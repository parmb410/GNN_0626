import os
import sys
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ, disable_inplace_relu
from datautil.getdataloader_single import get_act_dataloader, get_curriculum_loader
from torch_geometric.loader import DataLoader as PyGDataLoader 
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader
from network.act_network import ActNetwork

# Suppress TensorFlow and SHAP warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
import logging
logging.getLogger("shap").setLevel(logging.WARNING)  # Suppress SHAP warnings

# Unified SHAP utilities import
from shap_utils import (
    get_background_batch, safe_compute_shap_values, plot_summary,
    overlay_signal_with_shap, plot_shap_heatmap,
    evaluate_shap_impact, compute_flip_rate, compute_jaccard_topk,
    compute_kendall_tau,
    cosine_similarity_shap, save_shap_numpy,
    compute_confidence_change, _get_shap_array,
    compute_aopc, compute_feature_coherence, compute_shap_entropy,
    plot_emg_shap_4d, plot_4d_shap_surface, evaluate_advanced_shap_metrics
)

# ======================= GNN INTEGRATION START =======================
try:
    from gnn.temporal_gcn import TemporalGCN
    from gnn.graph_builder import GraphBuilder
    GNN_AVAILABLE = True
    print("GNN modules successfully imported")
except ImportError as e:
    print(f"[WARNING] GNN modules not available: {str(e)}")
    print("Falling back to CNN architecture")
    GNN_AVAILABLE = False
# ======================= GNN INTEGRATION END =======================

def automated_k_estimation(features, k_min=2, k_max=10):
    """Automatically determine optimal cluster count using silhouette score and Davies-Bouldin Index"""
    best_k = k_min
    best_score = -1
    scores = []
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features)
        labels = kmeans.labels_
        
        # Skip if only one cluster exists
        if len(np.unique(labels)) < 2:
            silhouette = -1
            dbi = float('inf')
            ch_score = -1
        else:
            silhouette = silhouette_score(features, labels)
            dbi = davies_bouldin_score(features, labels)
            ch_score = calinski_harabasz_score(features, labels)
        
        # Combine scores: higher silhouette and CH are better, lower DBI is better
        norm_silhouette = (silhouette + 1) / 2
        norm_dbi = 1 / (1 + dbi)
        norm_ch = ch_score / 1000
        
        # Combined score gives weight to all three metrics
        combined_score = (0.5 * norm_silhouette) + (0.3 * norm_ch) + (0.2 * norm_dbi)
        scores.append((k, silhouette, dbi, ch_score, combined_score))
        
        print(f"K={k}: Silhouette={silhouette:.4f}, DBI={dbi:.4f}, CH={ch_score:.4f}, Combined={combined_score:.4f}")
        
        if combined_score > best_score:
            best_k = k
            best_score = combined_score
            
    print(f"[INFO] Optimal K determined as {best_k} (Combined Score: {best_score:.4f})")
    return best_k

def calculate_h_divergence(features_source, features_target):
    """
    Calculate h-divergence between source and target domain features
    
    Args:
        features_source: Features from source domain (numpy array)
        features_target: Features from target domain (numpy array)
        
    Returns:
        h_divergence: Domain discrepancy measure
        domain_acc: Domain classifier accuracy
    """
    # Create domain labels: 0 for source, 1 for target
    labels_source = np.zeros(features_source.shape[0])
    labels_target = np.ones(features_target.shape[0])
    
    # Combine features and labels
    X = np.vstack([features_source, features_target])
    y = np.hstack([labels_source, labels_target])
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train and test sets (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train a simple domain classifier
    domain_classifier = LogisticRegression(max_iter=1000, random_state=42)
    domain_classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    domain_acc = domain_classifier.score(X_test, y_test)
    
    # Calculate h-divergence: d = 2(1 - 2ε)
    h_divergence = 2 * (1 - 2 * (1 - domain_acc))
    
    return h_divergence, domain_acc

def transform_for_gnn(x):
    """Robust transformation for GNN input handling various formats"""
    if not GNN_AVAILABLE:
        return x
    
    # Handle common 4D formats
    if x.dim() == 4:
        # Format 1: [batch, channels, 1, time] -> [batch, time, channels]
        if x.size(1) == 8 or x.size(1) == 200:
            return x.squeeze(2).permute(0, 2, 1)
        # Format 2: [batch, 1, channels, time] -> [batch, time, channels]
        elif x.size(2) == 8 or x.size(2) == 200:
            return x.squeeze(1).permute(0, 2, 1)
        # Format 3: [batch, time, 1, channels] -> [batch, time, channels]
        elif x.size(3) == 8 or x.size(3) == 200:
            return x.squeeze(2)
        # New format: [batch, time, channels, 1]
        elif x.size(3) == 1 and (x.size(2) == 8 or x.size(2) == 200):
            return x.squeeze(3)
    
    # Handle 3D formats
    elif x.dim() == 3:
        # Format 1: [batch, channels, time] -> [batch, time, channels]
        if x.size(1) == 8 or x.size(1) == 200:
            return x.permute(0, 2, 1)
        # Format 2: [batch, time, channels] - already correct
        elif x.size(2) == 8 or x.size(2) == 200:
            return x
    
    # Unsupported format
    raise ValueError(
        f"Cannot transform input of shape {x.shape} for GNN. "
        f"Expected formats: [B, C, 1, T], [B, 1, C, T], [B, T, 1, C], [B, T, C, 1], "
        f"or 3D formats [B, C, T] or [B, T, C] where C is 8 or 200."
    )

# ======================= TEMPORAL CONVOLUTION BLOCK =======================
class TemporalBlock(nn.Module):
    """Temporal Convolution Block for TCN"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
        self.padding = padding

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.conv1.bias is not None:
            self.conv1.bias.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # Remove excess padding from convolution
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.activation(out)
        out = self.dropout(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        return self.activation(out + residual)

# ======================= DATA AUGMENTATION MODULE =======================
class EMGDataAugmentation(nn.Module):
    """Augmentation module for EMG signals"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.jitter_scale = args.jitter_scale
        self.scaling_std = args.scaling_std
        self.warp_ratio = args.warp_ratio
        self.dropout = nn.Dropout(p=args.channel_dropout)
        self.aug_prob = getattr(args, 'aug_prob', 0.7)

    def forward(self, x):
        # Apply augmentations only during training
        if not self.training:
            return x

        # Random jitter (additive Gaussian noise)
        if torch.rand(1) < self.aug_prob:
            noise = torch.randn_like(x) * self.jitter_scale
            x = x + noise

        # Random scaling
        if torch.rand(1) < self.aug_prob:
            # Create scale factor with proper dimensions for broadcasting
            scale_factor = torch.randn(x.size(0), *([1] * (x.dim() - 1)), device=x.device)
            scale_factor = scale_factor * self.scaling_std + 1.0
            x = x * scale_factor

        # Random time warping - fixed for both 3D and 4D inputs
        if torch.rand(1) < self.aug_prob and self.warp_ratio > 0:
            # Calculate warp_amount first
            if x.dim() == 4:  # [batch, channels, 1, time]
                seq_len = x.size(3)
            elif x.dim() == 3:  # [batch, time, channels]
                seq_len = x.size(1)
            else:
                seq_len = x.size(-1)  # Last dimension as fallback
            
            warp_amount = int(torch.rand(1).item() * self.warp_ratio * seq_len)
            warp_amount = min(warp_amount, seq_len - 1)  # Ensure valid slice
            
            if warp_amount > 0:
                # Apply warping based on input dimensions
                if x.dim() == 4:  # CNN format: [batch, channels, 1, time]
                    if torch.rand(1) > 0.5:  # Forward warp
                        x = torch.cat([x[:, :, :, warp_amount:], x[:, :, :, :warp_amount]], dim=3)
                    else:  # Backward warp
                        x = torch.cat([x[:, :, :, -warp_amount:], x[:, :, :, :-warp_amount]], dim=3)
                elif x.dim() == 3:  # GNN format: [batch, time, channels]
                    if torch.rand(1) > 0.5:  # Forward warp
                        x = torch.cat([x[:, warp_amount:, :], x[:, :warp_amount, :]], dim=1)
                    else:  # Backward warp
                        x = torch.cat([x[:, -warp_amount:, :], x[:, :-warp_amount, :]], dim=1)

        # Random channel dropout
        if torch.rand(1) < self.aug_prob:
            x = self.dropout(x)

        return x

# ======================= OPTIMIZER FUNCTION =======================
def get_optimizer_adamw(algorithm, args, nettype='Diversify'):
    """Get optimizer with configurable parameters"""
    params = algorithm.parameters()
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params, 
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, 
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True)
    else:  # Default to Adam
        optimizer = torch.optim.Adam(
            params, 
            lr=args.lr,
            weight_decay=args.weight_decay)
    
    return optimizer
# ======================= TEMPORAL GCN LAYER =======================
class TemporalGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, graph_builder):
        super().__init__()
        self.graph_builder = graph_builder
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [batch, time, features]
        batch_size, seq_len, n_features = x.shape
        
        # Get edge indices for all samples in the batch
        edge_indices = self.graph_builder.build_graph_for_batch(x)
        
        outputs = []
        for i in range(batch_size):
            sample_features = x[i]
            edge_index = edge_indices[i]
            
            # Create sparse adjacency matrix
            if edge_index.numel() > 0:
                adj_matrix = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.size(1), device=x.device),
                    size=(seq_len, seq_len)
                ).to_dense()
            else:
                adj_matrix = torch.eye(seq_len, device=x.device)
                
            # Graph convolution: A * X
            conv_result = torch.mm(adj_matrix, sample_features)
            outputs.append(conv_result)
        
        x = torch.stack(outputs, dim=0)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
# ======================= ENHANCED GNN ARCHITECTURE =======================
class EnhancedTemporalGCN(TemporalGCN):
    def __init__(self, *args, **kwargs):
        # Extract parameters specific to EnhancedTemporalGCN
        self.n_layers = kwargs.pop('n_layers', 3)
        self.use_tcn = kwargs.pop('use_tcn', False)
        
        # Extract LSTM parameters
        lstm_hidden_size = kwargs.pop('lstm_hidden_size', 128)
        lstm_layers = kwargs.pop('lstm_layers', 1)
        bidirectional = kwargs.pop('bidirectional', False)
        lstm_dropout = kwargs.pop('lstm_dropout', 0.2)
        
        # Now call super with cleaned kwargs
        super().__init__(*args, **kwargs)
        
        # Define skip connection layer
        self.skip_conn = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Enhanced GNN architecture
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Build GNN layers with normalization
        for i in range(self.n_layers):
            layer = TemporalGCNLayer(
                input_dim=self.input_dim if i == 0 else self.hidden_dim,
                output_dim=self.hidden_dim,
                graph_builder=self.graph_builder
            )
            self.gnn_layers.append(layer)
            self.norms.append(nn.LayerNorm(self.hidden_dim))
        
        # Add attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Add TCN as alternative to LSTM
        if self.use_tcn:
            tcn_layers = []
            num_channels = [self.hidden_dim] * 3
            kernel_size = 5
            dropout = 0.1
            for i in range(len(num_channels)):
                dilation = 2 ** i
                in_channels = self.hidden_dim if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
                tcn_layers += [TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation, dropout=dropout
                )]
            self.tcn = nn.Sequential(*tcn_layers)
            # Update projection to match actual output channels
            self.tcn_proj = nn.Linear(num_channels[-1], self.output_dim)
        else:
            # LSTM remains as fallback
            self.lstm = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=lstm_dropout if lstm_layers > 1 else 0
            )
            # Calculate LSTM output dimension
            lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
            self.lstm_proj = nn.Linear(lstm_output_dim, self.output_dim)
            self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        
        # Add layer normalization after temporal processing
        self.temporal_norm = nn.LayerNorm(self.output_dim)
        
        # Add projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for layer in self.gnn_layers:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.norms:
            if hasattr(layer, 'weight'):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
        if hasattr(self, 'lstm'):
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1 for better learning
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)

    def forward(self, x):
        # Convert 4D input to 3D if necessary
        if x.dim() == 4:
            # Handle 4D input: [batch, channels, 1, time]
            # Convert to [batch, time, channels]
            x = x.squeeze(2).permute(0, 2, 1)
        
        # Input shape verification - allow both 8 (raw) and 200 (embedded)
        if x.size(-1) not in [8, 200]:
            raise ValueError(
                f"Input features dim mismatch! Expected 8 (raw) or 200 (embedded), "
                f"got {x.size(-1)}. Full shape: {x.shape}"
            )
        
        # Apply feature projection if needed (200 -> 8)
        if x.size(-1) == 200 and self.input_dim == 8:
            if not hasattr(self, 'feature_projection'):
                # Create projection layer if not exists
                self.feature_projection = nn.Linear(200, 8).to(x.device)
                print("Created feature projection layer: 200 -> 8")
            x = self.feature_projection(x)
        
        # Store original input for skip connection
        original_x = x.clone()
        
        # GNN processing with normalization
        gnn_features = x  # We'll keep track of the GNN processed features
        for layer, norm in zip(self.gnn_layers, self.norms):
            gnn_features = layer(gnn_features)
            gnn_features = norm(gnn_features)
            gnn_features = F.gelu(gnn_features)
        
        # Attention pooling
        attn_out, _ = self.attention(gnn_features, gnn_features, gnn_features)
        x = gnn_features + attn_out  # Residual connection
        
        # Temporal modeling
        if self.use_tcn:
            # TCN expects (batch, channels, time)
            tcn_in = x.permute(0, 2, 1)
            tcn_out = self.tcn(tcn_in)
            tcn_out = tcn_out.permute(0, 2, 1)
            # Project to output dimension
            temporal_out = self.tcn_proj(tcn_out)
        else:
            # LSTM processing
            lstm_out, _ = self.lstm(x)
            lstm_out = self.lstm_norm(lstm_out)
            temporal_out = self.lstm_proj(lstm_out)
        
        # Temporal aggregation using mean pooling
        gnn_out = temporal_out.mean(dim=1)  # [batch, output_dim]
        gnn_out = self.temporal_norm(gnn_out)
        
        # Skip connection processing - USE GNN PROCESSED FEATURES!
        # Process skip connection with temporal aggregation
        skip_out = gnn_features  # Use the GNN processed features instead of original input
        
        # Temporal aggregation for skip connection
        skip_out = skip_out.mean(dim=1)  # [batch, channels]
        skip_out = self.skip_conn(skip_out)  # [batch, output_dim]
        
        # Residual connection with gating mechanism
        gate = torch.sigmoid(0.5 * gnn_out + 0.5 * skip_out)
        output = gate * gnn_out + (1 - gate) * skip_out
        
        return output

# ======================= DOMAIN ADVERSARIAL LOSS =======================
class DomainAdversarialLoss(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, features, domain_labels):
        domain_pred = self.domain_classifier(features)
        return self.loss_fn(domain_pred.squeeze(), domain_labels.float())

# ======================= MAIN TRAINING FUNCTION =======================
def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)
    print_environ()
    print(s)
    # Add device configuration (GPU/CPU)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    # Create output directory if it exists
    os.makedirs(args.output, exist_ok=True)
    
    # Load datasets - IMPORTANT: This returns loader objects, not datasets
    loader_data = get_act_dataloader(args)
    train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata = loader_data[:7]
    
    # Automated K estimation if enabled
    if getattr(args, 'automated_k', False):
        print("\nRunning automated K estimation...")
        
        # Use GNN if enabled and available, otherwise use standard CNN
        if args.use_gnn and GNN_AVAILABLE:
            print("Using GNN for feature extraction")
            # Initialize graph builder for feature extraction
            graph_builder = GraphBuilder(
                method='correlation',
                threshold_type='adaptive',
                default_threshold=0.3,
                adaptive_factor=1.5
            )
            temp_model = EnhancedTemporalGCN(
                input_dim=8,  # EMG channels
                hidden_dim=args.gnn_hidden_dim,
                output_dim=args.gnn_output_dim,
                graph_builder=graph_builder,
                n_layers=getattr(args, 'gnn_layers', 3),
                use_tcn=getattr(args, 'use_tcn', True)
            ).to(args.device)
        else:
            print("Using CNN for feature extraction")
            temp_model = ActNetwork(args.dataset).to(args.device)
        
        temp_model.eval()
        feature_list = []
        
        with torch.no_grad():
            first_batch = True
            
            for batch in train_loader:
                # Handle GNN data differently
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    # Convert PyG Batch to dense tensor
                    from torch_geometric.utils import to_dense_batch
                    x_dense, mask = to_dense_batch(inputs.x, inputs.batch)
                    inputs = x_dense  # [batch_size, max_nodes, features]
                else:
                    inputs = batch[0].to(args.device).float()
                
                # Handle GNN input format if needed
                if args.use_gnn and GNN_AVAILABLE:
                    # Convert to (batch, time, channels) format
                    inputs = inputs.reshape(args.batch_size, -1, 8)
                    
                    # Ensure it's 3D: [batch, time, channels]
                    if inputs.dim() != 3:
                        # For 4D: [batch, channels, 1, time] -> [batch, time, channels]
                        if inputs.dim() == 4:
                            inputs = inputs.squeeze(2).permute(0, 2, 1)
                        else:
                            raise ValueError(f"Unsupported GNN input dimension: {inputs.dim()}")
                    
                    # Only print first batch shape for verification
                    if first_batch:
                        print(f"GNN input shape: {inputs.shape}")
                        first_batch = False
                
                features = temp_model(inputs)
                feature_list.append(features.detach().cpu().numpy())
        
        all_features = np.concatenate(feature_list, axis=0)
        optimal_k = automated_k_estimation(all_features)
        args.latent_domain_num = optimal_k
        print(f"Using automated latent_domain_num (K): {args.latent_domain_num}")
        
        del temp_model
        torch.cuda.empty_cache()
    
    # Batch size adjustment
    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num
    print(f"Adjusted batch size: {args.batch_size}")
    
    # Recreate data loaders with new batch size
    # Determine which loader class to use
    if args.use_gnn and GNN_AVAILABLE:
        LoaderClass = PyGDataLoader
    else:
        LoaderClass = TorchDataLoader
    
    train_loader = LoaderClass(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=min(4, args.N_WORKERS),
        drop_last=False,
        shuffle=True
    )
    
    train_loader_noshuffle = LoaderClass(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=min(4, args.N_WORKERS),
        drop_last=False,
        shuffle=False
    )
    
    valid_loader = LoaderClass(
        dataset=val,
        batch_size=args.batch_size,
        num_workers=min(4, args.N_WORKERS),
        drop_last=False,
        shuffle=False
    )
    
    target_loader = LoaderClass(
        dataset=targetdata,
        batch_size=args.batch_size,
        num_workers=min(4, args.N_WORKERS),
        drop_last=False,
        shuffle=False
    )
    
    # Initialize algorithm
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(args.device)
    
    # ======================= GNN INITIALIZATION START =======================
    if args.use_gnn and GNN_AVAILABLE:
        print("\n===== Initializing GNN Feature Extractor =====")
        
        # Initialize graph builder with research-optimized parameters
        graph_builder = GraphBuilder(
            method='correlation',
            threshold_type='adaptive',
            default_threshold=0.3,
            adaptive_factor=1.5,
            fully_connected_fallback=True
        )
        
        # Add GNN parameters to args
        args.gnn_layers = getattr(args, 'gnn_layers', 3)
        args.use_tcn = getattr(args, 'use_tcn', True)
        
        # Initialize enhanced GNN model
        gnn_model = EnhancedTemporalGCN(
            input_dim=8,  # EMG channels
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            graph_builder=graph_builder,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            bidirectional=args.bidirectional,
            lstm_dropout=args.lstm_dropout,
            n_layers=args.gnn_layers,
            use_tcn=args.use_tcn
        ).to(args.device)
        
        # Replace CNN feature extractor with GNN
        algorithm.featurizer = gnn_model
        
        # Create a function to build consistent bottleneck layers
        def create_bottleneck(input_dim, output_dim, layer_spec):
            """Create a bottleneck layer with consistent architecture"""
            try:
                num_layers = int(layer_spec)
                layers = []
                current_dim = input_dim
                
                # Add intermediate layers
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(current_dim, current_dim))
                    layers.append(nn.BatchNorm1d(current_dim))
                    layers.append(nn.ReLU(inplace=True))
                
                # Add final projection layer
                layers.append(nn.Linear(current_dim, output_dim))
                return nn.Sequential(*layers)
            except ValueError:
                # Fallback to simple linear projection
                return nn.Sequential(nn.Linear(input_dim, output_dim))
        
        # Create both bottlenecks with the correct dimensions
        input_dim = args.gnn_output_dim
        output_dim = int(args.bottleneck)
        
        # Create both bottlenecks (classifier and adversarial)
        algorithm.bottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        algorithm.abottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        algorithm.dbottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        
        print(f"Created bottlenecks: {input_dim} -> {output_dim}")
        print(f"Bottleneck architecture: {algorithm.bottleneck}")
        
        # Skip pretraining when using LSTM or TCN
        if hasattr(args, 'gnn_pretrain_epochs') and args.gnn_pretrain_epochs > 0:
            if args.lstm_layers > 0 or args.use_tcn:
                print("Skipping GNN pretraining due to LSTM/TCN integration")
                args.gnn_pretrain_epochs = 0
        
        # GNN Pretraining if enabled
        if hasattr(args, 'gnn_pretrain_epochs') and args.gnn_pretrain_epochs > 0:
            print(f"\n==== GNN Pretraining ({args.gnn_pretrain_epochs} epochs) ====")
            gnn_optimizer = torch.optim.AdamW(
                gnn_model.parameters(),
                lr=args.gnn_lr,
                weight_decay=args.gnn_weight_decay
            )
            
            for epoch in range(args.gnn_pretrain_epochs):
                gnn_model.train()
                total_loss = 0
                for batch in train_loader:
                    # Handle GNN data differently
                    if args.use_gnn and GNN_AVAILABLE:
                        inputs = batch[0].to(args.device)
                        labels = batch[1].to(args.device)
                        domains = batch[2].to(args.device)
                        x = inputs
                    else:
                        inputs = batch[0].to(args.device).float()
                        labels = batch[1].to(args.device).long()
                        domains = batch[2].to(args.device).long()
                        x = inputs
                    
                    # Convert to (batch, time, features) format
                    if args.use_gnn and GNN_AVAILABLE:
                        x = transform_for_gnn(x)
                        if x.dim() != 3:
                            raise ValueError(f"GNN requires 3D input (B,T,C), got {x.shape}")
                    
                    # Calculate mean across time dimension
                    target = torch.mean(x, dim=1)  # [batch, channels]
                    
                    # Forward pass
                    features = gnn_model(x)
                    
                    # Reconstruction loss
                    reconstructed = gnn_model.reconstruct(features)
                    loss = torch.nn.functional.mse_loss(reconstructed, target)
                    
                    # Skip update if NaN loss
                    if torch.isnan(loss):
                        print("NaN loss detected during pretraining, skipping update")
                        continue
                    
                    # Optimization with gradient clipping
                    gnn_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
                    gnn_optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f'GNN Pretrain Epoch {epoch+1}/{args.gnn_pretrain_epochs}: Loss {total_loss/len(train_loader):.4f}')
            
            print("GNN pretraining complete")
    # ======================= GNN INITIALIZATION END =======================
    
    algorithm.train()
    
    # Setup optimizers with AdamW
    optd = get_optimizer_adamw(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer_adamw(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer_adamw(algorithm, args, nettype='Diversify-all')
    
    # Add learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(opt, T_max=args.max_epoch)
    
    # Add data augmentation module
    augmenter = EMGDataAugmentation(args).cuda()
    
    # Add domain adversarial training if enabled
    if getattr(args, 'domain_adv_weight', 0.0) > 0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(
            bottleneck_dim=int(args.bottleneck)
        ).cuda()
        print(f"Added domain adversarial training (weight: {args.domain_adv_weight})")
    
    # Training metrics logging
    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'ent_loss',
                            'total_loss', 'train_acc', 'valid_acc', 'target_acc',
                            'total_cost_time', 'h_divergence', 'domain_acc']}
    best_valid_acc, target_acc = 0, 0
    # Determine loader class for entire source loader
if args.use_gnn and GNN_AVAILABLE:
    LoaderClass = PyGDataLoader
else:
    LoaderClass = TorchDataLoader
    # Create entire source loader for h-divergence calculation
    entire_source_loader = LoaderClass(
        tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, args.N_WORKERS))
    
    # Early stopping tracking
    best_valid_acc = 0
    epochs_without_improvement = 0
    early_stopping_patience = getattr(args, 'early_stopping_patience', 3)
    
    # Set gradient clipping norm
    MAX_GRAD_NORM = getattr(args, 'max_grad_norm', 5.0)
    
    # Main training loop
    global_step = 0
    for round_idx in range(args.max_epoch):
        # Dropout adjustment
        if hasattr(algorithm.featurizer, 'dropout'):
            if round_idx < 10:
                algorithm.featurizer.dropout.p = 0.1
            else:
                algorithm.featurizer.dropout.p = 0.5
        
        # Adaptive data augmentation
        if round_idx < 10:
            augmenter.aug_prob = 0.3
        else:
            augmenter.aug_prob = getattr(args, 'aug_prob', 0.7)
        
        print(f'\n======== ROUND {round_idx} ========')
        
        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {round_idx}")
            break
            
        # Determine epochs for this round
        if getattr(args, 'curriculum', False):
            # Ensure CL_PHASE_EPOCHS is a list
            if not hasattr(args, 'CL_PHASE_EPOCHS') or not isinstance(args.CL_PHASE_EPOCHS, list):
                args.CL_PHASE_EPOCHS = [3, 5, 8]  # Default to list of phases
            
            # Ensure CL_DIFFICULTY is a list
            if not hasattr(args, 'CL_DIFFICULTY') or not isinstance(args.CL_DIFFICULTY, list):
                args.CL_DIFFICULTY = [0.2, 0.5, 0.8]  # Default difficulty levels
            
            # Check if current round is within curriculum phases
            if round_idx < len(args.CL_PHASE_EPOCHS):
                current_epochs = args.CL_PHASE_EPOCHS[round_idx]
                print(f"Curriculum learning: Stage {round_idx} (using {current_epochs} epochs)")
            else:
                current_epochs = args.local_epoch
        else:
            current_epochs = args.local_epoch
        
        # Curriculum learning setup
        if getattr(args, 'curriculum', False) and round_idx < len(args.CL_PHASE_EPOCHS):
            print(f"Curriculum learning: Stage {round_idx}")
            
            # Set algorithm to evaluation mode
            algorithm.eval()
            
            # Create a prediction function that handles GNN transformation
            def curriculum_predict(x):
                if args.use_gnn and GNN_AVAILABLE:
                    x = transform_for_gnn(x)
                return algorithm.predict(x)
            
            # Create a simple object for domain evaluation
            class CurriculumEvaluator:
                def predict(self, x):
                    return curriculum_predict(x)
            
            evaluator = CurriculumEvaluator()
            
            # Get the curriculum loader
            train_loader = get_curriculum_loader(
                args,
                evaluator,  # Pass evaluator instead of algorithm
                tr,
                val,
                stage=round_idx,
                loader_class=PyGDataLoader  # Pass PyG loader class
            )
               # Determine loader class for entire source loader
            if args.use_gnn and GNN_AVAILABLE:
                LoaderClass = PyGDataLoader
            else:
                LoaderClass = TorchDataLoader

            # Update the no-shuffle loader
            train_loader_noshuffle = LoaderClass(
                train_loader.dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=min(2, args.N_WORKERS)
            )
        else:
            train_loader = get_curriculum_loader(
                args,
                algorithm,
                tr,
                val,
                stage=round_idx,
                loader_class=TorchDataLoader  # Pass standard loader class
            )
            train_loader_noshuffle = TorchDataLoader(
                train_loader.dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=min(2, args.N_WORKERS)
            )
            # Set algorithm back to training mode
            algorithm.train()
        
        # Phase 1: Feature update
        print('\n==== Feature update ====')
        print_row(['epoch', 'class_loss'], colwidth=15)
        for step in range(current_epochs):
            epoch_class_loss = 0.0
            batch_count = 0
            for batch in train_loader:
                # Handle GNN data differently
                if args.use_gnn and GNN_AVAILABLE:
                    # For GNN: batch[0] is a Batch object, batch[1] is labels, batch[2] is domains
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    data = [inputs, labels, domains]
                else:
                    # For non-GNN data
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    # Apply augmentation
                    inputs = augmenter(inputs)
                    data = [inputs, labels, domains]
                
                loss_result_dict = algorithm.update_a(data, opta)
                
                # Skip update if NaN loss
                if not np.isfinite(loss_result_dict['class']):
                    print("Skipping step due to non-finite loss")
                    continue
                
                epoch_class_loss += loss_result_dict['class']
                batch_count += 1
                
            if batch_count > 0:
                epoch_class_loss /= batch_count
                print_row([step, epoch_class_loss], colwidth=15)
                logs['class_loss'].append(epoch_class_loss)
        
        # Phase 2: Latent domain characterization
        print('\n==== Latent domain characterization ====')
        print_row(['epoch', 'total_loss', 'dis_loss', 'ent_loss'], colwidth=15)
        for step in range(current_epochs):
            epoch_total = 0.0
            epoch_dis = 0.0
            epoch_ent = 0.0
            batch_count = 0
            
            for batch in train_loader:
                # Handle GNN data differently
                if args.use_gnn and GNN_AVAILABLE:
                    # For GNN: batch[0] is a Batch object, batch[1] is labels, batch[2] is domains
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    data = [inputs, labels, domains]
                else:
                    # For non-GNN data
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    # Apply augmentation
                    inputs = augmenter(inputs)
                    data = [inputs, labels, domains]
                
                loss_result_dict = algorithm.update_d(data, optd)
                
                if any(not np.isfinite(v) for v in loss_result_dict.values()):
                    print("Skipping step due to non-finite loss")
                    continue
                    
                epoch_total += loss_result_dict['total']
                epoch_dis += loss_result_dict['dis']
                epoch_ent += loss_result_dict['ent']
                batch_count += 1
                
            if batch_count > 0:
                epoch_total /= batch_count
                epoch_dis /= batch_count
                epoch_ent /= batch_count
                print_row([step, epoch_total, epoch_dis, epoch_ent], colwidth=15)
                logs['dis_loss'].append(epoch_dis)
                logs['ent_loss'].append(epoch_ent)
                logs['total_loss'].append(epoch_total)
                
        algorithm.set_dlabel(train_loader)
        
        print('\n==== Domain-invariant feature learning ====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch'] + [f"{item}_loss" for item in loss_list] + \
                   [f"{item}_acc" for item in eval_dict] + ['total_cost_time']
        print_row(print_key, colwidth=15)
        
        round_start_time = time.time()
        for step in range(current_epochs):
            step_start_time = time.time()
            for batch in train_loader:
                # Handle GNN data differently
                if args.use_gnn and GNN_AVAILABLE:
                    # For GNN: batch[0] is a Batch object, batch[1] is labels, batch[2] is domains
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    data = [inputs, labels, domains]
                else:
                    # For non-GNN data
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    # Apply augmentation
                    inputs = augmenter(inputs)
                    data = [inputs, labels, domains]
                
                step_vals = algorithm.update(data, opt)
                
                # Apply gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(algorithm.parameters(), MAX_GRAD_NORM)
            
            # Create transform wrapper for GNN if needed
            transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
            
            # Calculate accuracies
            results = {
                'epoch': global_step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None, transform_fn=transform_fn),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None, transform_fn=transform_fn),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None, transform_fn=transform_fn),
                'total_cost_time': time.time() - step_start_time
            }
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Log losses
            for key in loss_list:
                results[f"{key}_loss"] = step_vals[key]
                logs[f"{key}_loss"].append(step_vals[key])
            
            # Log metrics
            for metric in ['train_acc', 'valid_acc', 'target_acc']:
                logs[metric].append(results[metric])
            
            # Update best validation accuracy
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
                epochs_without_improvement = 0
                # Save best model
                torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))
            else:
                epochs_without_improvement += 1
                
            print_row([results[key] for key in print_key], colwidth=15)
            global_step += 1
            
        logs['total_cost_time'].append(time.time() - round_start_time)
        
        # Calculate h-divergence every 5 epochs
        if round_idx % 5 == 0:
            print("\nCalculating h-divergence...")
            algorithm.eval()
            
            # Extract features for source domain
            source_features = []
            with torch.no_grad():
                for data in entire_source_loader:
                    # Handle GNN data differently
                    if args.use_gnn and GNN_AVAILABLE:
                        inputs = data[0].to(args.device)
                        # Convert to (batch, time, channels) format
                        inputs = transform_for_gnn(inputs)
                    else:
                        inputs = data[0].to(args.device).float()
                    
                    features = algorithm.featurizer(inputs).detach().cpu().numpy()
                    source_features.append(features)
            source_features = np.concatenate(source_features, axis=0)
            
            # Extract features for target domain
            target_features = []
            with torch.no_grad():
                for data in target_loader:
                    # Handle GNN data differently
                    if args.use_gnn and GNN_AVAILABLE:
                        inputs = data[0].to(args.device)
                        # Convert to (batch, time, channels) format
                        inputs = transform_for_gnn(inputs)
                    else:
                        inputs = data[0].to(args.device).float()
                    
                    features = algorithm.featurizer(inputs).detach().cpu().numpy()
                    target_features.append(features)
            target_features = np.concatenate(target_features, axis=0)
            
            # Calculate h-divergence
            h_div, domain_acc = calculate_h_divergence(source_features, target_features)
            logs['h_divergence'].append(h_div)
            logs['domain_acc'].append(domain_acc)
            print(f" H-Divergence: {h_div:.4f}, Domain Classifier Acc: {domain_acc:.4f}")
            
            algorithm.train()
            
    print(f'\n🎯 Final Target Accuracy: {target_acc:.4f}')
    
    # SHAP explainability analysis
    if getattr(args, 'enable_shap', False):
        print("\n📊 Running SHAP explainability...")
        try:
            # Prepare background and evaluation data
            if args.use_gnn and GNN_AVAILABLE:
                # For GNN, we need to sample individual graphs
                background = []
                for data in valid_loader:
                    background.extend(data[0].to_data_list())
                    if len(background) >= 64:
                        break
                background = background[:64]
                X_eval = background[:10]
            else:
                background = get_background_batch(valid_loader, size=64).cuda()
                X_eval = background[:10]
            
            # Disable inplace operations in the model
            disable_inplace_relu(algorithm)
            
            # Create transform wrapper for GNN if needed
            transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
            
            # Transform background and X_eval if necessary
            if transform_fn is not None and not args.use_gnn:
                background = transform_fn(background)
                X_eval = transform_fn(X_eval)
            
            # Compute SHAP values safely
            shap_explanation = safe_compute_shap_values(algorithm, background, X_eval)
            
            # Extract values from Explanation object
            shap_vals = shap_explanation.values
            print(f"SHAP values shape: {shap_vals.shape}")
            
            # Convert to numpy safely before visualization
            X_eval_np = X_eval
            if isinstance(X_eval, list):
                # Handle PyG data list
                X_eval_np = [d.x.detach().cpu().numpy() for d in X_eval]
            else:
                X_eval_np = X_eval.detach().cpu().numpy()
            
            # Handle GNN dimensionality for visualization
            if args.use_gnn and GNN_AVAILABLE:
                print(f"Original SHAP values shape: {shap_vals.shape}")
                print(f"Original X_eval shape: {X_eval_np[0].shape}")
                
                # If 4D, reduce to 3D by summing over classes
                if isinstance(shap_vals, list):
                    # For GNN, shap_vals is a list of arrays
                    shap_vals = [np.abs(sv).sum(axis=-1) for sv in shap_vals]
                    print(f"SHAP values after class sum: {shap_vals[0].shape}")
                
                # Now we should have list of 3D arrays: [channels, time] for each graph
                # We'll just visualize the first sample
                plot_emg_shap_4d(X_eval_np[0], shap_vals[0], 
                                output_path=os.path.join(args.output, "shap_gnn_sample.html"))
            else:
                # Generate core visualizations for non-GNN data
                try:
                    plot_summary(shap_vals, X_eval_np, 
                                output_path=os.path.join(args.output, "shap_summary.png"))
                except IndexError as e:
                    print(f"SHAP summary plot dimension error: {str(e)}")
                    print(f"Using fallback 3D visualization instead")
                    plot_emg_shap_4d(X_eval, shap_vals, 
                                    output_path=os.path.join(args.output, "shap_3d_fallback.html"))
                    
                overlay_signal_with_shap(X_eval_np[0], shap_vals, 
                                        output_path=os.path.join(args.output, "shap_overlay.png"))
                plot_shap_heatmap(shap_vals, 
                                 output_path=os.path.join(args.output, "shap_heatmap.png"))
            
            # Save SHAP values
            save_path = os.path.join(args.output, "shap_values.npy")
            save_shap_numpy(shap_vals, save_path=save_path)
            
            # Confusion matrix
            true_labels, pred_labels = [], []
            for data in valid_loader:
                # Handle GNN data differently
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = data[0].to(args.device)
                    y = data[1]
                else:
                    inputs = data[0].to(args.device).float()
                    y = data[1]
                
                with torch.no_grad():
                    # Apply transform for GNN if needed
                    if args.use_gnn and GNN_AVAILABLE:
                        inputs = transform_for_gnn(inputs)
                    preds = algorithm.predict(inputs).cpu()
                    true_labels.extend(y.cpu().numpy())
                    pred_labels.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())
            
            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix (Validation Set)")
            plt.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=300)
            plt.close()
            
            print("✅ SHAP analysis completed successfully")
        except Exception as e:
            print(f"[ERROR] SHAP analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Plot training metrics
    try:
        # Main training metrics plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        epochs = list(range(len(logs['class_loss'])))
        plt.plot(epochs, logs['class_loss'], label="Class Loss", marker='o')
        plt.plot(epochs, logs['dis_loss'], label="Dis Loss", marker='x')
        plt.plot(epochs, logs['total_loss'], label="Total Loss", linestyle='--')
        plt.title("Losses over Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        epochs = list(range(len(logs['train_acc'])))
        plt.plot(epochs, logs['train_acc'], label="Train Accuracy", marker='o')
        plt.plot(epochs, logs['valid_acc'], label="Valid Accuracy", marker='x')
        plt.plot(epochs, logs['target_acc'], label="Target Accuracy", linestyle='--')
        plt.title("Accuracy over Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "training_metrics.png"), dpi=300)
        plt.close()
        print("✅ Training metrics plot saved")
        
        # H-Divergence plot
        if logs['h_divergence']:
            plt.figure(figsize=(10, 6))
            h_epochs = [i * 5 for i in range(len(logs['h_divergence']))]
            plt.plot(h_epochs, logs['h_divergence'], 'o-', label='H-Divergence')
            plt.plot(h_epochs, logs['domain_acc'], 's-', label='Domain Classifier Acc')
            plt.title("Domain Discrepancy over Training")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(args.output, "domain_discrepancy.png"), dpi=300)
            plt.close()
            print("✅ Domain discrepancy plot saved")
            
    except Exception as e:
        print(f"[WARNING] Failed to generate training plots: {str(e)}")

if __name__ == '__main__':
    args = get_args()
    args.lambda_cls = getattr(args, 'lambda_cls', 1.0)
    args.lambda_dis = getattr(args, 'lambda_dis', 0.1)
    args.label_smoothing = getattr(args, 'label_smoothing', 0.1)
    args.max_grad_norm = getattr(args, 'max_grad_norm', 5.0)
    args.gnn_pretrain_epochs = getattr(args, 'gnn_pretrain_epochs', 20)

    # Add GNN-specific parameters to args
    if not hasattr(args, 'use_gnn'):
        args.use_gnn = False
        
    if args.use_gnn:
        if not GNN_AVAILABLE:
            print("[WARNING] GNN requested but not available. Falling back to CNN.")
            args.use_gnn = False
        else:
            # GNN hyperparameters
            args.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 64)
            args.gnn_output_dim = getattr(args, 'gnn_output_dim', 256)
            args.gnn_layers = getattr(args, 'gnn_layers', 3)
            args.gnn_lr = getattr(args, 'gnn_lr', 0.001)
            args.gnn_weight_decay = getattr(args, 'gnn_weight_decay', 0.0001)
            args.gnn_pretrain_epochs = getattr(args, 'gnn_pretrain_epochs', 5)
            
            # TCN/LSTM parameters
            args.use_tcn = getattr(args, 'use_tcn', True)
            args.lstm_hidden_size = getattr(args, 'lstm_hidden_size', 128)
            args.lstm_layers = getattr(args, 'lstm_layers', 1)
            args.bidirectional = getattr(args, 'bidirectional', False)
            args.lstm_dropout = getattr(args, 'lstm_dropout', 0.2)

    # Increase adversarial weight for better domain adaptation
    if not hasattr(args, 'adv_weight'):
        args.adv_weight = 2.0

    # Add new hyperparameters
    args.optimizer = getattr(args, 'optimizer', 'adamw')
    args.weight_decay = getattr(args, 'weight_decay', 1e-4)
    args.domain_adv_weight = getattr(args, 'domain_adv_weight', 0.5)

    # Augmentation parameters
    args.jitter_scale = getattr(args, 'jitter_scale', 0.05)
    args.scaling_std = getattr(args, 'scaling_std', 0.1)
    args.warp_ratio = getattr(args, 'warp_ratio', 0.1)
    args.channel_dropout = getattr(args, 'channel_dropout', 0.1)
    args.aug_prob = getattr(args, 'aug_prob', 0.7)

    # Training schedule
    args.max_epoch = getattr(args, 'max_epoch', 100)
    args.early_stopping_patience = getattr(args, 'early_stopping_patience', 10)

    # Domain adaptation
    if not hasattr(args, 'adv_weight'):
        args.adv_weight = 1.5  # Increased adversarial weight

    main(args)
