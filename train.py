import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ, disable_inplace_relu
from datautil.getdataloader_single import get_act_dataloader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from network.act_network import ActNetwork

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

def transform_for_gnn(x):
    """Robust transformation for GNN input handling various formats"""
    if not GNN_AVAILABLE:
        return x
    # (same implementation as before)
    if x.dim() == 4:
        if x.size(1) == 8 or x.size(1) == 200:
            return x.squeeze(2).permute(0, 2, 1)
        elif x.size(2) == 8 or x.size(2) == 200:
            return x.squeeze(1).permute(0, 2, 1)
        elif x.size(3) == 8 or x.size(3) == 200:
            return x.squeeze(2)
        elif x.size(3) == 1 and (x.size(2) == 8 or x.size(2) == 200):
            return x.squeeze(3)
    elif x.dim() == 3:
        if x.size(1) == 8 or x.size(1) == 200:
            return x.permute(0, 2, 1)
        elif x.size(2) == 8 or x.size(2) == 200:
            return x
    raise ValueError(f"Cannot transform input of shape {x.shape} for GNN.")

# ======================= TEMPORAL CONVOLUTION BLOCK =======================
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
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
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.activation(out)
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.activation(out + residual)

# ======================= DATA AUGMENTATION MODULE =======================
class EMGDataAugmentation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.jitter_scale = args.jitter_scale
        self.scaling_std = args.scaling_std
        self.warp_ratio = args.warp_ratio
        self.dropout = nn.Dropout(p=args.channel_dropout)
        self.aug_prob = getattr(args, 'aug_prob', 0.7)

    def forward(self, x):
        if not self.training:
            return x
        if torch.rand(1) < self.aug_prob:
            x = x + torch.randn_like(x) * self.jitter_scale
        if torch.rand(1) < self.aug_prob:
            scale_factor = torch.randn(x.size(0), *([1] * (x.dim() - 1)), device=x.device)
            scale_factor = scale_factor * self.scaling_std + 1.0
            x = x * scale_factor
        if torch.rand(1) < self.aug_prob and self.warp_ratio > 0:
            seq_len = x.size(-1) if x.dim() != 3 else x.size(1)
            warp_amount = min(int(torch.rand(1).item() * self.warp_ratio * seq_len), seq_len - 1)
            if warp_amount > 0:
                if x.dim() == 4:
                    if torch.rand(1) > 0.5:
                        x = torch.cat([x[:, :, :, warp_amount:], x[:, :, :, :warp_amount]], dim=3)
                    else:
                        x = torch.cat([x[:, :, :, -warp_amount:], x[:, :, :, :-warp_amount]], dim=3)
                else:
                    if torch.rand(1) > 0.5:
                        x = torch.cat([x[:, warp_amount:, :], x[:, :warp_amount, :]], dim=1)
                    else:
                        x = torch.cat([x[:, -warp_amount:, :], x[:, :-warp_amount, :]], dim=1)
        if torch.rand(1) < self.aug_prob:
            x = self.dropout(x)
        return x

# ======================= OPTIMIZER FUNCTION =======================
def get_optimizer_adamw(algorithm, args, nettype='Diversify'):
    params = algorithm.parameters()
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                               weight_decay=args.weight_decay, nesterov=True)
    else:
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

# ======================= TEMPORAL GCN LAYER =======================
class TemporalGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, graph_builder):
        super().__init__()
        self.graph_builder = graph_builder
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        edge_indices = self.graph_builder.build_graph_for_batch(x)
        outputs = []
        for i in range(batch_size):
            feat = x[i]
            edge_index = edge_indices[i]
            adj = (torch.sparse_coo_tensor(edge_index,
                    torch.ones(edge_index.size(1), device=x.device),
                    size=(seq_len, seq_len)).to_dense() if edge_index.numel() else torch.eye(seq_len, device=x.device))
            conv = torch.mm(adj, feat)
            outputs.append(conv)
        x = torch.stack(outputs, dim=0)
        x = self.linear(x)
        return self.activation(self.dropout(x))

# ======================= ENHANCED GNN ARCHITECTURE =======================
class EnhancedTemporalGCN(TemporalGCN):
    def __init__(self, *args, **kwargs):
        # Extract GNN params
        self.n_layers = kwargs.pop('n_layers', 3)
        self.use_tcn = kwargs.pop('use_tcn', False)
        lstm_params = {k: kwargs.pop(k) for k in ['lstm_hidden_size', 'lstm_layers', 'bidirectional', 'lstm_dropout'] if k in kwargs}
        super().__init__(*args, **kwargs)
        self.skip_conn = nn.Linear(self.hidden_dim, self.output_dim)
        self.gnn_layers = nn.ModuleList([TemporalGCNLayer(
            input_dim=(self.input_dim if i==0 else self.hidden_dim),
            output_dim=self.hidden_dim,
            graph_builder=self.graph_builder
        ) for i in range(self.n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layers)])
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4,
                                               dropout=0.1, batch_first=True)
        if self.use_tcn:
            tcn_layers, num_channels = [], [self.hidden_dim]*3
            for i in range(3):
                dilation = 2**i
                in_ch = (self.hidden_dim if i==0 else num_channels[i-1])
                out_ch = num_channels[i]
                tcn_layers.append(TemporalBlock(in_ch, out_ch, 5, 1, dilation, 0.1))
            self.tcn = nn.Sequential(*tcn_layers)
            self.tcn_proj = nn.Linear(num_channels[-1], self.output_dim)
        else:
            self.lstm = nn.LSTM(input_size=self.hidden_dim,
                                hidden_size=lstm_params['lstm_hidden_size'],
                                num_layers=lstm_params['lstm_layers'],
                                batch_first=True,
                                bidirectional=lstm_params['bidirectional'],
                                dropout=lstm_params['lstm_dropout'])
            lstm_out_dim = lstm_params['lstm_hidden_size']*(2 if lstm_params['bidirectional'] else 1)
            self.lstm_norm = nn.LayerNorm(lstm_out_dim)
            self.lstm_proj = nn.Linear(lstm_out_dim, self.output_dim)
        self.temporal_norm = nn.LayerNorm(self.output_dim)
        self.projection_head = nn.Sequential(nn.Linear(self.output_dim,self.output_dim),
                                             nn.ReLU(), nn.Linear(self.output_dim,self.output_dim))
        self._init_weights()

    def _init_weights(self):
        for layer in self.gnn_layers:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
        for norm in self.norms:
            if hasattr(norm, 'weight'):
                nn.init.constant_(norm.weight, 1.0)
                nn.init.constant_(norm.bias, 0.0)
        if hasattr(self, 'lstm'):
            for name,param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(2).permute(0,2,1)
        if x.size(-1) not in [8,200]:
            raise ValueError(f"Expected feature dim 8 or 200, got {x.size(-1)}")
        if x.size(-1)==200 and self.input_dim==8:
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Linear(200,8).to(x.device)
            x = self.feature_projection(x)
        gnn_out = x
        for layer,norm in zip(self.gnn_layers,self.norms):
            gnn_out = norm(layer(gnn_out))
            gnn_out = F.gelu(gnn_out)
        attn_out,_ = self.attention(gnn_out,gnn_out,gnn_out)
        gnn_out = gnn_out + attn_out
        if self.use_tcn:
            tcn_in = gnn_out.permute(0,2,1)
            gnn_out = self.tcn_proj(self.tcn(tcn_in).permute(0,2,1))
        else:
            lstm_out,_ = self.lstm(gnn_out)
            gnn_out = self.lstm_proj(self.lstm_norm(lstm_out))
        pooled = gnn_out.mean(dim=1)
        skip = self.skip_conn(gnn_out.mean(dim=1))
        gate = torch.sigmoid(0.5*pooled + 0.5*skip)
        return gate*pooled + (1-gate)*skip

# ======================= DOMAIN ADVERSARIAL LOSS =======================
class DomainAdversarialLoss(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.domain_classifier = nn.Sequential(nn.Linear(bottleneck_dim,50), nn.ReLU(), nn.Linear(50,1))
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, features, domain_labels):
        return self.loss_fn(self.domain_classifier(features).squeeze(), domain_labels.float())

# ======================= MAIN TRAINING FUNCTION =======================
def main(args):
    print_args(args, [])
    set_random_seed(args.seed)
    print_environ()
    # Device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    os.makedirs(args.output, exist_ok=True)

    # Data loaders
    train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata = get_act_dataloader(args)

    # Initialize algorithm
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(args.device)

    # GNN initialization and replacement
    if args.use_gnn and GNN_AVAILABLE:
        print("\n===== Initializing GNN Feature Extractor =====")
        graph_builder = GraphBuilder(method='correlation', threshold_type='adaptive', default_threshold=0.3, adaptive_factor=1.5, fully_connected_fallback=True)
        gnn_model = EnhancedTemporalGCN(input_dim=8, hidden_dim=args.gnn_hidden_dim, output_dim=args.gnn_output_dim, graph_builder=graph_builder, lstm_hidden_size=args.lstm_hidden_size, lstm_layers=args.lstm_layers, bidirectional=args.bidirectional, lstm_dropout=args.lstm_dropout, n_layers=args.gnn_layers, use_tcn=args.use_tcn).to(args.device)
        algorithm.featurizer = gnn_model
        # Bottleneck creation
        def create_bottleneck(in_dim, out_dim, spec):
            try:
                layers=[]; cur=in_dim
                for _ in range(int(spec)-1): layers += [nn.Linear(cur,cur), nn.BatchNorm1d(cur), nn.ReLU(inplace=True)]
                layers.append(nn.Linear(cur,out_dim)); return nn.Sequential(*layers)
            except:
                return nn.Sequential(nn.Linear(in_dim,out_dim))
        in_dim, out_dim = args.gnn_output_dim, int(args.bottleneck)
        algorithm.bottleneck = create_bottleneck(in_dim,out_dim,args.layer).to(args.device)
        algorithm.abottleneck = create_bottleneck(in_dim,out_dim,args.layer).to(args.device)
        algorithm.dbottleneck = create_bottleneck(in_dim,out_dim,args.layer).to(args.device)

    algorithm.train()

    # Optimizers & scheduler
    opt_d = get_optimizer_adamw(algorithm, args, 'Diversify-adv')
    opt = get_optimizer_adamw(algorithm, args, 'Diversify-cls')
    opt_a = get_optimizer_adamw(algorithm, args, 'Diversify-all')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epoch)

    # Augmenter
    augmenter = EMGDataAugmentation(args).to(args.device)

    # Domain adversarial loss
    if args.domain_adv_weight > 0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(int(args.bottleneck)).to(args.device)
        print(f"Added domain adversarial training (weight: {args
