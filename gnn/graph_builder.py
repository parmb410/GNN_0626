import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Union, List, Optional
import itertools

class GraphBuilder:
    """
    Builds dynamic correlation graphs from EMG time-series data with PyTorch support.
    Now handles 3D inputs (batch, time_steps, features) by processing each sample individually.
    Features:
    - Multiple similarity metrics (correlation, covariance, Euclidean distance)
    - Adaptive thresholding based on data distribution
    - Efficient star topology for small sequences
    - Batch processing support
    - Comprehensive validation checks
    
    Args:
        method: Similarity metric ('correlation', 'covariance', 'euclidean')
        threshold_type: 'fixed' or 'adaptive' (median-based)
        default_threshold: Default threshold value for fixed method
        adaptive_factor: Multiplier for adaptive threshold calculation
        fully_connected_fallback: Use fully connected graph when no edges found
    """
    
    def __init__(self,
                 method: str = 'correlation',
                 threshold_type: str = 'adaptive',
                 default_threshold: float = 0.3,
                 adaptive_factor: float = 1.5,
                 fully_connected_fallback: bool = True):
        self.method = method
        self.threshold_type = threshold_type
        self.default_threshold = default_threshold
        self.adaptive_factor = adaptive_factor
        self.fully_connected_fallback = fully_connected_fallback
        
        if method not in {'correlation', 'covariance', 'euclidean'}:
            raise ValueError(f"Invalid method '{method}'. Choose from 'correlation', 'covariance', or 'euclidean'")
            
        if threshold_type not in {'fixed', 'adaptive'}:
            raise ValueError(f"Invalid threshold_type '{threshold_type}'. Choose 'fixed' or 'adaptive'")

    def build_graph(self, feature_sequence: Union[torch.Tensor, np.ndarray]) -> torch.LongTensor:
        """
        Build temporal graph from feature sequence (post-convolution)
        
        Args:
            feature_sequence: Feature tensor of shape (batch, time_steps, features) or (time_steps, features)
            
        Returns:
            edge_index: Tensor of shape [2, num_edges]
        """
        """Handle both tensor and numpy inputs"""
        # Convert numpy arrays to tensors
        if isinstance(feature_sequence, np.ndarray):
            feature_sequence = torch.from_numpy(feature_sequence).float()
        # Handle 3D input by processing first sample only
        if feature_sequence.ndim == 3:
            batch_size, T, F = feature_sequence.shape
            if batch_size > 1:
                print(f"⚠️ GraphBuilder received batch size {batch_size}, using first sample only")
            return self._build_single_graph(feature_sequence[0])
            
        # Handle 2D input
        elif feature_sequence.ndim == 2:
            return self._build_single_graph(feature_sequence)
            
        else:
            raise ValueError(f"Input must be 2D or 3D tensor, got shape {feature_sequence.shape}")

    def _build_single_graph(self, feature_sequence: torch.Tensor) -> torch.LongTensor:
        """Build graph for a single sample (2D tensor)"""
        # Validate input
        if not isinstance(feature_sequence, torch.Tensor):
            raise TypeError(f"Input must be torch.Tensor, got {type(feature_sequence)}")
            
        if feature_sequence.ndim != 2:
            raise ValueError(f"Input must be 2D (time_steps, features), got shape {feature_sequence.shape}")
            
        T, F = feature_sequence.shape
        device = feature_sequence.device
        
        # Handle small sequences with star topology
        if T < 5:
            return self._create_star_topology(T).to(device)

        # Compute similarity matrix between TIME STEPS
        similarity_matrix = self._compute_similarity(feature_sequence)
        
        # Determine threshold
        threshold = self._determine_threshold(similarity_matrix)
        
        # Build edges using TIME STEPS as nodes
        return self._create_edges(similarity_matrix, threshold, T, device)

    def _compute_similarity(self, data: torch.Tensor) -> torch.Tensor:
        """Compute similarity between time steps (temporal correlation) using PyTorch"""
        T, F = data.shape
        
        if self.method == 'correlation':
            # Compute row-wise (time steps) std
            stds = torch.std(data, dim=1)
            constant_mask = stds < 1e-8
            if torch.any(constant_mask):
                # Add small noise to constant time steps
                noise = torch.randn_like(data) * 1e-8
                data = data + noise * constant_mask.unsqueeze(1)
            
            # Compute time-step correlation
            centered = data - torch.mean(data, dim=1, keepdim=True)
            cov_matrix = torch.mm(centered, centered.t()) / (F - 1)
            std_products = torch.outer(stds, stds)
            std_products[std_products < 1e-10] = 1e-10
            corr = cov_matrix / std_products
            return torch.clamp(corr, -1.0, 1.0)
            
        elif self.method == 'covariance':
            centered = data - torch.mean(data, dim=1, keepdim=True)
            cov = torch.mm(centered, centered.t()) / (F - 1)
            return torch.nan_to_num(cov, nan=0.0)
            
        elif self.method == 'euclidean':
            dist_matrix = torch.cdist(data, data, p=2)
            max_dist = torch.max(dist_matrix)
            if max_dist < 1e-8:
                return torch.ones_like(dist_matrix)
            similarity = 1 - (dist_matrix / max_dist)
            return torch.clamp(similarity, -1.0, 1.0)

    def _determine_threshold(self, matrix: torch.Tensor) -> float:
        """Calculate appropriate threshold based on type"""
        if self.threshold_type == 'fixed':
            return self.default_threshold
        
        # Adaptive threshold based on median absolute similarity
        abs_matrix = torch.abs(matrix)
        # Ignore self-connections
        abs_matrix.fill_diagonal_(0)
        
        # Flatten and remove zeros
        flat_matrix = abs_matrix.flatten()
        non_zero = flat_matrix[flat_matrix > 0]
        
        if non_zero.numel() == 0:
            return 0.0
            
        median_val = torch.median(non_zero).item()
        return median_val * self.adaptive_factor

    def _create_edges(self, matrix: torch.Tensor, threshold: float, 
                     num_nodes: int, device: torch.device) -> torch.LongTensor:
        """Create edge connections between TIME STEPS (nodes)"""
        # Vectorized approach for better performance
        indices = torch.triu_indices(num_nodes, num_nodes, 1, device=device)
        i, j = indices[0], indices[1]
        similarities = matrix[i, j]
        
        # Find edges above threshold
        mask = torch.abs(similarities) > threshold
        valid_i = i[mask]
        valid_j = j[mask]
        
        # Create bidirectional edges
        if valid_i.numel() > 0:
            edges = torch.stack([
                torch.cat([valid_i, valid_j]),
                torch.cat([valid_j, valid_i])
            ], dim=0)
        else:
            edges = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Handle no-edge case
        if edges.numel() == 0 and self.fully_connected_fallback:
            return self._create_fully_connected(num_nodes).to(device)
        
        # Validate indices
        if edges.numel() > 0:
            if torch.any(edges >= num_nodes) or torch.any(edges < 0):
                edges = torch.clamp(edges, 0, num_nodes-1)
        
        return edges

    def _create_star_topology(self, num_nodes: int) -> torch.LongTensor:
        """Create star topology for small sequences (more efficient than fully connected)"""
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long)
        
        center = num_nodes // 2
        edges = []
        for i in range(num_nodes):
            if i != center:
                edges.append([center, i])
                edges.append([i, center])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _create_fully_connected(self, num_nodes: int) -> torch.LongTensor:
        """Create fully connected graph between TIME STEPS"""
        # Create all possible edges except self-loops
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        
        return torch.tensor([rows, cols], dtype=torch.long)

    def build_graph_for_batch(self, batch_data: torch.Tensor) -> List[torch.LongTensor]:
        """
        Build graphs for a batch of EMG samples.
        
        Args:
            batch_data: EMG time-series of shape (batch_size, time_steps, channels)
            
        Returns:
            edge_indices: List of edge_index tensors for each sample
        """
        if batch_data.ndim != 3:
            raise ValueError(f"Batch input must be 3D (batch, time, channels), got shape {batch_data.shape}")
            
        edge_indices = []
        
        for i in range(batch_data.size(0)):
            sample = batch_data[i]
            edge_index = self._build_single_graph(sample)
            edge_indices.append(edge_index)
            
        return edge_indices
