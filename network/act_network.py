import torch
import torch.nn as nn
import numpy as np
from gnn.temporal_gcn import TemporalGCN  # Add GNN import

# Configuration for different sensor modalities
var_size = {
    'emg': {
        'in_size': 8,        # Number of input channels (EMG sensors)
        'ker_size': 9,       # Kernel size for temporal convolution
    }
}

def disable_inplace_relu(model):
    """Disable inplace operations in ReLU layers for compatibility"""
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

class ActNetwork(nn.Module):
    """
    Flexible network for sensor-based activity recognition
    Supports both CNN for raw data and GNN for graph data
    """
    def __init__(self, taskname='emg', args=None):
        """
        Initialize the network
        Args:
            taskname: Sensor modality (currently only 'emg' supported)
            args: Configuration arguments
        """
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        self.args = args
        
        # ===== GNN BRANCH =====
        if hasattr(args, 'model_type') and args.model_type == 'gnn':
            # Initialize GNN model
            self.gnn = TemporalGCN(
                input_dim=var_size[taskname]['in_size'],
                hidden_dim=args.gnn_hidden_dim,
                output_dim=args.bottleneck
            )
            self.in_features = args.bottleneck
            self.is_gnn = True
        # ===== CNN BRANCH =====
        else:
            # First convolutional block
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=var_size[taskname]['in_size'],
                    out_channels=16,
                    kernel_size=(1, var_size[taskname]['ker_size']),
                    padding=(0, var_size[taskname]['ker_size']//2)  # Maintain temporal dimension
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=2)
            )
            
            # Second convolutional block
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=(1, var_size[taskname]['ker_size']),
                    padding=(0, var_size[taskname]['ker_size']//2)
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=2)
            )
            
            # Calculate output size dynamically
            self.in_features = self._calculate_output_size()
            self.is_gnn = False
        
        # Disable inplace operations for compatibility
        disable_inplace_relu(self)

    def _calculate_output_size(self):
        """Dynamically calculate output feature size using dummy input"""
        with torch.no_grad():
            # Create dummy input with standard EMG dimensions
            dummy_input = torch.zeros(1, var_size[self.taskname]['in_size'], 1, 200)
            
            # Pass through convolutional layers
            features = self.conv1(dummy_input)
            features = self.conv2(features)
            
            # Calculate total feature size
            return int(np.prod(features.shape[1:]))  # Channels × Height × Width

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor - either raw data or graph data
        Returns:
            Flattened feature tensor of shape (batch, features)
        """
        # ===== GNN PROCESSING =====
        if self.is_gnn:
            # For graph data: x is PyG Data object
            return self.gnn(x)
        # ===== CNN PROCESSING =====
        else:
            # For raw data: shape (batch, channels, 1, time_steps)
            x = self.conv1(x)
            x = self.conv2(x)
            return x.view(x.size(0), -1)  # Flatten while preserving batch dimension
