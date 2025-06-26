import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm

class feat_bottleneck(nn.Module):
    """
    Bottleneck layer for feature transformation
    Options:
    - "ori": No additional processing
    - "bn": Apply batch normalization after linear layer
    """
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        """
        Initialize bottleneck layer
        Args:
            feature_dim: Input feature dimension (int or tuple for GNN)
            bottleneck_dim: Output feature dimension (int)
            type: Processing type ("ori" or "bn")
        """
        super(feat_bottleneck, self).__init__()
        self.type = type
        
        # Handle GNN input dimension (tuple -> int)
        if isinstance(feature_dim, tuple):
            feature_dim = feature_dim[0]
        
        # Ensure dimensions are integers
        feature_dim = int(feature_dim)
        bottleneck_dim = int(bottleneck_dim)
        
        # Linear transformation layer
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        
        # Batch normalization layer (only used when type="bn")
        if type == "bn":
            self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        
        # Optional components (not currently used in forward pass)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """Forward pass through bottleneck"""
        # For GNN models, x is already flattened
        if x.dim() > 2:
            # Flatten the input if it has more than 2 dimensions
            x = x.view(x.size(0), -1)
        
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    """
    Classifier head with optional weight normalization
    Options:
    - "linear": Standard linear classifier
    - "wn": Weight-normalized linear classifier
    """
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        """
        Initialize classifier
        Args:
            class_num: Number of output classes (must be int)
            bottleneck_dim: Input feature dimension (must be int)
            type: Classifier type ("linear" or "wn")
        """
        super(feat_classifier, self).__init__()
        self.type = type
        
        # Ensure parameters are integers
        class_num = int(class_num)
        bottleneck_dim = int(bottleneck_dim)
        
        # Create appropriate classifier type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        """Forward pass through classifier"""
        return self.fc(x)
