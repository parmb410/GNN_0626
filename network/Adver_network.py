import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    """
    Gradient Reversal Layer for Domain Adaptation
    Forward pass: Identity function
    Backward pass: Negates and scales the gradient by alpha
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """Identity forward pass"""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient reversal backward pass"""
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    """
    Domain Discriminator Network
    Classifies which domain a feature vector belongs to
    """
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=4):
        """
        Initialize discriminator
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_domains: Number of output domains
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Network architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        """Forward pass through discriminator"""
        return self.layers(x)
