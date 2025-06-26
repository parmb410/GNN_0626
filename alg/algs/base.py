import torch
import torch.nn as nn

class Algorithm(nn.Module):
    """
    Base class for all domain generalization algorithms.
    
    Provides common functionality for training/evaluation mode switching,
    and defines the required interface methods.
    """
    def __init__(self, args):
        super(Algorithm, self).__init__()
        self.args = args
        self.train()  # Start in training mode by default

    def update(self, minibatches):
        """Update the model parameters using a batch of data"""
        raise NotImplementedError("Subclasses must implement update method")

    def predict(self, x):
        """Make predictions on input data"""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def train(self, mode=True):
        """Set the model to training mode"""
        return super().train(mode)
    
    def eval(self):
        """Set the model to evaluation mode"""
        return super().eval()
    
    def explain(self, x):
        """Explain the model's prediction for input x"""
        # Default implementation just returns predictions
        return self.predict(x)
