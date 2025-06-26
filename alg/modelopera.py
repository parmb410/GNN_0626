import torch
from network import act_network
from gnn.temporal_gcn import TemporalGCN

def get_fea(args):
    """Initialize feature extractor network with GNN support"""
    if hasattr(args, 'model_type') and args.model_type == 'gnn':
        # Default values if not present
        input_dim = 8  # EMG channels
        hidden_dim = getattr(args, 'gnn_hidden_dim', 64)
        output_dim = getattr(args, 'gnn_output_dim', 256)
        
        net = TemporalGCN(input_dim, hidden_dim, output_dim)
        net.in_features = output_dim  # Needed for downstream bottleneck
        return net
    else:
        return act_network.ActNetwork(args.dataset)

def accuracy(network, loader, weights=None, usedpredict='p', transform_fn=None):
    """
    Calculate accuracy for a given data loader with support for:
    - Sample weighting
    - Multiple prediction methods
    - Both binary and multiclass classification
    - Handling of different dimensional outputs
    
    Args:
        network: Model to evaluate
        loader: Data loader (returns 0.0 if None)
        weights: Sample weights (optional)
        usedpredict: Prediction method ('p' for predict, otherwise predict1)
        
    Returns:
        Accuracy score (float)
    """
    if loader is None:
        return 0.0
    
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()

            if transform_fn:
                x = transform_fn(x)
            # Select prediction method
            if usedpredict == 'p':
                p = network.predict(x)
            else:
                p = network.predict1(x)
            
            # Handle multi-dimensional outputs
            if p.dim() > 2:
                p = p.squeeze(1)
            
            # Handle sample weights
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:weights_offset + len(x)]
                weights_offset += len(x)
            
            batch_weights = batch_weights.cuda()
            
            # Calculate correct predictions
            if p.size(1) == 1:  # Binary classification
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:  # Multiclass classification
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            
            total += batch_weights.sum().item()
    
    network.train()
    return correct / total if total > 0 else 0.0

def predict_proba(network, x):
    """
    Predict class probabilities with safety checks
    
    Args:
        network: Model to use for prediction
        x: Input tensor
        
    Returns:
        Class probabilities tensor
    """
    network.eval()
    with torch.no_grad():
        x = x.cuda().float()
        logits = network.predict(x)
        
        # Handle multi-dimensional outputs
        if logits.dim() > 2:
            logits = logits.squeeze(1)
            
        probs = torch.nn.functional.softmax(logits, dim=1)
    network.train()
    return probs
