from torchvision import transforms
import numpy as np
import torch
from datautil.graph_utils import convert_to_graph

class StandardScaler:
    """Normalize sensor data channel-wise"""
    def __call__(self, tensor):
        # tensor shape: [channels, timesteps, features]
        for c in range(tensor.size(0)):
            for f in range(tensor.size(2)):
                channel_data = tensor[c, :, f]
                mean = channel_data.mean()
                std = channel_data.std()
                if std > 0:
                    tensor[c, :, f] = (channel_data - mean) / std
                else:
                    tensor[c, :, f] = channel_data - mean
        return tensor

def act_train():
    """Original transformation for activity data"""
    return transforms.Compose([
        transforms.ToTensor(),
        StandardScaler(),
        lambda x: torch.tensor(x, dtype=torch.float32)
    ])

def act_to_graph_transform(args):
    """Transformation pipeline for GNN models"""
    return transforms.Compose([
        transforms.ToTensor(),
        StandardScaler(),
        # Changed to output [channels, time_steps] for GNN
        lambda x: x.squeeze(1)  # Remove middle dimension
    ])

def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/'):
    if dataset == 'pamap' and task == 'cross_people':
        x = np.load(root_dir+dataset+'/'+dataset+'_x1.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y1.npy')
    else:
        x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y.npy')
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy
