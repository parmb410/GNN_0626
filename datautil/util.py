import numpy as np
import torch
from torch_geometric.data import Data

def Nmax(args, d):
    """Find the position index for a given domain"""
    for i in range(len(args.test_envs)):
        if d < args.test_envs[i]:
            return i
    return len(args.test_envs)

class basedataset(torch.utils.data.Dataset):
    """Base dataset class with PyTorch Dataset inheritance"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class mydataset(torch.utils.data.Dataset):
    """Enhanced dataset class with improved functionality"""
    def __init__(self, args):
        super().__init__()
        self.x = None
        self.labels = None
        self.dlabels = None
        self.pclabels = None
        self.pdlabels = None
        self.task = None
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.loader = None
        self.args = args
        
        # Add data statistics tracking
        self.mean = None
        self.std = None
        self.class_distribution = None

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'pclabel':
            self.pclabels = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels = tlabels
        elif label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pclabel':
            self.pclabels[tindex] = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels
        elif label_type == 'domain_label':
            self.dlabels[tindex] = tlabels
        elif label_type == 'class_label':
            self.labels[tindex] = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def compute_statistics(self):
        """Compute and store dataset statistics for normalization"""
        if isinstance(self.x, np.ndarray):
            self.mean = np.mean(self.x, axis=0)
            self.std = np.std(self.x, axis=0)
        elif torch.is_tensor(self.x):
            self.mean = torch.mean(self.x, dim=0)
            self.std = torch.std(self.x, dim=0)
        
        # Compute class distribution
        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            self.class_distribution = dict(zip(unique, counts))
            print("Class distribution:", self.class_distribution)

    def normalize(self):
        """Normalize dataset using computed statistics"""
        if self.mean is None or self.std is None:
            self.compute_statistics()
            
        # Apply normalization
        if isinstance(self.x, np.ndarray):
            self.x = (self.x - self.mean) / (self.std + 1e-8)
        elif torch.is_tensor(self.x):
            self.x = (self.x - self.mean) / (self.std + 1e-8)

    def __getitem__(self, index):
        # Convert index to Python int
        if isinstance(index, np.integer):
            index = int(index)
        elif isinstance(index, np.ndarray):
            index = index.item()
        
        # Get and convert data
        x = self.input_trans(self.x[index])
        ctarget = self.labels[index] if self.labels is not None else -1
        dtarget = self.dlabels[index] if self.dlabels is not None else -1
        pctarget = self.pclabels[index] if self.pclabels is not None else -1
        pdtarget = self.pdlabels[index] if self.pdlabels is not None else -1
        
        # Convert numpy types to Python primitives
        if isinstance(ctarget, np.generic):
            ctarget = ctarget.item()
        if isinstance(dtarget, np.generic):
            dtarget = dtarget.item()
        if isinstance(pctarget, np.generic):
            pctarget = pctarget.item()
        if isinstance(pdtarget, np.generic):
            pdtarget = pdtarget.item()
            
        # For GNN models, ensure x is a Data object
        if hasattr(self.args, 'use_gnn') and self.args.use_gnn and not isinstance(x, Data):
            x = Data(x=x, edge_index=torch.tensor([[], []], dtype=torch.long), edge_attr=torch.tensor([]))
            
        return x, ctarget, dtarget, pctarget, pdtarget, index

    def __len__(self):
        return len(self.x)

class subdataset(mydataset):
    """Subset of a dataset"""
    def __init__(self, args, dataset, indices):
        super().__init__(args)
        
        # Convert indices to integer list
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, (torch.Tensor, np.ndarray)):
            indices = indices.tolist()
        indices = [int(i) for i in indices]
        
        # Extract data
        self.x = dataset.x[indices]
        self.labels = dataset.labels[indices] if dataset.labels is not None else None
        self.dlabels = dataset.dlabels[indices] if dataset.dlabels is not None else None
        self.pclabels = dataset.pclabels[indices] if dataset.pclabels is not None else None
        self.pdlabels = dataset.pdlabels[indices] if dataset.pdlabels is not None else None
        
        self.loader = dataset.loader
        self.task = dataset.task
        self.dataset = dataset.dataset
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        
        # Copy statistics
        self.mean = dataset.mean
        self.std = dataset.std
        self.class_distribution = dataset.class_distribution

class combindataset(mydataset):
    """Combined dataset from multiple sources"""
    def __init__(self, args, datalist):
        super().__init__(args)
        self.domain_num = len(datalist)
        self.loader = datalist[0].loader
        
        xlist = [item.x for item in datalist]
        cylist = [item.labels for item in datalist]
        dylist = [item.dlabels for item in datalist]
        pcylist = [item.pclabels for item in datalist]
        pdylist = [item.pdlabels for item in datalist]
        
        self.dataset = datalist[0].dataset
        self.task = datalist[0].task
        self.transform = datalist[0].transform
        self.target_transform = datalist[0].target_transform
        
        # Handle different data types
        if all(torch.is_tensor(x) for x in xlist):
            self.x = torch.vstack(xlist)
        else:
            self.x = np.vstack([x.numpy() if torch.is_tensor(x) else x for x in xlist])
        
        # Convert labels to tensors to avoid numpy types
        self.labels = torch.tensor(np.hstack(cylist), dtype=torch.long)
        self.dlabels = torch.tensor(np.hstack(dylist), dtype=torch.long)
        self.pclabels = torch.tensor(np.hstack(pcylist), dtype=torch.long) if pcylist[0] is not None else None
        self.pdlabels = torch.tensor(np.hstack(pdylist), dtype=torch.long) if pdylist[0] is not None else None
        
        # Compute new statistics for combined dataset
        self.compute_statistics()
