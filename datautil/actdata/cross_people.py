from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch

class ActList(mydataset):
    """
    Dataset class for cross-person activity recognition
    Combines data from multiple people and positions into a unified dataset
    """
    def __init__(self, args, dataset, root_dir, people_group, group_num, 
                 transform=None, target_transform=None, pclabels=None, 
                 pdlabels=None, shuffle_grid=True):
        """
        Initialize dataset
        Args:
            args: Configuration arguments
            dataset: Dataset name
            root_dir: Root directory of data
            people_group: List of people IDs to include
            group_num: Group identifier number
            transform: Input transformations
            target_transform: Target transformations
            pclabels: Precomputed class labels
            pdlabels: Precomputed domain labels
        """
        super(ActList, self).__init__(args)
        
        # Initialize dataset properties
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        
        # Load raw data
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        
        # Set up groups and positions
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))
        
        # Combine data from different people and positions
        self.comb_position(x, cy, py, sy)
        
        # Format data as tensors
        # Preserve both dimension expansion approaches
        self.x = self.x[:, :, np.newaxis, :]  # From first version
        self.transform = None  # From first version
        self.x = torch.tensor(self.x).float()
        
        # Handle pseudo-labels
        self.pclabels = pclabels if pclabels is not None else np.ones(self.labels.shape) * (-1)
        self.pdlabels = pdlabels if pdlabels is not None else np.ones(self.labels.shape) * 0
        
        # Domain labels
        self.tdlabels = np.ones(self.labels.shape) * group_num
        self.dlabels = np.ones(self.labels.shape) * (group_num - Nmax(args, group_num))

    def comb_position(self, x, cy, py, sy):
        """
        Combine data from different people and positions
        Args:
            x: Input features
            cy: Class labels
            py: Person IDs
            sy: Position/sensor IDs
        """
        # Preserve both implementation approaches
        # First version implementation
        for i, person_id in enumerate(self.people_group):
            # Get data for current person
            person_idx = np.where(py == person_id)[0]
            tx, tcy, tsy = x[person_idx], cy[person_idx], sy[person_idx]
            
            # Combine data from all positions for this person
            for j, position_id in enumerate(self.position):
                position_idx = np.where(tsy == position_id)[0]
                if j == 0:
                    ttx, ttcy = tx[position_idx], tcy[position_idx]
                else:
                    ttx = np.hstack((ttx, tx[position_idx]))
                    # Second version adds label stacking here
                    ttcy = np.hstack((ttcy, tcy[position_idx]))
            
            # Add to dataset
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))

    def set_x(self, x):
        """Update input features"""
        self.x = x
