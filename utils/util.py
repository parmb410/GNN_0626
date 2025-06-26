import random
import numpy as np
import torch
import sys
import os
import argparse
import torchvision
import PIL
from collections import defaultdict

def disable_inplace_relu(model):
    """Disable inplace operations in ReLU layers for compatibility"""
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

def set_random_seed(seed=0):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_valid_target_eval_names(args):
    """Generate evaluation names based on domain configuration"""
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(f'eval{i}_in')
            eval_name_dict['valid'].append(f'eval{i}_out')
        else:
            eval_name_dict['target'].append(f'eval{i}_out')
    return eval_name_dict

def alg_loss_dict(args):
    """Get loss names for different algorithms"""
    loss_dict = {
        'diversify': ['class', 'dis', 'total']
    }
    return loss_dict.get(args.algorithm, ['total'])

def print_args(args, print_list=[]):
    """Print selected arguments in a formatted way"""
    s = "==========================================\n"
    if not print_list:  # Print all arguments if list is empty
        print_list = args.__dict__.keys()
    
    for arg in print_list:
        if hasattr(args, arg):
            s += f"{arg}: {getattr(args, arg)}\n"
    return s

def print_row(row, colwidth=10, latex=False):
    """Print a formatted row of values"""
    sep = " & " if latex else "  "
    end_ = "\\\\" if latex else ""
    
    def format_val(x):
        if isinstance(x, float):
            return f"{x:.6f}".ljust(colwidth)[:colwidth]
        return str(x).ljust(colwidth)[:colwidth]
    
    print(sep.join(format_val(x) for x in row), end_)

def print_environ():
    """Print environment information"""
    print("Environment:")
    print(f"\tPython: {sys.version.split(' ')[0]}")
    print(f"\tPyTorch: {torch.__version__}")
    print(f"\tTorchvision: {torchvision.__version__}")
    print(f"\tCUDA: {torch.version.cuda}")
    print(f"\tCUDNN: {torch.backends.cudnn.version()}")
    print(f"\tNumPy: {np.__version__}")
    print(f"\tPIL: {PIL.__version__}")

class Tee:
    """Tee output to both console and file"""
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)
    
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()

def act_param_init(args):
    """Initialize activity recognition parameters"""
    # Default parameters for EMG dataset
    args.select_position = {'emg': [0]}
    args.select_channel = {'emg': np.arange(8)}
    args.hz_list = {'emg': 1000}
    args.act_people = {'emg': [[i*9 + j for j in range(9)] for i in range(4)]}
    
    # Dataset-specific parameters
    dataset_params = {
        'emg': ((8, 1, 200), 6, 10)  # (input_shape, num_classes, grid_size)
    }
    
    # Set parameters based on dataset
    params = dataset_params.get(args.dataset, ((0, 0, 0), 0, 0))
    args.input_shape = params[0]
    args.num_classes = params[1]
    args.grid_size = params[2]
    
    return args

def get_args():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Domain Generalization for Activity Recognition')
    
    # Algorithm parameters
    parser.add_argument('--algorithm', type=str, default="diversify", 
                        help="Algorithm to use")
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help="DANN discriminator alpha")
    parser.add_argument('--alpha1', type=float, default=0.1, 
                        help="DANN discriminator alpha for auxiliary network")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size")
    parser.add_argument('--beta1', type=float, default=0.5, 
                        help="Adam beta1 parameter")
    parser.add_argument('--checkpoint_freq', type=int, default=100, 
                        help='Checkpoint every N steps')
    parser.add_argument('--local_epoch', type=int, default=1, 
                        help='Local iterations per round')
    parser.add_argument('--max_epoch', type=int, default=120, 
                        help="Max training epochs")
    parser.add_argument('--lr', type=float, default=1e-2, 
                        help="Learning rate")
    parser.add_argument('--lr_decay1', type=float, default=1.0, 
                        help='LR decay for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0, 
                        help='LR decay for other components')
    parser.add_argument('--weight_decay', type=float, default=5e-4, 
                        help="Weight decay")
    
    # ====== ADDED REGULARIZATION PARAMETERS ======
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Dropout probability for regularization")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help="Label smoothing epsilon for CrossEntropyLoss")
    # ====== END ADDED REGULARIZATION PARAMETERS ======
    
    # Model architecture
    parser.add_argument('--bottleneck', type=int, default=256, 
                        help="Bottleneck dimension")
    parser.add_argument('--classifier', type=str, default="linear", 
                        choices=["linear", "wn"], 
                        help="Classifier type")
    parser.add_argument('--dis_hidden', type=int, default=256, 
                        help="Discriminator hidden dimension")
    parser.add_argument('--layer', type=str, default="bn", 
                        choices=["ori", "bn"], 
                        help="Bottleneck layer type")
    parser.add_argument('--model_size', default='median',
                        choices=['small', 'median', 'large', 'transformer'],
                        help="Model size variant")
    
    # Domain parameters
    parser.add_argument('--lam', type=float, default=0.0, 
                        help="Entropy regularization weight")
    parser.add_argument('--latent_domain_num', type=int, default=None, 
                        help="Number of latent domains (None for auto estimation)")
    parser.add_argument('--domain_num', type=int, default=0, 
                        help="Number of domains (auto-set during data loading)")
    
    # Data parameters
    parser.add_argument('--data_file', type=str, default='', 
                        help="Base data directory")
    parser.add_argument('--dataset', type=str, default='emg', 
                        help="Dataset name")
    parser.add_argument('--data_dir', type=str, default='', 
                        help="Data subdirectory")
    parser.add_argument('--task', type=str, default="cross_people", 
                        help="Task type")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0], 
                        help="Test environment indices")
    parser.add_argument('--N_WORKERS', type=int, default=4, 
                        help="Number of data workers")
    
    # Experimental features
    parser.add_argument('--automated_k', action='store_true', 
                        help='Enable automated K estimation')
    parser.add_argument('--curriculum', action='store_true', 
                        help='Enable curriculum learning')
    parser.add_argument('--CL_PHASE_EPOCHS', type=int, default=5, 
                        help='Epochs to apply curriculum learning')
    parser.add_argument('--enable_shap', action='store_true', 
                        help='Enable SHAP explainability')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Checkpoint path for SHAP evaluation')

    # ====== ADDED DEBUGGING PARAMETER ======
    parser.add_argument('--debug_mode', action='store_true',
                        help='Enable debugging mode with small dataset subset')
    # ====== END ADDED DEBUGGING PARAMETER ======
    
    # ======== GNN PARAMETERS ========
    parser.add_argument('--use_gnn', action='store_true', 
                        help='Use GNN instead of CNN')
    parser.add_argument('--gnn_hidden_dim', type=int, default=32,
                        help='Hidden dimension for GNN layers')
    parser.add_argument('--gnn_output_dim', type=int, default=128,
                        help='Output dimension for GNN layers')
    parser.add_argument('--gnn_lr', type=float, default=0.001,
                        help='Learning rate for GNN pretraining')
    parser.add_argument('--gnn_weight_decay', type=float, default=0.0001,
                        help='Weight decay for GNN pretraining')
    parser.add_argument('--gnn_pretrain_epochs', type=int, default=5,
                        help='Number of GNN pretraining epochs')
    # ======== END GNN PARAMETERS ========
    
    # System parameters
    parser.add_argument('--gpu_id', type=str, default='0', 
                        help="GPU device id to run")
    parser.add_argument('--seed', type=int, default=0, 
                        help="Random seed")
    parser.add_argument('--output', type=str, default="train_output", 
                        help="Output directory")
    parser.add_argument('--old', action='store_true', 
                        help="Use old model version")
    
    args = parser.parse_args()
    
    # Post-processing
    args.steps_per_epoch = 10000000000  # Large number, will be adjusted later
    args.data_dir = os.path.join(args.data_file, args.data_dir)
    
    # Setup environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    
    # Initialize dataset-specific parameters
    args = act_param_init(args)
    
    return args
