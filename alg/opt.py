import torch

def get_params(alg, args, nettype):
    """
    Get parameter groups with customized learning rates
    Args:
        alg: Algorithm/model instance
        args: Configuration arguments
        nettype: Network type identifier
    Returns:
        List of parameter groups with learning rate settings
    """
    init_lr = args.lr
    
    if nettype == 'Diversify-adv':
        return [
            {'params': alg.dbottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.dclassifier.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.ddiscriminator.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    
    elif nettype == 'Diversify-cls':
        return [
            {'params': alg.bottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.discriminator.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    
    elif nettype == 'Diversify-all':
        return [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.abottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.aclassifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    
    # Default case: return all parameters with base learning rate
    return [{'params': alg.parameters(), 'lr': init_lr}]

def get_optimizer(alg, args, nettype):
    """
    Create optimizer for specified network components
    Args:
        alg: Algorithm/model instance
        args: Configuration arguments
        nettype: Network type identifier
    Returns:
        Configured Adam optimizer
    """
    params = get_params(alg, args, nettype)
    return torch.optim.Adam(
        params, 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        betas=(args.beta1, 0.9)
    )
