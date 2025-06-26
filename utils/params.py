def get_params():
    paramname = {'diversify': [' --latent_domain_num ',
                               ' --alpha1 ',  ' --alpha ', ' --lam ', ' --use_gnn ', ' --gnn_hidden_dim ', ' --gnn_output_dim ']}
    paramlist = {
        'diversify': [
            # Expanded domain options (added 15,25)
            [2, 3, 5, 10, 15, 20, 25],
            
            # Finer adversarial control (added 0.01, 0.2)
            [0.01, 0.1, 0.2, 0.5, 1],
            
            # Wider alpha range (added 0.01, 5)
            [0.01, 0.1, 1, 5, 10],
            
            # Enable entropy regularization (added non-zero values)
            [0, 0.1, 0.5, 1],
            
            # GNN dimension enhancements
            [0, 1],  # use_gnn flag
            [32, 64, 128, 256],  # hidden_dim (increased max)
            [128, 256, 512]  # output_dim (increased max)
        ]
    }
    return paramname, paramlist
