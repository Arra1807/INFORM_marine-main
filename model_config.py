from adaptation_method.Autoencoder import ResNetAutoEncoder, Autoencoder
import wandb
def Configuration():    
    run = wandb.init(
        project= 'Adapter',
        config = { 
            'batch_size': 64,
            'learning_rate': 1e-3,        
            'epochs': 20,
            'Weight_decay': 1e-4
        }, 
    )
    return run