import wandb
def Configuration(model_name = ''):    
    run = wandb.init(
        project= 'Adapter',
        name = f'{model_name}_run',
        reinit=True,
        config = {
            'learning_rate': 1e-3,        
            'epochs': 10,
            'Weight_decay': 1e-4
        }, 
    )
    return run
