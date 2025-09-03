import torch
import wandb
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm 
from adaptation_method.EarlyStopping import EarlyStopping

def train_val_encoder(model, optimizer, Loss_func, num_epochs, train_dataloader, test_dataloader, run, device = None):
    avg_loss_train = []
    avg_loss_val = []
    best_val_loss = float('inf')
    earlystopping = EarlyStopping(patience= 10)
    stop_epoch = None
    model.to(device)
    
    wandb.watch(model, log="all", log_freq=100)
    
    #---Training---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        train_latent = []
        
        for label, train_data, mask in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        #for train_data in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            train_data = train_data.to(device, non_blocking = True)

            optimizer.zero_grad()
            
            outputs, latent = model(train_data)
            loss = Loss_func(outputs, train_data)
        
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            train_latent.append(latent.detach().cpu())


        train_avg_loss = epoch_loss / len(train_dataloader)
        avg_loss_train.append(train_avg_loss) 
        train_latent = torch.cat(train_latent, dim=0)

        print(f"Train encodings: min={train_latent.min():.4f}, max={train_latent.max():.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_latent = []
        with torch.no_grad():
            for label, test_data, mask in test_dataloader:
                test_data = test_data.to(device)
                
                val_outputs, batch_val_latent = model(test_data)
                loss = Loss_func(val_outputs, test_data)
                val_loss += loss.item()
                
                val_latent.append(batch_val_latent.detach().cpu())
                

        
                  
        val_avg_loss = val_loss / len(test_dataloader)
        avg_loss_val.append(val_avg_loss)
        val_latent = torch.cat(val_latent, dim=0)   
        
        print(f"Val latents: min={val_latent.min():.4f}, max={val_latent.max():.4f}")
        
        print(f" Train Loss = {train_avg_loss:.4f} ,Validation Loss = {val_avg_loss:.4f}")
        
        
        #Early stopping
        earlystopping(val_avg_loss)
        if earlystopping.early_stop and stop_epoch is None:
            stop_epoch = epoch
            print(f'Stopping early at epoch {epoch+1}')
            
        #Saving the best model
        best_model_path = 'best_autoencoder.pth'    
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_state = {
                'Model': model, 
                'state_dict':  model.state_dict(),
                'Optimizer': optimizer, 
                'Loss_func': Loss_func, 
                'Epochs:': num_epochs,
                'best_val_loss': best_val_loss,
            }
        torch.save(best_state, best_model_path)
        #print(f'Saved new best model at epoch {epoch+1} with val_loss = {val_avg_loss:.4f}')

            
        #Logging Hyperparameters
        run.log({
            'epoch': epoch+1, 
            'train_loss':  train_avg_loss,
            'val_loss': val_avg_loss,
            'Checkpoint/saved_at_epoch': epoch + 1,
            'Checkpoint/best_val_loss': best_val_loss
        })

    run.finish()

    return train_latent, val_latent, avg_loss_train, avg_loss_val, stop_epoch


def plot_loss(num_epochs, avg_loss_train, avg_loss_val, stop_epoch, run = None):
        plt.figure(figsize=(12,8))
        
        plt.plot(range(1, num_epochs+1), avg_loss_train, label = 'Training Loss')
        plt.plot(range(1, num_epochs+1), avg_loss_val, label = 'Validation Loss')
        if stop_epoch is not None:
            plt.axvline(x = stop_epoch, color = 'r', linestyle = '--', label = f'Early stop at Epoch{stop_epoch+1}')
            
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation')
        plt.legend()
        plt.grid(True)    
        if run is not None:
            run.log({'Loss curve': wandb.Image(plt)})
        plt.show()
    