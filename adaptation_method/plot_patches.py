import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_patches(data, tuple_frequencies, num_chan = 6, num_patches=5):
    """
    Plot echo patches from the dataset
    Args:
        data: tensor of shape [batch_size, channels, height, width]
        num_patches: number of patches to display
    """
    # Set up the figure
    fig, axes = plt.subplots(num_chan, num_patches, figsize=(15, 18))
    
    # Plot patches for all frequency channels
    for i in range(num_patches):
        if i < data.shape[0]:
            for freq in range(num_chan):  # Loop through all 6 frequency channels
                axes[freq, i].imshow(data[i, freq].cpu().numpy())
                axes[freq, i].set_title(f'Patch {i+1}\nFreq: {tuple_frequencies[freq]} kHz')
                axes[freq, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def plot_latent_patches(latent, num_chan = 3, num_patches= 10):
    """
    Plot latent patches from the autoencoder
    Args:
        latent: tensor of shape [batch_size, channels, height, width]
        num_patches: number of patches to display
        """
    # Set up the figure
    fig, axes = plt.subplots(num_chan, num_patches, figsize=(15, 9))
    
    # Plot patches for all frequency channels
    for i in range(num_patches):
        if i < latent.shape[0]:
            for freq in range(num_chan):  # Loop through all 3 latent channels (R,G,B)
                axes[freq, i].imshow(latent[i, freq].cpu().numpy(), cmap='viridis')
                axes[freq, i].set_title(f'Patch {i+1}\nChannel: {["R","G","B"][freq]}')
                axes[freq, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    
def plot_latent_dominance(latent, num_patches=3):
    """
    Plot latent patches showing RGB channel dominance
    Args:
        latent: tensor of shape [batch_size, 3, height, width]
        num_patches: number of patches to display
    """
    fig, axes = plt.subplots(4, num_patches, figsize=(15, 12))
    
    for i in range(num_patches):
        if i < latent.shape[0]:
            # Original RGB combination
            rgb_image = np.stack([
                latent[i, 0].cpu().numpy(),  # R channel
                latent[i, 1].cpu().numpy(),  # G channel
                latent[i, 2].cpu().numpy()   # B channel
            ], axis=2)
            axes[0, i].imshow(rgb_image)
            axes[0, i].set_title(f'Patch {i+1}\nRGB Combined')
            axes[0, i].axis('off')
            
            # Channel dominance map
            channels = ['Red', 'Green', 'Blue']
            data = latent[i].cpu().numpy()
            dominant = np.argmax(data, axis=0)  # Find dominant channel at each position
            
            # Create dominance heatmap
            heatmap = np.zeros((*dominant.shape, 3))
            colors = [[1,0,0], [0,1,0], [0,0,1]]  # RGB colors
            for ch in range(3):
                mask = (dominant == ch)
                heatmap[mask] = colors[ch]
            
            axes[1, i].imshow(heatmap)
            axes[1, i].set_title('Channel Dominance')
            axes[1, i].axis('off')
            
            # Individual channel intensities
            for ch in range(3):
                axes[ch+2, i].imshow(data[ch], cmap='hot')
                axes[ch+2, i].set_title(f'{channels[ch]} Intensity')
                axes[ch+2, i].axis('off')
    
    plt.tight_layout()
    plt.show()