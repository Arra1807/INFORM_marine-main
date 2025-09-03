import matplotlib.pyplot as plt

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