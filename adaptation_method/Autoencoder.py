import torch    
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

def spatial_preserving_resnet(in_channels = 6):
    model = resnet34(weights = ResNet34_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size= 3, stride=1, padding=1, bias= False)
    model.bn1 = nn.BatchNorm2d(64)
    model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return model

def freeze_layers(model):
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    return model

class ResNetAutoEncoder(nn.Module):
    """ResNet-based Autoencoder with spatial preserving architecture."""
    def __init__(self, in_channels = 6, latent_dim = 3):
        super().__init__()
        model = spatial_preserving_resnet(in_channels) 
        model = freeze_layers(model)  
    
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        self.compressor = nn.Sequential(
          nn.Conv2d(512, latent_dim, kernel_size =1), 
          nn.BatchNorm2d(latent_dim),
          nn.Sigmoid()
        )
            
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, in_channels, kernel_size=3, padding=1), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
     encoded = self.encoder(x)
     compressed = self.compressor(encoded)
     reconstructed = self.decoder(compressed)
     return reconstructed, compressed  



import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, in_channels = 6, out_channels = 3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding = 1),
            nn.GroupNorm(4,16),
            nn.ReLU(inplace= True),
            nn.Conv2d(16,16, kernel_size= 3, padding= 1),
            nn.GroupNorm(4,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace= True),
            nn.Softmax(dim = 1)
        )  
        
        
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )    
            
    def forward(self, x):
        z = self.encoder(x)
        perm = torch.randperm(z.size(1), device=z.device)
        z_perm = z[:, perm,:, :]
        x_hat = self.decoder(z_perm)
        return x_hat, z_perm



    
