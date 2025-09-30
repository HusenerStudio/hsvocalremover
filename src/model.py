import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class UNetVocalRemover(nn.Module):
    """U-Net architecture for vocal separation"""
    
    def __init__(self, input_channels=2, output_channels=2, hidden_channels=64, num_layers=6):
        super(UNetVocalRemover, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        in_ch = input_channels
        
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** i)
            if i == 0:
                # First layer without stride
                self.encoder_blocks.append(ConvBlock(in_ch, out_ch, stride=1))
            else:
                self.encoder_blocks.append(ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, in_ch * 2)
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        
        in_ch = in_ch * 2  # from bottleneck
        
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** (num_layers - i - 1))
            
            # Upsampling
            if i < num_layers - 1:
                self.upsampling_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))
            else:
                self.upsampling_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, 3, 1, 1))
            
            # Skip connection: double the channels
            self.decoder_blocks.append(ConvBlock(out_ch * 2, out_ch))
            in_ch = out_ch
        
        # Final output layer
        self.final_conv = nn.Conv2d(hidden_channels, output_channels, 1)
        self.final_activation = nn.Tanh()
        
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            if i < len(self.encoder_blocks) - 1:  # Don't store the last encoder output
                skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, (upsample, decoder_block) in enumerate(zip(self.upsampling_blocks, self.decoder_blocks)):
            x = upsample(x)
            
            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]  # Reverse order
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            x = decoder_block(x)
        
        # Final output
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss for audio separation"""
    
    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()
        
    def forward(self, pred, target):
        # Compute spectral convergence
        numerator = torch.norm(target - pred, p='fro', dim=(-2, -1))
        denominator = torch.norm(target, p='fro', dim=(-2, -1))
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)
        
        return torch.mean(numerator / denominator)


class MagnitudeLoss(nn.Module):
    """Magnitude loss for audio separation"""
    
    def __init__(self):
        super(MagnitudeLoss, self).__init__()
        
    def forward(self, pred, target):
        pred_mag = torch.abs(pred)
        target_mag = torch.abs(target)
        return F.l1_loss(pred_mag, target_mag)


class CombinedLoss(nn.Module):
    """Combined loss function for vocal separation"""
    
    def __init__(self, alpha=1.0, beta=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.spectral_loss = SpectralConvergenceLoss()
        self.magnitude_loss = MagnitudeLoss()
        
    def forward(self, pred, target):
        spec_loss = self.spectral_loss(pred, target)
        mag_loss = self.magnitude_loss(pred, target)
        return self.alpha * spec_loss + self.beta * mag_loss


def create_model(config):
    """Create model based on configuration"""
    model_config = config['model']
    
    model = UNetVocalRemover(
        input_channels=model_config['input_channels'],
        output_channels=model_config['output_channels'],
        hidden_channels=model_config['hidden_channels'],
        num_layers=model_config['num_layers']
    )
    
    return model


def create_loss_function(config):
    """Create loss function based on configuration"""
    loss_config = config['loss']
    loss_type = loss_config['type']
    
    if loss_type == 'spectral_convergence':
        return SpectralConvergenceLoss()
    elif loss_type == 'magnitude':
        return MagnitudeLoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'combined':
        return CombinedLoss(
            alpha=loss_config['alpha'],
            beta=loss_config['beta']
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")