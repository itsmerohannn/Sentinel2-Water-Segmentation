import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class CompactUNet(nn.Module):
    """
    Compact 4-Level U-Net for Efficient Water Body Semantic Segmentation
    (adapted from Ronneberger et al. 2015 and Cao et al. 2024).
    
    Designed to operate directly on the 4-channel compressed latent feature maps
    from the 1D-CNN Spectral Autoencoder.
    
    Args:
        in_channels (int): Input channel dimension (default: 4 from frozen AE encoder).
        out_channels (int): Output segmentation class logits (default: 1 for binary water).
        features (list): Channel widths per level (default: [32, 64, 128, 256]).
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 1, features: list = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
            
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features[0], features[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features[1], features[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features[2], features[3]))

        self.up1 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features[3], features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features[2], features[1])

        self.up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features[1], features[0])

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with skip concatenations
        d3 = self.up1(x4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.conv_up1(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2)

        d1 = self.up3(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.conv_up3(d1)

        logits = self.outc(d1)
        return logits
