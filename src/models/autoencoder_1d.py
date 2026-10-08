import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAutoencoder1D(nn.Module):
    """
    1D-Convolutional Autoencoder for Multispectral / Hyperspectral Data Compression
    (following Kuester et al. 2021 and Nalepa et al. 2020).
    
    Compresses 13 Sentinel-2 spectral bands along the spectral axis into a 4-channel
    latent representation while preserving spatial dimensions (H, W).
    
    Args:
        in_bands (int): Number of input spectral bands (default: 13 for Sentinel-2).
        latent_channels (int): Number of compressed bottleneck channels (default: 4).
    """
    def __init__(self, in_bands: int = 13, latent_channels: int = 4):
        super().__init__()
        self.in_bands = in_bands
        self.latent_channels = latent_channels

        # Encoder: 1D convolutions across contiguous spectral bands
        # Input to Conv1d: (N_pixels, 1, in_bands)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Linear projection to compressed latent channel dimension
        self.encoder_fc = nn.Linear(16 * in_bands, latent_channels)

        # Decoder: Expands latent vector back to 13 bands
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_channels, 16 * in_bands),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(), # Constrains output reflectance to [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses (B, in_bands, H, W) -> (B, latent_channels, H, W).
        """
        B, C, H, W = x.shape
        assert C == self.in_bands, f"Expected {self.in_bands} bands, got {C}"
        
        # Reshape to (B * H * W, 1, in_bands)
        # Permute (B, C, H, W) -> (B, H, W, C) -> (B * H * W, 1, C)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, 1, C)
        
        feat = self.encoder_conv(x_flat) # (N, 16, in_bands)
        feat = feat.view(feat.size(0), -1) # (N, 16 * in_bands)
        latent = self.encoder_fc(feat) # (N, latent_channels)
        
        # Reshape back to (B, latent_channels, H, W)
        latent = latent.view(B, H, W, self.latent_channels).permute(0, 3, 1, 2).contiguous()
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs (B, latent_channels, H, W) -> (B, in_bands, H, W).
        """
        B, C, H, W = latent.shape
        assert C == self.latent_channels, f"Expected {self.latent_channels} latent channels, got {C}"
        
        # Reshape (B, C, H, W) -> (B * H * W, C)
        latent_flat = latent.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        feat = self.decoder_fc(latent_flat) # (N, 16 * in_bands)
        feat = feat.view(-1, 16, self.in_bands) # (N, 16, in_bands)
        out = self.decoder_conv(feat) # (N, 1, in_bands)
        
        # Reshape back to (B, in_bands, H, W)
        out = out.view(B, H, W, self.in_bands).permute(0, 3, 1, 2).contiguous()
        return out

    def forward(self, x: torch.Tensor):
        """
        Returns:
            latent: (B, latent_channels, H, W)
            reconstruction: (B, in_bands, H, W)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
