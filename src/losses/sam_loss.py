import torch
import torch.nn as nn
import math

class SpectralAngleMapperLoss(nn.Module):
    """
    Differentiable Spectral Angle Mapper (SAM) Loss for Hyperspectral / Multispectral data.
    
    Computes the spectral angle (in radians) between the ground truth spectral vector x
    and the reconstructed spectral vector x_hat for each pixel:
        theta = arccos( <x, x_hat> / (||x||_2 * ||x_hat||_2) )
    
    Args:
        eps (float): Small constant to avoid division by zero and numerical instability at +/- 1 in arccos.
        reduction (str): 'mean' or 'none'.
    """
    def __init__(self, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_pred: Reconstructed tensor of shape (B, C, H, W) or (B, C) or (N, C)
            x_true: Target tensor of the same shape
            
        Returns:
            Mean or per-pixel SAM loss in radians.
        """
        # Determine spectral dimension: channel dimension is dim 1
        dim = 1
        
        # Dot product across spectral bands: sum(x_pred * x_true, dim=1)
        dot_product = torch.sum(x_pred * x_true, dim=dim)
        
        # L2 Norms across spectral bands
        norm_pred = torch.linalg.norm(x_pred, ord=2, dim=dim)
        norm_true = torch.linalg.norm(x_true, ord=2, dim=dim)
        
        # Cosine similarity clamped for stable gradient backprop
        cos_sim = dot_product / (norm_pred * norm_true + self.eps)
        cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
        
        # Spectral angle in radians
        sam_rad = torch.acos(cos_sim)
        
        if self.reduction == "mean":
            return torch.mean(sam_rad)
        elif self.reduction == "none":
            return sam_rad
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    @staticmethod
    def rad_to_deg(rad_tensor: torch.Tensor) -> torch.Tensor:
        """Helper to convert radians to degrees."""
        return rad_tensor * (180.0 / math.pi)
