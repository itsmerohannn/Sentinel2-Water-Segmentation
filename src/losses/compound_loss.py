import torch
import torch.nn as nn
import torch.nn.functional as F

class CompoundBCEDiceLoss(nn.Module):
    """
    Compound Loss Function combining Binary Cross Entropy (BCE) and Soft Dice Loss.
    Formulated specifically to handle foreground-background imbalance in remote sensing
    water body segmentation (Cao et al. 2024).
    
    L_seg = alpha * L_BCE + beta * L_Dice
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw network output logits of shape (B, 1, H, W)
            targets: Binary ground truth masks of shape (B, 1, H, W) with values in {0, 1}
            
        Returns:
            Scalar compound loss value.
        """
        # 1. Binary Cross-Entropy with Logits
        bce_loss = self.bce_fn(logits, targets)
        
        # 2. Soft Dice Loss
        probs = torch.sigmoid(logits)
        
        # Flatten across spatial dimensions
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat + targets_flat, dim=1)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = torch.mean(1.0 - dice_score)
        
        # Combined compound loss
        return self.alpha * bce_loss + self.beta * dice_loss
