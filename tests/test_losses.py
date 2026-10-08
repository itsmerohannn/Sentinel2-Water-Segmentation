import unittest
import torch
import math
from src.losses.sam_loss import SpectralAngleMapperLoss
from src.losses.compound_loss import CompoundBCEDiceLoss

class TestLosses(unittest.TestCase):
    def test_sam_loss_identical_spectra(self):
        sam_fn = SpectralAngleMapperLoss()
        x = torch.rand(2, 13, 32, 32) + 0.1 # Strictly positive
        # Scaled identical spectrum
        x_scaled = x * 2.5
        
        loss = sam_fn(x, x_scaled)
        # Angle between parallel vectors should be approximately 0.0 (within eps clamp)
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_sam_loss_orthogonal_spectra(self):
        sam_fn = SpectralAngleMapperLoss()
        # Create orthogonal vectors in spectral dimension
        x = torch.zeros(1, 2, 1, 1)
        y = torch.zeros(1, 2, 1, 1)
        x[0, 0, 0, 0] = 1.0 # [1, 0]
        y[0, 1, 0, 0] = 1.0 # [0, 1]
        
        loss = sam_fn(x, y)
        expected_rad = math.pi / 2.0
        self.assertAlmostEqual(loss.item(), expected_rad, places=3)

    def test_sam_loss_gradient(self):
        sam_fn = SpectralAngleMapperLoss()
        x = torch.rand(2, 13, 16, 16, requires_grad=True)
        target = torch.rand(2, 13, 16, 16)
        loss = sam_fn(x, target)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_compound_loss(self):
        loss_fn = CompoundBCEDiceLoss(alpha=0.5, beta=0.5)
        
        # High confidence match (logits large positive for 1s, large negative for 0s)
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        good_logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
        bad_logits = torch.tensor([[[[-10.0, 10.0], [10.0, -10.0]]]])

        loss_good = loss_fn(good_logits, targets).item()
        loss_bad = loss_fn(bad_logits, targets).item()

        self.assertLess(loss_good, loss_bad)
        self.assertGreaterEqual(loss_good, 0.0)

if __name__ == "__main__":
    unittest.main()
