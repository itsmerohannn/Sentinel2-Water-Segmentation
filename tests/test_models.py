import unittest
import torch
from src.models.autoencoder_1d import SpectralAutoencoder1D
from src.models.compact_unet import CompactUNet

class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.patch_size = 64 # Small patch for fast testing
        self.in_bands = 13
        self.latent_dim = 4
        self.dummy_input = torch.rand(self.batch_size, self.in_bands, self.patch_size, self.patch_size)

    def test_spectral_autoencoder_shapes(self):
        ae = SpectralAutoencoder1D(in_bands=self.in_bands, latent_channels=self.latent_dim)
        latent = ae.encode(self.dummy_input)
        self.assertEqual(latent.shape, (self.batch_size, self.latent_dim, self.patch_size, self.patch_size))

        recon = ae.decode(latent)
        self.assertEqual(recon.shape, (self.batch_size, self.in_bands, self.patch_size, self.patch_size))

        # Check values are constrained to [0, 1] by sigmoid
        self.assertTrue((recon >= 0.0).all() and (recon <= 1.0).all())

    def test_compact_unet_shape(self):
        unet = CompactUNet(in_channels=self.latent_dim, out_channels=1)
        dummy_latent = torch.rand(self.batch_size, self.latent_dim, self.patch_size, self.patch_size)
        logits = unet(dummy_latent)
        self.assertEqual(logits.shape, (self.batch_size, 1, self.patch_size, self.patch_size))

    def test_backward_pass(self):
        ae = SpectralAutoencoder1D(in_bands=self.in_bands, latent_channels=self.latent_dim)
        latent, recon = ae(self.dummy_input)
        loss = recon.mean()
        loss.backward()

        unet = CompactUNet(in_channels=self.latent_dim, out_channels=1)
        logits = unet(latent.detach())
        loss_unet = logits.mean()
        loss_unet.backward()

if __name__ == "__main__":
    unittest.main()
