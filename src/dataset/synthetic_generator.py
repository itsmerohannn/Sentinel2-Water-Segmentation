import numpy as np
import torch
import os
from typing import Dict, Tuple

# Reference physical reflectance signatures across 13 Sentinel-2 bands [0.0 - 1.0]
# Bands: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
WATER_SPECTRUM = np.array([0.08, 0.10, 0.09, 0.04, 0.02, 0.015, 0.01, 0.01, 0.008, 0.005, 0.001, 0.003, 0.002])
VEGETATION_SPECTRUM = np.array([0.03, 0.04, 0.08, 0.04, 0.15, 0.35, 0.45, 0.50, 0.52, 0.12, 0.02, 0.22, 0.12])
URBAN_SOIL_SPECTRUM = np.array([0.12, 0.14, 0.16, 0.19, 0.22, 0.25, 0.27, 0.29, 0.30, 0.15, 0.04, 0.35, 0.32])

def generate_synthetic_patch(patch_size: int = 128, has_water: bool = True, rng: np.random.RandomState = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single synthetic Sentinel-2 13-band patch and binary water mask.
    Returns:
        image: (13, H, W) in Digital Numbers [0, 10000]
        mask: (1, H, W) binary mask {0, 1}
    """
    if rng is None:
        rng = np.random.RandomState()
        
    mask = np.zeros((patch_size, patch_size), dtype=np.float32)
    image = np.zeros((13, patch_size, patch_size), dtype=np.float32)
    
    # Background is a mix of vegetation and urban/soil
    blend = rng.uniform(0.3, 0.7, size=(patch_size, patch_size))
    for b in range(13):
        bg_reflectance = blend * VEGETATION_SPECTRUM[b] + (1.0 - blend) * URBAN_SOIL_SPECTRUM[b]
        image[b, :, :] = bg_reflectance
        
    if has_water:
        # Create a synthetic geometric water body 
        center_x, center_y = rng.randint(patch_size // 4, 3 * patch_size // 4, size=2)
        radius = rng.randint(patch_size // 6, patch_size // 3)
        
        y, x = np.ogrid[:patch_size, :patch_size]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        water_region = dist_from_center <= radius
        
        mask[water_region] = 1.0
        for b in range(13):
            image[b, water_region] = WATER_SPECTRUM[b]
            
    # Add minor sensor noise
    noise = rng.normal(0.0, 0.005, size=image.shape)
    image = np.clip(image + noise, 0.0, 1.0)
    
    # Scale to Digital Numbers (DN) ~ 0 to 10000
    image_dn = (image * 10000.0).astype(np.float32)
    return image_dn, mask[np.newaxis, :, :]


def generate_synthetic_sentinel2_dataset(
    output_dir: str,
    n_train: int = 40,
    n_val: int = 10,
    n_test: int = 10,
    patch_size: int = 128,
    seed: int = 42
):
    """
    Creates a mock Sentinel-2 dataset partitioned into train/val/test directories.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    
    splits = {
        "train": n_train,
        "val": n_val,
        "test": n_test
    }
    
    for split_name, count in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks"), exist_ok=True)
        
        for i in range(count):
            # 75% water patches, 25% non-water patches
            has_water = rng.rand() < 0.75
            img, mask = generate_synthetic_patch(patch_size=patch_size, has_water=has_water, rng=rng)
            
            patch_id = f"scene_{split_name}_{i:04d}"
            np.save(os.path.join(split_dir, "images", f"{patch_id}.npy"), img)
            np.save(os.path.join(split_dir, "masks", f"{patch_id}.npy"), mask)
            
    print(f"Generated synthetic Sentinel-2 dataset at: {output_dir}")
