"""
AID Dataset loader for Rep-Mamba super-resolution.

AID (Aerial Image Dataset) – Xia et al., TGRS 2017
  30 scene categories, ~10 000 images total.

Split used in the paper (§IV-A):
  Train : 100 images/category × 30 categories = 3 000 images
  Test  :  30 images/category × 30 categories =   900 images
  All images cropped to 512 × 512 px.

Expected directory layout (set --data_root to its parent):

  <data_root>/
    Airport/
      airport_001.jpg
      ...
    Bare Land/
      ...
    ...   (30 categories)

Download: https://captain-whu.github.io/DiRS/
"""

import os
import random
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────────────────────
# AID category names (30 classes)
# ──────────────────────────────────────────────────────────────
AID_CLASSES = [
    'Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge',
    'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert',
    'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential',
    'Mountain', 'Park', 'Parking', 'Playground', 'Pond',
    'Port', 'RailwayStation', 'Resort', 'River', 'School',
    'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct',
]


# ──────────────────────────────────────────────────────────────
# Helper: build file list from AID root
# ──────────────────────────────────────────────────────────────

def build_aid_split(data_root: str, n_train: int = 100, n_test: int = 30,
                    seed: int = 42):
    """
    Scan all subdirectories under *data_root* and split into train/test.

    Returns
    -------
    train_paths, test_paths : list[Path]
    """
    root = Path(data_root)
    all_images = sorted(root.rglob('*.jpg')) + sorted(root.rglob('*.png'))

    # Group by category (sub-directory name)
    category_map: dict[str, list] = {}
    for p in all_images:
        cat = p.parent.name
        category_map.setdefault(cat, []).append(p)

    rng = random.Random(seed)
    train_paths, test_paths = [], []
    for cat, imgs in sorted(category_map.items()):
        imgs_sorted = sorted(imgs)
        rng.shuffle(imgs_sorted)
        train_paths.extend(imgs_sorted[:n_train])
        test_paths.extend(imgs_sorted[n_train: n_train + n_test])

    return train_paths, test_paths


# ──────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────

class AIDTrainDataset(Dataset):
    """
    Returns (LR, HR) patch pairs for training.

    Pipeline:
      1. Load full image (any size).
      2. Random crop to hr_size × hr_size (default 256 for ×4).
      3. Random horizontal / vertical flip + 90° rotation.
      4. Downscale by *scale* with bicubic → LR patch.
    """

    def __init__(self, image_paths: list,
                 scale: int = 4,
                 lr_patch_size: int = 64,
                 augment: bool = True):
        self.paths = image_paths
        self.scale = scale
        self.lr_size = lr_patch_size
        self.hr_size = lr_patch_size * scale
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')

        # ---- crop to hr_size × hr_size ----
        w, h = img.size
        # Pad if image is smaller than needed
        if w < self.hr_size or h < self.hr_size:
            pad_w = max(0, self.hr_size - w)
            pad_h = max(0, self.hr_size - h)
            img = TF.pad(img, (0, 0, pad_w, pad_h), padding_mode='reflect')
            w, h = img.size

        # Random crop
        x0 = random.randint(0, w - self.hr_size)
        y0 = random.randint(0, h - self.hr_size)
        hr = TF.crop(img, y0, x0, self.hr_size, self.hr_size)

        # ---- data augmentation ----
        if self.augment:
            if random.random() > 0.5:
                hr = TF.hflip(hr)
            if random.random() > 0.5:
                hr = TF.vflip(hr)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                hr = TF.rotate(hr, angle)

        # ---- generate LR via bicubic downscale ----
        lr = hr.resize(
            (self.lr_size, self.lr_size),
            resample=Image.BICUBIC
        )

        hr_t = TF.to_tensor(hr)   # [0, 1]
        lr_t = TF.to_tensor(lr)

        return lr_t, hr_t


class AIDTestDataset(Dataset):
    """
    Returns (LR, HR) pairs for evaluation.

    HR: centre-crop to nearest multiple of *scale* (up to 512×512).
    LR: bicubic downscale of HR.
    """

    def __init__(self, image_paths: list,
                 scale: int = 4,
                 max_size: int = 512):
        self.paths = image_paths
        self.scale = scale
        self.max_size = max_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        w, h = img.size

        # Crop to max_size × max_size (centre crop)
        crop_w = min(w, self.max_size)
        crop_h = min(h, self.max_size)
        # Make divisible by scale
        crop_w = (crop_w // self.scale) * self.scale
        crop_h = (crop_h // self.scale) * self.scale

        hr = TF.center_crop(img, (crop_h, crop_w))
        lr = hr.resize(
            (crop_w // self.scale, crop_h // self.scale),
            resample=Image.BICUBIC
        )

        hr_t = TF.to_tensor(hr)
        lr_t = TF.to_tensor(lr)

        return lr_t, hr_t, str(self.paths[idx])


# ──────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────

def get_dataloaders(data_root: str,
                    scale: int = 4,
                    lr_patch_size: int = 64,
                    batch_size: int = 16,
                    num_workers: int = 4,
                    n_train: int = 100,
                    n_test: int = 30,
                    seed: int = 42):
    """
    Build train/test DataLoaders for AID.

    Returns (train_loader, test_loader, n_train_total, n_test_total).
    """
    train_paths, test_paths = build_aid_split(data_root, n_train, n_test, seed)
    print(f"[Dataset] Train: {len(train_paths)} | Test: {len(test_paths)} images")

    train_ds = AIDTrainDataset(train_paths, scale=scale,
                               lr_patch_size=lr_patch_size, augment=True)
    test_ds = AIDTestDataset(test_paths, scale=scale)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,         # full images → variable size
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, len(train_paths), len(test_paths)


# ──────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else './data/AID'
    trl, tel, n_tr, n_te = get_dataloaders(root, scale=4, batch_size=4, num_workers=0)
    lr, hr = next(iter(trl))
    print(f"Train batch  LR={lr.shape}  HR={hr.shape}")
    lr, hr, path = next(iter(tel))
    print(f"Test  sample LR={lr.shape}  HR={hr.shape}  path={path[0]}")
