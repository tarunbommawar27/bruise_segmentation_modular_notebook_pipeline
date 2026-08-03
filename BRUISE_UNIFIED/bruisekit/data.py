"""One dataloader for every model: one disk read, one resize, one augmentation.

THE DATASET EMITS RAW [0,1] PIXELS -- IT DOES NOT NORMALISE
------------------------------------------------------------
SegFormer wants ImageNet-normalised input; YOLO wants plain /255. That is not a
style preference, it is a property of the trained weights: Ultralytics trains on
/255 and its BatchNorms carry frozen running statistics for that distribution.
Feeding YOLO ImageNet-normalised pixels makes it under-fire by 4x and caps it at
Dice 0.479 with NO threshold able to recover it.

So pixel scale belongs to the MODEL, not the loader (see models.py). The loader
emits raw [0,1] and each wrapper applies its own scale. Every model therefore
shares one disk read, one resize and one augmentation -- the geometry is
identical by construction -- while each still sees the distribution it was
trained for. The alternative (normalise here, un-normalise for YOLO) works but
carries a needless roundtrip and invites exactly the bug it is working around.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def build_augmentation(training: bool, img_size: int) -> A.Compose:
    """The augmentation pipeline. Identical for all five models.

    A.Resize is a NO-OP on the packaged 640x640 PNGs -- it is kept as a cheap
    guard so a wrong-sized file fails as a shape error here rather than as a
    silent geometry mismatch 40 minutes into training.

    A.Normalize(mean=0, std=1, max_pixel_value=255) is exactly x/255: albumentations
    computes (img - mean*max_pixel_value) / (std*max_pixel_value).
    """
    to_unit = A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)
    resize = [A.Resize(height=img_size, width=img_size)]
    if not training:
        return A.Compose(resize + [to_unit, ToTensorV2()])
    return A.Compose(resize + [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2),
        to_unit, ToTensorV2(),
    ])


class BruiseDataset(Dataset):
    """Reads the 640x640 PNG package. Returns (x[3,H,W] in [0,1], y[1,H,W] in {0,1}, stem)."""

    def __init__(self, df: pd.DataFrame, root: Path, img_size: int, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.img_size = img_size
        self.tfm = build_augmentation(training, img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]

        img = cv2.imread(str(self.root / r.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {r.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.root / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {r.mask_path}")
        # IMREAD_GRAYSCALE is documented to return (H, W) but returns (H, W, 1) in any
        # process where ultralytics has been imported -- it monkey-patches cv2.imread,
        # and import ORDER does not help. That trailing axis survives ToTensorV2 and
        # makes y [B,1,H,W,1]; dice(pred[H,W], gt[H,W,1]) then BROADCASTS to [H,W,H]
        # and returns nonsense -- a pixel-perfect prediction scores 63.9 instead of 1.0.
        # Squeezing here is a no-op on an unpatched cv2 and fixes every consumer at once.
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)

        aug = self.tfm(image=img, mask=mask)
        x = aug["image"].float()
        y = aug["mask"].unsqueeze(0).float()
        assert y.shape == (1, self.img_size, self.img_size), f"bad mask shape {y.shape} for {r.stem}"
        return x, y, str(r.stem)


def make_loader(df, root, img_size, batch_size, training, workers, seed=0):
    """DataLoader with seeded, reproducible shuffling and worker RNG."""
    ds = BruiseDataset(df, root, img_size, training=training)
    gen = torch.Generator()
    gen.manual_seed(seed)

    def _init_worker(worker_id: int) -> None:
        # Each worker gets a distinct but seed-derived RNG stream, so augmentation
        # is reproducible for a given seed AND workers never draw identical noise.
        np.random.seed(seed * 1000 + worker_id)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=training, drop_last=training,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=_init_worker, generator=gen,
    )