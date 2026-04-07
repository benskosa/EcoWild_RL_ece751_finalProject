"""
model.py

Defines:
  - SmokeDataset   – PyTorch Dataset that reads (frame1, frame2) pairs
                     and returns LBP-motion images on-the-fly.
  - build_model    – Returns a MobileNetV3-Small binary classifier.
                     MobileNetV3-Small is the natural successor to the
                     MobileNetV2 used in the paper; it is lighter (~2.5 M
                     params, ~0.06 GFLOPs) while maintaining accuracy.
                     Pass variant="v2" to reproduce the paper exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
)

from feature_extraction import make_lbp_motion_image, make_lbp_motion_image_nframes


# ---------------------------------------------------------------------------
# Standard ImageNet normalisation (used by all torchvision models)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Return the torchvision transform pipeline.

    During training we add random horizontal flipping and colour jitter
    for light augmentation.  At inference we only resize + normalise.

    The paper uses 240×180 frames.  MobileNet expects square input;
    we centre-crop to 180×180 then resize to 224×224 (standard ImageNet
    input size).  You can also simply resize directly to 224×224.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SmokeDataset(Dataset):
    """
    Expects a root directory with two sub-folders:

        root/
          smoke/          ← positive class  (label = 1)
              video_001/
                  frame_0001.jpg
                  frame_0002.jpg
                  ...
          no_smoke/       ← negative class  (label = 0)
              video_001/
                  frame_0001.jpg
                  ...

    Each video folder must contain frames named in chronological order.

    The dataset generates one LBP-motion image per sliding window of
    n_frames consecutive frames (with inter-frame gap of frame_gap).
    When n_frames=2 (default) this is the classic single-pair LBP-motion
    image from the paper.  When n_frames>2, the n_frames-1 pairwise
    LBP-motion images within the window are averaged into one image,
    capturing motion over a longer temporal span.

    Sliding window example with n_frames=3, frame_gap=1 on a 6-frame video:
        window 0: (f0, f1, f2)  → avg of LBP(f0,f1) and LBP(f1,f2)
        window 1: (f1, f2, f3)
        window 2: (f2, f3, f4)
        window 3: (f3, f4, f5)

    Cache mode (recommended for repeated training runs)
    ---------------------------------------------------
    Pass cache_root to point at a directory of pre-computed pairwise
    LBP-motion images built by precompute_lbp_cache.py.  The cache stores
    one PNG per *consecutive frame pair* (always gap=1), named pair_NNNN.png:

        cache_root/
          train/smoke/<fire_id>/pair_0000.png  pair_0001.png  ...
          train/no_smoke/<fire_id>/pair_0000.png  ...
          val/...

    With the cache, different n_frames values are nearly free — loading
    n_frames=3 just reads 2 cached PNGs and averages them in memory.
    No OpenCV or scikit-image computation happens at training time.

    Note: when using cache_root, frame_gap is baked into the cache itself
    (determined by how precompute_lbp_cache.py was run) and the frame_gap
    argument to SmokeDataset is ignored.  Organise multiple caches in
    subdirectories (e.g. lbp_cache/gap_1/, lbp_cache/gap_2/) and point
    cache_root at the one you want.  Varying n_frames within a given cache
    is free — no recomputation needed.

    Alternatively, if you have pre-computed LBP-motion images already saved
    as PNG/JPG files in  root/{smoke,no_smoke}/*.png, set
    precomputed=True and the dataset will load them directly.
    """

    def __init__(
        self,
        root: str,
        n_frames: int = 2,
        frame_gap: int = 1,
        target_size: tuple[int, int] = (240, 180),
        transform: transforms.Compose | None = None,
        precomputed: bool = False,
        cache_root: str | None = None,
        preload_cache: bool = False,
    ):
        """
        Parameters
        ----------
        root        : path to dataset split root (must contain smoke/ and no_smoke/)
        n_frames    : number of consecutive frames per LBP-motion sample.
                      n_frames=2 → classic single-pair (paper default).
                      n_frames>2 → average of n_frames-1 pairwise LBP images.
        frame_gap   : stride between frames within each window (on-the-fly mode
                      only; ignored when cache_root is set).
        target_size : (width, height) for on-the-fly feature extraction
        transform   : torchvision transform applied to each sample
        precomputed : if True, treats every image file directly as a
                      pre-built LBP-motion image (skips feature extraction)
        cache_root    : path to pre-computed pairwise LBP cache produced by
                        precompute_lbp_cache.py.  When set, on-the-fly
                        computation is skipped entirely; n_frames controls how
                        many consecutive cached pairs are averaged per sample.
        preload_cache : if True (and cache_root is set), load every cached pair
                        PNG into RAM at __init__ time.  __getitem__ then does
                        only dict lookups and numpy ops — no disk I/O per epoch.
                        Recommended on servers with ≥8 GB free RAM.
                        (~2.5 GB for the full train split at 240×180.)
        """
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2, got {n_frames}")

        self.root          = Path(root)
        self.n_frames      = n_frames
        self.frame_gap     = frame_gap
        self.target_size   = target_size
        self.transform     = transform
        self.precomputed   = precomputed
        self.cache_root    = Path(cache_root) if cache_root else None
        self.preload_cache = preload_cache and (cache_root is not None)
        self._mem: dict    = {}   # path → np.ndarray, populated if preload_cache

        self.samples: list[tuple] = []   # (path_group_or_single_path, label)

        # Total frames spanned by one window: gap * (n-1) + 1
        window_span = frame_gap * (n_frames - 1)
        # Number of cached pair images needed per sample window
        n_pairs = n_frames - 1

        for label, class_dir in enumerate(["no_smoke", "smoke"]):
            class_path = self.root / class_dir
            if not class_path.exists():
                raise FileNotFoundError(f"Expected directory: {class_path}")

            if precomputed:
                # Each file is already a complete LBP-motion image
                for img_path in sorted(class_path.glob("*.[jp][pn]g")):
                    self.samples.append((img_path, label))

            elif self.cache_root is not None:
                # Load from pre-computed pairwise LBP cache.
                # Each sample is a tuple of n_pairs consecutive pair PNGs.
                # Infer the split name from root path to locate cache subdir.
                split_name = self.root.name   # e.g. "train" or "val"
                for video_dir in sorted(class_path.iterdir()):
                    if not video_dir.is_dir():
                        continue
                    cache_video_dir = (
                        self.cache_root / split_name / class_dir / video_dir.name
                    )
                    if not cache_video_dir.is_dir():
                        continue
                    pair_paths = sorted(cache_video_dir.glob("pair_*.png"))
                    if len(pair_paths) < n_pairs:
                        continue
                    for i in range(len(pair_paths) - n_pairs + 1):
                        group = tuple(pair_paths[i + j] for j in range(n_pairs))
                        self.samples.append((group, label))

            else:
                # On-the-fly computation from raw frames
                for video_dir in sorted(class_path.iterdir()):
                    if not video_dir.is_dir():
                        continue
                    frame_paths = [
                        p for p in sorted(video_dir.glob("*.[jp][pn]g"))
                        if p.stat().st_size > 0   # skip empty/corrupt files
                    ]
                    if len(frame_paths) < n_frames:
                        continue   # skip videos too short for the window
                    for i in range(len(frame_paths) - window_span):
                        group = tuple(
                            frame_paths[i + j * frame_gap]
                            for j in range(n_frames)
                        )
                        self.samples.append((group, label))

        # Preload all unique cached pair PNGs into RAM
        if self.preload_cache and self.samples:
            from tqdm import tqdm
            all_paths = sorted({p for group, _ in self.samples for p in group})
            print(f"Preloading {len(all_paths)} cached pair images into RAM...")
            for p in tqdm(all_paths, unit="img", leave=False):
                self._mem[p] = np.array(Image.open(p).convert("RGB"))
            mem_mb = sum(a.nbytes for a in self._mem.values()) / 1024 ** 2
            print(f"  Cache preloaded: {mem_mb:.0f} MB in RAM")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample, label = self.samples[idx]

        if self.precomputed:
            # Load pre-built LBP-motion image directly
            img = Image.open(sample).convert("RGB")

        elif self.cache_root is not None:
            # Load n_pairs cached pair images and average them.
            # sample is a tuple of Path objects pointing to pair_NNNN.png files.
            # Use in-memory store if preload_cache=True, otherwise read from disk.
            if self.preload_cache:
                load = lambda p: self._mem[p]
            else:
                load = lambda p: np.array(Image.open(p).convert("RGB"))

            if len(sample) == 1:
                img = Image.fromarray(load(sample[0]))
            else:
                arrays = [load(p).astype(np.float32) for p in sample]
                averaged = np.mean(arrays, axis=0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(averaged)

        else:
            # On-the-fly: load raw frames and compute LBP-motion
            import cv2
            frames_bgr = []
            for p in sample:
                frame_rgb = np.array(Image.open(p).convert("RGB"))
                frames_bgr.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            lbp_motion = make_lbp_motion_image_nframes(frames_bgr, self.target_size)
            img = Image.fromarray(lbp_motion)   # RGB uint8

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(
    variant: str = "v3_small",
    pretrained: bool = True,
) -> nn.Module:
    """
    Return a MobileNet binary smoke classifier.

    Parameters
    ----------
    variant : "v3_small" | "v2"
        "v3_small"  – MobileNetV3-Small  (~2.5 M params, ~0.06 GFLOPs)
                      Lighter and more accurate than V2 on ImageNet.
                      Recommended default.
        "v2"        – MobileNetV2  (~3.4 M params, ~0.3 GFLOPs)
                      Reproduces the exact architecture from the paper.
    pretrained : bool
        If True, load ImageNet-pretrained weights (strongly recommended;
        the paper also uses transfer learning implicitly via MobileNetV2's
        pre-trained backbone).

    Returns
    -------
    model : nn.Module
        Binary classifier.  Output is a single logit (no sigmoid).
        Use BCEWithLogitsLoss during training.
    """
    if variant == "v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        # Replace the classifier head with a single binary output neuron
        # (paper uses sigmoid activation + BCE loss)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1),
        )

    elif variant == "v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # Replace the classifier head
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)

    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose 'v2' or 'v3_small'.")

    return model
