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

from feature_extraction import make_lbp_motion_image


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
    The dataset generates one LBP-motion image per consecutive frame pair
    within each video folder, inheriting the label of its parent class.

    Alternatively, if you have pre-computed LBP-motion images already saved
    as PNG/JPG files in  root/{smoke,no_smoke}/*.png, set
    precomputed=True and the dataset will load them directly.
    """

    def __init__(
        self,
        root: str,
        frame_gap: int = 1,
        target_size: tuple[int, int] = (240, 180),
        transform: transforms.Compose | None = None,
        precomputed: bool = False,
    ):
        """
        Parameters
        ----------
        root        : path to dataset root (must contain smoke/ and no_smoke/)
        frame_gap   : gap between frame pairs when building LBP-motion images
        target_size : (width, height) for feature_extraction.make_lbp_motion_image
        transform   : torchvision transform applied to each sample
        precomputed : if True, treats every image file directly as a
                      pre-built LBP-motion image (skips feature extraction)
        """
        self.root        = Path(root)
        self.frame_gap   = frame_gap
        self.target_size = target_size
        self.transform   = transform
        self.precomputed = precomputed

        self.samples: list[tuple] = []   # (path_or_pair, label)

        for label, class_dir in enumerate(["no_smoke", "smoke"]):
            class_path = self.root / class_dir
            if not class_path.exists():
                raise FileNotFoundError(f"Expected directory: {class_path}")

            if precomputed:
                # Each file is already an LBP-motion image
                for img_path in sorted(class_path.glob("*.[jp][pn]g")):
                    self.samples.append((img_path, label))
            else:
                # Each sub-folder is a video; files inside are raw frames
                for video_dir in sorted(class_path.iterdir()):
                    if not video_dir.is_dir():
                        continue
                    frame_paths = sorted(video_dir.glob("*.[jp][pn]g"))
                    for i in range(len(frame_paths) - frame_gap):
                        pair = (frame_paths[i], frame_paths[i + frame_gap])
                        self.samples.append((pair, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample, label = self.samples[idx]

        if self.precomputed:
            # Load pre-built LBP-motion image directly
            img = Image.open(sample).convert("RGB")
        else:
            # Build LBP-motion image on-the-fly from the frame pair
            path1, path2 = sample
            frame1 = np.array(Image.open(path1).convert("RGB"))
            frame2 = np.array(Image.open(path2).convert("RGB"))
            # make_lbp_motion_image expects BGR; convert RGB→BGR
            import cv2
            f1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            f2_bgr = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
            lbp_motion = make_lbp_motion_image(f1_bgr, f2_bgr, self.target_size)
            img = Image.fromarray(lbp_motion)   # already RGB

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
