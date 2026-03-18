"""
gate.py

Lightweight inference wrapper intended for integration into the EcoWild
pipeline as a first-stage gate that runs BEFORE the heavier
ResNet34 + YOLOv8 ensemble.

Usage example (EcoWild integration):
--------------------------------------
    from gate import LBPMotionGate

    gate = LBPMotionGate("checkpoints/best_model.pt", threshold=0.4)

    # In the EcoWild sensing loop, after DT fires:
    label, prob = gate.check(prev_frame_bgr, curr_frame_bgr)
    if label == 1:
        # Pass to heavy ResNet34 + YOLOv8 ensemble
        run_full_smoke_detection(curr_frame_bgr)
    # else: gate blocked, save energy

Design notes:
  - threshold=0.4  (lower than 0.5) biases toward fewer false negatives,
    which matches EcoWild's priority of not missing fires.
  - Returns (0, prob) on nighttime frames (mean luminance < night_threshold)
    since optical flow is uninformative in darkness; the caller should then
    skip the gate and go straight to the full ensemble for nighttime.
  - Fail-open policy: if an exception occurs during feature extraction
    (e.g. degenerate frame), returns (1, 1.0) to ensure the full ensemble
    is always invoked rather than silently dropping a potential fire frame.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from feature_extraction import make_lbp_motion_image
from model import build_model, get_transforms


class LBPMotionGate:
    """
    Wraps the trained LBP-motion MobileNet as a binary gate.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt checkpoint saved by train.py.
    threshold : float
        Probability threshold above which smoke is declared.
        Lower values → fewer false negatives (prefer for safety-critical
        applications like EcoWild).
    night_luminance_threshold : float in [0, 255]
        If the mean pixel value of the current frame is below this value,
        the frame is considered night-time and the gate is bypassed
        (returns is_night=True).
    device : str
        Torch device string.  Defaults to CUDA if available.
    frame_gap : int
        Not used here (gate always receives two frames from caller),
        kept for documentation consistency with training config.
    """

    def __init__(
        self,
        checkpoint_path: str,
        threshold: float = 0.4,
        night_luminance_threshold: float = 40.0,
        device: str | None = None,
    ):
        self.threshold = threshold
        self.night_lum = night_luminance_threshold
        self.device    = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.transform = get_transforms(train=False)
        self.model     = self._load_model(checkpoint_path)

    def _load_model(self, path: str) -> nn.Module:
        ckpt    = torch.load(path, map_location=self.device)
        variant = ckpt.get("variant", "v3_small")
        model   = build_model(variant=variant, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval()
        return model

    def is_nighttime(self, frame_bgr: np.ndarray) -> bool:
        """Return True if mean luminance is below the night threshold."""
        gray = frame_bgr.mean()
        return float(gray) < self.night_lum

    def check(
        self,
        frame1_bgr: np.ndarray,
        frame2_bgr: np.ndarray,
    ) -> tuple[int, float, bool]:
        """
        Run the LBP-motion gate on a pair of consecutive frames.

        Parameters
        ----------
        frame1_bgr : np.ndarray  shape (H, W, 3), dtype uint8
            Earlier frame (BGR, as returned by cv2.VideoCapture).
        frame2_bgr : np.ndarray  shape (H, W, 3), dtype uint8
            Later / current frame.

        Returns
        -------
        label    : int   1 = gate passes (potential smoke), 0 = gate blocks
        prob     : float sigmoid probability from the model
        is_night : bool  True if frame was too dark for reliable flow
        """
        # --- Night-time bypass (fail open) ----------------------------------
        if self.is_nighttime(frame2_bgr):
            return 1, 1.0, True

        # --- Feature extraction + inference (fail open on exception) --------
        try:
            lbp_motion = make_lbp_motion_image(frame1_bgr, frame2_bgr)
            img        = Image.fromarray(lbp_motion)
            tensor     = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logit = self.model(tensor).squeeze()
                prob  = torch.sigmoid(logit).item()

            label = int(prob >= self.threshold)
            return label, prob, False

        except Exception as exc:
            print(f"[LBPMotionGate] Feature extraction failed: {exc}. Failing open.")
            return 1, 1.0, False


# ---------------------------------------------------------------------------
# Quick demo: run the gate on two sample frames from disk
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) < 4:
        print("Usage: python gate.py <checkpoint.pt> <frame1.jpg> <frame2.jpg>")
        sys.exit(1)

    ckpt_path, f1_path, f2_path = sys.argv[1], sys.argv[2], sys.argv[3]

    gate   = LBPMotionGate(ckpt_path, threshold=0.4)
    frame1 = cv2.imread(f1_path)
    frame2 = cv2.imread(f2_path)

    label, prob, is_night = gate.check(frame1, frame2)

    print(f"Night-time frame: {is_night}")
    print(f"Smoke probability: {prob:.4f}")
    print(f"Gate decision:     {'SMOKE DETECTED – pass to full ensemble' if label else 'No smoke – gate blocked'}")
