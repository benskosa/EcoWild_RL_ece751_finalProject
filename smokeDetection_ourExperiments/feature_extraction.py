"""
feature_extraction.py

Implements the LBP-Motion image generation pipeline from:
  Shi et al., "Optimal Placement and Intelligent Smoke Detection Algorithm
  for Wildfire-Monitoring Cameras", IEEE Access 2020.

Pipeline (per frame pair):
  1. Convert both frames to grayscale
  2. Apply LBP to the first grayscale frame  → texture/shape descriptor
  3. Gaussian-smooth both grayscale frames
  4. Compute Farneback dense optical flow   → angle + magnitude matrices
  5. Pack [angle, magnitude, LBP] into HSV  → H, S, V channels
  6. Convert HSV → RGB                      → final LBP-motion image
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# ---------------------------------------------------------------------------
# LBP parameters (conventional operator used in the paper: P=8, R=1)
# ---------------------------------------------------------------------------
LBP_P = 8          # number of neighbours
LBP_R = 1          # radius
LBP_METHOD = "default"   # 'default' gives values in [0, 2^P - 1] = [0, 255]


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Compute the Local Binary Pattern image of a single-channel (grayscale)
    input.

    Parameters
    ----------
    gray : np.ndarray  shape (H, W), dtype uint8

    Returns
    -------
    lbp_norm : np.ndarray  shape (H, W), dtype float32, values in [0, 1]
        LBP codes normalised to [0, 1] so they can be used as a colour
        channel.
    """
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method=LBP_METHOD)
    # Normalise to [0, 1]
    lbp_norm = (lbp / (2 ** LBP_P - 1)).astype(np.float32)
    return lbp_norm


# ---------------------------------------------------------------------------
# Dense optical flow  (Farneback two-frame method – same as the paper [74])
# ---------------------------------------------------------------------------
def compute_dense_optical_flow(
    gray1: np.ndarray,
    gray2: np.ndarray,
    gaussian_ksize: int = 5,
    gaussian_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate dense optical flow between two grayscale frames using the
    Farneback polynomial-expansion method (cv2.calcOpticalFlowFarneback).

    Parameters
    ----------
    gray1, gray2 : np.ndarray  shape (H, W), dtype uint8
        Sequential grayscale frames.  gray1 is the earlier frame.
    gaussian_ksize : int
        Kernel size for Gaussian pre-smoothing (must be odd).
    gaussian_sigma : float
        Sigma for Gaussian pre-smoothing.

    Returns
    -------
    angle_norm : np.ndarray  shape (H, W), float32, values in [0, 1]
        Per-pixel flow angle mapped from [0°, 360°) → [0, 1].
    magnitude_norm : np.ndarray  shape (H, W), float32, values in [0, 1]
        Per-pixel flow magnitude, normalised to [0, 1] via min-max.
    """
    # Gaussian pre-smoothing (as described in the paper)
    g1 = cv2.GaussianBlur(gray1, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    g2 = cv2.GaussianBlur(gray2, (gaussian_ksize, gaussian_ksize), gaussian_sigma)

    # Farneback dense optical flow
    # Parameters follow OpenCV defaults which give a good speed/accuracy trade-off
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )  # shape (H, W, 2): [..., 0] = dx, [..., 1] = dy

    # Decompose into polar form
    magnitude, angle_rad = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Map angle from [0, 2π) → [0, 1]
    angle_norm = (angle_rad / (2 * np.pi)).astype(np.float32)

    # Normalise magnitude to [0, 1] via min-max  (per-frame, not global)
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        magnitude_norm = ((magnitude - mag_min) / (mag_max - mag_min)).astype(np.float32)
    else:
        magnitude_norm = np.zeros_like(magnitude, dtype=np.float32)

    return angle_norm, magnitude_norm


# ---------------------------------------------------------------------------
# LBP-motion image assembly
# ---------------------------------------------------------------------------
def make_lbp_motion_image(
    frame1: np.ndarray,
    frame2: np.ndarray,
    target_size: tuple[int, int] = (240, 180),
) -> np.ndarray:
    """
    Build a single LBP-motion image from two consecutive RGB video frames.

    The three channels of the output are assembled as:
        H  ←  optical flow angle    (motion direction)
        S  ←  optical flow magnitude (motion speed)
        V  ←  LBP of frame1         (texture / shape)

    This is then converted from HSV to RGB so it can be fed into any
    standard CNN that expects RGB input.

    Parameters
    ----------
    frame1, frame2 : np.ndarray  shape (H, W, 3), dtype uint8, BGR or RGB
        Two sequential frames from the same video.  The ordering (BGR/RGB)
        does not matter for grayscale conversion and is handled consistently.
    target_size : (width, height)
        Output spatial resolution.  Defaults to the 240×180 used in the
        paper.

    Returns
    -------
    lbp_motion_rgb : np.ndarray  shape (H, W, 3), dtype uint8
        LBP-motion image in RGB colour space, ready for CNN input after
        standard ImageNet normalisation.
    """
    # --- Resize both frames to target resolution ----------------------------
    f1 = cv2.resize(frame1, target_size)   # (W, H) for cv2
    f2 = cv2.resize(frame2, target_size)

    # --- Convert to grayscale -----------------------------------------------
    # cv2 expects BGR; if input is already BGR this is correct.
    # If the caller passes RGB frames, swap channels first.
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # --- Compute LBP on frame1 ----------------------------------------------
    lbp = compute_lbp(gray1)              # float32 in [0, 1]

    # --- Compute dense optical flow -----------------------------------------
    angle_norm, mag_norm = compute_dense_optical_flow(gray1, gray2)

    # --- Pack into HSV [0, 255] uint8 ---------------------------------------
    h = (angle_norm * 179).astype(np.uint8)   # OpenCV H range: [0, 179]
    s = (mag_norm   * 255).astype(np.uint8)
    v = (lbp        * 255).astype(np.uint8)

    hsv = np.stack([h, s, v], axis=-1)        # (H, W, 3)

    # --- Convert HSV → BGR → RGB -------------------------------------------
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


# ---------------------------------------------------------------------------
# N-frame LBP-motion image  (new: averages N-1 consecutive pair images)
# ---------------------------------------------------------------------------
def make_lbp_motion_image_nframes(
    frames: list,
    target_size: tuple = (240, 180),
) -> np.ndarray:
    """
    Build a single LBP-motion image from a sequence of N consecutive frames
    by averaging the N-1 pairwise LBP-motion images.

    Parameters
    ----------
    frames : list of np.ndarray  shape (H, W, 3), dtype uint8, BGR
        Ordered list of N frames (N >= 2).
    target_size : (width, height)
        Output spatial resolution passed to make_lbp_motion_image.

    Returns
    -------
    lbp_motion_rgb : np.ndarray  shape (H, W, 3), dtype uint8
        Averaged LBP-motion image in RGB colour space.

    Notes
    -----
    With N=2 this is identical to make_lbp_motion_image(frames[0], frames[1]).
    With N>2 each consecutive pair (f_i, f_{i+1}) produces one LBP-motion
    image; the per-pixel mean across all N-1 images is returned.  This
    captures motion accumulated over a longer temporal window while keeping
    the output shape (H, W, 3) compatible with MobileNet.
    """
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(frames)}")
    if len(frames) == 2:
        return make_lbp_motion_image(frames[0], frames[1], target_size)

    accum = np.zeros(
        (target_size[1], target_size[0], 3), dtype=np.float32
    )
    for i in range(len(frames) - 1):
        pair_img = make_lbp_motion_image(frames[i], frames[i + 1], target_size)
        accum += pair_img.astype(np.float32)

    averaged = (accum / (len(frames) - 1)).clip(0, 255).astype(np.uint8)
    return averaged


# ---------------------------------------------------------------------------
# Convenience: generate all LBP-motion images from a video file
# ---------------------------------------------------------------------------
def extract_lbp_motion_images_from_video(
    video_path: str,
    frame_gap: int = 1,
    max_frames: int | None = None,
    target_size: tuple[int, int] = (240, 180),
) -> list[np.ndarray]:
    """
    Read a video and produce a list of LBP-motion images.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_gap : int
        Number of frames between the two frames used to compute each
        LBP-motion image.  Use frame_gap=1 for time-lapse videos (where
        adjacent frames already show substantial motion, as in the paper's
        ALERTWildfire dataset).  For real-time videos at 25+ fps you may
        want a larger gap.
    max_frames : int or None
        If set, stop after producing this many LBP-motion images.
    target_size : (width, height)
        Spatial resolution of the output images.

    Returns
    -------
    lbp_motion_images : list of np.ndarray  shape (H, W, 3), dtype uint8
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) > max_frames + frame_gap:
            break
    cap.release()

    lbp_motion_images = []
    for i in range(len(frames) - frame_gap):
        img = make_lbp_motion_image(frames[i], frames[i + frame_gap], target_size)
        lbp_motion_images.append(img)
        if max_frames and len(lbp_motion_images) >= max_frames:
            break

    return lbp_motion_images
