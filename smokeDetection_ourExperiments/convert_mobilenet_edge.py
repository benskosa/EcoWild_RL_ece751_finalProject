"""
convert_mobilenet_edge.py
-------------------------
Converts a trained LBP+MobileNet checkpoint to edge-optimized formats:

  Step 1 (always): PyTorch .pt  →  ONNX  (.onnx)
  Step 2 (--trt) : ONNX         →  TensorRT engine (.trt)

TensorRT precision modes (--precision):
  fp32  Default, full precision GPU inference.
  fp16  Half-precision GPU inference (~2x faster, same accuracy).
  int8  INT8 quantized inference. Fastest and most energy efficient.
        Requires a calibration dataset (--calib_dir). Can run on GPU
        or on the Jetson DLA (--dla).

DLA (--dla):
  The Jetson AGX Xavier has 2 Deep Learning Accelerators dedicated to
  efficient CNN inference at ~1-2W vs ~10-15W for the GPU. Pass --dla
  to target DLA core 0 (use --dla_core 1 for core 1). INT8 precision
  is required for DLA. Not all layers are DLA-compatible; unsupported
  layers fall back to GPU automatically.

Why not TFLite INT8?
  TFLite targets Android/iOS. On Jetson the better CPU option is ONNX
  Runtime INT8 (--ort_int8), which uses ARM-optimized ONNX kernels and
  has a much simpler PyTorch conversion path.

Usage
-----
    # ONNX only — CPU inference via ONNX Runtime:
    python convert_mobilenet_edge.py \
        --checkpoint sweep_results/checkpoints/nf2_gap1/nf2_gap1_best_acc.pt \
        --out_dir    ../smokeDetection_baseline_ecoWild/Model

    # ONNX Runtime INT8 quantization (CPU, no calibration data needed):
    python convert_mobilenet_edge.py \
        --checkpoint sweep_results/checkpoints/nf2_gap1/nf2_gap1_best_acc.pt \
        --out_dir    ../smokeDetection_baseline_ecoWild/Model \
        --ort_int8

    # TensorRT FP16 on GPU:
    python convert_mobilenet_edge.py \
        --checkpoint sweep_results/checkpoints/nf2_gap1/nf2_gap1_best_acc.pt \
        --out_dir    ../smokeDetection_baseline_ecoWild/Model \
        --trt --precision fp16

    # TensorRT INT8 on DLA (recommended for lowest energy on Jetson Xavier):
    python convert_mobilenet_edge.py \
        --checkpoint sweep_results/checkpoints/nf2_gap1/nf2_gap1_best_acc.pt \
        --out_dir    ../smokeDetection_baseline_ecoWild/Model \
        --calib_dir  ../smokeDetection_baseline_ecoWild/Dataset/val/smoke \
        --trt --precision int8 --dla

    # Validate ONNX output matches PyTorch (requires onnxruntime):
    python convert_mobilenet_edge.py \
        --checkpoint sweep_results/checkpoints/nf2_gap1/nf2_gap1_best_acc.pt \
        --out_dir    ../smokeDetection_baseline_ecoWild/Model \
        --validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.onnx

# Allow imports from this directory
sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


# ---------------------------------------------------------------------------
# Step 1: PyTorch → ONNX
# ---------------------------------------------------------------------------

def export_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    imgsz: int = 224,
) -> None:
    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=18,       # 18 is the stable default for PyTorch >= 2.1
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        dynamo=False,           # use legacy exporter; avoids dynamic_axes warning
    )
    print(f"ONNX model saved: {onnx_path}")


# ---------------------------------------------------------------------------
# Step 2: ONNX → TensorRT  (with optional INT8 calibration and DLA)
# ---------------------------------------------------------------------------

class _Int8Calibrator:
    """
    Simple entropy calibrator for TensorRT INT8 quantization.
    Feeds random batches of calibration images from calib_dir to TRT
    so it can choose optimal INT8 scale factors per layer.
    """
    def __init__(self, calib_dir: Path, imgsz: int, n_batches: int = 50):
        from PIL import Image as PILImage

        self.imgsz = imgsz
        self.index = 0
        self.cache_file = str(calib_dir / "int8_calib.cache")

        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def preprocess(path: Path) -> np.ndarray:
            img = PILImage.open(path).convert("RGB").resize((imgsz, imgsz))
            arr = np.array(img, dtype=np.float32) / 255.0          # HWC [0,1]
            arr = (arr - _mean) / _std                              # normalize
            arr = arr.transpose(2, 0, 1)[np.newaxis]               # → NCHW
            return np.ascontiguousarray(arr, dtype=np.float32)

        img_paths = sorted(calib_dir.rglob("*.jpg")) + sorted(calib_dir.rglob("*.png"))
        img_paths = sorted(img_paths)[:n_batches]
        if not img_paths:
            raise FileNotFoundError(f"No .jpg/.png images found under {calib_dir}")

        print(f"  INT8 calibration: {len(img_paths)} images from {calib_dir}")
        self.batches = []
        for p in img_paths:
            try:
                self.batches.append(preprocess(p))
            except Exception:
                continue

    # --- TensorRT IInt8EntropyCalibrator2 interface ---
    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        import pycuda.driver as cuda
        if self.index >= len(self.batches):
            return None
        batch = self.batches[self.index]
        self.index += 1
        d_input = cuda.mem_alloc(batch.nbytes)
        cuda.memcpy_htod(d_input, batch)
        return [int(d_input)]

    def read_calibration_cache(self):
        import os
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def export_trt(
    onnx_path: Path,
    trt_path: Path,
    imgsz: int = 224,
    precision: str = "fp32",
    dla: bool = False,
    dla_core: int = 0,
    calib_dir: Path | None = None,
) -> None:
    """
    Convert an ONNX model to a TensorRT engine.

    precision : "fp32" | "fp16" | "int8"
    dla       : if True, target the Jetson DLA instead of GPU
                (requires precision="int8" or "fp16")
    dla_core  : DLA core index (0 or 1 on Xavier)
    calib_dir : directory of images for INT8 calibration (required for int8)
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        print("ERROR: tensorrt / pycuda not installed.")
        return

    if precision == "int8" and calib_dir is None:
        raise ValueError("--calib_dir is required when --precision int8")
    if dla and precision == "fp32":
        raise ValueError("DLA requires --precision fp16 or int8")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    success = parser.parse(onnx_model.SerializeToString())
    if not success:
        for i in range(parser.num_errors):
            print(f"  TRT parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model for TensorRT.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    # Precision flags
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("  WARNING: platform does not natively support FP16; falling back to FP32")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  Precision: FP16")
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("  WARNING: platform does not natively support INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _Int8Calibrator(calib_dir, imgsz)
        print("  Precision: INT8")
    else:
        print("  Precision: FP32")

    # DLA flags
    if dla:
        if builder.num_DLA_cores == 0:
            raise RuntimeError("No DLA cores found on this device.")
        print(f"  Target    : DLA core {dla_core}  "
              f"({builder.num_DLA_cores} core(s) available)")
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        # Allow layers unsupported by DLA to fall back to GPU
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    else:
        print(f"  Target    : GPU")

    # Fixed batch size = 1
    network.get_input(0).shape = [1, 3, imgsz, imgsz]

    print("  Building TensorRT engine (this may take several minutes) ...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    trt_path.write_bytes(engine)
    print(f"TensorRT engine saved: {trt_path}")


# ---------------------------------------------------------------------------
# ONNX Runtime INT8 quantization (CPU path)
# ---------------------------------------------------------------------------

def export_ort_int8(onnx_path: Path, ort_int8_path: Path, calib_dir: Path | None) -> None:
    """
    Post-training static INT8 quantization using ONNX Runtime.
    Faster CPU inference than FP32 ONNX; no GPU required.
    If calib_dir is None, dynamic quantization is used instead
    (no calibration data needed, slightly lower accuracy gain).
    """
    try:
        from onnxruntime.quantization import (
            quantize_static, quantize_dynamic,
            CalibrationDataReader, QuantType,
        )
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        return

    if calib_dir is not None:
        from PIL import Image as PILImage

        class _DataReader(CalibrationDataReader):
            def __init__(self, calib_dir: Path, imgsz: int, n: int = 100):
                _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                paths = sorted(list(calib_dir.rglob("*.jpg")) + list(calib_dir.rglob("*.png")))[:n]
                self.data = []
                for p in paths:
                    try:
                        img = PILImage.open(p).convert("RGB").resize((imgsz, imgsz))
                        arr = np.array(img, dtype=np.float32) / 255.0
                        arr = (arr - _mean) / _std
                        arr = arr.transpose(2, 0, 1)[np.newaxis]
                        self.data.append({"input": np.ascontiguousarray(arr)})
                    except Exception:
                        continue
                self.iter = iter(self.data)

            def get_next(self):
                return next(self.iter, None)

        print(f"  ORT INT8 static quantization (calib: {calib_dir}) ...")
        quantize_static(
            str(onnx_path),
            str(ort_int8_path),
            _DataReader(calib_dir, imgsz=224),
            weight_type=QuantType.QInt8,
        )
    else:
        print("  ORT INT8 dynamic quantization (no calib data) ...")
        quantize_dynamic(
            str(onnx_path),
            str(ort_int8_path),
            weight_type=QuantType.QInt8,
        )

    print(f"ORT INT8 model saved: {ort_int8_path}")


# ---------------------------------------------------------------------------
# Validation: compare PyTorch vs ONNX Runtime outputs
# ---------------------------------------------------------------------------

def validate_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    imgsz: int = 224,
    n_trials: int = 5,
    atol: float = 1e-4,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("WARNING: onnxruntime not installed — skipping validation.")
        print("  Install with: pip install onnxruntime")
        return

    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print(f"\nValidating ONNX output vs PyTorch ({n_trials} random inputs):")
    max_diff = 0.0
    for i in range(n_trials):
        x = torch.randn(1, 3, imgsz, imgsz)
        with torch.no_grad():
            pt_out = model(x).numpy()
        ort_out = session.run(["logit"], {"input": x.numpy()})[0]
        diff = np.abs(pt_out - ort_out).max()
        max_diff = max(max_diff, diff)
        status = "OK" if diff < atol else "FAIL"
        print(f"  Trial {i+1}: max |diff| = {diff:.2e}  [{status}]")

    print(f"Max absolute difference across all trials: {max_diff:.2e}")
    if max_diff < atol:
        print("Validation PASSED ✓")
    else:
        print(f"Validation FAILED ✗  (threshold={atol})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained MobileNet .pt checkpoint")
    parser.add_argument("--out_dir", default="../smokeDetection_baseline_ecoWild/Model",
                        help="Output directory (ONNX/ and TensorRT/ subdirs created inside)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Input image size (default: 224)")
    # TensorRT options
    parser.add_argument("--trt", action="store_true",
                        help="Convert ONNX → TensorRT engine (must run on target Jetson)")
    parser.add_argument("--precision", default="fp32",
                        choices=["fp32", "fp16", "int8"],
                        help="TensorRT precision (default: fp32). "
                             "int8 requires --calib_dir. "
                             "DLA requires fp16 or int8.")
    parser.add_argument("--dla", action="store_true",
                        help="Target Jetson DLA instead of GPU (lowest energy, ~1-2W). "
                             "Requires --precision fp16 or int8.")
    parser.add_argument("--dla_core", type=int, default=0,
                        help="DLA core index to use (0 or 1, default: 0)")
    parser.add_argument("--calib_dir", default=None,
                        help="Image directory for INT8 calibration "
                             "(required for --precision int8)")
    # ONNX Runtime INT8 (CPU path)
    parser.add_argument("--ort_int8", action="store_true",
                        help="Also export an ONNX Runtime INT8 quantized model for CPU inference. "
                             "Uses static quantization if --calib_dir is given, else dynamic.")
    # Misc
    parser.add_argument("--validate", action="store_true",
                        help="Validate ONNX output against PyTorch (requires onnxruntime)")
    parser.add_argument("--name", default=None,
                        help="Output filename stem (default: derived from checkpoint name)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    calib_dir = Path(args.calib_dir) if args.calib_dir else None

    out_dir  = Path(args.out_dir)
    onnx_dir = out_dir / "ONNX"
    trt_dir  = out_dir / "TensorRT"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    if args.trt:
        trt_dir.mkdir(parents=True, exist_ok=True)

    stem          = args.name or ckpt_path.stem
    onnx_path     = onnx_dir / f"{stem}.onnx"
    ort_int8_path = onnx_dir / f"{stem}_int8.onnx"
    trt_suffix    = f"_{args.precision}" + ("_dla" if args.dla else "")
    trt_path      = trt_dir  / f"{stem}{trt_suffix}.trt"

    # --- Load model ----------------------------------------------------------
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    variant = ckpt.get("variant", "v3_small")
    model   = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Variant : {variant}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")

    # --- Step 1: ONNX --------------------------------------------------------
    print(f"\n[Step 1] Exporting to ONNX ...")
    export_onnx(model, onnx_path, imgsz=args.imgsz)

    # --- Validate ------------------------------------------------------------
    if args.validate:
        validate_onnx(model, onnx_path, imgsz=args.imgsz)

    # --- Step 2a: ONNX Runtime INT8 (CPU path, optional) --------------------
    if args.ort_int8:
        print(f"\n[Step 2a] Exporting ORT INT8 quantized model (CPU) ...")
        export_ort_int8(onnx_path, ort_int8_path, calib_dir)

    # --- Step 2b: TensorRT (optional) ----------------------------------------
    if args.trt:
        print(f"\n[Step 2b] Converting ONNX → TensorRT  "
              f"(precision={args.precision}, dla={args.dla}) ...")
        print("  NOTE: TensorRT engines are device-specific.")
        print("        Always build on the Jetson, not on a dev machine.")
        export_trt(
            onnx_path, trt_path,
            imgsz=args.imgsz,
            precision=args.precision,
            dla=args.dla,
            dla_core=args.dla_core,
            calib_dir=calib_dir,
        )

    # --- Summary -------------------------------------------------------------
    print("\nDone.")
    print(f"  ONNX (FP32)      : {onnx_path}")
    if args.ort_int8:
        print(f"  ONNX (ORT INT8)  : {ort_int8_path}  ← use this for CPU inference")
    if args.trt:
        print(f"  TensorRT engine  : {trt_path}")
    print()
    if args.ort_int8:
        print("To run INT8 on CPU with ONNX Runtime:")
        print("  import onnxruntime as ort, numpy as np")
        print(f"  sess = ort.InferenceSession('{ort_int8_path}', providers=['CPUExecutionProvider'])")
        print("  logit = sess.run(['logit'], {'input': img_array})[0]")
        print("  prob  = 1 / (1 + np.exp(-logit))  # sigmoid")
    else:
        print("To run FP32 on CPU with ONNX Runtime:")
        print("  import onnxruntime as ort, numpy as np")
        print(f"  sess = ort.InferenceSession('{onnx_path}', providers=['CPUExecutionProvider'])")
        print("  logit = sess.run(['logit'], {'input': img_array})[0]")
        print("  prob  = 1 / (1 + np.exp(-logit))  # sigmoid")


if __name__ == "__main__":
    main()
