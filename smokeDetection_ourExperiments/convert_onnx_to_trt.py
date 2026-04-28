"""
convert_onnx_to_trt.py
----------------------
Converts an ONNX model to a TensorRT engine on the Jetson AGX Xavier.
Run this with system Python (which has TensorRT), after first exporting
the ONNX file using convert_mobilenet_edge.py in your micromamba env.

Usage
-----
    # FP32 (GPU):
    /usr/bin/python3 convert_onnx_to_trt.py \
        --onnx_path ../smokeDetection_baseline_ecoWild/Model/ONNX/nf2_gap16_best_acc.onnx \
        --trt_path  ../smokeDetection_baseline_ecoWild/Model/TensorRT/nf2_gap16_best_acc_fp32.trt

    # FP16 (GPU):
    /usr/bin/python3 convert_onnx_to_trt.py \
        --onnx_path  ../smokeDetection_baseline_ecoWild/Model/ONNX/nf2_gap16_best_acc.onnx \
        --trt_path   ../smokeDetection_baseline_ecoWild/Model/TensorRT/nf2_gap16_best_acc_fp16.trt \
        --precision  fp16

    # INT8 on DLA (lowest energy, recommended for MobileNet on Jetson Xavier):
    /usr/bin/python3 convert_onnx_to_trt.py \
        --onnx_path  ../smokeDetection_baseline_ecoWild/Model/ONNX/nf2_gap16_best_acc.onnx \
        --trt_path   ../smokeDetection_baseline_ecoWild/Model/TensorRT/nf2_gap16_best_acc_int8_dla.trt \
        --precision  int8 --dla \
        --calib_dir  ../smokeDetection_baseline_ecoWild/lbp_cache/gap_16/val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# INT8 calibrator (no torchvision — PIL + numpy only)
# ---------------------------------------------------------------------------

class _Int8Calibrator:
    def __init__(self, calib_dir: Path, imgsz: int, n_batches: int = 100):
        from PIL import Image as PILImage

        self.index = 0
        self.cache_file = str(calib_dir / "int8_calib.cache")

        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def preprocess(path: Path) -> np.ndarray:
            img = PILImage.open(path).convert("RGB").resize((imgsz, imgsz))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - _mean) / _std
            arr = arr.transpose(2, 0, 1)[np.newaxis]
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


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(
    onnx_path: Path,
    trt_path: Path,
    imgsz: int = 224,
    precision: str = "fp32",
    dla: bool = False,
    dla_core: int = 0,
    calib_dir: Path | None = None,
) -> None:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401

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

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("  Precision : FP16")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _Int8Calibrator(calib_dir, imgsz)
        print("  Precision : INT8")
    else:
        print("  Precision : FP32")

    if dla:
        if builder.num_DLA_cores == 0:
            raise RuntimeError("No DLA cores found on this device.")
        print(f"  Target    : DLA core {dla_core}  ({builder.num_DLA_cores} available)")
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    else:
        print("  Target    : GPU")

    network.get_input(0).shape = [1, 3, imgsz, imgsz]

    print("  Building engine (this may take several minutes) ...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    trt_path.parent.mkdir(parents=True, exist_ok=True)
    trt_path.write_bytes(engine)
    print(f"TensorRT engine saved: {trt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--onnx_path", required=True,
                        help="Path to input ONNX model")
    parser.add_argument("--trt_path", required=True,
                        help="Path to save the TensorRT engine (.trt)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--precision", default="fp32",
                        choices=["fp32", "fp16", "int8"],
                        help="TensorRT precision (default: fp32)")
    parser.add_argument("--dla", action="store_true",
                        help="Target Jetson DLA instead of GPU (requires fp16 or int8)")
    parser.add_argument("--dla_core", type=int, default=0,
                        help="DLA core index (0 or 1, default: 0)")
    parser.add_argument("--calib_dir", default=None,
                        help="Image directory for INT8 calibration (required for int8)")
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}")
        sys.exit(1)

    calib_dir = Path(args.calib_dir) if args.calib_dir else None

    print(f"ONNX input : {onnx_path}")
    print(f"TRT output : {args.trt_path}")

    convert(
        onnx_path  = onnx_path,
        trt_path   = Path(args.trt_path),
        imgsz      = args.imgsz,
        precision  = args.precision,
        dla        = args.dla,
        dla_core   = args.dla_core,
        calib_dir  = calib_dir,
    )


if __name__ == "__main__":
    main()
