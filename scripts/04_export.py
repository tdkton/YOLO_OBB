"""
Export the trained model to a hardware-optimized format for real-time inference.

  NVIDIA GPU : --format engine   (TensorRT, ~3-5 ms/frame)
  Intel CPU  : --format openvino
"""
import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--format", default="engine", choices=["engine", "openvino", "onnx"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (GPU only)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    out = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(f"[OK] Exported -> {out}")
    print("Update model.weights in config/system_config.yaml to point at this file.")


if __name__ == "__main__":
    main()
