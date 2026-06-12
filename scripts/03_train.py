"""
Transfer-learn YOLO26-OBB on the 3 PCB classes.

Derived from References_Code/Training_new.py, switched to the OBB task and the
recommended Nano weights (yolo26n-obb.pt). Override with --weights yolo26s-obb.pt
if Nano accuracy is insufficient.
"""
import os
os.environ.setdefault("USE_POLARS", "0")   # avoid Polars dependency (pandas fallback)

import argparse
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data", "data.yaml")
PROJECT = os.path.join(ROOT, "data", "runs", "train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n-obb.pt",
                    help="pretrained OBB weights (yolo26n-obb.pt / yolo26s-obb.pt)")
    ap.add_argument("--data", default=DATA_YAML,
                    help="path to data.yaml (e.g. a Roboflow export: "
                         "data/<roboflow_folder>/data.yaml). Default: data/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="pcb_obb")
    args = ap.parse_args()

    # Resolve a relative --data against the project root, then verify it exists.
    data_yaml = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    print(f"[OK] Using dataset config: {data_yaml}")

    if torch.cuda.is_available():
        device = 0
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[warn] No GPU -> training on CPU (slow).")

    os.makedirs(PROJECT, exist_ok=True)
    model = YOLO(args.weights)

    model.train(
        data=data_yaml,
        task="obb",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=PROJECT,
        name=args.name,
        exist_ok=True,
        cache=False,
        amp=True,
        patience=20,
        save_period=10,
        # --- OBB-safe augmentation (compensate mechanical tolerances) ---
        degrees=180.0,     # boards land at any orientation
        translate=0.10,
        scale=0.05,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        perspective=0.0,   # MUST stay 0 for OBB (corrupts angle labels)
        shear=0.0,         # MUST stay 0 for OBB
    )
    print(f"[OK] Training done. Best weights: {os.path.join(PROJECT, args.name, 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
