"""
Detect phân loại + góc xoay trên ảnh tĩnh — giống visualize realcam.

Vẽ:
  - Box OBB xanh lá  : board (QFP / TQFP)
  - Box OBB vàng mảnh: marker
  - Mũi tên magenta  : tâm PCB → tâm marker (vector góc)
  - Nhãn             : tên class + conf + góc deg   (hoặc "no marker")

Chạy từ thư mục gốc project:
    # 10 ảnh mặc định từ Test_orient_locate, lưu kết quả, không popup
    python scripts/06_detect_images.py

    # Xem từng ảnh (bấm phím bất kỳ để tiếp)
    python scripts/06_detect_images.py --show

    # Đổi số ảnh hoặc nguồn
    python scripts/06_detect_images.py --n 20
    python scripts/06_detect_images.py --source data/Native/test/images --n 5
"""
from __future__ import annotations
import argparse
import math
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from src.orientation import pick_marker, heading_from_marker_vector

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG  = os.path.join(ROOT, "config", "system_config.yaml")
SOURCE  = os.path.join(ROOT, "data", "Test_orient_locate")
OUT_DIR = os.path.join(ROOT, "data", "runs", "detect_images")
EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_obb(result):
    if result.obb is None:
        return
    for i in range(len(result.obb.cls)):
        cx  = float(result.obb.xywhr[i][0])
        cy  = float(result.obb.xywhr[i][1])
        w   = float(result.obb.xywhr[i][2])
        h   = float(result.obb.xywhr[i][3])
        th  = float(result.obb.xywhr[i][4])   # radians
        cid = int(result.obb.cls[i])
        cf  = float(result.obb.conf[i])
        yield cx, cy, w, h, th, cid, cf


def draw_roi_axes(frame, roi_polygon) -> None:
    """Vẽ ROI (viền đỏ) + trục X (xanh dương) + trục Y (xanh lá).
    Thứ tự polygon: [0]=TL [1]=TR [2]=BR [3]=BL(O)
    Trục X = cạnh dưới (O → BR), Trục Y = cạnh trái (O → TL).
    """
    if not roi_polygon or len(roi_polygon) != 4:
        return
    pts = [tuple(p) for p in roi_polygon]
    poly = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [poly], True, (0, 0, 255), 2)

    O  = pts[3]   # bottom-left = gốc O
    Xp = pts[2]   # bottom-right → X+
    Yp = pts[0]   # top-left     → Y+

    cv2.arrowedLine(frame, O, Xp, (255, 80, 0), 2, tipLength=0.04)
    mx = ((O[0] + Xp[0]) // 2, (O[1] + Xp[1]) // 2 + 20)
    cv2.putText(frame, "X", mx, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2)

    cv2.arrowedLine(frame, O, Yp, (0, 200, 0), 2, tipLength=0.04)
    my = (O[0] - 28, (O[1] + Yp[1]) // 2)
    cv2.putText(frame, "Y", my, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)

    cv2.circle(frame, O, 6, (0, 255, 255), -1)
    cv2.putText(frame, "O(0,0)", (O[0] + 8, O[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def roi_coords_cm(cx: float, cy: float, roi_polygon, pixels_per_mm: float):
    """Project (cx, cy) onto the ROI coordinate frame.
    O = roi_polygon[3] (bottom-left), X axis = O→BR, Y axis = O→TL.
    Returns (x_px, y_px, x_cm, y_cm) or None if inputs are invalid.
    """
    if not roi_polygon or len(roi_polygon) < 4 or pixels_per_mm <= 0:
        return None
    O  = np.array(roi_polygon[3], dtype=float)
    Xp = np.array(roi_polygon[2], dtype=float)
    Yp = np.array(roi_polygon[0], dtype=float)
    Xlen = np.linalg.norm(Xp - O)
    Ylen = np.linalg.norm(Yp - O)
    if Xlen == 0 or Ylen == 0:
        return None
    Xu = (Xp - O) / Xlen
    Yu = (Yp - O) / Ylen
    V  = np.array([cx, cy], dtype=float) - O
    x_px = float(np.dot(V, Xu))
    y_px = float(np.dot(V, Yu))
    ppcm = pixels_per_mm * 10.0
    return x_px, y_px, x_px / ppcm, y_px / ppcm


def draw_result(frame, pcb_dets, marker_dets, names, marker_map, offset_deg,
                roi_polygon=None, pixels_per_mm: float = 12.0):
    # ROI + trục X/Y
    draw_roi_axes(frame, roi_polygon)

    # Marker — box vàng mảnh
    for cx, cy, bw, bh, theta, cls_id, conf in marker_dets:
        box = cv2.boxPoints(((cx, cy), (bw, bh), math.degrees(theta))).astype(int)
        cv2.polylines(frame, [box], True, (0, 255, 255), 1)
        cv2.putText(frame, f"{names[cls_id]} {conf:.2f}",
                    (int(cx) - 20, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Board — box xanh lá + vector tâm→marker + tọa độ ROI
    for board in pcb_dets:
        cx, cy, bw, bh, theta, cls_id, conf = board
        box = cv2.boxPoints(((cx, cy), (bw, bh), math.degrees(theta))).astype(int)
        cv2.polylines(frame, [box], True, (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        marker, _ = pick_marker(board, marker_dets, names, marker_map,
                                 max_dist_px=pixels_per_mm * 20.0)
        angle = heading_from_marker_vector(board, marker, offset_deg)

        if marker is not None and angle is not None:
            mx, my = int(marker[0]), int(marker[1])
            cv2.arrowedLine(frame, (int(cx), int(cy)), (mx, my),
                            (255, 0, 255), 2, tipLength=0.2)
            cv2.circle(frame, (mx, my), 6, (255, 0, 255), -1)
            angle_label = f"{angle:.1f} deg"
        else:
            angle_label = "no marker"

        # Tọa độ (X, Y) tương đối so với gốc ROI O.
        rc = roi_coords_cm(cx, cy, roi_polygon, pixels_per_mm)
        if rc is not None:
            _, _, x_cm, y_cm = rc
            coord_label = f"X:{x_cm:.1f} Y:{y_cm:.1f} cm"
        else:
            coord_label = None

        cv2.putText(frame, f"{names[cls_id]} {conf:.2f}",
                    (int(cx) - 30, int(cy) - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, angle_label,
                    (int(cx) - 30, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        if coord_label:
            cv2.putText(frame, coord_label,
                        (int(cx) - 30, int(cy) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--source", default=SOURCE,
                    help="thư mục ảnh (mặc định: data/Test_orient_locate)")
    ap.add_argument("--n",     type=int, default=10,
                    help="số ảnh tối đa (mặc định 10)")
    ap.add_argument("--out",   default=OUT_DIR)
    ap.add_argument("--show",  action="store_true",
                    help="hiện cửa sổ xem ảnh (bấm phím bất kỳ để tiếp)")
    args = ap.parse_args()

    cfg          = load_config(args.config)
    model_cfg    = cfg["model"]
    ori          = cfg.get("orientation", {})
    marker_map   = ori.get("marker_map", {})
    pcb_classes  = set(ori.get("pcb_classes", []))
    offset_deg   = float(ori.get("offset_deg", 0.0))
    roi_polygon  = cfg.get("roi", {}).get("polygon", None)
    pixels_per_mm = float(cfg.get("coordinate", {}).get("pixels_per_mm", 12.0))

    model        = YOLO(model_cfg["weights"])
    names        = model.names
    conf_th      = float(model_cfg.get("conf", 0.6))
    conf_marker  = float(model_cfg.get("conf_marker", conf_th))
    iou_th       = float(model_cfg.get("iou", 0.7))
    imgsz        = int(model_cfg.get("imgsz", 640))
    device       = model_cfg.get("device", "") or None
    marker_cls   = set(marker_map.keys())

    img_files = sorted([
        os.path.join(args.source, f)
        for f in os.listdir(args.source)
        if os.path.splitext(f)[1].lower() in EXTS
    ])[:args.n]

    if not img_files:
        print(f"[err] Không tìm thấy ảnh trong: {args.source}")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"[OK] {len(img_files)} ảnh  |  model: {model_cfg['weights']}")
    print(f"     conf_board={conf_th}  conf_marker={conf_marker}  imgsz={imgsz}")
    print(f"     output → {args.out}\n")

    for i, img_path in enumerate(img_files):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [{i+1:02d}] ✗ {os.path.basename(img_path)}")
            continue

        result = model.predict(frame, conf=conf_marker, iou=iou_th,
                               imgsz=imgsz, device=device, verbose=False)[0]

        dets        = list(extract_obb(result))
        pcb_dets    = [d for d in dets if names[d[5]] in pcb_classes and d[6] >= conf_th]
        marker_dets = [d for d in dets if names[d[5]] in marker_cls]

        # In terminal
        status = []
        for board in pcb_dets:
            cx, cy, _, _, _, cls_id, conf = board
            marker, _ = pick_marker(board, marker_dets, names, marker_map,
                                     max_dist_px=pixels_per_mm * 20.0)
            angle = heading_from_marker_vector(board, marker, offset_deg)
            rc = roi_coords_cm(cx, cy, roi_polygon, pixels_per_mm)
            coord_str = f" X:{rc[2]:.1f}cm Y:{rc[3]:.1f}cm" if rc else ""
            if angle is not None:
                status.append(f"{names[cls_id]}({conf:.2f}) {angle:.1f}°{coord_str}")
            else:
                status.append(f"{names[cls_id]}({conf:.2f}) no_marker{coord_str}")
        print(f"  [{i+1:02d}] {os.path.basename(img_path)}")
        print(f"        → {', '.join(status) if status else 'không detect được board'}")

        draw_result(frame, pcb_dets, marker_dets, names, marker_map, offset_deg,
                    roi_polygon, pixels_per_mm)

        out_path = os.path.join(args.out, os.path.basename(img_path))
        cv2.imwrite(out_path, frame)

        if args.show:
            win = f"[{i+1}/{len(img_files)}] {os.path.basename(img_path)}"
            cv2.imshow(win, frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"\n[OK] Xong — ảnh kết quả lưu tại: {args.out}")


if __name__ == "__main__":
    main()
