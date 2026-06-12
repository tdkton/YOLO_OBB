"""
Real-time PCB sorting pipeline (YOLO26-OBB).

Per frame:  capture -> undistort -> YOLO26-OBB infer -> centroid track ->
            line-crossing trigger -> pixel->mm transform -> emit JSON package.

Emits exactly ONE package per board the instant its center crosses the fixed
trigger line. y_mm is constant (the line position); only type, x_mm and
angle_deg vary per board.

Run from the project root:
    python src/detect_realtime.py --config config/system_config.yaml [--view] [--no-comms]

Derived from References_Code/DetectRealtime_2_pixel_cm.py.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

# Allow "python src/detect_realtime.py" from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from src.geometry import PixelToMM, normalize_angle_deg
from src.tracker import CentroidTracker
from src.trigger import TriggerLine
from src.comms import PackageSender
from src.orientation import pick_marker, heading_from_marker_vector, resolve_heading_360


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_undistort(cfg: dict):
    u = cfg.get("undistort", {})
    if not u.get("enabled", False):
        return None, None
    K = np.array(u["camera_matrix"], dtype=np.float64)
    D = np.array(u["dist_coeffs"], dtype=np.float64)
    return K, D


def open_source(cfg: dict) -> cv2.VideoCapture:
    cam = cfg["camera"]
    src = cam["source"]
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 720))
        cap.set(cv2.CAP_PROP_FPS, cam.get("fps", 30))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {src!r}")
    return cap


def build_roi(cfg: dict):
    """Return the ROI polygon as an int32 (N,2) array, or None if disabled."""
    roi = cfg.get("roi", {})
    if not roi.get("enabled", False) or not roi.get("polygon"):
        return None
    return np.array(roi["polygon"], dtype=np.int32)


def in_roi(roi_poly, cx: float, cy: float) -> bool:
    """True if (cx, cy) is inside (or on) the ROI polygon (always True if no ROI)."""
    if roi_poly is None:
        return True
    return cv2.pointPolygonTest(roi_poly, (float(cx), float(cy)), False) >= 0


def extract_obb(result):
    """Yield (cx, cy, w, h, theta_rad, cls_id, conf) for each OBB detection."""
    obb = getattr(result, "obb", None)
    if obb is None or obb.xywhr is None:
        return
    xywhr = obb.xywhr.cpu().numpy()       # (N, 5): cx, cy, w, h, theta
    cls = obb.cls.cpu().numpy().astype(int)
    conf = obb.conf.cpu().numpy()
    for i in range(len(xywhr)):
        cx, cy, w, h, theta = xywhr[i]
        yield float(cx), float(cy), float(w), float(h), float(theta), int(cls[i]), float(conf[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/system_config.yaml")
    ap.add_argument("--view", action="store_true", help="show annotated window")
    ap.add_argument("--no-comms", action="store_true", help="do not open a TCP connection")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # --- Components -----------------------------------------------------------
    model = YOLO(cfg["model"]["weights"])
    names = model.names
    conf_th = float(cfg["model"].get("conf", 0.6))
    conf_marker = float(cfg["model"].get("conf_marker", conf_th))
    iou_th = float(cfg["model"].get("iou", 0.7))
    imgsz = int(cfg["model"].get("imgsz", 640))
    device = cfg["model"].get("device", "") or None

    K, D = build_undistort(cfg)
    roi_poly = build_roi(cfg)

    coord = cfg["coordinate"]
    mapper = PixelToMM(coord["homography"], coord.get("pixels_per_mm", 12.0))
    y_fixed_mm = float(coord.get("y_fixed_mm", 0.0))
    pixels_per_mm = float(coord.get("pixels_per_mm", 12.0))

    tl_cfg = cfg["trigger_line"]
    line = TriggerLine(tl_cfg["y_px"], tl_cfg.get("direction", "down"))
    min_conf = float(tl_cfg.get("min_conf", conf_th))

    ori = cfg.get("orientation", {})
    ori_enabled = ori.get("enabled", False)
    marker_map = ori.get("marker_map", {})
    marker_classes = set(marker_map.keys())
    cross_check = bool(ori.get("cross_check", True))
    pcb_classes = set(ori.get("pcb_classes", list(names.values())))
    heading_offset = float(ori.get("offset_deg", 0.0))
    symmetry_default = float(ori.get("symmetry_deg", 180.0))
    symmetry_by_class = ori.get("symmetry_by_class", {})

    # Bundle the orientation context so the viewer can compute live angles.
    ori_ctx = {
        "enabled": ori_enabled,
        "marker_map": marker_map,
        "symmetry_by_class": symmetry_by_class,
        "symmetry_default": symmetry_default,
        "offset": heading_offset,
    }

    tk = cfg.get("tracker", {})
    tracker = CentroidTracker(
        max_match_dist=tk.get("max_match_dist_px", 80),
        max_missing=tk.get("max_missing_frames", 15),
    )

    cm = cfg.get("comms", {})
    sender = PackageSender(
        host=cm.get("host", "127.0.0.1"),
        port=cm.get("port", 5000),
        reconnect_seconds=cm.get("reconnect_seconds", 2.0),
        enabled=cm.get("enabled", True) and not args.no_comms,
    )

    cap = open_source(cfg)
    print("[OK] Pipeline running. Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[warn] no frame; stopping.")
                break

            if K is not None:
                frame = cv2.undistort(frame, K, D)

            # Detect at the lower marker threshold so small markers are not dropped.
            result = model.predict(frame, conf=conf_marker, iou=iou_th,
                                   imgsz=imgsz, device=device, verbose=False)[0]

            dets = list(extract_obb(result))
            # Keep only detections whose center is inside the red ROI (the belt).
            dets = [d for d in dets if in_roi(roi_poly, d[0], d[1])]
            # Boards require the higher conf_th; markers keep the lower conf_marker.
            pcb_dets = [d for d in dets if names[d[5]] in pcb_classes and d[6] >= conf_th]
            marker_dets = [d for d in dets if names[d[5]] in marker_classes] if ori_enabled else []

            centroids = [(d[0], d[1]) for d in pcb_dets]
            active = tracker.update(centroids)

            # Map each active track back to its PCB detection (nearest centroid).
            for tid, trk in active.items():
                di = min(range(len(pcb_dets)),
                         key=lambda i: (pcb_dets[i][0] - trk.cx) ** 2 + (pcb_dets[i][1] - trk.cy) ** 2,
                         default=None) if pcb_dets else None
                if di is None:
                    continue
                board = pcb_dets[di]
                cx, cy, w, h, theta, cls_id, conf = board

                if conf < min_conf:
                    continue

                if line.crossed(trk):
                    rc = roi_coords_cm(cx, cy, roi_poly, pixels_per_mm) if roi_poly is not None else None
                    if rc is not None:
                        _, _, x_cm, y_cm = rc
                        x_mm = x_cm * 10.0
                        y_mm = y_cm * 10.0
                    else:
                        x_mm, _y = mapper.transform(cx, cy)
                        y_mm = y_fixed_mm
                    if ori_enabled:
                        marker, inferred = pick_marker(board, marker_dets, names, marker_map,
                                                       max_dist_px=pixels_per_mm * 20.0)
                        type_name = inferred if (cross_check and inferred) else names[cls_id]
                        # Primary: angle = board→marker vector vs downward reference.
                        angle = heading_from_marker_vector(board, marker, heading_offset)
                        if angle is None:
                            # Fallback when marker not detected: OBB theta mod symmetry.
                            sym = float(symmetry_by_class.get(type_name, symmetry_default))
                            angle = resolve_heading_360(board, None, heading_offset, sym)
                    else:
                        type_name = names[cls_id]
                        angle = normalize_angle_deg(theta)                          # [-90,90)
                    package = {
                        "type": type_name,
                        "track_id": tid,
                        "x_mm": round(x_mm, 2),
                        "y_mm": round(y_mm, 2),
                        "angle_deg": round(angle, 2),
                        "conf": round(conf, 3),
                        "ts": round(time.time(), 3),
                    }
                    sender.send(package)

            if args.view:
                draw(frame, line, pcb_dets, marker_dets, names, roi_poly, ori_ctx, pixels_per_mm,
                     active_tracks=active)
                cv2.imshow("PCB OBB", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[OK] Interrupted.")
    finally:
        cap.release()
        sender.close()
        if args.view:
            cv2.destroyAllWindows()


def draw_roi_axes(frame, roi_poly) -> None:
    """Vẽ ROI + trục toạ độ.
    Thứ tự polygon: [0]=TL [1]=TR [2]=BR [3]=BL(O)
    Trục X = cạnh dưới (O → BR), Trục Y = cạnh trái (O → TL).
    """
    if roi_poly is None or len(roi_poly) != 4:
        return
    pts = [tuple(p) for p in roi_poly]
    poly = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [poly], True, (0, 0, 255), 2)

    O  = pts[3]   # bottom-left = gốc
    Xp = pts[2]   # bottom-right → X+
    Yp = pts[0]   # top-left     → Y+

    cv2.arrowedLine(frame, O, Xp, (255, 80, 0), 2, tipLength=0.04)
    mx = ((O[0] + Xp[0]) // 2, (O[1] + Xp[1]) // 2 + 20)
    cv2.putText(frame, "X", mx, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

    cv2.arrowedLine(frame, O, Yp, (0, 200, 0), 2, tipLength=0.04)
    my = (O[0] - 28, (O[1] + Yp[1]) // 2)
    cv2.putText(frame, "Y", my, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    cv2.circle(frame, O, 6, (0, 255, 255), -1)
    cv2.putText(frame, "O", (O[0] + 8, O[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def roi_coords_cm(cx: float, cy: float, roi_polygon, pixels_per_mm: float):
    """Project (cx, cy) onto the ROI coordinate frame.
    O = roi_polygon[3] (bottom-left), X axis = O→BR, Y axis = O→TL.
    Returns (x_px, y_px, x_cm, y_cm) or None if inputs are invalid.
    """
    if roi_polygon is None or len(roi_polygon) < 4 or pixels_per_mm <= 0:
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


def _find_track_id(cx: float, cy: float, active_tracks: dict) -> int | None:
    """Trả về track ID của track gần (cx, cy) nhất, hoặc None nếu không có."""
    if not active_tracks:
        return None
    best_id, best_d = None, float("inf")
    for tid, trk in active_tracks.items():
        d = (trk.cx - cx) ** 2 + (trk.cy - cy) ** 2
        if d < best_d:
            best_d, best_id = d, tid
    return best_id


def draw(frame, line: TriggerLine, pcb_dets, marker_dets, names, roi_poly=None, ori_ctx=None,
         pixels_per_mm: float = 12.0, active_tracks: dict | None = None) -> None:
    """Visualize ROI+axes, trigger line, markers, and each board's OBB + angle + coords."""
    h, w = frame.shape[:2]
    draw_roi_axes(frame, roi_poly)
    cv2.line(frame, (0, line.y_px), (w, line.y_px), (0, 0, 255), 2)
    cv2.putText(frame, "TRIGGER LINE", (10, line.y_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Markers (white silkscreen boxes) drawn in yellow, for reference only.
    for cx, cy, bw, bh, theta, cls_id, conf in marker_dets:
        box = cv2.boxPoints(((cx, cy), (bw, bh), np.degrees(theta))).astype(int)
        cv2.polylines(frame, [box], True, (0, 255, 255), 1)

    # Boards: OBB (green) + center + class + PCB→marker vector angle + ROI coords.
    for board in pcb_dets:
        cx, cy, bw, bh, theta, cls_id, conf = board
        box = cv2.boxPoints(((cx, cy), (bw, bh), np.degrees(theta))).astype(int)
        cv2.polylines(frame, [box], True, (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)

        # Track ID — hiện góc trên-trái của box
        tid = _find_track_id(cx, cy, active_tracks)
        if tid is not None:
            cv2.putText(frame, f"#{tid}", (int(cx) - 30, int(cy) - 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        marker = None
        angle = None
        if ori_ctx and ori_ctx["enabled"]:
            marker, _ = pick_marker(board, marker_dets, names, ori_ctx["marker_map"],
                                     max_dist_px=pixels_per_mm * 20.0)
            angle = heading_from_marker_vector(board, marker, ori_ctx["offset"])

        # Magenta arrow: tâm PCB → tâm marker, đây là vector xác định góc.
        if marker is not None and angle is not None:
            mx, my = int(marker[0]), int(marker[1])
            cv2.arrowedLine(frame, (int(cx), int(cy)), (mx, my), (255, 0, 255), 2, tipLength=0.2)
            cv2.circle(frame, (mx, my), 6, (255, 0, 255), -1)
            angle_label = f"{angle:.1f} deg"
        else:
            angle_label = "no marker"

        # Tọa độ (X, Y) tương đối so với gốc ROI O.
        rc = roi_coords_cm(cx, cy, roi_poly, pixels_per_mm)
        if rc is not None:
            _, _, x_cm, y_cm = rc
            coord_label = f"X:{x_cm:.1f} Y:{y_cm:.1f} cm"
        else:
            coord_label = None

        cv2.putText(frame, f"{names[cls_id]} {conf:.2f}", (int(cx) - 30, int(cy) - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, angle_label, (int(cx) - 30, int(cy) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        if coord_label:
            cv2.putText(frame, coord_label, (int(cx) - 30, int(cy) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 2)


if __name__ == "__main__":
    main()
