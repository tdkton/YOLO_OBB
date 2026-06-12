"""
Hiệu chuẩn tỉ lệ pixel → cm.

Cách dùng:
  1. Đặt vật tham chiếu có kích thước đã biết (thước, tờ A4, PCB đo sẵn) vào khung camera.
  2. Chạy script — cửa sổ ảnh hiện lên.
  3. Click chuột trái vào ĐIỂM ĐẦU của đoạn tham chiếu.
  4. Click chuột trái vào ĐIỂM CUỐI của đoạn tham chiếu.
  5. Nhập khoảng cách thực (cm) vào terminal.
  6. Script tính pixels_per_cm và ghi vào config/system_config.yaml.

Chạy từ thư mục gốc project:
    # Dùng webcam (mặc định)
    python scripts/07_calib_pixel_cm.py

    # Dùng ảnh tĩnh
    python scripts/07_calib_pixel_cm.py --source data/Test_orient_locate/WIN_20260611_14_07_28_Pro.jpg

    # Chỉ tính, không ghi config
    python scripts/07_calib_pixel_cm.py --no-save
"""
from __future__ import annotations
import argparse
import math
import os
import sys

import cv2
import numpy as np
import yaml

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config", "system_config.yaml")

# ── state cho mouse callback ───────────────────────────────────────────────────
points: list[tuple[int, int]] = []
frame_display = None
_win_scale: tuple[float, float] = (1.0, 1.0)  # (scale_x, scale_y) window→image


def mouse_callback(event, x, y, flags, param):
    global points, frame_display, _win_scale
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        # Scale tọa độ click từ hệ window → hệ ảnh thật
        ix = int(round(x * _win_scale[0]))
        iy = int(round(y * _win_scale[1]))
        points.append((ix, iy))
        cv2.circle(frame_display, (ix, iy), 6, (0, 255, 255), -1)
        if len(points) == 2:
            cv2.line(frame_display, points[0], points[1], (0, 255, 255), 2)
            px_dist = math.dist(points[0], points[1])
            cv2.putText(frame_display,
                        f"{px_dist:.1f} px",
                        (min(points[0][0], points[1][0]),
                         min(points[0][1], points[1][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_roi_and_axes(img: np.ndarray, roi_polygon: list) -> np.ndarray:
    """Vẽ ROI (đỏ) + trục X (xanh dương) + trục Y (xanh lá) lên ảnh.

    Quy ước:
        polygon[0] = top-left
        polygon[1] = top-right
        polygon[2] = bottom-right   (X+)
        polygon[3] = bottom-left    (gốc O, X bắt đầu, Y bắt đầu)

    Trục X : cạnh dưới  — gốc O → bottom-right  (→)
    Trục Y : cạnh trái  — gốc O → top-left       (↑)
    """
    pts = [tuple(p) for p in roi_polygon]   # [(x,y), ...]
    poly = np.array(pts, dtype=np.int32)

    # ROI — đường viền đỏ
    cv2.polylines(img, [poly], True, (0, 0, 255), 2)

    # Gốc tọa độ O = bottom-left = pts[3]
    O  = pts[3]
    Xp = pts[2]   # bottom-right → hướng X+
    Yp = pts[0]   # top-left     → hướng Y+

    # Trục X — xanh dương (BGR: 255,0,0)
    cv2.arrowedLine(img, O, Xp, (255, 80, 0), 3, tipLength=0.04)
    mid_x = ((O[0] + Xp[0]) // 2, (O[1] + Xp[1]) // 2 + 22)
    cv2.putText(img, "X (cm)", mid_x,
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2)

    # Trục Y — xanh lá (BGR: 0,200,0)
    cv2.arrowedLine(img, O, Yp, (0, 200, 0), 3, tipLength=0.04)
    mid_y = (O[0] - 70, (O[1] + Yp[1]) // 2)
    cv2.putText(img, "Y (cm)", mid_y,
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)

    # Gốc O
    cv2.circle(img, O, 8, (0, 255, 255), -1)
    cv2.putText(img, "O (0,0)", (O[0] + 10, O[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # In tọa độ pixel 4 góc
    labels = ["TL", "TR", "BR", "BL(O)"]
    offsets = [(-55, -10), (8, -10), (8, 16), (8, 16)]
    for p, lbl, off in zip(pts, labels, offsets):
        cv2.putText(img, f"{lbl}{p}", (p[0] + off[0], p[1] + off[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    return img


def save_pixels_per_cm(config_path: str, pixels_per_cm: float) -> None:
    """Ghi pixels_per_cm vào coordinate.pixels_per_mm (đổi sang mm)."""
    cfg = load_config(config_path)
    # pixels_per_mm = pixels_per_cm / 10
    cfg["coordinate"]["pixels_per_mm"] = round(pixels_per_cm / 10.0, 4)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def grab_frame(source, cam_w: int = 0, cam_h: int = 0) -> np.ndarray | None:
    """Lay 1 frame tu camera hoac doc anh tinh."""
    if isinstance(source, str) and os.path.isfile(source):
        return cv2.imread(source)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print(f"[err] Khong mo duoc nguon: {source}")
        return None
    if cam_w > 0 and cam_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] Camera resolution: {actual_w}x{actual_h}")
    print("[info] Nhan SPACE de chup frame tu camera, Q de thoat...")
    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            break
        cv2.imshow("Chup frame de calib (SPACE=chup, Q=thoat)", f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frame = f.copy()
            break
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


def main():
    global points, frame_display, _win_scale

    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  default="0",
                    help="nguon anh: so (webcam index) hoac duong dan file anh")
    ap.add_argument("--config",  default=CONFIG)
    ap.add_argument("--no-save", action="store_true",
                    help="chi tinh, khong ghi vao config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    roi_polygon = cfg.get("roi", {}).get("polygon", [])
    cam_cfg     = cfg.get("camera", {})
    cam_w       = int(cam_cfg.get("width",  1280))
    cam_h       = int(cam_cfg.get("height", 720))

    # ── 1. Lay anh ──────────────────────────────────────────────────────────
    frame = grab_frame(args.source, cam_w, cam_h)
    if frame is None:
        print("[err] Khong lay duoc anh.")
        return

    frame_display = frame.copy()
    h, w = frame.shape[:2]
    print(f"\n[OK] Anh: {w}x{h} px")

    # ── 2. Vẽ ROI + trục toạ độ ─────────────────────────────────────────────
    if roi_polygon and len(roi_polygon) == 4:
        draw_roi_and_axes(frame_display, roi_polygon)
        print(f"[OK] ROI: {roi_polygon}")
        print(f"     Gốc O (bottom-left) = {roi_polygon[3]}")
        print(f"     Trục X → bottom-right = {roi_polygon[2]}")
        print(f"     Trục Y ↑ top-left     = {roi_polygon[0]}")
    else:
        print("[warn] Không tìm thấy roi.polygon trong config — chỉ calib tỉ lệ.")

    # ── 3. Hướng dẫn ────────────────────────────────────────────────────────
    cv2.putText(frame_display,
                "Click diem 1 va diem 2 tren doan tham chieu  |  R=reset  Q=thoat",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_display,
                "Click diem 1 va diem 2 tren doan tham chieu  |  R=reset  Q=thoat",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    win_w = min(w, 1280)
    win_h = min(h, 720)
    _win_scale = (w / win_w, h / win_h)
    if _win_scale != (1.0, 1.0):
        print(f"[info] Window {win_w}x{win_h}, anh that {w}x{h} "
              f"-> scale ({_win_scale[0]:.3f}, {_win_scale[1]:.3f})")

    win = "Calib pixel->cm  (click 2 diem)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, win_w, win_h)
    cv2.setMouseCallback(win, mouse_callback)

    # ── 3. Thu thập 2 điểm ──────────────────────────────────────────────────
    print("[info] Click vào điểm đầu, rồi điểm cuối của đoạn tham chiếu.")
    print("       R = đặt lại | Q = thoát\n")

    while True:
        cv2.imshow(win, frame_display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):          # Reset
            points.clear()
            frame_display = frame.copy()
            if roi_polygon and len(roi_polygon) == 4:
                draw_roi_and_axes(frame_display, roi_polygon)
            cv2.putText(frame_display,
                        "Click diem 1 va diem 2 tren doan tham chieu  |  R=reset  Q=thoat",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            print("[info] Đặt lại — click 2 điểm mới.")

        elif key == ord('q'):
            print("[info] Thoát.")
            cv2.destroyAllWindows()
            return

        elif len(points) == 2:
            break   # Đã có 2 điểm → tiếp tục

    cv2.destroyAllWindows()

    # ── 4. Tính khoảng cách pixel ────────────────────────────────────────────
    px_dist = math.dist(points[0], points[1])
    print(f"[OK] Điểm 1 : {points[0]}")
    print(f"[OK] Điểm 2 : {points[1]}")
    print(f"[OK] Khoảng cách : {px_dist:.2f} px")

    # ── 5. Nhập khoảng cách thực ─────────────────────────────────────────────
    while True:
        try:
            real_cm = float(input("\nNhập khoảng cách thực tế (cm): ").strip())
            if real_cm <= 0:
                raise ValueError
            break
        except ValueError:
            print("[err] Vui lòng nhập số dương (vd: 5.0)")

    # ── 6. Tính tỉ lệ ────────────────────────────────────────────────────────
    pixels_per_cm = px_dist / real_cm
    pixels_per_mm = pixels_per_cm / 10.0

    print(f"\n{'─'*45}")
    print(f"  Khoảng cách đo  : {px_dist:.2f} px")
    print(f"  Khoảng cách thực: {real_cm:.2f} cm")
    print(f"  ➜  1 cm  = {pixels_per_cm:.4f} px")
    print(f"  ➜  1 mm  = {pixels_per_mm:.4f} px")
    print(f"  ➜  1 px  = {1/pixels_per_cm:.4f} cm")
    print(f"{'─'*45}\n")

    # ── 7. Ghi config ─────────────────────────────────────────────────────────
    if not args.no_save:
        save_pixels_per_cm(args.config, pixels_per_cm)
        print(f"[OK] Đã ghi pixels_per_mm = {pixels_per_mm:.4f} vào {args.config}")
        print(f"     (coordinate.pixels_per_mm — dùng khi homography chưa hiệu chuẩn)")
    else:
        print("[info] --no-save: không ghi config.")

    # ── 8. Lưu ảnh kết quả ───────────────────────────────────────────────────
    result_img = frame.copy()
    # Vẽ ROI + trục trước (nền), rồi vẽ đoạn calib lên trên
    if roi_polygon and len(roi_polygon) == 4:
        draw_roi_and_axes(result_img, roi_polygon)
    cv2.circle(result_img, points[0], 8, (0, 255, 255), -1)
    cv2.circle(result_img, points[1], 8, (0, 255, 255), -1)
    cv2.line(result_img, points[0], points[1], (0, 255, 255), 2)
    label = f"{px_dist:.1f}px = {real_cm}cm  =>  {pixels_per_cm:.2f}px/cm"
    cv2.putText(result_img, label,
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out_path = os.path.join(ROOT, "data", "runs", "calib_pixel_cm.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, result_img)
    print(f"[OK] Ảnh kết quả lưu tại: {out_path}")


if __name__ == "__main__":
    main()
