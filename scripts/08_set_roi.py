"""
Chỉnh ROI bằng cách click 4 góc trực tiếp lên ảnh/camera.

Quy trình:
  1. Cửa sổ ảnh hiện lên, hiện ROI hiện tại (nếu có).
  2. Click lần lượt 4 góc theo thứ tự:
       Click 1 → top-left
       Click 2 → top-right
       Click 3 → bottom-right
       Click 4 → bottom-left
  3. Sau click 4, polygon tự đóng lại — xem có đúng không.
  4. Bấm ENTER để lưu vào config / R để click lại / Q để thoát.

Chạy từ thư mục gốc project:
    # Dùng ảnh tĩnh (khuyến nghị: chụp từ camera đúng vị trí gắn thật)
    python scripts/08_set_roi.py --source data/Test_orient_locate/WIN_20260611_14_07_28_Pro.jpg

    # Dùng webcam (chụp frame rồi click)
    python scripts/08_set_roi.py

    # Chỉ xem ROI hiện tại, không sửa
    python scripts/08_set_roi.py --source <ảnh> --view-only
"""
from __future__ import annotations
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config", "system_config.yaml")

LABELS  = ["1-TopLeft", "2-TopRight", "3-BotRight", "4-BotLeft(O)"]
COLORS  = [(0, 200, 255), (0, 200, 255), (0, 200, 255), (0, 255, 255)]
ROI_CLR = (0, 0, 255)
X_CLR   = (255, 80,  0)
Y_CLR   = (0,  200,  0)

# ── state ─────────────────────────────────────────────────────────────────────
clicks: list[tuple[int, int]] = []
frame_base: np.ndarray | None = None   # ảnh gốc không có annotation
_win_scale: tuple[float, float] = (1.0, 1.0)  # (scale_x, scale_y) window→image


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_roi(config_path: str, polygon: list[tuple[int, int]]) -> None:
    cfg = load_config(config_path)
    cfg["roi"]["enabled"] = True
    cfg["roi"]["polygon"] = [list(p) for p in polygon]
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def draw_axes(img: np.ndarray, polygon: list[tuple[int, int]]) -> None:
    """Vẽ trục X (cạnh dưới) và Y (cạnh trái) từ gốc O = bottom-left."""
    O  = polygon[3]   # bottom-left = gốc
    Xp = polygon[2]   # bottom-right → X+
    Yp = polygon[0]   # top-left     → Y+

    cv2.arrowedLine(img, O, Xp, X_CLR, 2, tipLength=0.04)
    mid_x = ((O[0] + Xp[0]) // 2, (O[1] + Xp[1]) // 2 + 20)
    cv2.putText(img, "X", mid_x, cv2.FONT_HERSHEY_SIMPLEX, 0.7, X_CLR, 2)

    cv2.arrowedLine(img, O, Yp, Y_CLR, 2, tipLength=0.04)
    mid_y = (O[0] - 30, (O[1] + Yp[1]) // 2)
    cv2.putText(img, "Y", mid_y, cv2.FONT_HERSHEY_SIMPLEX, 0.7, Y_CLR, 2)

    cv2.circle(img, O, 7, (0, 255, 255), -1)
    cv2.putText(img, "O(0,0)", (O[0] + 8, O[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def redraw(frame_base: np.ndarray) -> np.ndarray:
    """Vẽ lại toàn bộ annotation lên bản sao ảnh gốc."""
    img = frame_base.copy()
    h, w = img.shape[:2]

    # Hướng dẫn
    guide = "Click: TL > TR > BR > BL  |  ENTER=luu  R=reset  Q=thoat"
    cv2.putText(img, guide, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)
    cv2.putText(img, guide, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1)

    # Nhãn click tiếp theo
    if len(clicks) < 4:
        nxt = LABELS[len(clicks)]
        cv2.putText(img, f"Click tiep: {nxt}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Các điểm đã click
    for i, pt in enumerate(clicks):
        cv2.circle(img, pt, 7, COLORS[i], -1)
        cv2.putText(img, LABELS[i], (pt[0] + 8, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 1)

    # Đường nối các điểm
    for i in range(1, len(clicks)):
        cv2.line(img, clicks[i - 1], clicks[i], ROI_CLR, 1)

    # Polygon đóng khi đủ 4 điểm
    if len(clicks) == 4:
        poly = np.array(clicks, dtype=np.int32)
        cv2.polylines(img, [poly], True, ROI_CLR, 2)
        draw_axes(img, clicks)
        cv2.putText(img, "ENTER=luu  R=reset", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img


def mouse_callback(event, x, y, flags, param):
    global clicks, frame_base, _win_scale
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        # Scale tọa độ click từ hệ window → hệ ảnh thật
        ix = int(round(x * _win_scale[0]))
        iy = int(round(y * _win_scale[1]))
        clicks.append((ix, iy))
        cv2.imshow(param, redraw(frame_base))


def grab_frame(source: str, cam_w: int = 0, cam_h: int = 0) -> np.ndarray | None:
    if os.path.isfile(source):
        return cv2.imread(source)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"[err] Khong mo duoc: {source}")
        return None
    # Set resolution khớp với detect_realtime để ROI dùng đúng hệ tọa độ
    if cam_w > 0 and cam_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] Camera resolution: {actual_w}x{actual_h}")
    print("[info] SPACE = chup frame | Q = thoat")
    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            break
        preview = f.copy()
        cv2.putText(preview, "SPACE=chup frame | Q=thoat", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Chup frame (SPACE)", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frame = f.copy()
            break
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


def view_only(frame: np.ndarray, roi_polygon: list) -> None:
    """Chỉ hiện ROI hiện tại, không cho chỉnh."""
    img = frame.copy()
    h, w = img.shape[:2]
    if roi_polygon and len(roi_polygon) == 4:
        poly = np.array(roi_polygon, dtype=np.int32)
        cv2.polylines(img, [poly], True, ROI_CLR, 2)
        draw_axes(img, [tuple(p) for p in roi_polygon])
        for i, (pt, lbl) in enumerate(zip(roi_polygon, LABELS)):
            cv2.circle(img, tuple(pt), 7, COLORS[i], -1)
            cv2.putText(img, f"{lbl}{tuple(pt)}", (pt[0] + 8, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[i], 1)
    else:
        cv2.putText(img, "Chua co ROI trong config", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(img, "Q = thoat", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    win = "ROI hien tai (view only)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))
    cv2.imshow(win, img)
    while cv2.waitKey(20) & 0xFF != ord('q'):
        pass
    cv2.destroyAllWindows()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global clicks, frame_base

    ap = argparse.ArgumentParser()
    ap.add_argument("--source",    default="0")
    ap.add_argument("--config",    default=CONFIG)
    ap.add_argument("--view-only", action="store_true",
                    help="chỉ xem ROI hiện tại, không chỉnh")
    args = ap.parse_args()

    global _win_scale

    cfg         = load_config(args.config)
    roi_polygon = cfg.get("roi", {}).get("polygon", [])
    cam_cfg     = cfg.get("camera", {})
    cam_w       = int(cam_cfg.get("width",  1280))
    cam_h       = int(cam_cfg.get("height", 720))

    frame = grab_frame(args.source, cam_w, cam_h)
    if frame is None:
        print("[err] Khong lay duoc anh.")
        return
    h, w = frame.shape[:2]
    print(f"[OK] Anh: {w}x{h} px")

    if args.view_only:
        view_only(frame, roi_polygon)
        return

    # Hiện ROI cũ lên console
    if roi_polygon:
        print(f"[info] ROI hiện tại: {roi_polygon}")
    else:
        print("[info] Chưa có ROI trong config.")

    frame_base = frame.copy()

    # Tính scale để mouse callback convert tọa độ window → ảnh thật
    win_w = min(w, 1280)
    win_h = min(h, 720)
    _win_scale = (w / win_w, h / win_h)
    if _win_scale != (1.0, 1.0):
        print(f"[info] Window hien thi {win_w}x{win_h}, anh that {w}x{h} "
              f"-> scale ({_win_scale[0]:.3f}, {_win_scale[1]:.3f})")

    win = "Set ROI  (TL > TR > BR > BL)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, win_w, win_h)
    cv2.setMouseCallback(win, mouse_callback, win)
    cv2.imshow(win, redraw(frame_base))

    print("\n[info] Click 4 góc theo thứ tự: TopLeft → TopRight → BotRight → BotLeft")
    print("       ENTER = lưu | R = reset | Q = thoát\n")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            clicks.clear()
            cv2.imshow(win, redraw(frame_base))
            print("[info] Reset — click lại từ đầu.")

        elif key == ord('q'):
            print("[info] Thoát, không lưu.")
            break

        elif key == 13 and len(clicks) == 4:   # ENTER
            save_roi(args.config, clicks)
            print(f"[OK] Đã lưu ROI vào config:")
            for lbl, pt in zip(LABELS, clicks):
                print(f"     {lbl:16s}: {pt}")

            # Lưu ảnh kết quả
            result = redraw(frame_base)
            out = os.path.join(ROOT, "data", "runs", "roi_result.jpg")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cv2.imwrite(out, result)
            print(f"[OK] Ảnh kết quả: {out}")
            break

        elif key == 13 and len(clicks) < 4:
            print(f"[warn] Mới click {len(clicks)}/4 điểm — click đủ 4 rồi nhấn ENTER.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
