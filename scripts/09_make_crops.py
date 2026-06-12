"""
Tạo dataset mới (Native_Crop) = ảnh gốc Native + crop cận cảnh board.

Quy trình với mỗi split (train / valid / test):
  1. Copy toàn bộ images + labels gốc sang Native_Crop/<split>/
  2. Với mỗi board trong từng ảnh: crop vùng board + padding,
     re-normalize annotation, lưu thêm vào Native_Crop/<split>/
  3. Copy data.yaml gốc → Native_Crop/data.yaml

Kết quả:
  data/
    Native/          <- giữ nguyên, không bị thay đổi
    Native_Crop/     <- dataset mới
      train/
        images/      <- gốc + crops
        labels/
      valid/
        images/      <- gốc + crops
        labels/
      test/
        images/      <- chỉ copy gốc (không crop test)
        labels/
      data.yaml

Chạy từ thư mục gốc project:
    python scripts/09_make_crops.py

    # Tùy chỉnh padding và min crop size
    python scripts/09_make_crops.py --pad 0.6 --min-size 320

    # Đổi nguồn / đầu ra
    python scripts/09_make_crops.py --src data/Native --out data/Native_Crop
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


# ── OBB label helpers ─────────────────────────────────────────────────────────

def parse_label(line: str):
    """Phân tích 1 dòng OBB label: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    cls_id = int(parts[0])
    pts = np.array(parts[1:], dtype=float).reshape(4, 2)
    return cls_id, pts


def corners_to_pixel(pts_norm: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    return pts_norm * np.array([img_w, img_h], dtype=float)


def pixel_to_norm(pts_px: np.ndarray, crop_w: int, crop_h: int) -> np.ndarray:
    return pts_px / np.array([crop_w, crop_h], dtype=float)


def center_of(pts: np.ndarray) -> np.ndarray:
    return pts.mean(axis=0)


def bbox_of(pts: np.ndarray):
    """Axis-aligned bounding box → (x0, y0, x1, y1) pixel."""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return x0, y0, x1, y1


# ── crop logic ────────────────────────────────────────────────────────────────

def generate_crops_for_image(img: np.ndarray, parsed: list, board_classes: set,
                              padding: float, min_size: int) -> list[tuple]:
    """
    Tạo danh sách (crop_img, new_label_lines, suffix) cho từng board trong ảnh.
    parsed: [(cls_id, pts_px_4x2, orig_line_str), ...]
    """
    img_h, img_w = img.shape[:2]
    results = []

    boards = [(cls_id, pts_px)
              for cls_id, pts_px, _ in parsed if cls_id in board_classes]

    for b_idx, (_, b_pts) in enumerate(boards):
        bx0, by0, bx1, by1 = bbox_of(b_pts)
        bw, bh = bx1 - bx0, by1 - by0

        pad_x = bw * padding
        pad_y = bh * padding
        cx0 = int(max(0, bx0 - pad_x))
        cy0 = int(max(0, by0 - pad_y))
        cx1 = int(min(img_w, bx1 + pad_x))
        cy1 = int(min(img_h, by1 + pad_y))

        crop_w, crop_h = cx1 - cx0, cy1 - cy0
        if crop_w < min_size or crop_h < min_size:
            continue

        crop_img = img[cy0:cy1, cx0:cx1]

        new_lines = []
        for cls_id, pts_px, _ in parsed:
            cx, cy = center_of(pts_px)
            if not (cx0 <= cx <= cx1 and cy0 <= cy <= cy1):
                continue
            pts_crop = pts_px - np.array([cx0, cy0], dtype=float)
            pts_crop[:, 0] = pts_crop[:, 0].clip(0, crop_w)
            pts_crop[:, 1] = pts_crop[:, 1].clip(0, crop_h)
            pts_norm = pixel_to_norm(pts_crop, crop_w, crop_h)
            coords = " ".join(f"{v:.6f}" for v in pts_norm.flatten())
            new_lines.append(f"{cls_id} {coords}")

        if new_lines:
            results.append((crop_img, new_lines, f"_crop{b_idx}"))

    return results


# ── per-split processing ──────────────────────────────────────────────────────

def process_split(src_split: Path, out_split: Path, board_classes: set,
                  padding: float, min_size: int, do_crop: bool) -> tuple[int, int]:
    """
    Copy ảnh gốc + (tuỳ chọn) thêm crops vào out_split.
    Trả về (n_original, n_crops).
    """
    src_images = src_split / "images"
    src_labels = src_split / "labels"
    out_images = out_split / "images"
    out_labels = out_split / "labels"

    if not src_images.exists():
        return 0, 0

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted(p for p in src_images.iterdir()
                       if p.suffix.lower() in img_exts)

    n_orig = 0
    n_crop = 0

    for img_path in img_paths:
        label_path = src_labels / (img_path.stem + ".txt")

        # ── copy ảnh gốc ────────────────────────────────────────────────────
        shutil.copy2(img_path, out_images / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, out_labels / label_path.name)
        n_orig += 1

        if not do_crop or not label_path.exists():
            continue

        # ── tạo crops ───────────────────────────────────────────────────────
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        raw_lines = label_path.read_text(encoding="utf-8").splitlines()
        parsed = []
        for line in raw_lines:
            r = parse_label(line)
            if r is not None:
                cls_id, pts_norm = r
                pts_px = corners_to_pixel(pts_norm, img_w, img_h)
                parsed.append((cls_id, pts_px, line.strip()))

        crops = generate_crops_for_image(img, parsed, board_classes,
                                          padding, min_size)
        for crop_img, new_lines, suffix in crops:
            stem = img_path.stem + suffix
            out_img_p   = out_images / f"{stem}.jpg"
            out_label_p = out_labels / f"{stem}.txt"
            cv2.imwrite(str(out_img_p), crop_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_label_p.write_text("\n".join(new_lines), encoding="utf-8")
            n_crop += 1

    return n_orig, n_crop


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Tao dataset moi Native_Crop = anh goc + crop board.")
    ap.add_argument("--src", default=str(ROOT / "data" / "Native"),
                    help="Thu muc Native goc (chua train/valid/test)")
    ap.add_argument("--out", default=str(ROOT / "data" / "Native_Crop"),
                    help="Thu muc dau ra (mac dinh: data/Native_Crop)")
    ap.add_argument("--pad", type=float, default=0.6,
                    help="Padding moi phia so voi kich thuoc board (mac dinh 0.6 = 60%%)")
    ap.add_argument("--min-size", type=int, default=320,
                    help="Bo qua crop nho hon N pixel (mac dinh 320)")
    ap.add_argument("--board-classes", default="0,1",
                    help="Class ID cua board (mac dinh '0,1' = QFP,TQFP)")
    ap.add_argument("--no-crop-test", action="store_true", default=True,
                    help="Khong crop split test (mac dinh: bat)")
    args = ap.parse_args()

    src_root  = Path(args.src)
    out_root  = Path(args.out)
    board_cls = set(int(c) for c in args.board_classes.split(","))

    print("=" * 55)
    print(f"  Nguon       : {src_root}")
    print(f"  Dau ra      : {out_root}")
    print(f"  Padding     : {args.pad * 100:.0f}% moi phia")
    print(f"  Min size    : {args.min_size}px")
    print(f"  Board class : {board_cls}")
    print("=" * 55)

    if out_root.exists():
        print(f"\n[warn] Thu muc dau ra da ton tai: {out_root}")
        ans = input("  Ghi de? (y/N): ").strip().lower()
        if ans != "y":
            print("[info] Huy.")
            return
        shutil.rmtree(out_root)

    # ── xu ly tung split ────────────────────────────────────────────────────
    splits_crop = {"train": True, "valid": True, "test": False}
    total_orig = total_crop = 0

    for split, do_crop in splits_crop.items():
        src_split = src_root / split
        if not src_split.exists():
            print(f"\n  [{split}] khong tim thay -> bo qua")
            continue

        out_split = out_root / split
        n_orig, n_crop = process_split(src_split, out_split, board_cls,
                                        args.pad, args.min_size, do_crop)
        total_orig += n_orig
        total_crop += n_crop

        tag = f"+{n_crop} crops" if do_crop else "chi copy goc"
        print(f"\n  [{split:5s}] {n_orig} anh goc   {tag}")
        print(f"           -> {out_split}")

    # ── copy data.yaml ──────────────────────────────────────────────────────
    yaml_src = src_root / "data.yaml"
    if yaml_src.exists():
        yaml_dst = out_root / "data.yaml"
        shutil.copy2(yaml_src, yaml_dst)

        # Cap nhat duong dan path trong yaml sang out_root
        text = yaml_dst.read_text(encoding="utf-8")
        old_path = str(src_root).replace("\\", "/")
        new_path = str(out_root).replace("\\", "/")
        if old_path in text:
            text = text.replace(old_path, new_path)
            yaml_dst.write_text(text, encoding="utf-8")

        print(f"\n  [yaml ] {yaml_dst}")
    else:
        print("\n  [warn] Khong tim thay data.yaml trong nguon.")

    # ── tong ket ────────────────────────────────────────────────────────────
    total_all = total_orig + total_crop
    print()
    print("=" * 55)
    print(f"  Anh goc    : {total_orig}")
    print(f"  Crops them : {total_crop}")
    print(f"  Tong cong  : {total_all} anh trong Native_Crop")
    print("=" * 55)
    print()
    print("Buoc tiep theo — train voi dataset moi:")
    print(f"  yolo obb train model=yolov8n-obb.pt \\")
    print(f"    data={out_root / 'data.yaml'} \\")
    print(f"    imgsz=1920 epochs=300 batch=2 device=0")


if __name__ == "__main__":
    main()
