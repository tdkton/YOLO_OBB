"""
Extract frames from a recorded conveyor video for annotation.

Run the belt at the real operating speed (11-20 mm/s), record a video, then
sample frames at a fixed stride so consecutive images are not near-duplicates.

Usage:
    python scripts/00_extract_frames.py --video belt.mp4 --every 15 --out data/raw
"""
import argparse
import os
import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video file or camera index")
    ap.add_argument("--every", type=int, default=15, help="save 1 frame every N frames")
    ap.add_argument("--out", default="data/raw", help="output image folder")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    ap.add_argument("--prefix", default="frame", help="output filename prefix")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.video!r}")

    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.every == 0:
            fn = os.path.join(args.out, f"{args.prefix}_{saved:05d}.jpg")
            cv2.imwrite(fn, frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            saved += 1
        idx += 1

    cap.release()
    print(f"[OK] Read {idx} frames, saved {saved} images to {args.out}")


if __name__ == "__main__":
    main()
