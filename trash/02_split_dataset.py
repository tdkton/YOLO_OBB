"""
Split data/images + data/labels into train / validation / test (70 / 20 / 10).

Derived from References_Code/Split_new.py, fixed to:
  * use a 3-way 70/20/10 split (pipeline standard) instead of 90/10,
  * sample without replacement correctly,
  * copy each image together with its matching .txt OBB label.
"""
import os
import random
import shutil
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

TRAIN_PCT, VAL_PCT, TEST_PCT = 0.70, 0.20, 0.10
SEED = 42

IN_IMG = os.path.join(DATA, "images")
IN_LBL = os.path.join(DATA, "labels")

SPLITS = {
    "train": (os.path.join(DATA, "train", "images"), os.path.join(DATA, "train", "labels")),
    "validation": (os.path.join(DATA, "validation", "images"), os.path.join(DATA, "validation", "labels")),
    "test": (os.path.join(DATA, "test", "images"), os.path.join(DATA, "test", "labels")),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    if not os.path.isdir(IN_IMG):
        raise FileNotFoundError(f"Missing image folder: {IN_IMG}")
    for img_dir, lbl_dir in SPLITS.values():
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    images = [p for p in Path(IN_IMG).rglob("*") if p.suffix.lower() in IMG_EXTS]
    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_PCT)
    n_val = int(n * VAL_PCT)
    buckets = {
        "train": images[:n_train],
        "validation": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }
    print(f"Total images: {n}")
    for name, items in buckets.items():
        print(f"  {name:<11}: {len(items)}")

    for name, items in buckets.items():
        img_dir, lbl_dir = SPLITS[name]
        for img in items:
            shutil.copy(img, os.path.join(img_dir, img.name))
            txt = Path(IN_LBL) / f"{img.stem}.txt"
            if txt.exists():
                shutil.copy(txt, os.path.join(lbl_dir, txt.name))
            else:
                print(f"  [warn] no label for {img.name}")

    print("[OK] Split complete.")


if __name__ == "__main__":
    main()
