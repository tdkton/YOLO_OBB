"""
Build data/data.yaml for YOLO-OBB training from config/classes.txt.

Derived from References_Code/Configure_training.py, adapted for the OBB task
and the train/validation/test split produced by 02_split_dataset.py.
"""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES_TXT = os.path.join(ROOT, "config", "classes.txt")
DATA_DIR = os.path.join(ROOT, "data")
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")


def main() -> None:
    if not os.path.exists(CLASSES_TXT):
        raise FileNotFoundError(f"classes.txt not found at {CLASSES_TXT}")

    with open(CLASSES_TXT, "r", encoding="utf-8") as f:
        classes = [ln.strip() for ln in f if ln.strip()]

    os.makedirs(DATA_DIR, exist_ok=True)
    data = {
        "path": DATA_DIR,
        "train": os.path.join("train", "images"),
        "val": os.path.join("validation", "images"),
        "test": os.path.join("test", "images"),
        "nc": len(classes),
        "names": classes,
    }

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Wrote {DATA_YAML} with {len(classes)} classes: {classes}")
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        print("\n--- data.yaml ---\n" + f.read())


if __name__ == "__main__":
    main()
