# PCB Sorting Vision System — YOLO26-OBB on a Continuous Conveyor

A real-time computer-vision system that **classifies 3 PCB types by size**, measures each
board's **lateral position (X)** and **rotation angle (θ)**, and emits a single data
**package** the instant a board crosses a fixed **trigger line**. The **Y position is
constant** (it is the physical location of the trigger line), so only `class`, `X`, and
`angle` change per object.

The model is **Ultralytics YOLO26-OBB** (the "YOLO v26" family — NMS-free / end-to-end
oriented bounding boxes). The whole stack runs on **Ubuntu 24.04 inside WSL2**.

---

## 1. Problem Specification

| Item | Value |
|------|-------|
| PCB classes (by physical size) | `pcb_3x3` (30×30 mm), `pcb_3x4` (30×40 mm), `pcb_4x5` (40×50 mm) |
| Conveyor speed | Continuous, **11 – 20 mm/s** |
| Belt motion | Single direction (along image **Y** axis) |
| Output per board | `class`, `X_mm` (lateral / cross-belt), `angle_deg` (OBB θ) |
| Fixed quantity | `Y_mm` = trigger-line position (constant) |
| Trigger condition | Board **center crosses the fixed line** → detect → send package |
| Model | **YOLO26-OBB** (`yolo26*-obb`) |
| Runtime | Ubuntu 24.04 (WSL2), Python 3.10+ |

### Output package (JSON, one line per board)

```json
{ "type": "pcb_4x5", "track_id": 7, "x_mm": 142.6, "y_mm": 0.0, "angle_deg": -23.4, "conf": 0.94, "ts": 1733740000.123 }
```

`y_mm` is sent for completeness but is a **fixed constant** defined in
[config/system_config.yaml](config/system_config.yaml).

---

## 2. Model Recommendation (which fixed model to use)

> **Recommendation: `yolo26n-obb` (Nano). Fall back to `yolo26s-obb` only if mAP is insufficient.**

Reasoning for this specific application:

* **Only 3 classes, and they are separable by raw size.** This is an easy
  discrimination task — a Nano backbone has more than enough capacity. The dominant
  feature (bounding-box dimensions) is trivially learnable.
* **The belt is slow (11–20 mm/s).** A 30 mm board needs ≥ 1.5 s to traverse one
  body-length. Even at 15–30 FPS you get **dozens of frames per board**, so you do
  **not** need a heavy model for temporal coverage — latency budget is huge.
* **YOLO26 is NMS-free / end-to-end**, which removes the NMS post-processing step and
  gives lower, more deterministic latency than YOLO11 — ideal for a fixed-throughput
  industrial loop.
* **Nano exports cleanly to TensorRT/OpenVINO** and runs comfortably on a modest GPU
  or even CPU, which matters under WSL2.

| Model | When to pick it |
|-------|-----------------|
| **`yolo26n-obb`** ✅ default | 3 size-separable classes, slow belt, edge/WSL hardware |
| `yolo26s-obb` | Only if Nano's angle error θ or mAP50-95 is too high after tuning |
| `yolo26m-obb`+ | Not justified here — wasted latency/VRAM for a 3-class size task |

---

## 3. Do we need tracking? — **Yes, lightweight tracking.**

**Short answer: use a small centroid tracker, not a heavy one.**

### Why tracking is required at all
The trigger is "the board center crosses the fixed line." Because the belt is **very
slow**, the center sits *near* the line for **many consecutive frames**
(at 15 mm/s and 30 FPS a board moves only **0.5 mm/frame**). Without identity tracking,
a naive "is the center on the line?" test fires **dozens of duplicate packages** for the
*same* physical board.

### Why a *lightweight* tracker (not ByteTrack/BoT-SORT)
* Motion is **slow, unidirectional, and non-occluding** — the hard cases that justify
  ByteTrack/BoT-SORT (fast motion, crossing paths, occlusion, re-ID) **do not occur** on
  a single-lane conveyor.
* A **nearest-neighbour centroid tracker** (~30 lines, in
  [src/tracker.py](src/tracker.py)) assigns a stable ID to each board. We then fire
  **exactly one package per track ID** when that ID's center transitions from the
  *upstream* side of the line to the *downstream* side.
* This avoids any dependency on whether the Ultralytics OBB tracker emits IDs for your
  version, and it is fully deterministic.

So the design is: **detect (YOLO26-OBB) → centroid track → one-shot line-crossing
trigger per ID → package**. This is the robust replacement for the pure geometric
"Trigger Line" of the original pipeline.

---

## 4. System Architecture

```
                 ┌──────────────────────── Ubuntu 24.04 (WSL2) ────────────────────────┐
  Industrial     │                                                                      │
  Camera ──USB──►│  capture → undistort → YOLO26-OBB infer → centroid track            │
  (usbipd)       │                              │                                       │
                 │                              ▼                                       │
                 │                  line-crossing trigger (1 package / board)           │
                 │                              │                                       │
                 │                              ▼                                       │
                 │     pixel→mm transform  →  package {type, x_mm, y_mm, angle}         │
                 │                              │                                       │
                 └──────────────────────────────┼───────────────────────────────────────┘
                                                 ▼
                                    TCP / JSON  ──►  Host system / PLC / Delta robot
```

### Data flow (per frame)
1. **Capture & undistort** — `cv2.undistort()` with the calibration in
   [config/system_config.yaml](config/system_config.yaml).
2. **Inference** — `YOLO26-OBB` returns per board `(x, y, w, h, θ)` in pixels + class.
3. **Track** — centroid tracker assigns a persistent `track_id`.
4. **Trigger** — when a `track_id` center crosses the fixed line (upstream→downstream),
   fire **once**.
5. **Transform** — convert the pixel center to **`X_mm`** via the calibration homography;
   normalize **θ → angle_deg ∈ [−90, 90)**; attach the constant **`Y_mm`**.
6. **Emit** — send the JSON package over TCP to the host/PLC.

---

## 5. Repository Layout

```
pcb_obb_system/
├── README.md                     # this document
├── requirements.txt
├── setup_wsl.sh                  # one-shot env setup for Ubuntu 24.04 (WSL2)
├── config/
│   ├── classes.txt               # 3 class names (one per line)
│   └── system_config.yaml        # camera, calibration, trigger line, comms, model
├── data/                         # dataset root (images/, labels/, train/, validation/, test/)
├── scripts/
│   ├── 01_make_data_yaml.py      # build data.yaml from classes.txt   (← Configure_training.py)
│   ├── 02_split_dataset.py       # 70/20/10 train/val/test split      (← Split_new.py)
│   ├── 03_train.py               # YOLO26-OBB transfer learning        (← Training_new.py)
│   └── 04_export.py              # export best.pt → TensorRT / OpenVINO
└── src/
    ├── geometry.py               # pixel→mm homography + angle normalize
    ├── tracker.py                # lightweight centroid tracker
    ├── trigger.py                # one-shot line-crossing trigger
    ├── comms.py                  # TCP/JSON package sender (auto-reconnect)
    └── detect_realtime.py        # MAIN real-time pipeline (← DetectRealtime_2_pixel_cm.py)
```

---

## 6. End-to-End Usage (Ubuntu 24.04 / WSL2)

### 6.1 One-time setup
```bash
cd pcb_obb_system
bash setup_wsl.sh          # creates .venv and installs torch + ultralytics + opencv
source .venv/bin/activate
```

### 6.2 Attach the USB camera to WSL (from Windows PowerShell, as Admin)
```powershell
usbipd list                       # find the camera's BUSID
usbipd bind   --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```
> If USB passthrough is not available, point `camera.source` in the config to a **video
> file** or an **RTSP/HTTP stream** instead — the pipeline accepts any OpenCV source.

### 6.3 Train
```bash
python scripts/01_make_data_yaml.py        # writes data/data.yaml
python scripts/02_split_dataset.py         # 70/20/10 split
python scripts/03_train.py                 # transfer-learn yolo26n-obb.pt
```

### 6.4 Export for real-time speed
```bash
python scripts/04_export.py --weights data/runs/train/pcb_obb/weights/best.pt --format engine
# CPU-only host: use  --format openvino
```

### 6.5 Run the real-time pipeline
```bash
python src/detect_realtime.py --config config/system_config.yaml
# add  --no-comms  to test without a PLC/host listener
# add  --view      to show the annotated window (requires WSLg / X server)
```

---

## 7. Calibration (must be done once, on the real rig)

The accuracy of `X_mm` and `angle_deg` depends entirely on calibration.

1. **Intrinsics** — capture ~15 checkerboard images, run OpenCV
   `calibrateCamera()` to get `camera_matrix` + `dist_coeffs`, paste them into
   [config/system_config.yaml](config/system_config.yaml).
2. **Extrinsics (pixel→mm)** — place a board of known size at known positions, solve a
   **homography** (`cv2.findHomography`) mapping image pixels → table millimetres. Paste
   the 3×3 matrix into `coordinate.homography`.
3. **Trigger line** — set `trigger_line.y_px` to the pixel row the line sits on, and
   `coordinate.y_fixed_mm` to that line's physical Y in robot coordinates.
4. **Shutter/lighting** — fast shutter (≈1/500 s) + strong LED to kill motion blur.
   (At ≤20 mm/s blur is small, but solder-pad glare is the real enemy — diffuse the light.)

---

## 8. Training Notes (OBB-specific)

* Start from **`yolo26n-obb.pt`** (transfer learning) — see [scripts/03_train.py](scripts/03_train.py).
* **Augmentation:** rotation **±180°** (boards land at random orientation), translation
  ~10%, scale ±5%, mild motion-blur. **Disable perspective & shear** — they corrupt the
  OBB angle label geometry.
* **Watch the angle error (θ).** A few degrees of θ error can make a Delta-robot gripper
  miss/clip a component. Validate θ on a held-out video, not just `mAP50`.
* Dataset split is **70 / 20 / 10** (train/val/test) per the pipeline standard.

---

## 9. Mapping to the Original 5-Stage Pipeline

| Pipeline stage | Implemented by |
|----------------|----------------|
| 1–2 Data prep & annotation | manual capture → Label Studio (YOLO-OBB export) → `data/` |
| 2 data.yaml | [scripts/01_make_data_yaml.py](scripts/01_make_data_yaml.py) |
| 2 split | [scripts/02_split_dataset.py](scripts/02_split_dataset.py) |
| 3 training | [scripts/03_train.py](scripts/03_train.py) |
| 4 export/optimize | [scripts/04_export.py](scripts/04_export.py) |
| 5 inference + transform + trigger + comms | [src/detect_realtime.py](src/detect_realtime.py) + `src/*` |

> **Difference from the original doc:** here **Y is fixed** (a line, not a moving robot
> target) and the robot/PLC conveyor-tracking math (encoder FIFO) lives downstream of
> this vision package — this repo's responsibility ends at emitting the
> `{type, x_mm, y_mm, angle_deg}` package over TCP.
