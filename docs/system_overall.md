# System Overview — PCB Sorting Vision (YOLO26-OBB)

End-to-end reference for the PCB sorting vision system: **what was built**, the
**operating principle**, **how to prepare data and annotate in Roboflow**, **how the
0–360° rotation angle is determined**, the **output contract**, how to **run on WSL**,
and the **calibration steps** that gate accuracy.

The system classifies **2 PCB types** (`TQFP`, `QFP`) moving continuously on a conveyor
at **11–20 mm/s**, and for each board emits its **type**, **lateral position X**, and a
full **0–360° rotation angle** the instant the board crosses a **fixed trigger line**.
The **Y position is constant** (it is the trigger line). Model: **Ultralytics YOLO26-OBB**
(NMS-free / end-to-end). Runtime: **Ubuntu 24.04 on WSL2**.

---

## 1. What was built

```
pcb_obb_system/
├── README.md                     # full design document
├── system_overall.md              # this overview (principle + data prep + run)
├── docs/nguyen_ly_hoat_dong.md    # operating principle (Vietnamese)
├── requirements.txt
├── setup_wsl.sh                  # one-shot env setup for Ubuntu 24.04 (WSL2)
├── config/
│   ├── classes.txt               # TQFP / QFP / marker_TQFP / marker_QFP
│   └── system_config.yaml         # camera, calibration, trigger line, orientation, comms
├── data/                          # dataset root
├── scripts/
│   ├── 00_extract_frames.py       # sample frames from a recorded belt video
│   ├── 01_make_data_yaml.py       # build data.yaml from classes.txt
│   ├── 02_split_dataset.py        # 70/20/10 train/val/test split
│   ├── 03_train.py                # YOLO26-OBB transfer learning
│   └── 04_export.py               # export best.pt → TensorRT / OpenVINO
└── src/
    ├── geometry.py                # pixel→mm homography + angle normalize
    ├── tracker.py                 # lightweight centroid tracker
    ├── trigger.py                 # one-shot line-crossing trigger
    ├── orientation.py             # marker matching + 0–360° heading
    ├── comms.py                   # TCP/JSON package sender (auto-reconnect)
    └── detect_realtime.py         # MAIN real-time pipeline
```

### Classes (4 total)
| Index | Class | Role |
|------|-------|------|
| 0 | `TQFP` | PCB board type A — emits a package |
| 1 | `QFP` | PCB board type B — emits a package |
| 2 | `marker_TQFP` | white silkscreen box on TQFP — angle anchor only |
| 3 | `marker_QFP` | white silkscreen box on QFP — angle anchor only |

> **Type decision policy:** `cross_check: false` → the **PCB type comes only from the OBB
> class**. The markers are used **only** to resolve the 0–360° angle, never to decide the
> type.

---

## 2. Operating Principle

### 2.1 Per-frame data flow
```
Camera ─► [1] capture + undistort
            │
            ▼
        [2] YOLO26-OBB inference ─► OBBs: (cx, cy, w, h, θ, class, conf)
            │
            ▼
        [3] split: boards (TQFP/QFP)  vs  markers (white boxes)
            │
            ▼
        [4] centroid tracker on boards ─► stable track_id
            │
            ▼
        [5] trigger line: board crosses (upstream→downstream) → fire once
            │
            ▼
        [6] pixel→mm (homography) for X ;  Y = constant
            │
            ▼
        [7] 0–360° heading: OBB θ (mod 180°) + board→marker direction
            │
            ▼
        [8] JSON package ─► TCP ─► host / PLC / Delta robot
```

### 2.2 Step explanations
1. **Capture & undistort** — remove lens distortion with the camera matrix `K` and
   distortion coefficients `D`, so boards at the frame edges are not warped relative to
   the center (otherwise X and angle drift).
2. **YOLO26-OBB inference** — returns an **oriented** box `(cx, cy, w, h, θ)` per object
   plus class and confidence. OBB (not axis-aligned boxes) is mandatory because boards
   land at random orientations and we need the angle.
3. **Split boards vs markers** — by class name. `pcb_classes = {TQFP, QFP}` emit packages;
   `marker_*` objects are used only for the angle and are never tracked or emitted.
4. **Centroid tracking** — the belt is **very slow** (≈0.5 mm/frame at 15 mm/s, 30 FPS),
   so a board sits near the line for many frames. A lightweight nearest-neighbour
   **centroid tracker** assigns a **stable `track_id`** to each physical board. (Heavy
   trackers like ByteTrack/BoT-SORT are unnecessary: no fast motion, occlusion, or path
   crossing on a single-lane belt.)
5. **One-shot trigger line** — a fixed horizontal line at `y_px`. For each track we track
   which side of the line its center is on (`-1` upstream / `+1` downstream). A package
   fires **only** on the `-1 → +1` transition and **only once** per `track_id`
   (`triggered` flag). This eliminates duplicate packages.
6. **Coordinates (X measured, Y fixed)** — because the line is fixed, every board crosses
   at the same physical Y, so `y_mm = y_fixed_mm` (constant). The pixel center is mapped
   to `X_mm` with a calibrated **3×3 homography** `H`.
7. **0–360° heading** — see §4 (this is the core).
8. **Package & transmit** — build the JSON record and send it over TCP (newline-terminated)
   to the host/PLC. The PLC then performs encoder-based conveyor tracking (FIFO + HSC) to
   command the Delta robot — that part is downstream of this vision package.

---

## 3. Data Preparation & Frame Extraction

### 3.1 Record and extract frames
1. Run the conveyor at the **real operating speed (11–20 mm/s)** and record a video with
   the production camera, lighting, and working distance. Capture **natural variation**:
   random orientations, minor vibration, lighting changes, glare on solder pads.
2. Sample frames at a fixed stride so consecutive images are not near-duplicates:
   ```bash
   python scripts/00_extract_frames.py --video belt.mp4 --every 15 --out data/raw
   ```
3. Target roughly **300–500 frames per PCB type**. Make sure each type is seen at **many
   orientations** (the angle model only generalizes to angles it has seen).

### 3.2 Image format rules
- Save as **`.jpg`** (JPEG quality ~95). `.png` only if you need lossless.
- **Image and label filenames must match** (`frame_00007.jpg` ↔ `frame_00007.txt`).
- Use ASCII names — **no spaces, no Vietnamese diacritics**.

---

## 4. How the Rotation Angle (0–360°) Is Determined

This is the central idea, so it is spelled out fully — and it differs **per PCB type**
because **TQFP is rectangular** and **QFP is square**, which have different rotational
symmetry.

### 4.1 Why an OBB alone is not enough — and why it depends on shape
The OBB gives a precise **edge-aligned** angle `θ`, but the box repeats itself under
rotation by the board's **symmetry period**, so `θ` is ambiguous within that period:

| PCB type | Shape | Symmetry period | OBB resolves angle only to | Ambiguity to break |
|----------|-------|-----------------|----------------------------|--------------------|
| **TQFP** | rectangle (w ≠ h) | **180°** | 180° (long edge is known) | head vs tail (2 ways) |
| **QFP**  | square (w ≈ h)    | **90°**  | 90° (which edge is unknown) | 4 ways |

Ultralytics returns the raw angle in a 180°-wide window:

$$\theta \in [-\tfrac{\pi}{4}, \tfrac{3\pi}{4}) = [-45^\circ, 135^\circ)$$

For a **rectangle** this window already pins the long edge, so only the 180° head/tail
flip is unknown. For a **square** the long/short edge is undefined (w ≈ h), so the angle
is really only known to **90°** — there are **4** indistinguishable orientations.

### 4.2 The white box as an asymmetric anchor
Each PCB type carries a **distinct white silkscreen box** fixed at a known spot. Because it
is **asymmetric**, the direction from the board center to that box identifies the true
orientation. It is detected as its own class (`marker_TQFP` / `marker_QFP`) by the same
YOLO26-OBB model and used **only** for the angle (not for the type — see §1).

### 4.3 The computation (per board, at trigger time)
1. **Match the marker to the board** — the marker whose center lies inside the board's OBB
   polygon and is closest to the board center.
2. **Direction to the marker:**

   $$\phi = \operatorname{atan2}(m_y - c_y,\; m_x - c_x)$$

3. **Generate the candidate orientations** spaced by the board's symmetry period
   `S` (from `symmetry_by_class` in the config), with `n = 360 / S` candidates:

   $$\text{candidates} = \{\theta + k\,S \;:\; k = 0,1,\dots,n-1\}$$

   - **TQFP** (`S = 180°`): two candidates `{θ, θ+180°}`.
   - **QFP** (`S = 90°`): four candidates `{θ, θ+90°, θ+180°, θ+270°}`.

4. **Pick the candidate pointing toward the marker** (smallest angular distance to φ):

   $$heading = \arg\min_{\alpha \in \text{candidates}} \big|\angle(\alpha - \phi)\big|$$

5. **Apply the zero-offset and wrap to a full circle:**

   $$angle\_deg = (heading + offset\_deg) \bmod 360 \in [0, 360)$$

The OBB supplies the **precise edge angle**; the marker only **selects which candidate** is
the true heading. `offset_deg` defines which physical pose reads as 0°.
Code: [src/orientation.py](src/orientation.py) (`resolve_heading_360`).

### 4.4 Why the per-type symmetry matters
Using one symmetry for both types is wrong:
- `S = 90°` on a **TQFP** (rectangle) would add bogus `θ±90°` candidates → the marker could
  select an orientation rotated 90° off.
- `S = 180°` on a **QFP** (square) omits two valid candidates → can be 90° off.

So the symmetry is configured **per class** (`TQFP: 180`, `QFP: 90`). The type is taken
from the OBB class; since a square and a rectangle differ sharply in aspect ratio, that
classification is highly reliable, so `cross_check` is left **off** (markers are not needed
to decide the type).

### 4.5 Graceful degradation
If the marker is missing in a frame (blurred/occluded), the board still reports an angle,
but it falls back to the OBB-only value folded into `[0, S)` for that board (no full-circle
resolution that frame) — it never reports a wrong value.

### 4.6 Annotation requirements that make the angle correct
- Draw the white box **tightly** with 4 corners and label it with the **matching** marker
  class (`marker_TQFP` on TQFP, `marker_QFP` on QFP).
- Keep the marker on a **consistent side** of each board type.
- **For the square QFP especially:** prefer a marker that lies **along an edge**, not at a
  corner. A corner marker gives `φ ≈ 45°`, which sits exactly between two of the four
  candidates and makes the selection flip-prone. If unavoidable, use `offset_deg` to rotate
  the candidate grid so the marker falls clearly inside one quadrant.

---

## 5. Roboflow Annotation (full procedure)

### 5.1 Create the project
1. Roboflow → **Create New Project** → **Project Type: Object Detection**.
2. Annotation group: `pcb`.
3. Create the **4 classes** exactly as in [config/classes.txt](config/classes.txt):
   `TQFP`, `QFP`, `marker_TQFP`, `marker_QFP`.

### 5.2 Upload
Drag the extracted `data/raw/*.jpg` frames in → **Save and Continue**.

### 5.3 Annotate (oriented boxes)
For **every** image:
1. Use the **Polygon** tool and click the **4 physical corners** of each board, hugging the
   tilted edges (consistent corner order, e.g. top-left → top-right → bottom-right →
   bottom-left). Polygons give a more accurate θ than a box + rotate handle.
2. Assign the correct board class (`TQFP` or `QFP`).
3. Draw a second tight polygon around that board's **white silkscreen box**, and label it
   with the **matching** marker class (`marker_TQFP` for a TQFP board, `marker_QFP` for a
   QFP board). **Do not cross-assign** marker classes.
4. Annotate **only boards fully inside the frame**; skip boards cut off at the edges.

### 5.4 Preprocessing & augmentation (in Roboflow)
- **Preprocessing:** `Auto-Orient: On`, `Resize → 640×640`. Nothing else.
- **Augmentation:** **leave empty.** Roboflow's geometric augmentations (shear,
  perspective, crop) **corrupt OBB angle labels**. Augmentation is done safely inside
  [scripts/03_train.py](scripts/03_train.py) instead (`degrees=180`, `perspective=0`,
  `shear=0`).

### 5.5 Split & export
- Set **Train/Test Split = 70 / 20 / 10**.
- **Export → Format: `YOLOv8 Oriented Bounding Boxes`** (this is the Ultralytics OBB
  format; YOLO26-OBB uses it directly). Each `.txt` label line is:
  ```
  class_id  x1 y1  x2 y2  x3 y3  x4 y4      # 4 corners, normalized to [0,1]
  ```
- **Verify class order** in the exported `data.yaml`: it must be
  `[TQFP, QFP, marker_TQFP, marker_QFP]` (indices 0,1,2,3) to match
  [config/classes.txt](config/classes.txt). If Roboflow reordered them, update
  `classes.txt` to match.

### 5.6 Plug into the project
- **If Roboflow did the split:** drop its `train/`, `valid/`, `test/` under `data/`,
  rename `valid/` → `validation/`, then run `01_make_data_yaml.py` (skip
  `02_split_dataset.py`).
- **If you want the project's split:** export everything flat into `data/images` +
  `data/labels`, then run `02_split_dataset.py`.
- Always (re)generate `data.yaml` with `01_make_data_yaml.py` so class IDs stay consistent
  between training and runtime.

---

## 6. Training (OBB-specific)

```bash
python scripts/01_make_data_yaml.py     # 4-class data.yaml
python scripts/03_train.py              # transfer-learn yolo26n-obb.pt
python scripts/04_export.py --weights data/runs/train/pcb_obb/weights/best.pt --format engine
```
- Start from **`yolo26n-obb.pt`** (Nano is sufficient for 2 separable types; move to
  `yolo26s-obb` only if θ error or mAP is too high).
- Augmentation is OBB-safe (`degrees=180`, **`perspective=0`, `shear=0`**).
- Watch `mAP50`, `mAP50-95`, and **especially the angle error θ** — a few degrees can make
  a gripper clip a component. Validate θ on a real test video, not just on mAP.

---

## 7. Output package (one per board, at the trigger line)

```json
{ "type": "QFP", "track_id": 7, "x_mm": 142.6, "y_mm": 0.0, "angle_deg": 287.5, "conf": 0.94, "ts": 1733740000.123 }
```
- `type` — from the OBB class (markers do not change it; `cross_check: false`).
- `x_mm` — lateral position from the homography.
- `y_mm` — constant trigger-line position.
- `angle_deg` — full **0–360°** heading (or `[-90,90)` fallback if no marker that frame).

---

## 8. Run it on WSL Ubuntu 24.04

```bash
cd pcb_obb_system
bash setup_wsl.sh && source .venv/bin/activate
# attach the USB camera from Windows PowerShell (admin):
#   usbipd attach --wsl --busid <BUSID>
python src/detect_realtime.py --config config/system_config.yaml --view
# --no-comms : run without a TCP listener
# --view     : show the annotated window (needs WSLg / X server)
```
If USB passthrough is unavailable, point `camera.source` in the config at a **video file**
or **RTSP/HTTP URL** — any OpenCV source works.

---

## 9. Things you must calibrate before live runs (they gate accuracy)

1. **Intrinsics** `K`, `D` (from `cv2.calibrateCamera`) → `undistort` in
   [config/system_config.yaml](config/system_config.yaml).
2. **Extrinsics**: the pixel→mm **homography** `H` and `coordinate.y_fixed_mm` (the
   physical Y of the trigger line).
3. **Trigger line**: `trigger_line.y_px` set to the pixel row of the line.
4. **Angle zero**: `orientation.offset_deg` so the reference pose reads 0°.
5. **Optics**: fast shutter (~1/500 s) + diffuse LED lighting to kill motion blur and
   solder-pad glare.
