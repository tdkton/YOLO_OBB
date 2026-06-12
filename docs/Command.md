# Danh sách lệnh — PCB OBB System

Tổng hợp các chuỗi lệnh hay dùng. Copy nguyên khối, chạy tuần tự từ trên xuống.

---

## 1. Train trên WSL — môi trường .venv_cuda (GPU)

### Bước 1 — Bật môi trường (bắt buộc mỗi terminal mới)

```bash
cd /mnt/d/1_Uni_ute/Graduation_Project/Yolo_V26_OBB_PCB/pcb_obb_system
source .venv_cuda/bin/activate
unset TMPDIR
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- Prompt phải có `(.venv_cuda)` sau khi activate.
- Dòng cuối phải in `cuda True <tên card>`. Nếu `False` → dừng, không train.
- `unset TMPDIR` bắt buộc — nếu còn sẽ gây `OSError: [Errno 95]` khi dataloader spawn worker.

---

### Bước 2 — Chọn 1 trong 4 lệnh train tuỳ mục tiêu

> **Dùng `scripts/03_train.py` thay vì `yolo obb train` CLI** vì script đã cài sẵn:
> augmentation OBB-safe (`degrees=180`, `perspective=0`, `shear=0`), `cache=True`,
> `amp=True`, `patience=20`, `save_period=10` — không cần truyền tay mỗi lần.
>
> `--batch=-1` — tự chọn batch theo VRAM. Nếu báo `CUDA out of memory` → đổi `--batch 2`.

#### Train nano — dataset Native gốc, imgsz 1920
```bash
python scripts/03_train.py \
  --data data/Native/data.yaml \
  --imgsz 1920 \
  --epochs 300 \
  --batch=-1 \
  --name pcb_obb_native
```

#### Train nano — dataset Native_Crop (gốc + crop board), imgsz 1920  ← khuyến nghị
```bash
python scripts/03_train.py \
  --data data/Native_Crop/data.yaml \
  --imgsz 1920 \
  --epochs 300 \
  --batch 2 \
  --name pcb_obb_crop
```

#### Train nano — imgsz 1280 (nhanh hơn, dùng để thử nhanh)
```bash
python scripts/03_train.py \
  --data data/Native_Crop/data.yaml \
  --imgsz 1280 \
  --epochs 300 \
  --batch=-1 \
  --name pcb_obb_crop_1280
```

#### Train small — imgsz 1920 (sau khi có đủ data ≥ 300 ảnh/loại)
```bash
python scripts/03_train.py \
  --data data/Native_Crop/data.yaml \
  --weights yolo26s-obb.pt \
  --imgsz 1920 \
  --epochs 300 \
  --batch=-1 \
  --name pcb_obb_s_crop
```

---

### Bước 3 — Đo per-class sau train

```bash
yolo obb val \
  model=data/runs/train/pcb_obb_native/weights/best.pt \
  data=data/data1/data.yaml
```
Nhìn riêng dòng `marker_QFP` / `marker_TQFP` — recall phải cao hơn **48% / 59%** (baseline cũ).

---

### Bước 4 — Test nhanh trên tập test

```bash
yolo obb predict \
  model=data/runs/train/pcb_obb_native/weights/best.pt \
  source=data/data1/test/images \
  save=True \
  conf=0.5
```
Kết quả lưu ở `runs/obb/predict/`.

---

### So sánh các run

| `--name` | `--imgsz` | Pixel marker | VRAM | Ghi chú |
|---|---|---|---|---|
| `pcb_obb` | 640 | Ít nhất | Thấp | Baseline đã train (CPU) |
| `pcb_obb_1280` | 1280 | Trung bình | Trung bình | Cân bằng |
| `pcb_obb_native` | **1920** | **Nhiều nhất** | Cao | **Tốt nhất cho marker** |
| `pcb_obb_s_native` | 1920 | Nhiều nhất | Rất cao | Sau khi có đủ data |

---

## 2. Chạy camera thật trên Windows — môi trường .venv_win

### Bật venv và chạy (mỗi terminal mới)

```powershell
cd D:\1_Uni_ute\Graduation_Project\Yolo_V26_OBB_PCB\pcb_obb_system
.\.venv_win\Scripts\Activate.ps1
python src\detect_realtime.py --config config\system_config.yaml --view --no-comms
```

### Nếu Activate bị chặn (running scripts is disabled)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv_win\Scripts\Activate.ps1
```

### Cài lần đầu (chỉ 1 lần)

```powershell
cd D:\1_Uni_ute\Graduation_Project\Yolo_V26_OBB_PCB\pcb_obb_system
python -m venv .venv_win
.\.venv_win\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Trỏ tới model vừa train xong

Sửa trong `config/system_config.yaml`:
```yaml
model:
  weights: "data/runs/train/pcb_obb_native/weights/best.pt"
  imgsz: 1920
  conf: 0.60
```

---

## 3. Lệnh phụ hay dùng

### Kiểm tra GPU (WSL)
```bash
nvidia-smi
python -c "import torch; print(torch.__version__, '| cuda', torch.cuda.is_available())"
```

### Kiểm tra torch + ultralytics + cv2 (WSL)
```bash
python -c "import torch, ultralytics, cv2; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| cuda', torch.cuda.is_available())"
```

### Xem kết quả train
```bash
ls data/runs/train/
# Mở ảnh kết quả
eog data/runs/train/pcb_obb_native/results.png
eog data/runs/train/pcb_obb_native/confusion_matrix_normalized.png
```
