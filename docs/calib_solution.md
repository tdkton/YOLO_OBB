# Hướng dẫn cải thiện độ chính xác — PCB OBB System

Tài liệu tổng hợp đánh giá model hiện tại, các phương án cải thiện, lệnh train GPU (WSL),
lệnh chạy camera (Windows), và hướng dẫn xuất data từ Roboflow.

---

## 1. Đánh giá model hiện tại (train 100 epoch, imgsz 640, nano, CPU)

### Kết quả từ confusion_matrix

| Class | Recall | Nhận xét |
|---|---|---|
| **QFP** | **96%** (24/25) | Tốt — phân biệt vuông chắc |
| **TQFP** | **100%** (58/58) | Hoàn hảo |
| **marker_QFP** | **48%** (12/25) | Yếu — bỏ sót gần nửa |
| **marker_TQFP** | **59%** (34/58) | Trung bình — còn nhiều false positive |

### Nhận xét tổng

- **Board (QFP/TQFP): ổn**, không overfit, phân loại vuông/chữ nhật rất chắc.
- **Marker: nút thắt chính.** Recall 48–59% → marker không được detect đủ → góc kẹt 0–180° (fallback về `theta mod S` thay vì dùng vector marker).
- **Đường cong (results.png): val/loss vẫn giảm, mAP vẫn tăng ở epoch 100** → model chưa overfit, train thêm sẽ cải thiện.

---

## 2. Nguyên nhân marker yếu

1. **Ít data** — 100 ảnh tổng quá ít cho vật nhỏ như marker (cụm chữ trắng).
2. **imgsz 640 quá nhỏ** — marker chiếm rất ít pixel → model khó học đặc trưng.
3. **Epoch 100 chưa đủ** — đường cong còn dốc.

---

## 3. Các phương án cải thiện (theo thứ tự ưu tiên)

| Thứ tự | Phương án | Hiệu quả | Chi phí |
|---|---|---|---|
| 1 | **Tăng imgsz 640→1280** | Cao nhất cho marker nhỏ | Chỉ cần chạy lại lệnh |
| 2 | **Tăng epoch 100→300** | Trung bình, miễn phí | Chỉ cần chạy lại lệnh |
| 3 | **Thêm data marker** (300–500/loại) | Cao nhất dài hạn, fix gốc rễ | Phải gán nhãn thêm |
| 4 | **Đổi nano → small** | Có capacity hơn | Chỉ sau khi đã thêm data |
| 5 | **Export TensorRT/OpenVINO** | Tăng tốc inference, không tăng accuracy | Sau khi accuracy đạt |

### Lưu ý quan trọng
- **imgsz 1280 chỉ có lợi nếu ảnh gốc đủ nét** ở vùng marker. Nếu camera chụp 640 thì nâng imgsz chỉ phóng to mờ.
- **Đừng dùng augmentation của Roboflow** — YOLO đã augment on-the-fly, Roboflow augment chồng lên sẽ làm sai nhãn góc OBB (shear/perspective làm lệch angle label).
- **Nano → Small/Medium** chỉ nên làm sau khi có đủ data (≥300/loại). Với 100 ảnh, model to dễ overfit hơn.

---

## 4. Augmentation đã có trong code

File [scripts/03_train.py](../scripts/03_train.py) đã cấu hình augmentation on-the-fly mỗi epoch:

| Tham số | Giá trị | Tác dụng |
|---|---|---|
| `degrees` | **180** | Xoay ngẫu nhiên — board nằm mọi hướng |
| `translate` | 0.10 | Dịch vị trí |
| `scale` | 0.05 | Phóng to/thu nhỏ |
| `fliplr / flipud` | 0.5 / 0.5 | Lật ngang/dọc |
| `mosaic` | 1.0 | Ghép 4 ảnh |
| `perspective` | **0.0** | Phải = 0, làm hỏng nhãn OBB |
| `shear` | **0.0** | Phải = 0, làm hỏng nhãn OBB |

→ Không cần thêm augmentation ở Roboflow. Chỉ cần **ảnh thật + nhãn đúng**.

---

## 5. Hướng dẫn xuất data từ Roboflow

- **Format:** `YOLOv8 Oriented Bounding Boxes (YOLOv8-OBB)` — đúng định dạng Ultralytics OBB.
- **Preprocessing — chỉ bật DUY NHẤT 1 thứ:**
  - ✅ Bật **Auto-Orient** — sửa xoay EXIF (ảnh chụp điện thoại hay camera hay bị xoay).
  - ❌ **Resize: TẮT (để native)** — Ultralytics tự letterbox theo `--imgsz` lúc train. Resize trong Roboflow chỉ thêm bước thừa, nguy cơ mất pixel marker.
    - Ảnh gốc **1920×1080 (1080p 16:9)**: giữ nguyên native. Ultralytics sẽ tự scale về 1280×720 + pad đen 280px mỗi bên khi train imgsz 1280.
    - Nếu bắt buộc phải resize (giới hạn dung lượng Roboflow): chọn **"Stretch to" 1280×720** — an toàn vì 1920→1280 và 1080→720 cùng hệ số 0.667, không méo tỉ lệ, không lệch góc OBB. **Không dùng Stretch to 1280×1280** (méo, sai góc).
- **Augmentation:** ❌ **Tắt hết** — YOLO augment on-the-fly mỗi epoch, không cần Roboflow làm thêm.
- ⚠️ Sau khi xuất, kiểm tra thứ tự `names` trong `data.yaml` phải khớp `classes.txt`:
  ```
  QFP(0), TQFP(1), marker_QFP(2), marker_TQFP(3)
  ```

---

## 6. Lệnh train GPU trên WSL (.venv_cuda)

### Bước 1 — Bật môi trường (mỗi terminal mới, làm đủ 4 dòng)
```bash
# Vào thư mục project
cd /mnt/d/1_Uni_ute/Graduation_Project/Yolo_V26_OBB_PCB/pcb_obb_system

# Bật venv CUDA
source .venv_cuda/bin/activate

# BẮT BUỘC: bỏ TMPDIR — nếu còn sẽ gây lỗi OSError Errno 95 khi dataloader spawn worker
unset TMPDIR

# Xác nhận GPU (phải in "cuda True" + tên card)
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Prompt phải là `(.venv_cuda)`. Nếu in `cuda False` → GPU chưa nhận, không train tiếp.

### Bước 2a — Train nano imgsz 1280 (chạy đầu tiên)
```bash
python scripts/03_train.py \
  --data data/data1/data.yaml \
  --imgsz 1280 \
  --epochs 300 \
  --batch=-1 \
  --name pcb_obb_1280
```
- `--imgsz 1280` — Ultralytics **tự load ảnh gốc (native 1920×1080)**, scale về 1280×720, pad đen thành 1280×1280. Không cần khai báo native size ở đây — đó là tính chất của file ảnh trong folder data (không resize khi xuất Roboflow).
- `--batch=-1` — Ultralytics tự chọn batch theo VRAM (an toàn).
- `--name pcb_obb_1280` — không ghi đè run cũ `pcb_obb`, dễ so sánh.
- Nếu báo `CUDA out of memory` → đổi `--batch 8` hoặc `--batch 4`.

### Bước 2b — Train small imgsz 1280 (sau khi đã có đủ data ≥ 300/loại)
```bash
python scripts/03_train.py \
  --data data/data1/data.yaml \
  --weights yolo26s-obb.pt \
  --imgsz 1280 \
  --epochs 300 \
  --batch=-1 \
  --name pcb_obb_s1280
```

### Bước 3 — Đo per-class sau train (quan trọng: nhìn riêng marker)
```bash
yolo obb val \
  model=data/runs/train/pcb_obb_1280/weights/best.pt \
  data=data/data1/data.yaml
```
Nhìn riêng dòng `marker_QFP` / `marker_TQFP` — recall phải cao hơn **48% / 59%** (kết quả cũ).

### Bước 4 — Test nhanh trên tập test
```bash
yolo obb predict \
  model=data/runs/train/pcb_obb_1280/weights/best.pt \
  source=data/data1/test/images \
  save=True \
  conf=0.5
```
Kết quả lưu ở `runs/obb/predict/` — mở xem box + góc có đúng không.

---

## 7. Lệnh chạy camera thật trên Windows (.venv_win)

### Bật venv và chạy (mỗi lần mở terminal mới)
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
Hoặc dùng `.bat`:
```powershell
.\.venv_win\Scripts\activate.bat
```

### Cài lần đầu (chỉ 1 lần)
```powershell
cd D:\1_Uni_ute\Graduation_Project\Yolo_V26_OBB_PCB\pcb_obb_system
python -m venv .venv_win
.\.venv_win\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Trỏ tới model mới sau train 1280
Sửa trong [config/system_config.yaml](../config/system_config.yaml):
```yaml
model:
  weights: "data/runs/train/pcb_obb_1280/weights/best.pt"
  imgsz: 1280      # khớp với imgsz lúc train
  conf: 0.60
```

### Nguồn camera
```yaml
camera:
  source: 0          # webcam mặc định USB
  # source: 1        # webcam thứ 2 nếu có nhiều
  # source: "data/belt_test.mp4"   # video file để test
  # source: "rtsp://192.168.x.x:554/..."  # IP camera
```
⚠️ Số nguyên `0/1/2` viết trần, không ngoặc kép. Đường dẫn/URL phải có ngoặc kép.

---

## 8. Những gì hiển thị trên cửa sổ --view

Sau khi sửa [src/detect_realtime.py](../src/detect_realtime.py), mỗi vật hiện:
- **Box OBB xanh lá** ôm vật + **chấm tâm** xanh dương.
- **Tên class + conf** (dòng trên, xanh lá).
- **Góc 0–360° dạng `xx.x deg`** (dòng dưới, vàng).
- **Mũi tên vàng** từ tâm chỉ đúng hướng đầu vật (heading).
- **Box marker vàng mảnh** — để kiểm tra việc detect marker có hoạt động không.
- **Đường kích hoạt đỏ** (trigger line) + **polygon ROI đỏ** (vùng băng tải).

Bấm **`q`** để thoát cửa sổ.

---

## 9. Lỗi môi trường thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `OSError: [Errno 95]` lúc train | `TMPDIR=/mnt/d` → Unix socket không hỗ trợ trên 9p/DrvFs | `unset TMPDIR` rồi chạy lại |
| `CUDA out of memory` | `--batch=-1` chọn quá cao | Ép `--batch 8` hoặc `--batch 4` |
| WSL sập khi cài torch CUDA | Ổ C: chỉ còn 4.4 GB | Cài venv CUDA trên D: (92 GB trống) |
| Góc chỉ 0–180° | Marker không detect được → fallback `theta mod S` | Nâng imgsz 1280 + thêm data marker |
| `No module named 'torch'` | Nhầm venv (dùng pcb_venv thay vì .venv_cuda) | `source .venv_cuda/bin/activate` |
| Màn hình camera đen/lỗi | Backend OpenCV Windows | Thêm `cv2.CAP_DSHOW` vào VideoCapture |

---

## 10. Lộ trình cải thiện độ chính xác

```
Hiện tại (100 epoch, imgsz 640, nano, CPU)
  marker recall: 48–59%  →  góc 0–180° (fallback)

Bước 1 — Chạy ngay (GPU, không cần data mới):
  --imgsz 1280 --epochs 300 --batch=-1
  Kỳ vọng: marker recall lên 65–75%

Bước 2 — Thêm data marker (gán nhãn thêm ~300–500 ảnh/loại):
  Kỳ vọng: marker recall lên 85–90%  →  góc 0–360° ổn định

Bước 3 — Nâng model lên Small (sau khi có đủ data):
  --weights yolo26s-obb.pt --imgsz 1280 --epochs 300
  Kỳ vọng: mAP50 tăng thêm 3–5%

Bước 4 — Export TensorRT/OpenVINO (sau khi accuracy đạt):
  python scripts/04_export.py
  Tăng tốc inference 3–5× so với .pt thuần
```

---

## 11. Checklist trước khi deploy lên rig thật

- [ ] `trigger_line.y_px` — chỉnh theo vị trí vạch kích hoạt thật trên khung camera.
- [ ] `roi.polygon` — chỉnh 4 đỉnh khớp vùng băng tải trong khung ảnh.
- [ ] `calibration.K`, `D` — hiệu chuẩn lens (checkerboard 9×6).
- [ ] `calibration.homography` — chụp bảng chuẩn để pixel→mm chính xác.
- [ ] `coordinate.y_fixed_mm` — đo thực tế khoảng cách trigger line→robot.
- [ ] `orientation.offset_deg` — đặt PCB đúng hướng "0°" rồi đọc góc hiện tại, lấy delta làm offset.
- [ ] Kiểm tra `names` trong `data.yaml` khớp `classes.txt` (QFP=0, TQFP=1, marker_QFP=2, marker_TQFP=3).
