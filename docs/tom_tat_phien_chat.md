# Tóm tắt phiên làm việc — Hệ thống phân loại PCB YOLO26-OBB

Tài liệu tổng hợp toàn bộ những gì đã thực hiện, quyết định và xử lý trong phiên chat:
xây dựng hệ thống thị giác phân loại PCB trên băng tải, xác định loại + vị trí X + góc xoay
0–360°, chạy trên WSL Ubuntu 24.04 / Windows.

---

## 1. Bài toán

- **2 loại PCB** trên băng tải chạy liên tục **11–20 mm/s**:
  - **TQFP** — board **chữ nhật** (đối xứng xoay 180°)
  - **QFP** — board **vuông** (đối xứng xoay 90°)
- Mỗi loại có **một box silkscreen trắng riêng** làm mốc (`marker_TQFP`, `marker_QFP`).
- Đầu ra mỗi vật khi cắt **đường kích hoạt cố định**: `type`, `x_mm`, `angle_deg ∈ [0,360)`.
- **Y cố định** = vị trí trigger line. Model: **Ultralytics YOLO26-OBB**.

---

## 2. Các quyết định thiết kế chính

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Model | **`yolo26n-obb`** (Nano) | 2 loại tách biệt rõ, băng tải chậm → Nano đủ; xuất TensorRT/OpenVINO nhẹ |
| "YOLO v26" | = **Ultralytics YOLO26** (NMS-free, weights `yolo26*-obb.pt`, pretrained **DOTA** không phải COCO) | OBB không thể pretrained trên COCO (COCO không có hộp xoay) |
| Tracking | **Centroid tracker nhẹ (~30 dòng)**, không dùng ByteTrack/BoT-SORT | Băng tải 1 làn, chậm, không che khuất → tracker nặng thừa |
| Chống trùng gói | Trigger line **1 lần/track id** (cạnh upstream→downstream) | Băng tải chậm → tâm nằm gần vạch nhiều frame |
| Góc 0–360° | OBB cho góc cạnh (mod S) + **vector tâm→marker** chọn hướng | Hình chữ nhật/vuông đối xứng → OBB không đủ |
| Đối xứng theo loại | `symmetry_by_class: {TQFP: 180, QFP: 90}` | Vuông 4 ứng viên, chữ nhật 2 ứng viên — khác nhau |
| Phân loại | **Chỉ tin OBB** (`cross_check: false`) | Vuông vs chữ nhật khác tỉ lệ cạnh rõ → OBB tin cậy |
| Vật vuông (QFP) | Giữ OBB ôm sát + symmetry 90°, **không** dùng trick kéo chữ nhật | Trick làm lệch tâm/X, hỏng phân loại |
| ROI | Lọc detection theo **tâm nằm trong polygon đỏ** (băng tải) | Bỏ qua đống PCB ngoài băng tải |

---

## 3. Cấu trúc dự án

```
pcb_obb_system/
├── README.md                       # tài liệu thiết kế đầy đủ (EN)
├── system_overall.md                # tổng quan + data prep + Roboflow + cách tính góc (EN)
├── requirements.txt
├── setup_wsl.sh
├── config/
│   ├── classes.txt                  # QFP / TQFP / marker_QFP / marker_TQFP
│   └── system_config.yaml            # camera, ROI, calib, trigger line, orientation, comms
├── scripts/
│   ├── 00_extract_frames.py          # tách frame từ video
│   ├── 01_make_data_yaml.py          # sinh data.yaml từ classes.txt
│   ├── 02_split_dataset.py           # chia 70/20/10
│   ├── 03_train.py                   # train YOLO26-OBB (có cờ --data)
│   └── 04_export.py                  # export TensorRT/OpenVINO
├── src/
│   ├── geometry.py                   # pixel→mm homography + chuẩn hoá góc
│   ├── tracker.py                    # centroid tracker
│   ├── trigger.py                    # line-crossing 1 lần
│   ├── orientation.py                # ghép marker + heading 0–360 theo symmetry
│   ├── comms.py                      # gửi gói JSON qua TCP
│   └── detect_realtime.py            # pipeline chính
└── docs/
    ├── nguyen_ly_hoat_dong.md         # nguyên lý vận hành (VI)
    ├── nguyen_ly_toan_hoc_goc.tex     # nguyên lý toán học tính góc (LaTeX, VI)
    └── tom_tat_phien_chat.md          # file này
```

---

## 4. Công thức tính góc (cốt lõi)

```
phi      = atan2(my - cy, mx - cx)                 # hướng tâm board -> marker
S        = 180° (TQFP, chữ nhật) | 90° (QFP, vuông)
C        = { theta + k*S : k = 0 .. 360/S - 1 }    # 2 hoặc 4 ứng viên
heading  = argmin over a in C of  góc_lệch(a, phi)
angle_deg = (heading + offset_deg) mod 360°         # [0, 360)
```
Biên chọn đúng: `góc_lệch(phi, ung_vien_dung) < S/2` (TQFP 90°, QFP 45°).
Mất marker frame nào → fallback `theta mod S` (không có 360° riêng frame đó).
→ Chi tiết toán học: [nguyen_ly_toan_hoc_goc.tex](nguyen_ly_toan_hoc_goc.tex).

---

## 5. Lịch sử thay đổi trong phiên

1. Dựng toàn bộ hệ thống 5-stage từ `References_Code/` + pipeline gốc.
2. Đổi từ 3 loại (3×3, 3×4, 4×5) → 2 loại + marker để ra góc 360°.
3. Thêm marker theo từng loại (`marker_map`) + `cross_check` (sau đó tắt `cross_check`).
4. Đổi tên class: `TQFB/QFB` → **`TQFP/QFP`**.
5. Thêm `symmetry_by_class` (vuông 90° / chữ nhật 180°).
6. Đổi hình dạng: TQFP = **chữ nhật (180°)**, QFP = **vuông (90°)**.
7. Viết nguyên lý toán học (chuyển từ Markdown lỗi font → **LaTeX `.tex`**).
8. Thêm cờ `--data` cho `03_train.py` (trỏ thẳng data.yaml Roboflow).
9. Sửa `classes.txt` khớp thứ tự Roboflow (**QFP, TQFP, marker_QFP, marker_TQFP**).
10. Thêm **ROI polygon** (chỉ detect trong vùng đỏ băng tải).
11. Thêm **hiển thị góc 0–360° + mũi tên hướng** trên cửa sổ camera.
12. (Thử rồi hoàn lại) tách 2 ngưỡng conf/det_conf → **trở về 1 ngưỡng `conf = 0.60`**.

---

## 6. Cấu hình hiện tại (chốt)

- `classes.txt`: `QFP`(0), `TQFP`(1), `marker_QFP`(2), `marker_TQFP`(3) — khớp data.yaml Roboflow.
- `orientation`: `cross_check: false`; `symmetry_by_class: {TQFP: 180, QFP: 90}`; `offset_deg: 0`.
- `model`: `conf: 0.60`, `iou: 0.70`, `imgsz: 640`, weights `data/runs/train/pcb_obb/weights/best.pt`.
- `roi.enabled: true` (polygon 4 đỉnh — toạ độ ước lượng, cần chỉnh theo khung thật).
- Dataset đang dùng: `data/data1/` (hoặc folder Roboflow `Yolov26_OBB_PCB.v1i.yolov8-obb`), 70/20/10.

---

## 7. Quy trình làm việc

### Train (đã có `best.pt`)
```bash
python scripts/03_train.py --data data/data1/data.yaml
```
- KHÔNG chạy `00/01/02` nếu dùng data Roboflow đã chia sẵn.
- ⚠️ Thứ tự `names` trong data.yaml phải khớp `classes.txt`.

### Test nhanh
```bash
yolo obb predict model=data/runs/train/pcb_obb/weights/best.pt source=data/data1/test/images save=True
```

### Chạy camera thật (khuyến nghị trên Windows)
```powershell
cd D:\1_Uni_ute\Graduation_Project\Yolo_V26_OBB_PCB\pcb_obb_system
python -m venv .venv_win
.\.venv_win\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src\detect_realtime.py --config config\system_config.yaml --view --no-comms
```
`camera.source: 0` = webcam mặc định. Cửa sổ hiện box OBB + tên + góc 0–360° + mũi tên hướng.

---

## 8. Vấn đề môi trường đã gặp & cách xử lý

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Import `torch` đứng rất lâu | `.venv` nằm trên `/mnt/d` (DrvFs chậm) | Đặt venv trên `~` (ext4) — nhưng C: đầy nên cân nhắc |
| WSL sập `exit code 1` khi cài torch CUDA | **Ổ C: chỉ còn 4.4 GB** (CUDA cần ~6–7 GB) | Cài CUDA vào venv trên **D:** (92 GB) hoặc dọn C: |
| `py` not recognized (Windows) | Chưa có Python launcher; có `python` (3.9.1) | Dùng `python -m venv ...` |
| `OSError: [Errno 95]` lúc train | `TMPDIR=/mnt/d` → multiprocessing tạo Unix socket trên 9p (không hỗ trợ) | `unset TMPDIR` rồi train lại (hoặc `workers=0`) |
| Góc chỉ 0–180 | Marker không được dùng (không detect/ghép) → fallback | Kiểm tra box vàng marker trong `--view`; nếu thiếu → thêm data train marker |

**Lưu ý GPU vs đĩa:** chạy GPU trên D: **không chậm** phần tính toán (chỉ import lâu hơn chút).
Chỉ cần `nvidia-smi` thấy GPU + cài torch `cu124` là train GPU được.

---

## 9. Trạng thái & việc còn lại

**Đã xong:** dựng hệ thống, train ra `best.pt` (CPU), hiển thị góc 360° + ROI, môi trường chạy được.

**Cần làm tiếp:**
1. **Hiệu chuẩn rig thật** (gate độ chính xác): intrinsics `K,D`; homography pixel→mm;
   `trigger_line.y_px`; `coordinate.y_fixed_mm`; `roi.polygon`; `orientation.offset_deg`.
2. **Kiểm tra marker** thực sự được detect (nếu góc chưa đạt 360° → thiếu data marker → gán
   thêm ~300–500 ảnh/loại, phủ nhiều góc xoay).
3. (Tùy chọn) Export TensorRT (GPU) / OpenVINO (CPU) để tăng tốc.
4. Nối `comms` TCP tới PLC/robot Delta khi triển khai thật.
