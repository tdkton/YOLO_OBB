# Nguyên lý hoạt động — Hệ thống phân loại PCB bằng YOLO26-OBB

Tài liệu mô tả đầy đủ nguyên lý của hệ thống thị giác máy: phân biệt **2 loại PCB**
(`TQFP`, `QFP`) đang **di chuyển liên tục trên băng tải (11–20 mm/s)**, xuất ra **loại
PCB**, **vị trí X**, và **góc xoay 0–360°** ngay khi vật đi qua một **đường kích hoạt cố
định** (trigger line). Vị trí **Y là hằng số** (= vị trí đường kích hoạt). Model dùng
**Ultralytics YOLO26-OBB** (NMS-free / end-to-end), chạy trên **Ubuntu 24.04 (WSL2)**.

---

## 1. Bài toán & tín hiệu vào/ra

| Thành phần | Mô tả |
|---|---|
| Đầu vào | Khung hình từ camera công nghiệp đặt vuông góc nhìn xuống băng tải |
| Loại PCB | `TQFP`, `QFP` (phân biệt bằng hình/kích thước board → do OBB quyết định) |
| Mốc bất đối xứng | Mỗi loại có **một box silkscreen trắng riêng** (`marker_TQFP`, `marker_QFP`) |
| Tốc độ băng tải | 11–20 mm/s, một chiều |
| Đầu ra mỗi vật | `type`, `x_mm`, `angle_deg` ∈ [0,360) |
| Hằng số | `y_mm` = vị trí trigger line (cố định) |
| Điều kiện phát | Tâm board **cắt qua trigger line** → chốt dữ liệu → gửi 1 gói |

Gói dữ liệu (JSON, 1 dòng / 1 board):
```json
{ "type": "QFP", "track_id": 7, "x_mm": 142.6, "y_mm": 0.0, "angle_deg": 287.5, "conf": 0.94, "ts": 1733740000.123 }
```

---

## 2. Kiến trúc tổng thể

```
Camera ─► [1] Thu hình & undistort
            │
            ▼
        [2] YOLO26-OBB suy luận  ─► các OBB: (cx, cy, w, h, θ, class, conf)
            │
            ▼
        [3] Tách: board (TQFP/QFP)  vs  marker (box trắng)
            │
            ▼
        [4] Bám vết board bằng centroid tracker  ─► gán track_id ổn định
            │
            ▼
        [5] Trigger line: board cắt vạch (upstream→downstream) → phát 1 lần
            │
            ▼
        [6] Pixel→mm (homography) cho X ; Y = hằng số
            │
            ▼
        [7] Góc 0–360°: OBB (mod 180°) + hướng tâm→marker phá nhập nhằng
            │
            ▼
        [8] Đóng gói JSON ─► gửi TCP về hệ thống / PLC / robot Delta
```

Tương ứng mã nguồn: [src/detect_realtime.py](../src/detect_realtime.py) điều phối toàn bộ;
[src/tracker.py](../src/tracker.py), [src/trigger.py](../src/trigger.py),
[src/geometry.py](../src/geometry.py), [src/orientation.py](../src/orientation.py),
[src/comms.py](../src/comms.py) là các khối chức năng.

---

## 3. Vì sao YOLO26-OBB

- **OBB (Oriented Bounding Box):** thay vì hộp thẳng `(x, y, w, h)`, OBB thêm **góc θ**
  → hộp *nghiêng* ôm sát vật. Bắt buộc vì PCB rơi ngẫu nhiên trên băng tải; hộp thẳng
  không cho được góc gắp.
- **YOLO26** là dòng **NMS-free / end-to-end**: bỏ bước hậu xử lý NMS → độ trễ thấp và
  ổn định hơn, hợp dây chuyền công nghiệp.
- **Chọn bản Nano `yolo26n-obb`:** chỉ 2 loại, lại tách biệt rõ → Nano thừa sức; xuất
  TensorRT/OpenVINO nhẹ, chạy tốt trên WSL. Chỉ nâng `yolo26s-obb` nếu sai số góc θ
  hoặc mAP chưa đạt.

Mỗi vật model trả về `xywhr = (cx, cy, w, h, θ)` (pixel + radian) kèm `class`, `conf`.

---

## 4. Nguyên lý từng bước (xử lý mỗi khung hình)

### [1] Thu hình & hiệu chỉnh méo (undistort)
Ảnh thô bị méo do ống kính (nhất là ở rìa khung). Dùng ma trận nội (camera matrix `K`)
và hệ số méo `D` từ hiệu chuẩn để `cv2.undistort()`, đảm bảo PCB ở rìa khung không bị
biến dạng so với ở giữa → đo X và góc mới chính xác.

### [2] Suy luận YOLO26-OBB
Chạy model trên khung đã undistort, thu danh sách OBB. Mỗi OBB là một ứng viên: có thể
là **board** (`TQFP`/`QFP`) hoặc **marker** (`marker_TQFP`/`marker_QFP`).

### [3] Tách board và marker
Dựa vào tên class:
- `pcb_classes = {TQFP, QFP}` → đối tượng sẽ phát gói.
- `marker_map = {marker_TQFP→TQFP, marker_QFP→QFP}` → chỉ dùng **để tính góc**.

Marker **không** được bám vết, **không** phát gói riêng.

### [4] Bám vết (centroid tracker)
Vì băng tải **rất chậm** (≈0,5 mm/khung ở 15 mm/s, 30 FPS), tâm board nằm *gần* vạch
trong **nhiều khung liên tiếp**. Nếu không có định danh, một board sẽ kích hoạt **hàng
chục gói trùng**.

Giải pháp: **centroid tracker nhẹ** (~30 dòng) — mỗi khung, ghép mỗi tâm board với track
gần nhất trong bán kính `max_match_dist_px`; nếu không có thì tạo track mới; track không
thấy quá `max_missing_frames` khung thì bị xoá. Kết quả: **mỗi board vật lý có một
`track_id` ổn định**.

> Vì sao không dùng ByteTrack/BoT-SORT: chúng giải bài toán chuyển động nhanh, che khuất,
> giao cắt quỹ đạo, re-ID — **không xảy ra** trên băng tải một làn, một chiều, chậm. Dùng
> tracker nặng là thừa và làm tăng độ trễ.

### [5] Trigger line — phát đúng 1 lần/board
Đặt một **vạch ngang cố định** tại `y_px`. Với mỗi track tính "phía" của tâm so với vạch
theo chiều băng tải: `-1` (thượng nguồn) / `+1` (hạ nguồn).

Gói chỉ phát khi track **chuyển từ −1 sang +1** (đúng lúc cắt vạch) và **chưa từng phát**:

```
prev_side = -1  và  side = +1  và  not triggered   ⇒  PHÁT, đặt triggered = True
```

Định danh từ tracker + cờ `triggered` ⇒ **không bao giờ trùng**, dù vật đứng sát vạch
nhiều khung.

### [6] Hệ toạ độ: X đo được, Y cố định
- **Y cố định:** vì vạch nằm cố định trong khung, mọi board cắt vạch đều ở cùng một vị trí
  vật lý theo phương băng tải ⇒ `y_mm = y_fixed_mm` (hằng số trong config).
- **X đo:** đổi tâm pixel `(cx, cy)` sang mm bằng **đồng nhất thức homography** 3×3 `H`
  (giải từ hiệu chuẩn ngoại `cv2.findHomography`):

$$
\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = H \begin{bmatrix} c_x \\ c_y \\ 1 \end{bmatrix},
\qquad
X_{mm} = \frac{X'}{w}
$$

  Nếu chưa hiệu chuẩn (H = đơn vị), hệ dùng tạm tỉ lệ vô hướng `pixels_per_mm`.

### [7] Góc xoay 0–360° (cốt lõi)
**Bước a — góc OBB (mod 180°).** OBB cho góc θ chính xác theo cạnh board, nhưng hình chữ
nhật đối xứng nửa vòng nên θ **nhập nhằng 180°** (không biết "đầu" hay "đuôi"). Góc thô của
Ultralytics nằm trong dải:

$$\theta \in [-\tfrac{\pi}{4}, \tfrac{3\pi}{4}) = [-45^\circ, 135^\circ)$$

**Bước b — dùng box trắng phá nhập nhằng.** Box trắng là đặc trưng **bất đối xứng** gắn cố
định trên board. Ghép marker thuộc board (tâm marker nằm trong đa giác OBB của board, gần
tâm nhất). Lấy vector từ tâm board tới tâm marker:

$$\phi = \operatorname{atan2}(m_y - c_y,\; m_x - c_x)$$

Trong hai ứng viên $\{\theta, \theta+180^\circ\}$, chọn cái **cùng phía với marker** (sai
khác góc nhỏ nhất so với φ):

$$
heading = \arg\min_{\alpha \in \{\theta,\ \theta+180\}} \big|\angle(\alpha - \phi)\big|,
\qquad
angle\_deg = (heading + offset)\bmod 360
$$

→ Ra **góc duy nhất trong [0, 360)**. `offset_deg` để định nghĩa tư thế nào đọc là 0°.

> **Suy giảm mượt:** nếu khung đó không thấy marker (mờ/khuất), board vẫn xuất góc nhưng
> rơi về dải [-90°, 90°) (mất khả năng phân biệt 360° riêng khung đó), không báo sai.

### [8] Phân loại PCB — chỉ tin OBB
Cấu hình hiện tại `cross_check: false`: **loại PCB lấy thẳng từ class của OBB**
(`names[cls_id]`). Marker **không** tham gia quyết định loại — chỉ phục vụ tính góc. (Có
thể bật `cross_check: true` để marker đè/sửa loại khi hai size quá giống nhau, nhưng ở đây
đã tắt theo yêu cầu.)

### [9] Đóng gói & truyền
Tạo JSON `{type, track_id, x_mm, y_mm, angle_deg, conf, ts}` và gửi qua **TCP** (1 dòng kết
thúc `\n`) về hệ thống/PLC. Khối comms tự kết nối lại nếu phía nhận tạm mất, không làm
treo vòng lặp thị giác. Phía PLC sau đó lo phần bám băng tải bằng encoder (FIFO + HSC) để
robot Delta gắp — nằm ngoài phạm vi gói thị giác này.

---

## 5. Nguyên lý huấn luyện (train)

- **Khởi tạo từ `yolo26n-obb.pt`** (transfer learning) trên 4 class:
  `TQFP`, `QFP`, `marker_TQFP`, `marker_QFP`.
- **Tăng cường dữ liệu an toàn cho OBB** (trong [scripts/03_train.py](../scripts/03_train.py)):
  - `degrees=180` (board rơi mọi hướng), `translate≈0.10`, `scale≈0.05`, lật ngang/dọc,
    thêm motion-blur nhẹ.
  - **Tắt `perspective` và `shear`** (= 0): hai phép này **làm hỏng nhãn góc OBB**.
- **Chia tập 70/20/10** (train/val/test). Theo dõi `mAP50`, `mAP50-95`, và **đặc biệt sai
  số góc θ** — vài độ lệch cũng đủ làm gripper gắp trượt/kẹp lệch linh kiện.
- **Xuất tối ưu phần cứng:** TensorRT (`.engine`) cho GPU NVIDIA, OpenVINO cho CPU Intel →
  giảm độ trễ suy luận từ ~30 ms xuống 3–5 ms.

---

## 6. Tổng kết luồng dữ liệu

```
Khung hình
   │  undistort
   ▼
YOLO26-OBB ──► {boards: (cx,cy,w,h,θ,cls,conf)}  +  {markers}
   │
   ├─ board → centroid tracker → track_id
   │            │
   │            ▼  (khi cắt trigger line, đúng 1 lần)
   │      X_mm = homography(cx,cy) ;  Y_mm = hằng số
   │      type  = class OBB                         (cross_check=false)
   │      angle = OBB θ (mod 180) + hướng tâm→marker → [0,360)
   │            │
   ▼            ▼
 markers ─► chỉ để chọn nửa cho góc (không phát gói, không quyết loại)
                │
                ▼
        JSON {type, x_mm, y_mm, angle_deg, conf, ts} ──TCP──► PLC / Delta robot
```

---

## 7. Các tham số phải hiệu chỉnh trên rig thật

1. **Nội (intrinsics):** `K`, `D` từ bàn cờ → mục `undistort` trong
   [config/system_config.yaml](../config/system_config.yaml).
2. **Ngoại (pixel→mm):** homography `H` + `y_fixed_mm` (vị trí Y vật lý của trigger line).
3. **Trigger line:** `trigger_line.y_px` đặt đúng hàng pixel của vạch.
4. **Góc:** `orientation.offset_deg` để tư thế "chuẩn" đọc ra 0°.
5. **Ánh sáng/khẩu:** shutter nhanh (~1/500 s) + LED khuếch tán để khử nhoè và chống loá
   pad hàn.
