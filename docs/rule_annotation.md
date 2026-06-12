CHỈ ĐÁNH 4 Điểm

###### class: 
TQFB: PCB lớn/ QFB: PCB nhỏ/ marker_QFB: chữ PCB nhỏ/ marker_TQFB: dòng dài của chữ PCB lớn

##### Quy ước:
Bám sát viền (Tightness): Đây là luật bất thành văn của OBB.
-> box không thíu không thừa, càng khít càng tốt

Thứ tự điểm (Vertices Order): Dù Roboflow tự động xử lý khi export, nhưng thói quen tốt là luôn click điểm theo một chiều nhất định (ví dụ: thuận chiều kim đồng hồ) để tránh lỗi topology của Polygon.
-> thứ tự theo chiều kim đồng hồ

Nhất quán về trục (Orientation Consistency): Với các vật thể dài, chiều dài của box OBB phải luôn dọc theo chiều dài của vật thể, giúp model học góc $\theta$ đồng nhất.
-> giúp xác định điểm đầu tiên, 
vd: vật thẳng đứng thì điểm đầu tiên là góc trên bên phải -> góc dưới bên phải -> góc dưới bên trái -> góc trên bên trái ---> trục sẽ là từ góc trên bên phải xuống góc dưới bên phải ứng với đoạn dài nhất này

vật ngang thì điểm đầu là góc trên bên trái -> góc trên bên phải .... --> trục sẽ tương ứng, là đoạn dài nhất này

###### scenario
Bị che khuất (Occlusion / Overlap)
Mức độ nhẹ (< 30%): Tưởng tượng và đánh nhãn toàn bộ hình dáng gốc của vật thể,
Mức độ nặng (> 50%): Nếu vật thể bị đè quá nhiều -> chỉ đánh nhãn phần nhìn thấy được.
Mờ do chuyển động
Chỉ vẽ OBB ôm lấy phần "lõi" sắc nét của vật thể. Tuyệt đối bỏ qua các vệt mờ ảo
Ánh sáng thấp, bóng đổ: không gộp bóng đổ vào trong OBB
Bị biến dạng, méo mó: Vẫn vẽ 4 điểm OBB bám theo sát kích thước thực tế đang bị biến dạng đó. Không vẽ hình chữ nhật to như kích thước ban đầu
