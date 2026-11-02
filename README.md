# Phát hiện người qua camera (xử lý ảnh tự cài đặt)

Chương trình đọc hình từ webcam cố định, tự tạo ảnh nền, tìm vùng chuyển động và vẽ **khung xanh** quanh người ngồi trước camera (nhận dạng theo dáng **đầu + vai**).

Nguyên tắc dự án: các thuật toán xử lý ảnh (xám, Gaussian, threshold, morphology, thành phần liên thông) đều **tự cài đặt bằng NumPy**, không dùng model có sẵn (YOLO, MediaPipe, Haar, pretrained). OpenCV chỉ dùng để đọc camera, hiển thị và vẽ khung.

> Lưu ý: bước "xác nhận là người" hiện là **luật theo hình dáng** (% pixel + dáng đầu/vai), chưa phải bộ phân loại học từ dữ liệu (HOG + dataset). Vì vậy kết quả là "vùng có dáng đầu + vai", chưa phải nhận dạng người chính xác tuyệt đối.

---

## 1. Cấu trúc thư mục

```
project/
├── main.py                  # vòng lặp camera, tạo nền, to_grayscale
├── processing/
│   ├── gaussian.py          # gaussian_blur (kernel 3x3)
│   ├── threshold.py         # threshold_image
│   └── morphology.py        # erosion, dilation, opening, closing
└── detection/
    ├── motion.py            # detect_motion, calculate_motion_ratio
    ├── contours.py          # find_connected_components (Union-Find), merge_near_boxes
    └── person.py            # split_by_gap, has_head_and_shoulders, is_person
```

## 2. Yêu cầu

- Python 3 (đang chạy trên 3.12)
- `opencv-python`
- `numpy`
- Một webcam (mặc định `cv2.VideoCapture(0)`)

```bash
pip install opencv-python numpy
```

## 3. Hướng dẫn sử dụng

1. Chạy từ **thư mục gốc** của dự án (nơi có `main.py` và hai thư mục `processing`, `detection`):
   ```bash
   python main.py
   ```
2. **Đứng ra ngoài khung hình** và giữ yên cảnh khoảng **5 giây**. Terminal hiện `Đã tạo background` là đã lưu xong ảnh nền.
3. Bước vào khung hình. Khi có người ngồi trước camera, khung xanh hiện trên cửa sổ `Camera`.
4. Thoát bằng phím **`q`** hoặc **`Esc`**.

| Cửa sổ | Ý nghĩa |
|---|---|
| `Camera` | Hình thật kèm khung xanh |
| `Foreground` | Mặt nạ trắng/đen của vùng khác nền (dùng để chỉnh ngưỡng) |
| `Background` | Ảnh nền đã lưu |

> Nếu lúc khởi động đã có người trong khung, người đó bị lưu vào nền và sẽ không được phát hiện. Hãy khởi động lại và đứng ra ngoài.

## 4. Luồng hoạt động

```
Camera → to_grayscale (tự cài) → gaussian_blur
            │
            ├─ trừ frame liên tiếp → threshold → opening → motion_ratio
            │       > MOTION_RATIO : đang động
            │       ≤ MOTION_RATIO suốt STATIC_WAIT_TIME giây : tĩnh (is_static)
            │
            ├─ chưa có nền & tĩnh đủ lâu → LƯU NỀN
            │
            └─ đã có nền:
                 trừ nền → threshold → opening → bg_ratio
                 bg_ratio > LIGHT_RATIO ? → ánh sáng đổi: bù sáng (median) rồi so lại;
                                            kéo dài LIGHT_WAIT_TIME giây và cảnh tĩnh → cập nhật nền
                 ↓
                 find_connected_components → lọc MIN_AREA
                 ↓
                 merge_near_boxes (gộp ô cách nhau < 40 pixel)
                 ↓
                 split_by_gap (tách lại nếu có khe trống giữa 2 người)
                 ↓
                 is_person (điều kiện 1 + điều kiện 2)
                 ↓
                 vẽ khung xanh
```

### Xác nhận là người (`detection/person.py`)

Một box được giữ lại khi đạt **cả hai** điều kiện:

1. **Đủ lớn:** pixel trắng của box chiếm hơn `PERSON_RATIO`% cả khung hình (tính bằng `calculate_motion_ratio`).
2. **Dáng đầu + vai** (`has_head_and_shoulders`): đếm độ rộng theo từng hàng, đi từ trên xuống để tìm đỉnh độ rộng đầu tiên (đầu), gặp chỗ thắt lại (cổ), rồi phía dưới phải có hàng rộng ≥ `SHOULDER_RATIO` lần đầu (vai). Phần dưới vai không được xét, nên người đứng hay ngồi đều dùng được.

### Tách hai người (`split_by_gap`)

Đếm pixel trắng theo từng cột của box. Có khe trống từ `GAP_MIN` cột trở lên thì cắt thành hai box riêng, mỗi box tự đi qua `is_person`.

## 5. Các tham số có thể chỉnh

| Tham số | Nằm ở | Giá trị hiện tại | Tác dụng |
|---|---|---|---|
| `MOTION_THRESHOLD` | main.py | 30 | Độ lệch mức xám để tính là có chuyển động |
| `MOTION_RATIO` | main.py | 2 | % pixel động, vượt thì coi là cảnh đang động |
| `STATIC_WAIT_TIME` | main.py | 5 | Số giây tĩnh liên tục để tạo nền |
| `MIN_AREA` | main.py | 2000 | Vùng nhỏ hơn (w×h) bị bỏ |
| `LIGHT_RATIO` | main.py | 60 | Hơn % này khác nền thì coi là đổi ánh sáng |
| `LIGHT_WAIT_TIME` | main.py | 30 | Ánh sáng đổi liên tục bấy nhiêu giây thì cập nhật nền |
| `gap` | contours.py (`merge_near_boxes`) | 40 | Hai ô cách nhau dưới số pixel này thì bị gộp |
| `PERSON_RATIO` | person.py | 2 | % pixel tối thiểu của box so với khung hình |
| `SHOULDER_RATIO` | person.py | 1.4 | Vai phải rộng gấp bao nhiêu lần đầu |
| `NECK_DROP` | person.py | 0.85 | Độ rộng tụt dưới mức này thì coi là cổ |
| `MIN_HEAD_RATIO` | person.py | 0.15 | Đầu tối thiểu chiếm bao nhiêu bề ngang box |
| `GAP_MIN` | person.py | 3 | Khe trống tối thiểu (số cột) để tách |
| `EMPTY_MAX` | person.py | 2 | Cột có ≤ số pixel trắng này tính là cột trống |
| `MIN_PIECE_W` | person.py | 10 | Mảnh hẹp hơn bấy nhiêu cột bị bỏ |

Mẹo chỉnh: `is_person` in ra `ratio=...%` và `shoulder=...` thật của từng box. Đặt ngưỡng thấp hơn số đó một chút.

## 6. Xử lý sự cố

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `Khong doc duoc Camera` | Sai chỉ số camera hoặc camera đang bị app khác dùng | Đổi `VideoCapture(0)` thành `1`, đóng app khác |
| Mãi không tạo được nền | Cảnh luôn bị coi là động | Tăng `MOTION_RATIO`, giữ yên cảnh |
| Không có khung xanh khi ngồi | `PERSON_RATIO` hoặc `SHOULDER_RATIO` quá cao | Hạ thấp hơn số `ratio`/`shoulder` in ra; nếu `shoulder=0.00` mãi thì tăng `NECK_DROP` lên 0.92 |
| Khung xanh hiện cho vật khác | Ngưỡng quá thấp | Tăng `PERSON_RATIO`, `SHOULDER_RATIO` |
| Hai người cách nhau vẫn một khung | Khe nhỏ hoặc có nhiễu trong khe | Tăng `EMPTY_MAX` lên 5, hoặc giảm `GAP_MIN` xuống 2 |
| Một người bị chia thành hai khung | `GAP_MIN` quá nhỏ | Tăng `GAP_MIN` lên 6 đến 10 |
| Mất vai khi mặc đồ tối ngồi trước vật tối | Mức xám của áo và nền chênh dưới `MOTION_THRESHOLD`, mặt nạ bị thủng | Giảm `MOTION_THRESHOLD`, hoặc đổi nền/quần áo để tương phản hơn |
| `ModuleNotFoundError: detection` | Chạy sai thư mục | Chạy `python main.py` từ thư mục gốc |

## 7. Hạn chế hiện tại

- Hai người **dính sát hẳn** (vai chạm vai, không còn khe trống) chưa tách được.
- Chỉ nhận đúng dáng **đầu + vai** nhìn thẳng vào camera; người quay lưng hoặc nghiêng nhiều có thể không nhận.
- Dựa vào trừ nền nên camera phải đứng yên; thay đổi ánh sáng mạnh có thể làm hỏng nền (đã có bù sáng nhưng không hoàn hảo).
- Chưa có bộ phân loại học từ dữ liệu (HOG + dataset), phần xác nhận người đang là luật theo hình dáng.

## 8. Hướng phát triển

- Tách hai người dính sát (tìm "thung lũng" giữa hai đầu).
- HOG + dataset Person / Non-person + bộ phân loại tự cài đặt.
- Kiểm tra trong nhiều điều kiện ánh sáng và tối ưu tốc độ.
