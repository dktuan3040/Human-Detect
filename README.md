# Find Human

Ứng dụng phát hiện người qua webcam bằng xử lý ảnh cổ điển. Chương trình không sử dụng YOLO, MediaPipe hoặc mô hình học máy có sẵn; đối tượng được nhận diện từ vùng chuyển động có hình dạng gần giống đầu và vai.

## Tính năng

- Hiển thị hình ảnh webcam theo thời gian thực.
- Tự tạo ảnh nền khi cảnh đứng yên.
- Phát hiện vùng chuyển động bằng sai khác ảnh.
- Lọc nhiễu bằng Gaussian blur và phép opening.
- Gộp các vùng gần nhau, tách vùng theo khoảng trống.
- Vẽ khung màu xanh quanh vùng được xác định là người.
- Tự điều chỉnh ảnh nền khi ánh sáng môi trường thay đổi.

## Yêu cầu

- Windows có webcam.
- Python 3.12 hoặc mới hơn.
- Tkinter (thường có sẵn trong bản cài Python trên Windows).

Các thư viện Python cần dùng:

```text
numpy
opencv-python
pillow
```

## Cài đặt

Mở PowerShell tại thư mục dự án:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy opencv-python pillow
```

Nếu PowerShell không cho phép kích hoạt môi trường:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Chạy chương trình

### Giao diện chính

```powershell
python main.py
```

Trong cửa sổ chương trình:

- Nhấn `START` để mở webcam và bắt đầu xử lý.
- Nhấn `STOP` để dừng webcam.
- Nhấn `Esc` để đóng cửa sổ.

Giao diện được tạo trong [ui.py](./ui.py), còn điểm bắt đầu của chương trình là [main.py](./main.py).

### Pipeline OpenCV độc lập

```powershell
python pipeline.py
```

Pipeline này dùng `cv2.imshow()` để hiển thị các cửa sổ `Camera`, `Foreground` và `Background`. Nhấn `q` hoặc `Esc` để thoát.

## Cấu trúc dự án

```text
Find_Human/
├── main.py                 # Điểm bắt đầu của giao diện
├── ui.py                   # Giao diện Tkinter và vòng lặp webcam
├── pipeline.py             # Pipeline OpenCV độc lập
├── FindHuman.spec          # Cấu hình PyInstaller cho main.py
├── detection/
│   ├── motion.py           # Phát hiện chuyển động và tính tỷ lệ pixel
│   ├── contours.py         # Tìm và gộp vùng liên thông
│   └── person.py           # Kiểm tra hình dạng đầu và vai
└── processing/
    ├── gaussian.py         # Làm mờ Gaussian
    ├── threshold.py        # Phân ngưỡng ảnh
    └── morphology.py       # Erosion, dilation và opening
```

## Luồng xử lý

```text
Webcam
  ↓
Ảnh xám
  ↓
Gaussian blur
  ↓
Sai khác giữa các khung hình
  ↓
Phân ngưỡng và opening
  ↓
Tạo/cập nhật ảnh nền
  ↓
Tìm vùng liên thông
  ↓
Gộp và tách vùng
  ↓
Kiểm tra hình dạng người
  ↓
Vẽ khung phát hiện
```

## Tham số chính

Các tham số trong [pipeline.py](./pipeline.py) và phần tương ứng trong [ui.py](./ui.py) có thể được điều chỉnh:

| Tham số | Ý nghĩa |
| --- | --- |
| `MOTION_THRESHOLD` | Mức chênh lệch sáng tối thiểu để xem là chuyển động |
| `MOTION_RATIO` | Tỷ lệ pixel chuyển động để xác định cảnh đang động |
| `STATIC_WAIT_TIME` | Thời gian cảnh phải đứng yên trước khi tạo ảnh nền |
| `MIN_AREA` | Diện tích nhỏ nhất của vùng được xét |
| `LIGHT_RATIO` | Ngưỡng nhận biết thay đổi ánh sáng môi trường |
| `LIGHT_WAIT_TIME` | Thời gian chờ trước khi cập nhật ảnh nền |

## Đóng gói tùy chọn

`FindHuman.spec` là cấu hình PyInstaller. Cấu hình này đóng gói [main.py](./main.py) thành ứng dụng cửa sổ không hiện console, với tên `FindHuman`.

Nếu cần tạo lại bản đóng gói, chạy trong môi trường ảo đã kích hoạt:

```powershell
python -m PyInstaller --noconfirm FindHuman.spec
```

Đây chỉ là bước đóng gói tùy chọn; chạy dự án trong Python không cần tạo file `.exe`.

## Lưu ý

- Cần cấp quyền truy cập webcam cho Python/ứng dụng.
- Nếu không mở được camera, thử đóng các ứng dụng khác đang sử dụng webcam.
- Ánh sáng ổn định và phông nền ít nhiễu sẽ giúp kết quả tốt hơn.
- [ui.py](./ui.py) chứa logic hiển thị riêng cho giao diện; thay đổi trong [pipeline.py](./pipeline.py) không tự động cập nhật sang giao diện.
