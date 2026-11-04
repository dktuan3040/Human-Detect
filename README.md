# Find Human

Chương trình phát hiện người qua webcam bằng các kỹ thuật xử lý ảnh cơ bản. Dự án không dùng YOLO, MediaPipe hay model học máy có sẵn. Người được xác định dựa trên vùng chuyển động có hình dáng gần giống **đầu và vai**.

## 1. Cài đặt

Mở PowerShell tại thư mục dự án:

```powershell

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy opencv-python pillow
```

Nếu PowerShell chặn việc kích hoạt môi trường:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Chạy chương trình

### Chạy giao diện Tkinter

Đây là cách chạy chính:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

`main.py` chỉ tạo cửa sổ và gọi giao diện trong [ui.py](./ui.py). Bấm START để mở camera, STOP để dừng; Esc để đóng cửa sổ.

Giao diện có bản sao logic xử lý, không import [pipeline.py](./pipeline.py). Vì vậy sửa tham số trong một file không tự cập nhật file còn lại.

### Chạy pipeline OpenCV gốc

```powershell
python pipeline.py
```

File này mở các cửa sổ `Camera`, `Foreground` và `Background` bằng `cv2.imshow()`. Thoát bằng phím `q` hoặc `Esc`.

## 3. Cấu trúc thư mục

```text
Find_Human/
├── main.py                    # File chạy giao diện chính
├── ui.py                      # Giao diện Tkinter và vòng xử lý camera
├── pipeline.py                # Pipeline gốc dùng cv2.imshow
├── processing/
│   ├── gaussian.py            # Làm mờ Gaussian tự cài đặt
│   ├── threshold.py           # Chuyển ảnh thành đen/trắng
│   └── morphology.py          # Erosion, dilation, opening, closing
└── detection/
    ├── motion.py              # Tìm chuyển động và tính tỷ lệ pixel trắng
    ├── contours.py            # Tìm và gộp vùng liên thông
    └── person.py              # Tách vùng và kiểm tra hình đầu + vai
```

## 4. Luồng xử lý ảnh

```text
Camera
  ↓
Ảnh màu BGR
  ↓
to_grayscale()
  ↓
gaussian_blur()
  ↓
detect_motion(frame trước, frame hiện tại)
  ↓
threshold_image()
  ↓
opening()
  ↓
Tính motion_ratio
  ↓
Tạo background khi cảnh đứng yên
  ↓
Trừ background → trị tuyệt đối → ngưỡng → opening
  ↓
Nếu khác nền > 70%: bù sáng rồi tính lại foreground
  ↓
find_connected_components()
  ↓
Giữ khung có w × h >= MIN_AREA
  ↓
merge_near_boxes()
  ↓
split_by_gap()
  ↓
is_person()
  ↓
Vẽ khung xanh quanh người
```

## 5. Giải thích từng bước

### Bước 1: Đọc camera

```python
ret, frame = camera.read()
```

`frame` là một ảnh màu. OpenCV lưu màu theo thứ tự BGR:

```text
Blue - Green - Red
```

### Bước 2: Chuyển sang ảnh xám

Hàm `to_grayscale()` lấy ba kênh màu rồi tính độ sáng:

```python
0.114 * Blue + 0.587 * Green + 0.299 * Red
```

Ảnh xám chỉ có một giá trị cho mỗi pixel, từ 0 đến 255.

### Bước 3: Làm mờ Gaussian

Hàm `gaussian_blur()` dùng kernel 3x3:

```text
1 2 1
2 4 2   / 16
1 2 1
```

Mục đích là giảm nhiễu nhỏ do webcam trước khi tìm chuyển động.

### Bước 4: Tìm chuyển động

Hàm `detect_motion()` lấy giá trị tuyệt đối của:

```text
frame hiện tại - frame trước
```

Nếu độ chênh lệch lớn hơn `MOTION_THRESHOLD`, pixel được đổi thành trắng. Ngược lại, pixel là đen.

### Bước 5: Lọc nhiễu

Hàm `opening()` thực hiện:

```text
erosion → dilation
```

Nó loại các điểm trắng nhỏ và giữ lại vùng chuyển động lớn hơn.

### Bước 6: Tính tỷ lệ chuyển động

Hàm `calculate_motion_ratio()` tính:

```text
số pixel trắng / tổng số pixel × 100
```

Nếu tỷ lệ lớn hơn `MOTION_RATIO`, cảnh được coi là đang chuyển động. Nếu cảnh đứng yên đủ lâu, chương trình tạo background.

## 6. Background và Foreground

### Background

Background là ảnh nền khi chưa có người trong khung hình.

Để tạo background:

1. Đứng ngoài khung hình.
2. Giữ camera và cảnh vật ổn định.
3. Chờ khoảng **10 giây nếu cảnh tĩnh liên tục**: 5 giây xác nhận cảnh tĩnh, rồi thêm 5 giây chờ lưu nền.

Hai khoảng chờ dùng chung `STATIC_WAIT_TIME = 5`, nhưng được đếm bằng hai biến `start_time` và `start_bg_time`. Trong code hiện tại, chuyển động đặt lại `start_time`, nhưng không đặt lại `start_bg_time`; vì vậy khi cảnh bị gián đoạn, không nên hiểu đây là bộ đếm yêu cầu đúng 10 giây tĩnh liên tục trong mọi trường hợp.

Chương trình chỉ kiểm tra độ thay đổi, không biết nền có người hay không. Nếu bạn ngồi yên trong lúc tạo nền, hình của bạn cũng có thể bị lưu vào nền.

Chương trình lưu ảnh hiện tại vào:

```python
background_blurred
```

### Foreground

Foreground là kết quả so sánh ảnh hiện tại với background:

```text
Ảnh hiện tại và ảnh nền đã làm mờ
→ chuyển sang int16 rồi trừ nhau
→ lấy trị tuyệt đối
→ pixel có chênh lệch > ngưỡng thành 255, còn lại thành 0
→ opening để loại nhiễu nhỏ
→ foreground_opened
```

Vùng trắng là phần đã thay đổi so với background, thường là người bước vào khung hình.

Phân biệt: `motion_mask` so hai frame liên tiếp để biết cảnh đang động; `foreground_opened` so ảnh hiện tại với nền đã lưu. Người đứng yên vẫn có thể nằm trong foreground nếu họ khác nền.

## 7. Bù sáng

Khi đèn bật hoặc tắt, toàn bộ ảnh có thể sáng hơn hoặc tối hơn. Code chỉ vào nhánh bù sáng khi `bg_ratio > LIGHT_RATIO`, hiện là hơn **70%** pixel khác nền sau lọc nhiễu. Đây là dấu hiệu nghi ngờ đổi ánh sáng, không phải bằng chứng chắc chắn.

Các phép tính trong nhánh đó (dùng tên biến của pipeline):

```python
diff = blurred.astype(np.int16) - background_blurred.astype(np.int16)
shift = int(np.median(diff))
```

Sau đó tạo background đã điều chỉnh sáng/tối:

```python
adjusted_bg = np.clip(
    background_blurred.astype(np.int16) + shift, 0, 255
).astype(np.uint8)
```

`int16` chứa được số âm khi trừ ảnh; `np.clip` giới hạn giá trị về 0–255. `median` là **trung vị**, không phải trung bình cộng: lấy giá trị ở giữa sau khi sắp xếp. `shift` dương làm nền sáng hơn, âm làm nền tối hơn.

Sau đó chương trình so ảnh hiện tại với `adjusted_bg`, áp ngưỡng và opening lại để tạo foreground dùng cho nhận diện. Bù sáng diễn ra ngay trong nhánh này, không phải đợi 30 giây.

Việc lưu nền mới là bước riêng: nhánh đổi sáng phải kéo dài đủ `LIGHT_WAIT_TIME` và `is_static` phải đúng. Bộ đếm bị đặt lại nếu cảnh đang động hoặc tỷ lệ khác nền không còn vượt ngưỡng. Khi đủ điều kiện, ảnh hiện tại `blurred` được lưu làm nền mới. Khung BACKGROUND hiển thị nền đã lưu, không phải nền bù sáng tạm thời.

Mục tiêu là giảm nhầm lẫn do đổi sáng đồng đều. Cách này không bảo đảm xử lý được bóng đổ, ánh sáng cục bộ hoặc ảnh bị cháy sáng.

## 8. Tìm vùng chuyển động

### `find_connected_components()`

Hàm này tìm các vùng pixel trắng liên thông và trả về bounding box:

```python
(x, y, width, height)
```

Thuật toán dùng các đoạn pixel trắng theo từng hàng và lớp `UnionFind` để gộp các đoạn liên thông.

### `merge_near_boxes()`

Nếu hai box nằm gần nhau hoặc chồng lên nhau, chúng được gộp lại. Khoảng cách mặc định là 40 pixel.

### `split_by_gap()`

Nếu một box lớn có khe đen ở giữa, hàm tách box đó thành các phần nhỏ hơn. Điều này giúp tách hai người đứng cạnh nhau.

## 9. Kiểm tra có phải người không

Hàm `is_person()` kiểm tra hai điều kiện:

1. Số pixel trắng **bên trong khung ứng viên**, chia cho tổng pixel **của toàn bộ ảnh**, phải lớn hơn `PERSON_RATIO = 2%`.
2. Vùng trắng có hình dáng đầu và vai.

Hàm `has_head_and_shoulders()` đếm độ rộng pixel trắng theo từng hàng:

Cụ thể là số pixel trắng trên mỗi hàng, rồi làm mượt bằng trung bình 5 hàng; không phải đo khoảng cách giữa hai pixel ngoài cùng.

```text
       đầu hẹp
       cổ nhỏ
      vai rộng
```

Nếu phần vai rộng hơn phần đầu theo `SHOULDER_RATIO`, vùng đó được giữ lại và vẽ khung xanh.

Đây là nhận diện theo hình dáng, chưa phải nhận diện người bằng AI nên có thể nhận sai trong một số trường hợp.

## 10. Các tham số có thể chỉnh

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `MOTION_THRESHOLD` | pipeline: 50; UI: 40 | Chênh lệch độ sáng phải lớn hơn ngưỡng để pixel thành trắng |
| `MOTION_RATIO` | pipeline: 2%; UI: 5% | Tỷ lệ pixel trắng phải lớn hơn ngưỡng để coi cảnh đang động |
| `STATIC_WAIT_TIME` | 5 giây | Dùng cho cả hai giai đoạn chờ tạo nền; tổng khoảng 10 giây khi tĩnh liên tục |
| `MIN_AREA` | 2000 pixel² | Diện tích hình chữ nhật w × h tối thiểu trước bước gộp/tách; không phải số pixel trắng |
| `LIGHT_RATIO` | 70 | Ngưỡng nghi ngờ ánh sáng thay đổi |
| `LIGHT_WAIT_TIME` | 30 giây | Thời gian chờ trước khi cập nhật nền |
| `SHOULDER_RATIO` | 1.4 | Tỷ lệ độ rộng vai so với đầu |
| `NECK_DROP` | 0.85 | Mức giảm độ rộng để nhận biết cổ |
| `PERSON_RATIO` | 2% | Pixel trắng trong khung / tổng pixel toàn ảnh phải lớn hơn 2% |

Các tham số nhận diện hình dáng nằm trong `detection/person.py`. Các tham số camera và background nằm trong `pipeline.py` và `ui.py`.

Hai bản hiện có ngưỡng khác nhau nên có thể cho kết quả khác nhau. Khi chạy `main.py`, các ngưỡng của `ui.py` được sử dụng.

## 11. Lỗi thường gặp



### Không tạo được background

- Đứng yên ngoài khung hình.
- Giữ camera cố định.
- Tăng `MOTION_RATIO` nếu cảnh bị coi là luôn chuyển động.

### Không có khung xanh

- Đảm bảo đã tạo background.
- Giảm `MIN_AREA` nếu người ở xa.
- Giảm `PERSON_RATIO` hoặc `SHOULDER_RATIO` nếu điều kiện hình dáng quá cao.

### Nhận nhầm vật thể thành người

- Tăng `PERSON_RATIO`.
- Tăng `SHOULDER_RATIO`.
- Đảm bảo vùng vật thể không có hình đầu và vai giống người.

## 12. Hạn chế của dự án

- Camera cần đứng yên.
- Người nên nhìn tương đối thẳng về camera.
- Ánh sáng thay đổi mạnh có thể làm sai background.
- Hai người đứng sát nhau có thể không tách được.
- Đây là nhận diện theo hình dáng, chưa phải mô hình học máy.

## 13. Từ điển tên biến bằng tiếng Việt

Tên có dấu gạch dưới là các từ ghép: `previous_frame` = khung hình trước. `self.ten_bien` là biến được lưu trong đối tượng giao diện để lần xử lý sau vẫn dùng được. `True` = đúng, `False` = sai, `None` = chưa có giá trị.

### Ảnh và xử lý ảnh

| Tên trong code | Nghĩa và chức năng |
|---|---|
| `camera` | Webcam đang được mở |
| `ret` | Đọc ảnh thành công hay không |
| `frame`, `current_frame` | Khung hình, khung hình hiện tại |
| `previous_frame` | Ảnh xám đã làm mờ của lần trước |
| `blue`, `green`, `red` | Kênh xanh dương, xanh lá, đỏ |
| `grayscale`, `grayscale_frame` | Ảnh xám: chỉ còn độ sáng |
| `blurred` | Ảnh xám sau làm mờ Gaussian |
| `image` | Ảnh đầu vào; trong hàm hiển thị là đối tượng ảnh Pillow |
| `KERNEL` | Ma trận trọng số của bộ lọc |
| `padded` | Ảnh được thêm viền để xử lý các pixel ở mép |
| `difference` | Trị tuyệt đối chênh lệch giữa hai ảnh |
| `diff` | Chênh lệch có dấu giữa ảnh hiện tại và nền khi bù sáng |
| `threshold_value` | Ngưỡng so sánh giá trị pixel |
| `motion_mask` | Mặt nạ đen/trắng từ hai frame liên tiếp |
| `opened` | Mặt nạ chuyển động đã qua opening |
| `foreground_mask` | Mặt nạ khác nền trước opening |
| `foreground_opened` | Mặt nạ khác nền sau opening, dùng tìm người |
| `background_blurred` | Ảnh nền đã làm mờ được lưu lại |
| `background_float` | Bản ảnh dạng số thực; hiện được tạo nhưng không được sử dụng tiếp |
| `adjusted_bg` | Nền tạm thời sau bù sáng |
| `shift` | Mức tăng hoặc giảm độ sáng lấy từ trung vị chênh lệch |
| `white_pixels`, `total_pixels` | Pixel trắng và tổng số pixel |
| `motion_ratio` | Phần trăm pixel trắng khi so hai frame liên tiếp |
| `bg_ratio` | Phần trăm pixel trắng khi so ảnh hiện tại với nền |
| `result` | Kết quả của hàm; có thể là ảnh hoặc danh sách tùy hàm |

### Thời gian và trạng thái

| Tên trong code | Nghĩa và chức năng |
|---|---|
| `current_time` | Thời điểm hiện tại, tính bằng giây |
| `start_time` | Mốc bắt đầu chờ để xác nhận cảnh tĩnh |
| `start_bg_time` | Mốc chờ lưu nền sau khi xác nhận tĩnh |
| `light_start_time` | Mốc chờ cập nhật nền khi đổi sáng |
| `is_static` | Cảnh đã được xác nhận tĩnh hay chưa |
| `background_ready` | Đã có ảnh nền hay chưa |
| `running` | Giao diện có đang chạy camera không |
| `after_id` | Mã lịch gọi lại hàm xử lý, dùng hủy lịch khi STOP |
| `key` | Mã phím bấm trong bản OpenCV |

### Vùng trắng và nhận diện đầu–vai

| Tên trong code | Nghĩa và chức năng |
|---|---|
| `components` | Các khung bao vùng pixel trắng liên thông |
| `boxes`, `box` | Danh sách khung và một khung hình chữ nhật |
| `x`, `y` | Tọa độ góc trên trái; x tăng sang phải, y tăng xuống dưới |
| `w`, `h`, `width`, `height` | Chiều rộng và chiều cao |
| `people` | Các khung đạt luật nhận diện; không phải danh tính người |
| `gap` | Khoảng cách dùng xét gộp hai khung |
| `crop`, `sub` | Phần ảnh được cắt trong khung, phần nhỏ hơn của ảnh cắt |
| `has_white` | Mỗi cột có nhiều hơn EMPTY_MAX pixel trắng hay không |
| `pieces` | Các đoạn được tách theo khe trống |
| `start`, `last`, `empty_run` | Cột bắt đầu, cột có nội dung cuối cùng, số cột trống liên tiếp |
| `c0`, `c1`, `pw` | Cột đầu, cột cuối và chiều rộng một mảnh |
| `rows` | Trong hàm tách: chỉ số hàng có trắng; trong hàm đầu–vai: số pixel trắng mỗi hàng đã làm mượt |
| `box_w`, `head_w`, `shoulder_w` | Chiều rộng khung, lượng trắng đại diện độ rộng đầu, lượng trắng lớn nhất từ vùng cổ trở xuống |
| `neck_row` | Hàng được coi là cổ khi lượng trắng giảm đủ so với đầu |
| `ok_shape` | Hình dạng có đạt điều kiện đầu–vai không |
| `shoulder` | Tỷ lệ vai/đầu trả về để in kiểm tra, không phải độ rộng vai |
| `box_mask` | Ảnh đen bằng kích thước toàn ảnh, chỉ giữ pixel trong khung đang xét |
| `ratio` | Tùy hàm: phần trăm pixel trắng hoặc tỷ lệ vai/đầu |
| `frame_shape` | Kích thước ảnh truyền vào is_person; hiện tham số này chưa được sử dụng trong hàm |

### Tìm vùng liên thông và giao diện

| Tên trong code | Nghĩa và chức năng |
|---|---|
| `runs` | Các đoạn pixel trắng liên tiếp trên từng hàng |
| `splits`, `groups` | Vị trí tách đoạn và các nhóm đoạn/pixel |
| `uf` | Đối tượng UnionFind dùng gộp các đoạn cùng vùng |
| `parent` | Trong UnionFind: danh sách nhãn cha; trong giao diện: thành phần chứa thành phần con |
| `label` | Trong thuật toán: nhãn số của vùng; trong giao diện: ô hiển thị chữ/ảnh |
| `root` | Trong UnionFind: nhãn gốc; trong Tkinter: cửa sổ chính |
| `panel`, `content`, `controls` | Khung hiển thị, vùng nội dung, vùng nút điều khiển |
| `camera_label`, `foreground_label`, `background_label` | Ô hiển thị camera, mặt nạ khác nền, nền đã lưu |
| `start_button`, `stop_button` | Nút bắt đầu, nút dừng |
| `status`, `title`, `placeholder` | Trạng thái, tiêu đề, chữ tạm khi chưa có ảnh |
| `photo` | Ảnh đã chuyển sang định dạng Tkinter; giữ tham chiếu để ảnh không bị thu hồi |
| `color`, `size` | Có phải ảnh màu không và kích thước hiển thị tối đa |

Ví dụ: `crop = mask[y:y + h, x:x + w]` nghĩa là lấy phần mặt nạ từ hàng y đến trước y+h, từ cột x đến trước x+w. Thứ tự truy cập ảnh là **hàng trước, cột sau**.
