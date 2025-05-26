# human-tracking
Human tracking

## Frame exporter

Không phải lúc nào các model cũng có thể nhận diện được vậy thể mà hệ thống mong muốn, nhất là cac vật thể có tính đặc thù chuyên biệt, chưa có model sẵn có.
Frame exporter được sử dụng để export các frame trong video sample để sử dụng làm dữ liệu cho quá tình data label nhằm thực hiện nhận diện vật thể một các nhanh chóng và không tốn quá nhiều resource/tài nguyên cho quá trình tìm kiếm data.

```shell
pytohn frameexporter.py --video_path=<video_path>
```

## Model converter

Thực hiện convert định dạng model PT thành các định dạng model khác
- openvino (tối ưu cho CHIP Intel)
- onnx (có thể sử dụng trên các thiết bị nhúng, thiết bị di động)
- coreml (tối ưu cho CHIP Apple)

```shell
pytohn converter.py --model=<model>
```

## Thiết lập vùng đếm dữ liệu

Hỗ trợ GUI thiết lập vùng đếm dữ liệu, khi đối tượng đi vào (INSIDE) và đi ra (OUTSIDE) hoặc đi qua (OUTSIDE - INSIDE - OUTSIDE) vùng đếm thì sẽ thực hiện đếm đối tượng.
Click trên màn hình (pause video bằng space nếu cần) để tạo vùng đếm dữ liệu.

```shell
pytohn converter.py --action=config
```

## Nhận diện, theo dõi và đếm đối tượng

```shell
pytohn converter.py 
```

Note:
- File name đang hard code là file_name, có thể chỉnh sửa file name hoặc chọn là camera nếu muốn.
```python
file_name = "sample1.mp4"
videoCap = cv2.VideoCapture(file_name)  # Đọc video từ file
# videoCap = cv2.VideoCapture(0)  # Đọc video từ webcam nếu cần
```

- Model đang hard code là yolo11s, có thể chỉnh sửa model name tùy ý.
```python
# Khởi tạo mô hình YOLO với model được huấn luyện trước (YOLOv11 nano)
model_name = "yolo11s"
```

- Có thể size video để giảm khối lượng xử lý của model, giúp tăng FPS.
```shell
pytohn converter.py  --height=720
```

- Có thể lưu lại video quá trình nhận diện, ttheo dõi và đếm
```shell
pytohn converter.py --save
```

- Các option khác có thêm xem trong ArgumentParser.
