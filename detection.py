import os
import cv2
import json
import time
import math
import datetime
import argparse
import numpy as np
from ultralytics import YOLO
from collections import Counter
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips

# Khởi tạo mô hình YOLO với model được huấn luyện trước (YOLOv11 nano)
model_name = "yolo11s"

# Xác định nhãn đối tượng cần detect (ví dụ: 0 có thể là "person" nếu dùng COCO dataset)
# target_labels = [0]
target_labels = None

# Khởi tạo dictionary để lưu đường đi của các đối tượng
object_paths = {}

# Khởi tạo dictionary để lưu vị trí của các đối tượng theo từng frame
object_positions = {}

# Mở video từ file hoặc camera (chọn file video hoặc webcam)
file_name = "sample1.mp4"
videoCap = cv2.VideoCapture(file_name)  # Đọc video từ file
# videoCap = cv2.VideoCapture(0)  # Đọc video từ webcam nếu cần

# Cấu hình buffer để giảm độ trễ khi đọc video (giúp tăng FPS)
videoCap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# Bật tối ưu hóa OpenCV để tăng tốc xử lý hình ảnh
cv2.setUseOptimized(True)

# Kiểm tra xem OpenCV có hỗ trợ OpenCL không (nếu có thì bật lên để tận dụng GPU)
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)

def relative_position_analysis(point, points):
    """
    Xác định vị trí của điểm so với đa giác.

    Tham số:
    - point: Tuple (x, y), tọa độ của điểm cần kiểm tra.
    - points: Mảng numpy chứa tập hợp các điểm của đa giác.

    Trả về:
    - Chuỗi 'INSIDE' nếu điểm nằm bên trong đa giác.
    - Chuỗi 'OUTSIDE' nếu điểm nằm ngoài đa giác.
    """
    if cv2.pointPolygonTest(points, point, False) > 0:
        status = "INSIDE"
    else:
        status = "OUTSIDE"
    return status

def standardize_relative_position(object_positions):
    """
    Chuẩn hóa vị trí tương đối của các đối tượng trong khu vực quan tâm.

    Tham số:
    - object_positions: Dictionary chứa vị trí của các đối tượng, với key là ID và value là danh sách vị trí.

    Trả về:
    - Counter chứa số lượng các đối tượng ở vị trí cuối cùng, cùng với số lượng đối tượng đi qua khu vực quan tâm.
    """
    
    # Lọc ra các đối tượng có ít nhất 2 vị trí và có số lượng vị trí là số chẵn
    filtered_items = {k: v for k, v in object_positions.items() if len(v) >= 2 and len(v) % 2 == 0}

    # Lấy giá trị cuối cùng của mỗi đối tượng sau khi lọc
    last_values = [v[-1] for v in filtered_items.values()]
    
    # Đếm số lượng đối tượng dựa trên giá trị cuối cùng
    counted = Counter(last_values)

    # Đếm số lượng đối tượng chỉ đi qua khu vực quan tâm (số lẻ vị trí và có nhiều hơn 2 vị trí)
    through = sum(1 for key, value in object_positions.items() if len(value) > 2 and len(value) % 2 != 0)

    # Nếu có đối tượng đi qua, thêm thông tin này vào Counter
    if through > 0:
        counted['THROUGH'] = through

    return counted



def resize(frame, target_height):
    """
    Resize an image frame while maintaining its aspect ratio.

    Parameters:
    - frame (numpy.ndarray): The input image frame.
    - target_height (int): The desired height of the resized frame.

    Returns:
    - aspect_ratio (float): The aspect ratio (width/height) of the original frame.
    - resized_frame (numpy.ndarray): The resized frame maintaining the original aspect ratio.

    Processing:
    1. Extract the original width and height from the frame.
    2. Calculate the aspect ratio of the original frame.
    3. Compute the new width based on the aspect ratio and target height.
    4. Resize the frame using OpenCV while preserving aspect ratio.
    5. Return the aspect ratio and resized frame.
    """

    if target_height is None:
        return 1, frame
    original_height = frame.shape[0]  # Get original height
    original_width = frame.shape[1]   # Get original width
    aspect_ratio = original_width / original_height  # Calculate aspect ratio
    target_width = int(target_height * aspect_ratio)  # Calculate new width

    return aspect_ratio, cv2.resize(frame, (target_width, target_height))


def tracking(args, model):
    """
    Theo dõi và xác định trạng thái của các đối tượng trong video bằng YOLO.

    Mô tả:
    - Đọc video từ file hoặc webcam.
    - Tải cấu hình khu vực quan tâm (ROI) từ file `config.json`.
    - Theo dõi đối tượng được chỉ định (`target_labels`) bằng mô hình YOLO.
    - Tính toán FPS và hiển thị thông tin trên màn hình.
    - Xác định đối tượng nằm trong hoặc ngoài khu vực ROI.
    - Vẽ bounding box, đường di chuyển của đối tượng, và hiển thị thống kê.
    - Dừng quá trình khi nhấn phím 'q'.

    Tham số:
    - args (argparse.Namespace): Đối tượng chứa các tham số dòng lệnh từ argparse.
      - args.debug (bool): Nếu True, bật chế độ debug để hiển thị đối tượng.
      - args.device (string): Thiết bị được sử dụng cho quá trình nhận diện.
    - model: Đối tượng YOLO Model

    Giá trị trả về:
    - Không có giá trị trả về trực tiếp. Hiển thị kết quả theo dõi trên màn hình.

    Phụ thuộc:
    - `cv2`: OpenCV để xử lý hình ảnh và hiển thị video.
    - `numpy`: Xử lý ma trận và điểm khu vực.
    - `time`: Tính toán thời gian giữa các khung hình để đo FPS.
    - `json`: Đọc file cấu hình cho khu vực theo dõi.
    - `Counter`: Đếm số lượng đối tượng INSIDE/OUTSIDE.

    Ví dụ:
    >>> tracking()  # Chạy quá trình theo dõi đối tượng trong video

    Ghi chú:
    - Cần đảm bảo video hoặc camera được mở trước khi chạy hàm.
    - Nếu `args.debug` được bật, sẽ hiển thị khu vực ROI và đường đi của đối tượng.
    """

    # Lưu thời gian của khung hình trước để tính FPS
    prev_time = time.time()

    # Đọc file cấu hình chứa các điểm khu vực quan tâm (ROI)
    with open("config.json", "r") as file:
        config = json.load(file)

    # Lấy danh sách các điểm từ file config và chuyển thành dạng ma trận NumPy
    points = np.array([config['p1'], config['p2'],
                      config['p3'], config['p4']], np.int32)
    points = points.reshape((-1, 1, 2))
    out = None
    frame_count = 0

    while True:
        # Đọc frame từ video
        ret, frame = videoCap.read()
        if not ret:
            break
        ratio, frame = resize(frame, args.height)

        # Tính FPS bằng cách đo khoảng thời gian giữa hai khung hình
        new_time = time.time()
        fps = 1 / (new_time - prev_time)
        prev_time = new_time

        # Hiển thị FPS ở góc phải trên của màn hình
        cv2.putText(frame, f"FPS: {int(fps)}", (
            frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Sử dụng YOLO để theo dõi đối tượng được chỉ định
        if args.device is None:
            results = model.track(frame, persist=True,
                                  verbose=args.verbose, classes=target_labels)
        else:
            results = model.track(frame, persist=True, verbose=args.verbose, classes=target_labels, device=args.device)

        # Nếu chế độ debug được bật, tô màu khu vực ROI lên frame
        if args.debug:
            if args.save:
                cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                overlay = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillPoly(overlay, [points], (0, 0, 255,
                            128))  # Tô màu đỏ có độ mờ 50%

                # Áp dụng lớp phủ lên frame hiện tại bằng cách sử dụng alpha blending
                alpha = overlay[:, :, 3] / 255.0
                for c in range(3):
                    frame[:, :, c] = (1 - alpha) * frame[:, :, c] + \
                        alpha * overlay[:, :, c]

        # Duyệt qua kết quả từ YOLO để xử lý từng đối tượng phát hiện được
        for r in results:
            for box in r.boxes:
                if box is None or box.id is None:
                    continue

                # Lấy tọa độ của bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                obj_id = int(box.id[0])

                # Tính vị trí trung tâm của đối tượng
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                point = (center_x, center_y)

                # Kiểm tra xem đối tượng nằm trong khu vực quan tâm hay không
                status = relative_position_analysis(point, points)

                # Cập nhật trạng thái của đối tượng
                if obj_id in object_positions:
                    if object_positions[obj_id][-1] != status:
                        object_positions[obj_id].append(status)
                else:
                    object_positions[obj_id] = [status]

                # Lưu lịch sử đường đi của đối tượng
                if obj_id not in object_paths:
                    object_paths[obj_id] = []
                object_paths[obj_id].append((x1, y1))

                # Nếu chế độ debug được bật, vẽ bounding box và đường đi của đối tượng
                if args.debug:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y),
                               10, (255, 0, 255), -1)
                    cv2.putText(frame, f"{label}: {box.id[0]} [{status}]", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Vẽ đường nối giữa các vị trí trước đó của đối tượng
                    for i in range(1, len(object_paths[obj_id])):
                        cv2.line(
                            frame, object_paths[obj_id][i-1], object_paths[obj_id][i], (0, 0, 255), 2)
    
        counted = standardize_relative_position(object_positions)

        if len(counted) > 0:
            # Tạo khung hiển thị số lượng đối tượng INSIDE/OUTSIDE ở góc trên cùng
            top_left = (0, 0)
            width, height = 300, 10 * 2 + len(counted) * 50
            points2 = np.array([
                top_left,
                (top_left[0] + width, top_left[1]),
                (top_left[0] + width, top_left[1] + height),
                (top_left[0], top_left[1] + height)
            ], np.int32)
            points2 = points2.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [points2], (0, 0, 0))

            # Hiển thị số lượng đối tượng INSIDE/OUTSIDE
            for index, (key, value) in enumerate(counted.items()):
                cv2.putText(frame, f"{key}: {value}", (50, 50 + index * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_count += 1  # Tăng biến đếm số lượng frame đã xử lý
        if args.save:  # Kiểm tra xem có yêu cầu lưu video hay không
            if out is None:  # Nếu chưa khởi tạo VideoWriter, tiến hành khởi tạo
                fps = videoCap.get(cv2.CAP_PROP_FPS)  # Lấy số khung hình trên giây (FPS) từ video gốc
                original_height = frame.shape[0]  # Lấy chiều cao của frame
                original_width = frame.shape[1]  # Lấy chiều rộng của frame
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Xác định codec để ghi video (MP4)
                # Khởi tạo VideoWriter để lưu video đầu ra với định dạng, FPS và kích thước gốc
                out = cv2.VideoWriter(f"output_{file_name}", fourcc, math.ceil(fps), (original_width, original_height))
            out.write(frame)  # Ghi frame đã xử lý vào video đầu ra
        # Hiển thị video với các đối tượng được theo dõi
        cv2.imshow("YOLOv11s Human Tracking", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Giải phóng tài nguyên và đóng cửa sổ hiển thị
    if args.save:
        print(f"Save image to: output_{file_name}")
        out.release()
        video_clip = VideoFileClip(f"output_{file_name}")  # Đọc video đầu ra đã tạo
        audio_clip = AudioFileClip(file_name)  # Đọc file âm thanh cần ghép vào video
        final_clip = video_clip.with_audio(audio_clip)  # Ghép âm thanh vào video
        final_clip.write_videofile(f"final_{file_name}", codec='libx264', audio_codec='aac')  # Xuất video có âm thanh
    videoCap.release()
    cv2.destroyAllWindows()


def select_area(args):
    """
    Chọn vùng giám sát bằng cách click chuột trên video.

    Tham số:
    - args (argparse.Namespace): Đối tượng chứa tham số dòng lệnh từ argparse.
      - args.debug (bool): Hiện tại không được sử dụng trong hàm này nhưng có thể mở rộng trong tương lai.

    Mô tả:
    - Người dùng click **4 điểm** trên video để xác định khu vực quan tâm.
    - Khi đủ 4 điểm, tọa độ được lưu vào file `config.json`.
    - Hiển thị FPS để giám sát hiệu suất.
    - Nhấn 'q' để thoát.

    Giá trị trả về:
    - Không có giá trị trả về trực tiếp. Hiển thị video với khả năng tương tác.

    Ghi chú:
    - Nếu muốn điều chỉnh khu vực giám sát sau khi đã chọn, xóa `config.json` và chạy lại hàm.
    - File `config.json` có thể được sử dụng trong quá trình theo dõi đối tượng.
    """

    prev_time = time.time()  # Lưu thời gian của khung hình trước để tính FPS

    global clicked_points  # Danh sách lưu tọa độ các điểm đã click
    global last_point  # Lưu vị trí hiện tại của chuột để vẽ đường nối liên tục

    clicked_points = []  # Khởi tạo danh sách điểm đã click
    last_point = None  # Khởi tạo điểm cuối cùng là None

    def save_config():
        """
        Hàm lưu 4 điểm đã chọn vào file config.json.
        - Nếu file config đã tồn tại, tạo bản sao dự phòng trước khi ghi mới.
        - Ghi dữ liệu vào file JSON dưới dạng p1, p2, p3, p4.
        """
        file_name = "config.json"
        if os.path.exists(file_name):
            # Tạo bản sao dự phòng của file nếu đã tồn tại
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_name}.{timestamp}"
            os.rename(file_name, backup_name)

        # Ghi dữ liệu tọa độ vào file JSON
        with open(file_name, "w") as file:
            data = {
                'p1': clicked_points[0],
                'p2': clicked_points[1],
                'p3': clicked_points[2],
                'p4': clicked_points[3],
            }
            json.dump(data, file, indent=4)

    def mouse_event(event, x, y, flags, param):
        """
        Hàm xử lý sự kiện chuột:
        - Click chuột trái: Lưu tọa độ điểm vào danh sách clicked_points.
        - Nếu đã chọn đủ 4 điểm, lưu vào file config và thoát chương trình.
        - Khi di chuyển chuột, lưu vị trí hiện tại vào last_point để vẽ đường nối.
        """
        global last_point
        if event == cv2.EVENT_LBUTTONDOWN:  # Khi nhấn chuột trái
            clicked_points.append((x, y))  # Lưu tọa độ điểm click
            if len(clicked_points) >= 4:  # Nếu đủ 4 điểm thì lưu file config và thoát
                save_config()
                exit()
        elif event == cv2.EVENT_MOUSEMOVE:  # Khi di chuyển chuột
            last_point = (x, y)  # Cập nhật vị trí hiện tại của chuột
            tmpFrame = frame.copy()
            cv2.line(tmpFrame, clicked_points[-1], last_point, (0, 255, 0), 2)
            cv2.imshow("Video", tmpFrame)

    paused = False  # Biến kiểm soát trạng thái tạm dừng video
    cv2.namedWindow("Video")  # Tạo cửa sổ video
    # Gán hàm xử lý chuột cho cửa sổ video
    cv2.setMouseCallback("Video", mouse_event)

    while videoCap.isOpened():
        if not paused:
            ret, frame = videoCap.read()  # Đọc frame từ video
            if not ret:
                break
        ratio, frame = resize(frame, args.height)

        # Tính FPS bằng cách đo khoảng thời gian giữa hai khung hình
        new_time = time.time()
        fps = 1 / (new_time - prev_time)
        prev_time = new_time

        # Hiển thị FPS ở góc phải trên của màn hình
        cv2.putText(frame, f"FPS: {int(fps)}", (
            frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Nếu đã có ít nhất một điểm click, vẽ đường nối giữa các điểm
        if len(clicked_points) > 1:
            # Vẽ đường nối giữa các điểm đã click
            for i in range(len(clicked_points) - 1):
                cv2.line(frame, clicked_points[i], clicked_points[i + 1], (0, 255, 0), 2)

        # Hiển thị video với vùng giám sát được xác định
        if not paused:
            cv2.imshow("Video", frame)

        # Xử lý phím bấm
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):  # Nhấn 'q' để thoát
            break
        elif key == ord(" "):  # Nhấn SPACE để tạm dừng hoặc tiếp tục
            paused = not paused

    # Giải phóng tài nguyên và đóng cửa sổ hiển thị
    videoCap.release()
    cv2.destroyAllWindows()


def main():
    """
    Chương trình chính: Xử lý tham số dòng lệnh và gọi chức năng phù hợp.

    Tham số:
    - Không có tham số đầu vào trực tiếp (hàm lấy tham số từ dòng lệnh).

    Mô tả:
    - Nhận tham số từ người dùng (`tracking` hoặc `config`).
    - Nếu `--action config`, chạy `select_area(args)` để chọn vùng giám sát.
    - Nếu `--action tracking`, chạy `tracking(args)` để theo dõi đối tượng.

    Giá trị trả về:
    - Không có giá trị trả về. Chạy chương trình theo yêu cầu của người dùng.

    Ghi chú:
    - Chạy lệnh với `python script.py --action config` để chọn vùng.
    - Chạy lệnh với `python script.py --action tracking` để theo dõi đối tượng.
    - Chế độ `--debug` giúp hiển thị bounding box và đường đi của đối tượng.
    """
    parser = argparse.ArgumentParser(description="Tracking human")
    parser.add_argument(
        "--action", help="Action (tracking or config)", default='tracking')
    parser.add_argument("--debug", help="Debug mode",
                        action="store_true", default=True)
    parser.add_argument(
        "--height", help="Height of frame after resize", type=int)
    parser.add_argument("--model", help="Model: pt or onnx or openvino")
    parser.add_argument("--save", help="Save video tracking",
                        action="store_true", default=False)
    parser.add_argument("--verbose", help="Yolo verbose tracking",
                        action="store_true", default=False)
    parser.add_argument(
        "--device", help="Model: 0 (NVIDAI GPU), cpu, gpu, npu, mps, intel:gpu, intel:npu, intel:cpu")

    args = parser.parse_args()

    if args.action == 'config':
        select_area(args)
    else:
        if args.model == 'onnx':
            model = YOLO(f"{model_name}.onnx")
        if args.model == 'coreml':
            model = YOLO(f"{model_name}.mlpackage")
        if args.model == 'openvino':
            model = YOLO(f"{model_name}_openvino_model/")
        else:
            model = YOLO(f"{model_name}.pt")
        tracking(args, model)


if __name__ == "__main__":
    main()
