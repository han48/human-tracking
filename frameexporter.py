import cv2
import argparse
import os

# Khởi tạo argparse để nhận video_path từ dòng lệnh
parser = argparse.ArgumentParser(description="Trích xuất ảnh từ video")
parser.add_argument("video_path", type=str, help="Đường dẫn đến video")
args = parser.parse_args()

# Lấy tên file từ đường dẫn video (không bao gồm phần mở rộng)
video_name = os.path.splitext(os.path.basename(args.video_path))[0]

# Mở video
cap = cv2.VideoCapture(args.video_path)

frame_count = 0
save_every_n_frames = 15  # Cứ 15 frame lưu một ảnh

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Dừng nếu không còn frame nào

    if frame_count % save_every_n_frames == 0:
        image_path = f"frames/{video_name}_frame_{frame_count}.jpg"
        cv2.imwrite(image_path, frame)
        print(f"Đã lưu {image_path}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
