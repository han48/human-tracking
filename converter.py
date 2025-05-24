import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Load YOLO model")
parser.add_argument("--model", help="Model: pt or onnx or openvino", default="yolo11n")
args = parser.parse_args()

# Load the YOLO11 model
model = YOLO(f"{args.model}.pt")

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

# Export the model to CoreML format
model.export(format="coreml")  # creates 'yolo11n.mlpackage'