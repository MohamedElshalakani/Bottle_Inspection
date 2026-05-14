from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=11, nms = False)
print("Model exported successfully to ONNX format.")