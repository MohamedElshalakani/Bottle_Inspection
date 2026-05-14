import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

# Load model
session = ort.InferenceSession("yolov8n.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()
time.sleep(1)

frame = picam2.capture_array()
h, w = frame.shape[:2]

# Preprocess
img = cv2.resize(frame, (320, 320))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# Run inference
outputs = session.run(None, {input_name: img})

output = outputs[0]
print(f"Output shape: {output.shape}")

# Transpose from (1, 84, 2100) to (1, 2100, 84)
if output.shape[1] == 84 and output.shape[2] == 2100:
    output = output.transpose(0, 2, 1)
    print(f"Transposed shape: {output.shape}")

detections = output[0]  # Shape: (2100, 84)
print(f"Number of detections: {len(detections)}")
print("=" * 50)

bottle_detections = []

for i in range(len(detections)):
    detection = detections[i]
    obj_conf = detection[4]
    
    if obj_conf > 0.3:
        # Get class scores (positions 5 to 84)
        class_scores = detection[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        
        # Combined confidence
        total_conf = obj_conf * class_conf
        
        if class_id == 39 and total_conf > 0.3:
            x1, y1, x2, y2 = detection[:4]
            
            # Scale to original frame
            x1 = int(x1 * w / 320)
            y1 = int(y1 * h / 320)
            x2 = int(x2 * w / 320)
            y2 = int(y2 * h / 320)
            
            bottle_detections.append((x1, y1, x2, y2, total_conf))
            print(f"✅ BOTTLE {len(bottle_detections)}:")
            print(f"   Box: ({x1},{y1}) to ({x2},{y2})")
            print(f"   Confidence: {total_conf:.3f}")
            print("-" * 30)

if len(bottle_detections) == 0:
    print("❌ NO BOTTLES DETECTED")
    print(f"   Highest object confidence: {max([detections[i][4] for i in range(len(detections))]) if len(detections) > 0 else 0:.3f}")
else:
    print(f"\n✅ Found {len(bottle_detections)} bottle(s)!")

picam2.close()
print("\nDiagnostic complete.")