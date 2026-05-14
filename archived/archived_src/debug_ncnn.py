import cv2
import numpy as np
import ncnn
from picamera2 import Picamera2
import time

# Load NCNN model
net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.load_param("yolov8n_ncnn_model/model.ncnn.param")
net.load_model("yolov8n_ncnn_model/model.ncnn.bin")

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()
time.sleep(1)

print("Taking a picture...")
frame = picam2.capture_array()
h, w = frame.shape[:2]

# Preprocess
img = cv2.resize(frame, (320, 320))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create NCNN mat
mat_in = ncnn.Mat.from_pixels(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 320, 320)
mat_in.substract_mean_normalize([0, 0, 0], [1/255, 1/255, 1/255])

# Run inference
ex = net.create_extractor()
ex.input("in0", mat_in)

ret, mat_out = ex.extract("out0")

# The output needs to be reshaped properly
# Expected shape: (1, 84, 2100) or (1, 2100, 84)
data = np.array(mat_out)
print(f"Raw output shape: {data.shape}")

# Reshape to expected YOLO output format
# YOLO output is typically [batch, 84, num_detections] or [batch, num_detections, 84]
if len(data.shape) == 2:
    # Try to reshape to (84, 2100) -> (1, 84, 2100)
    if data.shape[0] == 84 and data.shape[1] == 2100:
        data = data.reshape(1, 84, 2100)
        data = data.transpose(0, 2, 1)  # Now (1, 2100, 84)
        print(f"Reshaped to: {data.shape}")

# Now data should be (1, num_detections, 84)
detections = data[0]  # (num_detections, 84)

print(f"\nNumber of detections: {detections.shape[0]}")
print("=" * 60)

bottle_count = 0
for i in range(detections.shape[0]):
    # Get object confidence (should be sigmoid activated)
    obj_conf = detections[i][4]
    
    # If confidence seems raw (large number), apply sigmoid
    if obj_conf > 1:
        obj_conf = 1 / (1 + np.exp(-obj_conf))
    
    if obj_conf > 0.3:
        # Get class scores
        class_scores = detections[i][5:]
        
        # Apply sigmoid to class scores if they're large
        if np.max(class_scores) > 10:
            class_scores = 1 / (1 + np.exp(-class_scores))
        
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        
        print(f"\nDetection {i}:")
        print(f"  Objectness: {obj_conf:.3f}")
        print(f"  Class ID: {class_id}")
        print(f"  Class Confidence: {class_conf:.3f}")
        
        if class_id == 39 and class_conf > 0.3:
            bottle_count += 1
            x1 = detections[i][0]
            y1 = detections[i][1]
            x2 = detections[i][2]
            y2 = detections[i][3]
            
            # Scale coordinates (they should already be normalized 0-1)
            if x1 > 1:  # If raw coordinates, normalize
                x1 = x1 / 320
                y1 = y1 / 320
                x2 = x2 / 320
                y2 = y2 / 320
            
            x1_scaled = int(x1 * w)
            y1_scaled = int(y1 * h)
            x2_scaled = int(x2 * w)
            y2_scaled = int(y2 * h)
            
            print(f"  ✅ BOTTLE DETECTED!")
            print(f"  Box: ({x1_scaled},{y1_scaled}) to ({x2_scaled},{y2_scaled})")
            
            # Draw on frame
            cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
            cv2.putText(frame, f"BOTTLE {obj_conf:.2f}", (x1_scaled, y1_scaled - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"\n{'='*60}")
print(f"Total bottles detected: {bottle_count}")

if bottle_count == 0:
    print("\n⚠️ NO BOTTLES DETECTED")
    print("The model is detecting other classes but not bottle (class 39)")
    print(f"Classes detected: check above for class IDs")

# Save image
cv2.imwrite("debug_output.jpg", frame)
print("\nImage saved as debug_output.jpg")

picam2.close()