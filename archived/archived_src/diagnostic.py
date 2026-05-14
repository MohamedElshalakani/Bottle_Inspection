import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()
time.sleep(1)

session = ort.InferenceSession("yolov8n.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

print("Running diagnostics for 30 frames...\n")

capture_times = []
preprocess_times = []
inference_times = []
total_times = []

for i in range(30):
    total_start = time.time()
    
    # Capture
    cap_start = time.time()
    frame = picam2.capture_array()
    capture_times.append(time.time() - cap_start)
    
    # Preprocess
    pre_start = time.time()
    img = cv2.resize(frame, (320, 320))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    preprocess_times.append(time.time() - pre_start)
    
    # Inference
    inf_start = time.time()
    _ = session.run(None, {input_name: img})
    inference_times.append(time.time() - inf_start)
    
    total_times.append(time.time() - total_start)
    
    if i % 10 == 0:
        print(f"Frame {i}: Total={total_times[-1]*1000:.1f}ms | "
              f"Capture={capture_times[-1]*1000:.1f}ms | "
              f"Inference={inference_times[-1]*1000:.1f}ms")

print(f"\n--- AVERAGES (30 frames) ---")
print(f"Capture time:    {sum(capture_times)/len(capture_times)*1000:.1f} ms")
print(f"Preprocess time: {sum(preprocess_times)/len(preprocess_times)*1000:.1f} ms")
print(f"Inference time:  {sum(inference_times)/len(inference_times)*1000:.1f} ms")
print(f"TOTAL per frame: {sum(total_times)/len(total_times)*1000:.1f} ms")
print(f"\nMax FPS possible: {1000 / (sum(total_times)/len(total_times)*1000):.1f}")

picam2.close()