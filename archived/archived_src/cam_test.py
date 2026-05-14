from picamera2 import Picamera2
import time
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

time.sleep(1)

frame = picam2.capture_array()

print("captured frame shape:", frame.shape)
cv2.imwrite("test.jpg", frame)

print("saved image as test.jpg")

