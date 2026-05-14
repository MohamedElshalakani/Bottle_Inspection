from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == ord('q'):
        break