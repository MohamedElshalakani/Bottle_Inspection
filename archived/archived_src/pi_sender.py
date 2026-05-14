from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import signal
import sys
import time

RELAY_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

RELAY_ON = 1
RELAY_OFF = 0
o
GPIO.output(RELAY_PIN, RELAY_OFF)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

PC_IP = "192.168.1.7"
PORT = 5000

s = socket.socket()
s.connect((PC_IP, PORT))
s.settimeout(1.0)

good_count = 0
no_cap_count = 0
no_label_count = 0
relay_off_time = None
last_frame = None

latest_id = "-"
latest_class = "waiting"
last_processed_id = -1

running = True

def stop(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

def send_frame():
    global last_frame
    last_frame = picam2.capture_array()
    _, buffer = cv2.imencode(".jpg", last_frame)
    data = buffer.tobytes()
    size = str(len(data)).ljust(16).encode()
    s.sendall(size)
    s.sendall(data)

while running:
    if relay_off_time and time.time() >= relay_off_time:
        GPIO.output(RELAY_PIN, RELAY_OFF)
        relay_off_time = None

    try:
        cmd = s.recv(16).decode().strip()
    except Exception as e:
        cmd = None

    if cmd == "CAPTURE":
        send_frame()

    elif cmd and cmd.startswith("ID:"):

        try:
            parts = cmd.split("|")
            latest_id = int(parts[0].split(":")[1])
            latest_class = parts[1]
        except Exception as e:
            print(f"Error parsing command: {e}")
            continue

        if latest_id != last_processed_id:

            last_processed_id = latest_id

            if latest_class == "Good":
                good_count += 1
                GPIO.output(RELAY_PIN, RELAY_OFF)

            elif latest_class in ["No_cap", "No_label"]:
                if latest_class == "No_cap": no_cap_count += 1
                else: no_label_count += 1
                GPIO.output(RELAY_PIN, RELAY_ON)
                relay_off_time = time.time() + 0.2
    frame = last_frame if last_frame is not None else picam2.capture_array()
    if frame is None:
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
    h, w, _ = frame.shape

    rx1, ry1, rx2, ry2 = 120, 80, 520, 400
   
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
    
    cv2.putText(frame, "ROI / INSPECTION ZONE", (rx1, ry1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(frame, "MYK AUTOMATION", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(frame, (10, h-120), (320, h-10), (255, 255, 255), -1)

    cv2.putText(frame, f"Good: {good_count}", (20, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"No Cap: {no_cap_count}", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"No Label: {no_label_count}", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 254), 2)

    status = f"Current: {latest_id} | {latest_class}"

    cv2.rectangle(frame, (w-300, h-60), (w-10, h-10), (255,255,255), -1)

    cv2.putText(frame, status, (w - 290, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    try:
        cv2.imshow("Bottle Inspection", frame)
    except Exception as e:
        print(f"Display error: {e}")
    try:
      if cv2.waitKey(1) == 27:
         running = False

      if cv2.getWindowProperty("Bottle Inspection", cv2.WND_PROP_VISIBLE) < 1:
         running = False
    except Exception as e:
        pass
    

GPIO.output(RELAY_PIN, RELAY_OFF)
GPIO.cleanup()
picam2.stop()
s.close()
cv2.destroyAllWindows()
sys.exit(0)