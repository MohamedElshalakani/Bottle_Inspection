from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import signal
import sys
import time

RELAY_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

RELAY_ON = 1
RELAY_OFF = 0

GPIO.output(RELAY_PIN, RELAY_OFF)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

PC_IP = "192.168.1.7"
PORT = 5000

s = socket.socket()
s.connect((PC_IP, PORT))
s.settimeout(30.0)

good_count = 0
no_cap_count = 0
no_label_count = 0
relay_off_time = None

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
    frame = picam2.capture_array()
    _, buffer = cv2.imencode(".jpg", frame)
    data = buffer.tobytes()
    size = str(len(data)).ljust(16).encode()
    s.sendall(size)
    s.sendall(data)

print("Connected to PC. Running inspection...")

while running:
    if relay_off_time and time.time() >= relay_off_time:
        GPIO.output(RELAY_PIN, RELAY_OFF)
        print("Relay OFF")
        relay_off_time = None

    try:
        cmd = s.recv(1024).decode().strip()
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
            print(f"Parse error: {e}")
            continue

        if latest_id != last_processed_id:
            last_processed_id = latest_id

            print(f"--- Bottle {latest_id} Result: {latest_class} ---")

            if latest_class == "Good":
                good_count += 1
                GPIO.output(RELAY_PIN, RELAY_OFF)
                print(f"RELAY: OFF (Good) | Good:{good_count} No_cap:{no_cap_count} No_label:{no_label_count}")

            elif latest_class in ["No_cap", "No_label"]:
                if latest_class == "No_cap":
                    no_cap_count += 1
                else:
                    no_label_count += 1
                GPIO.output(RELAY_PIN, RELAY_ON)
                relay_off_time = time.time() + 0.2
                print(f"RELAY: ON - REJECTING ({latest_class}) | Good:{good_count} No_cap:{no_cap_count} No_label:{no_label_count}")

GPIO.output(RELAY_PIN, RELAY_OFF)
GPIO.cleanup()
picam2.stop()
s.close()
print("Shutdown complete.")