from picamera2 import Picamera2
import socket
import cv2
import time

PC_IP = "192.168.1.1"
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((PC_IP, PORT))

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

print("Streaming started...")

while True:
    frame = picam2.capture_array()

    # compress frame
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    data = buffer.tobytes()

    # send size then data
    client.send(len(data).to_bytes(4, 'big'))
    client.sendall(data)

    # receive result from PC
    result = client.recv(1024).decode()
    print("PC:", result)

    time.sleep(0.1)