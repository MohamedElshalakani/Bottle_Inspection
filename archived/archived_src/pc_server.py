import socket
import numpy as np
import cv2
import torch
from collections import Counter
import time

print("[PC] Loading model...")

model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=r"C:\Users\mubar\OneDrive - Universities of Canada\Documents\Bottle_Inspection\yolov5\runs\train\exp10\weights\best.pt"
)

model.conf = 0.3


# ---------------- SOCKET ----------------
server = socket.socket()
server.bind(("0.0.0.0", 5000))
server.listen(1)

print("[PC] Waiting for Pi...")
conn, addr = server.accept()
print("[PC] Connected:", addr)

conn.settimeout(1.5)


# ---------------- SAFE RECEIVE ----------------
def recvall(n):
    data = b""
    while len(data) < n:
        try:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        except:
            return None
    return data


def get_frame():
    size_data = recvall(16)
    if not size_data:
        return None

    try:
        size = int(size_data.decode().strip())
    except:
        return None

    data = recvall(size)
    if data is None:
        return None

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return frame


# ---------------- STATE ----------------
active_id = 0


while True:

    try:
        buffer = []

        # ---------------- INCREMENT ID ----------------
        active_id += 1

        # ---------------- CAPTURE 5 FRAMES ----------------
        for i in range(5):

            conn.sendall(b"CAPTURE")
            frame = get_frame()

            if frame is None:
                continue

            results = model(frame)
            df = results.pandas().xyxy[0]

            if len(df) == 0:
                label = "no_bottle"
            else:
                df["area"] = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
                best = df.loc[df["area"].idxmax()]
                label = best["name"]

            buffer.append(label)
            print(f"[PC] Frame {i+1}: {label}")

        # ---------------- MAJORITY VOTE ----------------
        final = Counter(buffer).most_common(1)[0][0] if buffer else "error"

        print("\n====================")
        print("[PC] FINAL:", final)
        print("====================\n")

        # ---------------- SEND TO PI (FIXED FORMAT) ----------------
        message = f"ID:{active_id}|{final}"
        conn.sendall(message.encode())

        time.sleep(0.2)

    except KeyboardInterrupt:
        break

conn.close()
server.close()