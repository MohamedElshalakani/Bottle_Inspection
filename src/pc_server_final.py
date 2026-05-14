import socket
import numpy as np
import cv2
import math
import struct
import threading
import queue
import time
from collections import Counter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO(r"C:\UofCanada\Winter26\2220\Bottle_Inspection\models\best_fold5.pt")

tracker = DeepSort(
    max_age=20,
    n_init=4,
    max_cosine_distance=0.3,
    nn_budget=100
)

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", 5000))
server.listen(1)

print("Waiting for Pi to connect...")
conn, addr = server.accept()
conn.settimeout(5)
print(f"Pi connected from {addr}")

HEADER = 4

COUNT_STABLE_FRAMES = 3
MAX_EXPECTED_BOTTLES = 3

SINGLE_BOTTLE_PX = 80
OVERLAP_RATIO_THRESHOLD = 1.7

TRIGGER_X_FRAC = 0.70
MIN_HISTORY_FRAMES = 8
MIN_VOTE_FRAC = 0.60
SPATIAL_RADIUS_PX = 80
SPATIAL_COOLDOWN_S = 3.0
TRACK_CLEANUP_AGE = 60

frame_queue = queue.Queue(maxsize=1)
running_capture = True

track_history = {}
sent_ids = set()
track_last_cx = {}
track_triggered = set()
recently_counted = []
track_last_seen = {}
frame_count = 0


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def recv_msg(sock):
    raw_len = recv_exact(sock, HEADER)
    if raw_len is None:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    payload = recv_exact(sock, msg_len)
    return payload.decode() if payload else None


def send_msg(sock, text):
    encoded = text.encode()
    sock.sendall(struct.pack(">I", len(encoded)) + encoded)


def recv_frame(sock):
    raw_len = recv_exact(sock, HEADER)
    if raw_len is None:
        return None
    frame_len = struct.unpack(">I", raw_len)[0]
    data = recv_exact(sock, frame_len)
    if data is None:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def analyse_detections(boxes):
    if boxes is None or len(boxes) == 0:
        return [], []

    detections = []
    overlap_warnings = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1

        lbl = model.names[int(boxes.cls[i])]
        conf = boxes.conf[i].item()

        detections.append((cx, cy, lbl, conf, bw))

        if bw > SINGLE_BOTTLE_PX * OVERLAP_RATIO_THRESHOLD:
            msg = (
                f"[OVERLAP WARNING] Detection {i} width={bw:.0f}px "
                f">{OVERLAP_RATIO_THRESHOLD}× single bottle "
                f"({SINGLE_BOTTLE_PX}px). "
                f"Two bottles may be merged into one box."
            )
            overlap_warnings.append(msg)

    detections.sort(key=lambda d: d[0])
    return detections, overlap_warnings


count_frame_buffer = []
last_sent_count = -1

good_count = 0
no_cap_count = 0
no_label_count = 0
latest_class = "waiting"



def capture_thread():
    global running_capture

    while running_capture:
        try:
            send_msg(conn, "CAPTURE")
            frame = recv_frame(conn)

            if frame is None:
                running_capture = False
                break

            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(frame)

        except Exception as e:
            print(f"Capture thread error: {e}")
            running_capture = False
            break


threading.Thread(target=capture_thread, daemon=True).start()


try:
    while True:
        frame_count += 1

        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            if not running_capture:
                break
            continue

        results = model(frame, conf=0.50)[0]
        boxes = results.boxes

        detections, overlap_warnings = analyse_detections(boxes)

        for w in overlap_warnings:
            print(w)

        detections_for_tracker = []

        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = model.names[cls_id]

                w = x2 - x1
                h = y2 - y1

                detections_for_tracker.append(
                    ([x1, y1, w, h], conf, label)
                )

        tracks = tracker.update_tracks(
            detections_for_tracker,
            frame=frame
        )

        raw_count = sum(1 for t in tracks if t.is_confirmed())

        count_frame_buffer.append(raw_count)

        if len(count_frame_buffer) > COUNT_STABLE_FRAMES:
            count_frame_buffer.pop(0)

        if (
            len(count_frame_buffer) == COUNT_STABLE_FRAMES
            and len(set(count_frame_buffer)) == 1
        ):

            stable_count = count_frame_buffer[0]

            if stable_count != last_sent_count:

                if stable_count > MAX_EXPECTED_BOTTLES:
                    print(
                        f"[COUNT WARN] Stable count={stable_count} exceeds "
                        f"MAX_EXPECTED_BOTTLES={MAX_EXPECTED_BOTTLES}"
                    )
                    stable_count = MAX_EXPECTED_BOTTLES

                send_msg(conn, f"COUNT:{stable_count}")
                print(f"[COUNT STABLE] sending COUNT:{stable_count}")

                last_sent_count = stable_count

        h, w, _ = frame.shape
        trigger_px = int(w * TRIGGER_X_FRAC)

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id
            track_last_seen[track_id] = frame_count

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            label = track.get_det_class()

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            prev_cx = track_last_cx.get(track_id)
            if prev_cx is not None and prev_cx < trigger_px <= cx:
                track_triggered.add(track_id)
            track_last_cx[track_id] = cx

            if track_id not in track_history:
                track_history[track_id] = []

            if label is not None:
                track_history[track_id].append(label)

            if len(track_history[track_id]) > 15:
                track_history[track_id].pop(0)

            history = track_history[track_id]

            if track_id in sent_ids:
                continue

            if track_id not in track_triggered:
                continue

            if len(history) < MIN_HISTORY_FRAMES:
                continue

            final, win_count = Counter(history).most_common(1)[0]

            if win_count / len(history) < MIN_VOTE_FRAC:
                continue

            now = time.time()
            recently_counted = [
                (old_cx, old_cy, ts)
                for old_cx, old_cy, ts in recently_counted
                if now - ts <= SPATIAL_COOLDOWN_S * 2
            ]

            is_duplicate = any(
                math.hypot(cx - old_cx, cy - old_cy) <= SPATIAL_RADIUS_PX
                and now - ts <= SPATIAL_COOLDOWN_S
                for old_cx, old_cy, ts in recently_counted
            )

            if is_duplicate:
                sent_ids.add(track_id)
                print(f"[DEDUP] Skipping duplicate track ID:{track_id}")
                continue

            recently_counted.append((cx, cy, now))

            latest_class = final

            if final == "Good":
                good_count += 1
            elif final == "No_cap":
                no_cap_count += 1
            elif final == "No_label":
                no_label_count += 1

            send_msg(conn, f"ID:{track_id}|{final}")
            print(f"Sent classification → ID:{track_id}|{final}")

            sent_ids.add(track_id)

        stale_track_ids = [
            track_id
            for track_id, last_seen in track_last_seen.items()
            if frame_count - last_seen > TRACK_CLEANUP_AGE
        ]

        for track_id in stale_track_ids:
            track_history.pop(track_id, None)
            track_last_cx.pop(track_id, None)
            track_last_seen.pop(track_id, None)

        h, w, _ = frame.shape

        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]

                bw = x2 - x1

                color = (
                    (0, 140, 255)
                    if bw > SINGLE_BOTTLE_PX * OVERLAP_RATIO_THRESHOLD
                    else (128, 128, 128)
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            x1, y1, x2, y2 = map(int, track.to_ltrb())

            label = track.get_det_class()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"ID:{track_id} {label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.line(frame, (trigger_px, 0), (trigger_px, h), (0, 255, 255), 1)
        cv2.putText(
            frame,
            "TRIGGER",
            (trigger_px + 4, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

        frames_agreed = (
            count_frame_buffer.count(raw_count)
            if count_frame_buffer
            else 0
        )

        count_color = (
            (0, 200, 0)
            if frames_agreed == COUNT_STABLE_FRAMES
            else (0, 165, 255)
        )

        cv2.putText(
            frame,
            f"Visible: {raw_count} Stable: {frames_agreed}/{COUNT_STABLE_FRAMES}",
            (w - 320, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            count_color,
            2,
        )

        cv2.putText(
            frame,
            "MYK AUTOMATION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        cv2.rectangle(frame, (10, h - 120), (320, h - 10), (255, 255, 255), -1)

        cv2.putText(
            frame,
            f"Good: {good_count}",
            (20, h - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 0),
            2,
        )

        cv2.putText(
            frame,
            f"No Cap: {no_cap_count}",
            (20, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"No Label: {no_label_count}",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 254),
            2,
        )

        status = f"Current: {latest_class}"

        cv2.rectangle(
            frame,
            (w - 300, h - 60),
            (w - 10, h - 10),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            frame,
            status,
            (w - 290, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        cv2.imshow("Bottle Inspection - PC View", frame)

        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    print("Shutting down")

finally:
    running_capture = False
    conn.close()
    server.close()
    cv2.destroyAllWindows()
