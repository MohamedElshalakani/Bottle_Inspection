from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import signal
import struct
import threading

# ── Pin assignments ──────────────────────────────────────────────
RELAY_CLASSIFY  = 17
RELAY_COUNT_MSB = 27
RELAY_COUNT_LSB = 22

RELAY_ON  = GPIO.HIGH
RELAY_OFF = GPIO.LOW

REJECT_PULSE_SEC = 0.2

# ── GPIO setup ───────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)

GPIO.setup(RELAY_CLASSIFY, GPIO.OUT, initial=RELAY_OFF)
GPIO.setup(RELAY_COUNT_MSB, GPIO.OUT, initial=RELAY_OFF)
GPIO.setup(RELAY_COUNT_LSB, GPIO.OUT, initial=RELAY_OFF)

# ── Camera setup ─────────────────────────────────────────────────
picam2 = Picamera2()

picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480)}
    )
)

picam2.start()

# ── Socket setup ─────────────────────────────────────────────────
PC_IP = "192.168.1.100"
PORT  = 5000

s = socket.socket()
s.connect((PC_IP, PORT))
s.settimeout(30.0)

# ── State ────────────────────────────────────────────────────────
good_count        = 0
no_cap_count      = 0
no_label_count    = 0

classify_off_timer = None
relay_lock = threading.Lock()

current_count     = 0
processed_ids     = set()

running = True

HEADER = 4


# ── Messaging helpers ────────────────────────────────────────────
def recv_exact(sock, n):
    data = b""

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionError("Socket closed by peer")

        data += chunk

    return data


def recv_msg(sock):
    raw_len = recv_exact(sock, HEADER)

    msg_len = struct.unpack(">I", raw_len)[0]

    return recv_exact(sock, msg_len).decode()


def send_msg(sock, text):
    encoded = text.encode()

    sock.sendall(
        struct.pack(">I", len(encoded)) + encoded
    )


def send_frame(sock):

    frame = picam2.capture_array()

    # Lower JPEG quality slightly for faster transfer
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]

    _, buffer = cv2.imencode(
        ".jpg",
        frame,
        encode_param
    )

    data = buffer.tobytes()

    sock.sendall(
        struct.pack(">I", len(data)) + data
    )


# ── Relay helpers ────────────────────────────────────────────────
def set_count_relays(count):

    msb = RELAY_ON if (count & 0b10) else RELAY_OFF
    lsb = RELAY_ON if (count & 0b01) else RELAY_OFF

    GPIO.output(RELAY_COUNT_MSB, msb)
    GPIO.output(RELAY_COUNT_LSB, lsb)

    msb_str = "ON (1)" if msb == RELAY_ON else "OFF (0)"
    lsb_str = "ON (1)" if lsb == RELAY_ON else "OFF (0)"

    print(
        f"  [COUNT  ] GPIO27(MSB)={msb_str}  "
        f"GPIO22(LSB)={lsb_str}  "
        f"binary={int(msb==RELAY_ON)}{int(lsb==RELAY_ON)}  "
        f"→ PLC reads {count} bottle(s)"
    )


def log_classify_relay(state, result):

    state_str = (
        "ON  → REJECTING"
        if state == RELAY_ON
        else "OFF → PASS"
    )

    print(
        f"  [CLASSIFY] GPIO17={state_str}  ({result})"
    )


def _relay_off():
    global classify_off_timer

    with relay_lock:
        GPIO.output(RELAY_CLASSIFY, RELAY_OFF)

        log_classify_relay(
            RELAY_OFF,
            "pulse ended"
        )

        classify_off_timer = None


# ── Graceful shutdown ────────────────────────────────────────────
def stop(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


# ── Boot Message ─────────────────────────────────────────────────
print("=" * 60)
print(f"  Connected to PC at {PC_IP}:{PORT}")
print("  Waiting for PC commands...")
print("=" * 60)


# ── Main loop ────────────────────────────────────────────────────
while running:

    # Receive next command
    try:
        cmd = recv_msg(s)

    except Exception as e:
        print(f"[ERROR] recv failed: {e}")
        break

    # ── Frame request ────────────────────────────────────────────
    if cmd == "CAPTURE":

        try:
            send_frame(s)

        except Exception as e:
            print(f"[ERROR] send_frame failed: {e}")
            break

    # ── Stable bottle count ──────────────────────────────────────
    elif cmd.startswith("COUNT:"):

        try:
            count = int(cmd.split(":")[1])
            count = min(count, 3)

        except Exception as e:
            print(f"[ERROR] COUNT parse failed: {e}")
            continue

        if count != current_count:

            print(f"\n{'─'*60}")
            print(
                f"  Count update: "
                f"{current_count} → {count} bottle(s)"
            )
            print(f"{'─'*60}")

            current_count = count

            set_count_relays(current_count)

    # ── Classification result ────────────────────────────────────
    elif cmd.startswith("ID:"):

        try:
            parts = cmd.split("|")

            bottle_id = int(
                parts[0].split(":")[1]
            )

            result = parts[1]

        except Exception as e:
            print(f"[ERROR] parse failed: {e}")
            continue

        # Prevent duplicate processing
        if bottle_id in processed_ids:
            continue

        processed_ids.add(bottle_id)

        print(f"\n{'─'*60}")
        print(f"  Bottle #{bottle_id} → {result}")
        print(f"{'─'*60}")

        # GOOD bottle
        if result == "Good":

            good_count += 1

            if classify_off_timer is not None:
                classify_off_timer.cancel()
                classify_off_timer = None

            with relay_lock:
                GPIO.output(
                    RELAY_CLASSIFY,
                    RELAY_OFF
                )

                log_classify_relay(
                    RELAY_OFF,
                    result
                )

        # DEFECT bottle
        elif result in ("No_cap", "No_label"):

            if result == "No_cap":
                no_cap_count += 1
            else:
                no_label_count += 1

            if classify_off_timer is not None:
                classify_off_timer.cancel()

            with relay_lock:
                GPIO.output(
                    RELAY_CLASSIFY,
                    RELAY_ON
                )

                log_classify_relay(
                    RELAY_ON,
                    result
                )

            classify_off_timer = threading.Timer(
                REJECT_PULSE_SEC,
                _relay_off
            )
            classify_off_timer.start()

        # Running totals
        print(
            f"  [TOTALS ] "
            f"Good={good_count}  "
            f"No_cap={no_cap_count}  "
            f"No_label={no_label_count}"
        )

        print(
            f"  [COUNT  ] Waiting for stable "
            f"COUNT:X from PC"
        )

    else:
        print(f"[WARN] Unknown command: '{cmd}'")


# ── Cleanup ──────────────────────────────────────────────────────
print("\nShutting down — all relays OFF")

if classify_off_timer is not None:
    classify_off_timer.cancel()

with relay_lock:
    GPIO.output(RELAY_CLASSIFY, RELAY_OFF)
    GPIO.output(RELAY_COUNT_MSB, RELAY_OFF)
    GPIO.output(RELAY_COUNT_LSB, RELAY_OFF)

GPIO.cleanup()

picam2.stop()

s.close()

print("Shutdown complete.")
