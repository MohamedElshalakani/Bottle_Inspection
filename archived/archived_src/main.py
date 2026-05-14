import cv2
import numpy as np
from picamera2 import Picamera2
import joblib
import time

net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
clf = joblib.load("bottle_classifier.pkl")

label_map = {0: "GOOD", 1: "DEFECTIVE - Missing Cap", 2: "DEFECTIVE - No Label"}

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}, buffer_count=2)
picam2.configure(config)
picam2.start()
time.sleep(0.5)

good_count = defective_count = total_count = 0
last_counted_box = None
DIST_THRESHOLD = 50

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def detect(frame, conf_threshold=0.4, iou_threshold=0.45):
    h, w = frame.shape[:2]
    input_size = 320
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0]  # shape (84, 8400) or (8400, 84)
    
    if outputs.shape[0] == 84:
        outputs = outputs.transpose(1, 0)  # (8400, 84)
    
    boxes, confidences = [], []
    for pred in outputs:
        class_logits = pred[4:84]
        class_probs = sigmoid(class_logits)
        class_id = np.argmax(class_probs)
        score = class_probs[class_id]
        
        if class_id == 39 and score > conf_threshold:
            xc, yc, bw, bh = pred[0:4]
            xc = (xc / input_size) * w
            yc = (yc / input_size) * h
            bw = (bw / input_size) * w
            bh = (bh / input_size) * h
            x1 = int(xc - bw/2)
            y1 = int(yc - bh/2)
            boxes.append([x1, y1, int(bw), int(bh)])
            confidences.append(float(score))
    
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    result = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, wb, hb = boxes[i]
            result.append((x, y, x+wb, y+hb, confidences[i]))
    return result

print("="*50)
print("MKY Automation - Bottle Inspection System (FIXED)")
print("Press 'q' to quit, 'r' to reset")
print("="*50)

current_id = None
start_time = None
pred_buffer = []
DELAY = 0.7

while True:
    frame = picam2.capture_array()
    h, w = frame.shape[:2]
    boxes = detect(frame)
    
    center = (w//2, h//2)
    best_box = None
    best_dist = float('inf')
    for (x1,y1,x2,y2,conf) in boxes:
        bcx = (x1+x2)//2
        bcy = (y1+y2)//2
        dist = (bcx-center[0])**2 + (bcy-center[1])**2
        if dist < best_dist:
            best_dist = dist
            best_box = (x1,y1,x2,y2,conf)
    
    now = time.time()
    if best_box:
        x1,y1,x2,y2,conf = best_box
        bid = f"{x1//30}_{y1//30}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
        
        if current_id != bid:
            current_id = bid
            start_time = now
            pred_buffer = []
        
        if now - start_time < DELAY:
            cv2.putText(frame, f"ANALYZING... {int(DELAY-(now-start_time))}s", (x1, y1-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        else:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, (200,200))
                feat = gray.flatten().reshape(1,-1)
                pred = clf.predict(feat)[0]
                prob = max(clf.predict_proba(feat)[0]) * 100
                pred_buffer.append(pred)
                if len(pred_buffer) > 3:
                    pred_buffer.pop(0)
                final = max(set(pred_buffer), key=pred_buffer.count)
                color = (0,255,0) if final==0 else (0,0,255)
                label = f"{label_map[final]} ({prob:.1f}%)"
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                
                bc = ((x1+x2)//2, (y1+y2)//2)
                count_it = True
                if last_counted_box:
                    lc = ((last_counted_box[0]+last_counted_box[2])//2, (last_counted_box[1]+last_counted_box[3])//2)
                    if ((bc[0]-lc[0])**2 + (bc[1]-lc[1])**2)**0.5 < DIST_THRESHOLD:
                        count_it = False
                if count_it:
                    total_count += 1
                    if final == 0:
                        good_count += 1
                    else:
                        defective_count += 1
                    last_counted_box = (x1,y1,x2,y2)
    else:
        current_id = None
        start_time = None
        pred_buffer = []
        cv2.putText(frame, "NO BOTTLE", (w//2-60,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
   
    cv2.putText(frame, "MKY AUTOMATION", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"TOTAL: {total_count}", (w-180, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"GOOD: {good_count}", (w-180, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"DEFECT: {defective_count}", (w-180, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.drawMarker(frame, center, (255,255,0), cv2.MARKER_CROSS, 20, 1)
    
    cv2.imshow("Inspection System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        good_count = defective_count = total_count = 0
        last_counted_box = None

cv2.destroyAllWindows()
picam2.close()
print(f"\nFinal - Total: {total_count}, Good: {good_count}, Defective: {defective_count}")