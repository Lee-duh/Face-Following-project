#!/usr/bin/env python3
"""
improved_mobilenet.py

- RetinaFace (ONNX) 320x320
- Fixed thresholds and smoothing (no calibration)
- Inference every 2 frames
- Follows largest face when multiple faces present
- Prints intentions in terminal and overlays them BOTH at top and near bbox
- Streams to /video_feed via Flask

Usage:
    python3 improved_mobilenet.py
Open browser: http://<host-ip>:5000/video_feed
"""
#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
import threading
import time
from auppbot import AUPPBot

# ----------------- Flask app -----------------
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# ----------------- ONNX RetinaFace -----------------
MODEL_PATH = "retinaface_mnet0.25.onnx"
INPUT_SIZE = 320
CONF_THRESH = 0.5
NMS_THRESH = 0.4
TOP_K = 500

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

# ----------------- ROBOT INIT -----------------
try:
    bot = AUPPBot(port="/dev/ttyUSB0", baud=115200)
    print("[ROBOT] Initialized successfully")
except Exception as e:
    print("[ROBOT ERROR] Cannot initialize bot:", e)
    bot = None

# ----------------- ROBOT MOVEMENT FUNCTIONS -----------------
MOVE_SPEED = 35
TURN_SPEED = 35
MOVE_DURATION = 0.30

def move_forward():
    if bot:
        bot.motor1.forward(MOVE_SPEED)
        bot.motor2.forward(MOVE_SPEED)
        bot.motor3.forward(MOVE_SPEED)
        bot.motor4.forward(MOVE_SPEED)
        time.sleep(MOVE_DURATION)
        bot.stop_all()

def move_backward():
    if bot:
        bot.motor1.backward(MOVE_SPEED)
        bot.motor2.backward(MOVE_SPEED)
        bot.motor3.backward(MOVE_SPEED)
        bot.motor4.backward(MOVE_SPEED)
        time.sleep(MOVE_DURATION)
        bot.stop_all()

def turn_left():
    if bot:
        bot.motor1.backward(TURN_SPEED)
        bot.motor2.stop()
        bot.motor3.backward(TURN_SPEED)
        bot.motor4.stop()
        time.sleep(MOVE_DURATION)
        bot.stop_all()

def turn_right():
    if bot:
        bot.motor1.stop()
        bot.motor2.backward(TURN_SPEED)
        bot.motor3.stop()
        bot.motor4.backward(TURN_SPEED)
        time.sleep(MOVE_DURATION)
        bot.stop_all()

def stop_robot():
    if bot:
        bot.stop_all()

# ----------------------------------------------
# FACE DETECTION UTILITIES
# ----------------------------------------------
def generate_priors(image_size, min_sizes=[[16,32],[64,128],[256,512]], steps=[8,16,32], clip=False):
    priors = []
    img_h, img_w = image_size
    for k, min_sizes_k in enumerate(min_sizes):
        step = steps[k]
        fm_h = int(np.ceil(img_h / step))
        fm_w = int(np.ceil(img_w / step))
        for i in range(fm_h):
            for j in range(fm_w):
                for min_size in min_sizes_k:
                    cx = (j + 0.5) * step / img_w
                    cy = (i + 0.5) * step / img_h
                    s_kx = min_size / img_w
                    s_ky = min_size / img_h
                    priors.append([cx, cy, s_kx, s_ky])
    priors = np.array(priors, dtype=np.float32)
    if clip:
        priors = np.clip(priors, 0.0, 1.0)
    return priors

def decode_boxes(loc, priors, variances=(0.1,0.2)):
    boxes = np.empty_like(loc, dtype=np.float32)
    boxes_cx = priors[:,0] + loc[:,0] * variances[0] * priors[:,2]
    boxes_cy = priors[:,1] + loc[:,1] * variances[0] * priors[:,3]
    boxes_w = priors[:,2] * np.exp(loc[:,2] * variances[1])
    boxes_h = priors[:,3] * np.exp(loc[:,3] * variances[1])
    boxes[:,0] = boxes_cx - boxes_w/2
    boxes[:,1] = boxes_cy - boxes_h/2
    boxes[:,2] = boxes_cx + boxes_w/2
    boxes[:,3] = boxes_cy + boxes_h/2
    return boxes

def decode_landmarks(landms, priors, variances=(0.1,0.2)):
    lms = np.empty_like(landms)
    for i in range(5):
        lms[:,2*i] = priors[:,0] + landms[:,2*i] * variances[0] * priors[:,2]
        lms[:,2*i+1] = priors[:,1] + landms[:,2*i+1] * variances[0] * priors[:,3]
    return lms

def nms_numpy(boxes, scores, iou_threshold=0.4, top_k=500):
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1+1) * (y2-y1+1)
    order = scores.argsort()[::-1][:top_k]
    keep = []

    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        inter = w*h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter/union

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds+1]

    return np.array(keep, dtype=np.int32)

def preprocess_image(img, input_size, mean=(104,117,123)):
    h0, w0 = img.shape[:2]
    resized = cv2.resize(img, (input_size, input_size)).astype(np.float32)
    resized -= np.array(mean, dtype=np.float32)
    blob = resized.transpose(2,0,1)[None,:,:,:]
    scale_x = w0 / float(input_size)
    scale_y = h0 / float(input_size)
    return blob, (scale_x, scale_y), (h0, w0)

# -----------------------------------------------------------------
# DRAW DETECTIONS + TRIGGER ROBOT MOVEMENT BASED ON action_text
# -----------------------------------------------------------------
def draw_detections(img, boxes, scores, landms):
    h0, w0 = img.shape[:2]
    col_width = w0 / 3

    action_text = "No face detect, rotate"

    # Columns
    cv2.line(img, (int(col_width), 0), (int(col_width), h0), (255,255,0), 2)
    cv2.line(img, (int(2*col_width), 0), (int(2*col_width), h0), (255,255,0), 2)

    cv2.putText(img, "Left", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(img, "Middle", (int(col_width)+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(img, "Right", (int(2*col_width)+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # ------------------ IF FACE DETECTED ------------------
    if len(boxes) > 0:
        areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        idx = np.argmax(areas)

        x1, y1, x2, y2 = boxes[idx].astype(int)
        cx = (x1 + x2) / 2
        area = areas[idx]

        if cx < w0/3:
            action_text = "Move Left"
            turn_left()

        elif cx < 2*w0/3:
            action_text = "Face Align - Stop"
            stop_robot()

        else:
            action_text = "Move Right"
            turn_right()

        if area > 20000:
            action_text = "Move Backward"
            move_backward()

        elif area < 8000:
            action_text = "Move Forward"
            move_forward()

    # -------- Display action ------------
    cv2.putText(img, action_text, (10, h0 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    return img

# -----------------------------------------------------------------
# FRAME CAPTURE THREAD
# -----------------------------------------------------------------
def capture_frames():
    global output_frame
    priors = generate_priors((INPUT_SIZE, INPUT_SIZE))
    cap = cv2.VideoCapture(0)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h0, w0 = frame.shape[:2]

        blob, (scale_x, scale_y), _ = preprocess_image(frame, INPUT_SIZE)
        loc, conf, landms = sess.run(None, {input_name: blob})

        loc, conf, landms = loc[0], conf[0], landms[0]
        scores = conf[:,1]
        inds = np.where(scores > CONF_THRESH)[0]

        if inds.size:
            loc_sel = loc[inds]
            score_sel = scores[inds]
            land_sel = landms[inds]
            priors_sel = priors[inds]

            boxes = decode_boxes(loc_sel, priors_sel)
            land_d = decode_landmarks(land_sel, priors_sel)

            boxes_abs = np.zeros_like(boxes)
            boxes_abs[:,0] = boxes[:,0] * INPUT_SIZE * scale_x
            boxes_abs[:,1] = boxes[:,1] * INPUT_SIZE * scale_y
            boxes_abs[:,2] = boxes[:,2] * INPUT_SIZE * scale_x
            boxes_abs[:,3] = boxes[:,3] * INPUT_SIZE * scale_y

            land_abs = np.zeros_like(land_d)
            for i in range(5):
                land_abs[:,2*i] = land_d[:,2*i] * INPUT_SIZE * scale_x
                land_abs[:,2*i+1] = land_d[:,2*i+1] * INPUT_SIZE * scale_y

            keep = nms_numpy(boxes_abs, score_sel)
            boxes_keep = boxes_abs[keep]
            scores_keep = score_sel[keep]
            land_keep = land_abs[keep]

        else:
            boxes_keep = np.array([])
            scores_keep = np.array([])
            land_keep = np.array([])

        frame = draw_detections(frame, boxes_keep, scores_keep, land_keep)

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, h0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        with lock:
            output_frame = frame.copy()

# ----------------- FLASK STREAM -----------------
def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg", output_frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# ----------------- MAIN -----------------
if __name__ == "__main__":
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
