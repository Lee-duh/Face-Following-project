#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
import threading
import time

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
    boxes_cx = priors[:,0] + loc[:,0]*variances[0]*priors[:,2]
    boxes_cy = priors[:,1] + loc[:,1]*variances[0]*priors[:,3]
    boxes_w = priors[:,2]*np.exp(loc[:,2]*variances[1])
    boxes_h = priors[:,3]*np.exp(loc[:,3]*variances[1])
    boxes[:,0] = boxes_cx - boxes_w/2
    boxes[:,1] = boxes_cy - boxes_h/2
    boxes[:,2] = boxes_cx + boxes_w/2
    boxes[:,3] = boxes_cy + boxes_h/2
    return boxes

def decode_landmarks(landms, priors, variances=(0.1,0.2)):
    lms = np.empty_like(landms, dtype=np.float32)
    for i in range(5):
        lms[:,2*i] = priors[:,0] + landms[:,2*i]*variances[0]*priors[:,2]
        lms[:,2*i+1] = priors[:,1] + landms[:,2*i+1]*variances[0]*priors[:,3]
    return lms

def nms_numpy(boxes, scores, iou_threshold=0.4, top_k=5000):
    if boxes.shape[0]==0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    if top_k>0:
        order = order[:top_k]
    keep=[]
    while order.size>0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        rem_areas = areas[order[1:]]
        union = areas[i] + rem_areas - inter
        iou = inter/union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds+1]
    return np.array(keep, dtype=np.int32)

def preprocess_image(img, input_size, mean=(104,117,123)):
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img,(input_size,input_size)).astype(np.float32)
    img_resized -= np.array(mean, dtype=np.float32)
    blob = img_resized.transpose(2,0,1)[None,:,:,:].astype(np.float32)
    scale_x = w0 / float(input_size)
    scale_y = h0 / float(input_size)
    return blob, (scale_x, scale_y), (h0,w0)

def draw_detections(img, boxes, scores, landms):
    h0, w0 = img.shape[:2]
    col_width = w0 / 3.0
    action_text = "No face detect, rotate"

    # Draw column lines and labels
    col_width = w0 / 3.0
    cv2.line(img, (int(col_width), 0), (int(col_width), h0), (255,255,0), 2)
    cv2.line(img, (int(2*col_width), 0), (int(2*col_width), h0), (255,255,0), 2)
    cv2.putText(img, "Left", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(img, "Middle", (int(col_width)+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(img, "Right", (int(2*col_width)+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    if len(boxes) > 0:
        # Draw all boxes and landmarks with confidence
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img, f"{scores[i]:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            for k in range(5):
                lx = int(landms[i,2*k])
                ly = int(landms[i,2*k+1])
                cv2.circle(img, (lx,ly), 3, (0,255,255), -1)

        # Find the face with the largest area (closest face)
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        idx = np.argmax(areas)  # index of the largest box

        x1, y1, x2, y2 = boxes[idx].astype(int)
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2
        w = x2 - x1
        h = y2 - y1
        area = w*h

        # Determine horizontal position
        if cx < w0/3:
            action_text = "Move Left"
        elif cx < 2*w0/3:
            action_text = "Face Align - Stop"
        else:
            action_text = "Move Right"

        # Determine distance
        if area > 20000:  # face too close
            action_text = "Move Backward"
        elif area < 8000:  # face too far
            action_text = "Move Forward"

        
        # Draw all detected boxes and landmarks
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            for k in range(5):
                lx = int(landms[i,2*k])
                ly = int(landms[i,2*k+1])
                cv2.circle(img, (lx,ly), 3, (0,255,255), -1)

    # Display action text on top-left corner
    cv2.putText(img, action_text, (10, h0 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    return img

# ----------------- Frame capture thread -----------------
def capture_frames():
    global output_frame
    priors = generate_priors((INPUT_SIZE, INPUT_SIZE))
    cap = cv2.VideoCapture(0)

    # --------------- FPS START ----------------
    prev_time = time.time()
    fps = 0
    # --------------- FPS END ------------------

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate the frame 180 degrees if it's upside down
        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        # OR flip vertically (if needed)
        # frame = cv2.flip(frame, 0)

        # OR flip horizontally (if needed)
        # frame = cv2.flip(frame, 1)

        # Flip the frame to make it normal
        frame = cv2.flip(frame, 1)  # 1 = horizontal flip, 0 = vertical, -1 = both
        
        h0, w0 = frame.shape[:2]
        blob, (scale_x, scale_y), _ = preprocess_image(frame, INPUT_SIZE)
        outputs = sess.run(None, {input_name: blob.astype(np.float32)})
        loc, conf, landms = outputs
        loc, conf, landms = loc[0], conf[0], landms[0]

        scores = conf[:,1]
        inds = np.where(scores>CONF_THRESH)[0]

        if inds.size > 0:
            loc, scores_f, landms_sel, priors_sel = loc[inds], scores[inds], landms[inds], priors[inds,:]
            boxes = decode_boxes(loc, priors_sel)
            landms_d = decode_landmarks(landms_sel, priors_sel)

            # absolute coordinates
            boxes_abs = np.empty_like(boxes)
            boxes_abs[:,0] = boxes[:,0]*INPUT_SIZE*scale_x
            boxes_abs[:,1] = boxes[:,1]*INPUT_SIZE*scale_y
            boxes_abs[:,2] = boxes[:,2]*INPUT_SIZE*scale_x
            boxes_abs[:,3] = boxes[:,3]*INPUT_SIZE*scale_y

            landms_abs = np.empty_like(landms_d)
            for i in range(5):
                landms_abs[:,2*i] = landms_d[:,2*i]*INPUT_SIZE*scale_x
                landms_abs[:,2*i+1] = landms_d[:,2*i+1]*INPUT_SIZE*scale_y

            keep = nms_numpy(boxes_abs, scores_f, iou_threshold=NMS_THRESH, top_k=TOP_K)
            boxes_keep = boxes_abs[keep]; scores_keep = scores_f[keep]; landms_keep = landms_abs[keep]
        else:
            boxes_keep = np.array([]); scores_keep = np.array([]); landms_keep = np.array([])

        # draw
        frame = draw_detections(frame, boxes_keep, scores_keep, landms_keep)
        with lock:
            output_frame = frame.copy()

        # ----------------- FPS CALCULATION -----------------
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, h0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        # ---------------------------------------------------

        with lock:
            output_frame = frame.copy()
            
# ----------------- Flask streaming -----------------
def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg", output_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ----------------- Main -----------------
if __name__=="__main__":
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
