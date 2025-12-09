# #!/usr/bin/env python3
# import cv2
# import numpy as np
# import onnxruntime as ort
# import time

# # -----------------------------
# # CONFIG
# # -----------------------------
# MODEL_PATH = r"C:\Users\User\OneDrive\Desktop\Robotic\Pytorch_Retinaface\retinaface_mnet0.25.onnx"   # your ONNX file
# CONF_THRESH = 0.6
# IOU_THRESH  = 0.4
# IMG_SIZE = (640, 640)

# # -----------------------------
# # HELPER: Preprocess
# # -----------------------------
# def preprocess(img):
#     h, w = img.shape[:2]
#     img_resized = cv2.resize(img, IMG_SIZE)

#     # Convert BGR -> RGB
#     img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

#     # Normalize + convert to float32
#     img_resized = img_resized.astype(np.float32)
#     img_resized /= 255.0

#     # HWC -> CHW
#     img_resized = np.transpose(img_resized, (2, 0, 1))

#     # Add batch dimension
#     img_resized = np.expand_dims(img_resized, axis=0)
#     return img_resized

# # -----------------------------
# # HELPER: Decode retinaface output
# # -----------------------------
# def decode(loc, priors, variances=[0.1, 0.2]):
#     boxes = np.concatenate((
#         priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
#         priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
#     ), axis=1)
#     boxes[:, :2] -= boxes[:, 2:] / 2
#     boxes[:, 2:] += boxes[:, :2]
#     return boxes

# def decode_landmarks(pre, priors, variances=[0.1, 0.2]):
#     landms = np.concatenate((
#         priors[:, :2] + pre[:, 0:2] * variances[0] * priors[:, 2:],
#         priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
#         priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
#         priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
#         priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]
#     ), axis=1)
#     return landms

# # -----------------------------
# # MAIN
# # -----------------------------
# def main():
#     print(f"Loading ONNX model: {MODEL_PATH}")
#     providers = ['CPUExecutionProvider']
#     sess = ort.InferenceSession(MODEL_PATH, providers=providers)

#     input_name = sess.get_inputs()[0].name
#     output_names = [o.name for o in sess.get_outputs()]

#     cap = cv2.VideoCapture(0)  # 0 = default webcam

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Camera error!")
#             break

#         img_input = preprocess(frame)

#         loc, conf, landms = sess.run(
#             output_names,
#             {input_name: img_input.astype(np.float32)}   # IMPORTANT
#         )

#         # Remove batch dimension
#         loc = loc[0]
#         conf = conf[0][:, 1]      # face confidence
#         landms = landms[0]

#         # Select valid detections
#         mask = conf > CONF_THRESH
#         loc = loc[mask]
#         conf = conf[mask]
#         landms = landms[mask]

#         # (Optional) If you have priors saved:
#         # priors = np.load("priors_640x640.npy")
#         # boxes = decode(loc, priors)
#         # lms   = decode_landmarks(landms, priors)

#         # For now, simple display of confidence only
#         for i in range(len(conf)):
#             cv2.putText(frame, f"{conf[i]:.2f}", (10, 30 + i * 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         cv2.imshow("RetinaFace ONNX", frame)
#         if cv2.waitKey(1) == 27:  
#             break  # ESC to quit

#     cap.release()
#     cv2.destroyAllWindows()

# # -----------------------------
# # RUN
# # -----------------------------
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
onnx_face_detect.py
Detect faces with RetinaFace MobileNet0.25 ONNX using ONNX Runtime (CPU).
Usage examples:
  # Run webcam (default)
  python onnx_face_detect.py --model retinaface_mnet0.25.onnx

  # Run on a single image and save result
  python onnx_face_detect.py --model retinaface_mnet0.25.onnx --input C:\path\to\test.jpg --save result.jpg

Notes:
- This script expects the ONNX model to output 3 tensors: loc, conf, landms
- Preprocessing matches the common RetinaFace test: subtract mean (104,117,123), no scaling by 1/255
"""
import argparse
import os
import cv2
import time
import numpy as np
import onnxruntime as ort

# -------------------------
# Utilities: PriorBox, decode, NMS
# -------------------------
def generate_priors(image_size, min_sizes=[[16, 32], [64, 128], [256, 512]], steps=[8, 16, 32], clip=False):
    """Generate priors in normalized coordinates [cx, cy, w, h] (all relative 0..1)."""
    priors = []
    img_h, img_w = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
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
    return priors  # shape (num_priors, 4) normalized

def decode_boxes(loc, priors, variances=(0.1, 0.2)):
    """Decode bbox predictions back to x1,y1,x2,y2 in normalized coordinates (0..1)."""
    # loc: (N, 4)  priors: (N, 4) with (cx, cy, w, h)
    boxes = np.empty_like(loc, dtype=np.float32)
    # center
    boxes_cx = priors[:, 0] + loc[:, 0] * variances[0] * priors[:, 2]
    boxes_cy = priors[:, 1] + loc[:, 1] * variances[0] * priors[:, 3]
    boxes_w = priors[:, 2] * np.exp(loc[:, 2] * variances[1])
    boxes_h = priors[:, 3] * np.exp(loc[:, 3] * variances[1])
    # convert to x1,y1,x2,y2
    boxes[:, 0] = boxes_cx - boxes_w / 2.0
    boxes[:, 1] = boxes_cy - boxes_h / 2.0
    boxes[:, 2] = boxes_cx + boxes_w / 2.0
    boxes[:, 3] = boxes_cy + boxes_h / 2.0
    return boxes

def decode_landmarks(landms, priors, variances=(0.1, 0.2)):
    """Decode 5 landmarks. landms shape (N,10) -> returns (N,10) normalized coords."""
    # each landmark is (dx,dy) relative to prior center and prior size
    lms = np.empty_like(landms, dtype=np.float32)
    for i in range(5):
        lms[:, 2*i]   = priors[:, 0] + landms[:, 2*i]   * variances[0] * priors[:, 2]
        lms[:, 2*i+1] = priors[:, 1] + landms[:, 2*i+1] * variances[0] * priors[:, 3]
    return lms

def nms_numpy(boxes, scores, iou_threshold=0.4, top_k=5000):
    """Basic NMS on numpy arrays. boxes are (N,4) in absolute pixel coords (x1,y1,x2,y2)"""
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)
    scores = scores.astype(np.float32)

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    if top_k > 0:
        order = order[:top_k]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        rem_areas = areas[order[1:]]
        union = (areas[i] + rem_areas - inter)
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

# -------------------------
# Preprocess & draw helpers
# -------------------------
def preprocess_image(img, input_size, mean=(104, 117, 123)):
    """Resize image to square input_size and preprocess for RetinaFace:
       subtract mean, convert to CHW float32, no scaling by /255.
    Returns: blob (1,3,H,W), scale (scale_x, scale_y), resized image shape
    """
    target_h, target_w = input_size, input_size
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (target_w, target_h)).astype(np.float32)
    # BGR ordering kept (the model was trained with BGR - mean subtraction)
    img_resized -= np.array(mean, dtype=np.float32)
    blob = img_resized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    scale_x = w0 / float(target_w)
    scale_y = h0 / float(target_h)
    return blob, (scale_x, scale_y), (h0, w0)

def draw_detections(img, boxes, scores, landms, vis_thres=0.5):
    """Draw boxes and 5 landmarks on image. boxes in absolute pixel coords."""
    for i in range(len(boxes)):
        score = scores[i]
        if score < vis_thres:
            continue
        x1, y1, x2, y2 = boxes[i].astype(int).tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{score:.3f}", (max(0, x1), max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # landmarks: landms i is shape (10,) normalized? here pass absolute pixel
        for k in range(5):
            lx = int(landms[i, 2*k])
            ly = int(landms[i, 2*k+1])
            cv2.circle(img, (lx, ly), 2, (0, 255, 255), -1)
    return img

# -------------------------
# Main detection loop
# -------------------------
def run_detection(model_path,
                  input_source=None,
                  input_size=320,
                  conf_thresh=0.5,
                  nms_thresh=0.4,
                  keep_top_k=500,
                  save_path=None,
                  show=True):
    # Model path check
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    print("Loading ONNX model:", model_path)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print("ONNX inputs:", [i.name for i in sess.get_inputs()])
    print("ONNX outputs:", output_names)

    # generate priors for the chosen input size (normalized)
    priors = generate_priors(image_size=(input_size, input_size))
    num_priors = priors.shape[0]
    print(f"Generated {num_priors} priors for input size {input_size}x{input_size}")

    def process_frame(frame):
        blob, (scale_x, scale_y), (h0, w0) = preprocess_image(frame, input_size)
        # Run ONNX
        ort_inputs = {input_name: blob.astype(np.float32)}
        outputs = sess.run(None, ort_inputs)

        # Expect outputs: loc, conf, landms (order depends on model export)
        # Try to detect which is which by shapes: loc:(1,N,4), conf:(1,N,2), landms:(1,N,10)
        if len(outputs) == 3:
            a, b, c = outputs[0], outputs[1], outputs[2]
            # figure out which is conf (has last dim 2) and landms (last dim 10)
            shapes = [x.shape for x in (a, b, c)]
            # simple heuristics:
            if a.shape[-1] == 4:
                loc = a
                if b.shape[-1] == 2:
                    conf = b
                    landms = c
                else:
                    conf = c
                    landms = b
            else:
                # fallback assume order loc, conf, landms
                loc, conf, landms = a, b, c
        else:
            raise RuntimeError("Unexpected ONNX output count: expected 3 outputs (loc, conf, landms)")

        loc = loc[0]            # (num_priors, 4)
        conf = conf[0]         # (num_priors, 2)
        landms = landms[0]     # (num_priors, 10)

        # face confidence is conf[:,1]
        scores = conf[:, 1]

        # filter by threshold
        inds = np.where(scores > conf_thresh)[0]
        if inds.size == 0:
            return frame, []

        loc = loc[inds]
        scores_f = scores[inds]
        landms = landms[inds]
        priors_sel = priors[inds, :]

        # decode to normalized boxes and landmarks
        decoded_boxes = decode_boxes(loc, priors_sel)  # normalized
        decoded_landms = decode_landmarks(landms, priors_sel)  # normalized (x,y) pairs

        # convert normalized coords to absolute pixel coords on original image
        # note: priors were made relative to input_size; decode returned normalized wrt input_size
        # but preprocess used resizing from original->input_size. To map back:
        # x_abs = norm_x * input_size * scale_x
        # y_abs = norm_y * input_size * scale_y
        x_scale = w0 / float(input_size)
        y_scale = h0 / float(input_size)

        boxes_abs = np.empty_like(decoded_boxes)
        boxes_abs[:, 0] = decoded_boxes[:, 0] * input_size * x_scale
        boxes_abs[:, 1] = decoded_boxes[:, 1] * input_size * y_scale
        boxes_abs[:, 2] = decoded_boxes[:, 2] * input_size * x_scale
        boxes_abs[:, 3] = decoded_boxes[:, 3] * input_size * y_scale

        landms_abs = np.empty_like(decoded_landms)
        for i in range(5):
            landms_abs[:, 2*i]   = decoded_landms[:, 2*i]   * input_size * x_scale
            landms_abs[:, 2*i+1] = decoded_landms[:, 2*i+1] * input_size * y_scale

        # NMS
        keep = nms_numpy(boxes_abs, scores_f, iou_threshold=nms_thresh, top_k=keep_top_k)
        if keep.size == 0:
            return frame, []

        boxes_keep = boxes_abs[keep]
        scores_keep = scores_f[keep]
        landms_keep = landms_abs[keep]

        # draw
        out_img = frame.copy()
        out_img = draw_detections(out_img, boxes_keep, scores_keep, landms_keep, vis_thres=conf_thresh)
        dets = [{'box': boxes_keep[i].tolist(), 'score': float(scores_keep[i]), 'landms': landms_keep[i].tolist()} for i in range(len(keep))]
        return out_img, dets

    # Input handling: image file or webcam
    if input_source:
        # single image
        if not os.path.isfile(input_source):
            raise FileNotFoundError(f"Input image not found: {input_source}")
        img = cv2.imread(input_source)
        if img is None:
            raise RuntimeError("Failed to read image (cv2 returned None)")
        out_img, dets = process_frame(img)
        if save_path:
            cv2.imwrite(save_path, out_img)
            print("Saved result to", save_path)
        if show:
            cv2.imshow("RetinaFace ONNX", out_img)
            print("Detections:", dets)
            print("Press any key to exit")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Detections:", dets)
        return

    # webcam loop
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0). Try --input <image_path> instead.")
    fps_time = time.time()
    print("Starting webcam. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera, exiting...")
            break
        t0 = time.time()
        out_img, dets = process_frame(frame)
        t1 = time.time()
        fps = 1.0 / (t1 - fps_time + 1e-6)
        fps_time = t1
        cv2.putText(out_img, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("RetinaFace ONNX", out_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetinaFace ONNX (MobileNet0.25) inference (CPU)")
    parser.add_argument("--model", type=str, default="retinaface_mnet0.25.onnx", help="ONNX model filename")
    parser.add_argument("--input", type=str, default=None, help="Path to a single image. If omitted, webcam is used.")
    parser.add_argument("--input_size", type=int, default=320, help="ONNX model input long side (square).")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--topk", type=int, default=500, help="Top-K before NMS")
    parser.add_argument("--save", type=str, default=None, help="If set, save single-image output to this path")
    parser.add_argument("--noshow", action="store_true", help="Do not show result window (useful for headless runs)")
    args = parser.parse_args()

    run_detection(
        model_path=args.model,
        input_source=args.input,
        input_size=args.input_size,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        keep_top_k=args.topk,
        save_path=args.save,
        show=not args.noshow
    )
