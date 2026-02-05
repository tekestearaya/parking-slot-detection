""".............. webcam........
from flask import Flask, Response, jsonify, render_template
import cv2
import os

from src.model.inference import InferenceEngine
from src.model.yolo_wrapper import YOLOv8Model

app = Flask(__name__)

# Check for trained weights; fallback to pretrained YOLOv8n
weights_path = "weights/best.pt"
if not os.path.exists(weights_path):
    print("Weights not found, using pretrained YOLOv8n for demo")
    weights_path = "yolov8n.pt"

# Initialize YOLO model and inference engine
model = YOLOv8Model(weights_path)
engine = InferenceEngine(model)

cap = cv2.VideoCapture(0)  # 0 = default webcam

CLASS_MAP = {0: "occupied", 1: "available"}

# Stream frames to browser
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = engine.run(frame, conf_thres=0.5)

        for det in detections:
            label = CLASS_MAP.get(det["class_id"], "unknown")
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = (0, 0, 255) if label == "occupied" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/count")
def count():
    success, frame = cap.read()
    if not success:
        return jsonify({"occupied": 0, "available": 0})

    detections = engine.run(frame)
    return jsonify({
        "occupied": sum(1 for d in detections if d["class_id"] == 0),
        "available": sum(1 for d in detections if d["class_id"] == 1)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
"""











from flask import Flask, Response, jsonify, render_template
import cv2
import os
import threading
import time

from src.model.inference import InferenceEngine
from src.model.yolo_wrapper import YOLOv8Model

app = Flask(__name__)

# -------------------------------
# Model initialization
# -------------------------------
weights_path = "weights/best.pt"
if not os.path.exists(weights_path):
    print("Weights not found, using pretrained YOLOv8n for demo")
    weights_path = "yolov8n.pt"

model = YOLOv8Model(weights_path)
engine = InferenceEngine(model)

# -------------------------------
# Video initialization
# -------------------------------
VIDEO_PATH = "videos/parking_test.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

CLASS_MAP = {0: "occupied", 1: "available"}

# -------------------------------
# Shared frame (THREAD SAFE)
# -------------------------------
latest_frame = None
frame_lock = threading.Lock()

# -------------------------------
# Background video reader thread
# -------------------------------
def video_reader():
    global latest_frame
    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.03)  # ~30 FPS

# Start background thread
threading.Thread(target=video_reader, daemon=True).start()

# -------------------------------
# Video streaming generator
# -------------------------------
def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        detections = engine.run(frame, conf_thres=0.5)

        occupied = 0
        available = 0

        # Draw bounding boxes
        for det in detections:
            label = CLASS_MAP.get(det["class_id"], "unknown")
            x1, y1, x2, y2 = map(int, det["bbox"])

            if label == "occupied":
                color = (0, 0, 255)   # Red
                occupied += 1
            else:
                color = (0, 255, 0)   # Green
                available += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # -----------------------------
        # STATUS BOX (Top-left corner)
        # -----------------------------
        box_x, box_y = 10, 10
        box_width, box_height = 260, 80

        # Background rectangle
        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (50, 50, 50),
            -1
        )

        # Occupied (RED)
        cv2.putText(
            frame,
            f"OCCUPIED: {occupied}",
            (box_x + 10, box_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # Available (GREEN)
        cv2.putText(
            frame,
            f"AVAILABLE: {available}",
            (box_x + 10, box_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


# -------------------------------
# Flask routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/count")
def count():
    with frame_lock:
        if latest_frame is None:
            return jsonify({"occupied": 0, "available": 0})
        frame = latest_frame.copy()

    detections = engine.run(frame, conf_thres=0.5)

    return jsonify({
        "occupied": sum(1 for d in detections if d["class_id"] == 0),
        "available": sum(1 for d in detections if d["class_id"] == 1)
    })

# -------------------------------
# App entry point
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
