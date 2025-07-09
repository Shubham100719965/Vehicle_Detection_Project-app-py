import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("yolov8n")

vehicle_classes = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle"}
line_y = 180  # y-position of red counting line
offset = 6
pixels_to_meter = 0.05
speed_threshold = 30  # km/h

# Streamlit setup
st.set_page_config(page_title="YOLOv8 Vehicle Detection", layout="wide")
st.title("🚗 Vehicle Detection with Speed Estimation and Counting")

uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    vehicle_count = 0
    track_history = {}
    next_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        current_time = time.time()
        results = model(frame)[0]

        # Draw red counting line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            label = model.names[int(cls_id)].lower()
            if label not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Match vehicle by proximity
            matched_id = None
            for track_id, track in track_history.items():
                px, py = track["position"]
                if abs(cx - px) < 35 and abs(cy - py) < 35:
                    matched_id = track_id
                    break

            if matched_id is None:
                matched_id = next_id
                track_history[matched_id] = {
                    "position": (cx, cy),
                    "time": current_time,
                    "counted": False,
                    "speed": 0
                }
                next_id += 1
            else:
                prev = track_history[matched_id]
                distance_px = np.linalg.norm([cx - prev["position"][0], cy - prev["position"][1]])
                time_diff = current_time - prev["time"]
                speed_mps = (distance_px * pixels_to_meter) / time_diff if time_diff > 0 else 0
                speed_kmph = speed_mps * 3.6
                track_history[matched_id] = {
                    "position": (cx, cy),
                    "time": current_time,
                    "counted": prev["counted"],
                    "speed": speed_kmph
                }

                # Count vehicle if it crosses the red line
                if not prev["counted"] and prev["position"][1] < line_y <= cy:
                    vehicle_count += 1
                    track_history[matched_id]["counted"] = True

                # Draw box and speed
                color = (0, 255, 0) if speed_kmph < speed_threshold else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label.upper()}", (x1, y1 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"{speed_kmph:.1f} km/h", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Draw large vehicle count on frame
        cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"🚘 VEHICLE COUNT: {vehicle_count}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # Show the result frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()
    st.success("✅ Video processing complete")
