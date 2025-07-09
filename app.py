import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
vehicle_classes = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle"}

# Parameters
pos_linha = 550
offset = 6
largura_min = 80
altura_min = 80
pixels_to_meters = 0.1
speed_threshold = 30

# Speed calculation helper
def calculate_speed(distance_pixels, time_seconds):
    distance_meters = distance_pixels * pixels_to_meters
    return (distance_meters / time_seconds) * 3.6

# Center point
def pega_centro(x, y, w, h):
    return x + w // 2, y + h // 2

# Streamlit UI
st.title("🚗 Vehicle Detection, Speed Estimation & Counting")
video_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    carros = 0
    detec = []
    vehicle_tracks = {}
    next_track_id = 0
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, np.ones((5, 5)))
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_time = time.time()

        results = model(frame)[0]
        cv2.line(frame, (25, pos_linha), (1200, pos_linha), (255,127,0), 3)

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            label = model.names[int(cls_id)].lower()
            if label not in vehicle_classes:
                continue

            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            cx, cy = pega_centro(x, y, w, h)

            color = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if (w >= largura_min and h >= altura_min):
                detec.append((cx, cy))
                for (dcx, dcy) in detec:
                    if pos_linha - offset < dcy < pos_linha + offset:
                        carros += 1
                        cv2.line(frame, (25, pos_linha), (1200, pos_linha), (0,127,255), 3)
                        detec.remove((dcx, dcy))

            # Speed tracking
            matched_id = None
            for track_id, data in vehicle_tracks.items():
                px, py = data['last_position']
                if np.linalg.norm([cx - px, cy - py]) < 50:
                    matched_id = track_id
                    break

            if matched_id is None:
                matched_id = next_track_id
                vehicle_tracks[matched_id] = {
                    'last_position': (cx, cy),
                    'last_time': current_time,
                    'speed': 0
                }
                next_track_id += 1
            else:
                track = vehicle_tracks[matched_id]
                time_diff = current_time - track['last_time']
                if time_diff > 0:
                    pixel_distance = np.linalg.norm([cx - track['last_position'][0], cy - track['last_position'][1]])
                    speed = calculate_speed(pixel_distance, time_diff)
                    track['last_position'] = (cx, cy)
                    track['last_time'] = current_time
                    track['speed'] = speed

                    speed_text = f"{speed:.1f} km/h"
                    speed_color = (0, 0, 255) if speed > speed_threshold else (0, 255, 0)
                    cv2.putText(frame, speed_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)

        # Vehicle Count Display
        cv2.putText(frame, f"VEHICLE COUNT : {carros}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    st.success("✅ Video processing completed.")
