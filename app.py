import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8n model (auto-download from Ultralytics hub)
model = YOLO("yolov8n")

# Define vehicle-related classes
vehicle_classes = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle"}

# Set Streamlit page config
st.set_page_config(page_title="YOLOv8 Vehicle Detection", layout="wide")
st.title("🚗 Vehicle Detection using YOLOv8")

# Upload a video file
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))  # Resize for performance

        results = model(frame)[0]

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            label = model.names[int(cls_id)].lower()
            if label not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the result frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()
    st.success("✅ Video processing complete")
