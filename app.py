import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


MODEL_PATH = Path("models/face_cnn.keras")
LABELS_PATH = Path("models/labels.json")
IMG_SIZE = 160


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = {int(index): name for index, name in labels.items()}
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    return model, labels, face_detector, threading.Lock()


def predict_face(face_rgb, model, labels, lock):
    resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    with lock:
        probs = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probs))
    return labels[index], float(probs[index])


def detect_and_predict(image_rgb, model, labels, detector, lock):
    output = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
    results = []

    if len(faces) == 0:
        name, confidence = predict_face(image_rgb, model, labels, lock)
        return output, [(name, confidence, None)]

    for x, y, w, h in faces:
        pad = int(0.12 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_rgb.shape[1], x + w + pad)
        y2 = min(image_rgb.shape[0], y + h + pad)
        face_rgb = image_rgb[y1:y2, x1:x2]

        name, confidence = predict_face(face_rgb, model, labels, lock)
        results.append((name, confidence, (x1, y1, x2, y2)))

        label = f"{name} {confidence * 100:.1f}%"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 180, 80), 2)
        cv2.rectangle(output, (x1, max(0, y1 - 28)), (min(output.shape[1], x1 + 360), y1), (0, 180, 80), -1)
        cv2.putText(output, label, (x1 + 8, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    return output, results


class FaceRecognizer(VideoTransformerBase):
    def __init__(self):
        self.model, self.labels, self.detector, self.lock = load_assets()

    def transform(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        output_rgb, _ = detect_and_predict(image_rgb, self.model, self.labels, self.detector, self.lock)
        return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)


st.set_page_config(page_title="Nhận diện khuôn mặt", page_icon="camera", layout="centered")
st.title("Nhận diện khuôn mặt")

model, labels, detector, lock = load_assets()

tab_webcam, tab_upload = st.tabs(["Webcam realtime", "Upload ảnh"])

with tab_webcam:
    st.write("Bấm Start, cho phép trình duyệt truy cập camera, rồi đưa mặt vào khung hình.")
    webrtc_streamer(
        key="face-recognition",
        video_transformer_factory=FaceRecognizer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

with tab_upload:
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)
        output_rgb, results = detect_and_predict(image_rgb, model, labels, detector, lock)

        st.image(output_rgb, caption="Kết quả nhận diện", use_container_width=True)
        for name, confidence, box in results:
            suffix = "không phát hiện mặt, dự đoán trên cả ảnh" if box is None else "phát hiện mặt"
            st.success(f"{name} - {confidence * 100:.1f}% ({suffix})")
