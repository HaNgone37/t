import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown
from preprocess_utils import preprocess_image
from PIL import Image

# --- Tải mô hình từ Google Drive nếu chưa có ---
model_path = "hand_sign_cnn_model.h5"
if not os.path.exists(model_path):
    st.info("Đang tải mô hình từ Google Drive...")
    gdown.download(id="1-qtMLem63El7msIK84PzMmMTvgyr9T1_", output=model_path, quiet=False)

model = tf.keras.models.load_model(model_path)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]

st.title("🤟 Nhận diện Ký hiệu Tay bằng CNN")
st.write("Upload ảnh bàn tay hoặc dùng webcam (khi chạy local)")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    preprocessed = preprocess_image(img_np)
    pred = model.predict(preprocessed)[0]
    idx = np.argmax(pred)
    confidence = pred[idx]

    st.markdown(f"### 👉 Kết quả: **{labels[idx]}** ({confidence:.2f})")
