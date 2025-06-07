import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import requests
import os
import matplotlib.pyplot as plt

# # Tải model từ Google Drive
# @st.cache_resource
# def load_model():
#     model_path = "hand_sign_cnn_model.h5"
#     if not os.path.exists(model_path):
#         file_id = "1-qtMLem63El7msIK84PzMmMTvgyr9T1_"
#         url = f"https://drive.google.com/uc?export=download&id={file_id}"
#         r = requests.get(url)
#         with open(model_path, "wb") as f:
#             f.write(r.content)
#     return tf.keras.models.load_model(model_path)

import gdown

@st.cache_resource
def load_model():
    model_path = "hand_sign_cnn_model.h5"
    if not os.path.exists(model_path):
        file_id = "1-qtMLem63El7msIK84PzMmMTvgyr9T1_"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

# Tiền xử lý ảnh
def preprocess_image(img):
    img_size = 300
    img_white = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    h, w = img.shape[:2]
    aspect_ratio = h / w

    if aspect_ratio > 1:
        k = img_size / h
        w_cal = int(k * w)
        img_resize = cv2.resize(img, (w_cal, img_size))
        w_gap = (img_size - w_cal) // 2
        img_white[:, w_gap:w_gap + w_cal] = img_resize
    else:
        k = img_size / w
        h_cal = int(k * h)
        img_resize = cv2.resize(img, (img_size, h_cal))
        h_gap = (img_size - h_cal) // 2
        img_white[h_gap:h_gap + h_cal, :] = img_resize

    gray = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_img = cv2.filter2D(contrast_img, -1, sharp_kernel)
    norm_img = cv2.resize(sharp_img, (224, 224)).astype(np.float32) / 255.0
    return np.expand_dims(norm_img, axis=(0, -1))

# Giao diện Streamlit
st.set_page_config(page_title="Nhận diện tay", layout="centered")

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]

st.markdown("""
    <h1 style='text-align: center; font-size: 48px;'>🤟 Nhận diện Ký hiệu Tay bằng <span style='color:#ff4b4b'>CNN</span></h1>
    <p style='text-align: center; font-size: 18px;'>Upload ảnh bàn tay để dự đoán ký hiệu. Chạy webcam khi dùng local!</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# Ảnh upload
uploaded_file = st.file_uploader("📤 Chọn ảnh bàn tay", type=["jpg", "jpeg", "png"])

# Nút thử lại
if st.button("🔁 Thử lại"):
    st.experimental_rerun()

if uploaded_file:
    st.image(uploaded_file, caption="Ảnh đã chọn", width=300)
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    input_tensor = preprocess_image(img_bgr)
    model = load_model()
    prediction = model.predict(input_tensor)[0]
    pred_index = np.argmax(prediction)
    confidence = prediction[pred_index]

    st.markdown(f"<h2 style='text-align:center;'>🔤 Dự đoán: <span style='color:#4CAF50'>{labels[pred_index]}</span> ({confidence:.2f})</h2>", unsafe_allow_html=True)

    # Vẽ biểu đồ dự đoán
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(labels, prediction, color='skyblue')
    bars[pred_index].set_color('green')
    ax.set_title("Phân bố xác suất dự đoán", fontsize=16)
    ax.set_ylabel("Xác suất")
    ax.set_xticklabels(labels, rotation=45)
    st.pyplot(fig)
