import cv2
import numpy as np

def preprocess_image(img, img_size=224):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray_img)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_img = cv2.filter2D(contrast_img, -1, sharp_kernel)
    norm_img = cv2.resize(sharp_img, (img_size, img_size))
    norm_img = norm_img.astype(np.float32) / 255.0
    return np.expand_dims(norm_img, axis=(0, -1))  # shape: (1, 224, 224, 1)
