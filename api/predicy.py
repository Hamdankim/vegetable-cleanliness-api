import cv2
import joblib
import numpy as np
from src.utils import preprocess, extract_glcm_features

model = joblib.load("models/svm_model.pkl")

def predict_image(image_path):
    img = cv2.imread(image_path)
    _, gray, _ = preprocess(img)
    features = extract_glcm_features(gray)
    pred = model.predict([features])[0]
    return "Bersih" if pred == 0 else "Kotor"
