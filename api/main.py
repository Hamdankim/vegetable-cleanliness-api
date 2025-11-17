from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import joblib
from src.utils import preprocess, extract_glcm_features

app = FastAPI()

model = joblib.load("models/svm_model.pkl")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    _, gray, _ = preprocess(img)
    features = extract_glcm_features(gray)

    pred = model.predict([features])[0]

    return {
        "prediction": int(pred),
        "kelas": "Bersih" if pred == 0 else "Kotor"
    }
