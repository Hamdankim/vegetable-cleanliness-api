import io
import os
from typing import Dict

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .pipeline import build_features


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "svm_model.pkl")
    alt_path = os.path.join(os.getcwd(), "models", "svm_model.pkl")
    path = model_path if os.path.exists(model_path) else alt_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


app = FastAPI(title="Vegetable Cleanliness API", version="1.0.0")

# CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model at startup
MODEL = load_model()
IDX_TO_LABEL = {0: "bersih", 1: "kotor"}
COLOR_MODE = os.getenv("COLOR_MODE", "strict").strip().lower()


@app.get("/")
def root():
    return {"status": "ok", "service": "Vegetable Cleanliness API", "version": "1.0.0"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image_array = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        X = build_features(img, color_mode=COLOR_MODE)
        proba = MODEL.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        label = IDX_TO_LABEL.get(pred_idx, str(pred_idx))
        probs = {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(proba)}

        return PredictResponse(
            label=label,
            confidence=float(proba[pred_idx]),
            probabilities=probs,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-debug")
async def predict_debug(file: UploadFile = File(...)):
    try:
        content = await file.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        X = build_features(img, color_mode=COLOR_MODE)
        proba = MODEL.predict_proba(X)[0].tolist()
        pred_idx = int(np.argmax(proba))
        return {
            "color_mode": COLOR_MODE,
            "features": X.flatten().tolist(),
            "proba_order": [IDX_TO_LABEL[i] for i in range(len(proba))],
            "proba": proba,
            "pred_idx": pred_idx,
            "pred_label": IDX_TO_LABEL.get(pred_idx, str(pred_idx)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
