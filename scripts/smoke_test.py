import argparse
import json
import os

import cv2
import joblib
import numpy as np

from app.pipeline import build_features


def main():
    parser = argparse.ArgumentParser(description="Smoke test prediction")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default=os.path.join("models", "svm_model.pkl"))
    parser.add_argument("--color-mode", default=os.getenv("COLOR_MODE", "strict"))
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    clf = joblib.load(args.model)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    X = build_features(img, color_mode=args.color_mode)
    proba = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    idx_to_label = {0: "bersih", 1: "kotor"}
    res = {
        "label": idx_to_label.get(pred_idx, str(pred_idx)),
        "confidence": float(proba[pred_idx]),
        "probabilities": {idx_to_label[i]: float(p) for i, p in enumerate(proba)},
    }
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
