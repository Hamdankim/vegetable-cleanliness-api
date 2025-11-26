import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def grabcut_segment(img: np.ndarray):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    height, width = img.shape[:2]
    rect = (10, 10, width - 20, height - 20)

    cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    segmented = img * final_mask[:, :, np.newaxis]
    return segmented, final_mask


def extract_color_features(img: np.ndarray, color_mode: str = "strict"):
    # strict: correctly convert BGR->RGB->HSV (recommended after retraining)
    # compat: legacy mode mimicking original notebook (RGB->HSV on BGR array)
    if color_mode == "strict":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    features = {
        "H_mean": float(np.mean(H)),
        "H_std": float(np.std(H)),
        "S_mean": float(np.mean(S)),
        "S_std": float(np.std(S)),
        "V_mean": float(np.mean(V)),
        "V_std": float(np.std(V)),
    }
    return features


def extract_glcm(img_gray: np.ndarray):
    glcm = graycomatrix(
        img_gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "ASM": float(graycoprops(glcm, "ASM").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
    }
    return features


def build_features(bgr_img: np.ndarray, color_mode: str = "strict") -> np.ndarray:
    seg, _ = grabcut_segment(bgr_img)
    color_feats = extract_color_features(seg, color_mode=color_mode)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    glcm_feats = extract_glcm(gray)
    feats = list(color_feats.values()) + list(glcm_feats.values())
    return np.array(feats, dtype=np.float32).reshape(1, -1)
