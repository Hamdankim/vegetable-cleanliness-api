import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def preprocess_image(img: np.ndarray, size=(256, 256), blur='gaussian', ksize=3) -> np.ndarray:
    """
    Preprocessing sesuai training pipeline:
    1. Blur (gaussian/median/mean)
    2. Resize
    3. Normalize (0-1)
    """
    # 1. BLUR
    if blur == 'gaussian':
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif blur == 'median':
        img = cv2.medianBlur(img, ksize)
    elif blur == 'mean':
        img = cv2.blur(img, (ksize, ksize))
    
    # 2. RESIZE
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # 3. NORMALIZE (0-1)
    img = img.astype(np.float32) / 255.0
    
    return img


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
    # 1. PREPROCESSING (blur + resize + normalize) - SAMA SEPERTI TRAINING
    preprocessed = preprocess_image(bgr_img, size=(256, 256), blur='gaussian', ksize=3)
    
    # Convert back to uint8 for segmentation (0-255)
    img_uint8 = (preprocessed * 255).astype(np.uint8)
    
    # 2. SEGMENTATION
    seg, _ = grabcut_segment(img_uint8)
    
    # 3. FEATURE EXTRACTION
    color_feats = extract_color_features(seg, color_mode=color_mode)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    glcm_feats = extract_glcm(gray)
    
    feats = list(color_feats.values()) + list(glcm_feats.values())
    return np.array(feats, dtype=np.float32).reshape(1, -1)
