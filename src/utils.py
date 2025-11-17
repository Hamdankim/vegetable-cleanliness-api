import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def preprocess(image):
    img_resized = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return img_resized, gray, blur

def extract_glcm_features(gray):
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)

    features = [
        graycoprops(glcm, 'contrast')[0][0],
        graycoprops(glcm, 'homogeneity')[0][0],
        graycoprops(glcm, 'energy')[0][0],
        graycoprops(glcm, 'correlation')[0][0],
    ]
    return features
