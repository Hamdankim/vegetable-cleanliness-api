import os
import cv2
import joblib
from sklearn.svm import SVC
from src.utils import preprocess, extract_glcm_features

DATA_DIR = "data/raw/"
MODEL_PATH = "models/svm_model.pkl"

X = []
y = []

# Kelas = folder
classes = {
    "bersih": 0,
    "kotor": 1
}

print("📌 Mulai loading data dari folder bersih/ dan kotor/...")

for class_name, label in classes.items():
    folder_path = os.path.join(DATA_DIR, class_name)

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder tidak ditemukan: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is None:
            print("⚠️ Gagal membaca:", img_path)
            continue

        # Preprocess
        _, gray, _ = preprocess(img)

        # GLCM feature extraction
        features = extract_glcm_features(gray)

        X.append(features)
        y.append(label)

print(f"📁 Total data: {len(X)}")
print(f"🟩 Kelas bersih: {y.count(0)}")
print(f"🟥 Kelas kotor : {y.count(1)}")

if len(set(y)) < 2:
    raise ValueError("❌ ERROR: Dataset hanya memiliki 1 kelas! Pastikan folder bersih/ dan kotor/ berisi gambar.")

print("\n📌 Mulai training model SVM...")

model = SVC(kernel="linear")
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\n✅ Training selesai!")
print(f"💾 Model disimpan di: {MODEL_PATH}")
