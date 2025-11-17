# Vegetable Cleanliness Detection API

Model Machine Learning (GLCM + SVM) untuk mendeteksi apakah terong atau tomat dalam kondisi:
- Bersih
- Kotor

## Struktur Folder
- data/raw → gambar asli
- data/processed → gambar hasil preprocessing (blur/grayscale)
- src/ → notebook modelling + utils.py
- models/ → svm_model.pkl (hasil training)
- api/ → FastAPI backend

## Cara Menjalankan
1. Install dependency:
pip install -r requirements.txt
2. Jalankan API:
uvicorn api.main:app --reload
3. Test API dengan upload gambar ke:
POST http://localhost:8000/predict