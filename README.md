# Vegetable Cleanliness Detection

Sistem deteksi kebersihan sayuran menggunakan Computer Vision dan Machine Learning dengan teknik segmentasi GrabCut, ekstraksi fitur warna (HSV) dan tekstur (GLCM), serta klasifikasi menggunakan Support Vector Machine (SVM).

## 📋 Deskripsi Project

Project ini mengimplementasikan sistem klasifikasi otomatis untuk mendeteksi kebersihan sayuran menjadi dua kategori:
- **Bersih**: Sayuran yang bersih dan siap dikonsumsi
- **Kotor**: Sayuran yang kotor dan perlu dicuci

## 🏗️ Struktur Project

```
vegetable-cleanliness-api/
├── PCVK_Kelompok3.ipynb    # Notebook utama
├── data/
│   ├── raw/                # Dataset asli
│   │   ├── bersih/
│   │   └── kotor/
│   └── augmented/          # Hasil preprocessing & augmentasi
│       ├── bersih/
│       └── kotor/
└── models/
    └── svm_model.pkl       # Model SVM yang tersimpan
```

## 🔧 Requirements

```bash
opencv-python-headless
scikit-image
scikit-learn
joblib
numpy
matplotlib
```

Install dependencies:
```bash
pip install opencv-python-headless scikit-image scikit-learn joblib
```

## 🚀 Pipeline Sistem

### 1. **Preprocessing & Augmentasi**
- **Blur**: Gaussian/Median/Mean blur (default: Gaussian, ksize=3)
- **Resize**: 256x256 pixels
- **Normalization**: Scale ke range 0-1
- **Augmentasi**:
  - Rotation (90°, 180°, 270°)
  - Flip (Horizontal & Vertical)
  - Brightness adjustment (+12%, -12%)

### 2. **Segmentasi (GrabCut)**
- Menggunakan algoritma **GrabCut** dari OpenCV
- Memisahkan objek sayuran dari background
- Menghasilkan mask foreground/background

### 3. **Ekstraksi Fitur**

#### a. Fitur Warna (HSV)
- Mean & Standard Deviation dari channel H, S, V
- **Total: 6 fitur**

#### b. Fitur Tekstur (GLCM)
- Contrast, Dissimilarity, Homogeneity
- Energy, ASM, Correlation
- **Total: 6 fitur**

**Total Fitur: 12 fitur** (6 warna + 6 tekstur)

### 4. **Klasifikasi (SVM)**
- **Algorithm**: Support Vector Machine (RBF kernel)
- **Parameters**: 
  - C=10
  - gamma='scale'
  - probability=True
- **Preprocessing**: StandardScaler
- **Train/Test Split**: 80/20 dengan stratified sampling

## 📊 Hasil

```
Akurasi: 100%

              precision    recall  f1-score   support
      bersih       1.00      1.00      1.00        40
       kotor       1.00      1.00      1.00        40

    accuracy                           1.00        80
   macro avg       1.00      1.00      1.00        80
weighted avg       1.00      1.00      1.00        80
```

**Confusion Matrix:**
```
[[40  0]
 [ 0 40]]
```

**Dataset:**
- Total sampel: 400 (200 bersih, 200 kotor)
- Waktu eksekusi ekstraksi fitur: ~287 detik

## 💻 Cara Penggunaan

### Di Google Colab

1. **Mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Set path project:**
```python
DRIVE_PROJECT_DIR = '/content/drive/MyDrive/vegetable-cleanliness-api'
```

3. **Jalankan semua cell** di notebook `PCVK_Kelompok3.ipynb` secara berurutan

### Prediksi Gambar Baru

```python
# Load model
import joblib
clf = joblib.load('models/svm_model.pkl')

# Load dan preprocess gambar
img = cv2.imread('path/to/image.jpg')

# Segmentasi
seg, mask = grabcut_segment(img)

# Ekstraksi fitur
feat_color = extract_color_features(seg)
gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
feat_glcm = extract_glcm(gray)

# Gabungkan fitur
features = list(feat_color.values()) + list(feat_glcm.values())
features = np.array(features).reshape(1, -1)

# Prediksi
pred_label = clf.predict(features)[0]
pred_proba = clf.predict_proba(features)[0]

result = 'BERSIH' if pred_label == 0 else 'KOTOR'
confidence = pred_proba[pred_label] * 100

print(f'Prediksi: {result} ({confidence:.2f}%)')
```

## 🌐 API Deployment (FastAPI)

Service API disediakan di folder `app/` dengan endpoint utama:

- `GET /healthz` — cek kesehatan service
- `POST /predict` — unggah file gambar (`form-data` field: `file`)

Contoh respons:
```json
{
  "label": "bersih",
  "confidence": 0.97,
  "probabilities": {"bersih": 0.97, "kotor": 0.03}
}
```

### Menjalankan Secara Lokal

```bash
python -m venv .venv
source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t vegetable-cleanliness-api:latest .
docker run --rm -p 8000:8000 \
  -e ALLOWED_ORIGINS='*' \
  -e COLOR_MODE='strict' \
  vegetable-cleanliness-api:latest
```

### CORS

Atur origin yang diizinkan via env `ALLOWED_ORIGINS` (dipisah koma). Default `*`.

### Contoh `curl` prediksi

```bash
curl -X POST \
  -F "file=@data/raw/bersih/sample.jpg" \
  http://localhost:8000/predict
```

Endpoint debug (fitur + probabilitas mentah):
```bash
curl -X POST -F "file=@data/raw/bersih/sample.jpg" http://localhost:8000/predict-debug
```

### Smoke test tanpa server

```bash
# Jalankan sebagai module agar import app.* bekerja
python -m scripts.smoke_test data/raw/bersih/sample.jpg
# atau eksplisit color mode:
python -m scripts.smoke_test data/raw/bersih/sample.jpg --color-mode strict
```

---

## 📈 Visualisasi

Notebook menyediakan visualisasi untuk:
- Hasil preprocessing & augmentasi
- Hasil segmentasi GrabCut
- Prediksi dengan confidence score

## 🔬 Metodologi

1. **Preprocessing**: Normalisasi & augmentasi data untuk meningkatkan generalisasi
2. **Segmentasi**: GrabCut untuk fokus pada objek sayuran
3. **Feature Engineering**: Kombinasi fitur warna (HSV) dan tekstur (GLCM)
4. **Classification**: SVM dengan RBF kernel untuk klasifikasi non-linear
5. **Evaluation**: Stratified cross-validation untuk hasil yang robust

## 📝 Catatan

- Model mencapai akurasi 100% pada test set (80 sampel)
- Pipeline lengkap dari preprocessing hingga prediksi
- Support untuk augmentasi data otomatis
- Visualisasi hasil untuk interpretability

## 👥 Tim Pengembang

Kelompok 3 - PCVK

## 📄 License

Project ini dibuat untuk keperluan akademik.

---

**Link Colab:** [Open in Colab](https://colab.research.google.com/github/Hamdankim/vegetable-cleanliness-api/blob/main/PCVK_Kelompok3.ipynb)