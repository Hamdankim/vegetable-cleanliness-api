# Deployment Guide

## Opsi 1: Railway (Recommended - Gratis & Mudah)

### Setup Awal
1. **Sign up** di [Railway.app](https://railway.app) dengan GitHub
2. **New Project** → **Deploy from GitHub repo**
3. Pilih repository `vegetable-cleanliness-api`
4. Railway otomatis deteksi `Dockerfile` dan `railway.json`

### Environment Variables
Di Railway dashboard, tambahkan:
```
COLOR_MODE=strict
ALLOWED_ORIGINS=https://your-flutter-app.com,http://localhost:*
PORT=8000
```

### Auto Deploy
- Setiap push ke `main` branch otomatis trigger deploy
- Railway generate domain: `https://vegetable-cleanliness-api.up.railway.app`
- Bisa tambah custom domain di Settings → Domains

### Monitor
- Dashboard Railway: lihat logs, metrics, restart service
- Health check: otomatis via `/healthz`

---

## Opsi 2: Render

### Setup Awal
1. **Sign up** di [Render.com](https://render.com)
2. **New Web Service** → Connect GitHub repository
3. Render deteksi `render.yaml` untuk auto-config

### Blueprint (render.yaml sudah ada)
- Service name: `vegetable-cleanliness-api`
- Region: Singapore (free tier)
- Auto-deploy dari `main` branch
- Free tier: domain `https://vegetable-cleanliness-api.onrender.com`

### Deploy Hook (Opsional untuk CI/CD)
1. Di Render dashboard → Settings → Deploy Hook
2. Copy URL
3. Tambahkan ke GitHub Secrets: `RENDER_DEPLOY_HOOK_URL`

---

## Opsi 3: GitHub Container Registry + VPS

### Build & Push Manual
```bash
# Login ke GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Build & push
docker build -t ghcr.io/hamdankim/vegetable-cleanliness-api:latest .
docker push ghcr.io/hamdankim/vegetable-cleanliness-api:latest
```

### Pull & Run di VPS
```bash
# Di VPS (Ubuntu/Debian)
docker pull ghcr.io/hamdankim/vegetable-cleanliness-api:latest
docker run -d --name veg-api --restart unless-stopped \
  -p 8000:8000 \
  -e COLOR_MODE=strict \
  -e ALLOWED_ORIGINS='*' \
  ghcr.io/hamdankim/vegetable-cleanliness-api:latest
```

### Nginx Reverse Proxy (opsional HTTPS)
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Setup SSL dengan certbot:
```bash
sudo certbot --nginx -d api.yourdomain.com
```

---

## GitHub Actions Workflow

File `.github/workflows/deploy.yml` sudah dibuat dengan fitur:

### Auto Build & Push ke GHCR
- Trigger: push ke `main`, PR, atau manual
- Build Docker image dengan cache optimization
- Push ke GitHub Container Registry
- Tag: `latest`, `main-{sha}`, branch name

### Auto Deploy Hook
- **Render**: trigger via deploy hook (set secret `RENDER_DEPLOY_HOOK_URL`)
- **Railway**: otomatis deploy saat push ke main (no secret needed)

### Secrets yang Diperlukan
Di GitHub Settings → Secrets and variables → Actions:

- `RENDER_DEPLOY_HOOK_URL` (opsional, jika pakai Render)
- `GITHUB_TOKEN` (otomatis tersedia, untuk GHCR)

---

## Testing Deployment

### Health Check
```bash
curl https://your-deployed-url.com/healthz
```

### Predict Test
```bash
curl -X POST https://your-deployed-url.com/predict \
  -F "file=@test-image.jpg"
```

### Debug Endpoint
```bash
curl -X POST https://your-deployed-url.com/predict-debug \
  -F "file=@test-image.jpg"
```

---

## Flutter Integration

Update base URL di Flutter app:

```dart
class ApiConfig {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'https://vegetable-cleanliness-api.up.railway.app',
  );
}

class ApiClient {
  final Dio _dio = Dio(BaseOptions(baseUrl: ApiConfig.baseUrl));
  
  Future<PredictionResult> predict(String imagePath) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(imagePath, filename: 'image.jpg'),
    });
    final res = await _dio.post('/predict', data: formData);
    return PredictionResult.fromJson(res.data);
  }
}
```

Build dengan env:
```bash
flutter build apk --dart-define=API_BASE_URL=https://your-api.com
```

---

## Monitoring & Troubleshooting

### Logs
- **Railway**: Dashboard → Deployments → View Logs
- **Render**: Dashboard → Logs tab
- **VPS**: `docker logs -f veg-api`

### Common Issues

**502 Bad Gateway**
- Check container is running: `docker ps`
- Check logs: `docker logs veg-api`
- Verify port binding: container listens on `0.0.0.0:8000`

**CORS errors in Flutter**
- Update `ALLOWED_ORIGINS` env variable
- Production: set specific domains (no wildcard `*`)

**Model version mismatch**
- Warning normal (sklearn 1.6 → 1.7)
- Retrain atau pin `scikit-learn==1.6.1` di `requirements.txt`

**Slow cold starts (Render free tier)**
- Free tier spins down after 15min inactivity
- First request = ~30s startup
- Solution: upgrade plan atau use Railway (no spin down)

---

## Cost Estimasi

- **Railway**: Free $5 credit/month (~550 hours), then $0.000231/min
- **Render**: Free tier (spins down), $7/month for always-on
- **VPS** (DigitalOcean/Hetzner): $4-6/month for 1GB RAM

Untuk production (low-medium traffic): **Railway** recommended.
