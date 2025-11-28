FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# System deps for OpenCV headless and scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Railway injects PORT env, default 8000 for local
ENV PORT=8000

EXPOSE 8000

# Use $PORT from Railway, bind to 0.0.0.0, limit to 1 worker to prevent memory overflow
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
