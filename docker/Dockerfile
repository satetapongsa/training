# =======================================================
# AI Vision Training Studio - Production Dockerfile
# =======================================================
FROM python:3.11-slim AS base

# System dependencies for OpenCV, PyTorch, and WebRTC/Sockets
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create persistent storage directories
RUN mkdir -p /app/data/uploads \
             /app/data/datasets \
             /app/data/runs \
             /app/data/models \
             /app/data/exports

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["python", "-m", "app", "run", "--host", "0.0.0.0", "--port", "8000"]
