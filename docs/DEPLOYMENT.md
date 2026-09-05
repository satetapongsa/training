# Production Deployment Guide - AI Vision Training Studio

## 1. Local Machine / On-Premise GPU Server
To run directly on an on-premise server:
```bash
# Clone repository
cd ai_vision_studio

# Install production dependencies
pip install -r requirements.txt

# Run server with Uvicorn
python -m app run --host 0.0.0.0 --port 8000
```

---

## 2. Docker with NVIDIA GPU Acceleration

### Prerequisites:
- Docker 24.0+
- NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- NVIDIA Driver installed on host

### Step 1: Verify NVIDIA Runtime
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Step 2: Launch via Docker Compose
```bash
docker compose up -d --build
```
The Docker container automatically mounts host GPUs and runs the production training worker.

---

## 3. Storage & Database Scaling
- **Database**:
  To switch from SQLite to PostgreSQL, set:
  ```env
  DATABASE_URL=postgresql+asyncpg://user:password@postgres-host:5432/ai_vision_studio
  ```
- **Storage Layer**:
  The `StorageBackend` abstraction in `app/storage/base.py` can be extended with an `S3Storage` backend for AWS S3 or MinIO buckets.
