# Deployment & Git Guide - Vision Studio

## 1. Pushing to GitHub

The repository is already initialized and configured with a `.gitignore` that safely excludes heavy weight files (`*.pt`, `*.onnx`), SQLite databases, uploaded datasets, and `node_modules`.

```bash
# 1. Add your GitHub repository remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 2. Rename branch to main (if needed)
git branch -M main

# 3. Push your code to GitHub
git push -u origin main
```

---

## 2. Deploying Frontend to Vercel

The React frontend (`frontend/`) is powered by Vite and is ready for Vercel deployment.

### Option A: Import Monorepo Root directly into Vercel
Thanks to the root `vercel.json`, you can import your GitHub repo directly into Vercel without configuring directory overrides:
- **Build Command**: `cd frontend && npm run build` (auto-detected via `vercel.json`)
- **Output Directory**: `frontend/dist` (auto-detected via `vercel.json`)

### Option B: Set Root Directory to `frontend`
In the Vercel Project Creation screen:
1. Set **Root Directory** to `frontend`
2. Framework Preset: **Vite**
3. Build Command: `npm run build`
4. Output Directory: `dist`
5. **Environment Variables**:
   - `VITE_API_BASE_URL`: The public URL of your deployed Python FastAPI backend (e.g. `https://your-api.onrender.com` or `https://your-backend.railway.app`).

---

## 3. Deploying Python Backend (FastAPI + PyTorch)

Because model training requires Python, PyTorch, and disk access for datasets and checkpoints, host the backend on a Python-capable host such as **Render**, **Railway**, **Fly.io**, or an **AWS / DigitalOcean VPS**.

### Deploying to Render / Railway
1. Create a **Web Service** pointing to this repository.
2. Build Command:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Command:
   ```bash
   python -m app run --host 0.0.0.0 --port $PORT
   ```
4. Set Environment Variables:
   - `CORS_ORIGINS`: `["https://your-project.vercel.app","http://localhost:3000"]`
   - `SECRET_KEY`: `<Generate random 32-char secret>`
   - `APP_ENV`: `production`

### Deploying via Docker (NVIDIA GPU or CPU)
Run directly with Docker Compose:
```bash
docker compose up -d --build
```
Or build the Docker container:
```bash
docker build -t ai-vision-studio -f docker/Dockerfile .
docker run -p 8000:8000 --gpus all ai-vision-studio
```

---

## 4. Running Locally

### Development Mode:
- **Backend**:
  ```bash
  python -m app run --host 0.0.0.0 --port 8000
  ```
- **React Frontend (with Hot Reload & API Proxy)**:
  ```bash
  cd frontend
  npm run dev
  ```
  Open `http://localhost:3000` in your browser.

### Integrated Mode (FastAPI serves built React bundle):
```bash
cd frontend && npm run build
cd ..
python -m app run --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.
