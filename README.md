# AI Vision Training Studio ⚡

An enterprise-grade, end-to-end Computer Vision platform for training, evaluating, exporting, and serving object detection and image classification models.

Built with **Python 3.11+ / 3.14**, **FastAPI**, **PyTorch**, **TorchVision**, and **Ultralytics YOLO**, with real training execution, zero mock data, real-time WebSocket telemetry, interactive YOLO annotation canvas, and seamless ONNX/TorchScript export.

---

## 🌟 Key Features

- **Real Zero-Mock ML Pipeline**: Every button executes genuine PyTorch and Ultralytics routines. Checkpoints (`best.pt`, `last.pt`), real loss curves, and actual bounding box inference.
- **Interactive Annotation Studio**: In-browser YOLO annotation canvas with Pan (Space+drag), Zoom (mouse wheel), Bounding Box drawing, resize handles, and hotkey shortcuts.
- **Deep Dataset Validator**: Inspects magic bytes, detects corrupted images, verifies normalized bounding box boundaries, checks class index ranges, and detects duplicate images via SHA-256.
- **Stratified Auto-Split**: Splits Train (70%), Val (20%), and Test (10%) preserving class balances without data leakage, generating standard `dataset.yaml`.
- **Live Training Telemetry**: Real-time streaming over WebSockets: Epochs, Step, Loss Curve (SVG), mAP50 / Accuracy, learning rate, and CPU/RAM/GPU VRAM meters.
- **Model Registry & Provenance**: Tracks architecture, dataset version, hyperparameters, loss metrics, and confusion metrics with one-click ONNX export.
- **Inference Playground & Python SDK**: Test models on uploaded images with real bounding boxes and confidence badges, with a clean Python SDK:
  ```python
  from app.inference import Predictor
  predictor = Predictor("data/runs/my_project/exp1/checkpoints/best.pt")
  result = predictor.predict("sample.jpg", conf_threshold=0.25)
  result.save("detection_result.jpg")
  ```
- **CLI Subcommands**: Full command-line interface for training, validation, prediction, export, and running the server.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.11+ (Tested on Python 3.14)
- (Optional) NVIDIA CUDA GPU for accelerated training (automatically falls back to CPU if absent).

### 2. Installation
Clone or navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch Web Studio & REST API
```bash
python -m app run --port 8000
```
Then open your browser at:
- **Web Studio UI**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💻 CLI Commands

```bash
# Start Web Server & UI
python -m app run --port 8000

# Train a model directly via CLI config
python -m app train --config configs/train_template.yaml

# Validate a dataset folder
python -m app validate-dataset --dataset ./data/datasets/animal_dataset

# Run bounding box inference
python -m app predict --model ./data/runs/proj/exp/checkpoints/best.pt --source ./test.jpg --output ./out

# Export weights to ONNX
python -m app export --model ./best.pt --format onnx
```

---

## 🧪 Running Automated Tests

Run the complete test suite (17+ unit & integration tests):
```bash
python -m pytest tests/ -v
```

Run the real end-to-end training and ONNX export test:
```bash
python -m pytest tests/test_training_e2e.py -v -s
```

---

## 🐳 Docker Deployment

### Run with Docker Compose:
```bash
docker compose up -d --build
```
Access the studio at `http://localhost:8000`.

---

## 📚 Documentation Links
- [System Architecture](docs/ARCHITECTURE.md)
- [API Documentation](docs/API.md)
- [Model Training Guide](docs/MODEL_TRAINING.md)
- [Dataset Ingestion & Annotation](docs/DATASET_GUIDE.md)
- [Development Workflow](docs/DEVELOPMENT.md)
- [Production Deployment](docs/DEPLOYMENT.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
