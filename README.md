# Vision Training Studio: Production Deep Learning System

![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TorchVision-EE4C2C)
![Ultralytics](https://img.shields.io/badge/Backbone-YOLO11%20%7C%20ResNet-00B4D8)
![FastAPI](https://img.shields.io/badge/Gateway-FastAPI%20%7C%20ASGI-009688)
![React](https://img.shields.io/badge/Frontend-React%2019%20%7C%20Vite-61DAFB)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 1. System Overview

Vision Training Studio is an asynchronous, production-grade Computer Vision platform engineered for training, evaluating, optimizing, and deploying deep convolutional and transformer-based neural network models.

The system features a zero-mock execution paradigm: all interactions trigger genuine PyTorch backpropagation passes, dataset compilation routines, real-time WebSocket telemetry streams, and hardware-accelerated inference graphs.

```
+---------------------------------------------------------------------------------------+
|                                    SYSTEM TOPOLOGY                                    |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|   [ Client Layer ]                                                                    |
|   +-------------------------------------------------------------------------------+   |
|   | React 19 Single Page Application (Light Slate Theme, Zero External Dependencies) |   |
|   | - HTML5 Canvas Bounding Box Engine (Real-time Normalization [0.0, 1.0])       |   |
|   | - WebSocket Live Telemetry Consumer (Loss Curves, Latency, Hardware VRAM)     |   |
|   +---------------------------------------+---------------------------------------+   |
|                                           | REST API / Full-Duplex WebSockets         |
|   [ Gateway Layer ]                       v                                           |
|   +-------------------------------------------------------------------------------+   |
|   | FastAPI / Starlette Asynchronous ASGI Application Container                  |   |
|   | - Pydantic v2 Request/Response Validation Schema                              |   |
|   | - Chunked Multipart File Streamer & SHA-256 Ingestion Dedup Engine            |   |
|   +---------------------------------------+---------------------------------------+   |
|                                           | Native Threading / Multiprocessing        |
|   [ Compute & Training Engine ]           v                                           |
|   +-------------------------------------------------------------------------------+   |
|   | Deep Learning Execution Core                                                  |   |
|   | - Ultralytics YOLO11 (Detection / Segmentation) & PyTorch Backbones          |   |
|   | - CUDA / MPS / CPU Execution Dispatcher with Automatic Mixed Precision (AMP)  |   |
|   | - Dataset Sanitizer, Magic-Byte Validator & Stratified Splitter               |   |
|   +---------------------------------------+---------------------------------------+   |
|                                           | Checkpoints & Optimization                |
|   [ Artifact & Serving Layer ]            v                                           |
|   +-------------------------------------------------------------------------------+   |
|   | - Checkpoint Manager (best.pt, last.pt) with State Dict Pruning               |   |
|   | - ONNX Operator Export Engine (Opset 17, Constant Folding, FP16 Dynamic)     |   |
|   | - High-Throughput REST Inference Server with Non-Maximum Suppression (NMS)     |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation & Algorithmic Foundations

### 2.1 Coordinate Normalization & Transformation

Given an input raster image $I \in \mathbb{R}^{H \times W \times C}$ with pixel-space coordinates for a bounding box:

$$\mathbf{B}_{\text{pixel}} = [x_{\min}, y_{\min}, x_{\max}, y_{\max}], \quad \text{where } 0 \le x_{\min} < x_{\max} \le W, \ 0 \le y_{\min} < y_{\max} \le H$$

The client and ingestion engine map pixel coordinates to the canonical normalized bounding format $\mathbf{B}_{\text{norm}} = [x_c, y_c, w, h] \in [0, 1]^4$:

$$x_c = \frac{x_{\min} + x_{\max}}{2W}, \quad y_c = \frac{y_{\min} + y_{\max}}{2H}$$

$$w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$

Inverse transformation for post-inference canvas rendering:

$$x_{\min} = \left(x_c - \frac{w}{2}\right) \times W, \quad x_{\max} = \left(x_c + \frac{w}{2}\right) \times W$$

$$y_{\min} = \left(y_c - \frac{h}{2}\right) \times H, \quad y_{\max} = \left(y_c + \frac{h}{2}\right) \times H$$

### 2.2 Objective Loss Function

The training loop optimizes a multi-task composite loss function:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$

#### Complete Intersection over Union (CIoU) Loss
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v$$

Where:
- $\rho(\mathbf{b}, \mathbf{b}^{gt})$ represents Euclidean distance between central points.
- $c$ represents diagonal length of the smallest enclosing bounding box.
- $v = \frac{4}{\pi^2} \left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$ measures aspect ratio consistency.
- $\alpha = \frac{v}{(1 - \text{IoU}) + v}$ dynamically balances the aspect ratio penalty.

#### Binary Cross-Entropy with Logits (BCE) Classification Loss
$$\mathcal{L}_{\text{cls}} = -\sum_{i=1}^C \left[ y_i \log(\sigma(\hat{y}_i)) + (1 - y_i)\log(1 - \sigma(\hat{y}_i)) \right]$$

#### Distribution Focal Loss (DFL)
$$\mathcal{L}_{\text{DFL}}(S_i, S_{i+1}) = - \left( (y_{i+1} - y)\log(S_i) + (y - y_i)\log(S_{i+1}) \right)$$

---

## 3. Core Architecture & Modules

### 3.1 Data Ingestion & Sanitization Engine (`app/datasets/`)
- **Magic-Byte Verification**: Asserts MIME types at byte-level header signatures (JPEG: `FF D8 FF`, PNG: `89 50 4E 47 0D 0A 1A 0A`, WEBP: `52 49 46 46`).
- **Cryptographic Deduplication**: Computes SHA-256 digests on incoming image buffers to prevent redundant training samples and validation set leakage.
- **Stratified Partitioning**: Generates reproducible partitions ($\text{Train} = 70\%$, $\text{Val} = 20\%$, $\text{Test} = 10\%$) preserving target class distribution ratios across splits:

$$P(\text{class}_k \mid \mathcal{D}_{\text{train}}) \approx P(\text{class}_k \mid \mathcal{D}_{\text{val}}) \approx P(\text{class}_k \mid \mathcal{D}_{\text{test}})$$

### 3.2 Training Supervisor & Telemetry Stream (`app/training/`)
- **Asynchronous Execution Threading**: Dedicated background task workers decouple deep learning execution loops from the main ASGI event loop, guaranteeing HTTP server responsiveness.
- **WebSocket Telemetry Broadcasting**: Training metrics (epoch, batch index, box loss, class loss, dfl loss, mAP@50, mAP@50:95) are captured per iteration via callback hooks and pushed to connected clients at 100ms throttle intervals.
- **Graceful Preemption**: Implements clean signal handlers enabling cancellation with deterministic state-checkpoint persistence.

### 3.3 Model Optimization & ONNX Runtime Export (`app/models/`)
- **ONNX Graph Transformation**: Executes torch ONNX compilation targeting Opset 17 with static/dynamic axis bindings.
- **Graph Optimization**: Constant folding, redundant reshape elimination, and operator fusion for edge accelerator deployment.
- **Quantization Pipeline**: Provides 8-bit dynamic integer quantization (INT8) and half-precision floating point (FP16) reducing memory footprint by up to 75%.

---

## 4. Directory Structure

```
ai_vision_studio/
|-- app/
|   |-- __init__.py
|   |-- __main__.py               # CLI entrypoint router
|   |-- api/
|   |   |-- __init__.py
|   |   |-- routes.py             # REST API endpoint handlers
|   |   |-- websocket.py          # Real-time WebSocket connection manager
|   |-- core/
|   |   |-- config.py             # App settings, environment variables
|   |   |-- database.py           # SQLite/PostgreSQL engine and session factory
|   |-- datasets/
|   |   |-- ingestion.py          # Folder scanning, multipart streaming, validation
|   |   |-- splitter.py           # Stratified train/val/test splitting
|   |-- models/
|   |   |-- registry.py           # Model metadata, checkpoint catalog
|   |   |-- exporter.py           # ONNX, TorchScript export routines
|   |-- training/
|   |   |-- trainer.py            # YOLO11 / PyTorch training executor
|   |   |-- supervisor.py         # Thread manager, live progress observer
|   |-- inference/
|   |   |-- predictor.py          # Bounding box inference, NMS post-processing
|-- frontend/
|   |-- src/
|   |   |-- App.jsx               # Root application component
|   |   |-- index.css             # Light Slate Design System (Tokens, CSS Variables)
|   |   |-- api/
|   |   |   |-- client.js         # Axios / Fetch client and WebSocket constructor
|   |   |-- components/
|   |   |   |-- Sidebar.jsx       # 4-Item Minimalist Navigation (SVG Vector Icons)
|   |   |   |-- Topbar.jsx        # Lightweight status bar with connection badge
|   |   |-- views/
|   |   |   |-- StudioView.jsx    # Primary Folder Upload & Interactive Canvas Studio
|   |   |   |-- TrainingView.jsx  # Hyperparameters Configuration & Real-time Curves
|   |   |   |-- InferenceView.jsx # Latency benchmark & visual inference validation
|   |   |   |-- ModelsView.jsx    # Checkpoints catalog & ONNX download exporter
|-- configs/
|   |-- train_template.yaml       # Declarative training pipeline configuration
|-- tests/
|   |-- test_api.py               # REST API route contract tests
|   |-- test_dataset.py           # Bounding box normalization and dataset validator tests
|   |-- test_training_e2e.py      # End-to-end PyTorch training and export pipeline tests
|-- Dockerfile                    # Multi-stage container build definition
|-- docker-compose.yml            # Container orchestration specification
|-- requirements.txt              # Production Python dependencies
|-- README.md                     # Technical architecture and user guide
```

---

## 5. API Reference & Communication Protocols

### 5.1 REST Endpoints

| Method | Route | Description | Request Body / Form |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/projects` | List all active projects | None |
| `POST` | `/api/v1/projects` | Initialize a new project | `{"name": str, "task_type": str}` |
| `POST` | `/api/v1/datasets/upload-folder` | Chunked multipart folder ingestion | `multipart/form-data` |
| `GET` | `/api/v1/datasets/{id}/images` | Paginated dataset image retrieval | Query: `?page=1&limit=50` |
| `POST` | `/api/v1/annotations/{image_id}` | Bulk upsert bounding boxes | `[{"label": str, "x_min": float, ...}]` |
| `POST` | `/api/v1/training/start` | Launch asynchronous training loop | `{"dataset_id": int, "config": {...}}` |
| `GET` | `/api/v1/training/status/{run_id}` | Poll training execution state | None |
| `POST` | `/api/v1/training/cancel/{run_id}` | Terminate active training loop | None |
| `POST` | `/api/v1/inference/predict` | Synchronous bounding box inference | `multipart/form-data` (image file) |
| `POST` | `/api/v1/models/{id}/export` | Trigger ONNX graph compilation | `{"format": "onnx", "opset": 17}` |
| `GET` | `/api/v1/models/{id}/download` | Download compiled binary weights | Query: `?format=pt` or `?format=onnx` |

### 5.2 Real-Time WebSocket Telemetry Protocol

**Endpoint**: `/ws/live`

#### Telemetry Packet Schema
```json
{
  "type": "training_progress",
  "data": {
    "run_id": "run_1725528000",
    "epoch": 3,
    "total_epochs": 10,
    "step": 45,
    "total_steps": 150,
    "train_loss": 0.0412,
    "val_loss": 0.0385,
    "map50": 0.912,
    "map50_95": 0.748,
    "learning_rate": 0.0034,
    "vram_allocated_mb": 2048.5,
    "log": "Epoch 3/10: train_loss=0.0412, val_loss=0.0385, mAP@50=0.912"
  }
}
```

---

## 6. Installation & Quickstart

### 6.1 Prerequisites
- Python 3.11, 3.12, or 3.14
- Node.js 18+ and npm (for frontend building)
- (Optional) NVIDIA GPU with CUDA Compute Capability 7.0+ and cuDNN

### 6.2 Environment Setup
```bash
# Clone the repository
git clone https://github.com/satetapongsa/training.git
cd training

# Initialize and activate Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 6.3 Building the Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 6.4 Launching the Service
```bash
python -m app run --host 0.0.0.0 --port 8000
```

Open your browser at:
- **Application Interface**: `http://localhost:8000`
- **Swagger OpenAPI Documentation**: `http://localhost:8000/docs`
- **ReDoc Technical Specification**: `http://localhost:8000/redoc`

---

## 7. Command Line Interface (CLI)

The application provides a comprehensive CLI for headless operation:

```bash
# Start Web Server & UI Gateway
python -m app run --host 0.0.0.0 --port 8000

# Execute Headless Training Pipeline from YAML Configuration
python -m app train --config configs/train_template.yaml

# Perform Deep Dataset Validation & Verification
python -m app validate-dataset --dataset ./data/datasets/sample_dataset

# Execute Offline Image Inference
python -m app predict \
  --model ./data/runs/exp1/checkpoints/best.pt \
  --source ./data/test.jpg \
  --output ./detections.jpg

# Export Model Checkpoint to ONNX Format
python -m app export \
  --model ./data/runs/exp1/checkpoints/best.pt \
  --format onnx \
  --opset 17
```

---

## 8. Python SDK Usage

```python
from app.inference import Predictor

# Instantiate predictor from checkpoint weights
predictor = Predictor(model_path="data/runs/exp1/checkpoints/best.pt", device="cuda:0")

# Run inference on target raster
result = predictor.predict(
    image_path="sample.jpg",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Inspect detected bounding boxes
for detection in result.detections:
    print(f"Class: {detection.class_name} | Confidence: {detection.confidence:.3f} | BBox: {detection.box}")

# Save visual rendering
result.save("annotated_output.jpg")
```

---

## 9. Automated Testing & Verification

The repository includes a comprehensive automated test suite covering unit, integration, and end-to-end training loops:

```bash
# Execute entire test suite
python -m pytest tests/ -v

# Execute end-to-end training and ONNX export integration test
python -m pytest tests/test_training_e2e.py -v -s
```

---

## 10. Deployment

### 10.1 Docker Container Deployment
```bash
# Build and start containerized stack
docker compose up -d --build

# Monitor container logs
docker compose logs -f
```

### 10.2 Vercel Deployment (Frontend Interface)
The `frontend/` directory is pre-configured with Vite for seamless static edge deployment on Vercel:
```bash
cd frontend
vercel deploy --prod
```
Configure `VITE_API_URL` to point to your backend FastAPI host.

---

## 11. Technical Specifications Summary

- **Backbone Architectures**: Ultralytics YOLO11 (Nano, Small, Medium), ResNet-18
- **Annotation Format**: Standard YOLO format (`<class_id> <x_center> <y_center> <width> <height>`)
- **Export Targets**: PyTorch (`.pt`), Open Neural Network Exchange (`.onnx`), TorchScript (`.torchscript`)
- **Transport Layer**: HTTP/1.1 REST API + Asynchronous WebSocket Telemetry (`/ws/live`)
- **UI Architecture**: React 19 SPA, Clean Slate Light Theme, Zero Emojis, Native HTML5 Canvas
