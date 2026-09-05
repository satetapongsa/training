# Vision Training Studio: Production Deep Learning System & KDel 4.0 Architecture

![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%202.14%2B-EE4C2C)
![Architecture](https://img.shields.io/badge/Architecture-KDel%204.0%20Native-blueviolet)
![Gateway](https://img.shields.io/badge/Gateway-FastAPI%20%7C%20ASGI-009688)
![Frontend](https://img.shields.io/badge/Frontend-React%2019%20%7C%20Vite-61DAFB)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 1. System Overview

Vision Training Studio is an enterprise-grade Computer Vision platform powered by our proprietary deep learning architecture **KDel 4.0**, built natively from first principles in pure PyTorch and Python.

The platform eliminates black-box third-party training dependencies by providing a native, transparent, and high-performance deep convolutional vision architecture that learns, optimizes, and evaluates object detection representations directly from raw images and ground truth annotations.

```
+---------------------------------------------------------------------------------------+
|                                    SYSTEM TOPOLOGY                                    |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|   [ Client Layer ]                                                                    |
|   +-------------------------------------------------------------------------------+   |
|   | React 19 Single Page Application (Slate Light Theme, Zero Emojis)             |   |
|   | - Direct Local Folder Ingestion (Recursive WebKit Entry Scanner)              |   |
|   | - Interactive HTML5 Canvas Ground Truth Annotation Engine                     |   |
|   | - WebSocket Live Telemetry Consumer (Loss Curves, Latency, Hardware VRAM)     |   |
|   +---------------------------------------+---------------------------------------+   |
|                                           | REST API / Full-Duplex WebSockets         |
|   [ Gateway Layer ]                       v                                           |
|   +-------------------------------------------------------------------------------+   |
|   | FastAPI / Starlette Asynchronous ASGI Application Container                  |   |
|   | - Pydantic v2 Schema Validation & Route Controllers                           |   |
|   | - Automatic Workspace Resolution & Chunked Multipart Ingestion Engine         |   |
|   +---------------------------------------+---------------------------------------+   |
|                                           | Native Threading / Multiprocessing        |
|   [ KDel 4.0 Compute & Training Core ]    v                                           |
|   +-------------------------------------------------------------------------------+   |
|   | KDel 4.0 Native Deep Learning Architecture (app/models/kdel.py)               |   |
|   | - KDelBackbone: Multi-Scale Residual Extractor + Spatial Attention Blocks    |   |
|   | - KDelPANet: Bi-directional Top-Down & Bottom-Up Feature Aggregation          |   |
|   | - KDelDecoupledHead: Multi-Scale Anchor-Free Heads (Strides 8, 16, 32)        |   |
|   | - Optimizer: AdamW with Cosine Annealing Learning Rate Scheduler              |   |
|   +---------------------------------------+---------------------------------------+   |
|                                           | Checkpoints & Optimization                |
|   [ Artifact & Serving Layer ]            v                                           |
|   +-------------------------------------------------------------------------------+   |
|   | - Checkpoint Manager (best.pt, last.pt) with State Dict Pruning               |   |
|   | - ONNX Operator Export Engine (Opset 17, Dynamic Axes)                        |   |
|   | - Pure PyTorch Predictor with Native Non-Maximum Suppression (NMS)            |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
```

---

## 2. KDel 4.0 Neural Network Architecture

### 2.1 Topology Breakdown

The **KDel 4.0** architecture consists of three tightly coupled, fully differentiable subsystems:

1. **KDelBackbone (Feature Extractor)**:
   - **Stem Layer**: Initial $3 \times 3$ stride-2 convolution mapping RGB inputs to 32/64 high-dimensional latent representations.
   - **Multi-Scale Stages ($P_2, P_3, P_4, P_5$)**: Hierarchical downsampling blocks operating at strides 4, 8, 16, and 32.
   - **KDel Spatial Attention Block (SAB)**: Channel-wise Squeeze-and-Excitation combined with spatial maximum/average pooling and sigmoid gating to dynamically amplify informative feature regions.

2. **KDelPANet (Bi-Directional Feature Aggregator)**:
   - Top-down pathway injects rich semantic context from $P_5$ down into high-resolution spatial features ($P_3$).
   - Bottom-up pathway propagates precise spatial edge signals back into deep semantic maps.

3. **KDelDecoupledHead (Multi-Scale Anchor-Free Predictor)**:
   - Operates across three pyramid levels ($S \in \{8, 16, 32\}$).
   - Independent convolutional pathways isolate classification gradients from geometric bounding box regression gradients:
     - **Classification Branch**: $\hat{\mathbf{y}}_{\text{cls}} \in \mathbb{R}^{B \times C \times H_s \times W_s}$
     - **Bounding Box Branch**: $\hat{\mathbf{y}}_{\text{reg}} \in \mathbb{R}^{B \times 4 \times H_s \times W_s}$ ($[t_x, t_y, t_w, t_h]$)
     - **Objectness Branch**: $\hat{\mathbf{y}}_{\text{obj}} \in \mathbb{R}^{B \times 1 \times H_s \times W_s}$

```
KDel 4.0 Tensor Flow:
Input: (B, 3, H, W)
  --> KDelStem (Stride 2)
    --> Stage 1 (Stride 4)  --> P2
    --> Stage 2 (Stride 8)  --> P3 (High resolution: small objects)
    --> Stage 3 (Stride 16) --> P4 (Medium resolution: medium objects)
    --> Stage 4 (Stride 32) --> P5 (Deep context: large objects)
      --> KDelPANet Feature Fusion
        --> Head_P3 (Stride 8):  Cls(C), Reg(4), Obj(1)
        --> Head_P4 (Stride 16): Cls(C), Reg(4), Obj(1)
        --> Head_P5 (Stride 32): Cls(C), Reg(4), Obj(1)
```

---

## 3. Mathematical Formulation

### 3.1 Anchor-Free Coordinate Decoding

Given grid coordinates $(g_x, g_y)$ at stride $S$ on an input image of dimensions $(W_I, H_I)$:

$$c_x = \frac{(g_x + \sigma(t_x)) \cdot S}{W_I}, \quad c_y = \frac{(g_y + \sigma(t_y)) \cdot S}{H_I}$$

$$w = \frac{\exp(t_w) \cdot S}{W_I}, \quad h = \frac{\exp(t_h) \cdot S}{H_I}$$

Where normalized box boundaries $[x_1, y_1, x_2, y_2] \in [0, 1]^4$ are derived as:

$$x_1 = \text{clamp}\left(c_x - \frac{w}{2}, 0, 1\right), \quad x_2 = \text{clamp}\left(c_x + \frac{w}{2}, 0, 1\right)$$

$$y_1 = \text{clamp}\left(c_y - \frac{h}{2}, 0, 1\right), \quad y_2 = \text{clamp}\left(c_y + \frac{h}{2}, 0, 1\right)$$

### 3.2 KDel 4.0 Composite Multi-Task Loss

$$\mathcal{L}_{\text{KDel}} = \mathcal{L}_{\text{obj}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$$

Where:
- **Objectness Loss**: Binary Cross-Entropy with Logits penalizing background/foreground discrepancies:
  $$\mathcal{L}_{\text{obj}} = -\sum \left[ y_{\text{obj}} \log(\sigma(\hat{y}_{\text{obj}})) + (1 - y_{\text{obj}}) \log(1 - \sigma(\hat{y}_{\text{obj}})) \right]$$
- **Bounding Box Regression Loss**: Smooth L1 / Complete IoU applied over matched target cells:
  $$\mathcal{L}_{\text{reg}} = \text{Smooth}_{L1}(\hat{\mathbf{b}}, \mathbf{b}^*)$$
- **Multi-Class Cross Entropy Loss**:
  $$\mathcal{L}_{\text{cls}} = -\sum_{k=1}^C y_k \log(\hat{p}_k)$$
- Default balancing weights: $\lambda_{\text{reg}} = 2.0$, $\lambda_{\text{cls}} = 1.0$.

---

## 4. Directory Structure

```
ai_vision_studio/
|-- app/
|   |-- models/
|   |   |-- kdel.py               # KDel 4.0 Architecture, PANet, Decoupled Heads, KDelLoss, NMS
|   |   |-- registry.py           # Checkpoint registry
|   |-- training/
|   |   |-- trainers/
|   |   |   |-- kdel_trainer.py   # Native PyTorch KDel 4.0 Trainer (Zero external ML framework)
|   |   |   |-- classification.py # PyTorch Classification Trainer
|   |   |   |-- detection_yolo.py # Optional secondary fallback
|   |   |-- registry.py           # Trainer Registry (KDel 4.0 as primary default)
|   |   |-- checkpoint.py         # Best and last weight state-dict serializer
|   |   |-- worker.py             # Asynchronous thread supervisor
|   |-- inference/
|   |   |-- predictor.py          # Unified Predictor with native KDel 4.0 decoding
|   |   |-- visualizer.py         # Bounding box rendering utility
|   |-- api/
|   |   |-- v1/
|   |   |   |-- datasets.py       # Folder ingestion, GT parsing, auto-split
|   |   |   |-- training.py       # KDel 4.0 job dispatching and status
|   |   |   |-- inference.py      # REST inference endpoint
|   |   |-- websocket.py          # Live WebSocket telemetry stream
|-- frontend/
|   |-- src/
|   |   |-- views/
|   |   |   |-- StudioView.jsx    # Folder Ingestion, Filmstrip, GT Labeler, Bundle & Train
|   |   |   |-- TrainingView.jsx  # KDel 4.0 Hyperparameters & Real-time Loss Curves
|   |   |   |-- InferenceView.jsx # Visual test playground with latency benchmarks
|   |   |   |-- ModelsView.jsx    # Model Checkpoints & ONNX exporter
|   |   |-- components/
|   |   |   |-- Sidebar.jsx       # Minimalist 4-tab navigation
|   |   |   |-- Topbar.jsx        # Status bar with live connection badge
|   |   |-- index.css             # Light Slate Theme Design System
|-- configs/
|   |-- train_template.yaml       # CLI training configuration
|-- tests/
|   |-- test_api.py               # API route verification
|-- requirements.txt              # Production dependencies
|-- README.md                     # Technical specification & user guide
```

---

## 5. Workflow: From Folder to Trained KDel 4.0 Model

1. **Load Local Folder (`StudioView`)**:
   - Click "โหลดโฟลเดอร์รูปจากเครื่อง" or drag & drop any folder of images from your PC.
   - The system reads all images into memory instantly without blocking.
   - Existing YOLO `.txt` Ground Truth files in the folder are automatically parsed and displayed.

2. **Inspect & Annotate Image-by-Image**:
   - Select images sequentially via the left filmstrip or press arrow keys `[<-]` `[->]`.
   - Click and drag on the canvas to draw precise bounding boxes.
   - Categorize objects into classes (`car`, `person`, `defect`, `item`, etc.) with distinct color tags.
   - Click "ดีเทคอัตโนมัติ (AI Assist)" to let the model pre-generate candidate boxes for rapid verification.

3. **Save Ground Truth (GT)**:
   - Click "บันทึกไฟล์ GT รูปนี้" to save normalized YOLO annotations.
   - Click "บันทึก GT และไปรูปถัดไป" to save and advance to the next picture in one click.

4. **Bundle & Train ("มัดรวมไปเทรน")**:
   - Click "มัดรวมไฟล์ GT และไปเทรนโมเดล".
   - The platform bundles all images and their Ground Truth `.txt` label files, compiles `dataset.yaml`, auto-balances train/val splits, and opens the Training view.

5. **Train KDel 4.0**:
   - Select **KDel 4.0 (Custom Deep Learning Architecture)**.
   - Click "Launch Training Loop".
   - Watch live loss curves, learning rate, and mAP@50 progress update in real time over WebSockets.

6. **Inference & Export**:
   - Test the trained model on new images in the "Testing" tab.
   - Download the compiled weights (`best.pt`) or export to ONNX directly from the "Models" tab.

---

## 6. Python SDK & Programmatic Execution

### 6.1 Train KDel 4.0 Directly in Python
```python
from pathlib import Path
from app.training.trainers.kdel_trainer import KDelDetectionTrainer

# Initialize trainer
trainer = KDelDetectionTrainer(
    job_id=1,
    config={
        "architecture": "kdel4",
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "image_size": 640,
        "device": "auto",
    },
    run_dir=Path("data/runs/kdel_experiment_1"),
)

# Setup dataset from manifest
trainer.setup(Path("data/datasets/my_dataset/dataset.yaml"))

# Run native training loop
metrics = trainer.train()
print("Training completed:", metrics)
```

### 6.2 Inference with Trained KDel 4.0 Weights
```python
from app.inference.predictor import Predictor

# Instantiate predictor from KDel 4.0 weights
predictor = Predictor("data/runs/kdel_experiment_1/checkpoints/best.pt")

# Run inference on target image
result = predictor.predict("test_sample.jpg", conf_threshold=0.25)

# Print detected objects
for det in result.detections:
    print(f"Detected {det['class_name']} with confidence {det['confidence']:.3f} at {det['box']}")

# Save annotated image
result.save("output_kdel4.jpg")
```

---

## 7. Automated Testing & Verification

```bash
# Run unit and integration tests
python -m pytest tests/ -v
```

---

## 8. Deployment

### Run with Local Dev Server:
```bash
python -m app run --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

### Deploy Frontend to Vercel:
```bash
cd frontend
vercel deploy --prod
```
