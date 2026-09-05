# Developer & Contribution Guide - AI Vision Training Studio

## 1. Project Layout
```
ai_vision_studio/
├── app/
│   ├── api/             # FastAPI routers & WebSocket live endpoint
│   ├── core/            # Config, telemetry, security, logging
│   ├── db/              # SQLAlchemy 2.0 async/sync models & sessions
│   ├── datasets/        # Ingestion, deep validator, splitter, augmentor
│   ├── annotations/     # YOLO, COCO, VOC parsers
│   ├── training/        # TrainerBase, worker, registry, YOLO & PyTorch trainers
│   ├── models_registry/ # Exporters (ONNX/TorchScript), metadata
│   ├── inference/       # Predictor SDK, batch processor, visualizer
│   └── static/          # Single Page Application (HTML/CSS/JS)
├── configs/             # YAML configurations
├── data/                # Runtime data directory (uploads, runs, models)
├── docker/              # Dockerfile and compose
├── tests/               # Automated pytest suite
└── requirements.txt
```

---

## 2. Adding a New Model Architecture

To add a new architecture (e.g. `RetinaNet` or custom model):
1. Create a trainer class subclassing `app.training.base.TrainerBase`:
   ```python
   from app.training.base import TrainerBase

   class MyCustomTrainer(TrainerBase):
       def setup(self, manifest_path): ...
       def train(self): ...
       def stop(self): ...
       def export(self, fmt, out_path): ...
   ```
2. Register it with `TrainerRegistry`:
   ```python
   from app.training.registry import TrainerRegistry
   TrainerRegistry.register("my_model_name", MyCustomTrainer)
   ```
No core database or frontend routes need to be modified!

---

## 3. Running Code Style & Linting
```bash
# Type checking
mypy app/

# Code formatting
black app/ tests/

# Test suite execution
pytest tests/ -v
```
