# API Reference - AI Vision Training Studio

All REST endpoints are prefixed with `/api/v1`. Live telemetry is broadcast over WebSocket at `/ws/live`.

---

## 1. Projects API (`/api/v1/projects`)
- `GET /api/v1/projects` - List all projects with dataset/model counters.
- `POST /api/v1/projects` - Create project (`name`, `task_type`: detection/classification, `description`).
- `GET /api/v1/projects/{id}` - Get project details.
- `DELETE /api/v1/projects/{id}` - Delete project and cascaded records.

---

## 2. Datasets API (`/api/v1/datasets`)
- `GET /api/v1/datasets?project_id={id}` - List datasets for project.
- `POST /api/v1/datasets` - Create dataset (`name`, `classes`: `["cat", "dog"]`).
- `POST /api/v1/datasets/{id}/upload` - Upload multiple image files (`multipart/form-data`).
- `POST /api/v1/datasets/{id}/import-folder` - Scan local folder on server and import images with `.txt` labels.
- `GET /api/v1/datasets/{id}/images` - Paginated image list with annotation details.
- `GET /api/v1/datasets/images/{image_id}/file` - Stream raw image file.
- `POST /api/v1/datasets/{id}/validate` - Perform deep image and annotation validation.
- `POST /api/v1/datasets/{id}/split` - Stratified split into train/val/test and generate `dataset.yaml`.
- `POST /api/v1/datasets/{id}/augment-preview` - Returns augmented image with bounding boxes drawn.

---

## 3. Annotations API (`/api/v1/annotations`)
- `GET /api/v1/annotations/{image_id}` - Get all YOLO annotations for an image.
- `POST /api/v1/annotations` - Create single annotation (`bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`).
- `PUT /api/v1/annotations/batch` - Save/replace all annotations for an image in batch.
- `DELETE /api/v1/annotations/{id}` - Delete single annotation.

---

## 4. Training API (`/api/v1/training`)
- `GET /api/v1/training/architectures` - List supported models (YOLO11, YOLOv8, ResNet, MobileNet).
- `GET /api/v1/training/jobs?project_id={id}` - List training jobs and recent epoch metrics.
- `POST /api/v1/training/start` - Launch background training job.
- `GET /api/v1/training/{id}` - Get job status, current epoch, and best metric.
- `POST /api/v1/training/{id}/stop` - Gracefully stop active training job.
- `GET /api/v1/training/{id}/logs` - Fetch raw training log stream.

---

## 5. Models Registry API (`/api/v1/models`)
- `GET /api/v1/models?project_id={id}` - List registered models with metrics.
- `GET /api/v1/models/{id}` - Get model card and provenance.
- `POST /api/v1/models/{id}/export` - Export model to ONNX or TorchScript.
- `GET /api/v1/models/{id}/download?format=pt|onnx` - Download model weight file.
- `DELETE /api/v1/models/{id}` - Delete model record and files.

---

## 6. Inference API (`/api/v1/inference`)
- `POST /api/v1/inference/predict` - Upload image, run inference, returns detections list and URL of annotated output image.
- `POST /api/v1/inference/batch` - Run inference over folder of images, outputs JSON/CSV/TXT.
- `GET /api/v1/inference/output/{filename}` - Retrieve rendered detection image.

---

## 7. System API (`/api/v1/system`)
- `GET /api/v1/system/info` - Static hardware specs, Python/PyTorch/CUDA versions.
- `GET /api/v1/system/metrics` - Real-time CPU %, RAM %, and GPU VRAM telemetry.

---

## 8. WebSocket Stream (`/ws/live`)
Broadcasts real-time events:
- `epoch_update`: `{ "epoch": 1, "total_epochs": 20, "loss": 0.42, "metrics": {"mAP50": 0.85} }`
- `gpu_update`: `{ "cpu_percent": 15.2, "ram_percent": 42.1, "gpus": [...] }`
- `training_complete`: `{ "job_id": 1, "model_id": 2, "metrics": {...} }`
- `training_error`: `{ "job_id": 1, "error": "CUDA OOM" }`
