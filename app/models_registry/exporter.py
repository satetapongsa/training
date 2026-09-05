import json
import sys
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO

from app.core.logging import logger


class ModelExporterBase(ABC):
    """Abstract interface for exporting model formats."""

    @abstractmethod
    def export(self, weight_path: Path, output_path: Path, config: Dict[str, Any]) -> Path:
        pass


class YOLONNExporter(ModelExporterBase):
    """Exports Ultralytics YOLO models to ONNX or TorchScript."""

    def export(self, weight_path: Path, output_path: Path, config: Dict[str, Any]) -> Path:
        fmt = config.get("format", "onnx").lower()
        imgsz = int(config.get("image_size", 640))
        dynamic = bool(config.get("dynamic", True))

        model = YOLO(str(weight_path))
        exported_file = model.export(format=fmt, imgsz=imgsz, dynamic=dynamic)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_file), str(output_path))
        return output_path


class ModelMetadataGenerator:
    """Generates standard metadata.json for exported models."""

    @staticmethod
    def generate(
        model_name: str,
        version: str,
        architecture: str,
        dataset_name: str,
        classes: list,
        metrics: Dict[str, Any],
        hyperparams: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "model_name": model_name,
            "model_version": version,
            "architecture": architecture,
            "dataset": dataset_name,
            "classes": classes,
            "metrics": metrics,
            "hyperparameters": hyperparams,
            "framework": "PyTorch",
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "cuda_available": torch.cuda.is_available(),
        }
