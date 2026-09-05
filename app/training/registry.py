from typing import Dict, Type
from app.training.base import TrainerBase
from app.training.trainers.detection_yolo import YOLODetectionTrainer
from app.training.trainers.classification import PyTorchClassificationTrainer


class TrainerRegistry:
    """Registry pattern allowing dynamic addition of model architectures."""

    _registry: Dict[str, Type[TrainerBase]] = {
        # Detection architectures
        "yolo11n": YOLODetectionTrainer,
        "yolo11s": YOLODetectionTrainer,
        "yolov8n": YOLODetectionTrainer,
        "yolov8s": YOLODetectionTrainer,
        "detection_default": YOLODetectionTrainer,
        # Classification architectures
        "resnet18": PyTorchClassificationTrainer,
        "resnet50": PyTorchClassificationTrainer,
        "mobilenet_v3": PyTorchClassificationTrainer,
        "classification_default": PyTorchClassificationTrainer,
    }

    @classmethod
    def register(cls, name: str, trainer_cls: Type[TrainerBase]) -> None:
        cls._registry[name.lower()] = trainer_cls

    @classmethod
    def get(cls, name: str, task_type: str = "detection") -> Type[TrainerBase]:
        arch = name.lower()
        if arch in cls._registry:
            return cls._registry[arch]
        
        # Fallback based on task type
        if task_type == "classification":
            return cls._registry["classification_default"]
        return cls._registry["detection_default"]

    @classmethod
    def list_supported_models(cls) -> Dict[str, list]:
        detection_models = [k for k, v in cls._registry.items() if v is YOLODetectionTrainer and "default" not in k]
        classification_models = [k for k, v in cls._registry.items() if v is PyTorchClassificationTrainer and "default" not in k]
        return {
            "detection": detection_models,
            "classification": classification_models,
        }
