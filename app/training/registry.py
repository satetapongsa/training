from typing import Dict, Type
from app.training.base import TrainerBase
from app.training.trainers.kdel_trainer import KDelDetectionTrainer
from app.training.trainers.detection_yolo import YOLODetectionTrainer
from app.training.trainers.classification import PyTorchClassificationTrainer


class TrainerRegistry:
    """Registry pattern allowing dynamic addition of model architectures."""

    _registry: Dict[str, Type[TrainerBase]] = {
        # KDel 4.0 Native PyTorch Detection architectures (Primary Default)
        "kdel4": KDelDetectionTrainer,
        "kdel-4.0": KDelDetectionTrainer,
        "kdel4.0": KDelDetectionTrainer,
        "kdel4_nano": KDelDetectionTrainer,
        "kdel4_pro": KDelDetectionTrainer,
        "detection_default": KDelDetectionTrainer,

        # Standard Detection architectures
        "yolo11n": YOLODetectionTrainer,
        "yolo11s": YOLODetectionTrainer,
        "yolov8n": YOLODetectionTrainer,
        "yolov8s": YOLODetectionTrainer,

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
        detection_models = [k for k, v in cls._registry.items() if v in (KDelDetectionTrainer, YOLODetectionTrainer) and "default" not in k]
        classification_models = [k for k, v in cls._registry.items() if v is PyTorchClassificationTrainer and "default" not in k]
        return {
            "detection": detection_models,
            "classification": classification_models,
        }


def get_trainer_class(name: str, task_type: str = "detection") -> Type[TrainerBase]:
    """Helper function to retrieve trainer class by architecture name."""
    return TrainerRegistry.get(name, task_type=task_type)
