from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import json
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, Enum, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_type: Mapped[str] = mapped_column(String(50), default="detection")  # detection, classification, segmentation
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    datasets: Mapped[List["Dataset"]] = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    training_jobs: Mapped[List["TrainingJob"]] = relationship("TrainingJob", back_populates="project", cascade="all, delete-orphan")
    models: Mapped[List["Model"]] = relationship("Model", back_populates="project", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_type: Mapped[str] = mapped_column(String(50), default="detection")
    classes: Mapped[List[str]] = mapped_column(JSON, default=list)  # e.g. ["cat", "dog"]
    status: Mapped[str] = mapped_column(String(50), default="ready")  # ready, validating, error
    validation_report: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    total_images: Mapped[int] = mapped_column(Integer, default=0)
    total_annotations: Mapped[int] = mapped_column(Integer, default=0)
    train_count: Mapped[int] = mapped_column(Integer, default=0)
    val_count: Mapped[int] = mapped_column(Integer, default=0)
    test_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    project: Mapped["Project"] = relationship("Project", back_populates="datasets")
    images: Mapped[List["Image"]] = relationship("Image", back_populates="dataset", cascade="all, delete-orphan")
    versions: Mapped[List["DatasetVersion"]] = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")
    training_jobs: Mapped[List["TrainingJob"]] = relationship("TrainingJob", back_populates="dataset")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), default="v1.0.0")
    manifest_path: Mapped[str] = mapped_column(String(255), nullable=False)  # Path to dataset.yaml
    split_ratio: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)  # {"train": 0.7, "val": 0.2, "test": 0.1}
    augmentation_config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="versions")
    training_jobs: Mapped[List["TrainingJob"]] = relationship("TrainingJob", back_populates="dataset_version")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(50), default="image/jpeg")
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    split: Mapped[str] = mapped_column(String(20), default="train")  # train, val, test, unassigned
    is_annotated: Mapped[bool] = mapped_column(Boolean, default=False)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="images")
    annotations: Mapped[List["Annotation"]] = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Normalized YOLO format (0.0 to 1.0)
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_w: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_h: Mapped[float] = mapped_column(Float, nullable=False)
    segmentation: Mapped[Optional[List[List[float]]]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    image: Mapped["Image"] = relationship("Image", back_populates="annotations")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    dataset_version_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True)
    
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    architecture: Mapped[str] = mapped_column(String(100), nullable=False)  # yolo11n, resnet18, etc.
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending, running, paused, completed, failed, cancelled
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=50)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)

    best_metric_val: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    best_metric_name: Mapped[str] = mapped_column(String(50), default="mAP50")

    run_dir: Mapped[str] = mapped_column(String(500), nullable=False)
    checkpoint_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    project: Mapped["Project"] = relationship("Project", back_populates="training_jobs")
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="training_jobs")
    dataset_version: Mapped[Optional["DatasetVersion"]] = relationship("DatasetVersion", back_populates="training_jobs")
    metrics: Mapped[List["TrainingMetric"]] = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")
    produced_model: Mapped[Optional["Model"]] = relationship("Model", back_populates="job", uselist=False)


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    step: Mapped[int] = mapped_column(Integer, default=0)
    loss: Mapped[float] = mapped_column(Float, nullable=False)
    val_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)  # {"precision": 0.85, "recall": 0.82, "mAP50": 0.88}
    lr: Mapped[float] = mapped_column(Float, default=0.001)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    job: Mapped["TrainingJob"] = relationship("TrainingJob", back_populates="metrics")


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    job_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("training_jobs.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), default="v1.0.0")
    architecture: Mapped[str] = mapped_column(String(100), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), default="detection")
    classes: Mapped[List[str]] = mapped_column(JSON, default=list)

    weights_path: Mapped[str] = mapped_column(String(500), nullable=False)  # best.pt
    onnx_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    torchscript_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    metadata_info: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    is_deployed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    project: Mapped["Project"] = relationship("Project", back_populates="models")
    job: Mapped[Optional["TrainingJob"]] = relationship("TrainingJob", back_populates="produced_model")
    inference_jobs: Mapped[List["InferenceJob"]] = relationship("InferenceJob", back_populates="model")
    versions: Mapped[List["ModelVersion"]] = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), default="v1.0.0")
    weights_path: Mapped[str] = mapped_column(String(500), nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    model: Mapped["Model"] = relationship("Model", back_populates="versions")



class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), default="single")  # single, batch, folder
    total_images: Mapped[int] = mapped_column(Integer, default=1)
    processed_images: Mapped[int] = mapped_column(Integer, default=0)
    output_dir: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="completed")
    results_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    model: Mapped["Model"] = relationship("Model", back_populates="inference_jobs")


class SystemConfig(Base):
    __tablename__ = "system_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
