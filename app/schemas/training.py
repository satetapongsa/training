from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class TrainingConfigSchema(BaseModel):
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(8, ge=1, le=256)
    image_size: int = Field(640, ge=64, le=1920)
    learning_rate: float = Field(0.001, gt=0.0, le=1.0)
    optimizer: str = Field("AdamW", pattern="^(AdamW|SGD|Adam)$")
    weight_decay: float = Field(0.0005, ge=0.0)
    scheduler: str = Field("cosine", pattern="^(cosine|linear|step)$")
    warmup_epochs: int = Field(3, ge=0)
    early_stopping_patience: int = Field(10, ge=0)
    checkpoint_interval: int = Field(5, ge=1)
    device: str = Field("auto", pattern="^(auto|cuda|cpu)$")
    seed: int = 42
    deterministic: bool = True
    augmentation: Dict[str, Any] = Field(default_factory=dict)


class TrainingJobStartRequest(BaseModel):
    project_id: int
    dataset_id: int
    model_name: str = Field(..., min_length=2, max_length=100)
    architecture: str = Field("yolo11n", min_length=2, max_length=100)
    config: TrainingConfigSchema


class TrainingMetricResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    epoch: int
    step: int
    loss: float
    val_loss: Optional[float] = None
    metrics: Dict[str, Any] = {}
    lr: float
    timestamp: datetime


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    dataset_id: int
    model_name: str
    architecture: str
    status: str
    config: Dict[str, Any]
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    best_metric_val: Optional[float] = None
    best_metric_name: str
    run_dir: str
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    recent_metrics: List[TrainingMetricResponse] = []
