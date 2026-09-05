from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from app.schemas.annotation import AnnotationResponse


class DatasetBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = None
    task_type: str = Field("detection", pattern="^(detection|classification|segmentation)$")
    classes: List[str] = Field(default_factory=list)


class DatasetCreate(DatasetBase):
    project_id: int


class DatasetSplitRequest(BaseModel):
    train_ratio: float = Field(0.70, ge=0.1, le=0.9)
    val_ratio: float = Field(0.20, ge=0.05, le=0.5)
    test_ratio: float = Field(0.10, ge=0.0, le=0.4)
    seed: int = 42


class AugmentationConfig(BaseModel):
    enabled: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotation: int = 15
    brightness_contrast: float = 0.2
    blur: float = 0.0
    noise: float = 0.0


class ImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    filename: str
    original_name: str
    mime_type: str
    file_size: int
    width: int
    height: int
    split: str
    is_annotated: bool
    annotations: List[AnnotationResponse] = []
    image_url: Optional[str] = None
    created_at: datetime


class DatasetResponse(DatasetBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    status: str
    validation_report: Optional[Dict[str, Any]] = None
    total_images: int = 0
    total_annotations: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    created_at: datetime
    updated_at: datetime
