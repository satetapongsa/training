from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class AnnotationBase(BaseModel):
    class_id: int = Field(..., ge=0)
    class_name: str = Field(..., min_length=1)
    # Normalized YOLO format 0..1
    bbox_x: float = Field(..., ge=0.0, le=1.0)
    bbox_y: float = Field(..., ge=0.0, le=1.0)
    bbox_w: float = Field(..., ge=0.0, le=1.0)
    bbox_h: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    segmentation: Optional[List[List[float]]] = None


class AnnotationCreate(AnnotationBase):
    image_id: int


class AnnotationUpdate(BaseModel):
    class_id: Optional[int] = Field(None, ge=0)
    class_name: Optional[str] = None
    bbox_x: Optional[float] = Field(None, ge=0.0, le=1.0)
    bbox_y: Optional[float] = Field(None, ge=0.0, le=1.0)
    bbox_w: Optional[float] = Field(None, ge=0.0, le=1.0)
    bbox_h: Optional[float] = Field(None, ge=0.0, le=1.0)


class AnnotationResponse(AnnotationBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    created_at: datetime
    updated_at: datetime


class BatchAnnotationsUpdate(BaseModel):
    image_id: int
    annotations: List[AnnotationBase]
