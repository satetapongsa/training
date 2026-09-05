from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    # Normalized coordinates 0..1
    x1: float
    y1: float
    x2: float
    y2: float
    # Pixel coordinates
    box_pixels: Optional[List[int]] = None


class InferencePredictionResult(BaseModel):
    model_id: int
    model_name: str
    image_width: int
    image_height: int
    detections: List[BoundingBox]
    total_detections: int
    inference_time_ms: float
    annotated_image_url: Optional[str] = None


class BatchInferenceRequest(BaseModel):
    model_id: int
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    save_annotated_images: bool = True
    output_format: str = Field("json", pattern="^(json|csv|txt)$")
