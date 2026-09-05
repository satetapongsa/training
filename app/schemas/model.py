from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict


class ModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    job_id: Optional[int] = None
    name: str
    version: str
    architecture: str
    task_type: str
    classes: List[str]
    weights_path: str
    onnx_path: Optional[str] = None
    torchscript_path: Optional[str] = None
    metrics: Dict[str, Any] = {}
    metadata_info: Dict[str, Any] = {}
    size_bytes: int
    is_deployed: bool
    created_at: datetime


class ModelExportRequest(BaseModel):
    format: str = "onnx"  # onnx, torchscript, pt
    image_size: int = 640
    batch_size: int = 1
    dynamic: bool = True
