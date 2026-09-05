from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class SystemInfoResponse(BaseModel):
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    device_count: int
    gpus: List[Dict[str, Any]]
    cpu_count: int
    physical_cpu_count: int
    total_ram_gb: float


class RealtimeMetricsResponse(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpus: List[Dict[str, Any]]
