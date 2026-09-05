from fastapi import APIRouter
from app.core.telemetry import telemetry
from app.schemas.system import SystemInfoResponse, RealtimeMetricsResponse

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/info", response_model=SystemInfoResponse)
async def get_system_hardware_info():
    """Returns static hardware configuration and ML framework details."""
    return telemetry.get_hardware_info()


@router.get("/metrics", response_model=RealtimeMetricsResponse)
async def get_system_realtime_metrics():
    """Returns real-time resource utilization (CPU, RAM, GPU)."""
    return telemetry.get_realtime_metrics()
