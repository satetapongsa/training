import sys
from typing import Dict, Any, List
import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


class SystemTelemetry:
    """Monitors system resources (CPU, RAM, GPU) and framework specs."""

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Returns static hardware configuration and framework versions."""
        cuda_available = torch.cuda.is_available()
        gpu_info: List[Dict[str, Any]] = []

        if cuda_available:
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "id": i,
                    "name": prop.name,
                    "total_memory_mb": round(prop.total_memory / (1024 * 1024), 2),
                    "multi_processor_count": prop.multi_processor_count,
                })

        return {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda if cuda_available else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "gpus": gpu_info,
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }

    @staticmethod
    def get_realtime_metrics() -> Dict[str, Any]:
        """Returns live CPU %, RAM %, and GPU utilization & VRAM."""
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        gpu_metrics: List[Dict[str, Any]] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                total = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                
                utilization = 0.0
                if HAS_NVML:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        utilization = float(util.gpu)
                    except Exception:
                        pass

                gpu_metrics.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_used_mb": round(allocated, 2),
                    "vram_reserved_mb": round(reserved, 2),
                    "vram_total_mb": round(total, 2),
                    "vram_percent": round((allocated / total) * 100, 2) if total > 0 else 0,
                    "utilization_percent": utilization,
                })

        return {
            "cpu_percent": cpu_percent,
            "ram_used_gb": round(ram.used / (1024**3), 2),
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "ram_percent": ram.percent,
            "gpus": gpu_metrics,
        }


telemetry = SystemTelemetry()
