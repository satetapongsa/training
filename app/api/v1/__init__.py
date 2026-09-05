from fastapi import APIRouter
from app.api.v1.projects import router as projects_router
from app.api.v1.datasets import router as datasets_router
from app.api.v1.annotations import router as annotations_router
from app.api.v1.training import router as training_router
from app.api.v1.models import router as models_router
from app.api.v1.inference import router as inference_router
from app.api.v1.system import router as system_router

api_v1_router = APIRouter()
api_v1_router.include_router(projects_router)
api_v1_router.include_router(datasets_router)
api_v1_router.include_router(annotations_router)
api_v1_router.include_router(training_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(inference_router)
api_v1_router.include_router(system_router)

__all__ = ["api_v1_router"]
