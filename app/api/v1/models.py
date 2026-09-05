from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_database_session
from app.db.models import Model
from app.schemas.model import ModelResponse, ModelExportRequest
from app.models_registry.manager import ModelRegistryManager

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=List[ModelResponse])
async def list_models(
    project_id: Optional[int] = None,
    db: AsyncSession = Depends(get_database_session),
):
    stmt = select(Model).order_by(Model.created_at.desc())
    if project_id:
        stmt = stmt.filter(Model.project_id == project_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: AsyncSession = Depends(get_database_session)):
    stmt = select(Model).filter(Model.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    return model


@router.post("/{model_id}/export")
async def export_model(
    model_id: int,
    payload: ModelExportRequest,
    db: AsyncSession = Depends(get_database_session),
):
    stmt = select(Model).filter(Model.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        exported_path = await ModelRegistryManager.export_model_format(
            db=db,
            model=model,
            target_format=payload.format,
            image_size=payload.image_size,
        )
        return {
            "success": True,
            "format": payload.format,
            "export_path": str(exported_path),
            "filename": exported_path.name,
            "download_url": f"/api/v1/models/{model.id}/download?format={payload.format}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/{model_id}/download")
async def download_model_weights(
    model_id: int,
    format: str = Query("pt", pattern="^(pt|onnx|torchscript)$"),
    db: AsyncSession = Depends(get_database_session),
):
    stmt = select(Model).filter(Model.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    if format == "onnx":
        if not model.onnx_path or not Path(model.onnx_path).exists():
            # Auto-export if not yet exported
            exported = await ModelRegistryManager.export_model_format(db, model, "onnx")
            target_file = exported
        else:
            target_file = Path(model.onnx_path)
    elif format == "torchscript":
        if not model.torchscript_path or not Path(model.torchscript_path).exists():
            exported = await ModelRegistryManager.export_model_format(db, model, "torchscript")
            target_file = exported
        else:
            target_file = Path(model.torchscript_path)
    else:
        target_file = Path(model.weights_path)

    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Model weight file not found at {target_file}")

    return FileResponse(
        str(target_file),
        media_type="application/octet-stream",
        filename=f"{model.name}_{model.version}.{format}",
    )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: int, db: AsyncSession = Depends(get_database_session)):
    stmt = select(Model).filter(Model.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    await db.delete(model)
    await db.commit()
