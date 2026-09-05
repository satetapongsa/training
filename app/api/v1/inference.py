import io
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from PIL import Image

from app.api.deps import get_database_session
from app.core.config import settings
from app.core.security import verify_image_file, sanitize_filename
from app.db.models import Model
from app.schemas.inference import InferencePredictionResult, BoundingBox
from app.inference.predictor import Predictor
from app.inference.batch import BatchInferenceProcessor

router = APIRouter(prefix="/inference", tags=["Inference"])

# Cache loaded predictors for fast interactive inferences
_predictor_cache = {}


def get_cached_predictor(model_path: str) -> Predictor:
    if model_path not in _predictor_cache:
        _predictor_cache[model_path] = Predictor(model_path)
    return _predictor_cache[model_path]


@router.post("/predict", response_model=InferencePredictionResult)
async def predict_single_image(
    model_id: int = Form(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_database_session),
):
    stmt = select(Model).filter(Model.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    weight_path = Path(model.weights_path)
    if not weight_path.exists():
        raise HTTPException(status_code=404, detail=f"Model weights missing: {weight_path}")

    # Read uploaded image bytes
    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        predictor = get_cached_predictor(str(weight_path))
        pred_res = predictor.predict(
            pil_img, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )

        # Save annotated image into exports/inference
        output_dir = settings.EXPORT_DIR / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex[:10]}_{sanitize_filename(file.filename)}"
        annotated_path = output_dir / unique_name
        pred_res.save(annotated_path)

        box_schemas = [BoundingBox(**b) for b in pred_res.detections]

        return InferencePredictionResult(
            model_id=model.id,
            model_name=model.name,
            image_width=pred_res.width,
            image_height=pred_res.height,
            detections=box_schemas,
            total_detections=len(box_schemas),
            inference_time_ms=pred_res.inference_time_ms,
            annotated_image_url=f"/api/v1/inference/output/{unique_name}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/output/{filename}")
async def get_inference_output_image(filename: str):
    safe_name = sanitize_filename(filename)
    target = settings.EXPORT_DIR / "inference" / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Inference output image not found.")
    return FileResponse(target, media_type="image/jpeg")


@router.post("/batch")
async def batch_inference(
    model_id: int = Form(...),
    folder_path: str = Form(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    db: AsyncSession = Depends(get_database_session),
):
    stmt = select(Model).filter(Model.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder path does not exist.")

    allowed_exts = {f".{ext}" for ext in settings.ALLOWED_IMAGE_EXTENSIONS}
    images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
    if not images:
        raise HTTPException(status_code=400, detail="No valid images found in folder.")

    predictor = get_cached_predictor(model.weights_path)
    output_dir = settings.EXPORT_DIR / "batch" / f"batch_{uuid.uuid4().hex[:8]}"

    report = BatchInferenceProcessor.process_batch(
        predictor=predictor,
        image_sources=images,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        save_images=True,
        export_formats=["json", "csv", "txt"],
    )

    return report
