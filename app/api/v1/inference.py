import io
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from PIL import Image

from app.api.deps import get_database_session
from app.core.config import settings
from app.core.security import verify_image_file, sanitize_filename
from app.core.logging import logger
from app.db.models import Model, TrainingJob
from app.schemas.inference import InferencePredictionResult, BoundingBox
from app.inference.predictor import Predictor
from app.inference.batch import BatchInferenceProcessor

router = APIRouter(prefix="/inference", tags=["Inference"])

# Cache loaded predictors with modification timestamp for automatic hot-reloading
_predictor_cache: Dict[str, tuple[float, Predictor]] = {}


def get_cached_predictor(model_path: str) -> Predictor:
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    mtime = path_obj.stat().st_mtime
    if model_path in _predictor_cache:
        cached_mtime, cached_predictor = _predictor_cache[model_path]
        if cached_mtime == mtime:
            return cached_predictor

    logger.info(f"Loading/Updating predictor for model: {model_path}")
    predictor = Predictor(model_path)
    _predictor_cache[model_path] = (mtime, predictor)
    return predictor


async def resolve_model_weight_path(
    db: AsyncSession,
    run_id: Optional[int] = None,
    model_id: Optional[int] = None,
) -> tuple[Path, str, int]:
    """Resolves target model weight file path, name, and ID with automatic fallback."""
    target_id = run_id or model_id

    # 1. Search by specific target ID in TrainingJob
    if target_id is not None:
        job_stmt = select(TrainingJob).filter(TrainingJob.id == target_id)
        job_res = await db.execute(job_stmt)
        job = job_res.scalar_one_or_none()
        if job:
            run_dir = Path(job.run_dir)
            candidates = []
            if job.checkpoint_path:
                candidates.append(Path(job.checkpoint_path))
            candidates.extend([
                run_dir / "checkpoints" / "best.pt",
                run_dir / "weights" / "best.pt",
                run_dir / "checkpoints" / "last.pt",
                run_dir / "weights" / "last.pt",
            ])
            for cand in candidates:
                if cand.exists() and cand.is_file():
                    return cand, job.model_name, job.id

        # Search by specific target ID in Model table
        model_stmt = select(Model).filter(Model.id == target_id)
        model_res = await db.execute(model_stmt)
        model_obj = model_res.scalar_one_or_none()
        if model_obj and model_obj.weights_path:
            p = Path(model_obj.weights_path)
            if p.exists():
                return p, model_obj.name, model_obj.id

    # 2. Auto-fallback: Find latest completed training job with existing weights
    latest_jobs_stmt = (
        select(TrainingJob)
        .order_by(TrainingJob.id.desc())
    )
    latest_jobs_res = await db.execute(latest_jobs_stmt)
    all_jobs = latest_jobs_res.scalars().all()
    for j in all_jobs:
        run_dir = Path(j.run_dir)
        for cand in [
            Path(j.checkpoint_path) if j.checkpoint_path else None,
            run_dir / "checkpoints" / "best.pt",
            run_dir / "weights" / "best.pt",
            run_dir / "checkpoints" / "last.pt",
        ]:
            if cand and cand.exists() and cand.is_file():
                return cand, j.model_name, j.id

    # 3. Search Model registry table
    latest_models_stmt = select(Model).order_by(Model.id.desc())
    latest_models_res = await db.execute(latest_models_stmt)
    all_models = latest_models_res.scalars().all()
    for m in all_models:
        if m.weights_path:
            p = Path(m.weights_path)
            if p.exists():
                return p, m.name, m.id

    # 4. Search runs directory directly
    if settings.RUNS_DIR.exists():
        for pt_file in sorted(settings.RUNS_DIR.rglob("best.pt"), reverse=True):
            if pt_file.is_file():
                return pt_file, "KDel 4.0 Auto-Discovered", 1

    # 5. Base fallback if available
    base_default = settings.BASE_DIR / "yolo11n.pt"
    if base_default.exists():
        return base_default, "YOLO Baseline", 1

    raise FileNotFoundError("ไม่พบไฟล์โมเดลที่เทรนเสร็จในระบบ กรุณาเทรนโมเดล KDel 4.0 ก่อนทำการทดสอบภาพ")


@router.post("/detect")
@router.post("/predict")
async def detect_single_image(
    file: UploadFile = File(...),
    run_id: Optional[int] = Form(None),
    model_id: Optional[int] = Form(None),
    confidence: Optional[float] = Form(None),
    conf_threshold: Optional[float] = Form(None),
    iou_threshold: float = Form(0.45),
    db: AsyncSession = Depends(get_database_session),
):
    """Executes single image object detection using KDel 4.0 or specified model."""
    try:
        weight_path, model_name, active_id = await resolve_model_weight_path(
            db=db, run_id=run_id, model_id=model_id
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Read uploaded image bytes
    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ไฟล์รูปภาพไม่ถูกต้อง: {str(e)}")

    effective_conf = (
        confidence
        if confidence is not None
        else (conf_threshold if conf_threshold is not None else 0.25)
    )

    try:
        predictor = get_cached_predictor(str(weight_path))
        pred_res = predictor.predict(
            pil_img, conf_threshold=effective_conf, iou_threshold=iou_threshold
        )

        # Save annotated image into exports/inference
        output_dir = settings.EXPORT_DIR / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex[:10]}_{sanitize_filename(file.filename or 'image.jpg')}"
        annotated_path = output_dir / unique_name
        pred_res.save(annotated_path)

        # Ensure normalized coordinates and box dictionary are populated for both interfaces
        formatted_detections = []
        for d in pred_res.detections:
            x1 = float(d.get("x1", d.get("box", {}).get("x1", 0.0)))
            y1 = float(d.get("y1", d.get("box", {}).get("y1", 0.0)))
            x2 = float(d.get("x2", d.get("box", {}).get("x2", 1.0)))
            y2 = float(d.get("y2", d.get("box", {}).get("y2", 1.0)))
            box_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            formatted_detections.append({
                "class_id": int(d.get("class_id", 0)),
                "class_name": str(d.get("class_name", "object")),
                "confidence": float(d.get("confidence", 1.0)),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box": box_dict,
                "box_pixels": d.get("box_pixels", [
                    int(x1 * pred_res.width),
                    int(y1 * pred_res.height),
                    int(x2 * pred_res.width),
                    int(y2 * pred_res.height),
                ]),
            })

        return {
            "model_id": active_id,
            "model_name": model_name,
            "image_width": pred_res.width,
            "image_height": pred_res.height,
            "detections": formatted_detections,
            "total_detections": len(formatted_detections),
            "inference_time_ms": round(pred_res.inference_time_ms, 2),
            "annotated_image_url": f"/api/v1/inference/output/{unique_name}",
        }

    except Exception as e:
        logger.exception(f"Inference execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"การตรวจจับล้มเหลว: {str(e)}")


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
