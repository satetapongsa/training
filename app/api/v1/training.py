from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import get_database_session
from app.core.config import settings
from app.core.logging import logger
from app.db.models import TrainingJob, TrainingMetric, Dataset, Project, Image
from app.schemas.training import (
    TrainingJobStartRequest,
    TrainingJobResponse,
    TrainingMetricResponse,
)
from app.training.worker import training_worker
from app.training.registry import TrainerRegistry
from app.datasets.splitter import DatasetSplitter

router = APIRouter(prefix="/training", tags=["Training"])


@router.get("/architectures")
async def get_architectures():
    """Returns available model architectures separated by task type."""
    return TrainerRegistry.list_supported_models()


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    project_id: Optional[int] = None,
    db: AsyncSession = Depends(get_database_session),
):
    stmt = (
        select(TrainingJob)
        .options(selectinload(TrainingJob.metrics))
        .order_by(TrainingJob.created_at.desc())
    )
    if project_id:
        stmt = stmt.filter(TrainingJob.project_id == project_id)
    result = await db.execute(stmt)
    jobs = result.scalars().all()

    resp = []
    for j in jobs:
        item = TrainingJobResponse.model_validate(j)
        item.recent_metrics = [
            TrainingMetricResponse.model_validate(m) for m in j.metrics[-20:]
        ]
        resp.append(item)
    return resp


@router.get("/runs", response_model=List[TrainingJobResponse])
async def list_training_runs(
    project_id: Optional[int] = None,
    db: AsyncSession = Depends(get_database_session),
):
    return await list_training_jobs(project_id=project_id, db=db)


@router.post("/start", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def start_training(
    payload: TrainingJobStartRequest,
    db: AsyncSession = Depends(get_database_session),
):
    # Verify project & dataset
    proj_res = await db.execute(select(Project).filter(Project.id == payload.project_id))
    project = proj_res.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    ds_res = await db.execute(
        select(Dataset).options(selectinload(Dataset.images).selectinload(Image.annotations)).filter(Dataset.id == payload.dataset_id)
    )
    dataset = ds_res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not dataset.images:
        raise HTTPException(status_code=400, detail="Dataset has no images to train on.")

    # Check if dataset manifest exists; if not, auto-generate it
    manifest_file = settings.DATASET_DIR / dataset.name / "dataset.yaml"
    if not manifest_file.exists():
        images_with_annots = []
        for img in dataset.images:
            ann_dicts = [
                {"class_id": a.class_id, "bbox_x": a.bbox_x, "bbox_y": a.bbox_y, "bbox_w": a.bbox_w, "bbox_h": a.bbox_h}
                for a in img.annotations
            ]
            images_with_annots.append(({"id": img.id, "filename": img.filename, "file_path": img.file_path, "split": img.split}, ann_dicts))

        manifest_file = DatasetSplitter.generate_yolo_manifest_structure(
            dataset_dir=settings.DATASET_DIR,
            dataset_name=dataset.name,
            classes=dataset.classes,
            images_with_annotations=images_with_annots,
        )

    # Setup run directory: runs/{project_name}/{experiment_name}
    import re
    safe_proj = re.sub(r"[^a-zA-Z0-9_]", "_", project.name)
    safe_exp = re.sub(r"[^a-zA-Z0-9_]", "_", payload.model_name)
    run_dir = settings.RUNS_DIR / safe_proj / safe_exp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create TrainingJob in DB
    job = TrainingJob(
        project_id=project.id,
        dataset_id=dataset.id,
        model_name=payload.model_name,
        architecture=payload.architecture,
        status="pending",
        config=payload.config.model_dump(),
        total_epochs=payload.config.epochs,
        run_dir=str(run_dir),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Start training in background worker
    training_worker.start_job(job_id=job.id, dataset_manifest_path=manifest_file)

    resp = TrainingJobResponse.model_validate(job)
    return resp


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: int, db: AsyncSession = Depends(get_database_session)):
    stmt = (
        select(TrainingJob)
        .options(selectinload(TrainingJob.metrics))
        .filter(TrainingJob.id == job_id)
    )
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")

    resp = TrainingJobResponse.model_validate(job)
    resp.recent_metrics = [
        TrainingMetricResponse.model_validate(m) for m in job.metrics
    ]
    return resp


@router.get("/status/{job_id}", response_model=TrainingJobResponse)
async def get_training_status_alias(job_id: int, db: AsyncSession = Depends(get_database_session)):
    return await get_training_job(job_id=job_id, db=db)


@router.post("/cancel/{job_id}")
async def cancel_training_alias(job_id: int, db: AsyncSession = Depends(get_database_session)):
    return await stop_training(job_id=job_id, db=db)


@router.post("/{job_id}/stop")
async def stop_training(job_id: int, db: AsyncSession = Depends(get_database_session)):
    stmt = select(TrainingJob).filter(TrainingJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")

    stopped = training_worker.stop_job(job_id)
    if stopped:
        job.status = "cancelled"
        await db.commit()
        return {"success": True, "message": "Training job cancelled successfully."}

    return {"success": False, "message": "Job is not currently active."}


@router.get("/{job_id}/logs")
async def get_training_logs(job_id: int, db: AsyncSession = Depends(get_database_session)):
    stmt = select(TrainingJob).filter(TrainingJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")

    log_file = Path(job.run_dir) / "logs" / "training.log"
    if not log_file.exists():
        return {"logs": "Log file not created yet."}

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    return {"logs": content}
