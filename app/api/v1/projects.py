from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.api.deps import get_database_session
from app.db.models import Project, Dataset, Model, TrainingJob
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get("", response_model=List[ProjectResponse])
async def list_projects(db: AsyncSession = Depends(get_database_session)):
    stmt = select(Project).order_by(Project.created_at.desc())
    result = await db.execute(stmt)
    projects = result.scalars().all()

    response_list = []
    for p in projects:
        # Count datasets
        ds_res = await db.execute(select(func.count(Dataset.id)).filter(Dataset.project_id == p.id))
        ds_count = ds_res.scalar() or 0

        # Count models
        mod_res = await db.execute(select(func.count(Model.id)).filter(Model.project_id == p.id))
        mod_count = mod_res.scalar() or 0

        # Count jobs
        job_res = await db.execute(select(func.count(TrainingJob.id)).filter(TrainingJob.project_id == p.id))
        job_count = job_res.scalar() or 0

        resp = ProjectResponse.model_validate(p)
        resp.datasets_count = ds_count
        resp.models_count = mod_count
        resp.training_jobs_count = job_count
        response_list.append(resp)

    return response_list


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreate, db: AsyncSession = Depends(get_database_session)):
    # Check if project name exists
    existing = await db.execute(select(Project).filter(Project.name == payload.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Project '{payload.name}' already exists.")

    project = Project(
        name=payload.name,
        description=payload.description,
        task_type=payload.task_type,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return ProjectResponse.model_validate(project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, db: AsyncSession = Depends(get_database_session)):
    stmt = select(Project).filter(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return ProjectResponse.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: int, db: AsyncSession = Depends(get_database_session)):
    stmt = select(Project).filter(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    await db.delete(project)
    await db.commit()
