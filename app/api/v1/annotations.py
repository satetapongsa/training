from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.api.deps import get_database_session
from app.db.models import Annotation, Image, Dataset
from app.schemas.annotation import (
    AnnotationCreate,
    AnnotationResponse,
    BatchAnnotationsUpdate,
)

router = APIRouter(prefix="/annotations", tags=["Annotations"])


@router.get("/{image_id}", response_model=List[AnnotationResponse])
async def get_image_annotations(image_id: int, db: AsyncSession = Depends(get_database_session)):
    stmt = select(Annotation).filter(Annotation.image_id == image_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED)
async def create_annotation(payload: AnnotationCreate, db: AsyncSession = Depends(get_database_session)):
    img_res = await db.execute(select(Image).filter(Image.id == payload.image_id))
    img = img_res.scalar_one_or_none()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found.")

    annot = Annotation(
        image_id=payload.image_id,
        class_id=payload.class_id,
        class_name=payload.class_name,
        bbox_x=payload.bbox_x,
        bbox_y=payload.bbox_y,
        bbox_w=payload.bbox_w,
        bbox_h=payload.bbox_h,
        segmentation=payload.segmentation,
        confidence=payload.confidence,
    )
    db.add(annot)
    img.is_annotated = True
    await db.commit()
    await db.refresh(annot)
    return annot


@router.put("/batch", response_model=List[AnnotationResponse])
async def batch_save_annotations(payload: BatchAnnotationsUpdate, db: AsyncSession = Depends(get_database_session)):
    img_res = await db.execute(select(Image).filter(Image.id == payload.image_id))
    img = img_res.scalar_one_or_none()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found.")

    # Remove existing annotations for this image
    await db.execute(delete(Annotation).filter(Annotation.image_id == payload.image_id))

    new_annots = []
    for a in payload.annotations:
        annot = Annotation(
            image_id=payload.image_id,
            class_id=a.class_id,
            class_name=a.class_name,
            bbox_x=a.bbox_x,
            bbox_y=a.bbox_y,
            bbox_w=a.bbox_w,
            bbox_h=a.bbox_h,
            segmentation=a.segmentation,
            confidence=a.confidence,
        )
        db.add(annot)
        new_annots.append(annot)

    img.is_annotated = len(new_annots) > 0
    await db.commit()

    # Update dataset classes list if new classes were created
    if img.dataset_id and new_annots:
        ds_res = await db.execute(select(Dataset).filter(Dataset.id == img.dataset_id))
        ds = ds_res.scalar_one_or_none()
        if ds:
            existing_classes = set(ds.classes or [])
            for a in payload.annotations:
                if a.class_name:
                    existing_classes.add(a.class_name.strip())
            ds.classes = list(existing_classes)
            await db.commit()

    # Re-query
    for annot in new_annots:
        await db.refresh(annot)

    return new_annots


@router.post("/{image_id}", response_model=List[AnnotationResponse])
async def save_image_annotations_alias(
    image_id: int,
    payload: List[AnnotationBase],
    db: AsyncSession = Depends(get_database_session),
):
    batch_payload = BatchAnnotationsUpdate(image_id=image_id, annotations=payload)
    return await batch_save_annotations(payload=batch_payload, db=db)



@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_annotation(annotation_id: int, db: AsyncSession = Depends(get_database_session)):
    res = await db.execute(select(Annotation).filter(Annotation.id == annotation_id))
    annot = res.scalar_one_or_none()
    if not annot:
        raise HTTPException(status_code=404, detail="Annotation not found.")
    await db.delete(annot)
    await db.commit()
