import io
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, Query
from fastapi.responses import Response, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.api.deps import get_database_session
from app.core.config import settings
from app.core.logging import logger
from app.db.models import Dataset, Image, Annotation, Project, DatasetVersion
from app.schemas.dataset import (
    DatasetCreate,
    DatasetResponse,
    ImageResponse,
    DatasetSplitRequest,
    AugmentationConfig,
)
from app.datasets.manager import DatasetManager
from app.datasets.validator import DatasetValidator
from app.datasets.splitter import DatasetSplitter
from app.datasets.augmentor import DatasetAugmentor

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    project_id: Optional[int] = None,
    db: AsyncSession = Depends(get_database_session),
):
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if project_id:
        stmt = stmt.filter(Dataset.project_id == project_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(payload: DatasetCreate, db: AsyncSession = Depends(get_database_session)):
    # Verify or auto-create project
    project = None
    if payload.project_id:
        proj_res = await db.execute(select(Project).filter(Project.id == payload.project_id))
        project = proj_res.scalar_one_or_none()
    if not project:
        proj_first = await db.execute(select(Project).order_by(Project.id))
        project = proj_first.scalars().first()
        if not project:
            project = Project(name="Workspace", task_type=payload.task_type or "detection")
            db.add(project)
            await db.commit()
            await db.refresh(project)
        payload.project_id = project.id

    dataset = Dataset(
        project_id=payload.project_id,
        name=payload.name,
        description=payload.description,
        task_type=payload.task_type,
        classes=payload.classes,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_database_session)):
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset


@router.post("/{dataset_id}/upload", status_code=status.HTTP_201_CREATED)
async def upload_images(
    dataset_id: int,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_database_session),
):
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    uploaded = []
    errors = []
    
    # Separate image files and potential label .txt files
    image_files = []
    label_files = {}  # stem -> text content

    for f in files:
        fname = Path(f.filename).name
        suffix = Path(f.filename).suffix.lower()
        if suffix == ".txt":
            try:
                content = (await f.read()).decode("utf-8", errors="replace")
                label_files[Path(fname).stem] = content
            except Exception:
                pass
        else:
            image_files.append(f)

    # Process images
    total_annotations_added = 0
    for file in image_files:
        try:
            content = await file.read()
            raw_name = Path(file.filename).name
            rec = await DatasetManager.process_and_save_image(
                db=db,
                dataset=dataset,
                filename=raw_name,
                file_stream=io.BytesIO(content),
                original_name=raw_name,
            )
            if rec:
                uploaded.append({"id": rec.id, "filename": rec.filename})

                # Check if matching label file exists
                stem = Path(raw_name).stem
                if stem in label_files:
                    lines = label_files[stem].splitlines()
                    has_annot = False
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            from app.annotations.yolo import YOLOAnnotationHelper
                            cid, bx, by, bw, bh = YOLOAnnotationHelper.parse_line(line)
                            c_name = dataset.classes[cid] if cid < len(dataset.classes) else f"class_{cid}"
                            annot = Annotation(
                                image_id=rec.id,
                                class_id=cid,
                                class_name=c_name,
                                bbox_x=bx,
                                bbox_y=by,
                                bbox_w=bw,
                                bbox_h=bh,
                                confidence=1.0,
                            )
                            db.add(annot)
                            total_annotations_added += 1
                            has_annot = True
                        except Exception as pe:
                            errors.append(f"Label parse error in {stem}.txt: {pe}")
                    
                    if has_annot:
                        rec.is_annotated = True
                        await db.commit()
            else:
                errors.append(f"Failed verification for {file.filename}")
        except Exception as e:
            errors.append(f"Upload error {file.filename}: {str(e)}")

    if total_annotations_added > 0:
        dataset.total_annotations += total_annotations_added
        await db.commit()

    return {
        "success": True,
        "uploaded_count": len(uploaded),
        "imported_annotations": total_annotations_added,
        "errors_count": len(errors),
        "uploaded": uploaded,
        "errors": errors,
    }



@router.post("/{dataset_id}/import-folder")
async def import_folder(
    dataset_id: int,
    folder_path: str = Form(...),
    db: AsyncSession = Depends(get_database_session),
):
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        imported_imgs, imported_annots, errors = await DatasetManager.import_folder(
            db=db, dataset=dataset, folder_path_str=folder_path
        )
        return {
            "success": True,
            "imported_images": imported_imgs,
            "imported_annotations": imported_annots,
            "errors": errors,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/images", response_model=List[ImageResponse])
async def list_dataset_images(
    dataset_id: int,
    skip: int = 0,
    limit: int = 100,
    split: Optional[str] = None,
    db: AsyncSession = Depends(get_database_session),
):
    stmt = (
        select(Image)
        .options(selectinload(Image.annotations))
        .filter(Image.dataset_id == dataset_id)
        .order_by(Image.id.asc())
        .offset(skip)
        .limit(limit)
    )
    if split:
        stmt = stmt.filter(Image.split == split)

    result = await db.execute(stmt)
    images = result.scalars().all()

    resp = []
    for img in images:
        item = ImageResponse.model_validate(img)
        item.image_url = f"/api/v1/datasets/images/{img.id}/file"
        resp.append(item)
    return resp


@router.get("/images/{image_id}/file")
async def get_image_file(image_id: int, db: AsyncSession = Depends(get_database_session)):
    result = await db.execute(select(Image).filter(Image.id == image_id))
    img = result.scalar_one_or_none()
    if not img or not Path(img.file_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found.")
    return FileResponse(img.file_path, media_type=img.mime_type)


@router.post("/{dataset_id}/validate")
async def validate_dataset(dataset_id: int, db: AsyncSession = Depends(get_database_session)):
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Fetch all images and annotations
    img_res = await db.execute(
        select(Image).options(selectinload(Image.annotations)).filter(Image.dataset_id == dataset_id)
    )
    images = img_res.scalars().all()

    image_dicts = [
        {"id": img.id, "filename": img.filename, "file_path": img.file_path, "split": img.split, "sha256": img.sha256}
        for img in images
    ]

    annots_by_img = {}
    for img in images:
        annots_by_img[img.id] = [
            {
                "class_id": a.class_id,
                "class_name": a.class_name,
                "bbox_x": a.bbox_x,
                "bbox_y": a.bbox_y,
                "bbox_w": a.bbox_w,
                "bbox_h": a.bbox_h,
            }
            for a in img.annotations
        ]

    report = DatasetValidator.validate_dataset_records(
        dataset_name=dataset.name,
        classes=dataset.classes,
        images=image_dicts,
        annotations_by_image_id=annots_by_img,
    )

    dataset.validation_report = report
    dataset.status = "ready" if report["is_valid"] else "error"
    await db.commit()

    return report


@router.post("/{dataset_id}/split")
async def split_dataset(
    dataset_id: int,
    payload: DatasetSplitRequest,
    db: AsyncSession = Depends(get_database_session),
):
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    img_res = await db.execute(
        select(Image).options(selectinload(Image.annotations)).filter(Image.dataset_id == dataset_id)
    )
    images = img_res.scalars().all()
    if not images:
        raise HTTPException(status_code=400, detail="Cannot split an empty dataset.")

    image_dicts = [{"id": img.id} for img in images]
    splits = DatasetSplitter.calculate_splits(
        images=image_dicts,
        train_ratio=payload.train_ratio,
        val_ratio=payload.val_ratio,
        test_ratio=payload.test_ratio,
        seed=payload.seed,
    )

    # Update splits on Image records
    img_map = {img.id: img for img in images}
    train_count = 0
    val_count = 0
    test_count = 0

    for split_name, ids in splits.items():
        for i_id in ids:
            if i_id in img_map:
                img_map[i_id].split = split_name
                if split_name == "train":
                    train_count += 1
                elif split_name == "val":
                    val_count += 1
                elif split_name == "test":
                    test_count += 1

    dataset.train_count = train_count
    dataset.val_count = val_count
    dataset.test_count = test_count

    # Generate physical YOLO manifest structure
    images_with_annots = []
    for img in images:
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

    # Create DatasetVersion record
    version = DatasetVersion(
        dataset_id=dataset.id,
        version_tag="v1.0.0",
        manifest_path=str(manifest_file),
        split_ratio={
            "train": payload.train_ratio,
            "val": payload.val_ratio,
            "test": payload.test_ratio,
        },
    )
    db.add(version)
    await db.commit()

    return {
        "success": True,
        "splits": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "manifest_path": str(manifest_file),
    }


@router.post("/{dataset_id}/augment-preview")
async def augment_preview(
    dataset_id: int,
    image_id: int = Query(...),
    config: AugmentationConfig = AugmentationConfig(),
    db: AsyncSession = Depends(get_database_session),
):
    img_res = await db.execute(
        select(Image).options(selectinload(Image.annotations)).filter(Image.id == image_id, Image.dataset_id == dataset_id)
    )
    img = img_res.scalar_one_or_none()
    if not img or not Path(img.file_path).exists():
        raise HTTPException(status_code=404, detail="Image not found.")

    annots = [
        {"class_id": a.class_id, "class_name": a.class_name, "bbox_x": a.bbox_x, "bbox_y": a.bbox_y, "bbox_w": a.bbox_w, "bbox_h": a.bbox_h}
        for a in img.annotations
    ]

    try:
        jpeg_bytes = DatasetAugmentor.generate_preview(
            image_path=Path(img.file_path),
            annotations=annots,
            config=config.model_dump(),
        )
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Augmentation preview failed: {str(e)}")


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_database_session)):
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    await db.delete(dataset)
    await db.commit()
