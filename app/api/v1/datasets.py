import io
import re
import zipfile
import yaml
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


async def _extract_and_import_zip(
    zip_bytes: bytes,
    dataset: Dataset,
    db: AsyncSession,
) -> Dict[str, Any]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ไฟล์ไม่ใช่ ZIP ที่ถูกต้อง: {str(e)}")

    namelist = [n for n in zf.namelist() if not n.startswith("__MACOSX/") and not n.endswith("/")]

    # 1. Discover classes from classes.txt or dataset.yaml
    extracted_classes = list(dataset.classes) if dataset.classes else []
    for fname in namelist:
        lower_name = fname.lower()
        if lower_name.endswith("classes.txt"):
            try:
                content = zf.read(fname).decode("utf-8", errors="replace")
                cls_lines = [c.strip() for c in content.splitlines() if c.strip()]
                if cls_lines:
                    for c in cls_lines:
                        if c not in extracted_classes:
                            extracted_classes.append(c)
            except Exception:
                pass
        elif lower_name.endswith("dataset.yaml") or lower_name.endswith("data.yaml"):
            try:
                content = zf.read(fname).decode("utf-8", errors="replace")
                yaml_data = yaml.safe_load(content)
                raw_names = yaml_data.get("names", [])
                if isinstance(raw_names, dict):
                    names_list = [raw_names[k] for k in sorted(raw_names.keys())]
                elif isinstance(raw_names, list):
                    names_list = raw_names
                else:
                    names_list = []
                for c in names_list:
                    if str(c) not in extracted_classes:
                        extracted_classes.append(str(c))
            except Exception:
                pass

    if not extracted_classes:
        extracted_classes = ["object"]
    dataset.classes = extracted_classes

    # 2. Collect label txt files: map stem -> text content
    label_files: Dict[str, str] = {}
    image_entries: List[str] = []
    valid_img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for fname in namelist:
        p = Path(fname)
        ext = p.suffix.lower()
        if ext == ".txt" and p.name.lower() not in ("classes.txt", "requirements.txt"):
            try:
                txt_content = zf.read(fname).decode("utf-8", errors="replace")
                label_files[p.stem.lower()] = txt_content
            except Exception:
                pass
        elif ext in valid_img_exts:
            image_entries.append(fname)

    if not image_entries:
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์รูปภาพ (.jpg, .png, .webp, .bmp) ในไฟล์ ZIP นี้")

    uploaded = []
    errors = []
    total_annotations_added = 0

    for img_entry in image_entries:
        try:
            content = zf.read(img_entry)
            raw_name = Path(img_entry).name
            rec = await DatasetManager.process_and_save_image(
                db=db,
                dataset=dataset,
                filename=raw_name,
                file_stream=io.BytesIO(content),
                original_name=raw_name,
            )
            if rec:
                uploaded.append({"id": rec.id, "filename": rec.filename})
                stem = Path(raw_name).stem.lower()
                if stem in label_files:
                    lines = label_files[stem].splitlines()
                    has_annot = False
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                cid = int(parts[0])
                                if cid < len(dataset.classes):
                                    c_name = dataset.classes[cid]
                                else:
                                    c_name = f"class_{cid}"
                                    if c_name not in dataset.classes:
                                        dataset.classes.append(c_name)

                                if len(parts) == 5:
                                    cx = float(parts[1])
                                    cy = float(parts[2])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    seg = None
                                else:
                                    coords = [float(x) for x in parts[1:]]
                                    xs = coords[0::2]
                                    ys = coords[1::2]
                                    if xs and ys:
                                        xmin = min(xs)
                                        xmax = max(xs)
                                        ymin = min(ys)
                                        ymax = max(ys)
                                        cx = (xmin + xmax) / 2.0
                                        cy = (ymin + ymax) / 2.0
                                        w = max(0.001, xmax - xmin)
                                        h = max(0.001, ymax - ymin)
                                        seg = [[round(xs[i], 6), round(ys[i], 6)] for i in range(len(xs))]
                                    else:
                                        continue

                                annot = Annotation(
                                    image_id=rec.id,
                                    class_id=cid,
                                    class_name=c_name,
                                    bbox_x=cx,
                                    bbox_y=cy,
                                    bbox_w=w,
                                    bbox_h=h,
                                    segmentation=seg,
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
                errors.append(f"Failed verification for {raw_name}")
        except Exception as e:
            errors.append(f"Upload error {img_entry}: {str(e)}")

    if total_annotations_added > 0:
        dataset.total_annotations += total_annotations_added

    # Auto-split & generate physical YOLO structure
    try:
        img_res = await db.execute(
            select(Image).options(selectinload(Image.annotations)).filter(Image.dataset_id == dataset.id)
        )
        all_imgs = img_res.scalars().all()
        if all_imgs:
            image_dicts = [{"id": img.id} for img in all_imgs]
            splits = DatasetSplitter.calculate_splits(
                images=image_dicts,
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0,
            )
            img_map = {img.id: img for img in all_imgs}
            train_count = 0
            val_count = 0
            for split_name, ids in splits.items():
                for i_id in ids:
                    if i_id in img_map:
                        img_map[i_id].split = split_name
                        if split_name == "train":
                            train_count += 1
                        elif split_name == "val":
                            val_count += 1

            dataset.train_count = train_count
            dataset.val_count = val_count

            images_with_annots = []
            for img in all_imgs:
                ann_dicts = [
                    {
                        "class_id": a.class_id,
                        "bbox_x": a.bbox_x,
                        "bbox_y": a.bbox_y,
                        "bbox_w": a.bbox_w,
                        "bbox_h": a.bbox_h,
                        "segmentation": a.segmentation,
                    }
                    for a in img.annotations
                ]
                images_with_annots.append(
                    ({"id": img.id, "filename": img.filename, "file_path": img.file_path, "split": img.split}, ann_dicts)
                )

            DatasetSplitter.generate_yolo_manifest_structure(
                dataset_dir=settings.DATASET_DIR,
                dataset_name=dataset.name,
                classes=dataset.classes,
                images_with_annotations=images_with_annots,
            )
    except Exception as se:
        logger.warning(f"Auto-split warning on zip import: {se}")

    await db.commit()
    await db.refresh(dataset)

    return {
        "success": True,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "uploaded_count": len(uploaded),
        "imported_annotations": total_annotations_added,
        "classes": dataset.classes,
        "errors_count": len(errors),
        "errors": errors,
    }


@router.post("/upload-zip")
async def upload_dataset_zip(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    project_id: Optional[int] = Form(None),
    task_type: Optional[str] = Form("detection"),
    db: AsyncSession = Depends(get_database_session),
):
    """Upload a ZIP containing images and companion GT labels to create a new dataset."""
    project = None
    if project_id:
        proj_res = await db.execute(select(Project).filter(Project.id == project_id))
        project = proj_res.scalar_one_or_none()
    if not project:
        proj_first = await db.execute(select(Project).order_by(Project.id))
        project = proj_first.scalars().first()
        if not project:
            project = Project(name="Workspace", task_type=task_type or "detection")
            db.add(project)
            await db.commit()
            await db.refresh(project)

    import time
    clean_base = Path(file.filename or "dataset").stem
    clean_base = re.sub(r"[^a-zA-Z0-9_\-\u0E00-\u0E7F]", "_", clean_base)
    name = dataset_name or f"{clean_base}_{int(time.time())}"

    dataset = Dataset(
        project_id=project.id,
        name=name,
        description=f"นำเข้าจากไฟล์ ZIP: {file.filename}",
        task_type=task_type or "detection",
        classes=["object"],
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    zip_bytes = await file.read()
    result = await _extract_and_import_zip(zip_bytes=zip_bytes, dataset=dataset, db=db)
    result["dataset"] = DatasetResponse.model_validate(dataset)
    return result


@router.post("/{dataset_id}/upload-zip")
async def upload_zip_to_existing_dataset(
    dataset_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_database_session),
):
    """Upload a ZIP file into an existing dataset."""
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    zip_bytes = await file.read()
    res = await _extract_and_import_zip(zip_bytes=zip_bytes, dataset=dataset, db=db)
    res["dataset"] = DatasetResponse.model_validate(dataset)
    return res


@router.get("/{dataset_id}/download-zip")
async def download_dataset_zip(
    dataset_id: int,
    db: AsyncSession = Depends(get_database_session),
):
    """Packages the dataset images, YOLO Ground Truth labels, classes.txt, and dataset.yaml into a ZIP file."""
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    img_res = await db.execute(
        select(Image).options(selectinload(Image.annotations)).filter(Image.dataset_id == dataset_id)
    )
    images = img_res.scalars().all()
    if not images:
        raise HTTPException(status_code=400, detail="ชุดข้อมูลนี้ยังไม่มีรูปภาพสำหรับดาวน์โหลด")

    classes_list = list(dataset.classes) if dataset.classes else ["object"]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. classes.txt
        zf.writestr("classes.txt", "\n".join(classes_list) + "\n")

        # 2. dataset.yaml
        yaml_data = {
            "path": "./",
            "train": "images",
            "val": "images",
            "names": {i: name for i, name in enumerate(classes_list)},
            "nc": len(classes_list),
        }
        zf.writestr("dataset.yaml", yaml.dump(yaml_data, sort_keys=False))

        # 3. images/ and labels/
        for img in images:
            img_filename = img.original_name or img.filename
            img_path = Path(img.file_path)

            if img_path.exists():
                try:
                    zf.write(img_path, f"images/{img_filename}")
                except Exception as e:
                    logger.warning(f"Error packing image {img_filename} to zip: {e}")

            # Ground Truth label .txt
            label_stem = Path(img_filename).stem
            label_lines = []
            for a in img.annotations:
                cid = a.class_id
                if a.segmentation and len(a.segmentation) >= 3:
                    pts_str = " ".join(f"{float(pt[0]):.6f} {float(pt[1]):.6f}" for pt in a.segmentation)
                    label_lines.append(f"{cid} {pts_str}")
                else:
                    label_lines.append(
                        f"{cid} {float(a.bbox_x):.6f} {float(a.bbox_y):.6f} {float(a.bbox_w):.6f} {float(a.bbox_h):.6f}"
                    )

            zf.writestr(f"labels/{label_stem}.txt", "\n".join(label_lines) + ("\n" if label_lines else ""))

    zip_buffer.seek(0)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", dataset.name)
    filename = f"{safe_name}_gt.zip"
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
            {
                "class_id": a.class_id,
                "bbox_x": a.bbox_x,
                "bbox_y": a.bbox_y,
                "bbox_w": a.bbox_w,
                "bbox_h": a.bbox_h,
                "segmentation": a.segmentation,
            }
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
