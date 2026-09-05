import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, BinaryIO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func

from app.core.config import settings
from app.core.security import verify_image_file, calculate_sha256, sanitize_filename
from app.core.logging import logger
from app.storage.local import storage
from app.db.models import Dataset, Image, Annotation
from app.annotations.yolo import YOLOAnnotationHelper


class DatasetManager:
    """High-level manager for ingestion, folder scanning, and annotation synchronization."""

    @staticmethod
    async def process_and_save_image(
        db: AsyncSession,
        dataset: Dataset,
        filename: str,
        file_stream: BinaryIO,
        original_name: str,
    ) -> Optional[Image]:
        """Saves an image, checks validity, and stores metadata in DB."""
        safe_name = sanitize_filename(filename)
        dest_subpath = f"datasets/{dataset.name}/images/{safe_name}"

        # Write to storage
        saved_path = storage.save_file(dest_subpath, file_stream)

        # Deep verification
        is_valid, mime_or_err, w, h = verify_image_file(saved_path)
        if not is_valid:
            logger.warning(f"File {filename} rejected: {mime_or_err}")
            storage.delete_file(dest_subpath)
            return None

        file_size = saved_path.stat().st_size
        img_hash = calculate_sha256(saved_path)

        image_record = Image(
            dataset_id=dataset.id,
            filename=safe_name,
            file_path=str(saved_path),
            original_name=original_name,
            mime_type=f"image/{mime_or_err}",
            file_size=file_size,
            width=w,
            height=h,
            split="train",
            is_annotated=False,
            sha256=img_hash,
        )
        db.add(image_record)
        await db.commit()
        await db.refresh(image_record)

        # Update dataset image counter
        dataset.total_images += 1
        dataset.train_count += 1
        await db.commit()

        return image_record

    @staticmethod
    async def import_folder(
        db: AsyncSession,
        dataset: Dataset,
        folder_path_str: str,
    ) -> Tuple[int, int, List[str]]:
        """
        Scans a local directory, imports valid images, and parses matching .txt YOLO annotations.
        Returns: (imported_images_count, imported_annotations_count, errors)
        """
        folder = Path(folder_path_str)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path_str}")

        imported_images = 0
        imported_annotations = 0
        errors: List[str] = []

        allowed_exts = {f".{ext}" for ext in settings.ALLOWED_IMAGE_EXTENSIONS}
        image_files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in allowed_exts]

        for img_path in image_files:
            try:
                with open(img_path, "rb") as f:
                    rec = await DatasetManager.process_and_save_image(
                        db=db,
                        dataset=dataset,
                        filename=img_path.name,
                        file_stream=f,
                        original_name=img_path.name,
                    )
                if not rec:
                    errors.append(f"Image verification failed for: {img_path.name}")
                    continue

                imported_images += 1

                # Check if matching YOLO label file exists (e.g. img_01.txt)
                label_path = img_path.with_suffix(".txt")
                if not label_path.exists():
                    # Also check ../labels/img_01.txt
                    alt_label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
                    if alt_label_path.exists():
                        label_path = alt_label_path

                if label_path.exists():
                    with open(label_path, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()
                    
                    has_annot = False
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
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
                            imported_annotations += 1
                            has_annot = True
                        except Exception as pe:
                            errors.append(f"Label parse error {label_path.name}: {pe}")
                    
                    if has_annot:
                        rec.is_annotated = True
                        await db.commit()

            except Exception as ex:
                errors.append(f"Error importing {img_path.name}: {str(ex)}")

        # Update dataset totals
        dataset.total_annotations += imported_annotations
        await db.commit()

        return imported_images, imported_annotations, errors
