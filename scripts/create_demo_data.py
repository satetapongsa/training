import os
import sys
import io
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw


from app.core.config import settings
from app.db.session import async_engine, Base, AsyncSessionLocal, init_db
from app.db.models import Project, Dataset, Image as DBImage, Annotation, DatasetVersion
from app.datasets.splitter import DatasetSplitter
from app.storage.local import storage


async def seed_demo():
    print("Initializing database and demo dataset...")
    await init_db()

    async with AsyncSessionLocal() as db:
        # 1. Create or get Project
        from sqlalchemy import select
        res = await db.execute(select(Project).filter(Project.name == "Demo Detection Project"))
        project = res.scalar_one_or_none()
        if not project:
            project = Project(
                name="Demo Detection Project",
                description="Pre-configured starter project for quick training and testing.",
                task_type="detection",
            )
            db.add(project)
            await db.commit()
            await db.refresh(project)
            print(f"Created project: {project.name} (ID: {project.id})")

        # 2. Create or get Dataset
        res = await db.execute(select(Dataset).filter(Dataset.name == "shapes_dataset"))
        dataset = res.scalar_one_or_none()
        if not dataset:
            classes = ["circle", "square"]
            dataset = Dataset(
                project_id=project.id,
                name="shapes_dataset",
                description="Geometric shapes dataset with circles and squares.",
                task_type="detection",
                classes=classes,
            )
            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)
            print(f"Created dataset: {dataset.name} (ID: {dataset.id})")

            # 3. Generate 6 sample images with real annotations
            images_with_annots = []
            colors = [(239, 68, 68), (59, 130, 246), (16, 185, 129), (245, 158, 11), (139, 92, 246), (6, 182, 212)]

            for i in range(6):
                w, h = 320, 320
                img = Image.new("RGB", (w, h), color=(15, 23, 42))
                draw = ImageDraw.Draw(img)

                # Draw shapes
                color = colors[i % len(colors)]
                shape_type = i % 2  # 0: circle, 1: square

                # Box coordinates
                x1, y1, x2, y2 = 80, 80, 240, 240
                if shape_type == 0:
                    draw.ellipse([x1, y1, x2, y2], fill=color, outline="#ffffff", width=2)
                else:
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline="#ffffff", width=2)

                # Save to storage
                filename = f"shape_{i+1}.jpg"
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)

                dest_subpath = f"datasets/{dataset.name}/images/{filename}"
                saved_path = storage.save_file(dest_subpath, buf)

                split = "train" if i < 4 else "val"
                db_img = DBImage(
                    dataset_id=dataset.id,
                    filename=filename,
                    file_path=str(saved_path),
                    original_name=filename,
                    mime_type="image/jpeg",
                    file_size=saved_path.stat().st_size,
                    width=w,
                    height=h,
                    split=split,
                    is_annotated=True,
                )
                db.add(db_img)
                await db.commit()
                await db.refresh(db_img)

                # YOLO normalized coords: center (0.5, 0.5), size (0.5, 0.5)
                annot = Annotation(
                    image_id=db_img.id,
                    class_id=shape_type,
                    class_name=classes[shape_type],
                    bbox_x=0.5,
                    bbox_y=0.5,
                    bbox_w=0.5,
                    bbox_h=0.5,
                    confidence=1.0,
                )
                db.add(annot)
                await db.commit()

                images_with_annots.append(({
                    "id": db_img.id,
                    "filename": filename,
                    "file_path": str(saved_path),
                    "split": split,
                }, [{
                    "class_id": shape_type,
                    "bbox_x": 0.5,
                    "bbox_y": 0.5,
                    "bbox_w": 0.5,
                    "bbox_h": 0.5,
                }]))

            # Update dataset counts
            dataset.total_images = 6
            dataset.total_annotations = 6
            dataset.train_count = 4
            dataset.val_count = 2
            dataset.status = "ready"
            await db.commit()

            # Generate physical manifest
            manifest_file = DatasetSplitter.generate_yolo_manifest_structure(
                dataset_dir=settings.DATASET_DIR,
                dataset_name=dataset.name,
                classes=classes,
                images_with_annotations=images_with_annots,
            )

            # Register DatasetVersion
            v = DatasetVersion(
                dataset_id=dataset.id,
                version_tag="v1.0.0",
                manifest_path=str(manifest_file),
                split_ratio={"train": 0.7, "val": 0.2, "test": 0.1},
            )
            db.add(v)
            await db.commit()

            print(f"Created 6 sample images and manifest at: {manifest_file}")


if __name__ == "__main__":
    asyncio.run(seed_demo())
