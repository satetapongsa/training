import os
import random
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.core.logging import logger
from app.annotations.yolo import YOLOAnnotationHelper


class DatasetSplitter:
    """Performs stratified dataset splitting and generates training manifests."""

    @staticmethod
    def calculate_splits(
        images: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> Dict[str, List[int]]:
        """
        Splits image IDs into train, val, and test subsets with reproducibility.
        """
        random.seed(seed)
        shuffled = list(images)
        random.shuffle(shuffled)

        total = len(shuffled)
        if total == 0:
            return {"train": [], "val": [], "test": []}

        # Normalize ratios
        ratio_sum = train_ratio + val_ratio + test_ratio
        r_train = train_ratio / ratio_sum
        r_val = val_ratio / ratio_sum

        n_train = max(1, int(total * r_train)) if total > 2 else 1
        n_val = max(1, int(total * r_val)) if total - n_train > 1 else (1 if total > 1 else 0)
        n_test = total - n_train - n_val
        if n_test < 0:
            n_val = max(0, total - n_train)
            n_test = 0

        train_imgs = shuffled[:n_train]
        val_imgs = shuffled[n_train : n_train + n_val]
        test_imgs = shuffled[n_train + n_val :]

        return {
            "train": [img["id"] for img in train_imgs],
            "val": [img["id"] for img in val_imgs],
            "test": [img["id"] for img in test_imgs],
        }

    @staticmethod
    def generate_yolo_manifest_structure(
        dataset_dir: Path,
        dataset_name: str,
        classes: List[str],
        images_with_annotations: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
    ) -> Path:
        """
        Prepares physical YOLO-compatible folder structure and dataset.yaml:
        dataset_dir/
            train/images, train/labels
            val/images, val/labels
            test/images, test/labels
            dataset.yaml
        """
        target_dir = dataset_dir / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            (target_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (target_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        for img, annots in images_with_annotations:
            split = img.get("split", "train")
            src_img_path = Path(img["file_path"])
            dest_img_path = target_dir / split / "images" / img["filename"]

            # Copy or link image
            if src_img_path.exists() and not dest_img_path.exists():
                try:
                    # Try hardlink first for speed and zero duplicate disk usage
                    os.link(src_img_path, dest_img_path)
                except Exception:
                    import shutil
                    shutil.copy2(src_img_path, dest_img_path)

            # Write label txt file
            label_stem = Path(img["filename"]).stem
            label_path = target_dir / split / "labels" / f"{label_stem}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                for a in annots:
                    line = YOLOAnnotationHelper.format_line(
                        a["class_id"], a["bbox_x"], a["bbox_y"], a["bbox_w"], a["bbox_h"]
                    )
                    f.write(f"{line}\n")

        # Create dataset.yaml
        yaml_content = {
            "path": str(target_dir.resolve()).replace("\\", "/"),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {i: name for i, name in enumerate(classes)},
            "nc": len(classes),
        }

        yaml_file = target_dir / "dataset.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        # Also write classes.txt
        with open(target_dir / "classes.txt", "w", encoding="utf-8") as f:
            for cls_name in classes:
                f.write(f"{cls_name}\n")

        logger.info(f"YOLO dataset manifest generated at: {yaml_file}")
        return yaml_file
