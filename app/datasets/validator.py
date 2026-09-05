from pathlib import Path
from typing import Dict, Any, List, Set
from collections import Counter
from app.core.security import verify_image_file, calculate_sha256
from app.core.logging import logger


class DatasetValidator:
    """Production validator for Object Detection and Classification datasets."""

    @staticmethod
    def validate_dataset_records(
        dataset_name: str,
        classes: List[str],
        images: List[Dict[str, Any]],
        annotations_by_image_id: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Validates database-backed dataset records and returns detailed report.
        """
        errors: List[str] = []
        warnings: List[str] = []
        seen_hashes: Dict[str, str] = {}
        class_counter = Counter()
        split_counter = Counter()
        annotated_count = 0

        num_classes = len(classes)
        if num_classes == 0:
            errors.append("Dataset has no classes defined.")

        for img in images:
            img_id = img["id"]
            filename = img["filename"]
            file_path = Path(img["file_path"])
            split = img.get("split", "train")
            split_counter[split] += 1

            # 1. File existence and integrity
            if not file_path.exists():
                errors.append(f"Image '{filename}' (ID {img_id}) missing from disk: {file_path}")
                continue

            valid, fmt_or_err, w, h = verify_image_file(file_path)
            if not valid:
                errors.append(f"Image '{filename}' verification failed: {fmt_or_err}")
                continue

            # 2. Duplicate detection
            img_hash = img.get("sha256") or calculate_sha256(file_path)
            if img_hash in seen_hashes:
                warnings.append(
                    f"Duplicate image content detected: '{filename}' is identical to '{seen_hashes[img_hash]}'"
                )
            else:
                seen_hashes[img_hash] = filename

            # 3. Check Annotations
            annots = annotations_by_image_id.get(img_id, [])
            if annots:
                annotated_count += 1

            for ann in annots:
                c_id = ann["class_id"]
                c_name = ann.get("class_name", "")
                bx, by, bw, bh = ann["bbox_x"], ann["bbox_y"], ann["bbox_w"], ann["bbox_h"]

                # Check class boundaries
                if c_id < 0 or c_id >= num_classes:
                    errors.append(
                        f"Image '{filename}': Invalid class_id {c_id} (allowed: 0..{num_classes-1})."
                    )
                else:
                    class_counter[classes[c_id]] += 1

                # Check box normalization boundaries
                if not (0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0):
                    errors.append(
                        f"Image '{filename}': Bounding box center ({bx:.3f}, {by:.3f}) out of bounds [0, 1]."
                    )
                if bw <= 0.0 or bw > 1.0 or bh <= 0.0 or bh > 1.0:
                    errors.append(
                        f"Image '{filename}': Bounding box dimension w={bw:.3f}, h={bh:.3f} invalid."
                    )

        total_images = len(images)
        if total_images == 0:
            errors.append("Dataset contains 0 images.")
        elif annotated_count == 0 and num_classes > 0:
            warnings.append("Dataset has no labeled images yet.")

        is_valid = len(errors) == 0

        return {
            "dataset_name": dataset_name,
            "is_valid": is_valid,
            "total_images": total_images,
            "annotated_images": annotated_count,
            "unannotated_images": total_images - annotated_count,
            "total_annotations": sum(class_counter.values()),
            "errors_count": len(errors),
            "warnings_count": len(warnings),
            "errors": errors[:50],  # Cap for UI display
            "warnings": warnings[:50],
            "class_distribution": dict(class_counter),
            "split_distribution": dict(split_counter),
        }
