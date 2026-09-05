import pytest
from pathlib import Path
from app.annotations.yolo import YOLOAnnotationHelper
from app.datasets.validator import DatasetValidator
from app.datasets.augmentor import DatasetAugmentor


def test_yolo_annotation_parsing():
    line = "0 0.500000 0.500000 0.200000 0.300000"
    cid, x, y, w, h = YOLOAnnotationHelper.parse_line(line)
    assert cid == 0
    assert abs(x - 0.5) < 1e-4
    assert abs(y - 0.5) < 1e-4
    assert abs(w - 0.2) < 1e-4
    assert abs(h - 0.3) < 1e-4

    formatted = YOLOAnnotationHelper.format_line(cid, x, y, w, h)
    assert formatted.startswith("0 0.500000 0.500000")


def test_yolo_coordinate_conversions():
    # Convert normalized (0.5, 0.5, 0.4, 0.4) on 100x100 image
    x1, y1, x2, y2 = YOLOAnnotationHelper.yolo_to_xyxy(0.5, 0.5, 0.4, 0.4, 100, 100)
    assert x1 == 30 and y1 == 30
    assert x2 == 70 and y2 == 70

    # Inverse
    nx, ny, nw, nh = YOLOAnnotationHelper.xyxy_to_yolo(30, 30, 70, 70, 100, 100)
    assert abs(nx - 0.5) < 1e-3
    assert abs(ny - 0.5) < 1e-3


def test_dataset_validator_success(sample_image_path):
    images = [{"id": 1, "filename": sample_image_path.name, "file_path": str(sample_image_path), "split": "train"}]
    annots = {1: [{"class_id": 0, "class_name": "cat", "bbox_x": 0.5, "bbox_y": 0.5, "bbox_w": 0.2, "bbox_h": 0.2}]}
    report = DatasetValidator.validate_dataset_records(
        dataset_name="test_dset",
        classes=["cat"],
        images=images,
        annotations_by_image_id=annots,
    )
    assert report["is_valid"] is True
    assert report["total_images"] == 1
    assert report["total_annotations"] == 1
    assert report["errors_count"] == 0


def test_dataset_validator_detects_out_of_bounds_and_bad_classes(sample_image_path):
    images = [{"id": 1, "filename": sample_image_path.name, "file_path": str(sample_image_path)}]
    # Class ID 5 is out of bounds for classes=["cat"]
    # Bbox x=1.5 is out of bounds
    annots = {1: [{"class_id": 5, "bbox_x": 1.5, "bbox_y": 0.5, "bbox_w": 0.2, "bbox_h": 0.2}]}
    report = DatasetValidator.validate_dataset_records(
        dataset_name="bad_dset",
        classes=["cat"],
        images=images,
        annotations_by_image_id=annots,
    )
    assert report["is_valid"] is False
    assert report["errors_count"] >= 2


def test_augmentation_preview(sample_image_path):
    annots = [{"class_id": 0, "class_name": "cat", "bbox_x": 0.5, "bbox_y": 0.5, "bbox_w": 0.2, "bbox_h": 0.2}]
    cfg = {"horizontal_flip": 1.0, "rotation": 10}
    preview_bytes = DatasetAugmentor.generate_preview(sample_image_path, annots, cfg)
    assert isinstance(preview_bytes, bytes)
    assert len(preview_bytes) > 500  # Valid JPEG bytes
