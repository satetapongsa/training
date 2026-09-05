import pytest
from pathlib import Path
from app.datasets.splitter import DatasetSplitter


def test_calculate_splits_ratios():
    images = [{"id": i} for i in range(10)]
    splits = DatasetSplitter.calculate_splits(
        images=images,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )
    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 1
    # Check disjoint
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_ids) == 10


def test_generate_yolo_manifest_structure(tmp_path, sample_image_path):
    dataset_dir = tmp_path / "datasets"
    images_with_annots = [
        (
            {"id": 1, "filename": "sample.jpg", "file_path": str(sample_image_path), "split": "train"},
            [{"class_id": 0, "bbox_x": 0.5, "bbox_y": 0.5, "bbox_w": 0.3, "bbox_h": 0.3}],
        )
    ]
    manifest = DatasetSplitter.generate_yolo_manifest_structure(
        dataset_dir=dataset_dir,
        dataset_name="test_proj_ds",
        classes=["animal"],
        images_with_annotations=images_with_annots,
    )
    assert manifest.exists()
    assert (manifest.parent / "train" / "images" / "sample.jpg").exists()
    assert (manifest.parent / "train" / "labels" / "sample.txt").exists()
