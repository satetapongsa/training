import pytest
from pathlib import Path
from PIL import Image
import torch

from app.core.config import settings
from app.datasets.splitter import DatasetSplitter
from app.training.trainers.detection_yolo import YOLODetectionTrainer
from app.inference.predictor import Predictor
from app.models_registry.exporter import YOLONNExporter


@pytest.mark.slow
def test_real_training_and_inference_pipeline(tmp_path):
    # 1. Setup mini real dataset
    dataset_dir = tmp_path / "test_dataset"
    classes = ["circle", "square"]
    
    # Create 6 synthetic images with drawn shapes
    images_with_annots = []
    for i in range(6):
        img = Image.new("RGB", (160, 160), color=(30 + i * 20, 40, 50))
        img_path = tmp_path / f"synth_{i}.jpg"
        img.save(img_path)
        # Class 0 or 1, centered
        annots = [{"class_id": i % 2, "bbox_x": 0.5, "bbox_y": 0.5, "bbox_w": 0.4, "bbox_h": 0.4}]
        images_with_annots.append(({"id": i, "filename": img_path.name, "file_path": str(img_path), "split": "train" if i < 4 else "val"}, annots))

    # 2. Generate YOLO dataset manifest
    manifest_yaml = DatasetSplitter.generate_yolo_manifest_structure(
        dataset_dir=dataset_dir,
        dataset_name="mini_cv_dataset",
        classes=classes,
        images_with_annotations=images_with_annots,
    )
    assert manifest_yaml.exists()

    # 3. Setup real YOLO Detection Trainer (2 epochs, lightweight)
    run_dir = tmp_path / "run_e2e_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = YOLODetectionTrainer(
        job_id=101,
        config={
            "architecture": "yolo11n",
            "epochs": 2,
            "batch_size": 2,
            "image_size": 160,
            "learning_rate": 0.001,
            "device": "cpu",
            "seed": 42,
        },
        run_dir=run_dir,
        on_progress_callback=lambda p: None,
    )
    trainer.setup(manifest_yaml)

    # 4. Train real model
    final_metrics = trainer.train()
    assert isinstance(final_metrics, dict)

    # 5. Verify physical weights were created
    best_pt = run_dir / "checkpoints" / "best.pt"
    assert best_pt.exists(), f"Expected best.pt at {best_pt}"
    assert best_pt.stat().st_size > 1000  # Non-empty real model weight

    # 6. Load real weight with Predictor and run inference
    predictor = Predictor(best_pt, device="cpu")
    test_img = Image.new("RGB", (160, 160), color=(100, 150, 200))
    res = predictor.predict(test_img, conf_threshold=0.01)
    assert res.inference_time_ms > 0
    saved_annotated = res.save(tmp_path / "detection_output.jpg")
    assert saved_annotated.exists()
    assert saved_annotated.stat().st_size > 0

    # 7. Real ONNX Export
    onnx_out = tmp_path / "model.onnx"
    exporter = YOLONNExporter()
    exported = exporter.export(
        weight_path=best_pt,
        output_path=onnx_out,
        config={"format": "onnx", "image_size": 160},
    )
    assert exported.exists()
    assert exported.stat().st_size > 1000
