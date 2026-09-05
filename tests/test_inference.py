import pytest
from pathlib import Path
from PIL import Image
from app.inference.visualizer import DetectionVisualizer
from app.inference.predictor import PredictionResult


def test_visualizer_draw_detections(sample_image_path, tmp_path):
    detections = [
        {
            "class_id": 0,
            "class_name": "cat",
            "confidence": 0.95,
            "x1": 0.2,
            "y1": 0.2,
            "x2": 0.8,
            "y2": 0.8,
        }
    ]
    img = DetectionVisualizer.draw_detections(sample_image_path, detections)
    assert isinstance(img, Image.Image)
    assert img.size == (320, 240)

    # Convert to bytes
    raw_bytes = DetectionVisualizer.to_bytes(img)
    assert len(raw_bytes) > 0


def test_prediction_result_save_and_dict(sample_image_path, tmp_path):
    pil_img = Image.open(sample_image_path)
    detections = [
        {
            "class_id": 0,
            "class_name": "car",
            "confidence": 0.88,
            "x1": 0.1,
            "y1": 0.1,
            "x2": 0.6,
            "y2": 0.6,
            "box_pixels": [32, 24, 192, 144],
        }
    ]
    res = PredictionResult(
        original_image=pil_img,
        detections=detections,
        inference_time_ms=25.4,
        model_name="test_model",
    )
    d = res.to_dict()
    assert d["model_name"] == "test_model"
    assert d["total_detections"] == 1
    assert d["inference_time_ms"] == 25.4

    out_file = tmp_path / "out_pred.jpg"
    saved = res.save(out_file)
    assert saved.exists()
    assert saved.stat().st_size > 0
