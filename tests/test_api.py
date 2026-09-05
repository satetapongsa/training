import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_system_info_endpoints():
    info_resp = client.get("/api/v1/system/info")
    assert info_resp.status_code == 200
    data = info_resp.json()
    assert "pytorch_version" in data
    assert "python_version" in data

    metrics_resp = client.get("/api/v1/system/metrics")
    assert metrics_resp.status_code == 200
    m_data = metrics_resp.json()
    assert "cpu_percent" in m_data
    assert "ram_percent" in m_data


def test_architectures_endpoint():
    resp = client.get("/api/v1/training/architectures")
    assert resp.status_code == 200
    archs = resp.json()
    assert "detection" in archs
    assert "classification" in archs
    assert "yolo11n" in archs["detection"]


def test_end_to_end_project_dataset_flow():
    # 1. Create Project
    p_resp = client.post(
        "/api/v1/projects",
        json={"name": "test_e2e_project", "task_type": "detection", "description": "Automated test project"},
    )
    assert p_resp.status_code in [201, 400]
    if p_resp.status_code == 201:
        proj_id = p_resp.json()["id"]
    else:
        # Fetch existing
        projects = client.get("/api/v1/projects").json()
        proj_id = [p["id"] for p in projects if p["name"] == "test_e2e_project"][0]

    # 2. Create Dataset
    d_resp = client.post(
        "/api/v1/datasets",
        json={"project_id": proj_id, "name": "e2e_dataset", "classes": ["cat", "dog"]},
    )
    assert d_resp.status_code == 201
    dataset_id = d_resp.json()["id"]

    # 3. Upload image
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(200, 50, 50)).save(buf, format="JPEG")
    buf.seek(0)

    upload_resp = client.post(
        f"/api/v1/datasets/{dataset_id}/upload",
        files=[("files", ("cat_test.jpg", buf, "image/jpeg"))],
    )
    assert upload_resp.status_code == 201
    uploaded = upload_resp.json()
    assert uploaded["uploaded_count"] == 1
    image_id = uploaded["uploaded"][0]["id"]

    # 4. Add Annotation
    ann_resp = client.post(
        "/api/v1/annotations",
        json={
            "image_id": image_id,
            "class_id": 0,
            "class_name": "cat",
            "bbox_x": 0.5,
            "bbox_y": 0.5,
            "bbox_w": 0.3,
            "bbox_h": 0.3,
            "confidence": 1.0,
        },
    )
    assert ann_resp.status_code == 201

    # 5. Validate Dataset
    val_resp = client.post(f"/api/v1/datasets/{dataset_id}/validate")
    assert val_resp.status_code == 200
    report = val_resp.json()
    assert report["is_valid"] is True
    assert report["total_images"] == 1
    assert report["total_annotations"] == 1
