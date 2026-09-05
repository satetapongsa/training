import os
import shutil
import pytest
from pathlib import Path
from PIL import Image

# Setup test data directory
TEST_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    from app.db.session import init_db_sync
    init_db_sync()
    yield
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)



@pytest.fixture
def sample_image_path(tmp_path) -> Path:
    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def mini_dataset_dir(tmp_path) -> Path:
    dset_dir = tmp_path / "mini_dataset"
    dset_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (dset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Create 4 sample images and labels
    for i in range(4):
        img = Image.new("RGB", (128, 128), color=(i * 40, 100, 150))
        img.save(dset_dir / "train" / "images" / f"img_{i}.jpg")
        with open(dset_dir / "train" / "labels" / f"img_{i}.txt", "w") as f:
            f.write(f"0 0.5 0.5 0.4 0.4\n")

    return dset_dir
