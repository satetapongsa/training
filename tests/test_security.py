import pytest
from pathlib import Path
from PIL import Image
from app.core.security import (
    sanitize_filename,
    is_safe_path,
    calculate_sha256,
    verify_image_file,
)


def test_sanitize_filename():
    assert sanitize_filename("safe_image.jpg") == "safe_image.jpg"
    assert sanitize_filename("../../evil.png") == "evil.png"
    assert sanitize_filename("image\x00with_null.jpg") == "image_with_null.jpg"
    assert sanitize_filename("....double_dot.png") == "file_double_dot.png"



def test_path_traversal_detection(tmp_path):
    base_dir = tmp_path / "sandbox"
    base_dir.mkdir()

    safe_target = base_dir / "uploads" / "image.png"
    assert is_safe_path(base_dir, safe_target) is True

    unsafe_target = base_dir / ".." / "system32"
    assert is_safe_path(base_dir, unsafe_target) is False


def test_verify_image_file(tmp_path):
    # Valid JPEG
    jpg_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(jpg_path, format="JPEG")
    valid, fmt, w, h = verify_image_file(jpg_path)
    assert valid is True
    assert fmt == "jpeg"
    assert w == 100 and h == 100

    # Fake JPEG (text file with .jpg extension)
    fake_jpg = tmp_path / "fake.jpg"
    with open(fake_jpg, "w") as f:
        f.write("echo 'I am not an image';")
    valid, fmt, _, _ = verify_image_file(fake_jpg)
    assert valid is False

    # Empty file
    empty_file = tmp_path / "empty.jpg"
    empty_file.touch()
    valid, fmt, _, _ = verify_image_file(empty_file)
    assert valid is False


def test_calculate_sha256(sample_image_path):
    h1 = calculate_sha256(sample_image_path)
    assert isinstance(h1, str)
    assert len(h1) == 64
