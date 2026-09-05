import os
import re
import hashlib
from pathlib import Path
from typing import Tuple
from PIL import Image
from app.core.config import settings

# Allowed Magic Byte Signatures for Image Formats
IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",  # WebP has RIFF...WEBP
    b"BM": "bmp",
    b"II*\x00": "tiff",  # Little-endian TIFF
    b"MM\x00*": "tiff",  # Big-endian TIFF
}


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename against path traversal and dangerous characters."""
    # Remove directory separators and null bytes
    cleaned = os.path.basename(filename).replace("\x00", "_")
    # Remove leading dots or invalid characters
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned)
    # Remove consecutive dots
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    if not cleaned or cleaned.startswith("."):
        cleaned = f"file_{cleaned.lstrip('.')}"
    return cleaned


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Validates that target_path is strictly inside base_dir to prevent path traversal."""
    try:
        resolved_base = base_dir.resolve()
        resolved_target = target_path.resolve()
        return resolved_target == resolved_base or resolved_base in resolved_target.parents
    except Exception:
        return False


def calculate_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file efficiently using chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_image_file(file_path: Path) -> Tuple[bool, str, int, int]:
    """
    Verifies image validity using file magic bytes and PIL decoding.
    Returns: (is_valid, mime_or_format, width, height)
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False, "empty_file", 0, 0

    # 1. Check Magic Bytes
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
            valid_signature = False
            for sig, fmt in IMAGE_SIGNATURES.items():
                if header.startswith(sig):
                    valid_signature = True
                    break
                # Check for RIFF WebP specifically
                if header.startswith(b"RIFF") and b"WEBP" in header[:12]:
                    valid_signature = True
                    break
            if not valid_signature:
                return False, "invalid_magic_signature", 0, 0
    except Exception as e:
        return False, f"read_error: {str(e)}", 0, 0

    # 2. Verify with PIL Image integrity verification
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Re-open to read width and height since verify() can close the file descriptor
        with Image.open(file_path) as img:
            w, h = img.size
            fmt = img.format.lower() if img.format else "unknown"
            if w <= 0 or h <= 0:
                return False, "invalid_dimensions", 0, 0
            return True, fmt, w, h
    except Exception as e:
        return False, f"corrupted_image: {str(e)}", 0, 0
