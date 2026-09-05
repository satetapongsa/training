import os
import shutil
from pathlib import Path
from typing import BinaryIO, List
from app.core.config import settings
from app.core.security import is_safe_path, sanitize_filename
from app.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Production Local Filesystem storage with strict path traversal checks."""

    def __init__(self, base_dir: Path = settings.BASE_STORAGE_DIR):
        self.base_dir = base_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, subpath: str) -> Path:
        """Resolves subpath and verifies that it does not escape base_dir."""
        # Sanitize parts to prevent ../ manipulation
        parts = [sanitize_filename(p) for p in Path(subpath).parts if p not in (".", "..")]
        target = self.base_dir.joinpath(*parts)
        if not is_safe_path(self.base_dir, target):
            raise PermissionError(f"Path traversal detected: {subpath}")
        return target

    def save_file(self, destination_subpath: str, data: BinaryIO) -> Path:
        target_path = self._resolve_safe_path(destination_subpath)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "wb") as f:
            shutil.copyfileobj(data, f)

        return target_path

    def get_file_path(self, subpath: str) -> Path:
        return self._resolve_safe_path(subpath)

    def delete_file(self, subpath: str) -> bool:
        target_path = self._resolve_safe_path(subpath)
        if target_path.exists() and target_path.is_file():
            target_path.unlink()
            return True
        return False

    def exists(self, subpath: str) -> bool:
        target_path = self._resolve_safe_path(subpath)
        return target_path.exists()

    def list_files(self, subpath_prefix: str) -> List[Path]:
        target_dir = self._resolve_safe_path(subpath_prefix)
        if not target_dir.exists() or not target_dir.is_dir():
            return []
        return [p for p in target_dir.rglob("*") if p.is_file()]


# Global storage instance
storage = LocalStorage()
