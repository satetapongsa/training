from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, List, Optional


class StorageBackend(ABC):
    """Abstract interface for storage backends (Local, S3, MinIO)."""

    @abstractmethod
    def save_file(self, destination_subpath: str, data: BinaryIO) -> Path:
        """Saves a binary file to the storage backend."""
        pass

    @abstractmethod
    def get_file_path(self, subpath: str) -> Path:
        """Returns the local or cached Path for direct processing."""
        pass

    @abstractmethod
    def delete_file(self, subpath: str) -> bool:
        """Deletes a file from storage."""
        pass

    @abstractmethod
    def exists(self, subpath: str) -> bool:
        """Checks if a file exists."""
        pass

    @abstractmethod
    def list_files(self, subpath_prefix: str) -> List[Path]:
        """Lists files matching prefix."""
        pass
