from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Callable, Optional


class TrainerBase(ABC):
    """Abstract Base Class for all Computer Vision Trainers."""

    def __init__(
        self,
        job_id: int,
        config: Dict[str, Any],
        run_dir: Path,
        on_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.job_id = job_id
        self.config = config
        self.run_dir = run_dir
        self.on_progress_callback = on_progress_callback
        self.is_stopped = False
        self.is_paused = False

    @abstractmethod
    def setup(self, dataset_manifest_path: Path) -> None:
        """Initializes model architecture, datasets, optimizer, and loss function."""
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Runs the complete training loop. Returns final evaluation metrics."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Gracefully halts training and saves latest checkpoint."""
        self.is_stopped = True

    @abstractmethod
    def pause(self) -> None:
        """Pauses training loop."""
        self.is_paused = True

    @abstractmethod
    def resume(self, checkpoint_path: Path) -> None:
        """Resumes training from a saved checkpoint."""
        pass

    @abstractmethod
    def export(self, export_format: str, output_path: Path) -> Path:
        """Exports trained weights to specified format (e.g. onnx, torchscript, pt)."""
        pass
