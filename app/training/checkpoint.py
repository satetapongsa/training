import os
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from app.core.logging import logger


class CheckpointManager:
    """Manages model checkpoints, saving best.pt, last.pt, and resuming."""

    def __init__(self, checkpoints_dir: Path, metric_name: str = "mAP50", higher_is_better: bool = True):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.best_metric_val: Optional[float] = None

    def is_better(self, current_val: float) -> bool:
        if self.best_metric_val is None:
            return True
        if self.higher_is_better:
            return current_val > self.best_metric_val
        return current_val < self.best_metric_val

    def save(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        metric_val: float,
        is_best_override: Optional[bool] = None,
    ) -> Dict[str, Path]:
        """
        Saves latest checkpoint as last.pt and conditionally copies to best.pt.
        """
        last_path = self.checkpoints_dir / "last.pt"
        best_path = self.checkpoints_dir / "best.pt"

        # Save last.pt
        torch.save(state_dict, last_path)

        # Check if best
        is_best = is_best_override if is_best_override is not None else self.is_better(metric_val)
        if is_best:
            self.best_metric_val = metric_val
            shutil.copy2(last_path, best_path)
            logger.info(f"New best model saved at epoch {epoch} with {self.metric_name}={metric_val:.4f}")

        return {"last": last_path, "best": best_path, "is_best": is_best}

    def load(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Loads checkpoint state dictionary from disk."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
