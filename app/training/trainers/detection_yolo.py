import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import torch
from ultralytics import YOLO

from app.core.logging import logger
from app.training.base import TrainerBase
from app.training.checkpoint import CheckpointManager


class YOLODetectionTrainer(TrainerBase):
    """Production Object Detection Trainer powered by Ultralytics YOLO engine."""

    def __init__(
        self,
        job_id: int,
        config: Dict[str, Any],
        run_dir: Path,
        on_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__(job_id, config, run_dir, on_progress_callback)
        self.model: Optional[YOLO] = None
        self.dataset_manifest_path: Optional[Path] = None
        self.checkpoint_manager = CheckpointManager(
            checkpoints_dir=self.run_dir / "checkpoints",
            metric_name="mAP50",
            higher_is_better=True,
        )

    def setup(self, dataset_manifest_path: Path) -> None:
        self.dataset_manifest_path = dataset_manifest_path
        arch = self.config.get("architecture", "yolo11n")
        # If architecture does not end with .pt, append it
        model_weight = f"{arch}.pt" if not arch.endswith(".pt") else arch
        
        logger.info(f"[Job {self.job_id}] Initializing YOLO model: {model_weight}")
        self.model = YOLO(model_weight)

        # Attach custom event callbacks to YOLO model
        def on_train_epoch_end(trainer):
            if self.is_stopped:
                trainer.stop = True
                return

            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            # Retrieve training loss
            loss = float(trainer.loss.item()) if hasattr(trainer, "loss") and trainer.loss is not None else 0.0
            lr = float(trainer.optimizer.param_groups[0]["lr"]) if hasattr(trainer, "optimizer") else 0.001

            # Retrieve validation metrics if available
            metrics_dict = {}
            if hasattr(trainer, "metrics") and trainer.metrics:
                m = trainer.metrics
                metrics_dict = {
                    "precision": round(float(m.get("metrics/precision(B)", 0.0)), 4),
                    "recall": round(float(m.get("metrics/recall(B)", 0.0)), 4),
                    "mAP50": round(float(m.get("metrics/mAP50(B)", 0.0)), 4),
                    "mAP50_95": round(float(m.get("metrics/mAP50-95(B)", 0.0)), 4),
                }

            if self.on_progress_callback:
                self.on_progress_callback({
                    "type": "epoch_update",
                    "job_id": self.job_id,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "loss": round(loss, 5),
                    "lr": lr,
                    "metrics": metrics_dict,
                })

        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

    def train(self) -> Dict[str, Any]:
        if self.model is None or self.dataset_manifest_path is None:
            raise RuntimeError("Trainer setup() must be called before train().")

        epochs = int(self.config.get("epochs", 10))
        batch_size = int(self.config.get("batch_size", 8))
        imgsz = int(self.config.get("image_size", 640))
        lr0 = float(self.config.get("learning_rate", 0.001))
        seed = int(self.config.get("seed", 42))
        device_cfg = self.config.get("device", "auto")

        device = "0" if (device_cfg in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        logger.info(f"[Job {self.job_id}] Starting YOLO training on device: {device}")

        # Execute real training loop
        results = self.model.train(
            data=str(self.dataset_manifest_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            lr0=lr0,
            seed=seed,
            device=device,
            project=str(self.run_dir.parent),
            name=self.run_dir.name,
            exist_ok=True,
            plots=True,
            verbose=False,
        )

        # Copy generated weights into runs/{project}/{experiment}/checkpoints/
        weights_dir = self.run_dir / "weights"
        dest_checkpoints = self.run_dir / "checkpoints"
        dest_checkpoints.mkdir(parents=True, exist_ok=True)

        best_weight = weights_dir / "best.pt"
        last_weight = weights_dir / "last.pt"

        if best_weight.exists():
            shutil.copy2(best_weight, dest_checkpoints / "best.pt")
        if last_weight.exists():
            shutil.copy2(last_weight, dest_checkpoints / "last.pt")

        # Compile final metrics report
        final_metrics = {}
        if hasattr(results, "results_dict"):
            final_metrics = {
                "precision": round(float(results.results_dict.get("metrics/precision(B)", 0.0)), 4),
                "recall": round(float(results.results_dict.get("metrics/recall(B)", 0.0)), 4),
                "mAP50": round(float(results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
                "mAP50_95": round(float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
            }

        # Save metrics.json
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)

        return final_metrics

    def stop(self) -> None:
        self.is_stopped = True

    def pause(self) -> None:
        self.is_paused = True

    def resume(self, checkpoint_path: Path) -> None:
        logger.info(f"[Job {self.job_id}] Resuming from checkpoint: {checkpoint_path}")
        self.model = YOLO(str(checkpoint_path))

    def export(self, export_format: str, output_path: Path) -> Path:
        """Exports the trained YOLO model into ONNX, TorchScript, or PyTorch."""
        best_pt = self.run_dir / "checkpoints" / "best.pt"
        if not best_pt.exists():
            best_pt = self.run_dir / "weights" / "best.pt"

        model = YOLO(str(best_pt))
        exported_file = model.export(format=export_format, dynamic=True)
        exported_path = Path(exported_file)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(exported_path, output_path)
        return output_path
