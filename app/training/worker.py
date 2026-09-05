import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from sqlalchemy import select, update
from app.core.config import settings
from app.core.logging import logger, setup_logger
from app.core.telemetry import telemetry
from app.db.session import SyncSessionLocal
from app.db.models import TrainingJob, TrainingMetric, Model, Dataset, Project
from app.training.registry import TrainerRegistry
from app.training.events import event_bus
from app.training.base import TrainerBase


class TrainingWorker:
    """Manages background execution, telemetry streaming, and lifecycle of training jobs."""

    def __init__(self):
        self._active_trainers: Dict[int, TrainerBase] = {}
        self._active_threads: Dict[int, threading.Thread] = {}
        self._telemetry_running: Dict[int, bool] = {}

    def is_job_running(self, job_id: int) -> bool:
        return job_id in self._active_threads and self._active_threads[job_id].is_alive()

    def has_active_jobs(self) -> bool:
        """Returns True if any training thread is currently alive and active."""
        return any(t.is_alive() for t in self._active_threads.values())

    def get_active_job_id(self) -> Optional[int]:
        """Returns the ID of the currently active training job if any."""
        for j_id, thread in self._active_threads.items():
            if thread.is_alive():
                return j_id
        return None

    def start_job(self, job_id: int, dataset_manifest_path: Path) -> None:
        """Launches training job in a dedicated background worker thread."""
        thread = threading.Thread(
            target=self._run_job_sync,
            args=(job_id, dataset_manifest_path),
            daemon=True,
            name=f"TrainingWorker-{job_id}",
        )
        self._active_threads[job_id] = thread
        thread.start()

    def stop_job(self, job_id: int) -> bool:
        """Requests graceful cancellation of a running training job."""
        trainer = self._active_trainers.get(job_id)
        if trainer:
            trainer.stop()
            self._telemetry_running[job_id] = False
            return True
        return False

    def _run_telemetry_loop(self, job_id: int):
        """Streams hardware telemetry while job is running."""
        while self._telemetry_running.get(job_id, False):
            try:
                metrics = telemetry.get_realtime_metrics()
                # Run event emission safely from thread
                loop = asyncio.new_event_loop()
                loop.run_until_complete(event_bus.emit("gpu_update", {"job_id": job_id, **metrics}))
                loop.close()
            except Exception:
                pass
            time.sleep(1.0)

    def _run_job_sync(self, job_id: int, dataset_manifest_path: Path):
        """Synchronous execution function running within the worker thread."""
        session = SyncSessionLocal()
        job: Optional[TrainingJob] = None
        try:
            job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                logger.error(f"[Worker] Job {job_id} not found in database.")
                return

            job.status = "running"
            job.pid = os.getpid()
            session.commit()

            run_dir = Path(job.run_dir)
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            job_logger = setup_logger(f"job_{job_id}", log_file=logs_dir / "training.log")

            job_logger.info(f"=== Starting Training Job {job_id}: {job.model_name} ({job.architecture}) ===")

            # Setup Telemetry thread
            self._telemetry_running[job_id] = True
            telemetry_thread = threading.Thread(
                target=self._run_telemetry_loop, args=(job_id,), daemon=True
            )
            telemetry_thread.start()

            # Determine task type
            dataset = session.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            task_type = dataset.task_type if dataset else "detection"

            # Progress callback for real-time DB recording and WS emission
            def on_progress(event_data: Dict[str, Any]):
                try:
                    epoch = int(event_data.get("epoch", job.current_epoch or 1))
                    total_epochs = int(event_data.get("total_epochs", job.total_epochs or 10))
                    step = int(event_data.get("step", job.current_step or 0))
                    total_steps = int(event_data.get("total_steps", job.total_steps or 0))
                    loss = float(event_data.get("train_loss", event_data.get("loss", 0.0)))
                    val_loss = float(event_data.get("val_loss", 0.0))
                    lr = float(event_data.get("learning_rate", event_data.get("lr", 0.001)))
                    map50 = float(event_data.get("map50", 0.0))
                    val_metrics = event_data.get("metrics", {})
                    if map50 and "mAP50" not in val_metrics:
                        val_metrics["mAP50"] = map50

                    # Record metric to DB
                    metric_record = TrainingMetric(
                        job_id=job_id,
                        epoch=epoch,
                        step=step,
                        loss=loss,
                        metrics=val_metrics,
                        lr=lr,
                    )
                    session.add(metric_record)

                    # Update job progress in DB
                    job.current_epoch = epoch
                    job.total_epochs = total_epochs
                    job.current_step = step
                    job.total_steps = total_steps
                    if map50 > (job.best_metric_val or 0.0):
                        job.best_metric_val = map50
                        job.best_metric_name = "mAP50"
                    elif "accuracy" in val_metrics and val_metrics["accuracy"] > (job.best_metric_val or 0.0):
                        job.best_metric_val = val_metrics["accuracy"]
                        job.best_metric_name = "accuracy"

                    session.commit()

                    # Emit via event bus for WebSocket clients
                    progress_payload = {
                        "job_id": job_id,
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "step": step,
                        "total_steps": total_steps,
                        "train_loss": loss,
                        "val_loss": val_loss,
                        "map50": map50,
                        "map50_95": event_data.get("map50_95", map50 * 0.75),
                        "learning_rate": lr,
                        "log": event_data.get("log", f"Epoch {epoch}/{total_epochs} Step {step}/{total_steps} Loss: {loss:.4f}"),
                    }
                    event_bus.emit_threadsafe("training_progress", progress_payload)
                    event_bus.emit_threadsafe("epoch_update", progress_payload)
                except Exception as cb_err:
                    logger.warning(f"[Worker] on_progress warning: {cb_err}")

            # Instantiate Trainer
            trainer_cls = TrainerRegistry.get(job.architecture, task_type=task_type)
            trainer: TrainerBase = trainer_cls(
                job_id=job_id,
                config=job.config,
                run_dir=run_dir,
                on_progress_callback=on_progress,
            )
            self._active_trainers[job_id] = trainer

            # Run Setup & Training
            trainer.setup(dataset_manifest_path)
            final_metrics = trainer.train()

            # Stop telemetry
            self._telemetry_running[job_id] = False

            # Check if cancelled or completed
            if trainer.is_stopped:
                job.status = "cancelled"
                job_logger.info(f"Training Job {job_id} was cancelled by user.")
            else:
                job.status = "completed"
                checkpoints_dir = run_dir / "checkpoints"
                best_pt = checkpoints_dir / "best.pt"
                last_pt = checkpoints_dir / "last.pt"
                job.checkpoint_path = str(best_pt if best_pt.exists() else last_pt)

                # Register Model in Database
                weight_file = best_pt if best_pt.exists() else last_pt
                weight_size = weight_file.stat().st_size if weight_file.exists() else 0

                registered_model = Model(
                    project_id=job.project_id,
                    job_id=job.id,
                    name=f"{job.model_name}_final",
                    version="v1.0.0",
                    architecture=job.architecture,
                    task_type=task_type,
                    classes=dataset.classes if dataset else [],
                    weights_path=str(weight_file),
                    metrics=final_metrics,
                    metadata_info={
                        "epochs": job.total_epochs,
                        "architecture": job.architecture,
                        "dataset": dataset.name if dataset else "unknown",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    },
                    size_bytes=weight_size,
                    is_deployed=True,
                )
                session.add(registered_model)
                session.commit()

                job_logger.info(f"Training completed successfully! Model registered: {registered_model.name}")

                # Emit completion event
                complete_payload = {
                    "job_id": job_id,
                    "model_id": registered_model.id,
                    "metrics": final_metrics,
                    "weights_path": str(weight_file),
                }
                event_bus.emit_threadsafe("training_completed", complete_payload)
                event_bus.emit_threadsafe("training_complete", complete_payload)

            session.commit()

        except Exception as e:
            logger.exception(f"[Worker] Error in job {job_id}: {e}")
            self._telemetry_running[job_id] = False
            if job:
                job.status = "failed"
                job.error_message = str(e)
                session.commit()

            fail_payload = {"job_id": job_id, "error": str(e)}
            event_bus.emit_threadsafe("training_failed", fail_payload)
            event_bus.emit_threadsafe("training_error", fail_payload)

        finally:
            self._active_trainers.pop(job_id, None)
            self._active_threads.pop(job_id, None)
            session.close()


training_worker = TrainingWorker()
