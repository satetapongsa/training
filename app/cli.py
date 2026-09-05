import argparse
import sys
import yaml
from pathlib import Path
import uvicorn
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.datasets.validator import DatasetValidator
from app.inference.predictor import Predictor
from app.inference.batch import BatchInferenceProcessor
from app.models_registry.exporter import YOLONNExporter


def cmd_run(args):
    """Starts the FastAPI Web Server and Web Studio."""
    host = args.host or settings.HOST
    port = args.port or settings.PORT
    print(f"\n=======================================================")
    print(f"  AI Vision Training Studio")
    print(f"  Web Studio & API: http://{host}:{port}")
    print(f"  Interactive API Docs: http://{host}:{port}/docs")
    print(f"=======================================================\n")
    uvicorn.run("app.main:app", host=host, port=port, reload=args.reload)


def cmd_train(args):
    """Runs a training job directly via CLI."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from app.training.registry import TrainerRegistry
    from app.training.base import TrainerBase

    task = cfg.get("task_type", "detection")
    arch = cfg.get("architecture", "yolo11n")
    manifest_path = Path(cfg["dataset"]["path"])

    run_dir = settings.RUNS_DIR / cfg.get("project_name", "cli_proj") / cfg.get("experiment_name", "cli_run")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training: {arch} on {manifest_path}...")
    trainer_cls = TrainerRegistry.get(arch, task_type=task)
    trainer: TrainerBase = trainer_cls(
        job_id=999,
        config=cfg.get("hyperparameters", {}),
        run_dir=run_dir,
        on_progress_callback=lambda p: print(f"Progress: {p}"),
    )
    trainer.setup(manifest_path)
    final_metrics = trainer.train()
    print(f"\nTraining completed! Final Metrics:\n{final_metrics}")
    print(f"Weights saved in: {run_dir / 'checkpoints'}")


def cmd_validate_dataset(args):
    """Validates dataset images and annotations from a directory."""
    dset_path = Path(args.dataset)
    if not dset_path.exists():
        print(f"Error: Dataset directory not found: {dset_path}")
        sys.exit(1)

    # Scan images
    allowed_exts = {f".{ext}" for ext in settings.ALLOWED_IMAGE_EXTENSIONS}
    image_files = [f for f in dset_path.rglob("*") if f.is_file() and f.suffix.lower() in allowed_exts]

    images = [{"id": idx, "filename": f.name, "file_path": str(f)} for idx, f in enumerate(image_files)]
    annots_by_img = {}

    # Try to read classes.txt
    classes_file = dset_path / "classes.txt"
    classes = []
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]

    report = DatasetValidator.validate_dataset_records(
        dataset_name=dset_path.name,
        classes=classes,
        images=images,
        annotations_by_image_id=annots_by_img,
    )
    print(yaml.dump(report, sort_keys=False))


def cmd_predict(args):
    """Runs inference on an image or directory."""
    predictor = Predictor(args.model)
    source = Path(args.source)
    output_dir = Path(args.output or "./inference_output")

    if source.is_file():
        res = predictor.predict(source, conf_threshold=args.conf)
        output_file = output_dir / f"pred_{source.name}"
        res.save(output_file)
        print(f"Detected {len(res.detections)} objects in {res.inference_time_ms:.1f}ms")
        print(f"Result saved to: {output_file}")
    elif source.is_dir():
        allowed_exts = {f".{ext}" for ext in settings.ALLOWED_IMAGE_EXTENSIONS}
        imgs = [f for f in source.rglob("*") if f.is_file() and f.suffix.lower() in allowed_exts]
        report = BatchInferenceProcessor.process_batch(
            predictor=predictor,
            image_sources=imgs,
            output_dir=output_dir,
            conf_threshold=args.conf,
        )
        print(f"Batch prediction complete: {report}")


def cmd_export(args):
    """Exports model weights to ONNX or TorchScript."""
    model_path = Path(args.model)
    output_path = Path(args.output or f"{model_path.stem}.{args.format}")
    exporter = YOLONNExporter()
    exported = exporter.export(
        weight_path=model_path,
        output_path=output_path,
        config={"format": args.format, "image_size": args.imgsz},
    )
    print(f"Successfully exported {args.format} model to: {exported}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-vision-studio",
        description="AI Vision Training Studio CLI - Production Computer Vision Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # run command
    p_run = subparsers.add_parser("run", help="Start FastAPI Web Server & Studio UI")
    p_run.add_argument("--host", default=None, help="Host address to bind")
    p_run.add_argument("--port", type=int, default=None, help="Port to bind (default: 8000)")
    p_run.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")

    # train command
    p_train = subparsers.add_parser("train", help="Run model training from config")
    p_train.add_argument("--config", required=True, help="Path to YAML training config")

    # validate-dataset command
    p_val = subparsers.add_parser("validate-dataset", help="Deep validate a dataset")
    p_val.add_argument("--dataset", required=True, help="Path to dataset directory")

    # predict command
    p_pred = subparsers.add_parser("predict", help="Run inference on image or folder")
    p_pred.add_argument("--model", required=True, help="Path to model weights (.pt or .onnx)")
    p_pred.add_argument("--source", required=True, help="Path to image file or directory")
    p_pred.add_argument("--output", default=None, help="Output directory for annotated images")
    p_pred.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    # export command
    p_exp = subparsers.add_parser("export", help="Export weights to ONNX/TorchScript")
    p_exp.add_argument("--model", required=True, help="Path to model .pt")
    p_exp.add_argument("--format", default="onnx", choices=["onnx", "torchscript"], help="Export format")
    p_exp.add_argument("--output", default=None, help="Output file path")
    p_exp.add_argument("--imgsz", type=int, default=640, help="Image resolution for export")

    return parser


def cli_entrypoint():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "run": cmd_run,
        "train": cmd_train,
        "validate-dataset": cmd_validate_dataset,
        "predict": cmd_predict,
        "export": cmd_export,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    cli_entrypoint()
