from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.db.models import Model
from app.core.config import settings
from app.models_registry.exporter import YOLONNExporter, ModelMetadataGenerator


class ModelRegistryManager:
    """Manages model registry entries, packaging, and version comparisons."""

    @staticmethod
    async def export_model_format(
        db: AsyncSession,
        model: Model,
        target_format: str = "onnx",
        image_size: int = 640,
    ) -> Path:
        weight_path = Path(model.weights_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weight_path}")

        export_dir = settings.EXPORT_DIR / f"{model.name}_{model.version}"
        export_dir.mkdir(parents=True, exist_ok=True)
        dest_file = export_dir / f"model.{target_format}"

        exporter = YOLONNExporter()
        exported = exporter.export(
            weight_path=weight_path,
            output_path=dest_file,
            config={"format": target_format, "image_size": image_size},
        )

        if target_format == "onnx":
            model.onnx_path = str(exported)
        elif target_format == "torchscript":
            model.torchscript_path = str(exported)

        # Write metadata.json alongside
        meta = ModelMetadataGenerator.generate(
            model_name=model.name,
            version=model.version,
            architecture=model.architecture,
            dataset_name=model.metadata_info.get("dataset", "unknown"),
            classes=model.classes,
            metrics=model.metrics,
            hyperparams=model.metadata_info,
        )
        import json
        with open(export_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        await db.commit()
        await db.refresh(model)
        return exported
