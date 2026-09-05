import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Union
from app.core.config import settings
from app.core.logging import logger
from app.inference.predictor import Predictor, PredictionResult


class BatchInferenceProcessor:
    """Processes directories or lists of images with a trained model."""

    @staticmethod
    def process_batch(
        predictor: Predictor,
        image_sources: List[Path],
        output_dir: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_images: bool = True,
        export_formats: List[str] = ["json", "csv"],
    ) -> Dict[str, Any]:
        """
        Runs inference across all images and exports results into output_dir.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        img_out_dir = output_dir / "images"
        if save_images:
            img_out_dir.mkdir(parents=True, exist_ok=True)

        all_results: List[Dict[str, Any]] = []
        flat_rows: List[Dict[str, Any]] = []

        for img_path in image_sources:
            try:
                res: PredictionResult = predictor.predict(
                    img_path, conf_threshold=conf_threshold, iou_threshold=iou_threshold
                )
                res_dict = res.to_dict()
                res_dict["filename"] = img_path.name

                if save_images:
                    saved_path = res.save(img_out_dir / f"pred_{img_path.name}")
                    res_dict["annotated_image"] = str(saved_path)

                all_results.append(res_dict)

                # Flatten rows for CSV
                for det in res.detections:
                    flat_rows.append({
                        "filename": img_path.name,
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "x1": det["x1"],
                        "y1": det["y1"],
                        "x2": det["x2"],
                        "y2": det["y2"],
                    })

            except Exception as e:
                logger.error(f"Error inferencing {img_path.name}: {e}")

        # Export JSON
        if "json" in export_formats:
            with open(output_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

        # Export CSV
        if "csv" in export_formats and flat_rows:
            with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
                writer.writeheader()
                writer.writerows(flat_rows)

        # Export TXT (YOLO format predictions)
        if "txt" in export_formats:
            labels_dir = output_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            for res_item in all_results:
                stem = Path(res_item["filename"]).stem
                with open(labels_dir / f"{stem}.txt", "w", encoding="utf-8") as lf:
                    for d in res_item.get("detections", []):
                        # Convert xyxy to yolo center w h
                        w = d["x2"] - d["x1"]
                        h = d["y2"] - d["y1"]
                        cx = d["x1"] + w / 2.0
                        cy = d["y1"] + h / 2.0
                        lf.write(f"{d['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {d['confidence']:.4f}\n")

        return {
            "total_processed": len(all_results),
            "output_directory": str(output_dir),
            "results_count": len(flat_rows),
        }
