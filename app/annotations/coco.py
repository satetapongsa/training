import json
from typing import List, Dict, Any


class COCOExporter:
    """Exports dataset annotations to standard COCO JSON format."""

    @staticmethod
    def export(
        images: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        categories: List[str],
    ) -> str:
        """
        Builds COCO format dictionary and returns JSON string.
        """
        coco_categories = [
            {"id": idx, "name": name, "supercategory": "none"}
            for idx, name in enumerate(categories)
        ]

        coco_images = []
        for img in images:
            coco_images.append({
                "id": img["id"],
                "file_name": img["filename"],
                "width": img["width"],
                "height": img["height"],
            })

        coco_annotations = []
        annot_id = 1
        for ann in annotations:
            img_w = ann["image_width"]
            img_h = ann["image_height"]
            # Convert YOLO to COCO bbox: [top-left-x, top-left-y, width, height] in pixels
            w_px = ann["bbox_w"] * img_w
            h_px = ann["bbox_h"] * img_h
            x_px = (ann["bbox_x"] * img_w) - (w_px / 2.0)
            y_px = (ann["bbox_y"] * img_h) - (h_px / 2.0)

            coco_annotations.append({
                "id": annot_id,
                "image_id": ann["image_id"],
                "category_id": ann["class_id"],
                "bbox": [round(x_px, 2), round(y_px, 2), round(w_px, 2), round(h_px, 2)],
                "area": round(w_px * h_px, 2),
                "iscrowd": 0,
            })
            annot_id += 1

        dataset_dict = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories,
        }
        return json.dumps(dataset_dict, indent=2)
