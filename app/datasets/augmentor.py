import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from app.core.logging import logger


class DatasetAugmentor:
    """Configurable image & bounding-box augmentation pipeline using Albumentations."""

    @staticmethod
    def build_pipeline(config: Dict[str, Any]) -> A.Compose:
        """Constructs an Albumentations Compose pipeline for detection."""
        transforms = []

        if config.get("horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=float(config["horizontal_flip"])))

        if config.get("vertical_flip", 0) > 0:
            transforms.append(A.VerticalFlip(p=float(config["vertical_flip"])))

        if config.get("rotation", 0) > 0:
            deg = int(config["rotation"])
            transforms.append(A.Rotate(limit=(-deg, deg), p=0.7, border_mode=cv2.BORDER_REFLECT))

        if config.get("brightness_contrast", 0) > 0:
            val = float(config["brightness_contrast"])
            transforms.append(A.RandomBrightnessContrast(brightness_limit=val, contrast_limit=val, p=0.7))

        if config.get("blur", 0) > 0:
            transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=float(config["blur"])))

        if config.get("color_jitter", 0) > 0:
            val = float(config["color_jitter"])
            transforms.append(A.ColorJitter(brightness=val, contrast=val, saturation=val, hue=0.1, p=0.6))

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
            ),
        )

    @staticmethod
    def generate_preview(
        image_path: Path,
        annotations: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> bytes:
        """
        Applies augmentation and renders bounding boxes on the resulting image.
        Returns JPEG encoded bytes.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Extract YOLO bboxes [x_center, y_center, width, height]
        bboxes = []
        labels = []
        for a in annotations:
            bboxes.append([a["bbox_x"], a["bbox_y"], a["bbox_w"], a["bbox_h"]])
            labels.append(a.get("class_name", str(a["class_id"])))

        pipeline = DatasetAugmentor.build_pipeline(config)
        try:
            transformed = pipeline(image=image, bboxes=bboxes, class_labels=labels)
            aug_img = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]
        except Exception as e:
            logger.warning(f"Augmentation failed, falling back to original: {e}")
            aug_img = image
            aug_boxes = bboxes
            aug_labels = labels

        # Render bounding boxes with PIL
        pil_img = Image.fromarray(aug_img)
        draw = ImageDraw.Draw(pil_img)
        img_w, img_h = pil_img.size

        colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]

        for idx, (box, label) in enumerate(zip(aug_boxes, aug_labels)):
            bx, by, bw, bh = box
            x1 = int((bx - bw / 2) * img_w)
            y1 = int((by - bh / 2) * img_h)
            x2 = int((bx + bw / 2) * img_w)
            y2 = int((by + bh / 2) * img_h)
            color = colors[idx % len(colors)]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Label background banner
            draw.rectangle([x1, max(0, y1 - 18), x1 + len(label) * 8 + 10, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - 16)), label, fill="#ffffff")

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
