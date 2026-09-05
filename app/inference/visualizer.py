import io
from pathlib import Path
from typing import List, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont


class DetectionVisualizer:
    """Renders high-contrast bounding boxes, class tags, and confidence chips."""

    # Curated palette for clear differentiation
    PALETTE = [
        "#6366f1", "#10b981", "#f59e0b", "#ef4444",
        "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6",
        "#f97316", "#84cc16", "#3b82f6", "#a855f7"
    ]

    @staticmethod
    def draw_detections(
        image_input: Union[Path, str, Image.Image],
        detections: List[Dict[str, Any]],
        box_thickness: int = 3,
    ) -> Image.Image:
        """
        Draws bounding boxes and labels on the image.
        Detections contain: x1, y1, x2, y2 (normalized 0..1), class_id, class_name, confidence.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.copy().convert("RGB")

        draw = ImageDraw.Draw(img)
        w, h = img.size

        for det in detections:
            cid = det.get("class_id", 0)
            cname = det.get("class_name", "object")
            conf = det.get("confidence", 1.0)
            color = DetectionVisualizer.PALETTE[cid % len(DetectionVisualizer.PALETTE)]

            # Calculate pixel coords
            x1 = int(det["x1"] * w)
            y1 = int(det["y1"] * h)
            x2 = int(det["x2"] * w)
            y2 = int(det["y2"] * h)

            # Clamp
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Draw outer rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)

            # Text label
            label_text = f"{cname} {conf:.1%}"
            text_w = len(label_text) * 8 + 12
            text_h = 20

            # Draw label banner
            banner_y1 = max(0, y1 - text_h)
            banner_y2 = y1
            draw.rectangle([x1, banner_y1, x1 + text_w, banner_y2], fill=color)
            draw.text((x1 + 6, banner_y1 + 3), label_text, fill="#ffffff")

        return img

    @staticmethod
    def to_bytes(img: Image.Image, format: str = "JPEG", quality: int = 90) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format=format, quality=quality)
        return buf.getvalue()
