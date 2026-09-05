from typing import List, Tuple, Dict, Any


class YOLOAnnotationHelper:
    """Utilities for parsing, normalizing, and serializing YOLO bounding boxes."""

    @staticmethod
    def parse_line(line: str) -> Tuple[int, float, float, float, float]:
        """Parses a single YOLO annotation line: `class_id x_center y_center width height`."""
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO annotation format: '{line}'")
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        return class_id, x, y, w, h

    @staticmethod
    def format_line(class_id: int, x: float, y: float, w: float, h: float) -> str:
        """Formats bounding box into standard YOLO annotation string."""
        return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def yolo_to_xyxy(
        x: float, y: float, w: float, h: float, img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        """Converts normalized YOLO (center_x, center_y, width, height) to pixel (x1, y1, x2, y2)."""
        x1 = int((x - w / 2) * img_width)
        y1 = int((y - h / 2) * img_height)
        x2 = int((x + w / 2) * img_width)
        y2 = int((y + h / 2) * img_height)
        # Clamp to image boundaries
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        return x1, y1, x2, y2

    @staticmethod
    def xyxy_to_yolo(
        x1: float, y1: float, x2: float, y2: float, img_width: int, img_height: int
    ) -> Tuple[float, float, float, float]:
        """Converts pixel (x1, y1, x2, y2) to normalized YOLO (center_x, center_y, width, height)."""
        w_px = max(0.0, x2 - x1)
        h_px = max(0.0, y2 - y1)
        x_center = x1 + w_px / 2.0
        y_center = y1 + h_px / 2.0
        return (
            min(1.0, max(0.0, x_center / img_width)),
            min(1.0, max(0.0, y_center / img_height)),
            min(1.0, max(0.0, w_px / img_width)),
            min(1.0, max(0.0, h_px / img_height)),
        )
