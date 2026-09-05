import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import torch
from ultralytics import YOLO

from app.core.logging import logger
from app.inference.visualizer import DetectionVisualizer


class PredictionResult:
    """Encapsulates inference output and offers convenient visualization & export."""

    def __init__(
        self,
        original_image: Image.Image,
        detections: List[Dict[str, Any]],
        inference_time_ms: float,
        model_name: str,
    ):
        self.original_image = original_image
        self.detections = detections
        self.inference_time_ms = inference_time_ms
        self.model_name = model_name
        self.width, self.height = original_image.size

    def render(self, box_thickness: int = 3) -> Image.Image:
        """Renders bounding boxes and returns PIL Image."""
        return DetectionVisualizer.draw_detections(
            self.original_image, self.detections, box_thickness=box_thickness
        )

    def save(self, output_path: Union[str, Path]) -> Path:
        """Draws bounding boxes and saves output image to disk."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = self.render()
        rendered.save(target)
        return target

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "image_width": self.width,
            "image_height": self.height,
            "total_detections": len(self.detections),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "detections": self.detections,
        }


class Predictor:
    """
    Unified High-Level Computer Vision Inference Engine.
    
    Usage:
        predictor = Predictor("models/best.pt")
        result = predictor.predict("image.jpg", conf_threshold=0.25)
        result.save("result.jpg")
    """

    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.device = "cuda" if (device != "cpu" and torch.cuda.is_available()) else "cpu"
        self.model_name = self.model_path.stem
        self.suffix = self.model_path.suffix.lower()
        self.is_kdel = False
        self.kdel_classes = []
        self.variant = "standard"

        logger.info(f"Loading inference model '{self.model_path.name}' on {self.device}...")

        # 1. Attempt loading as proprietary KDel 4.0 model first
        if self.suffix == ".pt":
            try:
                ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or ckpt.get("architecture", "").startswith("kdel")):
                    from app.models.kdel import KDel4Model
                    self.kdel_classes = ckpt.get("classes", ["object"])
                    num_classes = max(1, len(self.kdel_classes))
                    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                    initial_variant = str(ckpt.get("variant", ckpt.get("architecture", "standard"))).lower()

                    # Try loading with specified variant or auto-discover
                    loaded_model = None
                    for cand_v in [initial_variant, "nano", "standard", "pro"]:
                        try:
                            m = KDel4Model(num_classes=num_classes, variant=cand_v)
                            m.load_state_dict(state_dict)
                            loaded_model = m
                            self.variant = cand_v
                            break
                        except Exception:
                            continue

                    if loaded_model is not None:
                        self.model = loaded_model
                        self.model.to(self.device)
                        self.model.eval()
                        self.is_kdel = True
                        logger.info(f"Loaded KDel 4.0 ({self.variant}) Deep Learning Model with {num_classes} classes: {self.kdel_classes}")
                        return
            except Exception as e:
                logger.debug(f"Not a KDel state_dict ({e}), falling back to YOLO format...")

        # 2. Fallback to Ultralytics YOLO loader if not KDel
        if self.suffix in [".pt", ".onnx", ".engine"]:
            self.model = YOLO(str(self.model_path))
        else:
            raise ValueError(f"Unsupported model format: {self.suffix}")

    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> PredictionResult:
        """
        Executes model inference on a single image.
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = image_input.copy().convert("RGB")

        w, h = pil_img.size
        t0 = time.perf_counter()

        detections: List[Dict[str, Any]] = []

        # --- Native KDel 4.0 Inference ---
        if self.is_kdel:
            from app.models.kdel import kdel_nms
            from torchvision import transforms

            preprocess = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_t = preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                decoded = self.model(img_t)
                kdel_dets = kdel_nms(decoded[0], conf_threshold=conf_threshold, iou_threshold=iou_threshold)

            latency_ms = (time.perf_counter() - t0) * 1000.0

            for d in kdel_dets:
                cid = d["class_id"]
                cname = self.kdel_classes[cid] if cid < len(self.kdel_classes) else f"class_{cid}"
                x1 = d["box"]["x1"]
                y1 = d["box"]["y1"]
                x2 = d["box"]["x2"]
                y2 = d["box"]["y2"]

                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": round(d["confidence"], 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box": d["box"],
                    "box_pixels": [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)],
                })

            return PredictionResult(
                original_image=pil_img,
                detections=detections,
                inference_time_ms=latency_ms,
                model_name=f"KDel 4.0 ({self.model_name})",
            )

        # --- Fallback YOLO Inference ---
        results = self.model(
            pil_img,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                names = res.names

                for box, conf, cls_id in zip(xyxy, confs, classes):
                    x1_px, y1_px, x2_px, y2_px = box
                    norm_x1 = max(0.0, min(1.0, float(x1_px / w)))
                    norm_y1 = max(0.0, min(1.0, float(y1_px / h)))
                    norm_x2 = max(0.0, min(1.0, float(x2_px / w)))
                    norm_y2 = max(0.0, min(1.0, float(y2_px / h)))

                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": names.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(float(conf), 4),
                        "x1": norm_x1,
                        "y1": norm_y1,
                        "x2": norm_x2,
                        "y2": norm_y2,
                        "box": {"x1": norm_x1, "y1": norm_y1, "x2": norm_x2, "y2": norm_y2},
                        "box_pixels": [int(x1_px), int(y1_px), int(x2_px), int(y2_px)],
                    })

        return PredictionResult(
            original_image=pil_img,
            detections=detections,
            inference_time_ms=latency_ms,
            model_name=self.model_name,
        )
