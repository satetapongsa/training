import xml.etree.ElementTree as ET
from typing import Dict, List, Any


class VOCExporter:
    """Exports annotations for a single image to Pascal VOC XML format."""

    @staticmethod
    def export_xml(
        image_info: Dict[str, Any], annotations: List[Dict[str, Any]]
    ) -> str:
        annotation = ET.Element("annotation")

        folder = ET.SubElement(annotation, "folder")
        folder.text = "images"

        filename = ET.SubElement(annotation, "filename")
        filename.text = image_info["filename"]

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_info["width"])
        height = ET.SubElement(size, "height")
        height.text = str(image_info["height"])
        depth = ET.SubElement(size, "depth")
        depth.text = "3"

        img_w = image_info["width"]
        img_h = image_info["height"]

        for ann in annotations:
            obj = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj, "name")
            name.text = ann["class_name"]

            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"

            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            w_px = ann["bbox_w"] * img_w
            h_px = ann["bbox_h"] * img_h
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(max(0, (ann["bbox_x"] * img_w) - (w_px / 2.0))))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(max(0, (ann["bbox_y"] * img_h) - (h_px / 2.0))))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(min(img_w, (ann["bbox_x"] * img_w) + (w_px / 2.0))))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(min(img_h, (ann["bbox_y"] * img_h) + (h_px / 2.0))))

        return ET.tostring(annotation, encoding="unicode")
