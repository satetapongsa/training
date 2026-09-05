"""KDel 4.0: Proprietary Deep Convolutional & Multi-Scale Vision Neural Network.
Engineered natively in pure PyTorch with zero external framework dependencies.

Architecture Topology:
  Input (B, 3, H, W)
      │
      ▼
  KDelStem (Stride 2)
      │
      ▼
  KDelBackbone
      ├─ Stage 1 (Stride 4)  ──> P2
      ├─ Stage 2 (Stride 8)  ──> P3 (High resolution, small targets)
      ├─ Stage 3 (Stride 16) ──> P4 (Medium resolution, medium targets)
      └─ Stage 4 (Stride 32) ──> P5 (Deep receptive field, large targets)
      │
      ▼
  KDelPANet (Bi-directional Feature Pyramid Network + Spatial Attention)
      ├─ Fused P3
      ├─ Fused P4
      └─ Fused P5
      │
      ▼
  KDelDecoupledHead (Multi-Scale Anchor-Free Heads)
      ├─ Cls Logits: (B, NumClasses, H_i, W_i)
      ├─ Reg BBox:   (B, 4, H_i, W_i) [cx, cy, w, h]
      └─ Objectness: (B, 1, H_i, W_i)
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. Core Convolutions & Attention Blocks ---

class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv2d + BatchNorm2d + SiLU."""

    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: Optional[int] = None, act: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class KDelSpatialAttention(nn.Module):
    """KDel Spatial & Channel Attention Module (Squeeze-and-Excitation + Spatial Gating)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid_c = max(8, channels // reduction)
        # Channel attention branch
        self.fc1 = nn.Conv2d(channels, mid_c, 1, bias=False)
        self.fc2 = nn.Conv2d(mid_c, channels, 1, bias=False)
        # Spatial attention branch
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel gate
        gap = F.adaptive_avg_pool2d(x, (1, 1))
        ca = torch.sigmoid(self.fc2(F.silu(self.fc1(gap))))
        x_ca = x * ca

        # Spatial gate
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x_ca * sa


class KDelResidualBlock(nn.Module):
    """Residual bottleneck with attention gating."""

    def __init__(self, channels: int):
        super().__init__()
        self.c1 = ConvBlock(channels, channels, k=3, s=1)
        self.c2 = ConvBlock(channels, channels, k=3, s=1, act=False)
        self.attn = KDelSpatialAttention(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.c2(self.c1(x))
        out = self.attn(out)
        return self.act(out + residual)


# --- 2. Backbone Feature Extractor ---

class KDelBackbone(nn.Module):
    """Multi-Scale Deep Residual Feature Extractor."""

    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(256 * width_mult)
        c5 = int(512 * width_mult)

        d1 = max(1, int(1 * depth_mult))
        d2 = max(1, int(2 * depth_mult))
        d3 = max(1, int(3 * depth_mult))
        d4 = max(1, int(2 * depth_mult))

        # Stem: Stride 2
        self.stem = nn.Sequential(
            ConvBlock(3, c1, k=3, s=2),
            ConvBlock(c1, c2, k=3, s=1),
        )

        # Stage 1: Stride 4
        self.stage1 = nn.Sequential(
            ConvBlock(c2, c2, k=3, s=2),
            *[KDelResidualBlock(c2) for _ in range(d1)],
        )

        # Stage 2: Stride 8 (P3)
        self.stage2 = nn.Sequential(
            ConvBlock(c2, c3, k=3, s=2),
            *[KDelResidualBlock(c3) for _ in range(d2)],
        )

        # Stage 3: Stride 16 (P4)
        self.stage3 = nn.Sequential(
            ConvBlock(c3, c4, k=3, s=2),
            *[KDelResidualBlock(c4) for _ in range(d3)],
        )

        # Stage 4: Stride 32 (P5)
        self.stage4 = nn.Sequential(
            ConvBlock(c4, c5, k=3, s=2),
            *[KDelResidualBlock(c5) for _ in range(d4)],
        )

        self.out_channels = [c3, c4, c5]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # Stride 8
        p4 = self.stage3(p3)  # Stride 16
        p5 = self.stage4(p4)  # Stride 32
        return p3, p4, p5


# --- 3. Path Aggregation Feature Pyramid Network (PANet) ---

class KDelPANet(nn.Module):
    """Bi-directional Top-Down & Bottom-Up Feature Aggregation Network."""

    def __init__(self, in_channels: List[int]):
        super().__init__()
        c3, c4, c5 = in_channels

        # Top-down lateral layers
        self.lateral_p5 = ConvBlock(c5, c4, k=1, s=1)
        self.lateral_p4 = ConvBlock(c4, c3, k=1, s=1)

        # Smooth layers
        self.smooth_p4 = ConvBlock(c4, c4, k=3, s=1)
        self.smooth_p3 = ConvBlock(c3, c3, k=3, s=1)

        # Bottom-up path
        self.down_p3 = ConvBlock(c3, c4, k=3, s=2)
        self.smooth_down_p4 = ConvBlock(c4 * 2, c4, k=3, s=1)

        self.down_p4 = ConvBlock(c4, c5, k=3, s=2)
        self.smooth_down_p5 = ConvBlock(c5 + c5, c5, k=3, s=1)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Top-down flow
        p5_up = F.interpolate(self.lateral_p5(p5), scale_factor=2, mode="nearest")
        p4_fused = self.smooth_p4(p4 + p5_up)

        p4_up = F.interpolate(self.lateral_p4(p4_fused), scale_factor=2, mode="nearest")
        p3_fused = self.smooth_p3(p3 + p4_up)

        # Bottom-up flow
        p3_down = self.down_p3(p3_fused)
        p4_out = self.smooth_down_p4(torch.cat([p4_fused, p3_down], dim=1))

        p4_down = self.down_p4(p4_out)
        p5_out = self.smooth_down_p5(torch.cat([p5, p4_down], dim=1))

        return p3_fused, p4_out, p5_out


# --- 4. Decoupled Multi-Scale Detection Head ---

class KDelDecoupledHead(nn.Module):
    """Decoupled Classification, Bounding Box Regression, and Objectness Head."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, k=3),
            ConvBlock(in_channels, in_channels, k=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

        # Bounding box regression branch (tx, ty, tw, th)
        self.reg_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, k=3),
            ConvBlock(in_channels, in_channels, k=3),
            nn.Conv2d(in_channels, 4, kernel_size=1),
        )

        # Objectness branch
        self.obj_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, k=3),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_conv(x)  # (B, num_classes, H, W)
        reg_preds = self.reg_conv(x)   # (B, 4, H, W)
        obj_logits = self.obj_conv(x)  # (B, 1, H, W)
        return cls_logits, reg_preds, obj_logits


# --- 5. Full KDel 4.0 Neural Network Model ---

class KDel4Model(nn.Module):
    """KDel 4.0: Complete End-to-End Object Detection & Recognition Architecture."""

    def __init__(self, num_classes: int = 80, variant: str = "standard"):
        super().__init__()
        self.num_classes = max(2, num_classes)
        self.variant = variant.lower()

        # Variant scaling factors
        if "nano" in self.variant or "lite" in self.variant:
            width_mult, depth_mult = 0.5, 0.67
        elif "pro" in self.variant or "large" in self.variant:
            width_mult, depth_mult = 1.25, 1.33
        else:
            width_mult, depth_mult = 1.0, 1.0

        self.backbone = KDelBackbone(width_mult=width_mult, depth_mult=depth_mult)
        c3, c4, c5 = self.backbone.out_channels
        self.panet = KDelPANet([c3, c4, c5])

        # Heads at strides 8, 16, 32
        self.head_p3 = KDelDecoupledHead(c3, self.num_classes)
        self.head_p4 = KDelDecoupledHead(c4, self.num_classes)
        self.head_p5 = KDelDecoupledHead(c5, self.num_classes)
        self.strides = [8, 16, 32]

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Any:
        # 1. Multi-scale feature extraction
        p3, p4, p5 = self.backbone(x)

        # 2. Bi-directional feature fusion
        f3, f4, f5 = self.panet(p3, p4, p5)

        # 3. Head predictions
        out_p3 = self.head_p3(f3)
        out_p4 = self.head_p4(f4)
        out_p5 = self.head_p5(f5)

        raw_outputs = [out_p3, out_p4, out_p5]

        if self.training:
            return raw_outputs

        # Inference: decode raw outputs into normalized [x1, y1, x2, y2, score, class_id]
        return self._decode_predictions(raw_outputs, x.shape[2:])

    def _decode_predictions(self, raw_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], img_shape: Tuple[int, int]) -> torch.Tensor:
        """Decodes multi-scale grid offsets into unified bounding boxes [0.0, 1.0]."""
        batch_size = raw_outputs[0][0].shape[0]
        all_boxes = []

        h_img, w_img = img_shape

        for (cls_logits, reg_preds, obj_logits), stride in zip(raw_outputs, self.strides):
            b, c, h, w = cls_logits.shape

            # Sigmoids
            obj_conf = torch.sigmoid(obj_logits)               # (B, 1, H, W)
            cls_probs = torch.softmax(cls_logits, dim=1)        # (B, C, H, W)

            # Generate grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=cls_logits.device),
                torch.arange(w, device=cls_logits.device),
                indexing="ij"
            )
            grid_x = grid_x.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            grid_y = grid_y.float().unsqueeze(0).unsqueeze(0)

            # Decode center and size (normalized to [0, 1])
            cx = (grid_x + torch.sigmoid(reg_preds[:, 0:1, :, :])) * stride / w_img
            cy = (grid_y + torch.sigmoid(reg_preds[:, 1:2, :, :])) * stride / h_img
            bw = (torch.exp(reg_preds[:, 2:3, :, :].clamp(-4, 4)) * stride) / w_img
            bh = (torch.exp(reg_preds[:, 3:4, :, :].clamp(-4, 4)) * stride) / h_img

            # Convert to [x1, y1, x2, y2]
            x1 = (cx - bw / 2).clamp(0.0, 1.0)
            y1 = (cy - bh / 2).clamp(0.0, 1.0)
            x2 = (cx + bw / 2).clamp(0.0, 1.0)
            y2 = (cy + bh / 2).clamp(0.0, 1.0)

            # Flatten spatial dims: (B, H*W, ...)
            boxes = torch.cat([x1, y1, x2, y2], dim=1).permute(0, 2, 3, 1).reshape(b, -1, 4)
            obj = obj_conf.permute(0, 2, 3, 1).reshape(b, -1, 1)
            cls_p = cls_probs.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)

            # Output per scale: (B, H*W, 4 + 1 + num_classes)
            scale_out = torch.cat([boxes, obj, cls_p], dim=-1)
            all_boxes.append(scale_out)

        return torch.cat(all_boxes, dim=1)  # (B, TotalAnchors, 5 + num_classes)


# --- 6. KDel 4.0 Loss Function ---

class KDelLoss(nn.Module):
    """Composite Multi-Task Loss: Smooth-L1 / CIoU + BCE Objectness + BCE Classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = max(2, num_classes)
        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.CrossEntropyLoss()
        self.l1_reg = nn.SmoothL1Loss(reduction="mean")

    def forward(self, raw_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculates loss between raw multi-scale feature maps and ground truth boxes."""
        device = raw_outputs[0][0].device
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)

        batch_size = raw_outputs[0][0].shape[0]

        # For each scale head
        for cls_logits, reg_preds, obj_logits in raw_outputs:
            b, _, h, w = cls_logits.shape

            # Target objectness tensor: default zeros (background)
            target_obj = torch.zeros_like(obj_logits)
            matched_regs = []
            target_regs = []
            matched_cls = []
            target_cls = []

            for bi in range(batch_size):
                sample_targets = targets[bi] if bi < len(targets) else None
                if sample_targets is None or len(sample_targets) == 0:
                    continue

                for t in sample_targets:
                    # t: [class_id, cx, cy, w, h] in normalized [0, 1]
                    cid = min(self.num_classes - 1, max(0, int(t[0].item())))
                    cx, cy, tw, th = t[1].item(), t[2].item(), t[3].item(), t[4].item()

                    gx = min(w - 1, max(0, int(cx * w)))
                    gy = min(h - 1, max(0, int(cy * h)))

                    target_obj[bi, 0, gy, gx] = 1.0

                    matched_regs.append(reg_preds[bi, :, gy, gx])
                    target_regs.append(torch.tensor([cx * w - gx, cy * h - gy, math.log(max(1e-4, tw * w)), math.log(max(1e-4, th * h))], device=device))

                    matched_cls.append(cls_logits[bi, :, gy, gx])
                    target_cls.append(torch.tensor(cid, device=device, dtype=torch.long))

            # Objectness loss
            scale_obj_loss = self.bce_obj(obj_logits, target_obj)
            total_obj_loss = total_obj_loss + scale_obj_loss

            # Regression loss if targets exist
            if matched_regs:
                pred_r = torch.stack(matched_regs)
                targ_r = torch.stack(target_regs)
                total_reg_loss = total_reg_loss + self.l1_reg(pred_r, targ_r)

            # Classification loss if targets exist
            if matched_cls:
                pred_c = torch.stack(matched_cls)
                targ_c = torch.stack(target_cls)
                total_cls_loss = total_cls_loss + self.bce_cls(pred_c, targ_c)

        total_loss = total_obj_loss + 2.0 * total_reg_loss + 1.0 * total_cls_loss
        return {
            "loss": total_loss,
            "obj_loss": total_obj_loss,
            "reg_loss": total_reg_loss,
            "cls_loss": total_cls_loss,
        }


# --- 7. Non-Maximum Suppression (NMS) ---

def kdel_nms(predictions: torch.Tensor, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    """Pure PyTorch Non-Maximum Suppression on batch prediction tensor."""
    detections = []
    # predictions: (N, 4 + 1 + num_classes) -> [x1, y1, x2, y2, obj_conf, c0, c1, ...]
    if predictions.ndim == 3:
        predictions = predictions[0]

    boxes = predictions[:, :4]
    obj_confs = predictions[:, 4]
    cls_probs = predictions[:, 5:]

    max_cls_probs, class_ids = torch.max(cls_probs, dim=1)
    scores = obj_confs * max_cls_probs

    # Filter confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Sort descending by score
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    scores = scores[order]
    class_ids = class_ids[order]

    keep = []
    while len(order) > 0:
        idx = order[0].item()
        keep.append(idx)
        if len(order) == 1:
            break

        cur_box = boxes[0:1]
        other_boxes = boxes[1:]

        # Calculate IoU
        x1 = torch.maximum(cur_box[:, 0], other_boxes[:, 0])
        y1 = torch.maximum(cur_box[:, 1], other_boxes[:, 1])
        x2 = torch.minimum(cur_box[:, 2], other_boxes[:, 2])
        y2 = torch.minimum(cur_box[:, 3], other_boxes[:, 3])

        inter_area = torch.clamp(x2 - x1, min=0.0) * torch.clamp(y2 - y1, min=0.0)
        box_area = (cur_box[:, 2] - cur_box[:, 0]) * (cur_box[:, 3] - cur_box[:, 1])
        other_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

        iou = inter_area / torch.clamp(box_area + other_area - inter_area, min=1e-6)

        # Keep indices with IoU < threshold
        keep_mask = iou <= iou_threshold
        order = order[1:][keep_mask]
        boxes = other_boxes[keep_mask]
        scores = scores[1:][keep_mask]
        class_ids = class_ids[1:][keep_mask]

    # Convert kept detections to list of dicts
    result = []
    for i in range(min(100, len(boxes))):
        result.append({
            "box": {
                "x1": float(boxes[i, 0].item()),
                "y1": float(boxes[i, 1].item()),
                "x2": float(boxes[i, 2].item()),
                "y2": float(boxes[i, 3].item()),
            },
            "confidence": float(scores[i].item()),
            "class_id": int(class_ids[i].item()),
        })
    return result


# Aliases and Model Builders
KDelModel = KDel4Model


def build_kdel4(variant: str = "standard", num_classes: int = 80) -> KDel4Model:
    """Factory helper to construct KDel 4.0 neural network model."""
    return KDel4Model(num_classes=num_classes, variant=variant)
