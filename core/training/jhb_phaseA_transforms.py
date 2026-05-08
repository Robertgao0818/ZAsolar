"""Boundary-aware augmentation transforms.

Mirrors train.py's TrainTransforms but synchronises ``target["ignore_masks"]``
with ``target["masks"]`` for every spatial op (flip, rotate, scale jitter).
``mask_weights`` is per-instance scalar so it doesn't need spatial transform,
only re-indexing when instances are filtered.
"""
from __future__ import annotations

import torch


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    n = masks.shape[0]
    boxes = []
    for i in range(n):
        pos = torch.where(masks[i])
        if len(pos[0]) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            y_min = pos[0].min().item()
            y_max = pos[0].max().item()
            x_min = pos[1].min().item()
            x_max = pos[1].max().item()
            boxes.append([x_min, y_min, x_max + 1, y_max + 1])
    return torch.tensor(boxes, dtype=torch.float32)


class BoundaryAwareTrainTransforms:
    """Same augmentation as train.py's TrainTransforms, but propagates to
    ``ignore_masks`` and ``mask_weights``."""

    def __init__(self, chip_size: int = 400):
        self.chip_size = chip_size

    def __call__(self, image, target):
        # Horizontal flip
        if torch.rand(1) < 0.5:
            image = image.flip(-1)
            if target["boxes"].numel() > 0:
                w = image.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
                target["masks"] = target["masks"].flip(-1)
                if "ignore_masks" in target:
                    target["ignore_masks"] = target["ignore_masks"].flip(-1)

        # Vertical flip
        if torch.rand(1) < 0.5:
            image = image.flip(-2)
            if target["boxes"].numel() > 0:
                h = image.shape[-2]
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
                target["masks"] = target["masks"].flip(-2)
                if "ignore_masks" in target:
                    target["ignore_masks"] = target["ignore_masks"].flip(-2)

        # 90-degree rotations
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, [-2, -1])
            if target["boxes"].numel() > 0:
                target["masks"] = torch.rot90(target["masks"], k, [-2, -1])
                if "ignore_masks" in target:
                    target["ignore_masks"] = torch.rot90(target["ignore_masks"], k, [-2, -1])
                target["boxes"] = _masks_to_boxes(target["masks"])

        # Color jitter (image only)
        if torch.rand(1) < 0.8:
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            image = image * brightness
            contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = (image - mean) * contrast + mean
            saturation = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
            gray = image.mean(dim=0, keepdim=True)
            image = (image - gray) * saturation + gray
            image = image.clamp(0.0, 1.0)

        # Scale jitter
        if torch.rand(1) < 0.5:
            scale = 0.8 + torch.rand(1).item() * 0.4
            _, h, w = image.shape
            new_h = int(h * scale)
            new_w = int(w * scale)
            image_resized = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0)
            out_img = torch.zeros_like(image)
            ph = min(new_h, h)
            pw = min(new_w, w)
            out_img[:, :ph, :pw] = image_resized[:, :ph, :pw]
            image = out_img

            if target["masks"].numel() > 0:
                masks_resized = torch.nn.functional.interpolate(
                    target["masks"].unsqueeze(1).float(),
                    size=(new_h, new_w), mode="nearest"
                ).squeeze(1).byte()
                out_m = torch.zeros(masks_resized.shape[0], h, w, dtype=torch.uint8)
                out_m[:, :ph, :pw] = masks_resized[:, :ph, :pw]
                target["masks"] = out_m

                if "ignore_masks" in target and target["ignore_masks"].numel() > 0:
                    ig_resized = torch.nn.functional.interpolate(
                        target["ignore_masks"].unsqueeze(1).float(),
                        size=(new_h, new_w), mode="nearest"
                    ).squeeze(1).byte()
                    out_ig = torch.zeros(ig_resized.shape[0], h, w, dtype=torch.uint8)
                    out_ig[:, :ph, :pw] = ig_resized[:, :ph, :pw]
                    target["ignore_masks"] = out_ig

                target["boxes"] = _masks_to_boxes(target["masks"])

                # Filter instances whose fg mask was scaled out of the chip
                valid = target["masks"].sum(dim=(-2, -1)) > 0
                if valid.any():
                    target["boxes"] = target["boxes"][valid]
                    target["labels"] = target["labels"][valid]
                    target["masks"] = target["masks"][valid]
                    if "ignore_masks" in target:
                        target["ignore_masks"] = target["ignore_masks"][valid]
                    if "mask_weights" in target:
                        target["mask_weights"] = target["mask_weights"][valid]
                    target["area"] = target["area"][valid] if valid.sum() <= len(target["area"]) \
                        else target["boxes"][:, 2:].prod(dim=1)
                else:
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                    target["labels"] = torch.zeros(0, dtype=torch.int64)
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
                    if "ignore_masks" in target:
                        target["ignore_masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
                    if "mask_weights" in target:
                        target["mask_weights"] = torch.zeros(0, dtype=torch.float32)
                    target["area"] = torch.zeros(0, dtype=torch.float32)

        return image, target


class ValTransforms:
    def __call__(self, image, target):
        return image, target
