"""Mask R-CNN builder for ZAsolar detection.

Single source of truth for constructing the detection model across
`train.py` and future standalone inference runners. Keeps the torchvision
`maskrcnn_resnet50_fpn` backbone we've always used, so existing
checkpoints load unchanged — this module replaces geoai's opaque wrapper
without changing model topology.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_solar_maskrcnn(
    pretrained_path: str | Path | None = None,
    num_classes: int = 2,
    image_mean: Sequence[float] | None = None,
    image_std: Sequence[float] | None = None,
    strict_load: bool = False,
) -> torch.nn.Module:
    """Build Mask R-CNN ResNet50-FPN with ImageNet-normalized inputs.

    Args:
        pretrained_path: optional .pth to load as initial weights. Accepts raw
            state_dicts as well as checkpoints nested under 'model' or
            'state_dict'. 'module.' prefixes (DataParallel) are stripped.
        num_classes: detection classes including background (default: 2).
        image_mean / image_std: override normalization; default is ImageNet.
        strict_load: forwarded to load_state_dict. Default False to tolerate
            head size mismatches when transferring across dataset schemas.
    """
    mean = list(image_mean) if image_mean is not None else list(IMAGENET_MEAN)
    std = list(image_std) if image_std is not None else list(IMAGENET_STD)

    model = maskrcnn_resnet50_fpn(
        weights=None,
        progress=False,
        num_classes=num_classes,
        weights_backbone=None,
        image_mean=mean,
        image_std=std,
    )

    if pretrained_path is not None:
        state_dict = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
        print(
            f"[MODEL] Loaded weights from {pretrained_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    return model
