"""Boundary-aware Mask R-CNN loss patch.

Replaces ``torchvision.models.detection.roi_heads.maskrcnn_loss`` with a
variant that supports:

  1. Per-pixel ignore band: skips BCE on a band straddling the polygon edge.
  2. Per-instance mask weight: scales (or zeros out) the BCE for entire
     instances. ``mask_weight=0`` means an instance is dropped from mask
     supervision entirely (still contributes to box + cls).

Mechanism: torchvision's RoIHeads.forward looks up the loss function via
``roi_heads.maskrcnn_loss`` at call time, so replacing the module-level
attribute is enough — no subclass / forward override needed.

The patch reads the per-image ignore_masks and mask_weights from a
module-level batch state (``stash_batch_supervision``) because torchvision's
RoIHeads.forward only forwards ``t["masks"]`` and ``t["labels"]`` to the loss
fn. The training loop is responsible for stashing and clearing state per
batch.

Usage:
    from core.training.boundary_aware_mask import (
        install_patch, stash_batch_supervision, clear_batch_supervision,
    )
    install_patch()  # once at startup
    ...
    for images, targets in loader:
        stash_batch_supervision(targets)
        loss_dict = model(images, targets)  # patched maskrcnn_loss reads stash
        clear_batch_supervision()
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.detection import roi_heads as _rh
from torchvision.ops import roi_align


_BATCH_STATE: dict = {"ignore_masks": None, "mask_weights": None}


def stash_batch_supervision(targets: list[dict]) -> None:
    """Stash per-image supervision tensors. Call before model(images, targets)."""
    _BATCH_STATE["ignore_masks"] = [t.get("ignore_masks") for t in targets]
    _BATCH_STATE["mask_weights"] = [t.get("mask_weights") for t in targets]


def clear_batch_supervision() -> None:
    _BATCH_STATE["ignore_masks"] = None
    _BATCH_STATE["mask_weights"] = None


def _project_masks_on_boxes(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    matched_idxs: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """Same as torchvision's project_masks_on_boxes but reusable for ignore masks."""
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    masks = masks[:, None].to(rois)
    return roi_align(masks, rois, (M, M), 1.0)[:, 0]


def patched_maskrcnn_loss(
    mask_logits: torch.Tensor,
    proposals: list[torch.Tensor],
    gt_masks: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    mask_matched_idxs: list[torch.Tensor],
) -> torch.Tensor:
    """Boundary-aware BCE: per-pixel ignore + per-instance weight."""
    M = mask_logits.shape[-1]

    labels = [gl[idxs] for gl, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        _project_masks_on_boxes(m, p, i, M)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    weights_per_img = _BATCH_STATE.get("mask_weights")
    has_weights = (
        weights_per_img is not None
        and any(w is not None for w in weights_per_img)
    )
    if has_weights:
        per_prop_w = []
        for w_full, idxs in zip(weights_per_img, mask_matched_idxs):
            if w_full is None or len(w_full) == 0:
                per_prop_w.append(torch.ones(len(idxs), device=mask_logits.device))
            else:
                w_full_dev = w_full.to(idxs.device)
                per_prop_w.append(w_full_dev[idxs].to(mask_logits.device))
        per_prop_w = torch.cat(per_prop_w, dim=0) if per_prop_w else None
    else:
        per_prop_w = None

    ignores_per_img = _BATCH_STATE.get("ignore_masks")
    has_ignores = (
        ignores_per_img is not None
        and any(im is not None for im in ignores_per_img)
    )
    if has_ignores:
        ig_targets = []
        for im, p, i in zip(ignores_per_img, proposals, mask_matched_idxs):
            if im is None or len(im) == 0:
                ig_targets.append(torch.zeros(len(p), M, M, device=p.device))
            else:
                ig_targets.append(_project_masks_on_boxes(im, p, i, M))
        ig_targets = torch.cat(ig_targets, dim=0) if ig_targets else None
    else:
        ig_targets = None

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    selected_logits = mask_logits[
        torch.arange(labels.shape[0], device=labels.device), labels
    ]
    bce_per_pixel = F.binary_cross_entropy_with_logits(
        selected_logits, mask_targets, reduction="none"
    )

    if ig_targets is not None:
        valid_pixel = (ig_targets <= 0.5).to(bce_per_pixel.dtype)
    else:
        valid_pixel = torch.ones_like(bce_per_pixel)

    if per_prop_w is not None:
        w_pp = per_prop_w[:, None, None].to(bce_per_pixel.dtype)
        weighted = bce_per_pixel * valid_pixel * w_pp
        denom = (valid_pixel * w_pp).sum().clamp(min=1.0)
    else:
        weighted = bce_per_pixel * valid_pixel
        denom = valid_pixel.sum().clamp(min=1.0)

    return weighted.sum() / denom


_ORIGINAL_LOSS = _rh.maskrcnn_loss


def install_patch() -> None:
    """Replace torchvision's maskrcnn_loss. Idempotent."""
    _rh.maskrcnn_loss = patched_maskrcnn_loss


def restore_original() -> None:
    """Restore torchvision's stock maskrcnn_loss (e.g. for tests)."""
    _rh.maskrcnn_loss = _ORIGINAL_LOSS


def is_installed() -> bool:
    return _rh.maskrcnn_loss is patched_maskrcnn_loss
