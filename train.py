"""
Fine-tune Mask R-CNN (ResNet50-FPN) on Cape Town solar panel annotations.

Two-stage training:
  Stage 1 — Heads-only:  ROI heads + mask head, backbone frozen, 3 epochs, LR=1e-3
  Stage 2 — Full fine-tune: all parameters, up to 20 epochs, LR=1e-4, cosine decay

Checkpoint selection: best validation segm_AP50.
Output: .pth state_dict, loadable via core.models.build_solar_maskrcnn.

Usage:
    python train.py --coco-dir data/coco [--output-dir checkpoints] [--epochs2 20]

Requires CUDA GPU.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from core.models import build_solar_maskrcnn
from core.profiling import StageProfiler

assert torch.cuda.is_available(), (
    "CUDA is required for training. WSL2 does not currently expose CUDA to PyTorch. "
    "Run this script on a CUDA-capable machine."
)


# ════════════════════════════════════════════════════════════════════════
# COCO Dataset
# ════════════════════════════════════════════════════════════════════════
# Boundary BCE weight by annotation provenance.
#
# Rule: any annotation whose detection originated from a V3-C model prediction
# (and was then accepted via human review, with or without SAM re-segmentation)
# must contribute zero weight to mask-boundary supervision — its edges carry
# V3-C halo bias and feeding them back into mask-head training closes the
# reviewed-prediction → halo cycle.  Region (CT vs JHB) is irrelevant: the
# disqualifier is "V3-C-derived detection", not the imagery source.
#
# Annotator-initiated annotations (pure human, SAM-as-drawing-tool, SAM-added
# true-FN catches that V3-C missed) keep full weight on boundary supervision —
# they break the cycle by injecting fresh edge information.
_LABEL_SOURCE_TO_BOUNDARY_W = {
    # H (annotator-initiated, per-instance interactive): full boundary weight
    "human_manual": 1.0,
    "human_manual_sam_assisted": 1.0,
    "human_manual_qgis_geosam": 1.0,
    "sam_added_browser": 1.0,           # browser SAM point-prompt, interactive (post-2026-04-13)
    # R / S / batch-SAM / legacy: zero on edge (mask_trusted=False sources)
    "sam_refined_review": 0.0,
    "reviewed_prediction": 0.0,
    "sam_added_true_fn": 0.0,           # Batch 003/004 non-interactive SAM cut
    "legacy_weak_supervision": 0.0,
}

# Per-instance mask supervision gate (mirror of export_coco_dataset._MASK_TRUSTED).
# True  → instance contributes full mask BCE.
# False → mask BCE skipped entirely (box + cls still apply).
_LABEL_SOURCE_TO_MASK_TRUSTED = {
    "human_manual": True,
    "human_manual_sam_assisted": True,
    "human_manual_qgis_geosam": True,
    "sam_added_browser": True,
    "reviewed_prediction": False,
    "sam_refined_review": False,
    "sam_added_true_fn": False,
    "legacy_weak_supervision": False,
}


def _boundary_pixel_weights(mask_np, label_source, band_iters=2):
    """Per-pixel BCE weight: 1.0 outside boundary band, source-dependent inside.

    Band built by dilate(mask, k) XOR erode(mask, k) — a ~2*band_iters px ring
    centered on the polygon edge. Inside band, weight = source map (H=1.0,
    S=0.5, R=0.0). Foreground core and background pixels stay weight=1.0 so
    pred-vs-bg classification is preserved for all sources; only boundary
    shape supervision is down-weighted for noisy labels.
    """
    bw = _LABEL_SOURCE_TO_BOUNDARY_W.get(label_source, 1.0)
    if bw == 1.0:
        return np.ones_like(mask_np, dtype=np.float32)
    import cv2
    kernel = np.ones((3, 3), dtype=np.uint8)
    dil = cv2.dilate(mask_np, kernel, iterations=band_iters)
    ero = cv2.erode(mask_np, kernel, iterations=band_iters)
    band = (dil.astype(np.int8) ^ ero.astype(np.int8)) > 0
    out = np.ones_like(mask_np, dtype=np.float32)
    out[band] = bw
    return out


def _resolve_mask_trusted(ann: dict) -> bool:
    """Return mask_trusted for one COCO annotation.

    Priority:
    1. explicit ann["mask_trusted"] (written by export_coco_dataset when
       label_source is known) → use as-is.
    2. ann["label_source"] in the trusted-source map → use mapped value.
    3. ann has label_source but value is unknown → False (conservative:
       unknown provenance is treated as untrusted to avoid silently feeding
       potentially halo-biased data into mask BCE).
    4. ann has neither field (legacy export) → True (backwards compat: old
       COCO files predate the trusted-source machinery and should keep their
       prior full-supervision behavior)."""
    if "mask_trusted" in ann:
        return bool(ann["mask_trusted"])
    ls = ann.get("label_source")
    if ls is None:
        return True
    return _LABEL_SOURCE_TO_MASK_TRUSTED.get(str(ls), False)


class CocoSolarDataset(torch.utils.data.Dataset):
    """COCO instance segmentation dataset for solar panel chips."""

    def __init__(self, coco_json: Path, root_dir: Path, transforms=None,
                 per_source_mask_weight: bool = False, boundary_band_iters: int = 2,
                 per_instance_mask_trusted: bool = False):
        with open(coco_json, "r") as f:
            data = json.load(f)

        self.root_dir = root_dir
        self.transforms = transforms
        self.per_source_mask_weight = per_source_mask_weight
        self.boundary_band_iters = boundary_band_iters
        self.per_instance_mask_trusted = per_instance_mask_trusted

        self.images = {img["id"]: img for img in data["images"]}
        self.image_ids = [img["id"] for img in data["images"]]

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in data["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        if per_source_mask_weight or per_instance_mask_trusted:
            from collections import Counter
            srcs = Counter(a.get("label_source", "MISSING") for a in data["annotations"])
            print(f"[DATASET] per_source_mask_weight={per_source_mask_weight} "
                  f"per_instance_mask_trusted={per_instance_mask_trusted}; "
                  f"label_source counts: {dict(srcs)}")
            if per_instance_mask_trusted:
                trusted = sum(
                    1 for a in data["annotations"]
                    if _resolve_mask_trusted(a)
                )
                untrusted = len(data["annotations"]) - trusted
                print(f"[DATASET] mask_trusted breakdown: trusted={trusted} "
                      f"untrusted={untrusted}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.root_dir / img_info["file_name"]
        import rasterio
        with rasterio.open(str(img_path)) as src:
            data = src.read()  # (C, H, W)
        image = torch.as_tensor(data, dtype=torch.float32) / 255.0

        # Build target
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        masks = []
        areas = []
        pix_weights = []
        per_instance_weights = []
        label_sources = []

        h, w = img_info["height"], img_info["width"]

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            label_sources.append(ann.get("label_source") or "")

            # Decode segmentation to mask
            mask = np.zeros((h, w), dtype=np.uint8)
            for seg in ann["segmentation"]:
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                import cv2
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

            if self.per_source_mask_weight:
                pix_weights.append(
                    _boundary_pixel_weights(
                        mask, ann.get("label_source"), self.boundary_band_iters
                    )
                )

            if self.per_instance_mask_trusted:
                per_instance_weights.append(
                    1.0 if _resolve_mask_trusted(ann) else 0.0
                )

        if len(boxes) == 0:
            # Empty image — return empty targets
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
                "label_sources": [],
            }
            if self.per_source_mask_weight:
                target["mask_pixel_weights"] = torch.zeros((0, h, w), dtype=torch.float32)
            if self.per_instance_mask_trusted:
                target["mask_weights"] = torch.zeros(0, dtype=torch.float32)
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
                "label_sources": label_sources,
            }
            if self.per_source_mask_weight:
                target["mask_pixel_weights"] = torch.as_tensor(
                    np.array(pix_weights), dtype=torch.float32
                )
            if self.per_instance_mask_trusted:
                target["mask_weights"] = torch.as_tensor(
                    per_instance_weights, dtype=torch.float32
                )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# ════════════════════════════════════════════════════════════════════════
# Data Augmentation
# ════════════════════════════════════════════════════════════════════════
class TrainTransforms:
    """Random augmentations for training: flips, rotations, color jitter, scale."""

    def __init__(self, chip_size=400):
        self.chip_size = chip_size

    def __call__(self, image, target):
        # image: (C, H, W), target: dict with boxes, masks, etc.
        has_pix_w = "mask_pixel_weights" in target

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = image.flip(-1)  # flip W
            if target["boxes"].numel() > 0:
                w = image.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
                target["masks"] = target["masks"].flip(-1)
                if has_pix_w and target["mask_pixel_weights"].numel() > 0:
                    target["mask_pixel_weights"] = target["mask_pixel_weights"].flip(-1)

        # Random vertical flip
        if torch.rand(1) < 0.5:
            image = image.flip(-2)  # flip H
            if target["boxes"].numel() > 0:
                h = image.shape[-2]
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
                target["masks"] = target["masks"].flip(-2)
                if has_pix_w and target["mask_pixel_weights"].numel() > 0:
                    target["mask_pixel_weights"] = target["mask_pixel_weights"].flip(-2)

        # Random 90° rotation (0, 90, 180, 270)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, [-2, -1])
            if target["boxes"].numel() > 0:
                masks = target["masks"]
                masks = torch.rot90(masks, k, [-2, -1])
                target["masks"] = masks
                if has_pix_w and target["mask_pixel_weights"].numel() > 0:
                    target["mask_pixel_weights"] = torch.rot90(
                        target["mask_pixel_weights"], k, [-2, -1]
                    )
                # Recompute boxes from masks
                target["boxes"] = masks_to_boxes(masks)

        # Color jitter (brightness, contrast, saturation)
        if torch.rand(1) < 0.8:  # apply 80% of the time
            # Brightness: ±0.2
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            image = image * brightness

            # Contrast: ±0.2
            contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = (image - mean) * contrast + mean

            # Saturation: ±0.15
            saturation = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
            gray = image.mean(dim=0, keepdim=True)
            image = (image - gray) * saturation + gray

            image = image.clamp(0.0, 1.0)

        # Random scale jitter: 0.8x – 1.2x
        if torch.rand(1) < 0.5:
            scale = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            _, h, w = image.shape
            new_h = int(h * scale)
            new_w = int(w * scale)

            image_resized = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), mode="bilinear",
                align_corners=False
            ).squeeze(0)

            # Pad or crop to original size
            out = torch.zeros_like(image)
            ph = min(new_h, h)
            pw = min(new_w, w)
            out[:, :ph, :pw] = image_resized[:, :ph, :pw]
            image = out

            if target["masks"].numel() > 0:
                masks_resized = torch.nn.functional.interpolate(
                    target["masks"].unsqueeze(1).float(),
                    size=(new_h, new_w), mode="nearest"
                ).squeeze(1).byte()

                out_masks = torch.zeros(
                    masks_resized.shape[0], h, w, dtype=torch.uint8
                )
                out_masks[:, :ph, :pw] = masks_resized[:, :ph, :pw]
                target["masks"] = out_masks
                target["boxes"] = masks_to_boxes(out_masks)

                if has_pix_w and target["mask_pixel_weights"].numel() > 0:
                    pw_resized = torch.nn.functional.interpolate(
                        target["mask_pixel_weights"].unsqueeze(1),
                        size=(new_h, new_w), mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    out_pw = torch.ones(
                        pw_resized.shape[0], h, w, dtype=torch.float32
                    )
                    out_pw[:, :ph, :pw] = pw_resized[:, :ph, :pw]
                    target["mask_pixel_weights"] = out_pw

                # Filter out empty masks
                valid = target["masks"].sum(dim=(-2, -1)) > 0
                has_inst_w = "mask_weights" in target
                has_label_src = "label_sources" in target
                if valid.any():
                    target["boxes"] = target["boxes"][valid]
                    target["labels"] = target["labels"][valid]
                    target["masks"] = target["masks"][valid]
                    target["area"] = target["area"][valid] if valid.sum() <= len(target["area"]) else target["boxes"][:, 2:].prod(dim=1)
                    if has_pix_w and target["mask_pixel_weights"].numel() > 0:
                        target["mask_pixel_weights"] = target["mask_pixel_weights"][valid]
                    if has_inst_w and target["mask_weights"].numel() > 0:
                        target["mask_weights"] = target["mask_weights"][valid]
                    if has_label_src and target["label_sources"]:
                        valid_list = valid.tolist()
                        target["label_sources"] = [
                            ls for ls, v in zip(target["label_sources"], valid_list) if v
                        ]
                else:
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                    target["labels"] = torch.zeros(0, dtype=torch.int64)
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
                    target["area"] = torch.zeros(0, dtype=torch.float32)
                    if has_pix_w:
                        target["mask_pixel_weights"] = torch.zeros((0, h, w), dtype=torch.float32)
                    if has_inst_w:
                        target["mask_weights"] = torch.zeros(0, dtype=torch.float32)
                    if has_label_src:
                        target["label_sources"] = []

        return image, target


def masks_to_boxes(masks):
    """Compute bounding boxes from binary masks. masks: (N, H, W)."""
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


class ValTransforms:
    """No augmentations for validation."""
    def __call__(self, image, target):
        return image, target


# ════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════
def build_model(pretrained_path: str | None = None, num_classes: int = 2):
    """Thin wrapper over core.models.build_solar_maskrcnn for backcompat."""
    return build_solar_maskrcnn(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
    )


def freeze_backbone(model):
    """Freeze backbone + FPN, keep ROI heads + mask head trainable."""
    for name, param in model.named_parameters():
        if name.startswith("backbone") or name.startswith("fpn"):
            param.requires_grad = False
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total = sum(1 for p in model.parameters())
    print(f"[FREEZE] {n_frozen}/{n_total} parameters frozen")


def reinit_mask_head(model):
    """Replace mask_head conv stack + mask_predictor with fresh weights.

    Wipes halo bias accumulated in mask supervision. Use with --pretrained
    pointing at a halo-iterated checkpoint (e.g. V3-C) to keep backbone/RPN/box
    prior while clearing the mask boundary classifier.
    """
    from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor

    in_channels = model.backbone.out_channels
    mask_layers = (256, 256, 256, 256)
    mask_dilation = 1
    num_classes = model.roi_heads.box_predictor.cls_score.out_features

    model.roi_heads.mask_head = MaskRCNNHeads(in_channels, mask_layers, mask_dilation)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_layers[-1], mask_layers[-1], num_classes)
    print(f"[REINIT] mask_head + mask_predictor reset (num_classes={num_classes})")


def reinit_box_predictor(model):
    """Replace box_predictor (cls_score + bbox_pred) with fresh weights."""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = model.roi_heads.box_predictor.cls_score.out_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"[REINIT] box_predictor reset (num_classes={num_classes})")


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    print("[UNFREEZE] All parameters trainable")


def freeze_mask_head(model):
    """Freeze mask_head conv stack + mask_predictor.

    Keeps backbone + RPN + box_head trainable so they can adapt to new data
    without mask_head learning noise from a fresh/halo'd prior. Per
    2026-05-09-training-supervision-layering.md item #1 and the unified_reviewall
    plan Stage 1.
    """
    n = 0
    for p in model.roi_heads.mask_head.parameters():
        p.requires_grad = False
        n += 1
    for p in model.roi_heads.mask_predictor.parameters():
        p.requires_grad = False
        n += 1
    print(f"[FREEZE-MASK] mask_head + mask_predictor frozen ({n} tensors)")


def unfreeze_mask_head(model):
    """Unfreeze mask_head + mask_predictor at Stage 2 boundary."""
    n = 0
    for p in model.roi_heads.mask_head.parameters():
        p.requires_grad = True
        n += 1
    for p in model.roi_heads.mask_predictor.parameters():
        p.requires_grad = True
        n += 1
    print(f"[UNFREEZE-MASK] mask_head + mask_predictor unfrozen ({n} tensors)")


# ════════════════════════════════════════════════════════════════════════
# Evaluation (COCO AP)
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _compute_f1_at_conf(coco_gt_data: dict, coco_results: list,
                        conf_threshold: float = 0.85,
                        iou_threshold: float = 0.5) -> dict:
    """Compute deployment-threshold F1 on chip val set.

    Mirrors inference-time behavior: filter preds by conf, greedy IoU match
    to GT per image. Catches V4.2-style failure mode where AP50 stays high
    but deployment-threshold precision collapses.
    """
    from pycocotools import mask as mask_util

    gt_by_img: dict[int, list] = {}
    for ann in coco_gt_data["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann["segmentation"])
    preds_by_img: dict[int, list] = {}
    for r in coco_results:
        if r["score"] < conf_threshold:
            continue
        preds_by_img.setdefault(r["image_id"], []).append(r)

    tp = fp = fn = 0
    all_img_ids = {im["id"] for im in coco_gt_data["images"]}
    for img_id in all_img_ids:
        gt_rles = gt_by_img.get(img_id, [])
        pred_list = sorted(preds_by_img.get(img_id, []),
                           key=lambda r: -r["score"])
        if not gt_rles and not pred_list:
            continue
        if not pred_list:
            fn += len(gt_rles)
            continue
        if not gt_rles:
            fp += len(pred_list)
            continue
        matched_gt = set()
        for pr in pred_list:
            ious = mask_util.iou([pr["segmentation"]], gt_rles,
                                 [0] * len(gt_rles))[0]
            best_j, best_iou = -1, iou_threshold
            for j, iou in enumerate(ious):
                if j in matched_gt:
                    continue
                if iou >= best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                matched_gt.add(best_j)
                tp += 1
            else:
                fp += 1
        fn += len(gt_rles) - len(matched_gt)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "conf_threshold": conf_threshold, "iou_threshold": iou_threshold}


@torch.no_grad()
def evaluate_coco(model, data_loader, device):
    """Compute COCO segm AP50 + deployment-conf F1 on the validation set."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_util

    model.eval()

    coco_results = []
    coco_gt_data = {
        "info": {
            "description": "Cape Town Solar Panel Detection - validation eval",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            # Works with both "solar_panel" and "solar_installation" COCO datasets;
            # pycocotools matches by category_id, not name.
            {"id": 1, "name": "solar_panel"}
        ],
    }
    ann_id = 1

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            img_id = target["image_id"].item()
            h = target["masks"].shape[-2] if target["masks"].numel() > 0 else 400
            w = target["masks"].shape[-1] if target["masks"].numel() > 0 else 400

            coco_gt_data["images"].append({
                "id": img_id, "width": w, "height": h
            })

            # Add GT annotations
            for i in range(target["boxes"].shape[0]):
                mask_np = target["masks"][i].cpu().numpy().astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(mask_np))
                rle["counts"] = rle["counts"].decode("utf-8")
                bx, by, bx2, by2 = target["boxes"][i].tolist()
                coco_gt_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "bbox": [bx, by, bx2 - bx, by2 - by],
                    "area": (bx2 - bx) * (by2 - by),
                    "iscrowd": 0,
                })
                ann_id += 1

            # Predictions
            if len(output["scores"]) == 0:
                continue
            for i in range(len(output["scores"])):
                score = output["scores"][i].item()
                mask_np = (output["masks"][i, 0].cpu().numpy() > 0.5).astype(np.uint8)
                if mask_np.sum() == 0:
                    continue
                rle = mask_util.encode(np.asfortranarray(mask_np))
                rle["counts"] = rle["counts"].decode("utf-8")
                bx, by, bx2, by2 = output["boxes"][i].tolist()
                coco_results.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score": score,
                    "bbox": [bx, by, bx2 - bx, by2 - by],
                })

    if not coco_gt_data["annotations"]:
        print("[EVAL] No GT annotations in val set")
        return {"ap50": 0.0, "f1@85": 0.0, "p@85": 0.0, "r@85": 0.0}

    # Run COCO evaluation
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    coco_gt.createIndex()

    if not coco_results:
        print("[EVAL] No predictions")
        return {"ap50": 0.0, "f1@85": 0.0, "p@85": 0.0, "r@85": 0.0}

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap50 = coco_eval.stats[1]

    # Deployment-threshold F1 — catches V4.2 failure mode where AP50 is high
    # but conf-gated precision collapses
    f1_stats = _compute_f1_at_conf(coco_gt_data, coco_results,
                                   conf_threshold=0.85, iou_threshold=0.5)
    return {
        "ap50": ap50,
        "f1@85": f1_stats["f1"],
        "p@85": f1_stats["p"],
        "r@85": f1_stats["r"],
        "tp@85": f1_stats["tp"],
        "fp@85": f1_stats["fp"],
        "fn@85": f1_stats["fn"],
    }


# ════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    lr_scheduler=None, scaler=None, profiler=None):
    """Train for one epoch, return average loss. Supports AMP via scaler.

    If `profiler` is a StageProfiler, per-stage wall/GPU times are accumulated
    into it (stages: data_wait, to_device, forward, backward, step).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    prof = profiler  # alias
    data_iter = iter(data_loader)
    while True:
        # ── data_wait: time blocked waiting for next batch ────────────
        if prof is not None:
            with prof("data_wait"):
                try:
                    images, targets = next(data_iter)
                except StopIteration:
                    break
        else:
            try:
                images, targets = next(data_iter)
            except StopIteration:
                break

        # ── to_device: H2D transfer ────────────────────────────────────
        # Skip non-tensor target fields (e.g. label_sources is a Python list).
        def _to_dev(t, non_blocking=False):
            out = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device, non_blocking=non_blocking)
                else:
                    out[k] = v
            return out

        if prof is not None:
            with prof("to_device"):
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [_to_dev(t, non_blocking=True) for t in targets]
        else:
            images = [img.to(device) for img in images]
            targets = [_to_dev(t) for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            if prof is not None:
                with prof("forward", cuda=True):
                    with torch.amp.autocast("cuda"):
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                with prof("backward", cuda=True):
                    scaler.scale(losses).backward()
                with prof("step"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            if prof is not None:
                with prof("forward", cuda=True):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                with prof("backward", cuda=True):
                    losses.backward()
                with prof("step"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += losses.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mask R-CNN for solar panel detection")
    parser.add_argument("--coco-dir", default="data/coco", help="COCO dataset directory")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory for selection artifacts (best_model, best_ap50_model, final_model, training_history)")
    parser.add_argument(
        "--scratch-dir", default=None,
        help="If set, heavy resume checkpoints (stageN_epochM.pth with optimizer "
             "state) land here instead of --output-dir. Lets you put rotating epoch "
             "checkpoints on fast local disk (e.g. /root or /dev/shm on RunPod) "
             "while best/final state_dicts persist to slow network volume "
             "(e.g. /workspace). Defaults to --output-dir (existing behavior).",
    )
    parser.add_argument(
        "--pretrained", default="checkpoints/exp003_C_targeted_hn/best_model.pth",
        help="Path to pretrained .pth used as initial weights (default: V3-C best_model.pth)",
    )
    parser.add_argument("--epochs1", type=int, default=3, help="Stage 1 epochs (heads-only)")
    parser.add_argument("--epochs2", type=int, default=20, help="Stage 2 epochs (full fine-tune)")
    parser.add_argument("--lr1", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--lr2", type=float, default=1e-4, help="Stage 2 learning rate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Batches each worker prefetches (PyTorch default 2)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint (.pth) to resume training (auto-detects stage)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Print per-stage wall/GPU timing breakdown after each epoch",
    )
    parser.add_argument(
        "--reinit-mask-head", action="store_true",
        help="After loading --pretrained, reset mask_head + mask_predictor to "
             "fresh weights. Use to wipe halo bias from a halo-iterated checkpoint.",
    )
    parser.add_argument(
        "--reinit-box-predictor", action="store_true",
        help="After loading --pretrained, reset box_predictor (cls_score + "
             "bbox_pred) to fresh weights.",
    )
    parser.add_argument(
        "--jhb-phaseA-spec", default=None,
        help="If set, use JHBRawPartsDataset + boundary-aware mask supervision "
             "(see core.training.boundary_aware_mask). Path to "
             "configs/datasets/jhb_phaseA.yaml. Bypasses --coco-dir.",
    )
    parser.add_argument(
        "--per-source-mask-weight", action="store_true",
        help="Down-weight mask boundary BCE by COCO ann label_source. "
             "Annotator-initiated (human_manual / human_manual_sam_assisted / "
             "sam_added_true_fn) keep weight=1.0; V3-C-derived "
             "(reviewed_prediction / sam_refined_review / "
             "legacy_weak_supervision) drop to 0 on the polygon edge band. "
             "Foreground/background core stays weight=1.0 for all sources, so "
             "pred-vs-bg classification is preserved. Reads label_source from "
             "each annotation in the COCO json. Implements docs/plans/"
             "2026-05-09-training-supervision-layering.md item #4 (R=0 per "
             "literature, sam_refined_review folded into V3-C-derived because "
             "the detection — not just the boundary — comes from V3-C).",
    )
    parser.add_argument(
        "--boundary-band-iters", type=int, default=2,
        help="Boundary-band width for --per-source-mask-weight: dilate XOR "
             "erode with this many 3x3 iterations gives a ~2*iters px ring "
             "around polygon edge.",
    )
    parser.add_argument(
        "--per-instance-mask-trusted", action="store_true",
        help="Per-instance mask-loss gate. Reads ann['mask_trusted'] from COCO "
             "(falls back to label_source mapping). Untrusted instances (R-type "
             "reviewed_prediction, sam_refined_review, sam_added_true_fn, "
             "legacy_weak_supervision) contribute box + cls loss but zero mask "
             "BCE. Trusted instances (human_manual / *_sam_assisted / "
             "sam_added_browser / qgis_geosam) contribute full mask loss. "
             "Implements unified_reviewall plan (docs/plans/"
             "review-aerial-2023-jhb-opengeoai-v3c-parallel-biscuit.md). "
             "Combines with --per-source-mask-weight (per-pixel band weighting "
             "still applies to trusted instances).",
    )
    parser.add_argument(
        "--freeze-mask-head", action="store_true",
        help="In Stage 1, freeze ONLY mask_head + mask_predictor; backbone + "
             "RPN + box_head stay trainable. Replaces the legacy Stage 1 "
             "freeze-backbone semantics. At Stage 2 entry, mask head is "
             "unfrozen and the full model trains with differential LR. "
             "Per docs/plans/2026-05-09-training-supervision-layering.md item "
             "#1 and the unified_reviewall plan.",
    )
    parser.add_argument(
        "--diff-lr-backbone-mult", type=float, default=None,
        help="Stage 2 LR multiplier for backbone params (default: 1.0 = legacy "
             "flat LR). Set to 0.1 for V3-C warm-start.",
    )
    parser.add_argument(
        "--diff-lr-rpn-box-mult", type=float, default=None,
        help="Stage 2 LR multiplier for RPN + box_head params. Default 1.0.",
    )
    parser.add_argument(
        "--diff-lr-mask-mult", type=float, default=None,
        help="Stage 2 LR multiplier for mask_head + mask_predictor. Default 1.0.",
    )
    parser.add_argument(
        "--log-per-source-box-reg-loss", action="store_true",
        help="Bucket box regression loss by GT label_source per epoch (R/S/H "
             "etc). Signal-only — does not alter gradients. If (mean_H - "
             "mean_R) / mean_H > 0.25 persists for >=3 consecutive eval epochs, "
             "writes box_trusted_recommended:true into "
             "<output_dir>/next_round_signal.json. Per the unified_reviewall "
             "plan, this is the next-round box_trusted decision signal.",
    )
    parser.add_argument(
        "--eval-schedule", default=None,
        help='Uneven eval schedule spec like "2:10:2,11:25:3" (epochs 2,4,6,8,10 '
             'then 13,16,19,22,25). If omitted, evaluation runs every epoch. '
             'Schedule is interpreted relative to the absolute epoch number '
             'across BOTH stages (Stage 1 epochs 1..epochs1, Stage 2 epochs '
             '1..epochs2 are mapped to global epoch indices).',
    )
    parser.add_argument(
        "--early-stop-metrics", default=None,
        help="Comma-separated metric names from the eval dict to drive early "
             "stop. OR-stop semantics: any metric's patience exhausted → stop. "
             "Defaults to f1_85,ap50. The plan's pass criteria are area_f1 + "
             "ch2_recall@0.5; these are grid-level and not yet computed in-loop "
             "— for now use chip-level proxies and run scripts/analysis/"
             "area_aggregate_eval.py post-training for final pass-criteria check.",
    )
    parser.add_argument(
        "--early-stop-min-delta", type=float, default=0.005,
        help="Minimum improvement (best-so-far + min_delta) to reset patience.",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="If > 0, enable early stopping with this patience (consecutive "
             "eval epochs without improvement on any --early-stop-metrics).",
    )
    parser.add_argument(
        "--best-ckpt-bulk-range", default=None,
        help='Spec "lo,hi" (e.g. "0.85,1.15"). Recorded in training_history.json '
             "for the post-hoc best-ckpt picker; train.py does NOT currently "
             "compute grid-level bulk_ratio in-loop, so this gate is enforced "
             "by an external selector that reads training_history + grid-level "
             "eval results.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else output_dir
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if scratch_dir != output_dir:
        print(f"[CKPT] scratch_dir={scratch_dir} (rotating epoch ckpts), "
              f"output_dir={output_dir} (best/final selection artifacts)")

    # ── AMP setup ─────────────────────────────────────────────────────
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[AMP] Mixed precision training enabled")

    # ── Resolve pretrained weights ────────────────────────────────────
    pretrained_path = Path(args.pretrained)
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrained_path}. "
            "Pass --pretrained <path-to-.pth> explicitly."
        )
    print(f"[MODEL] Using pretrained weights: {pretrained_path}")

    # ── Datasets ──────────────────────────────────────────────────────
    if args.jhb_phaseA_spec:
        from core.training.boundary_aware_mask import install_patch, stash_batch_supervision
        from core.training.jhb_phaseA_dataset import JHBRawPartsDataset, load_spec
        from core.training.jhb_phaseA_transforms import (
            BoundaryAwareTrainTransforms,
            ValTransforms as _PhaseAValTransforms,
        )

        install_patch()
        print(f"[PATCH] boundary-aware maskrcnn_loss installed")

        spec = load_spec(args.jhb_phaseA_spec)
        train_ds = JHBRawPartsDataset(
            spec, "train", transforms=BoundaryAwareTrainTransforms(args.chip_size)
        )
        val_ds = JHBRawPartsDataset(
            spec, "val", transforms=_PhaseAValTransforms()
        )
        # Stash supervision tensors before each forward (only when training).
        _stash_fn = stash_batch_supervision

        def _pre_forward_hook(module, args_in):
            if module.training and len(args_in) >= 2 and args_in[1] is not None:
                _stash_fn(args_in[1])
    else:
        train_ds = CocoSolarDataset(
            coco_dir / "train.json", coco_dir, transforms=TrainTransforms(args.chip_size),
            per_source_mask_weight=args.per_source_mask_weight,
            boundary_band_iters=args.boundary_band_iters,
            per_instance_mask_trusted=args.per_instance_mask_trusted,
        )
        val_ds = CocoSolarDataset(
            coco_dir / "val.json", coco_dir, transforms=ValTransforms(),
            per_source_mask_weight=args.per_source_mask_weight,
            boundary_band_iters=args.boundary_band_iters,
            per_instance_mask_trusted=args.per_instance_mask_trusted,
        )
        _install_aux_resize = (
            args.per_source_mask_weight or args.per_instance_mask_trusted
        )
        if _install_aux_resize:
            from core.training.boundary_aware_mask import install_patch
            install_patch()
            modes = []
            if args.per_source_mask_weight:
                modes.append("per-source-mask-weight")
            if args.per_instance_mask_trusted:
                modes.append("per-instance-mask-trusted")
            print(f"[PATCH] boundary-aware maskrcnn_loss installed "
                  f"({'+'.join(modes)})")
    print(f"[DATA] Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(pretrained_path, num_classes=2)
    if args.reinit_mask_head:
        reinit_mask_head(model)
    if args.reinit_box_predictor:
        reinit_box_predictor(model)
    model.to(device)

    # Boundary-aware supervision: wrap model.transform so per-image aux fields
    # (mask_pixel_weights, ignore_masks) are resized to match torchvision's
    # post-transform mask shape before stashing — fixes the spatial mismatch
    # between pre-transform 400-space weights and post-transform 800-space
    # proposals used by patched mask loss.
    if _install_aux_resize:
        from core.training.boundary_aware_mask import install_transform_aux_resize
        install_transform_aux_resize(model)
        print("[HOOK] model.transform wrapped for post-transform aux resize + stash")

    # ── Per-source box reg loss: install fastrcnn patch + wrap RoI sampler ──
    if args.log_per_source_box_reg_loss:
        from core.training.boundary_aware_mask import (
            install_patch as _install_mask_patch,
            install_fastrcnn_patch,
            install_transform_aux_resize as _install_transform_box,
            wrap_select_training_samples,
        )
        _install_mask_patch()    # idempotent; ensures stash hook also runs
        install_fastrcnn_patch()
        wrap_select_training_samples(model)
        # Reuse the same transform wrapper for box-loss bucketing. The wrapper
        # also handles non-mask cases gracefully (no-op when aux fields absent),
        # so it's safe even when neither per_source_mask_weight nor
        # per_instance_mask_trusted is set.
        if not _install_aux_resize:
            _install_transform_box(model)
            print("[HOOK] model.transform wrapped (box-loss bucketing path)")
        print("[PATCH] fastrcnn_loss patched + select_training_samples wrapped "
              "(per-source box reg loss bucketing ON)")

    # Trusted source classification mirrors _LABEL_SOURCE_TO_MASK_TRUSTED.
    _TRUSTED_SRC = {k for k, v in _LABEL_SOURCE_TO_MASK_TRUSTED.items() if v}
    _UNTRUSTED_SRC = {k for k, v in _LABEL_SOURCE_TO_MASK_TRUSTED.items() if not v}

    def _summarize_box_loss(epoch_idx: int, into_history: dict) -> None:
        """Read boundary_aware_mask._BOX_LOSS_BUCKETS, group into H / R / S
        (sam_added_true_fn) and write into into_history + log."""
        from core.training.boundary_aware_mask import (
            box_loss_bucket_means, reset_box_loss_buckets,
        )
        per_src = box_loss_bucket_means()
        if not per_src:
            return
        # Group
        h_vals = [per_src[s] for s in _TRUSTED_SRC if s in per_src]
        r_vals = [per_src[s] for s in _UNTRUSTED_SRC
                  if s in per_src and s != "sam_added_true_fn"]
        s_only = per_src.get("sam_added_true_fn")

        mean_H = float(sum(h_vals) / len(h_vals)) if h_vals else None
        mean_R = float(sum(r_vals) / len(r_vals)) if r_vals else None
        gap_HR = None
        if mean_H and mean_R and mean_H > 0:
            gap_HR = (mean_H - mean_R) / mean_H

        msg = (f"  [BOX-LOSS-PER-SOURCE] H={mean_H} R={mean_R} "
               f"sam_added_true_fn={s_only} gap_HR={gap_HR}")
        print(msg)
        into_history["box_loss_per_source"] = per_src
        into_history["box_loss_mean_H"] = mean_H
        into_history["box_loss_mean_R"] = mean_R
        into_history["box_loss_gap_HR"] = gap_HR
        reset_box_loss_buckets()

    best_ap50 = 0.0
    best_f1 = 0.0
    best_path = output_dir / "best_model.pth"           # selected by F1@0.85 (deployment metric)
    best_ap50_path = output_dir / "best_ap50_model.pth" # secondary, for comparison
    history = []
    box_gap_streak = 0  # consecutive eval epochs with gap_HR > 0.25

    # ── Early stopping + uneven eval schedule ─────────────────────────
    def _parse_eval_schedule(spec: str | None) -> set | None:
        if spec is None:
            return None
        epochs = set()
        for chunk in spec.split(","):
            parts = chunk.strip().split(":")
            if len(parts) != 3:
                raise ValueError(f"invalid eval-schedule chunk {chunk!r}; expected start:end:step")
            s, e, st = int(parts[0]), int(parts[1]), int(parts[2])
            for ep in range(s, e + 1, st):
                epochs.add(ep)
        return epochs

    eval_epochs_global = _parse_eval_schedule(args.eval_schedule)
    if eval_epochs_global is not None:
        print(f"[EVAL] uneven schedule: {sorted(eval_epochs_global)}")

    class EarlyStop:
        """Best-so-far + patience, OR-stop on any of multiple metrics."""
        def __init__(self, metric_names, min_delta=0.005, patience=3):
            self.names = metric_names
            self.min_delta = min_delta
            self.patience = patience
            self.best = {m: -float("inf") for m in metric_names}
            self.counters = {m: 0 for m in metric_names}

        def step(self, metrics):
            for m in self.names:
                v = metrics.get(m)
                if v is None:
                    continue
                if v > self.best[m] + self.min_delta:
                    self.best[m] = v
                    self.counters[m] = 0
                else:
                    self.counters[m] += 1
            return any(c >= self.patience for c in self.counters.values())

        def state(self):
            return {
                "best": {k: float(v) for k, v in self.best.items()},
                "counters": dict(self.counters),
            }

    early_stop_enabled = args.early_stop_patience > 0
    es_metrics = (args.early_stop_metrics or "f1_85,ap50").split(",")
    es_metrics = [m.strip() for m in es_metrics if m.strip()]
    early_stop = EarlyStop(
        es_metrics, args.early_stop_min_delta, args.early_stop_patience
    ) if early_stop_enabled else None
    if early_stop_enabled:
        print(f"[EARLY-STOP] metrics={es_metrics} min_delta={args.early_stop_min_delta} "
              f"patience={args.early_stop_patience} (OR-stop semantics)")

    def _metrics_for_earlystop(eval_metrics):
        """Map evaluate_coco output keys → early-stop metric names.

        evaluate_coco returns {'ap50', 'f1@85', 'p@85', 'r@85', ...}; alias
        them so users can write --early-stop-metrics f1_85,ap50.
        """
        if eval_metrics is None:
            return {}
        return {
            "ap50": eval_metrics.get("ap50"),
            "f1_85": eval_metrics.get("f1@85"),
            "p_85": eval_metrics.get("p@85"),
            "r_85": eval_metrics.get("r@85"),
            # Grid-level placeholders for forward compat with plan literal names:
            "area_f1": eval_metrics.get("area_f1"),
            "ch2_recall": eval_metrics.get("ch2_recall"),
        }

    bulk_range_lo, bulk_range_hi = None, None
    if args.best_ckpt_bulk_range:
        parts = args.best_ckpt_bulk_range.split(",")
        bulk_range_lo, bulk_range_hi = float(parts[0]), float(parts[1])
        print(f"[BEST-CKPT] bulk_ratio gate recorded for post-hoc selection: "
              f"[{bulk_range_lo:.3f}, {bulk_range_hi:.3f}] (not enforced in-loop)")

    def _should_eval_this_epoch(stage: int, epoch_in_stage: int) -> bool:
        """Map (stage, epoch_in_stage) → global epoch index, compare to schedule."""
        if eval_epochs_global is None:
            return True
        if stage == 1:
            global_ep = epoch_in_stage
        else:
            global_ep = args.epochs1 + epoch_in_stage
        return global_ep in eval_epochs_global

    # ── Resume handling (auto-detect stage) ───────────────────────────
    resume_stage = 0
    resume_epoch = 0
    ckpt = None
    if args.resume:
        print(f"\n[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        resume_stage = ckpt.get("stage", 2)
        resume_epoch = ckpt["epoch"]
        best_ap50 = ckpt.get("best_ap50", best_ap50)
        best_f1 = ckpt.get("best_f1", best_f1)
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[RESUME] Stage {resume_stage}, epoch {resume_epoch}, "
              f"best_ap50={best_ap50:.4f}, best_f1={best_f1:.4f}")

    def save_checkpoint(stage, epoch, model, optimizer, best_ap50, best_f1, scaler=None):
        """Save checkpoint with all state needed for resume."""
        ckpt_path = scratch_dir / f"stage{stage}_epoch{epoch}.pth"
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "best_ap50": best_ap50,
            "best_f1": best_f1,
        }
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        torch.save(state, ckpt_path)
        # Keep only last 2 checkpoints per stage to save disk
        old = sorted(scratch_dir.glob(f"stage{stage}_epoch*.pth"))
        for f in old[:-2]:
            f.unlink()
        return ckpt_path

    # ── Stage 1: Heads-only ───────────────────────────────────────────
    skip_stage1 = (args.resume and resume_stage >= 2)
    stage1_start = resume_epoch if (args.resume and resume_stage == 1) else 0

    if not skip_stage1:
        print("\n" + "=" * 60)
        if args.freeze_mask_head:
            print(f"Stage 1: freeze-mask-head warm-up ({args.epochs1} epochs, LR={args.lr1})")
        else:
            print(f"Stage 1: Heads-only training ({args.epochs1} epochs, LR={args.lr1})")
        print("=" * 60)

        if args.freeze_mask_head:
            # unified_reviewall semantics: mask_head frozen, everything else trainable
            freeze_mask_head(model)
        else:
            # legacy semantics: backbone+FPN frozen, all heads (incl. mask) trainable
            freeze_backbone(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer1 = torch.optim.SGD(
            trainable_params, lr=args.lr1, momentum=0.9, weight_decay=1e-4
        )
        if args.resume and resume_stage == 1 and ckpt and "optimizer" in ckpt:
            optimizer1.load_state_dict(ckpt["optimizer"])
            print(f"[RESUME] Stage 1 optimizer restored, continuing from epoch {stage1_start + 1}")

        stage1_stopped_early = False
        for epoch in range(stage1_start + 1, args.epochs1 + 1):
            t0 = time.time()
            epoch_prof = StageProfiler(cuda=True) if args.profile else None
            avg_loss = train_one_epoch(
                model, optimizer1, train_loader, device, epoch,
                scaler=scaler, profiler=epoch_prof,
            )
            dt = time.time() - t0
            if epoch_prof is not None:
                print(epoch_prof.summary(header=f"stage1 epoch {epoch}"))

            do_eval = _should_eval_this_epoch(1, epoch)
            hist_entry = {
                "stage": 1, "epoch": epoch, "loss": avg_loss,
            }
            metrics = None
            if do_eval:
                metrics = evaluate_coco(model, val_loader, device)
                ap50 = metrics["ap50"]
                f1_85 = metrics["f1@85"]
                print(f"  Epoch {epoch}/{args.epochs1}  loss={avg_loss:.4f}  "
                      f"val_AP50={ap50:.4f}  val_F1@85={f1_85:.4f} "
                      f"(P={metrics['p@85']:.3f} R={metrics['r@85']:.3f} "
                      f"TP={metrics['tp@85']} FP={metrics['fp@85']} FN={metrics['fn@85']})  "
                      f"time={dt:.0f}s")
                hist_entry.update({
                    "val_ap50": ap50, "val_f1_85": f1_85,
                    "val_p_85": metrics["p@85"], "val_r_85": metrics["r@85"],
                })
            else:
                print(f"  Epoch {epoch}/{args.epochs1}  loss={avg_loss:.4f}  "
                      f"[eval skipped by --eval-schedule]  time={dt:.0f}s")

            if args.log_per_source_box_reg_loss:
                _summarize_box_loss(epoch, hist_entry)
                gap_HR = hist_entry.get("box_loss_gap_HR")
                if do_eval:
                    if gap_HR is not None and gap_HR > 0.25:
                        box_gap_streak += 1
                    else:
                        box_gap_streak = 0
                    if box_gap_streak >= 3:
                        sig_path = output_dir / "next_round_signal.json"
                        sig_path.write_text(json.dumps({
                            "box_trusted_recommended": True,
                            "reason": f"gap_HR > 0.25 for {box_gap_streak} consecutive eval epochs",
                            "last_epoch": f"stage1_e{epoch}",
                            "last_gap_HR": gap_HR,
                        }, indent=2) + "\n")
                        print(f"  >> [SIGNAL] gap_HR > 0.25 streak={box_gap_streak}, "
                              f"wrote {sig_path}")
            history.append(hist_entry)

            if do_eval:
                if f1_85 > best_f1:
                    best_f1 = f1_85
                    torch.save(model.state_dict(), best_path)
                    print(f"  >> New best F1@85={f1_85:.4f}, saved to {best_path}")
                if ap50 > best_ap50:
                    best_ap50 = ap50
                    torch.save(model.state_dict(), best_ap50_path)

                if early_stop is not None:
                    es_input = _metrics_for_earlystop(metrics)
                    if early_stop.step(es_input):
                        print(f"  >> [EARLY-STOP] triggered at stage1 epoch {epoch}; "
                              f"state={early_stop.state()}")
                        save_checkpoint(1, epoch, model, optimizer1, best_ap50, best_f1, scaler)
                        stage1_stopped_early = True
                        break

            save_checkpoint(1, epoch, model, optimizer1, best_ap50, best_f1, scaler)

        if stage1_stopped_early:
            print("[STAGE-1] early-stopped; proceeding to Stage 2")

    # ── Stage 2: Full fine-tune ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Stage 2: Full fine-tune ({args.epochs2} epochs, LR={args.lr2}, cosine decay)")
    print("=" * 60)

    stage2_start = resume_epoch if (args.resume and resume_stage == 2) else 0

    unfreeze_all(model)

    diff_lr_on = any(
        m is not None
        for m in (args.diff_lr_backbone_mult, args.diff_lr_rpn_box_mult, args.diff_lr_mask_mult)
    )
    if diff_lr_on:
        bb_mult = args.diff_lr_backbone_mult if args.diff_lr_backbone_mult is not None else 1.0
        rb_mult = args.diff_lr_rpn_box_mult if args.diff_lr_rpn_box_mult is not None else 1.0
        mk_mult = args.diff_lr_mask_mult if args.diff_lr_mask_mult is not None else 1.0

        bb_params, rb_params, mk_params, other_params = [], [], [], []
        bb_total = rb_total = mk_total = oth_total = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                bb_params.append(p); bb_total += p.numel()
            elif "mask_head" in name or "mask_predictor" in name:
                mk_params.append(p); mk_total += p.numel()
            elif name.startswith("rpn.") or "box_predictor" in name or "box_head" in name:
                rb_params.append(p); rb_total += p.numel()
            else:
                other_params.append(p); oth_total += p.numel()

        param_groups = []
        if bb_params:
            param_groups.append({"params": bb_params, "lr": args.lr2 * bb_mult, "name": "backbone"})
        if rb_params:
            param_groups.append({"params": rb_params, "lr": args.lr2 * rb_mult, "name": "rpn_box"})
        if mk_params:
            param_groups.append({"params": mk_params, "lr": args.lr2 * mk_mult, "name": "mask"})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr2, "name": "other"})

        # sanity check: per-group param count vs total
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sum_groups = bb_total + rb_total + mk_total + oth_total
        print(f"[DIFF-LR] param groups (lr2={args.lr2:.2e}):")
        print(f"          backbone:  {bb_total:>10,d} params  lr={args.lr2 * bb_mult:.2e}  (×{bb_mult})")
        print(f"          rpn_box:   {rb_total:>10,d} params  lr={args.lr2 * rb_mult:.2e}  (×{rb_mult})")
        print(f"          mask:      {mk_total:>10,d} params  lr={args.lr2 * mk_mult:.2e}  (×{mk_mult})")
        print(f"          other:     {oth_total:>10,d} params  lr={args.lr2:.2e}  (×1.0)")
        print(f"          sum:       {sum_groups:>10,d}  vs trainable total {total_trainable:,d}")
        assert sum_groups == total_trainable, (
            f"diff-LR param group sum {sum_groups} != trainable total {total_trainable}; "
            f"name-pattern split missed parameters"
        )
        optimizer2 = torch.optim.SGD(
            param_groups, momentum=0.9, weight_decay=1e-4
        )
    else:
        optimizer2 = torch.optim.SGD(
            model.parameters(), lr=args.lr2, momentum=0.9, weight_decay=1e-4
        )
    total_steps = args.epochs2 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=total_steps, eta_min=1e-6
    )

    if args.resume and resume_stage == 2 and ckpt and "optimizer" in ckpt:
        optimizer2.load_state_dict(ckpt["optimizer"])
        for _ in range(stage2_start * len(train_loader)):
            scheduler.step()
        print(f"[RESUME] Stage 2 optimizer restored, scheduler advanced to epoch {stage2_start}")

    stage2_stopped_early = False
    for epoch in range(stage2_start + 1, args.epochs2 + 1):
        t0 = time.time()
        epoch_prof = StageProfiler(cuda=True) if args.profile else None
        avg_loss = train_one_epoch(
            model, optimizer2, train_loader, device, epoch,
            lr_scheduler=scheduler, scaler=scaler, profiler=epoch_prof,
        )
        dt = time.time() - t0
        if epoch_prof is not None:
            print(epoch_prof.summary(header=f"stage2 epoch {epoch}"))

        current_lr = optimizer2.param_groups[0]["lr"]
        do_eval = _should_eval_this_epoch(2, epoch)
        hist_entry = {
            "stage": 2, "epoch": epoch, "loss": avg_loss, "lr": current_lr,
        }
        metrics = None
        if do_eval:
            metrics = evaluate_coco(model, val_loader, device)
            ap50 = metrics["ap50"]
            f1_85 = metrics["f1@85"]
            print(f"  Epoch {epoch}/{args.epochs2}  loss={avg_loss:.4f}  "
                  f"val_AP50={ap50:.4f}  val_F1@85={f1_85:.4f} "
                  f"(P={metrics['p@85']:.3f} R={metrics['r@85']:.3f} "
                  f"TP={metrics['tp@85']} FP={metrics['fp@85']} FN={metrics['fn@85']})  "
                  f"lr={current_lr:.2e}  time={dt:.0f}s")
            hist_entry.update({
                "val_ap50": ap50, "val_f1_85": f1_85,
                "val_p_85": metrics["p@85"], "val_r_85": metrics["r@85"],
            })
        else:
            print(f"  Epoch {epoch}/{args.epochs2}  loss={avg_loss:.4f}  "
                  f"[eval skipped by --eval-schedule]  lr={current_lr:.2e}  time={dt:.0f}s")

        if args.log_per_source_box_reg_loss:
            _summarize_box_loss(epoch, hist_entry)
            gap_HR = hist_entry.get("box_loss_gap_HR")
            if do_eval:
                if gap_HR is not None and gap_HR > 0.25:
                    box_gap_streak += 1
                else:
                    box_gap_streak = 0
                if box_gap_streak >= 3:
                    sig_path = output_dir / "next_round_signal.json"
                    sig_path.write_text(json.dumps({
                        "box_trusted_recommended": True,
                        "reason": f"gap_HR > 0.25 for {box_gap_streak} consecutive eval epochs",
                        "last_epoch": f"stage2_e{epoch}",
                        "last_gap_HR": gap_HR,
                    }, indent=2) + "\n")
                    print(f"  >> [SIGNAL] gap_HR > 0.25 streak={box_gap_streak}, "
                          f"wrote {sig_path}")
        history.append(hist_entry)

        if do_eval:
            if f1_85 > best_f1:
                best_f1 = f1_85
                torch.save(model.state_dict(), best_path)
                print(f"  >> New best F1@85={f1_85:.4f}, saved to {best_path}")
            if ap50 > best_ap50:
                best_ap50 = ap50
                torch.save(model.state_dict(), best_ap50_path)

            if early_stop is not None:
                es_input = _metrics_for_earlystop(metrics)
                if early_stop.step(es_input):
                    print(f"  >> [EARLY-STOP] triggered at stage2 epoch {epoch}; "
                          f"state={early_stop.state()}")
                    save_checkpoint(2, epoch, model, optimizer2, best_ap50, best_f1, scaler)
                    stage2_stopped_early = True
                    break

        save_checkpoint(2, epoch, model, optimizer2, best_ap50, best_f1, scaler)

    if stage2_stopped_early:
        print("[STAGE-2] early-stopped")

    # ── Save final outputs ────────────────────────────────────────────
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)

    history_path = output_dir / "training_history.json"
    history_payload = {
        "history": history,
        "best_ap50": best_ap50,
        "best_f1": best_f1,
        "early_stop": early_stop.state() if early_stop is not None else None,
        "best_ckpt_bulk_range": (
            [bulk_range_lo, bulk_range_hi]
            if bulk_range_lo is not None else None
        ),
        "eval_schedule": args.eval_schedule,
        "diff_lr": {
            "backbone_mult": args.diff_lr_backbone_mult,
            "rpn_box_mult": args.diff_lr_rpn_box_mult,
            "mask_mult": args.diff_lr_mask_mult,
        },
        "per_instance_mask_trusted": args.per_instance_mask_trusted,
        "per_source_mask_weight": args.per_source_mask_weight,
        "freeze_mask_head": args.freeze_mask_head,
        "log_per_source_box_reg_loss": args.log_per_source_box_reg_loss,
    }
    history_path.write_text(json.dumps(history_payload, indent=2) + "\n")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best val AP50: {best_ap50:.4f}")
    print(f"  Best model:    {best_path}")
    print(f"  Final model:   {final_path}")
    print(f"  History:       {history_path}")
    print()
    print("To run inference with the fine-tuned model:")
    print(f"  python detect_and_evaluate.py --model-path {best_path}")


if __name__ == "__main__":
    main()
