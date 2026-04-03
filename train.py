"""
Fine-tune geoai's Mask R-CNN (ResNet50-FPN) on Cape Town solar panel annotations.

Two-stage training:
  Stage 1 — Heads-only:  ROI heads + mask head, backbone frozen, 3 epochs, LR=1e-3
  Stage 2 — Full fine-tune: all parameters, up to 20 epochs, LR=1e-4, cosine decay

Checkpoint selection: best validation segm_AP50.
Output: .pth file compatible with geoai SolarPanelDetector(model_path=...).

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
from torchvision.models.detection import maskrcnn_resnet50_fpn

assert torch.cuda.is_available(), (
    "CUDA is required for training. WSL2 does not currently expose CUDA to PyTorch. "
    "Run this script on a CUDA-capable machine."
)


# ════════════════════════════════════════════════════════════════════════
# COCO Dataset
# ════════════════════════════════════════════════════════════════════════
class CocoSolarDataset(torch.utils.data.Dataset):
    """COCO instance segmentation dataset for solar panel chips."""

    def __init__(self, coco_json: Path, root_dir: Path, transforms=None):
        with open(coco_json, "r") as f:
            data = json.load(f)

        self.root_dir = root_dir
        self.transforms = transforms

        self.images = {img["id"]: img for img in data["images"]}
        self.image_ids = [img["id"] for img in data["images"]]

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in data["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

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

        h, w = img_info["height"], img_info["width"]

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
            areas.append(ann["area"])

            # Decode segmentation to mask
            mask = np.zeros((h, w), dtype=np.uint8)
            for seg in ann["segmentation"]:
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                import cv2
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

        if len(boxes) == 0:
            # Empty image — return empty targets
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            }

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

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = image.flip(-1)  # flip W
            if target["boxes"].numel() > 0:
                w = image.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
                target["masks"] = target["masks"].flip(-1)

        # Random vertical flip
        if torch.rand(1) < 0.5:
            image = image.flip(-2)  # flip H
            if target["boxes"].numel() > 0:
                h = image.shape[-2]
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
                target["masks"] = target["masks"].flip(-2)

        # Random 90° rotation (0, 90, 180, 270)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, [-2, -1])
            if target["boxes"].numel() > 0:
                masks = target["masks"]
                masks = torch.rot90(masks, k, [-2, -1])
                target["masks"] = masks
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

                # Filter out empty masks
                valid = target["masks"].sum(dim=(-2, -1)) > 0
                if valid.any():
                    target["boxes"] = target["boxes"][valid]
                    target["labels"] = target["labels"][valid]
                    target["masks"] = target["masks"][valid]
                    target["area"] = target["area"][valid] if valid.sum() <= len(target["area"]) else target["boxes"][:, 2:].prod(dim=1)
                else:
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                    target["labels"] = torch.zeros(0, dtype=torch.int64)
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
                    target["area"] = torch.zeros(0, dtype=torch.float32)

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
    """Build Mask R-CNN ResNet50-FPN, optionally loading geoai pretrained weights."""
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    model = maskrcnn_resnet50_fpn(
        weights=None,
        progress=False,
        num_classes=num_classes,
        weights_backbone=None,
        image_mean=image_mean,
        image_std=image_std,
    )

    if pretrained_path is not None:
        print(f"[MODEL] Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        # Remove 'module.' prefix if present (DataParallel)
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)
        print("[MODEL] Pretrained weights loaded")

    return model


def freeze_backbone(model):
    """Freeze backbone + FPN, keep ROI heads + mask head trainable."""
    for name, param in model.named_parameters():
        if name.startswith("backbone") or name.startswith("fpn"):
            param.requires_grad = False
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total = sum(1 for p in model.parameters())
    print(f"[FREEZE] {n_frozen}/{n_total} parameters frozen")


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    print("[UNFREEZE] All parameters trainable")


# ════════════════════════════════════════════════════════════════════════
# Evaluation (COCO AP)
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_coco(model, data_loader, device):
    """Compute COCO segm AP50 on the validation set."""
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
        return 0.0

    # Run COCO evaluation
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    coco_gt.createIndex()

    if not coco_results:
        print("[EVAL] No predictions")
        return 0.0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # AP50 is index 1 in stats
    ap50 = coco_eval.stats[1]
    return ap50


# ════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    lr_scheduler=None, scaler=None):
    """Train for one epoch, return average loss. Supports AMP via scaler."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
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
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument(
        "--pretrained", default=None,
        help="Path to geoai pretrained .pth (default: auto-download from HuggingFace)",
    )
    parser.add_argument("--epochs1", type=int, default=3, help="Stage 1 epochs (heads-only)")
    parser.add_argument("--epochs2", type=int, default=20, help="Stage 2 epochs (full fine-tune)")
    parser.add_argument("--lr1", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--lr2", type=float, default=1e-4, help="Stage 2 learning rate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint (.pth) to resume training (auto-detects stage)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── AMP setup ─────────────────────────────────────────────────────
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[AMP] Mixed precision training enabled")

    # ── Resolve pretrained weights ────────────────────────────────────
    pretrained_path = args.pretrained
    if pretrained_path is None:
        from huggingface_hub import hf_hub_download
        pretrained_path = hf_hub_download(
            repo_id="giswqs/geoai",
            filename="solar_panel_detection.pth",
        )
        print(f"[MODEL] Using geoai weights: {pretrained_path}")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = CocoSolarDataset(
        coco_dir / "train.json", coco_dir, transforms=TrainTransforms(args.chip_size)
    )
    val_ds = CocoSolarDataset(
        coco_dir / "val.json", coco_dir, transforms=ValTransforms()
    )
    print(f"[DATA] Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(pretrained_path, num_classes=2)
    model.to(device)

    best_ap50 = 0.0
    best_path = output_dir / "best_model.pth"
    history = []

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
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[RESUME] Stage {resume_stage}, epoch {resume_epoch}, best_ap50={best_ap50:.4f}")

    def save_checkpoint(stage, epoch, model, optimizer, best_ap50, scaler=None):
        """Save checkpoint with all state needed for resume."""
        ckpt_path = output_dir / f"stage{stage}_epoch{epoch}.pth"
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "best_ap50": best_ap50,
        }
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        torch.save(state, ckpt_path)
        # Keep only last 2 checkpoints per stage to save disk
        old = sorted(output_dir.glob(f"stage{stage}_epoch*.pth"))
        for f in old[:-2]:
            f.unlink()
        return ckpt_path

    # ── Stage 1: Heads-only ───────────────────────────────────────────
    skip_stage1 = (args.resume and resume_stage >= 2)
    stage1_start = resume_epoch if (args.resume and resume_stage == 1) else 0

    if not skip_stage1:
        print("\n" + "=" * 60)
        print(f"Stage 1: Heads-only training ({args.epochs1} epochs, LR={args.lr1})")
        print("=" * 60)

        freeze_backbone(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer1 = torch.optim.SGD(
            trainable_params, lr=args.lr1, momentum=0.9, weight_decay=1e-4
        )
        if args.resume and resume_stage == 1 and ckpt and "optimizer" in ckpt:
            optimizer1.load_state_dict(ckpt["optimizer"])
            print(f"[RESUME] Stage 1 optimizer restored, continuing from epoch {stage1_start + 1}")

        for epoch in range(stage1_start + 1, args.epochs1 + 1):
            t0 = time.time()
            avg_loss = train_one_epoch(
                model, optimizer1, train_loader, device, epoch, scaler=scaler
            )
            dt = time.time() - t0

            ap50 = evaluate_coco(model, val_loader, device)
            print(f"  Epoch {epoch}/{args.epochs1}  loss={avg_loss:.4f}  "
                  f"val_AP50={ap50:.4f}  time={dt:.0f}s")

            history.append({
                "stage": 1, "epoch": epoch, "loss": avg_loss, "val_ap50": ap50
            })

            if ap50 > best_ap50:
                best_ap50 = ap50
                torch.save(model.state_dict(), best_path)
                print(f"  >> New best AP50={ap50:.4f}, saved to {best_path}")

            save_checkpoint(1, epoch, model, optimizer1, best_ap50, scaler)

    # ── Stage 2: Full fine-tune ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Stage 2: Full fine-tune ({args.epochs2} epochs, LR={args.lr2}, cosine decay)")
    print("=" * 60)

    stage2_start = resume_epoch if (args.resume and resume_stage == 2) else 0

    unfreeze_all(model)
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

    for epoch in range(stage2_start + 1, args.epochs2 + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, optimizer2, train_loader, device, epoch,
            lr_scheduler=scheduler, scaler=scaler,
        )
        dt = time.time() - t0

        ap50 = evaluate_coco(model, val_loader, device)
        current_lr = optimizer2.param_groups[0]["lr"]
        print(f"  Epoch {epoch}/{args.epochs2}  loss={avg_loss:.4f}  "
              f"val_AP50={ap50:.4f}  lr={current_lr:.2e}  time={dt:.0f}s")

        history.append({
            "stage": 2, "epoch": epoch, "loss": avg_loss, "val_ap50": ap50,
            "lr": current_lr,
        })

        if ap50 > best_ap50:
            best_ap50 = ap50
            torch.save(model.state_dict(), best_path)
            print(f"  >> New best AP50={ap50:.4f}, saved to {best_path}")

        save_checkpoint(2, epoch, model, optimizer2, best_ap50, scaler)

    # ── Save final outputs ────────────────────────────────────────────
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n")

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
