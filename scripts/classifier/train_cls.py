"""
Train PV vs non-PV binary classifier (EfficientNet-B0).

Two-stage training:
  Stage 1: backbone frozen, train classifier head only (5 epochs)
  Stage 2: full fine-tune with cosine annealing (25 epochs)

Usage:
    python scripts/classifier/train_cls.py \
        --data-dir data/cls_pv_thermal \
        --output-dir checkpoints/cls_pv_thermal

    # Resume from checkpoint
    python scripts/classifier/train_cls.py \
        --data-dir data/cls_pv_thermal \
        --output-dir checkpoints/cls_pv_thermal \
        --resume checkpoints/cls_pv_thermal/last_cls.pth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

NUM_CLASSES = 2
CLASS_NAMES = ["non_pv", "pv"]

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


SUPPORTED_ARCHS = ("efficientnet_b0", "resnet18", "convnext_tiny", "dinov2_vits14")


class DinoV2Classifier(nn.Module):
    """DINOv2 ViT backbone + linear classification head on CLS token.

    Uses `timm` to load pretrained DINOv2 features. The CLS token output
    (768 dims for ViT-S/14 at img_size=224 → patches=16×16) is projected to
    `num_classes` by a linear head. Keeps things simple so frozen-warmup and
    full-FT both behave identically to the CNN branches.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        num_classes: int,
        img_size: int = 224,
    ):
        super().__init__()
        import timm  # imported lazily to keep optional
        # DINOv2 ViT defaults to img_size=518 in timm; override to match our
        # 224 crop policy so positional embeddings are re-interpolated.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # return features (CLS token)
            img_size=img_size,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


DINOV2_TIMM_NAMES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
}


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    """Build classifier model with replaced final layer."""
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        # convnext.classifier is Sequential: [LayerNorm2d, Flatten, Linear]
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    elif arch in DINOV2_TIMM_NAMES:
        model = DinoV2Classifier(
            DINOV2_TIMM_NAMES[arch],
            pretrained=pretrained,
            num_classes=NUM_CLASSES,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Supported: {SUPPORTED_ARCHS}")
    return model


AUG_PROFILES = ("current", "flip_only")


def get_transforms(
    img_size: int = 224,
    is_train: bool = True,
    aug_profile: str = "current",
) -> transforms.Compose:
    """Get data augmentation transforms.

    `aug_profile`:
      - `current`: HFlip + VFlip + Rotate90 + ColorJitter (existing baseline)
      - `flip_only`: HFlip only + ColorJitter (removes rotation priors for
        nadir orientation hypothesis; see exp_cls_augmentation_ablation.md)
    """
    if not is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    if aug_profile not in AUG_PROFILES:
        raise ValueError(f"Unknown aug_profile '{aug_profile}'. Supported: {AUG_PROFILES}")

    augs: list = [
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if aug_profile == "current":
        augs += [
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        ]
    augs += [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(augs)


def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Create weighted sampler for class balance."""
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Freeze all parameters except the classification head."""
    if arch in ("efficientnet_b0", "convnext_tiny"):
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    elif arch == "resnet18":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    elif arch in DINOV2_TIMM_NAMES:
        # DinoV2Classifier exposes `backbone` + `head`; freeze everything
        # except the linear head.
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
    else:
        raise ValueError(f"freeze_backbone: unsupported arch '{arch}'")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Train for one epoch, return metrics."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Validate, return metrics including balanced accuracy and per-class stats."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total = len(all_labels)

    # Per-class metrics
    per_class = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = all_labels == cls_idx
        pred_mask = all_preds == cls_idx
        tp = ((all_preds == cls_idx) & (all_labels == cls_idx)).sum()
        fp = ((all_preds == cls_idx) & (all_labels != cls_idx)).sum()
        fn = ((all_preds != cls_idx) & (all_labels == cls_idx)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[cls_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "count": int(mask.sum()),
        }

    # Balanced accuracy
    recalls = [per_class[c]["recall"] for c in CLASS_NAMES]
    balanced_acc = float(np.mean(recalls))

    return {
        "loss": running_loss / total,
        "accuracy": float((all_preds == all_labels).sum() / total),
        "balanced_accuracy": balanced_acc,
        "per_class": per_class,
    }


def _load_config_file(path: Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    # Strip documentation-only fields
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def main():
    parser = argparse.ArgumentParser(description="Train PV vs non-PV classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/cls_pv_thermal"))
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON config that pre-sets other args (CLI flags override)",
    )
    parser.add_argument("--arch", default="efficientnet_b0", choices=SUPPORTED_ARCHS)
    parser.add_argument(
        "--training-mode", default="full_ft",
        choices=["full_ft", "linear_probe"],
        help="full_ft: frozen warmup → unfreeze (default). linear_probe: keep backbone frozen throughout.",
    )
    parser.add_argument(
        "--aug-profile", default="current", choices=AUG_PROFILES,
        help="Augmentation profile (see exp_cls_augmentation_ablation.md)",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Stage 1: epochs with frozen backbone")
    parser.add_argument("--finetune-epochs", type=int, default=25,
                        help="Stage 2: epochs with full fine-tune (ignored in linear_probe mode)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Stage 1 learning rate")
    parser.add_argument("--lr-finetune", type=float, default=1e-4,
                        help="Stage 2 learning rate")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Config file overrides defaults, CLI flags override config.
    if args.config and args.config.exists():
        cfg = _load_config_file(args.config)
        print(f"[config] Loaded defaults from {args.config}: {cfg}")
        # Map config fields onto args only if they weren't explicitly set.
        # (We rely on argparse defaults matching; user-provided CLI flags
        # won't be overwritten because argparse already parsed them.)
        for key, value in cfg.items():
            if not hasattr(args, key):
                continue
            default_val = parser.get_default(key)
            if getattr(args, key) == default_val:
                setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}, Arch: {args.arch}, "
          f"Mode: {args.training_mode}, Aug: {args.aug_profile}")

    # --- Data ---
    train_dataset = datasets.ImageFolder(
        args.data_dir / "train",
        transform=get_transforms(args.img_size, is_train=True, aug_profile=args.aug_profile),
    )
    val_dataset = datasets.ImageFolder(
        args.data_dir / "val",
        transform=get_transforms(args.img_size, is_train=False, aug_profile=args.aug_profile),
    )

    print(f"Train: {len(train_dataset)} images, "
          f"classes: {train_dataset.class_to_idx}")
    print(f"Val: {len(val_dataset)} images")

    sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # --- Model ---
    model = build_model(args.arch, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Metadata saved with every checkpoint so classify_predictions.py can
    # rebuild the right backbone and preprocessing from the file alone.
    ckpt_meta = {
        "arch": args.arch,
        "training_mode": args.training_mode,
        "aug_profile": args.aug_profile,
        "img_size": args.img_size,
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "preprocessing": {
            "resize": args.img_size,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_balanced_acc = 0.0
    history = []

    start_stage = 1
    start_epoch = 0

    resume_ckpt = None
    if args.resume and args.resume.exists():
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        best_balanced_acc = resume_ckpt.get("best_balanced_acc", 0.0)
        start_stage = resume_ckpt.get("stage", 1)
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        history_path = args.output_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])
        print(f"Resumed from stage {start_stage}, epoch {start_epoch}, "
              f"best_bal_acc={best_balanced_acc:.4f}")

    # --- Stage 1: Frozen backbone ---
    if start_stage <= 1:
        print(f"\n=== Stage 1: Frozen backbone ({args.freeze_epochs} epochs) ===")
        freeze_backbone(model, args.arch)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )
        # Restore optimizer state if resuming into stage 1
        if resume_ckpt and start_stage == 1 and "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])

        s1_start = start_epoch if start_stage == 1 else 0
        for epoch in range(s1_start, args.freeze_epochs):
            t0 = time.time()
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, use_amp
            )
            val_metrics = validate(model, val_loader, criterion, device, use_amp)
            dt = time.time() - t0

            bal_acc = val_metrics["balanced_accuracy"]
            entry = {
                "stage": 1, "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_balanced_acc": bal_acc,
                "val_per_class": val_metrics["per_class"],
            }
            history.append(entry)

            improved = ""
            if bal_acc > best_balanced_acc:
                best_balanced_acc = bal_acc
                _save_checkpoint(model, scaler, 1, epoch, best_balanced_acc,
                                 args.output_dir / "best_cls.pth",
                                 optimizer=optimizer, meta=ckpt_meta)
                improved = " *best*"

            print(f"  S1 E{epoch:02d} | "
                  f"loss={train_metrics['loss']:.4f} "
                  f"acc={train_metrics['accuracy']:.3f} | "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"bal_acc={bal_acc:.3f}{improved} "
                  f"[{dt:.1f}s]")

        # Save last S1 checkpoint
        _save_checkpoint(model, scaler, 1, args.freeze_epochs - 1,
                         best_balanced_acc, args.output_dir / "last_cls.pth",
                         optimizer=optimizer, meta=ckpt_meta)

    # Linear probe mode skips Stage 2 entirely — backbone stays frozen.
    if args.training_mode == "linear_probe":
        print("\n[linear_probe] Skipping Stage 2 (backbone stays frozen).")
        _finalize(args, best_balanced_acc, history, ckpt_meta)
        return

    # --- Stage 2: Full fine-tune ---
    print(f"\n=== Stage 2: Full fine-tune ({args.finetune_epochs} epochs) ===")
    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=1e-6
    )
    # Restore optimizer + scheduler state if resuming into stage 2
    if resume_ckpt and start_stage == 2:
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])

    s2_start = start_epoch if start_stage == 2 else 0
    for epoch in range(s2_start, args.finetune_epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_metrics = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        dt = time.time() - t0

        bal_acc = val_metrics["balanced_accuracy"]
        entry = {
            "stage": 2, "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_balanced_acc": bal_acc,
            "val_per_class": val_metrics["per_class"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(entry)

        improved = ""
        if bal_acc > best_balanced_acc:
            best_balanced_acc = bal_acc
            _save_checkpoint(model, scaler, 2, epoch, best_balanced_acc,
                             args.output_dir / "best_cls.pth",
                             optimizer=optimizer, scheduler=scheduler,
                             meta=ckpt_meta)
            improved = " *best*"

        print(f"  S2 E{epoch:02d} | "
              f"loss={train_metrics['loss']:.4f} "
              f"acc={train_metrics['accuracy']:.3f} | "
              f"val_loss={val_metrics['loss']:.4f} "
              f"bal_acc={bal_acc:.3f}{improved} "
              f"lr={scheduler.get_last_lr()[0]:.2e} "
              f"[{dt:.1f}s]")

    # Save last checkpoint
    _save_checkpoint(model, scaler, 2, args.finetune_epochs - 1,
                     best_balanced_acc, args.output_dir / "last_cls.pth",
                     optimizer=optimizer, scheduler=scheduler, meta=ckpt_meta)

    _finalize(args, best_balanced_acc, history, ckpt_meta)


def _finalize(
    args: argparse.Namespace,
    best_balanced_acc: float,
    history: list,
    ckpt_meta: dict,
) -> None:
    """Write training history + summary config. Called at end of both
    `full_ft` and `linear_probe` paths."""
    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    config = {
        **ckpt_meta,
        "freeze_epochs": args.freeze_epochs,
        "finetune_epochs": args.finetune_epochs if args.training_mode == "full_ft" else 0,
        "lr_stage1": args.lr,
        "lr_stage2": args.lr_finetune if args.training_mode == "full_ft" else None,
        "batch_size": args.batch_size,
        "best_balanced_accuracy": best_balanced_acc,
        "data_dir": str(args.data_dir),
    }
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n=== Training Complete ===")
    print(f"  Best balanced accuracy: {best_balanced_acc:.4f}")
    print(f"  Best model: {args.output_dir / 'best_cls.pth'}")
    print(f"  History: {history_path}")


def _save_checkpoint(
    model: nn.Module,
    scaler: torch.amp.GradScaler,
    stage: int,
    epoch: int,
    best_balanced_acc: float,
    path: Path,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    meta: dict | None = None,
) -> None:
    state = {
        "model": model.state_dict(),
        "stage": stage,
        "epoch": epoch,
        "best_balanced_acc": best_balanced_acc,
        "scaler": scaler.state_dict(),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if meta is not None:
        state["meta"] = meta
    torch.save(state, path)


if __name__ == "__main__":
    main()
