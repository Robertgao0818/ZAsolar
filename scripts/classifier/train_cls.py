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
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


def get_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Get data augmentation transforms."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


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
    if arch == "efficientnet_b0":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    elif arch == "resnet18":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False


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


def main():
    parser = argparse.ArgumentParser(description="Train PV vs non-PV classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/cls_pv_thermal"))
    parser.add_argument("--arch", default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet18"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Stage 1: epochs with frozen backbone")
    parser.add_argument("--finetune-epochs", type=int, default=25,
                        help="Stage 2: epochs with full fine-tune")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Stage 1 learning rate")
    parser.add_argument("--lr-finetune", type=float, default=1e-4,
                        help="Stage 2 learning rate")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}, Arch: {args.arch}")

    # --- Data ---
    train_dataset = datasets.ImageFolder(
        args.data_dir / "train",
        transform=get_transforms(args.img_size, is_train=True),
    )
    val_dataset = datasets.ImageFolder(
        args.data_dir / "val",
        transform=get_transforms(args.img_size, is_train=False),
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
                                 optimizer=optimizer)
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
                         optimizer=optimizer)

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
                             optimizer=optimizer, scheduler=scheduler)
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
                     optimizer=optimizer, scheduler=scheduler)

    # --- Save training history ---
    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        "arch": args.arch,
        "img_size": args.img_size,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "freeze_epochs": args.freeze_epochs,
        "finetune_epochs": args.finetune_epochs,
        "lr_stage1": args.lr,
        "lr_stage2": args.lr_finetune,
        "batch_size": args.batch_size,
        "best_balanced_accuracy": best_balanced_acc,
        "data_dir": str(args.data_dir),
    }
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Training Complete ===")
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
    torch.save(state, path)


if __name__ == "__main__":
    main()
