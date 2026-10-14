"""Optional DINOv2 presence pregate for one paired trial.

The Cape Town 26-grid decision keeps a chip when its out-of-fold score is
at least 0.45. Those scores came from four fold heads (vit30 folds 0–3):
each Cape Town grid was scored by the single head that did not train on it.
Fold 4 and vit30_final were not trained.

A Durban chip was in none of those folds. This module scores every supplied
head on one frozen backbone and keeps the chip when the highest score is
at least the threshold. With the four fold heads, a chip is dropped only
when every head is below the threshold. That is more permissive than any
one head, and it is not the Cape Town operating point transferred onto
Vexcel Durban imagery.

The pregate never downloads imagery or weights. Callers leave it off by
not supplying a head.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

TRIAL_THRESHOLD = 0.45
TRAINED_CHIP_SIZE = 400
VIT_IMG = 392
VIT_PATCHES = (VIT_IMG // 14) ** 2
VIT_DIM = 384
VIT_MEAN = (0.485, 0.456, 0.406)
VIT_STD = (0.229, 0.224, 0.225)
BACKBONE_NAME = "vit_small_patch14_dinov2.lvd142m"
DINO_CACHE = Path.home() / "zasolar_data" / "models" / "dinov2_floor" / "hf_cache"
LEARNING_CURVE_RUNS = (
    Path.home()
    / "zasolar_data"
    / "gate_pilot_2026-09"
    / "learning_curve_2026-09-28"
    / "train"
    / "runs"
)
TIMING_SCHEMA = "pregate_stage_timing_v1"
KEEP_RULE = (
    "max head score >= threshold; a chip is rejected only when every "
    "supplied head is below the threshold"
)
TIMING_NOTE = (
    "batch_wait_s is main-thread stall on the DataLoader, not the full "
    "tile-read time, because worker reads overlap the GPU. pregate_s and "
    "detect_s are synchronized stage times. Their sum is not loop_wall_s. "
    "chip_skip_fraction is the share of chips that skipped the detector. "
    "It is not the share of wall-clock time saved. loop_wall_s already "
    "includes tile reads, the pregate, and detection."
)


def default_head_paths() -> tuple[Path, ...]:
    """The four heads that produced the Cape Town 26-grid scores.

    Fold 4 is absent on purpose: that fold was not trained.
    """
    return tuple(
        LEARNING_CURVE_RUNS / f"vit30_fold{fold}" / "head.pt"
        for fold in range(4)
    )


@dataclass(frozen=True)
class PregateSettings:
    head_paths: tuple[Path, ...]
    threshold: float
    head_labels: tuple[str, ...]


def resolve_pregate_settings(
    *,
    heads: Sequence[str] | None,
    threshold: float | None,
    chip_size: int,
    allow_non400: bool,
) -> PregateSettings | None:
    """Return None when the gate is off.

    A threshold without a head is refused. Leaving the gate off in that
    case would look like a threshold run and would not be one.
    """
    supplied = [str(item) for item in (heads or [])]
    if not supplied:
        if threshold is not None:
            raise ValueError(
                "--pregate-threshold was set without --pregate-head; "
                "refusing to leave the gate off by accident"
            )
        return None
    if int(chip_size) != TRAINED_CHIP_SIZE and not allow_non400:
        raise ValueError(
            f"pregate was trained on {TRAINED_CHIP_SIZE}px chips; "
            f"got chip_size={chip_size}. Pass --pregate-allow-non400 to override."
        )
    chosen = TRIAL_THRESHOLD if threshold is None else float(threshold)
    if not math.isfinite(chosen) or not (0.0 <= chosen <= 1.0):
        raise ValueError(f"pregate threshold must be inside [0, 1], got {chosen}")
    paths: list[Path] = []
    labels: list[str] = []
    for raw in supplied:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"pregate head is not a file: {path}")
        label = path.parent.name
        if not label or label in labels:
            raise ValueError(f"duplicate or empty pregate head label: {label!r}")
        paths.append(path)
        labels.append(label)
    return PregateSettings(
        head_paths=tuple(paths),
        threshold=chosen,
        head_labels=tuple(labels),
    )


def keep_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    """True where the highest head score is >= threshold.

    `scores` is (n,) for one head or (n, n_heads) for several. A score equal
    to the threshold is kept, matching the Cape Town rule score >= 0.45.
    """
    chosen = float(threshold)
    if not math.isfinite(chosen) or not (0.0 <= chosen <= 1.0):
        raise ValueError(f"pregate threshold must be inside [0, 1], got {chosen}")
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"pregate scores must have shape (n,) or (n, heads), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("pregate scores must be finite")
    return array.max(axis=1) >= chosen


def prepare_gate_batch(chips: Sequence["torch.Tensor"]) -> "torch.Tensor":
    """Resize float CHW chips in [0, 1] the way the fold heads were trained.

    Training opened 400px PNG files, divided by 255, then bilinear-resized
    to 392 with align_corners False and ImageNet normalization. No crop.
    """
    import torch
    import torch.nn.functional as F

    if not chips:
        raise ValueError("pregate received an empty chip batch")
    stacked = []
    for chip in chips:
        if chip.ndim != 3 or int(chip.shape[0]) != 3:
            raise ValueError(f"pregate chip must be CHW with 3 channels, got {tuple(chip.shape)}")
        if chip.dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError(
                "pregate expects float chips already scaled to [0, 1], "
                f"got dtype {chip.dtype}"
            )
        stacked.append(chip.detach().to(dtype=torch.float32).cpu())
    batch = torch.stack(stacked, dim=0)
    if not torch.isfinite(batch).all():
        raise ValueError("pregate chips must be finite")
    peak = float(batch.max()) if batch.numel() else 0.0
    floor = float(batch.min()) if batch.numel() else 0.0
    if peak > 1.5 or floor < -0.01:
        raise ValueError(
            "pregate chips are not in [0, 1] "
            f"(min={floor:.4f}, max={peak:.4f}). "
            "detect_direct leaves float32 rasters unscaled. Refusing to score them."
        )
    batch = F.interpolate(
        batch, size=(VIT_IMG, VIT_IMG), mode="bilinear", align_corners=False,
    )
    mean = torch.tensor(VIT_MEAN, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(VIT_STD, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PresencePregate:
    """Frozen DINOv2 backbone plus one or more linear patch heads."""

    def __init__(
        self,
        head_paths: Sequence[Path],
        device: str,
        backbone: "torch.nn.Module | None" = None,
        cache_dir: Path | None = None,
    ):
        import torch

        self.device = device
        self.head_paths = tuple(Path(path) for path in head_paths)
        self.head_labels = tuple(path.parent.name for path in self.head_paths)
        if backbone is None:
            backbone = load_backbone(device, cache_dir)
        self.backbone = backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.heads = []
        for path in self.head_paths:
            head = load_head(path, device)
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad_(False)
            self.heads.append(head)
        self._torch = torch

    def score(self, chips: Sequence["torch.Tensor"]) -> np.ndarray:
        """Return shape (n_chips, n_heads) sigmoid scores."""
        torch = self._torch
        if not chips:
            return np.zeros((0, len(self.heads)), dtype=np.float32)
        batch = prepare_gate_batch(chips).to(self.device, non_blocking=True)
        with torch.inference_mode():
            features = self.backbone.forward_features(batch)
            if features.ndim != 3 or int(features.shape[1]) != VIT_PATCHES + 1:
                raise RuntimeError(
                    f"expected {(VIT_PATCHES + 1)} tokens, got {tuple(features.shape)}"
                )
            if int(features.shape[2]) != VIT_DIM:
                raise RuntimeError(f"expected embed dim {VIT_DIM}, got {features.shape[2]}")
            patch = features[:, 1:, :]
            columns = []
            for head in self.heads:
                logit = head(patch).squeeze(-1).amax(dim=1)
                columns.append(torch.sigmoid(logit))
            stacked = torch.stack(columns, dim=1)
        return stacked.detach().to(dtype=torch.float32).cpu().numpy()


def load_backbone(device: str, cache_dir: Path | None = None):
    """Load the frozen local DINOv2 backbone. Does not download weights."""
    cache = Path(cache_dir) if cache_dir is not None else DINO_CACHE
    if not cache.is_dir():
        raise FileNotFoundError(
            f"DINOv2 cache is not a directory: {cache}. "
            "The pregate does not download weights."
        )
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required to load the DINOv2 pregate backbone") from exc
    try:
        model = timm.create_model(
            BACKBONE_NAME, pretrained=True, num_classes=0, img_size=VIT_IMG,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the frozen DINOv2 backbone from the local cache. "
            "The pregate does not download weights."
        ) from exc
    model = model.eval().to(device)
    if int(model.num_features) != VIT_DIM:
        raise RuntimeError(f"expected embed dim {VIT_DIM}, got {model.num_features}")
    config = timm.data.resolve_data_config({}, model=model)
    if tuple(config["mean"]) != VIT_MEAN or tuple(config["std"]) != VIT_STD:
        raise RuntimeError(f"DINOv2 normalization changed: {config['mean']} {config['std']}")
    return model


def load_head(path: Path, device: str):
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(f"{path} has no state_dict")
    head = torch.nn.Linear(VIT_DIM, 1)
    head.load_state_dict(checkpoint["state_dict"])
    return head.eval().to(device)


@dataclass
class StageClock:
    batch_wait_s: float = 0.0
    pregate_s: float = 0.0
    detect_s: float = 0.0
    loop_wall_s: float = 0.0
    n_chips: int = 0
    n_kept: int = 0
    n_rejected: int = 0
    n_detections: int = 0


def chip_skip_fraction(clock: StageClock) -> float | None:
    if clock.n_chips <= 0:
        return None
    return clock.n_rejected / clock.n_chips


def build_timing_payload(
    *,
    clock: StageClock,
    settings: PregateSettings | None,
    head_sha256: Sequence[str] = (),
) -> dict:
    heads = []
    if settings is not None:
        if len(head_sha256) not in (0, len(settings.head_paths)):
            raise ValueError("head_sha256 must be empty or one digest per head")
        for index, (label, path) in enumerate(zip(settings.head_labels, settings.head_paths)):
            item = {"label": label, "path": str(path)}
            if head_sha256:
                item["sha256"] = head_sha256[index]
            heads.append(item)
    skip = chip_skip_fraction(clock)
    return {
        "schema": TIMING_SCHEMA,
        "gate_enabled": settings is not None,
        "threshold": None if settings is None else settings.threshold,
        "keep_rule": None if settings is None else KEEP_RULE,
        "threshold_source": (
            None
            if settings is None
            else "cape_town_26_grid_owner_confirmed_0.45_not_validated_on_durban_vexcel"
        ),
        "heads": heads,
        "batch_wait_s": clock.batch_wait_s,
        "pregate_s": clock.pregate_s,
        "detect_s": clock.detect_s,
        "loop_wall_s": clock.loop_wall_s,
        "n_chips": clock.n_chips,
        "n_kept": clock.n_kept,
        "n_rejected": clock.n_rejected,
        "n_detections": clock.n_detections,
        "chip_skip_fraction": skip,
        "note": TIMING_NOTE,
    }


def timing_path(raw_path: Path) -> Path:
    return Path(raw_path).with_name("stage_timing.json")


def decisions_path(raw_path: Path) -> Path:
    return Path(raw_path).with_name("pregate_decisions.csv")


def write_timing(path: Path, payload: dict) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_decisions(path: Path, rows: Sequence[dict], head_labels: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chip_index", "kept", "score_max", *list(head_labels)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
