"""
Calibrate per-imagery-source thresholds for cls_pv_thermal_v2 backbones.

The v2 protocol replaces v1's single 0.5/0.85 threshold with one threshold
per imagery layer:

  - `aerial_2025`     (CT)         calibrated on CT v1 val
  - `aerial_2023`     (JHB suburbs) calibrated on JHB v4 holdout (also in v1)
  - `geid_2024_02`    (JHB CBD)     calibrated on the new audit holdout

The val chip filenames carry the imagery layer in their prefix (set when
v1/v2 builders write them):

  - `cape_town_*`   → aerial_2025
  - `johannesburg_*`→ aerial_2023
  - `jhbcbd_*`      → geid_2024_02

For each (backbone, layer), this script:

  1. Scores every val chip with the backbone's `best_cls.pth`.
  2. Finds the threshold T s.t. PV-recall ≥ target on that layer's val.
  3. Reports non-PV-kill rate at T.
  4. Writes `configs/classifier/thresholds_v2.json`.
  5. Applies the v2 promotion rule (PV-recall ≥ 0.95 → non-PV-kill ≥ 0.40
     on geid_2024_02), reporting per-backbone pass/fail.

Usage:
    python scripts/classifier/calibrate_v2_thresholds.py \\
        --data-dir data/cls_pv_thermal_v2 \\
        --output configs/classifier/thresholds_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.classifier.classify_predictions import load_classifier  # noqa: E402

DEFAULT_BACKBONES = {
    "efficientnet_b0": "checkpoints/cls_pv_thermal_v2_efficientnet_b0/best_cls.pth",
    "convnext_tiny":   "checkpoints/cls_pv_thermal_v2_convnext_tiny/best_cls.pth",
    "dinov2_vits14":   "checkpoints/cls_pv_thermal_v2_dinov2_vits14/best_cls.pth",
}

LAYER_FROM_PREFIX = {
    "cape_town":    "aerial_2025",
    "johannesburg": "aerial_2023",
    "jhbcbd":       "geid_2024_02",
}

PROMOTION_TARGET_RECALL = 0.95
PROMOTION_MIN_KILL = 0.40
PROMOTION_LAYER = "geid_2024_02"


class ChipDataset(Dataset):
    def __init__(self, paths: list[Path], img_size: int, mean, std):
        self.paths = paths
        self.img_size = img_size
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return self.tf(img), idx


def imagery_layer_for(path: Path) -> str | None:
    """Map a val chip filename → imagery layer."""
    name = path.name
    for prefix, layer in LAYER_FROM_PREFIX.items():
        if name.startswith(prefix + "_"):
            return layer
    return None


def collect_val_paths(data_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """Gather val chips with (path, y_true, layer) tuples."""
    paths: list[Path] = []
    y: list[int] = []
    layers: list[str] = []

    for cls_idx, cls in enumerate(("non_pv", "pv")):
        for p in sorted((data_dir / "val" / cls).glob("*.png")):
            layer = imagery_layer_for(p)
            if layer is None:
                continue  # unknown source, skip
            paths.append(p)
            y.append(cls_idx)  # 0=non_pv, 1=pv
            layers.append(layer)

    return paths, np.array(y, dtype=int), layers


def score_paths(arch: str, ckpt: Path, paths: list[Path],
                device: torch.device, batch_size: int = 64) -> tuple[np.ndarray, dict]:
    print(f"\n  [{arch}] loading {ckpt}")
    model, config = load_classifier(ckpt, device)
    img_size = int(config.get("img_size", 224))
    mean = config["preprocessing"]["mean"]
    std = config["preprocessing"]["std"]
    pv_idx = config.get("class_names", ["non_pv", "pv"]).index("pv")

    ds = ChipDataset(paths, img_size, mean, std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    scores = np.zeros(len(paths), dtype=np.float32)
    with torch.no_grad():
        for tensors, idxs in loader:
            tensors = tensors.to(device)
            probs = torch.softmax(model(tensors), dim=1).cpu().numpy()
            for k, idx in enumerate(idxs.numpy()):
                scores[idx] = probs[k, pv_idx]
    return scores, {"img_size": img_size, "pv_idx": int(pv_idx)}


def threshold_at_pv_recall(scores: np.ndarray, y: np.ndarray,
                            target_recall: float) -> tuple[float, dict]:
    pv_mask = (y == 1)
    n_pv = int(pv_mask.sum())
    n_npv = int((~pv_mask).sum())
    if n_pv == 0:
        return float("nan"), {"n_pv": 0, "n_nonpv": n_npv}
    pv_sorted = np.sort(scores[pv_mask])[::-1]
    k = max(1, min(int(np.ceil(target_recall * n_pv)), n_pv))
    thr = float(pv_sorted[k - 1])
    y_pred = scores >= thr
    tp = int(((y_pred) & pv_mask).sum())
    tn = int(((~y_pred) & (~pv_mask)).sum())
    return thr, {
        "n_pv": n_pv, "n_nonpv": n_npv,
        "pv_recall": tp / n_pv,
        "nonpv_kill": (tn / n_npv) if n_npv else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "cls_pv_thermal_v2")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "configs" / "classifier" / "thresholds_v2.json")
    p.add_argument("--target-recall", type=float, default=PROMOTION_TARGET_RECALL)
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="(arch=path) overrides; defaults to cls_pv_thermal_v2_*")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--also-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.9, 0.95],
                   help="Additional thresholds to report metrics at, per layer.")
    args = p.parse_args()

    device = torch.device(args.device)

    print("[1/3] Collect v2 val chips")
    paths, y, layers = collect_val_paths(args.data_dir)
    layer_arr = np.array(layers)
    print(f"  Total val chips: {len(paths)}")
    for layer in sorted(set(layers)):
        mask = layer_arr == layer
        n_pv = int(((y == 1) & mask).sum())
        n_npv = int(((y == 0) & mask).sum())
        print(f"    {layer}: pv={n_pv} non_pv={n_npv} (total {n_pv+n_npv})")

    if args.checkpoints is None:
        ckpt_map = {arch: Path(p) for arch, p in DEFAULT_BACKBONES.items()}
    else:
        ckpt_map = {}
        for entry in args.checkpoints:
            arch, path = entry.split("=", 1)
            ckpt_map[arch] = Path(path)

    print(f"\n[2/3] Score val chips with {len(ckpt_map)} backbones")
    output: dict = {
        "_meta": {
            "description": "Per-imagery-source PV-vs-non-PV classifier thresholds (v2 protocol)",
            "spec_doc": "docs/experiments/exp_cls_dataset_protocol.md (v2 protocol)",
            "promotion_rule": {
                "target_pv_recall": PROMOTION_TARGET_RECALL,
                "min_nonpv_kill_on_layer": PROMOTION_LAYER,
                "min_nonpv_kill": PROMOTION_MIN_KILL,
            },
            "calibration_dataset": str(args.data_dir.relative_to(PROJECT_ROOT)),
        },
        "by_backbone": {},
    }
    promotion_summary: list[dict] = []

    for arch, ckpt in ckpt_map.items():
        if not ckpt.exists():
            print(f"\n  [{arch}] checkpoint missing: {ckpt} — skipping")
            continue
        scores, _meta = score_paths(arch, ckpt, paths, device, args.batch_size)
        per_layer: dict = {}
        for layer in sorted(set(layers)):
            mask = layer_arr == layer
            thr, m = threshold_at_pv_recall(
                scores[mask], y[mask], args.target_recall,
            )
            per_layer[layer] = {
                "threshold": thr,
                "calibration": {
                    "target_pv_recall": args.target_recall,
                    "achieved_pv_recall": round(m.get("pv_recall", float("nan")), 4),
                    "nonpv_kill": round(m.get("nonpv_kill", float("nan")), 4),
                    "n_pv": m["n_pv"], "n_nonpv": m["n_nonpv"],
                },
                "diagnostic_at_fixed_thresholds": {},
            }
            for t in args.also_thresholds:
                pred = scores[mask] >= t
                pv = (y[mask] == 1)
                tp = int(((pred) & pv).sum())
                tn = int(((~pred) & (~pv)).sum())
                per_layer[layer]["diagnostic_at_fixed_thresholds"][f"{t:.2f}"] = {
                    "pv_recall": round(tp / max(int(pv.sum()), 1), 4),
                    "nonpv_kill": round(tn / max(int((~pv).sum()), 1), 4),
                }
            print(f"  {arch}/{layer}: thr={thr:.4f} "
                  f"pv_rec={per_layer[layer]['calibration']['achieved_pv_recall']:.3f} "
                  f"npv_kill={per_layer[layer]['calibration']['nonpv_kill']:.3f}")

        output["by_backbone"][arch] = {
            "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
            "thresholds": per_layer,
        }

        # Apply promotion rule (only on geid_2024_02)
        if PROMOTION_LAYER in per_layer:
            kill = per_layer[PROMOTION_LAYER]["calibration"]["nonpv_kill"]
            promotable = (
                per_layer[PROMOTION_LAYER]["calibration"]["achieved_pv_recall"]
                >= args.target_recall
                and kill >= PROMOTION_MIN_KILL
            )
            promotion_summary.append({
                "arch": arch,
                "geid_threshold": per_layer[PROMOTION_LAYER]["threshold"],
                "geid_pv_recall": per_layer[PROMOTION_LAYER]["calibration"]["achieved_pv_recall"],
                "geid_nonpv_kill": kill,
                "promotable": promotable,
            })

    print("\n[3/3] Write outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"  Wrote {args.output}")

    print("\n=== v2 promotion rule (PV-recall ≥ %.2f → non-PV-kill ≥ %.2f on %s) ==="
          % (PROMOTION_TARGET_RECALL, PROMOTION_MIN_KILL, PROMOTION_LAYER))
    if promotion_summary:
        df = pd.DataFrame(promotion_summary)
        print(df.to_string(index=False))
    else:
        print("  No promotion summary (geid_2024_02 layer absent or no checkpoints scored)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
