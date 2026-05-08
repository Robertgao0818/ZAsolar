"""
Zero-shot evaluation of pre-trained classifiers on the JHB CBD audit set.

Three classifiers were trained on `cls_pv_thermal_v1` (CT batch003/004
reviewed predictions). The 462-row V3-C ∩ V4.2 shared-FP audit set is
JHB CBD, GEID 2024-02 imagery, V3-C+SAM mask+box predictions — none of
these chips entered classifier training. This is therefore an OOD
generalization test:
- can a CT-trained PV-vs-non-PV classifier rescue the 119
  actually_pv_mislabeled chips (treat them as PV) without nuking the
  remaining 343 true non-PV?
- which backbone (efficientnet_b0 / convnext_tiny / dinov2_vits14)
  generalizes best?

Reads pre-extracted chips at `chips/nonpv/v3c_*.png` (built by
build_cls_dataset_cascade.py) and classifies each chip with each
backbone's `best_cls.pth`.

Usage:
    python scripts/classifier/eval_on_audit_set.py \\
        --pool-dir data/cls_pv_nonpv_v3c_v42_cascade \\
        --labeled-csv data/cls_pv_nonpv_v3c_v42_cascade/labeler/v3c__both/nonpv_subtype_labeled.csv \\
        --output-dir results/analysis/cls_audit_eval_20260427
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.classifier.classify_predictions import build_model, load_classifier  # noqa: E402

DEFAULT_CHECKPOINTS = {
    "efficientnet_b0": "checkpoints/cls_pv_thermal_v1_effb0/best_cls.pth",
    "convnext_tiny":   "checkpoints/cls_pv_thermal_v1_convnext_tiny/best_cls.pth",
    "dinov2_vits14":   "checkpoints/cls_pv_thermal_v1_dinov2_vits14/best_cls.pth",
}

POSITIVE_HUMAN_LABEL = "actually_pv_mislabeled"  # → binary y_true=1 (PV)


class ChipFileDataset(Dataset):
    def __init__(self, chip_paths: list[Path], img_size: int, mean, std):
        self.chip_paths = chip_paths
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.chip_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.chip_paths[idx]))
        if img is None:
            raise RuntimeError(f"Failed to read {self.chip_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return self.transform(img), idx


def evaluate_one(
    arch: str,
    checkpoint: Path,
    chip_paths: list[Path],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict]:
    print(f"\n  [{arch}] loading {checkpoint}")
    model, config = load_classifier(checkpoint, device)
    img_size = int(config.get("img_size", 224))
    mean = config["preprocessing"]["mean"]
    std = config["preprocessing"]["std"]
    pv_idx = config.get("class_names", ["non_pv", "pv"]).index("pv")

    ds = ChipFileDataset(chip_paths, img_size, mean, std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    pv_scores = np.zeros(len(chip_paths), dtype=np.float32)
    with torch.no_grad():
        for tensors, idxs in loader:
            tensors = tensors.to(device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for k, idx in enumerate(idxs.numpy()):
                pv_scores[idx] = probs[k, pv_idx]
    return pv_scores, {"arch": arch, "img_size": img_size, "pv_idx": int(pv_idx)}


def threshold_metrics(pv_scores: np.ndarray, y_true: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (pv_scores >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        n_pv = int((y_true == 1).sum())
        n_npv = int((y_true == 0).sum())
        recall_pv = tp / max(n_pv, 1)
        prec_pv = tp / max(tp + fp, 1)
        # In the cascade we want the classifier to KEEP PV (y_pred=1 → keep)
        # and DROP non-PV (y_pred=0 → drop). Thus:
        # - PV preserved rate = recall on PV
        # - non-PV killed rate = TN / N(non-PV) = specificity on PV-positive
        npv_killed = tn / max(n_npv, 1)
        bal_acc = (recall_pv + npv_killed) / 2
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "pv_recall": round(recall_pv, 4),
            "pv_precision": round(prec_pv, 4),
            "nonpv_kill_rate": round(npv_killed, 4),
            "bal_acc": round(bal_acc, 4),
        })
    return pd.DataFrame(rows)


def find_threshold_at_pv_recall(
    pv_scores: np.ndarray, y_true: np.ndarray, target_recall: float
) -> tuple[float, float]:
    """Return (threshold, achieved_kill_rate) where pv_recall ≥ target_recall."""
    pv_mask = y_true == 1
    if pv_mask.sum() == 0:
        return float("nan"), float("nan")
    # Sort PV scores descending; threshold = score of the (target_recall * n_pv)-th
    pv_scores_sorted = np.sort(pv_scores[pv_mask])[::-1]
    n_pv = pv_mask.sum()
    k = int(np.ceil(target_recall * n_pv))
    k = max(1, min(k, n_pv))
    thr = float(pv_scores_sorted[k - 1])
    y_pred = pv_scores >= thr
    npv = (~pv_mask).sum()
    tn = int(((~y_pred) & (~pv_mask)).sum())
    return thr, tn / max(int(npv), 1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--labeled-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="(arch=path) overrides; defaults to cls_pv_thermal_v1_*")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[1/4] Read manifest + audit labels")
    manifest = pd.read_csv(args.pool_dir / "manifest.csv")
    audit = pd.read_csv(args.labeled_csv)
    df = audit.merge(
        manifest[["chip_id", "chip_path"]], on="chip_id", how="left"
    )
    df["chip_full"] = df["chip_path"].apply(lambda r: str(args.pool_dir / r))
    df["y_true_pv"] = (df["human_label"] == POSITIVE_HUMAN_LABEL).astype(int)
    n_pv = int(df["y_true_pv"].sum())
    n_npv = int((df["y_true_pv"] == 0).sum())
    print(f"  rows: {len(df)} | PV (mislabeled): {n_pv} | non-PV: {n_npv}")

    chip_paths = [Path(p) for p in df["chip_full"]]
    missing = [p for p in chip_paths if not p.exists()]
    if missing:
        print(f"  [WARN] {len(missing)} chip files missing; first: {missing[0]}")
    chip_paths = [p for p in chip_paths if p.exists()]

    if args.checkpoints is None:
        ckpt_map = {arch: Path(p) for arch, p in DEFAULT_CHECKPOINTS.items()}
    else:
        ckpt_map = {}
        for entry in args.checkpoints:
            arch, path = entry.split("=", 1)
            ckpt_map[arch] = Path(path)

    print(f"\n[2/4] Run inference for {len(ckpt_map)} backbones on {len(chip_paths)} chips")
    score_table = df[["chip_id", "grid_id", "pred_idx", "human_label", "y_true_pv"]].copy()
    backbone_summaries = []
    for arch, ckpt in ckpt_map.items():
        scores, meta = evaluate_one(arch, ckpt, chip_paths, device, args.batch_size)
        col = f"pv_score_{arch}"
        score_table[col] = scores

        # Threshold sweep
        thr_df = threshold_metrics(
            scores, df["y_true_pv"].values,
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        thr_df.to_csv(args.output_dir / f"threshold_sweep_{arch}.csv", index=False)

        # Threshold at high PV-recall (we don't want to suppress real PV)
        thr_r95, kill_r95 = find_threshold_at_pv_recall(scores, df["y_true_pv"].values, 0.95)
        thr_r99, kill_r99 = find_threshold_at_pv_recall(scores, df["y_true_pv"].values, 0.99)

        # Bal acc at threshold 0.5
        row_05 = thr_df[thr_df["threshold"] == 0.5].iloc[0]

        backbone_summaries.append({
            "arch": arch,
            "checkpoint": str(ckpt),
            "n_chips": int(len(chip_paths)),
            "n_pv": n_pv, "n_nonpv": n_npv,
            "thr_0.5_pv_recall": float(row_05["pv_recall"]),
            "thr_0.5_nonpv_kill": float(row_05["nonpv_kill_rate"]),
            "thr_0.5_bal_acc": float(row_05["bal_acc"]),
            "thr_at_pv_recall_0.95": float(thr_r95),
            "nonpv_kill_at_pv_recall_0.95": float(kill_r95),
            "thr_at_pv_recall_0.99": float(thr_r99),
            "nonpv_kill_at_pv_recall_0.99": float(kill_r99),
        })
        print(
            f"  {arch}: @0.5  pv_recall={row_05['pv_recall']:.3f}  "
            f"nonpv_kill={row_05['nonpv_kill_rate']:.3f}  bal_acc={row_05['bal_acc']:.3f}"
        )
        print(
            f"  {arch}: @PV-recall=0.95 thr={thr_r95:.3f} nonpv_kill={kill_r95:.3f} | "
            f"@PV-recall=0.99 thr={thr_r99:.3f} nonpv_kill={kill_r99:.3f}"
        )

    print("\n[3/4] Write outputs")
    score_table.to_csv(args.output_dir / "scores.csv", index=False)
    pd.DataFrame(backbone_summaries).to_csv(args.output_dir / "summary.csv", index=False)

    print("\n[4/4] Per-subtype kill-rate at threshold=0.5 (per backbone)")
    rows = []
    for arch in ckpt_map:
        col = f"pv_score_{arch}"
        for sub, sub_df in score_table.groupby("human_label"):
            kill = (sub_df[col] < 0.5).sum() / len(sub_df)
            rows.append({"arch": arch, "subtype": sub, "n": len(sub_df), "kill_rate@0.5": round(float(kill), 4)})
    by_sub = pd.DataFrame(rows)
    by_sub.to_csv(args.output_dir / "per_subtype_kill.csv", index=False)
    pivot = by_sub.pivot(index="subtype", columns="arch", values="kill_rate@0.5")
    pivot["n"] = by_sub.groupby("subtype")["n"].first().values
    print(pivot.to_string())

    print(f"\nOutputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
