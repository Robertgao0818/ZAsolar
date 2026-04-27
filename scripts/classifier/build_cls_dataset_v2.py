"""
Build cls_pv_thermal_v2 — multi-subtype FP suppressor classifier dataset.

V2 protocol freezes the role pivot (water-heater filter → multi-subtype FP
suppressor) by combining the v1 CT-trained pool with the JHB CBD GEID
audit core. See `docs/experiments/exp_cls_dataset_protocol.md` "v2 protocol"
for the full spec.

Inputs
------
- v1 chips at `data/cls_pv_thermal_v1/{train,val}/{pv,non_pv}/*.png`
  (already split by region-stratified whole-grid; preserved as-is)
- Cascade pool at `data/cls_pv_nonpv_v3c_v42_cascade/chips/{pv,nonpv}/*.png`
  with manifest at `data/cls_pv_nonpv_v3c_v42_cascade/manifest.csv`
- Audit subtype labels at `.../labeler/v3c__both/nonpv_subtype_labeled.csv`
  (462 V3-C-side rows) + propagated `.../nonpv_subtype_labeled_v4_2.csv`
  (441 V4.2-side rows from per-grid IoU≥0.3 pairing)

Pipeline
--------
1. Take the 903 audited chips (V3-C 462 + V4.2 propagated 441) — these are
   `source_detector=both` AND `label=nonpv` in the cascade manifest, joined
   with the audit CSVs on `chip_id`.
2. Class flip: `human_label == "actually_pv_mislabeled"` → pv class
   (119+106 = 225 chips). Other 8 subtypes → non_pv class.
3. Subtype-stratified holdout (25-30%) on this new bucket only:
   - No grid leakage (same `grid_id` cannot appear in both train and val).
   - Each subtype with ≥6 total samples must contribute ≥1 chip to val.
   - Try seeds 0..N until both constraints met.
4. v1 splits are preserved (CT v1 buckets keep their train/val from
   `cls_pv_thermal_v1/`).
5. Output `data/cls_pv_thermal_v2/{train,val}/{pv,non_pv}/*.png` as
   symlinks (no chip duplication on disk) and a manifest with the
   per-bucket per-subtype counts.

Usage
-----
    python scripts/classifier/build_cls_dataset_v2.py \\
        --v1-root data/cls_pv_thermal_v1 \\
        --cascade-root data/cls_pv_nonpv_v3c_v42_cascade \\
        --output-dir data/cls_pv_thermal_v2 \\
        --val-fraction 0.27
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

NEW_BUCKET = "johannesburg:v3c_sam_mask_geid_2024_02"
PV_FLIP_LABEL = "actually_pv_mislabeled"


def load_audit_chips(cascade_root: Path) -> pd.DataFrame:
    """Load 903 audited chips (V3-C 462 + V4.2 propagated 441) with subtype.

    Returns DataFrame with columns:
        chip_id, detector, grid_id, pred_idx, area_m2, source_detector,
        chip_path (relative to cascade_root), human_label (subtype),
        label (post-flip: pv / non_pv).
    """
    manifest = pd.read_csv(cascade_root / "manifest.csv")
    v3c_csv = cascade_root / "labeler/v3c__both/nonpv_subtype_labeled.csv"
    v42_csv = cascade_root / "labeler/v3c__both/nonpv_subtype_labeled_v4_2.csv"

    audit_v3c = pd.read_csv(v3c_csv)[["chip_id", "human_label"]]
    audit_v42 = pd.read_csv(v42_csv)[["chip_id", "human_label"]]
    audit = pd.concat([audit_v3c, audit_v42], ignore_index=True)
    print(f"  Audit rows loaded: V3-C {len(audit_v3c)} + V4.2 propagated {len(audit_v42)} = {len(audit)}")

    # Join on chip_id; manifest's `label` is "nonpv" pre-flip — replace with post-flip.
    df = audit.merge(
        manifest[["chip_id", "grid_id", "detector", "pred_idx", "area_m2",
                  "source_detector", "chip_path"]],
        on="chip_id", how="left", validate="one_to_one",
    )
    missing = df["grid_id"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} audit chips missing from cascade manifest")

    # Class flip
    df["label"] = df["human_label"].map(
        lambda s: "pv" if s == PV_FLIP_LABEL else "non_pv"
    )
    df["source_bucket"] = NEW_BUCKET

    n_pv = (df["label"] == "pv").sum()
    n_npv = (df["label"] == "non_pv").sum()
    print(f"  After class flip: pv={n_pv}, non_pv={n_npv} (flipped from "
          f"{(df['human_label'] == PV_FLIP_LABEL).sum()} actually_pv_mislabeled)")
    return df


def stratified_grid_split(
    df: pd.DataFrame,
    val_fraction: float,
    min_subtype_n: int = 6,
    max_seeds: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Find a grid-grouped split that preserves subtype representation.

    Constraints:
    - No grid leakage: same `grid_id` cannot appear in both train and val.
    - Subtype coverage: every subtype with total count ≥ min_subtype_n
      must have ≥1 chip in val.
    """
    subtype_totals = df["human_label"].value_counts()
    must_cover = set(subtype_totals[subtype_totals >= min_subtype_n].index)
    print(f"  Subtypes requiring val coverage (N >= {min_subtype_n}): "
          f"{len(must_cover)} of {df['human_label'].nunique()}")

    for seed in range(max_seeds):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr_idx, va_idx = next(splitter.split(df, groups=df["grid_id"]))
        train, val = df.iloc[tr_idx], df.iloc[va_idx]
        val_subtypes = set(val["human_label"].unique())
        missing = must_cover - val_subtypes
        if not missing:
            print(f"  Seed {seed}: train={len(train)} val={len(val)} "
                  f"(val_frac={len(val)/len(df):.3f}); all {len(must_cover)} "
                  f"subtypes covered")
            return train, val, seed
    raise RuntimeError(
        f"No seed in 0..{max_seeds-1} satisfied subtype coverage; "
        f"loosen val_fraction or min_subtype_n"
    )


def link_v1_chips(v1_root: Path, output_dir: Path) -> dict[str, int]:
    """Symlink existing v1 chips into v2 output. Returns counts per split/class."""
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        for cls in ("pv", "non_pv"):
            src_dir = v1_root / split / cls
            dst_dir = output_dir / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            for src in src_dir.glob("*.png"):
                dst = dst_dir / src.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
                n += 1
            counts[f"{split}/{cls}"] = n
            print(f"  Linked v1 {split}/{cls}: {n} chips")
    return counts


def link_audit_chips(
    df_split: pd.DataFrame,
    cascade_root: Path,
    output_dir: Path,
    split: str,
) -> dict[str, int]:
    """Symlink audit-bucket chips into v2 output. Disambiguates filenames
    so v1 chip names (e.g. region_grid_predN_source.png) cannot collide.
    """
    counts: dict[str, int] = {"pv": 0, "non_pv": 0}
    for cls in ("pv", "non_pv"):
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for row in df_split.itertuples(index=False):
        src = cascade_root / row.chip_path
        if not src.exists():
            print(f"  WARN: missing source chip {src}")
            continue
        # Audit chips already have unique chip_id like v3c_G0772_p0000;
        # prefix with "jhbcbd_" so they cannot collide with v1 filenames.
        dst_name = f"jhbcbd_{row.chip_id}__{row.human_label}.png"
        dst = output_dir / split / row.label / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        counts[row.label] += 1

    print(f"  Linked audit {split}: pv={counts['pv']}, non_pv={counts['non_pv']}")
    return counts


def write_subtype_labels_csv(train_df: pd.DataFrame, val_df: pd.DataFrame,
                              output_dir: Path) -> Path:
    """Write a chip_id → subtype map for per-subtype reporting in train_cls.py."""
    rows = []
    for split, df in (("train", train_df), ("val", val_df)):
        for r in df.itertuples(index=False):
            rows.append({
                "chip_filename": f"jhbcbd_{r.chip_id}__{r.human_label}.png",
                "split": split,
                "label": r.label,
                "subtype": r.human_label,
                "grid_id": r.grid_id,
                "detector": r.detector,
                "source_bucket": NEW_BUCKET,
            })
    out = pd.DataFrame(rows)
    path = output_dir / "subtype_labels.csv"
    out.to_csv(path, index=False)
    return path


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    v1_counts: dict[str, int],
    audit_train_counts: dict[str, int],
    audit_val_counts: dict[str, int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    chosen_seed: int,
) -> Path:
    def subtype_breakdown(df: pd.DataFrame) -> dict:
        return {
            sub: {
                "n": int(grp_size),
                "by_class": df[df["human_label"] == sub]["label"].value_counts().to_dict(),
                "by_detector": df[df["human_label"] == sub]["detector"].value_counts().to_dict(),
                "n_grids": int(df[df["human_label"] == sub]["grid_id"].nunique()),
            }
            for sub, grp_size in df["human_label"].value_counts().items()
        }

    manifest = {
        "description": "cls_pv_thermal_v2 — multi-subtype FP suppressor "
                       "(CT v1 + JHB CBD GEID audit)",
        "built_by": "scripts/classifier/build_cls_dataset_v2.py",
        "spec_doc": "docs/experiments/exp_cls_dataset_protocol.md (v2 protocol)",
        "v1_root": str(args.v1_root),
        "cascade_root": str(args.cascade_root),
        "val_fraction_target": args.val_fraction,
        "min_subtype_coverage_n": args.min_subtype_n,
        "chosen_seed_for_jhb_split": chosen_seed,
        "buckets": {
            "ct_v1": {
                "source": f"{args.v1_root} (preserved splits)",
                "train": {
                    "pv": v1_counts["train/pv"],
                    "non_pv": v1_counts["train/non_pv"],
                    "total": v1_counts["train/pv"] + v1_counts["train/non_pv"],
                },
                "val": {
                    "pv": v1_counts["val/pv"],
                    "non_pv": v1_counts["val/non_pv"],
                    "total": v1_counts["val/pv"] + v1_counts["val/non_pv"],
                },
            },
            NEW_BUCKET: {
                "source": f"{args.cascade_root} (audited V3-C + propagated V4.2)",
                "class_flip_rule": f"human_label == '{PV_FLIP_LABEL}' → pv",
                "split_policy": "subtype-stratified, grid-grouped, no leakage, "
                                "≥1 chip per subtype with N >= min_subtype_n",
                "train": {
                    "pv": audit_train_counts["pv"],
                    "non_pv": audit_train_counts["non_pv"],
                    "total": audit_train_counts["pv"] + audit_train_counts["non_pv"],
                    "n_grids": int(train_df["grid_id"].nunique()),
                    "by_subtype": subtype_breakdown(train_df),
                },
                "val": {
                    "pv": audit_val_counts["pv"],
                    "non_pv": audit_val_counts["non_pv"],
                    "total": audit_val_counts["pv"] + audit_val_counts["non_pv"],
                    "n_grids": int(val_df["grid_id"].nunique()),
                    "by_subtype": subtype_breakdown(val_df),
                },
            },
        },
        "totals": {
            "train": {
                "pv": v1_counts["train/pv"] + audit_train_counts["pv"],
                "non_pv": v1_counts["train/non_pv"] + audit_train_counts["non_pv"],
            },
            "val": {
                "pv": v1_counts["val/pv"] + audit_val_counts["pv"],
                "non_pv": v1_counts["val/non_pv"] + audit_val_counts["non_pv"],
            },
        },
    }
    path = output_dir / "dataset_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1-root", type=Path,
                   default=PROJECT_ROOT / "data" / "cls_pv_thermal_v1")
    p.add_argument("--cascade-root", type=Path,
                   default=PROJECT_ROOT / "data" / "cls_pv_nonpv_v3c_v42_cascade")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "cls_pv_thermal_v2")
    p.add_argument("--val-fraction", type=float, default=0.27,
                   help="Target val fraction for new JHB CBD bucket (CT v1 splits preserved)")
    p.add_argument("--min-subtype-n", type=int, default=6,
                   help="Subtypes with total count >= this must appear in val")
    p.add_argument("--max-seeds", type=int, default=200)
    args = p.parse_args()

    if not args.v1_root.exists():
        print(f"ERROR: v1 root not found: {args.v1_root}")
        return 1
    if not args.cascade_root.exists():
        print(f"ERROR: cascade root not found: {args.cascade_root}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Load JHB CBD audit chips with subtype labels")
    audit_df = load_audit_chips(args.cascade_root)

    print(f"\n[2/4] Subtype-stratified grid split (val~{args.val_fraction:.2f})")
    train_df, val_df, seed = stratified_grid_split(
        audit_df, val_fraction=args.val_fraction,
        min_subtype_n=args.min_subtype_n, max_seeds=args.max_seeds,
    )

    print("\n[3/4] Symlink chips into v2 output")
    v1_counts = link_v1_chips(args.v1_root, args.output_dir)
    audit_train_counts = link_audit_chips(train_df, args.cascade_root, args.output_dir, "train")
    audit_val_counts = link_audit_chips(val_df, args.cascade_root, args.output_dir, "val")

    print("\n[4/4] Write manifest + subtype labels CSV")
    subtype_path = write_subtype_labels_csv(train_df, val_df, args.output_dir)
    manifest_path = write_manifest(
        args.output_dir, args, v1_counts,
        audit_train_counts, audit_val_counts,
        train_df, val_df, seed,
    )

    print("\n=== v2 dataset ready ===")
    print(f"  Output: {args.output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Subtype labels: {subtype_path}")
    print(f"  Train: pv={v1_counts['train/pv'] + audit_train_counts['pv']}, "
          f"non_pv={v1_counts['train/non_pv'] + audit_train_counts['non_pv']}")
    print(f"  Val:   pv={v1_counts['val/pv'] + audit_val_counts['pv']}, "
          f"non_pv={v1_counts['val/non_pv'] + audit_val_counts['non_pv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
