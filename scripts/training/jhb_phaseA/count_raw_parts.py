"""Per-grid raw-parts inventory for the JHB Phase A training pool.

Reads ``results/johannesburg/v3c_vexcel_2024_ch1_sample/<grid>/review/`` directly
(no dissolve, no clean_gt). Outputs a CSV with the four source counts that drive
mask supervision strategy:

  V3C_correct    -> mask_weight = 0 (det+cls only, halo not into mask BCE)
  V3C_edit       -> drop (replaced by sam_added)
  V3C_delete     -> hard negative (box+cls, no mask)
  sam_added      -> mask_weight = 1.0 + boundary_ignore_px band

Usage:
    python scripts/training/jhb_phaseA/count_raw_parts.py
    -> writes results/analysis/jhb_phaseA_prep/raw_parts_inventory.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REVIEW_ROOT = PROJECT_ROOT / "results/johannesburg/v3c_vexcel_2024_ch1_sample"
OUT_CSV = PROJECT_ROOT / "results/analysis/jhb_phaseA_prep/raw_parts_inventory.csv"

TRAIN_GRIDS = ["G0772","G0773","G0774","G0775","G0814","G0815","G0818",
               "G0853","G0854","G0855","G0856","G0857","G0888","G0889",
               "G0890","G0892","G0922","G0923","G0924","G0926"]
VAL_GRIDS = ["G0816","G0817","G0925"]


def count_grid(grid: str, split: str) -> dict:
    rev_p = REVIEW_ROOT / grid / "review" / f"{grid}_reviewed.gpkg"
    sam_p = REVIEW_ROOT / grid / "review" / f"{grid}_sam_added.gpkg"
    if not rev_p.exists():
        print(f"[MISS] {grid}: {rev_p}", file=sys.stderr)
        return {}
    rev = gpd.read_file(rev_p)
    sam = gpd.read_file(sam_p) if sam_p.exists() else None
    return {
        "grid": grid,
        "split": split,
        "v3c_correct": int((rev.review_status == "correct").sum()),
        "v3c_edit": int((rev.review_status == "edit").sum()),
        "v3c_delete": int((rev.review_status == "delete").sum()),
        "sam_added": int(len(sam)) if sam is not None else 0,
    }


def main():
    rows = []
    for g in TRAIN_GRIDS:
        rows.append(count_grid(g, "train"))
    for g in VAL_GRIDS:
        rows.append(count_grid(g, "val"))
    df = pd.DataFrame([r for r in rows if r])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[SAVE] {OUT_CSV}")

    for split in ("train", "val"):
        sub = df[df.split == split]
        print(f"\n[{split.upper()}] n_grid={len(sub)}")
        for col in ("v3c_correct", "v3c_edit", "v3c_delete", "sam_added"):
            print(f"  {col:14s}: {int(sub[col].sum())}")
    total = {col: int(df[col].sum()) for col in ("v3c_correct","v3c_edit","v3c_delete","sam_added")}
    print(f"\n[ALL] {total}")


if __name__ == "__main__":
    main()
