"""
Propagate V3-C-side subtype labels to V4.2-side via per-grid spatial pairing.

V3-C ∩ V4.2 共同 nonpv 核心 (source_detector=both, label=nonpv) 包含：
- 462 V3-C polygons (已人工审核)
- 441 V4.2 polygons (未审核)

V3-C 和 V4.2 通过相同的 IoU≥0.3 或 50% 包含规则配对得到 source_detector=both
标记。这个脚本把 V3-C 的 human_label 沿这个配对关系传给 V4.2 polygon。

每个 V4.2 polygon 找它最大重叠的 V3-C 邻居（同 grid 内），继承该 V3-C 的
subtype。多 V3-C 邻居时取 IoU 最大者。

Output:
- v4_2 propagation CSV (chip_id / human_label / propagated_from / pair_iou)
- merged labeled CSV (462 v3c + 441 v4_2)

Usage:
    python scripts/classifier/propagate_subtype_to_v42.py \\
        --pool-dir data/cls_pv_nonpv_v3c_v42_cascade \\
        --labeled-csv data/cls_pv_nonpv_v3c_v42_cascade/labeler/v3c__both/nonpv_subtype_labeled.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--labeled-csv", type=Path, required=True)
    p.add_argument("--metric-crs", default="EPSG:32735")
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--contain-threshold", type=float, default=0.5)
    args = p.parse_args()

    print(f"[1/3] Read manifest + V3-C labels")
    manifest = gpd.read_file(args.pool_dir / "manifest.gpkg")
    if str(manifest.crs) != args.metric_crs:
        manifest = manifest.to_crs(args.metric_crs)
    labels = pd.read_csv(args.labeled_csv)

    v3c_label_by_chip = dict(zip(labels["chip_id"], labels["human_label"]))

    v3c = manifest[
        (manifest["detector"] == "v3c")
        & (manifest["label"] == "nonpv")
        & (manifest["source_detector"] == "both")
    ].copy()
    v4_2 = manifest[
        (manifest["detector"] == "v4_2")
        & (manifest["label"] == "nonpv")
        & (manifest["source_detector"] == "both")
    ].copy()
    v3c["human_label"] = v3c["chip_id"].map(v3c_label_by_chip)
    print(f"  V3-C nonpv·both: {len(v3c)} (labeled: {v3c['human_label'].notna().sum()})")
    print(f"  V4.2 nonpv·both: {len(v4_2)} (to label)")

    print(f"\n[2/3] Pair per-grid (IoU≥{args.iou_threshold} or {args.contain_threshold:.0%} contain)")
    v4_2_rows = []
    multi_match_count = 0

    for grid in sorted(set(v4_2["grid_id"])):
        v3c_g = v3c[v3c["grid_id"] == grid].reset_index(drop=True)
        v4_2_g = v4_2[v4_2["grid_id"] == grid].reset_index(drop=True)
        if len(v3c_g) == 0 or len(v4_2_g) == 0:
            continue
        sj = gpd.sjoin(
            v4_2_g, v3c_g[["geometry"]], how="inner", predicate="intersects"
        )
        # For each V4.2 row, pick best V3-C neighbour by IoU
        best_for_v42: dict[int, dict] = {}
        for _, row in sj.iterrows():
            i = int(row.name)
            j = int(row["index_right"])
            ag = v4_2_g.geometry.iloc[i]
            bg = v3c_g.geometry.iloc[j]
            if not ag.intersects(bg):
                continue
            inter = ag.intersection(bg).area
            if inter <= 0:
                continue
            union = ag.area + bg.area - inter
            iou = inter / union if union > 0 else 0.0
            if (
                iou < args.iou_threshold
                and inter / ag.area < args.contain_threshold
                and inter / bg.area < args.contain_threshold
            ):
                continue
            cur = best_for_v42.get(i)
            if cur is None or iou > cur["iou"]:
                best_for_v42[i] = {
                    "i": i, "j": j, "iou": iou,
                    "v3c_chip_id": v3c_g.iloc[j]["chip_id"],
                    "v3c_label": v3c_g.iloc[j]["human_label"],
                }

        # Track multi-match counts (one V4.2 → many V3-C neighbours)
        per_v42 = sj.groupby(sj.index).size()
        multi_match_count += int((per_v42 > 1).sum())

        for i, info in best_for_v42.items():
            v4_2_row = v4_2_g.iloc[i]
            v4_2_rows.append({
                "chip_id": v4_2_row["chip_id"],
                "detector": "v4_2",
                "source_detector": "both",
                "grid_id": grid,
                "pred_idx": int(v4_2_row["pred_idx"]),
                "iou_to_gt": float(v4_2_row["iou_to_gt"]),
                "area_m2": float(v4_2_row["area_m2"]),
                "human_label": info["v3c_label"],
                "propagated_from": info["v3c_chip_id"],
                "pair_iou": round(info["iou"], 4),
            })

    propagated = pd.DataFrame(v4_2_rows)
    n_unmatched = len(v4_2) - len(propagated)
    print(f"  Propagated: {len(propagated)}/{len(v4_2)}")
    print(f"  V4.2 polygons with multiple V3-C neighbours: {multi_match_count} (kept best-IoU)")
    if n_unmatched:
        print(f"  [WARN] {n_unmatched} V4.2 polygons in 'both' pool have no IoU≥{args.iou_threshold} V3-C neighbour")

    print(f"\n  Subtype distribution after propagation (V4.2-side):")
    print(propagated["human_label"].value_counts(dropna=False).to_string())

    # Persist V4.2 labels
    out_v42 = args.labeled_csv.parent / "nonpv_subtype_labeled_v4_2.csv"
    propagated.to_csv(out_v42, index=False)
    print(f"\n  Wrote {out_v42}")

    # Merge with V3-C labels for a unified labeled view
    v3c_lab = labels.copy()
    v3c_lab["propagated_from"] = ""
    v3c_lab["pair_iou"] = 1.0  # self
    union = pd.concat([v3c_lab, propagated], ignore_index=True)
    out_union = args.labeled_csv.parent / "nonpv_subtype_labeled_union.csv"
    union.to_csv(out_union, index=False)
    print(f"  Wrote {out_union} ({len(union)} rows)")

    print(f"\n[3/3] Combined V3-C + V4.2 subtype distribution (n={len(union)}):")
    print(union["human_label"].value_counts(dropna=False).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
