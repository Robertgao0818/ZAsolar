"""Build the v4_2_jhb_ft COCO dataset.

Composition:
  - 25 JHB CBD batch1 grids (G0772..G0926)
  - 25 JHB Sandton batch2 grids (G1110..G1254)
  - 25 CT v4_base subsample (panel-count stratified, 3 strata, seed=42)

Val split (whole-grid, no tile leakage):
  - 5 CBD + 5 Sandton stratified by polygon count → 10 val grids
  - All other 65 grids go to train

Single build_base_coco() call; whole-grid split is driven by val_grids.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.annotation_loader import discover_annotations, load_annotation_gdf  # noqa: E402
from export_coco_dataset import build_base_coco  # noqa: E402

CBD_BATCH1 = [
    "G0772", "G0773", "G0774", "G0775", "G0776",
    "G0814", "G0815", "G0816", "G0817", "G0818",
    "G0853", "G0854", "G0855", "G0856", "G0857",
    "G0888", "G0889", "G0890", "G0891", "G0892",
    "G0922", "G0923", "G0924", "G0925", "G0926",
]
SANDTON_BATCH2 = [
    "G1110", "G1111", "G1112", "G1113", "G1114",
    "G1144", "G1145", "G1146", "G1147", "G1148",
    "G1179", "G1180", "G1181", "G1182", "G1183",
    "G1214", "G1215", "G1216", "G1217", "G1218",
    "G1250", "G1251", "G1252", "G1253", "G1254",
]


def grid_polygon_counts(grid_ids: list[str], region: str) -> dict[str, int]:
    entries = discover_annotations(regions=[region])
    missing = [gid for gid in grid_ids if gid not in entries]
    if missing:
        raise ValueError(
            f"{len(missing)} grids not in {region} annotation registry: "
            f"{missing}. Check configs/datasets/regions.yaml or annotation files."
        )
    counts = {}
    for gid in grid_ids:
        gdf = load_annotation_gdf(entries[gid])
        counts[gid] = len(gdf)
    return counts


def stratified_sample(counts: dict[str, int], n: int, seed: int) -> list[str]:
    """Sample n grids stratified by polygon count (3 terciles)."""
    rng = random.Random(seed)
    sorted_grids = sorted(counts.items(), key=lambda kv: kv[1])
    n_total = len(sorted_grids)
    if n >= n_total:
        return sorted([g for g, _ in sorted_grids])

    # Tercile boundaries
    t1_end = n_total // 3
    t2_end = 2 * n_total // 3
    strata = [
        sorted_grids[:t1_end],
        sorted_grids[t1_end:t2_end],
        sorted_grids[t2_end:],
    ]

    # Per-stratum sample size (proportional)
    per_stratum = []
    remaining = n
    for i, stratum in enumerate(strata):
        if i < len(strata) - 1:
            k = round(n * len(stratum) / n_total)
            k = min(k, len(stratum), remaining)
        else:
            k = min(remaining, len(stratum))
        per_stratum.append(k)
        remaining -= k

    sampled = []
    for k, stratum in zip(per_stratum, strata):
        if k > 0:
            sampled.extend(rng.sample([g for g, _ in stratum], k))
    return sorted(sampled)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="/mnt/d/ZAsolar/coco_v4_2_jhb_ft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ct-subsample-n", type=int, default=25)
    ap.add_argument("--val-cbd-n", type=int, default=5)
    ap.add_argument("--val-sandton-n", type=int, default=5)
    ap.add_argument("--neg-ratio", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print grid lists and exit without building COCO")
    args = ap.parse_args()

    print("=== Step 1/3: CT panel-count stratified subsample ===")
    jhb_target_ids = set(CBD_BATCH1 + SANDTON_BATCH2)
    ct_entries = discover_annotations(regions=["cape_town"])
    ct_pool = sorted(g for g in ct_entries if g not in jhb_target_ids)
    excluded_overlap = sorted(set(ct_entries) - set(ct_pool))
    print(f"  CT grids discovered: {len(ct_entries)}")
    if excluded_overlap:
        print(f"  Excluded CT grids overlapping with JHB target: {excluded_overlap}")
    print(f"  CT pool after dedup: {len(ct_pool)}")
    ct_counts = grid_polygon_counts(ct_pool, region="cape_town")
    print(f"  CT grids with annotations loaded: {len(ct_counts)}")
    ct_sampled = stratified_sample(ct_counts, args.ct_subsample_n, args.seed)
    print(f"  CT subsampled ({len(ct_sampled)}): {ct_sampled}")
    if len(ct_sampled) != args.ct_subsample_n:
        raise ValueError(
            f"CT stratified sample returned {len(ct_sampled)} grids, "
            f"expected {args.ct_subsample_n}. CT pool size = {len(ct_counts)}."
        )

    print("\n=== Step 2/3: JHB val stratified split ===")
    cbd_counts = grid_polygon_counts(CBD_BATCH1, region="johannesburg")
    sandton_counts = grid_polygon_counts(SANDTON_BATCH2, region="johannesburg")
    cbd_val = stratified_sample(cbd_counts, args.val_cbd_n, args.seed)
    sandton_val = stratified_sample(sandton_counts, args.val_sandton_n, args.seed + 1)
    val_grids = sorted(cbd_val + sandton_val)
    print(f"  CBD val ({len(cbd_val)}): {cbd_val}")
    print(f"  Sandton val ({len(sandton_val)}): {sandton_val}")
    if len(cbd_val) != args.val_cbd_n:
        raise ValueError(
            f"CBD val sample returned {len(cbd_val)} grids, expected {args.val_cbd_n}."
        )
    if len(sandton_val) != args.val_sandton_n:
        raise ValueError(
            f"Sandton val sample returned {len(sandton_val)} grids, "
            f"expected {args.val_sandton_n}."
        )

    include_grids = sorted(ct_sampled + CBD_BATCH1 + SANDTON_BATCH2)
    train_grids = sorted(set(include_grids) - set(val_grids))
    print(f"\nTotal include_grids: {len(include_grids)}  (CT={len(ct_sampled)}, CBD={len(CBD_BATCH1)}, Sandton={len(SANDTON_BATCH2)})")
    print(f"Train grids: {len(train_grids)}")
    print(f"Val grids:   {len(val_grids)}")
    val_not_in_include = set(val_grids) - set(include_grids)
    if val_not_in_include:
        raise ValueError(
            f"Val grids not in include_grids: {sorted(val_not_in_include)}. "
            f"This would silently drop val grids from the dataset."
        )

    if args.dry_run:
        print("\n[dry-run] Skipping COCO build.")
        return

    print("\n=== Step 3/3: Build COCO ===")
    params = {
        # JHB first so JHB grids take precedence on grid-ID collisions
        # (e.g. G0854/G0855 exist in both Capetown/ and Joburg/ but represent
        # different physical locations).
        "regions": ["johannesburg", "cape_town"],
        "output_dir": args.output_dir,
        "chip_size": 400,
        "overlap": 0.25,
        "val_fraction": 0.2,    # ignored when val_grids is set
        "seed": args.seed,
        "no_balance": False,
        "manifest": None,
        "tier_filter": "T1+T2",
        "category_name": "solar_panel",
        "neg_ratio": args.neg_ratio,
        "exclude_grids": None,
        "include_grids": include_grids,
        "val_grids": val_grids,
        "audit_csv": None,
        "exclude_audit_labels": ["heater_or_non_pv", "uncertain"],
    }
    result = build_base_coco(params)
    print(f"\n[done] output_dir={result.get('output_dir')}")
    for k in ("train_images", "val_images", "train_annotations", "val_annotations"):
        if k in result:
            print(f"  {k}: {result[k]}")


if __name__ == "__main__":
    main()
