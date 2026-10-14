#!/usr/bin/env python3
"""V1.4 Channel 1 stratified-precision sampler.

For each grid in a model_run's results, draw a stratified random sample of
predictions across (size_bucket × confidence_bin) cells. Saves per-grid
sampled predictions in a parallel results tree compatible with
`review_detections.py --predictions-dir`.

After review, compute stratified precision estimates with proper Horvitz-
Thompson weights to recover the unbiased grid-level / pooled precision.

Defaults target the JHB CBD 25-grid V3-C-on-Vexcel run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024_ch1_sample"

# Stratification cells (size m² × confidence).
# V3-C-on-Vexcel confidence distribution is heavily skewed (median ≈ 0.97), so
# the "low_conf" cut sits near the median to keep both conf strata populated.
SIZE_EDGES = [0, 30, 100, 1e9]                   # 3 buckets: <30 / 30-100 / ≥100
SIZE_LABELS = ["s_xs", "s_md", "s_lg"]
CONF_EDGES = [0, 0.95, 1.0001]                   # split near median
CONF_LABELS = ["c_lo", "c_hi"]


def stratify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["size_bucket"] = pd.cut(df["area_m2"], bins=SIZE_EDGES, labels=SIZE_LABELS, include_lowest=True)
    df["conf_bucket"] = pd.cut(df["confidence"], bins=CONF_EDGES, labels=CONF_LABELS, include_lowest=True)
    df["stratum"] = df["size_bucket"].astype(str) + "_" + df["conf_bucket"].astype(str)
    return df


def sample_grid(pred_gdf: gpd.GeoDataFrame, per_stratum: int, rng: random.Random) -> gpd.GeoDataFrame:
    pred_gdf = stratify(pred_gdf)
    out = []
    for stratum, grp in pred_gdf.groupby("stratum", observed=True):
        n_pop = len(grp)
        n_take = min(per_stratum, n_pop)
        idx = rng.sample(grp.index.tolist(), n_take)
        sub = grp.loc[idx].copy()
        sub["stratum_pop"] = n_pop          # for HT weighting later
        sub["stratum_sampled"] = n_take
        sub["sample_weight"] = n_pop / n_take
        out.append(sub)
    if not out:
        return pred_gdf.iloc[0:0]
    return gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=pred_gdf.crs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                   help="Source model_run dir with <grid>/predictions_metric.gpkg")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Output dir for sampled predictions (parallel tree)")
    p.add_argument("--per-stratum", type=int, default=5,
                   help="Sample size per (size×conf) stratum per grid; 4 strata × 5 = 20/grid max")
    p.add_argument("--grid-budget", type=int, default=None,
                   help="Optional total cap per grid; trims after stratified sampling")
    p.add_argument("--grids", nargs="*", default=None,
                   help="Restrict to these grid IDs (default: all in results-root)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    grid_dirs = sorted(d for d in args.results_root.iterdir() if d.is_dir() and d.name.startswith("G"))
    if args.grids:
        grid_dirs = [d for d in grid_dirs if d.name in set(args.grids)]

    summary_rows = []
    for gdir in grid_dirs:
        gid = gdir.name
        src = gdir / "predictions_metric.gpkg"
        if not src.exists():
            print(f"[SKIP] {gid}: no predictions_metric.gpkg")
            continue
        gdf = gpd.read_file(src)
        if len(gdf) == 0:
            print(f"[SKIP] {gid}: 0 predictions")
            continue
        sample = sample_grid(gdf, args.per_stratum, rng)
        if args.grid_budget and len(sample) > args.grid_budget:
            sample = sample.sample(args.grid_budget, random_state=args.seed)

        out_dir = args.output_root / gid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions_metric.gpkg"
        sample.to_file(out_path, driver="GPKG")

        # Carry over the original config.json so review tool / downstream
        # tools see the right metadata
        src_cfg = gdir / "config.json"
        if src_cfg.exists():
            (out_dir / "config.json").write_text(src_cfg.read_text())

        cell_counts = {f"{s}_{c}": int(((sample.size_bucket==s)&(sample.conf_bucket==c)).sum())
                       for s in SIZE_LABELS for c in CONF_LABELS}
        summary_rows.append({
            "grid": gid,
            "n_pred_total": len(gdf),
            "n_sampled": len(sample),
            **cell_counts,
        })
        cells_str = " ".join(f"{k}={v}" for k,v in cell_counts.items())
        print(f"[{gid}] sampled {len(sample)}/{len(gdf)}  {cells_str}")

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_root / "ch1_sample_manifest.csv"
    summary.to_csv(summary_path, index=False)
    meta = {
        "source_results_root": str(args.results_root),
        "per_stratum_target": args.per_stratum,
        "size_edges_m2": SIZE_EDGES,
        "conf_edges": CONF_EDGES,
        "seed": args.seed,
        "n_grids": len(summary),
        "n_total_sampled": int(summary.n_sampled.sum()) if len(summary) else 0,
        "n_total_pop": int(summary.n_pred_total.sum()) if len(summary) else 0,
    }
    (args.output_root / "ch1_sample_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{summary_path}: {len(summary)} grids, {meta['n_total_sampled']}/{meta['n_total_pop']} sampled")
    print(f"Output root: {args.output_root}")
    print(f"\nLaunch review with:\n"
          f"  python scripts/annotations/review_detections.py \\\n"
          f"    --grid-id {' '.join(summary['grid'].tolist())} \\\n"
          f"    --region jhb --imagery-layer vexcel_2024 \\\n"
          f"    --predictions-dir {args.output_root.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
