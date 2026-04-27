"""Audit non-PV subtype distribution in the classifier training pool.

Diagnostic: quantifies how much of the reviewed `delete` pool has a
known subtype (heater / skylight / shadow / pergola) versus "unknown"
(reviewed delete without taxonomy coverage). Run before tuning non-PV
sampling strategy in `scripts/classifier/build_cls_dataset.py`.

For each reviewed non-PV chip (area < cutoff), attempts to join a
taxonomy subtype label by spatial proximity (same grid, centroid
within `--match-threshold-m`). Rows without a match are labeled
"unknown" and a sample is rendered for visual inspection.

Outputs:
  results/analysis/classifier_nonpv_audit/<run_id>/
    summary.json
    summary.md
    annotated_nonpv.csv
    unknown_samples/<bucket>/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.classifier.build_cls_dataset import (  # noqa: E402
    LABEL_NONPV,
    PROJECT_ROOT,
    discover_grid_sources,
    extract_chip,
    load_reviewed_predictions,
)

AUDIT_ROOT = PROJECT_ROOT / "results" / "analysis" / "classifier_nonpv_audit"
DEFAULT_TAXONOMY = (
    PROJECT_ROOT / "results" / "analysis" / "small_fp" / "taxonomy_run"
    / "small_fp_taxonomy_labeled.csv"
)


def _meters_between(lon1, lat1, lon2, lat2):
    """Haversine distance in meters. Works scalar-to-array via broadcasting."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_taxonomy_with_centroids(taxonomy_csv: Path, source_lookup: dict) -> pd.DataFrame:
    """Read taxonomy CSV and resolve (lon, lat) via each grid's predictions_metric.gpkg.

    Reads each grid's gpkg at most once; faster than the per-row helper in
    build_cls_dataset when auditing the full taxonomy set.
    """
    df = pd.read_csv(taxonomy_csv)
    enriched: list[dict] = []
    missing_grids: list[str] = []

    for grid_id, part in df.groupby("grid_id"):
        src = source_lookup.get(grid_id)
        if src is None:
            missing_grids.append(grid_id)
            continue
        pred_path = src.results_path / grid_id / "predictions_metric.gpkg"
        if not pred_path.exists():
            missing_grids.append(grid_id)
            continue
        try:
            preds = gpd.read_file(pred_path)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: failed to read {pred_path}: {e}")
            continue
        preds_4326 = preds.to_crs(epsg=4326) if preds.crs and preds.crs.to_epsg() != 4326 else preds
        for _, row in part.iterrows():
            pred_id = int(row["pred_id"])
            if pred_id >= len(preds_4326):
                continue
            centroid = preds_4326.iloc[pred_id].geometry.centroid
            enriched.append({
                "grid_id": grid_id,
                "pred_id": pred_id,
                "human_label": row["human_label"],
                "area_m2": float(row.get("area_m2", 0) or 0),
                "confidence": float(row.get("confidence", 0) or 0),
                "centroid_lon": centroid.x,
                "centroid_lat": centroid.y,
            })

    if missing_grids:
        print(f"  INFO: skipped {len(missing_grids)} taxonomy grids "
              f"(no predictions_metric.gpkg under registered model_run)")
    return pd.DataFrame(enriched)


def attach_subtype(
    nonpv_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    match_threshold_m: float,
) -> pd.DataFrame:
    """For each reviewed non-PV row, find nearest taxonomy row in same grid.

    If the nearest neighbor is within threshold, attach its `human_label`
    as subtype; else mark "unknown".
    """
    out = nonpv_df.reset_index(drop=True).copy()
    subtypes = ["unknown"] * len(out)
    dists = [np.nan] * len(out)

    tax_by_grid = {g: d.reset_index(drop=True) for g, d in taxonomy_df.groupby("grid_id")}

    for i, r in out.iterrows():
        tx = tax_by_grid.get(r["grid_id"])
        if tx is None or len(tx) == 0:
            continue
        d = _meters_between(
            r["centroid_lon"], r["centroid_lat"],
            tx["centroid_lon"].to_numpy(), tx["centroid_lat"].to_numpy(),
        )
        j = int(np.argmin(d))
        if d[j] <= match_threshold_m:
            subtypes[i] = tx.iloc[j]["human_label"]
            dists[i] = float(d[j])

    out["subtype"] = subtypes
    out["subtype_match_dist_m"] = dists
    return out


def crosstab(df: pd.DataFrame) -> list[dict]:
    buckets = sorted(df["source_bucket"].unique())
    subs = sorted(df["subtype"].unique().tolist())
    rows = []
    for b in buckets:
        part = df[df["source_bucket"] == b]
        total = int(len(part))
        entry: dict = {"source_bucket": b, "total": total}
        for s in subs:
            n = int((part["subtype"] == s).sum())
            entry[s] = n
            entry[f"{s}_pct"] = round(100.0 * n / total, 1) if total else 0.0
        rows.append(entry)
    return rows


def sample_unknowns(
    df: pd.DataFrame,
    output_dir: Path,
    tiles_root: Path | None,
    per_bucket: int,
    seed: int,
) -> dict[str, list[str]]:
    unknown = df[df["subtype"] == "unknown"]
    sampled: dict[str, list[str]] = defaultdict(list)
    tile_cache: dict = {}

    try:
        for b, part in unknown.groupby("source_bucket"):
            n = min(per_bucket, len(part))
            if n == 0:
                continue
            picks = part.sample(n=n, random_state=seed)
            bucket_slug = b.replace(":", "__")
            bucket_dir = output_dir / "unknown_samples" / bucket_slug
            bucket_dir.mkdir(parents=True, exist_ok=True)
            for _, r in picks.iterrows():
                chip = extract_chip(
                    r["centroid_lon"], r["centroid_lat"],
                    r["grid_id"], r["region"],
                    tiles_root, tile_cache,
                )
                if chip is None:
                    continue
                fname = f"{r['region']}_{r['grid_id']}_pred{int(r['pred_id'])}.png"
                out_path = bucket_dir / fname
                cv2.imwrite(str(out_path), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
                sampled[b].append(fname)
    finally:
        for h in tile_cache.values():
            h.close()

    return dict(sampled)


def format_md(
    run_id: str,
    area_cutoff: float,
    match_threshold_m: float,
    taxonomy_total: int,
    rows: list[dict],
    sampled: dict[str, list[str]],
) -> str:
    lines = [
        f"# Classifier non-PV Subtype Audit — {run_id}",
        "",
        "Diagnostic for non-PV class heterogeneity in the classifier training "
        "pool. Joins reviewed `delete` decisions to the small-FP taxonomy run "
        "by spatial proximity; rows without a match are flagged as `unknown` "
        "(no subtype label available, NOT a different FP type).",
        "",
        f"- `area_cutoff_m2`: {area_cutoff}",
        f"- Match threshold: {match_threshold_m} m (centroid distance)",
        f"- Taxonomy labeled rows with resolvable centroid: {taxonomy_total}",
        "",
        "## Per-bucket subtype breakdown (non-PV pool)",
        "",
    ]
    if not rows:
        lines.append("_no non-PV rows loaded._")
        return "\n".join(lines) + "\n"

    subs = sorted({k for r in rows for k in r if k not in ("source_bucket", "total")
                   and not k.endswith("_pct")})
    header = "| bucket | total | " + " | ".join(subs) + " |"
    sep = "|---|---|" + "---|" * len(subs)
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = [r["source_bucket"], str(r["total"])]
        for s in subs:
            cells.append(f"{r.get(s, 0)} ({r.get(s + '_pct', 0.0)}%)")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Unknown samples (for visual inspection)")
    lines.append("")
    if sampled:
        for b, files in sorted(sampled.items()):
            slug = b.replace(":", "__")
            lines.append(f"- `{b}`: {len(files)} chips in `unknown_samples/{slug}/`")
    else:
        lines.append("_no unknown samples rendered._")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", default=date.today().isoformat())
    p.add_argument("--area-cutoff", type=float, default=30.0)
    p.add_argument("--taxonomy-csv", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--match-threshold-m", type=float, default=5.0)
    p.add_argument("--per-bucket-samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tiles-root", type=Path, default=None,
                   help="Override tile root (else per-grid via region registry)")
    p.add_argument("--output-root", type=Path, default=AUDIT_ROOT)
    p.add_argument("--skip-chip-extraction", action="store_true",
                   help="Skip rendering unknown sample chips (faster dry run)")
    args = p.parse_args()

    out_dir = args.output_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Discovering reviewed grids via region_registry...")
    sources = discover_grid_sources(include_deprecated=False, include_legacy_flat=True)
    by_grid: dict = {}
    for s in sources:
        by_grid.setdefault(s.grid_id, s)
    print(f"  {len(sources)} (region, model_run, grid) tuples")

    print(f"\n[2/4] Loading reviewed non-PV pool (area < {args.area_cutoff})...")
    reviewed = load_reviewed_predictions(sources, area_cutoff=args.area_cutoff)
    nonpv = reviewed[reviewed["label"] == LABEL_NONPV].reset_index(drop=True)
    print(f"  non-PV reviewed: {len(nonpv)} "
          f"across {nonpv['source_bucket'].nunique()} buckets / "
          f"{nonpv['grid_id'].nunique()} grids")

    print(f"\n[3/4] Loading taxonomy and spatial-joining subtypes...")
    taxonomy = load_taxonomy_with_centroids(args.taxonomy_csv, by_grid)
    print(f"  taxonomy rows with resolved centroids: {len(taxonomy)}")
    annotated = attach_subtype(nonpv, taxonomy, args.match_threshold_m)

    matched = (annotated["subtype"] != "unknown").sum()
    print(f"  matched (≤{args.match_threshold_m}m): {matched} / {len(annotated)} "
          f"({100.0 * matched / max(1, len(annotated)):.1f}%)")
    rows = crosstab(annotated)

    sampled: dict[str, list[str]] = {}
    if not args.skip_chip_extraction:
        print(f"\n[4/4] Sampling unknowns and rendering chips "
              f"(per_bucket={args.per_bucket_samples})...")
        sampled = sample_unknowns(
            annotated, out_dir, args.tiles_root,
            per_bucket=args.per_bucket_samples, seed=args.seed,
        )
        for b, files in sampled.items():
            print(f"  {b}: rendered {len(files)} chips")
    else:
        print(f"\n[4/4] Skipping chip extraction (--skip-chip-extraction)")

    summary = {
        "run_id": args.run_id,
        "area_cutoff_m2": args.area_cutoff,
        "match_threshold_m": args.match_threshold_m,
        "per_bucket_samples_target": args.per_bucket_samples,
        "taxonomy_rows_total": int(len(taxonomy)),
        "reviewed_nonpv_total": int(len(nonpv)),
        "reviewed_nonpv_matched": int(matched),
        "per_bucket": rows,
        "unknown_samples_rendered_per_bucket": {b: len(v) for b, v in sampled.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "summary.md").write_text(format_md(
        args.run_id, args.area_cutoff, args.match_threshold_m,
        len(taxonomy), rows, sampled,
    ))
    annotated.to_csv(out_dir / "annotated_nonpv.csv", index=False)

    print(f"\nWrote:")
    print(f"  {out_dir / 'summary.md'}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'annotated_nonpv.csv'}")
    if sampled:
        print(f"  {out_dir / 'unknown_samples'}/<bucket>/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
