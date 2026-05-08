#!/usr/bin/env python3
"""Probe whether sibling sub-array misses are NMS suppression victims.

Compares two raw_detections.pkl artifacts (one at default NMS, one at
relaxed NMS) against 补标 GT, classifying each GT polygon as:

- covered_by_baseline  : default NMS already detects it (≥ min_iou)
- nms_recoverable      : default misses, relaxed catches (NMS suppression)
- no_proposal          : neither catches it (proposal stage failure)

If `nms_recoverable` is a meaningful fraction of currently-missed GT,
relaxing NMS in production is worth it. Otherwise the bottleneck is
at the proposal/feature-extraction stage — needs training-time fixes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.inference.raw_artifact import read_artifact  # noqa: E402


def detections_to_gdf(artifact_path: Path, score_threshold: float) -> gpd.GeoDataFrame:
    """Project all chip detections into source-CRS bbox polygons."""
    art = read_artifact(artifact_path)
    rows = []
    crs = None
    for chip in art.chips:
        transform = Affine(*chip.source_transform)
        if crs is None:
            crs = chip.source_crs
        for det in chip.detections:
            if det.score < score_threshold:
                continue
            x1, y1, x2, y2 = det.box_source_xyxy
            ulx, uly = transform * (x1, y1)
            lrx, lry = transform * (x2, y2)
            poly = box(min(ulx, lrx), min(uly, lry), max(ulx, lrx), max(uly, lry))
            rows.append({
                "geometry": poly,
                "score": float(det.score),
                "chip_index": int(chip.chip_index),
            })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    print(f"  loaded {len(gdf)} detections (≥{score_threshold:.2f}) from {artifact_path.name} "
          f"(artifact NMS={art.nms_thresh})")
    return gdf


def post_hoc_nms(gdf: gpd.GeoDataFrame, iou_threshold: float) -> gpd.GeoDataFrame:
    """Greedy NMS by score on box polygons. iou_threshold close to 1.0 = effectively no NMS."""
    if len(gdf) == 0:
        return gdf
    gdf = gdf.sort_values("score", ascending=False).reset_index(drop=True)
    polygons = list(gdf.geometry)
    keep_idx = []
    for i in range(len(polygons)):
        keep = True
        pi = polygons[i]
        for j in keep_idx:
            pj = polygons[j]
            try:
                inter = pi.intersection(pj).area
                if inter == 0:
                    continue
                union = pi.area + pj.area - inter
                iou = inter / union if union > 0 else 0
                if iou > iou_threshold:
                    keep = False
                    break
            except Exception:
                continue
        if keep:
            keep_idx.append(i)
    return gdf.iloc[keep_idx].reset_index(drop=True)


def max_iou_vs_set(gt_poly, pred_gdf: gpd.GeoDataFrame) -> float:
    """Max IoU between gt_poly and any prediction polygon. 0 if pred set is empty."""
    if len(pred_gdf) == 0:
        return 0.0
    gt_buf = gt_poly.buffer(0)
    if not gt_buf.is_valid or gt_buf.is_empty:
        return 0.0
    candidates = pred_gdf.sindex.query(gt_buf, predicate="intersects")
    if len(candidates) == 0:
        return 0.0
    best = 0.0
    for cidx in candidates:
        pp = pred_gdf.geometry.iloc[cidx]
        if not pp.is_valid:
            pp = pp.buffer(0)
        try:
            inter = gt_buf.intersection(pp).area
            if inter == 0:
                continue
            union = gt_buf.area + pp.area - inter
            iou = inter / union if union > 0 else 0
            if iou > best:
                best = iou
        except Exception:
            continue
    return best


def stratify_by_area(area_m2: float) -> str:
    if area_m2 < 30:
        return "s_xs_lt30"
    if area_m2 < 100:
        return "s_md_30_100"
    return "s_lg_ge100"


def classify(
    gt_gdf: gpd.GeoDataFrame,
    baseline_gdf: gpd.GeoDataFrame,
    relaxed_gdf: gpd.GeoDataFrame,
    min_iou: float,
) -> pd.DataFrame:
    rows = []
    for idx, gt_row in gt_gdf.iterrows():
        gt_poly = gt_row.geometry
        baseline_iou = max_iou_vs_set(gt_poly, baseline_gdf)
        relaxed_iou = max_iou_vs_set(gt_poly, relaxed_gdf)
        if baseline_iou >= min_iou:
            cat = "covered_by_baseline"
        elif relaxed_iou >= min_iou:
            cat = "nms_recoverable"
        else:
            cat = "no_proposal"
        area = float(gt_row.get("area_m2") or gt_poly.area)
        rows.append({
            "gt_idx": int(idx),
            "clean_id": gt_row.get("clean_id"),
            "area_m2": area,
            "stratum": stratify_by_area(area),
            "best_baseline_iou": baseline_iou,
            "best_relaxed_iou": relaxed_iou,
            "category": cat,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    overall = df["category"].value_counts(normalize=True).reindex(
        ["covered_by_baseline", "nms_recoverable", "no_proposal"], fill_value=0.0
    )
    overall_n = df["category"].value_counts().reindex(
        ["covered_by_baseline", "nms_recoverable", "no_proposal"], fill_value=0
    )
    rows = [{
        "stratum": "ALL",
        "n_gt": len(df),
        "n_covered_by_baseline": int(overall_n["covered_by_baseline"]),
        "n_nms_recoverable": int(overall_n["nms_recoverable"]),
        "n_no_proposal": int(overall_n["no_proposal"]),
        "pct_covered_by_baseline": float(overall["covered_by_baseline"]),
        "pct_nms_recoverable": float(overall["nms_recoverable"]),
        "pct_no_proposal": float(overall["no_proposal"]),
    }]
    for stratum in ["s_xs_lt30", "s_md_30_100", "s_lg_ge100"]:
        sub = df[df["stratum"] == stratum]
        if len(sub) == 0:
            continue
        n = sub["category"].value_counts().reindex(
            ["covered_by_baseline", "nms_recoverable", "no_proposal"], fill_value=0
        )
        rows.append({
            "stratum": stratum,
            "n_gt": len(sub),
            "n_covered_by_baseline": int(n["covered_by_baseline"]),
            "n_nms_recoverable": int(n["nms_recoverable"]),
            "n_no_proposal": int(n["no_proposal"]),
            "pct_covered_by_baseline": int(n["covered_by_baseline"]) / len(sub),
            "pct_nms_recoverable": int(n["nms_recoverable"]) / len(sub),
            "pct_no_proposal": int(n["no_proposal"]) / len(sub),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-pkl", type=Path, required=True,
                   help="raw_detections.pkl from a NMS=0.5 run (current default)")
    p.add_argument("--relaxed-pkl", type=Path, required=True,
                   help="raw_detections.pkl from a NMS=0.99 run (relaxed probe). "
                        "May be the same file as baseline if it was run at 0.99 "
                        "and we derive baseline via post-hoc NMS — see --derive-baseline")
    p.add_argument("--gt-gpkg", type=Path, required=True,
                   help="Channel-2 clean_gt gpkg, e.g. data/annotations_channel2_clean/G0817/G0817_clean_gt.gpkg")
    p.add_argument("--score-threshold", type=float, default=0.3,
                   help="Score floor for predictions (mirror inference-time threshold). "
                        "0.3 matches the standard V3-C eval threshold.")
    p.add_argument("--min-iou", type=float, default=0.1,
                   help="GT-vs-prediction IoU floor for a 'covers' verdict")
    p.add_argument("--derive-baseline", action="store_true",
                   help="If set, ignore --baseline-pkl and derive baseline by applying "
                        "post-hoc NMS at IoU=0.5 to relaxed-pkl detections")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GT…")
    gt_gdf = gpd.read_file(args.gt_gpkg)
    print(f"  {len(gt_gdf)} GT polygons, CRS={gt_gdf.crs}")

    print("Loading relaxed (NMS=0.99) pkl…")
    relaxed_gdf = detections_to_gdf(args.relaxed_pkl, args.score_threshold)
    if relaxed_gdf.crs != gt_gdf.crs:
        print(f"  reprojecting relaxed predictions {relaxed_gdf.crs} → {gt_gdf.crs}")
        relaxed_gdf = relaxed_gdf.to_crs(gt_gdf.crs)

    if args.derive_baseline:
        print("Deriving baseline by post-hoc NMS @ IoU=0.5 on relaxed set…")
        baseline_gdf = post_hoc_nms(relaxed_gdf, iou_threshold=0.5)
        print(f"  {len(baseline_gdf)} baseline detections after post-hoc NMS")
    else:
        print("Loading baseline pkl…")
        baseline_gdf = detections_to_gdf(args.baseline_pkl, args.score_threshold)
        if baseline_gdf.crs != gt_gdf.crs:
            print(f"  reprojecting baseline predictions {baseline_gdf.crs} → {gt_gdf.crs}")
            baseline_gdf = baseline_gdf.to_crs(gt_gdf.crs)

    print(f"\nClassifying {len(gt_gdf)} GT polygons at min_iou={args.min_iou}…")
    df = classify(gt_gdf, baseline_gdf, relaxed_gdf, args.min_iou)
    summary = summarize(df)

    per_poly_path = args.output_dir / "per_polygon_classification.csv"
    summary_path = args.output_dir / "summary_by_stratum.csv"
    df.to_csv(per_poly_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\n  per-polygon → {per_poly_path}")
    print(f"  summary    → {summary_path}\n")
    print(summary.to_string(index=False))

    print("\nKey ratio for (6b) decision:")
    overall = summary[summary["stratum"] == "ALL"].iloc[0]
    missed = overall["n_nms_recoverable"] + overall["n_no_proposal"]
    if missed > 0:
        recoverable_share = overall["n_nms_recoverable"] / missed
        print(f"  of {missed} currently-missed GT, {overall['n_nms_recoverable']} "
              f"({recoverable_share*100:.1f}%) are NMS-recoverable")
        if recoverable_share >= 0.25:
            print("  → ≥25%: relaxing NMS in production is worth pursuing")
        elif recoverable_share <= 0.10:
            print("  → ≤10%: skip (6b), bottleneck is at proposal stage — go to (1) retrain")
        else:
            print("  → 10–25%: marginal; weigh against precision cost on full grid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
