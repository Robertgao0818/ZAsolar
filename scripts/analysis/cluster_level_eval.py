#!/usr/bin/env python3
"""Cluster-level compromise evaluation for split/merge-tolerant polygon matching.

This script evaluates GT and predicted polygons using many-to-many overlap clusters.
It is designed for cases where one large prediction may correspond to multiple
manual polygons, or vice versa.

Core idea:
- Build a bipartite overlap graph between GT and prediction polygons.
- Collapse each connected component into one local cluster.
- Score each cluster using unioned geometry coverage/purity plus a cardinality penalty.
- Aggregate cluster-level precision/recall/F1 while preserving structural penalties.

Typical Joburg CBD usage:
  python3 scripts/analysis/cluster_level_eval.py \
      --annotation-dir data/annotations_channel2_clean/G0816 \
      --results-root results \
      --region jhb \
      --output-dir results/analysis/joburg_cbd_cluster_eval_<run_id>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grid_utils import get_metric_crs


def ensure_crs(gdf: gpd.GeoDataFrame, assumed_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(assumed_crs)
    return gdf


def to_metric_crs(gdf: gpd.GeoDataFrame, *, assumed_crs: str, metric_crs: str) -> gpd.GeoDataFrame:
    gdf = ensure_crs(gdf, assumed_crs)
    if str(gdf.crs) != metric_crs:
        gdf = gdf.to_crs(metric_crs)
    return gdf


@dataclass(frozen=True)
class ClusterComponent:
    gt_indices: set[int]
    pred_indices: set[int]


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def compute_iou(geom_a, geom_b) -> float:
    if geom_a.is_empty or geom_b.is_empty:
        return 0.0
    intersection = geom_a.intersection(geom_b).area
    union = geom_a.area + geom_b.area - intersection
    return safe_ratio(intersection, union)


def _candidate_overlap_edges(
    gt: gpd.GeoDataFrame,
    pred: gpd.GeoDataFrame,
    *,
    edge_iou_threshold: float = 0.0,
) -> list[tuple[int, int]]:
    if len(gt) == 0 or len(pred) == 0:
        return []

    pred_sindex = pred.sindex
    edges: list[tuple[int, int]] = []
    for gt_idx, gt_geom in enumerate(gt.geometry):
        for pred_idx in pred_sindex.intersection(gt_geom.bounds):
            pred_geom = pred.iloc[pred_idx].geometry
            if not gt_geom.intersects(pred_geom):
                continue
            if edge_iou_threshold <= 0:
                edges.append((gt_idx, pred_idx))
                continue
            if compute_iou(gt_geom, pred_geom) >= edge_iou_threshold:
                edges.append((gt_idx, pred_idx))
    return edges


def build_overlap_clusters(
    gt: gpd.GeoDataFrame,
    pred: gpd.GeoDataFrame,
    *,
    edge_iou_threshold: float = 0.0,
) -> list[dict]:
    """Build connected components over a GT↔Pred overlap graph.

    Returns a list of dicts with `gt_indices` and `pred_indices` sets.
    Components with only GT or only Pred nodes are included so FP/FN-only clusters
    remain visible in downstream metrics.
    """
    gt_count = len(gt)
    pred_count = len(pred)
    gt_adj: dict[int, set[int]] = {i: set() for i in range(gt_count)}
    pred_adj: dict[int, set[int]] = {i: set() for i in range(pred_count)}

    for gt_idx, pred_idx in _candidate_overlap_edges(gt, pred, edge_iou_threshold=edge_iou_threshold):
        gt_adj[gt_idx].add(pred_idx)
        pred_adj[pred_idx].add(gt_idx)

    visited_gt: set[int] = set()
    visited_pred: set[int] = set()
    components: list[ClusterComponent] = []

    for start_gt in range(gt_count):
        if start_gt in visited_gt or not gt_adj[start_gt]:
            continue
        stack: list[tuple[str, int]] = [("gt", start_gt)]
        comp_gt: set[int] = set()
        comp_pred: set[int] = set()
        while stack:
            kind, idx = stack.pop()
            if kind == "gt":
                if idx in visited_gt:
                    continue
                visited_gt.add(idx)
                comp_gt.add(idx)
                for pred_idx in gt_adj[idx]:
                    if pred_idx not in visited_pred:
                        stack.append(("pred", pred_idx))
            else:
                if idx in visited_pred:
                    continue
                visited_pred.add(idx)
                comp_pred.add(idx)
                for gt_idx in pred_adj[idx]:
                    if gt_idx not in visited_gt:
                        stack.append(("gt", gt_idx))
        components.append(ClusterComponent(comp_gt, comp_pred))

    for gt_idx in range(gt_count):
        if gt_idx not in visited_gt:
            components.append(ClusterComponent({gt_idx}, set()))
    for pred_idx in range(pred_count):
        if pred_idx not in visited_pred:
            components.append(ClusterComponent(set(), {pred_idx}))

    return [
        {"gt_indices": comp.gt_indices, "pred_indices": comp.pred_indices}
        for comp in components
    ]


def compute_cluster_metrics(gt: gpd.GeoDataFrame, pred: gpd.GeoDataFrame, cluster: dict) -> dict:
    gt_idx = sorted(cluster["gt_indices"])
    pred_idx = sorted(cluster["pred_indices"])

    gt_union = unary_union(gt.iloc[gt_idx].geometry.tolist()) if gt_idx else None
    pred_union = unary_union(pred.iloc[pred_idx].geometry.tolist()) if pred_idx else None

    gt_area = float(gt_union.area) if gt_union and not gt_union.is_empty else 0.0
    pred_area = float(pred_union.area) if pred_union and not pred_union.is_empty else 0.0
    inter_geom = gt_union.intersection(pred_union) if (gt_area > 0 and pred_area > 0) else None
    inter_area = float(inter_geom.area) if inter_geom and not inter_geom.is_empty else 0.0

    area_precision = safe_ratio(inter_area, pred_area)
    area_recall = safe_ratio(inter_area, gt_area)
    area_f1 = safe_ratio(2 * area_precision * area_recall, area_precision + area_recall) if (area_precision + area_recall) else 0.0
    cluster_iou = safe_ratio(inter_area, gt_area + pred_area - inter_area) if (gt_area + pred_area - inter_area) else 0.0

    gt_n = len(gt_idx)
    pred_n = len(pred_idx)
    if gt_n > 0 and pred_n > 0:
        cardinality_penalty = min(gt_n, pred_n) / max(gt_n, pred_n)
    else:
        cardinality_penalty = 0.0
    balanced_score = area_f1 * cardinality_penalty

    return {
        "gt_indices": set(gt_idx),
        "pred_indices": set(pred_idx),
        "gt_count": gt_n,
        "pred_count": pred_n,
        "gt_area": gt_area,
        "pred_area": pred_area,
        "intersection_area": inter_area,
        "area_precision": area_precision,
        "area_recall": area_recall,
        "area_f1": area_f1,
        "cluster_iou": cluster_iou,
        "cardinality_penalty": cardinality_penalty,
        "balanced_score": balanced_score,
    }


def summarize_cluster_metrics(
    gt: gpd.GeoDataFrame,
    pred: gpd.GeoDataFrame,
    *,
    edge_iou_threshold: float = 0.0,
    match_coverage_threshold: float = 0.5,
    match_purity_threshold: float = 0.3,
) -> dict:
    components = build_overlap_clusters(gt, pred, edge_iou_threshold=edge_iou_threshold)
    cluster_rows = [compute_cluster_metrics(gt, pred, cluster) for cluster in components]

    matched = []
    fp_only = []
    fn_only = []
    partial = []
    for row in cluster_rows:
        has_gt = row["gt_count"] > 0
        has_pred = row["pred_count"] > 0
        if has_gt and has_pred:
            is_match = (
                row["area_recall"] >= match_coverage_threshold
                and row["area_precision"] >= match_purity_threshold
            )
            (matched if is_match else partial).append(row)
        elif has_pred:
            fp_only.append(row)
        else:
            fn_only.append(row)

    tp = len(matched)
    fp = len(fp_only) + len(partial)
    fn = len(fn_only) + len(partial)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    matched_df = pd.DataFrame(matched)
    all_df = pd.DataFrame(cluster_rows)
    return {
        "cluster_rows": cluster_rows,
        "matched_cluster_count": tp,
        "fp_cluster_count": fp,
        "fn_cluster_count": fn,
        "cluster_precision": precision,
        "cluster_recall": recall,
        "cluster_f1": f1,
        "mean_matched_area_precision": float(matched_df["area_precision"].mean()) if not matched_df.empty else 0.0,
        "mean_matched_area_recall": float(matched_df["area_recall"].mean()) if not matched_df.empty else 0.0,
        "mean_matched_area_f1": float(matched_df["area_f1"].mean()) if not matched_df.empty else 0.0,
        "mean_cardinality_penalty": float(matched_df["cardinality_penalty"].mean()) if not matched_df.empty else 0.0,
        "mean_balanced_score": float(matched_df["balanced_score"].mean()) if not matched_df.empty else 0.0,
        "mean_cluster_iou": float(matched_df["cluster_iou"].mean()) if not matched_df.empty else 0.0,
        "mean_gt_per_cluster": float(all_df["gt_count"].mean()) if not all_df.empty else 0.0,
        "mean_pred_per_cluster": float(all_df["pred_count"].mean()) if not all_df.empty else 0.0,
    }


def load_largest_layer_gpkg(path: Path) -> tuple[gpd.GeoDataFrame, str | None]:
    layers = pyogrio.list_layers(str(path))
    best_gdf = None
    best_layer = None
    best_count = -1
    for layer_name, _ in layers:
        gdf = gpd.read_file(str(path), layer=layer_name)
        if len(gdf) > best_count:
            best_gdf = gdf
            best_layer = layer_name
            best_count = len(gdf)
    if best_gdf is None:
        best_gdf = gpd.read_file(str(path))
    return best_gdf, best_layer


def load_gt_from_path(grid_id: str, path: Path, region: str | None) -> tuple[gpd.GeoDataFrame, str | None]:
    metric_crs = get_metric_crs(grid_id, region=region)
    gdf, layer = load_largest_layer_gpkg(path)
    gdf = to_metric_crs(gdf, assumed_crs="EPSG:4326", metric_crs=metric_crs)
    gdf = gdf[gdf.geometry.notna() & gdf.is_valid & ~gdf.geometry.is_empty].copy().reset_index(drop=True)
    return gdf, layer


def load_pred_from_results(grid_id: str, path: Path, region: str | None) -> gpd.GeoDataFrame:
    metric_crs = get_metric_crs(grid_id, region=region)
    gdf = gpd.read_file(str(path))
    gdf = to_metric_crs(gdf, assumed_crs=metric_crs, metric_crs=metric_crs)
    gdf = gdf[gdf.geometry.notna() & gdf.is_valid & ~gdf.geometry.is_empty].copy().reset_index(drop=True)
    return gdf


def discover_annotation_files(annotation_dir: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(annotation_dir.glob("*.gpkg")):
        m = re.match(r"(G\d{4}|JHB\d{2})", p.name)
        if m:
            out[m.group(1)] = p
    return out


def evaluate_annotation_dir(
    *,
    annotation_dir: Path,
    results_root: Path,
    output_dir: Path,
    region: str | None,
    edge_iou_threshold: float,
    match_coverage_threshold: float,
    match_purity_threshold: float,
) -> dict:
    annotation_map = discover_annotation_files(annotation_dir)
    if not annotation_map:
        raise FileNotFoundError(f"No GPKG annotations found in {annotation_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_grid_rows = []
    cluster_rows = []
    all_gt = []
    all_pred = []
    missing_predictions = []

    for grid_id, gt_path in sorted(annotation_map.items()):
        pred_path = results_root / grid_id / "predictions_metric.gpkg"
        if not pred_path.exists():
            missing_predictions.append(grid_id)
            continue

        gt, layer = load_gt_from_path(grid_id, gt_path, region)
        pred = load_pred_from_results(grid_id, pred_path, region)
        summary = summarize_cluster_metrics(
            gt,
            pred,
            edge_iou_threshold=edge_iou_threshold,
            match_coverage_threshold=match_coverage_threshold,
            match_purity_threshold=match_purity_threshold,
        )

        per_grid_rows.append({
            "grid_id": grid_id,
            "annotation_path": str(gt_path),
            "annotation_layer": layer,
            "predictions_path": str(pred_path),
            "gt_polygon_count": len(gt),
            "pred_polygon_count": len(pred),
            **{k: v for k, v in summary.items() if k != "cluster_rows"},
        })

        for idx, row in enumerate(summary["cluster_rows"]):
            cluster_rows.append({
                "grid_id": grid_id,
                "cluster_id": f"{grid_id}_C{idx:03d}",
                **{k: v for k, v in row.items() if not isinstance(v, set)},
                "gt_indices": sorted(row["gt_indices"]),
                "pred_indices": sorted(row["pred_indices"]),
            })

        gt2 = gt[["geometry"]].copy(); gt2["grid_id"] = grid_id
        pred2 = pred[["geometry"]].copy(); pred2["grid_id"] = grid_id
        all_gt.append(gt2); all_pred.append(pred2)

    per_grid_df = pd.DataFrame(per_grid_rows).sort_values("grid_id").reset_index(drop=True)
    cluster_df = pd.DataFrame(cluster_rows)
    per_grid_df.to_csv(output_dir / "per_grid_cluster_metrics.csv", index=False)
    cluster_df.to_json(output_dir / "cluster_details.json", orient="records", indent=2)
    if missing_predictions:
        (output_dir / "missing_predictions.txt").write_text("\n".join(missing_predictions) + "\n", encoding="utf-8")

    if per_grid_df.empty:
        raise RuntimeError("No grids evaluated successfully")

    agg_summary = summarize_cluster_metrics(
        pd.concat(all_gt, ignore_index=True),
        pd.concat(all_pred, ignore_index=True),
        edge_iou_threshold=edge_iou_threshold,
        match_coverage_threshold=match_coverage_threshold,
        match_purity_threshold=match_purity_threshold,
    )

    result = {
        "output_dir": str(output_dir),
        "annotation_dir": str(annotation_dir),
        "results_root": str(results_root),
        "n_grids": int(len(per_grid_df)),
        "edge_iou_threshold": edge_iou_threshold,
        "match_coverage_threshold": match_coverage_threshold,
        "match_purity_threshold": match_purity_threshold,
        "aggregate_cluster_metrics": {k: v for k, v in agg_summary.items() if k != "cluster_rows"},
        "macro_means": {
            "cluster_precision": float(per_grid_df["cluster_precision"].mean()),
            "cluster_recall": float(per_grid_df["cluster_recall"].mean()),
            "cluster_f1": float(per_grid_df["cluster_f1"].mean()),
            "mean_matched_area_f1": float(per_grid_df["mean_matched_area_f1"].mean()),
            "mean_cardinality_penalty": float(per_grid_df["mean_cardinality_penalty"].mean()),
            "mean_balanced_score": float(per_grid_df["mean_balanced_score"].mean()),
        },
        "top_grids_by_balanced_score": per_grid_df.nlargest(5, "mean_balanced_score")[[
            "grid_id", "cluster_f1", "mean_matched_area_f1", "mean_cardinality_penalty", "mean_balanced_score"
        ]].to_dict(orient="records"),
        "bottom_grids_by_balanced_score": per_grid_df.nsmallest(5, "mean_balanced_score")[[
            "grid_id", "cluster_f1", "mean_matched_area_f1", "mean_cardinality_penalty", "mean_balanced_score"
        ]].to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster-level split/merge-tolerant evaluation")
    parser.add_argument("--annotation-dir", type=Path, required=True, help="Directory containing GT .gpkg files")
    parser.add_argument("--results-root", type=Path, required=True, help="Root containing <grid>/predictions_metric.gpkg")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: results/analysis/cluster_eval_<timestamp>)")
    parser.add_argument("--region", default=None, help="Optional region alias for metric CRS (e.g. jhb)")
    parser.add_argument("--edge-iou-threshold", type=float, default=0.0, help="Minimum polygon-pair IoU to add a graph edge")
    parser.add_argument("--match-coverage-threshold", type=float, default=0.5, help="Minimum cluster recall to count as matched")
    parser.add_argument("--match-purity-threshold", type=float, default=0.3, help="Minimum cluster precision to count as matched")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "results" / "analysis" / f"cluster_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    result = evaluate_annotation_dir(
        annotation_dir=args.annotation_dir,
        results_root=args.results_root,
        output_dir=output_dir,
        region=args.region,
        edge_iou_threshold=args.edge_iou_threshold,
        match_coverage_threshold=args.match_coverage_threshold,
        match_purity_threshold=args.match_purity_threshold,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
