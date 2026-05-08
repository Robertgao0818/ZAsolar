#!/usr/bin/env python3
"""Envelope-group fill trigger sweep over (env_area_min, group_density_min,
n_clusters_min). For each variant, compute overall + source-stratified metrics
against a clean_gt with `source` labels (V3C_TP / SAM_supp+V3C_TP / ...).

Inputs (all per-grid):
- ``--per-detection-gpkg``  finalize.py --merge-mode per-detection output
- ``--pixel-or-gpkg``       finalize.py --merge-mode pixel-or output
- ``--clean-gt``            clean_gt with `source` column
- ``--ref-sam-maskbox-gpkg`` (optional) reference for diagnostic comparison

The sweep:
1. Builds mutual-IoU clusters from per_detection (default IoU >= 0.3).
2. Assigns each cluster to a pixel-or envelope when the cluster is mostly
   inside one (intersection / cluster_area >= ``--cluster-in-envelope``,
   default 0.8).
3. For each (a, d, c) trigger, replaces a cluster group with its envelope when
   ``env_area >= a``, ``group_density >= d``, ``n_clusters >= c``.

Outputs (under ``--output-dir``):
- ``overall_metrics.csv``           one row per variant
- ``source_metrics.csv``            one row per (variant, source)
- ``group_assignments.csv``         cluster→envelope map with trigger features
- ``summary.md``                    top variants + recommendation
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import get_metric_crs, normalize_grid_id  # noqa: E402


# ───────────────────────── helpers ─────────────────────────

def _polyize(geom: BaseGeometry) -> list[Polygon]:
    """Drop non-polygonal pieces; return list of Polygons."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_polyize(g))
        return out
    return []


def _safe_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _unary_safe(geoms: list[BaseGeometry]) -> BaseGeometry | None:
    if not geoms:
        return None
    if len(geoms) == 1:
        return geoms[0]
    return unary_union(geoms)


# ───────────────────────── clustering ─────────────────────────

def mutual_iou_clusters(
    polys: list[BaseGeometry],
    iou_threshold: float,
) -> list[int]:
    """Union-find clustering by symmetric IoU >= threshold. Returns cluster id
    per input polygon (root index after path compression)."""
    n = len(polys)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    if n > 1:
        tree = STRtree(polys)
        for i, p in enumerate(polys):
            for j in tree.query(p):
                j = int(j)
                if j <= i:
                    continue
                if _safe_iou(p, polys[j]) >= iou_threshold:
                    union(i, j)
    return [find(i) for i in range(n)]


# ───────────────────────── envelope assignment ─────────────────────────

@dataclass
class ClusterRow:
    cluster_id: int
    members: list[int]
    geom: BaseGeometry
    area: float
    score: float

@dataclass
class EnvelopeRow:
    env_id: int
    geom: BaseGeometry
    area: float
    score: float


def build_clusters_from_per_detection(
    per_det: gpd.GeoDataFrame,
    iou_threshold: float,
) -> list[ClusterRow]:
    polys = list(per_det.geometry.values)
    scores = per_det["confidence"].astype(float).values if "confidence" in per_det.columns \
        else (per_det["score"].astype(float).values if "score" in per_det.columns
              else np.ones(len(per_det)))
    cluster_id = mutual_iou_clusters(polys, iou_threshold)
    members_by_cid: dict[int, list[int]] = {}
    for i, cid in enumerate(cluster_id):
        members_by_cid.setdefault(cid, []).append(i)
    out: list[ClusterRow] = []
    for new_id, (_, members) in enumerate(sorted(members_by_cid.items())):
        geoms = [polys[i] for i in members]
        merged = _unary_safe(geoms)
        if merged is None or merged.is_empty:
            continue
        out.append(ClusterRow(
            cluster_id=new_id,
            members=members,
            geom=merged,
            area=float(merged.area),
            score=float(max(scores[i] for i in members)),
        ))
    return out


def assign_clusters_to_envelopes(
    clusters: list[ClusterRow],
    envelopes: list[EnvelopeRow],
    inside_ratio: float,
) -> dict[int, int | None]:
    """For each cluster_id, return env_id (or None if no envelope qualifies)."""
    if not envelopes:
        return {c.cluster_id: None for c in clusters}
    env_geoms = [e.geom for e in envelopes]
    env_index_by_pos = [e.env_id for e in envelopes]
    tree = STRtree(env_geoms)
    out: dict[int, int | None] = {}
    for c in clusters:
        candidates = tree.query(c.geom)
        best_env: int | None = None
        best_intersect = 0.0
        for pos in candidates:
            pos = int(pos)
            inter = c.geom.intersection(env_geoms[pos]).area
            if c.area > 0 and inter / c.area >= inside_ratio and inter > best_intersect:
                best_env = env_index_by_pos[pos]
                best_intersect = inter
        out[c.cluster_id] = best_env
    return out


# ───────────────────────── trigger sweep ─────────────────────────

@dataclass
class GroupSummary:
    env_id: int
    env_area: float
    n_clusters: int
    cluster_union_area: float
    group_density: float
    member_cluster_ids: list[int]


def summarize_groups(
    clusters: list[ClusterRow],
    envelopes: list[EnvelopeRow],
    cluster_to_env: dict[int, int | None],
) -> list[GroupSummary]:
    by_env: dict[int, list[ClusterRow]] = {}
    for c in clusters:
        eid = cluster_to_env.get(c.cluster_id)
        if eid is None:
            continue
        by_env.setdefault(eid, []).append(c)
    env_by_id = {e.env_id: e for e in envelopes}
    out: list[GroupSummary] = []
    for eid, members in by_env.items():
        env = env_by_id[eid]
        merged = _unary_safe([m.geom for m in members])
        cu_area = 0.0 if merged is None or merged.is_empty else float(merged.area)
        density = cu_area / env.area if env.area > 0 else 0.0
        out.append(GroupSummary(
            env_id=eid,
            env_area=float(env.area),
            n_clusters=len(members),
            cluster_union_area=cu_area,
            group_density=float(density),
            member_cluster_ids=sorted([m.cluster_id for m in members]),
        ))
    return out


def build_variant_geometries(
    clusters: list[ClusterRow],
    envelopes: list[EnvelopeRow],
    cluster_to_env: dict[int, int | None],
    groups: list[GroupSummary],
    *,
    env_area_min: float,
    density_min: float,
    n_clusters_min: int,
) -> tuple[list[BaseGeometry], int, list[float]]:
    """Returns (variant_geoms, n_envelope_group_fill, fill_areas)."""
    triggered_envs: dict[int, EnvelopeRow] = {}
    fill_areas: list[float] = []
    env_by_id = {e.env_id: e for e in envelopes}
    triggered_cluster_ids: set[int] = set()
    for g in groups:
        if (
            g.env_area >= env_area_min
            and g.group_density >= density_min
            and g.n_clusters >= n_clusters_min
        ):
            triggered_envs[g.env_id] = env_by_id[g.env_id]
            fill_areas.append(env_by_id[g.env_id].area)
            triggered_cluster_ids.update(g.member_cluster_ids)
    out: list[BaseGeometry] = []
    for c in clusters:
        if c.cluster_id in triggered_cluster_ids:
            continue
        out.append(c.geom)
    for env in triggered_envs.values():
        out.append(env.geom)
    return out, len(triggered_envs), fill_areas


# ───────────────────────── metrics ─────────────────────────

def _pixel_area_recall(
    pred_union: BaseGeometry | None,
    gt_subset: gpd.GeoDataFrame,
) -> tuple[float, float, float]:
    """Returns (intersection_area, gt_area, pred_area) over the subset."""
    if pred_union is None or pred_union.is_empty:
        return 0.0, float(gt_subset.geometry.area.sum()), 0.0
    gt_union = unary_union(list(gt_subset.geometry))
    inter = pred_union.intersection(gt_union).area
    return float(inter), float(gt_union.area), float(pred_union.area)


def compute_overall(
    variant_geoms: list[BaseGeometry],
    clean_gt: gpd.GeoDataFrame,
    *,
    iou_threshold: float = 0.3,
    n_envelope_fill: int = 0,
    fill_areas: list[float] | None = None,
) -> dict:
    fill_areas = fill_areas or []
    pred_areas = np.array([g.area for g in variant_geoms]) if variant_geoms else np.zeros(0)
    pred_union = _unary_safe(variant_geoms)
    gt_union = unary_union(list(clean_gt.geometry)) if len(clean_gt) > 0 else None

    pred_area_total = float(pred_areas.sum())
    gt_area_total = float(gt_union.area) if gt_union is not None else 0.0
    if pred_union is not None and gt_union is not None:
        inter_area = float(pred_union.intersection(gt_union).area)
    else:
        inter_area = 0.0
    pixel_area_p = inter_area / pred_area_total if pred_area_total > 0 else 0.0
    pixel_area_r = inter_area / gt_area_total if gt_area_total > 0 else 0.0
    if pixel_area_p + pixel_area_r > 0:
        pixel_area_f1 = 2 * pixel_area_p * pixel_area_r / (pixel_area_p + pixel_area_r)
    else:
        pixel_area_f1 = 0.0

    # polygon-level matching at IoU >= iou_threshold.
    # TP/FP are pred-side: pred is TP if best IoU with any GT >= threshold.
    # FN is GT-side: GT counted matched if best IoU with any pred >= threshold.
    tp_pred = 0
    fp = 0
    n_matched_gt = 0
    n_pred = len(variant_geoms)
    if variant_geoms and len(clean_gt) > 0:
        gt_geoms = list(clean_gt.geometry.values)
        gt_tree = STRtree(gt_geoms)
        pred_tree = STRtree(variant_geoms)
        for p in variant_geoms:
            best_iou = 0.0
            for j in gt_tree.query(p):
                j = int(j)
                iou = _safe_iou(p, gt_geoms[j])
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= iou_threshold:
                tp_pred += 1
            else:
                fp += 1
        for gt in gt_geoms:
            best_iou = 0.0
            for j in pred_tree.query(gt):
                j = int(j)
                iou = _safe_iou(variant_geoms[j], gt)
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= iou_threshold:
                n_matched_gt += 1
    fn_gt = len(clean_gt) - n_matched_gt
    poly_p = tp_pred / n_pred if n_pred > 0 else 0.0
    poly_r = n_matched_gt / len(clean_gt) if len(clean_gt) > 0 else 0.0
    poly_f1 = 2 * poly_p * poly_r / (poly_p + poly_r) if (poly_p + poly_r) > 0 else 0.0

    return {
        "n_pred": n_pred,
        "pred_area_m2": round(pred_area_total, 1),
        "n_gt": len(clean_gt),
        "gt_area_m2": round(gt_area_total, 1),
        "pred_gt": round(pred_area_total / gt_area_total, 3) if gt_area_total > 0 else 0.0,
        "max_m2": round(float(pred_areas.max()), 1) if n_pred > 0 else 0.0,
        "p95_m2": round(float(np.percentile(pred_areas, 95)), 1) if n_pred > 0 else 0.0,
        "p99_m2": round(float(np.percentile(pred_areas, 99)), 1) if n_pred > 0 else 0.0,
        "n_ge_200": int((pred_areas >= 200).sum()),
        "area_p": round(pixel_area_p, 3),
        "area_r": round(pixel_area_r, 3),
        "area_f1": round(pixel_area_f1, 3),
        "tp_pred_iou03": tp_pred,
        "fp_iou03": fp,
        "fn_gt_iou03": fn_gt,
        "poly_p_iou03": round(poly_p, 3),
        "poly_r_iou03": round(poly_r, 3),
        "poly_f1_iou03": round(poly_f1, 3),
        "n_envelope_group_fill": int(n_envelope_fill),
        "mean_fill_area_m2": round(float(np.mean(fill_areas)), 1) if fill_areas else 0.0,
        "max_fill_area_m2": round(float(np.max(fill_areas)), 1) if fill_areas else 0.0,
    }


def compute_source_metrics(
    variant_geoms: list[BaseGeometry],
    clean_gt_subset: gpd.GeoDataFrame,
) -> dict:
    n_gt = len(clean_gt_subset)
    if n_gt == 0:
        return {
            "n_gt": 0, "gt_area_sum": 0.0,
            "any_overlap_rate": 0.0, "recall_iou03": 0.0, "recall_iou05": 0.0,
            "median_iou": 0.0, "mean_iou": 0.0, "share_iou_lt_05": 0.0,
            "area_recall": 0.0, "median_coverage": 0.0,
            "mean_n_pred_hits": 0.0, "median_n_pred_hits": 0.0,
        }
    gt_geoms = list(clean_gt_subset.geometry.values)
    gt_areas = np.array([g.area for g in gt_geoms])
    gt_area_sum = float(gt_areas.sum())

    if not variant_geoms:
        return {
            "n_gt": n_gt, "gt_area_sum": round(gt_area_sum, 1),
            "any_overlap_rate": 0.0, "recall_iou03": 0.0, "recall_iou05": 0.0,
            "median_iou": 0.0, "mean_iou": 0.0, "share_iou_lt_05": 1.0,
            "area_recall": 0.0, "median_coverage": 0.0,
            "mean_n_pred_hits": 0.0, "median_n_pred_hits": 0.0,
        }
    pred_tree = STRtree(variant_geoms)
    best_iou = np.zeros(n_gt)
    coverage = np.zeros(n_gt)
    n_hits = np.zeros(n_gt, dtype=int)
    intersection_total = 0.0
    for i, gt in enumerate(gt_geoms):
        candidate_idx = [int(j) for j in pred_tree.query(gt)]
        if not candidate_idx:
            continue
        candidate_geoms = [variant_geoms[j] for j in candidate_idx]
        pred_local_union = _unary_safe(candidate_geoms)
        if pred_local_union is None or pred_local_union.is_empty:
            continue
        inter = pred_local_union.intersection(gt)
        if inter.is_empty:
            continue
        intersection_total += inter.area
        # per-pred IoUs
        ious = []
        for pg in candidate_geoms:
            inter_pg = pg.intersection(gt).area
            if inter_pg <= 0:
                continue
            n_hits[i] += 1
            union_pg = pg.area + gt.area - inter_pg
            if union_pg > 0:
                ious.append(inter_pg / union_pg)
        if ious:
            best_iou[i] = max(ious)
        coverage[i] = min(inter.area / gt.area, 1.0) if gt.area > 0 else 0.0

    any_overlap_rate = float((best_iou > 0).sum()) / n_gt
    recall_iou03 = float((best_iou >= 0.3).sum()) / n_gt
    recall_iou05 = float((best_iou >= 0.5).sum()) / n_gt
    median_iou_val = float(np.median(best_iou[best_iou > 0])) if (best_iou > 0).any() else 0.0
    mean_iou_val = float(best_iou[best_iou > 0].mean()) if (best_iou > 0).any() else 0.0
    share_iou_lt_05 = float((best_iou < 0.5).sum()) / n_gt
    area_recall = intersection_total / gt_area_sum if gt_area_sum > 0 else 0.0
    median_coverage = float(np.median(coverage[coverage > 0])) if (coverage > 0).any() else 0.0
    mean_n_pred_hits = float(n_hits.mean())
    median_n_pred_hits = float(np.median(n_hits))

    return {
        "n_gt": n_gt,
        "gt_area_sum": round(gt_area_sum, 1),
        "any_overlap_rate": round(any_overlap_rate, 3),
        "recall_iou03": round(recall_iou03, 3),
        "recall_iou05": round(recall_iou05, 3),
        "median_iou": round(median_iou_val, 3),
        "mean_iou": round(mean_iou_val, 3),
        "share_iou_lt_05": round(share_iou_lt_05, 3),
        "area_recall": round(area_recall, 3),
        "median_coverage": round(median_coverage, 3),
        "mean_n_pred_hits": round(mean_n_pred_hits, 3),
        "median_n_pred_hits": round(median_n_pred_hits, 3),
    }


# ───────────────────────── orchestration ─────────────────────────

def _load_gpkg(path: Path, metric_crs: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError(f"{path} has no CRS")
    if str(g.crs) != metric_crs:
        g = g.to_crs(metric_crs)
    return g.reset_index(drop=True)


def _parse_grid(arg: str | None, default: list[float]) -> list[float]:
    if arg is None:
        return default
    return [float(x) for x in arg.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-id", required=True)
    ap.add_argument("--region", default="jhb")
    ap.add_argument("--per-detection-gpkg", required=True, type=Path)
    ap.add_argument("--pixel-or-gpkg", required=True, type=Path)
    ap.add_argument("--clean-gt", required=True, type=Path)
    ap.add_argument("--ref-sam-maskbox-gpkg", type=Path, default=None,
                    help="Optional SAM maskbox baseline for diagnostic")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--mutual-iou", type=float, default=0.3)
    ap.add_argument("--cluster-in-envelope", type=float, default=0.8,
                    help="Cluster assigned to envelope when "
                         "intersection / cluster_area >= this threshold")
    ap.add_argument("--area-mins", default=None,
                    help="Comma-list of env_area_min m². Default: 100,150,200")
    ap.add_argument("--density-mins", default=None,
                    help="Comma-list of group_density_min. Default: 0.45,0.50,0.55,0.60,0.65,0.75")
    ap.add_argument("--cluster-mins", default=None,
                    help="Comma-list of n_clusters_min. Default: 2,3,4")
    args = ap.parse_args()

    grid_id = normalize_grid_id(args.grid_id)
    metric_crs = get_metric_crs(grid_id, region=args.region)
    metric_crs_str = str(metric_crs)

    print(f"[sweep] grid={grid_id} region={args.region} crs={metric_crs_str}")

    per_det = _load_gpkg(args.per_detection_gpkg, metric_crs_str)
    pixel_or = _load_gpkg(args.pixel_or_gpkg, metric_crs_str)
    clean_gt = _load_gpkg(args.clean_gt, metric_crs_str)
    ref_sam: gpd.GeoDataFrame | None = None
    if args.ref_sam_maskbox_gpkg is not None and args.ref_sam_maskbox_gpkg.exists():
        ref_sam = _load_gpkg(args.ref_sam_maskbox_gpkg, metric_crs_str)

    print(f"[sweep] per_det={len(per_det)}  pixel_or={len(pixel_or)}  "
          f"clean_gt={len(clean_gt)}  ref_sam={None if ref_sam is None else len(ref_sam)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) clusters from per_detection
    clusters = build_clusters_from_per_detection(per_det, args.mutual_iou)
    print(f"[sweep] mutual_iou_{args.mutual_iou} clusters: {len(clusters)}")

    # 2) envelopes from pixel_or
    envelopes = [
        EnvelopeRow(env_id=i, geom=row.geometry,
                    area=float(row.geometry.area),
                    score=float(row.get("confidence", row.get("score", 0.0))))
        for i, row in pixel_or.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    ]
    print(f"[sweep] envelopes: {len(envelopes)}")

    # 3) cluster→envelope assignment
    cluster_to_env = assign_clusters_to_envelopes(clusters, envelopes, args.cluster_in_envelope)
    n_assigned = sum(1 for v in cluster_to_env.values() if v is not None)
    print(f"[sweep] clusters assigned to an envelope: {n_assigned} / {len(clusters)}")

    # 4) groups
    groups = summarize_groups(clusters, envelopes, cluster_to_env)
    print(f"[sweep] envelope groups (>=1 cluster): {len(groups)}")

    # write group_assignments.csv
    group_rows = []
    for g in groups:
        group_rows.append({
            "env_id": g.env_id,
            "env_area_m2": round(g.env_area, 1),
            "n_clusters": g.n_clusters,
            "cluster_union_area_m2": round(g.cluster_union_area, 1),
            "group_density": round(g.group_density, 3),
            "member_cluster_ids": ";".join(str(x) for x in g.member_cluster_ids),
        })
    pd.DataFrame(group_rows).to_csv(args.output_dir / "group_assignments.csv", index=False)

    # 5) variants
    sources = list(clean_gt["source"].unique()) if "source" in clean_gt.columns else []

    variants: dict[str, list[BaseGeometry]] = {}
    variant_meta: dict[str, dict] = {}

    # baseline: pixel-or
    pixel_or_geoms = [e.geom for e in envelopes]
    variants["old_pixel_or"] = pixel_or_geoms
    variant_meta["old_pixel_or"] = {"n_envelope_fill": 0, "fill_areas": []}

    # baseline: per_detection (no merge)
    pd_geoms = list(per_det.geometry.values)
    variants["per_detection"] = pd_geoms
    variant_meta["per_detection"] = {"n_envelope_fill": 0, "fill_areas": []}

    # baseline: mutual_iou_<t>
    mu_name = f"mutual_iou_{args.mutual_iou}"
    variants[mu_name] = [c.geom for c in clusters]
    variant_meta[mu_name] = {"n_envelope_fill": 0, "fill_areas": []}

    # optional ref SAM maskbox
    if ref_sam is not None:
        variants["ref_sam_maskbox"] = list(ref_sam.geometry.values)
        variant_meta["ref_sam_maskbox"] = {"n_envelope_fill": 0, "fill_areas": []}

    # group fill grid
    a_grid = _parse_grid(args.area_mins, [100.0, 150.0, 200.0])
    d_grid = _parse_grid(args.density_mins, [0.45, 0.50, 0.55, 0.60, 0.65, 0.75])
    c_grid = [int(x) for x in _parse_grid(args.cluster_mins, [2, 3, 4])]
    print(f"[sweep] grid: a={a_grid} d={d_grid} c={c_grid} ({len(a_grid)*len(d_grid)*len(c_grid)} variants)")

    for a, d, c in itertools.product(a_grid, d_grid, c_grid):
        name = f"group_a{int(a)}_d{d:.2f}_c{c}"
        geoms, n_fill, fill_areas = build_variant_geometries(
            clusters, envelopes, cluster_to_env, groups,
            env_area_min=a, density_min=d, n_clusters_min=c,
        )
        variants[name] = geoms
        variant_meta[name] = {"n_envelope_fill": n_fill, "fill_areas": fill_areas,
                              "area_min_m2": a, "density_min": d, "min_clusters": c}

    # 6) metrics
    overall_rows = []
    source_rows = []
    for vname, geoms in variants.items():
        meta = variant_meta[vname]
        ovr = compute_overall(geoms, clean_gt,
                              n_envelope_fill=meta["n_envelope_fill"],
                              fill_areas=meta["fill_areas"])
        ovr["variant"] = vname
        ovr["area_min_m2"] = meta.get("area_min_m2", "")
        ovr["density_min"] = meta.get("density_min", "")
        ovr["min_clusters"] = meta.get("min_clusters", "")
        overall_rows.append(ovr)
        for src in sources:
            sub = clean_gt[clean_gt["source"] == src].reset_index(drop=True)
            srm = compute_source_metrics(geoms, sub)
            srm["variant"] = vname
            srm["source"] = src
            source_rows.append(srm)

    overall_df = pd.DataFrame(overall_rows)
    overall_df = overall_df[[
        "variant", "area_min_m2", "density_min", "min_clusters",
        "n_pred", "pred_area_m2", "n_gt", "gt_area_m2", "pred_gt",
        "max_m2", "p95_m2", "p99_m2", "n_ge_200",
        "area_p", "area_r", "area_f1",
        "tp_pred_iou03", "fp_iou03", "fn_gt_iou03",
        "poly_p_iou03", "poly_r_iou03", "poly_f1_iou03",
        "n_envelope_group_fill", "mean_fill_area_m2", "max_fill_area_m2",
    ]]
    overall_df.to_csv(args.output_dir / "overall_metrics.csv", index=False)

    source_df = pd.DataFrame(source_rows)
    source_df = source_df[[
        "variant", "source",
        "n_gt", "gt_area_sum",
        "any_overlap_rate", "recall_iou03", "recall_iou05",
        "median_iou", "mean_iou", "share_iou_lt_05",
        "area_recall", "median_coverage",
        "mean_n_pred_hits", "median_n_pred_hits",
    ]]
    source_df.to_csv(args.output_dir / "source_metrics.csv", index=False)

    # 7) summary
    write_summary(args.output_dir, args.grid_id, overall_df, source_df, sources)
    print(f"[sweep] wrote {args.output_dir}/overall_metrics.csv "
          f"({len(overall_df)} variants), source_metrics.csv "
          f"({len(source_df)} rows), group_assignments.csv ({len(group_rows)} groups)")
    return 0


def _md_table(df: pd.DataFrame, fmt: str = ".3f") -> str:
    if len(df) == 0:
        return "(no rows)\n"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---:" if pd.api.types.is_numeric_dtype(df[c]) else "---"
                              for c in cols]) + " |\n"
    body_lines = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, fmt))
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v))
        body_lines.append("| " + " | ".join(cells) + " |")
    return header + sep + "\n".join(body_lines) + "\n"


def write_summary(
    out_dir: Path, grid_id: str,
    overall_df: pd.DataFrame, source_df: pd.DataFrame,
    sources: list[str],
) -> None:
    g = overall_df.copy()
    group_rows = g[g["variant"].str.startswith("group_a")]
    if len(group_rows):
        best_f1 = group_rows.sort_values("area_f1", ascending=False).iloc[0]
        cons = group_rows[(group_rows["density_min"] == 0.50)
                          & (group_rows["area_min_m2"] == 150.0)
                          & (group_rows["min_clusters"] == 3)]
        loose = group_rows[(group_rows["density_min"] == 0.45)
                           & (group_rows["area_min_m2"] == 100.0)
                           & (group_rows["min_clusters"] == 3)]
    else:
        best_f1 = None
        cons = loose = pd.DataFrame()

    lines = []
    lines.append(f"# {grid_id} envelope-group sweep summary\n\n")
    lines.append("## Overall (baselines + leaders)\n\n")
    keep_cols = ["variant", "n_pred", "pred_gt", "area_p", "area_r", "area_f1",
                 "poly_r_iou03", "n_envelope_group_fill"]
    mu_name = ""
    if (g["variant"].str.startswith("mutual_iou_")).any():
        mu_name = g[g["variant"].str.startswith("mutual_iou_")]["variant"].iloc[0]
    base_rows = g[g["variant"].isin(["old_pixel_or", "per_detection", mu_name, "ref_sam_maskbox"])][keep_cols]
    extras = []
    if len(cons):
        extras.append(cons.iloc[0][keep_cols])
    if len(loose):
        extras.append(loose.iloc[0][keep_cols])
    if best_f1 is not None and (len(cons) == 0 or best_f1["variant"] != cons.iloc[0]["variant"]) \
            and (len(loose) == 0 or best_f1["variant"] != loose.iloc[0]["variant"]):
        extras.append(best_f1[keep_cols])
    if extras:
        base_rows = pd.concat([base_rows, pd.DataFrame(extras)], ignore_index=True)
    lines.append(_md_table(base_rows.reset_index(drop=True)))

    if best_f1 is not None:
        lines.append("\n## Group-fill area-F1 leader\n\n")
        lines.append(f"- variant: `{best_f1['variant']}`  \n")
        lines.append(f"- area_F1 = {best_f1['area_f1']:.3f}, "
                     f"area_R = {best_f1['area_r']:.3f}, "
                     f"poly_R@0.3 = {best_f1['poly_r_iou03']:.3f}, "
                     f"n_pred = {int(best_f1['n_pred'])}, "
                     f"envelope_fills = {int(best_f1['n_envelope_group_fill'])}\n")
        if len(cons):
            r = cons.iloc[0]
            lines.append("\n## Conservative trigger (a150_d0.50_c3)\n\n")
            lines.append(f"- area_F1 = {r['area_f1']:.3f}, area_R = {r['area_r']:.3f}, "
                         f"poly_R@0.3 = {r['poly_r_iou03']:.3f}, "
                         f"n_pred = {int(r['n_pred'])}, "
                         f"envelope_fills = {int(r['n_envelope_group_fill'])}\n")
        if len(loose):
            r = loose.iloc[0]
            lines.append("\n## Loose trigger (a100_d0.45_c3)\n\n")
            lines.append(f"- area_F1 = {r['area_f1']:.3f}, area_R = {r['area_r']:.3f}, "
                         f"poly_R@0.3 = {r['poly_r_iou03']:.3f}, "
                         f"n_pred = {int(r['n_pred'])}, "
                         f"envelope_fills = {int(r['n_envelope_group_fill'])}\n")

    lines.append("\n## Source diagnostics (conservative + loose vs baselines)\n")
    pivot_variants = ["old_pixel_or", "ref_sam_maskbox"]
    mu_var = source_df[source_df["variant"].str.startswith("mutual_iou_")]["variant"]
    if len(mu_var):
        pivot_variants.append(mu_var.iloc[0])
    if len(cons):
        pivot_variants.append(cons.iloc[0]["variant"])
    if len(loose):
        pivot_variants.append(loose.iloc[0]["variant"])
    for src in sources:
        sub = source_df[(source_df["source"] == src)
                        & (source_df["variant"].isin(pivot_variants))]
        if len(sub) == 0:
            continue
        lines.append(f"\n### source = `{src}`\n\n")
        cols = ["variant", "n_gt", "recall_iou05", "median_iou",
                "share_iou_lt_05", "area_recall"]
        lines.append(_md_table(sub[cols].reset_index(drop=True)))

    (out_dir / "summary.md").write_text("".join(lines))


if __name__ == "__main__":
    sys.exit(main())
