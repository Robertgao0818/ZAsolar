#!/usr/bin/env python3
"""Build model area/coverage tracker from existing prediction and GT polygons.

This is the reusable version of the 25-grid GEID Li area-eval frame.  It
answers two questions per model/grid:

1. Coverage / discovery: how much of the human-cut solar area did the model
   find, and what fraction of GT installations have substantial model coverage?
2. Area accuracy / purity: how much predicted solar area overlaps human-cut
   solar area, and how different is predicted total area from GT total area?

Outputs:
    results/analysis/model_area_coverage_tracker.csv
    results/analysis/model_area_coverage_tracker.md

No inference is re-run.  The script reads rows from
results/analysis/model_grid_metrics_tracker.csv, uses each row's result_dir for
predictions_metric.gpkg, and resolves GT from configs/datasets/regions.yaml.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyogrio
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.grid_utils import get_metric_crs  # noqa: E402
from core.region_registry import get_region_config  # noqa: E402
from scripts.analysis.build_model_grid_metrics_tracker import (  # noqa: E402
    family_of as _family_of_run,
    infer_grid_id as _infer_grid_id,
    infer_imagery_layer as _infer_imagery_layer,
    infer_model_run as _infer_model_run,
    infer_region as _infer_region,
    read_json_safe as _read_config_json,
)

INPUT_TRACKER = REPO_ROOT / "results" / "analysis" / "model_grid_metrics_tracker.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "analysis"
OUTPUT_CSV = OUTPUT_DIR / "model_area_coverage_tracker.csv"
OUTPUT_MD = OUTPUT_DIR / "model_area_coverage_tracker.md"

REGION_KEY = {"ct": "cape_town", "jhb": "johannesburg", "cape_town": "cape_town", "johannesburg": "johannesburg"}
MAX_PLAUSIBLE_POLY_M2 = 20_000.0

# Li independent GT locations (human annotations that are NOT the reviewed-prediction
# self-loop GT). Evaluating predictions against these gives the blueprint benchmark:
# how much human-cut solar area did the model find, and how close is the total area.
LI_GT_DIRS: dict[str, list[Path]] = {
    "cape_town": [REPO_ROOT / "data" / "annotations" / "Capetown_Li"],
    "johannesburg": [
        Path("/mnt/d/ZAsolar/annotations_inbox/Joburg_CBD_Li"),
    ],
}

COLUMNS = [
    "region",
    "model_family",
    "model_run",
    "imagery_layer",
    "grid_id",
    "result_dir",
    "gt_source_type",
    "benchmark_alias",
    "gt_source",
    "gt_layer",
    "gt_polygon_count",
    "pred_polygon_count",
    "gt_union_area_m2",
    "pred_union_area_m2",
    "intersection_area_m2",
    "area_precision",
    "area_recall",
    "area_f1",
    "pred_gt_area_ratio",
    "signed_area_error",
    "abs_area_error",
    "gt_coverage_mean",
    "gt_coverage_median",
    "gt_coverage_ge_05_rate",
    "gt_coverage_ge_08_rate",
    "pred_purity_mean",
    "pred_purity_median",
    "pred_purity_ge_05_rate",
    "pred_purity_ge_08_rate",
    "gt_with_any_pred_rate",
    "pred_with_any_gt_rate",
    "gt_with_multi_pred_intersections_rate",
    "pred_with_multi_gt_intersections_rate",
    "mean_pred_intersections_per_gt",
    "mean_gt_intersections_per_pred",
    "n_dropped_gt",
    "n_dropped_pred",
    "eval_status",
    "skip_reason",
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None or x == "":
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(v) or math.isinf(v):
        return "-"
    return f"{v:.{digits}f}"


def _geometry_finite(geom) -> bool:
    try:
        bounds = geom.bounds
    except Exception:
        return False
    return all(math.isfinite(v) and abs(v) < 1e18 for v in bounds)


def _clean_metric_gdf(gdf: gpd.GeoDataFrame, *, assumed_crs: str, metric_crs: str) -> tuple[gpd.GeoDataFrame, int]:
    if gdf.empty:
        return gdf, 0
    if gdf.crs is None:
        gdf = gdf.set_crs(assumed_crs)
    if str(gdf.crs) != metric_crs:
        gdf = gdf.to_crs(metric_crs)
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    if not gdf.empty:
        gdf = gdf[gdf.geometry.apply(_geometry_finite)].copy()
    if not gdf.empty:
        areas = gdf.geometry.area
        gdf = gdf[areas <= MAX_PLAUSIBLE_POLY_M2].copy()
    return gdf.reset_index(drop=True), before - len(gdf)


def _read_largest_layer(path: Path) -> tuple[gpd.GeoDataFrame, str | None]:
    layers = pyogrio.list_layers(str(path))
    best_layer = None
    best_gdf = None
    best_count = -1
    for layer_name, _geometry_type in layers:
        gdf = gpd.read_file(str(path), layer=layer_name)
        if len(gdf) > best_count:
            best_layer = str(layer_name)
            best_gdf = gdf
            best_count = len(gdf)
    if best_gdf is None:
        return gpd.read_file(str(path)), None
    return best_gdf, best_layer


def _read_layer_or_largest(path: Path, layer: str | None) -> tuple[gpd.GeoDataFrame, str | None]:
    if layer:
        available = {str(row[0]) for row in pyogrio.list_layers(str(path))}
        if layer in available:
            return gpd.read_file(str(path), layer=layer), layer
    return _read_largest_layer(path)


def _is_exact_grid_match(stem: str, grid_id: str) -> bool:
    """Guard against partial-prefix matches (e.g. G1238 vs G12380)."""
    if not stem.startswith(grid_id):
        return False
    rest = stem[len(grid_id):]
    # Next char must be non-alphanumeric, or end-of-name
    return not rest or not rest[0].isalnum()


def _find_gpkg_in_dir(ann_dir: Path, grid_id: str) -> Path | None:
    if not ann_dir.exists():
        return None
    candidates = [p for p in ann_dir.glob(f"{grid_id}*.gpkg") if _is_exact_grid_match(p.stem, grid_id)]
    return sorted(candidates)[-1] if candidates else None


def _reviewed_gt_spec(region: str, grid_id: str) -> tuple[Path, str | None] | None:
    """Default (reviewed) GT from regions.yaml."""
    cfg = get_region_config(region)
    entry = cfg.grids.get(grid_id) or {}
    src = entry.get("annotation_source")
    layer = entry.get("annotation_layer")
    if src:
        path = _resolve(src)
        if path.exists():
            return path, layer
    # Conservative fallback: only exact grid-prefixed GPKGs in configured annotation dir.
    ann_dir = _resolve(cfg.paths.annotations_dir)
    found = _find_gpkg_in_dir(ann_dir, grid_id)
    if found:
        return found, None
    return None


def _li_gt_spec(region: str, grid_id: str) -> tuple[Path, str | None] | None:
    """Independent Li human annotation GT. Returns None if no Li file for this grid."""
    for ann_dir in LI_GT_DIRS.get(region, []):
        found = _find_gpkg_in_dir(ann_dir, grid_id)
        if found:
            return found, None
    return None


# Back-compat alias; nothing else should call this, but keep it so external scripts
# that may have imported `_gt_spec` don't break silently.
def _gt_spec(region: str, grid_id: str) -> tuple[Path, str | None] | None:
    return _reviewed_gt_spec(region, grid_id)


def _union_area(gdf: gpd.GeoDataFrame) -> tuple[Any, float]:
    if gdf.empty:
        return None, 0.0
    geom = unary_union(gdf.geometry.tolist())
    return geom, float(geom.area) if geom and not geom.is_empty else 0.0


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def evaluate_pair(gt: gpd.GeoDataFrame, pred: gpd.GeoDataFrame) -> dict[str, float]:
    gt_union, gt_area = _union_area(gt)
    pred_union, pred_area = _union_area(pred)
    if gt_area > 0 and pred_area > 0:
        intersection_area = float(gt_union.intersection(pred_union).area)
    else:
        intersection_area = 0.0

    area_precision = _safe_ratio(intersection_area, pred_area)
    area_recall = _safe_ratio(intersection_area, gt_area)
    area_f1 = _safe_ratio(2 * area_precision * area_recall, area_precision + area_recall) if (area_precision + area_recall) else 0.0

    gt_coverages: list[float] = []
    pred_counts_per_gt: list[int] = []
    pred_sindex = pred.sindex if not pred.empty else None
    for gt_geom in gt.geometry:
        inter_area = 0.0
        count = 0
        if pred_sindex is not None:
            for pred_idx in pred_sindex.intersection(gt_geom.bounds):
                pred_geom = pred.iloc[pred_idx].geometry
                if not gt_geom.intersects(pred_geom):
                    continue
                ia = float(gt_geom.intersection(pred_geom).area)
                if ia > 0:
                    inter_area += ia
                    count += 1
        gt_coverages.append(min(1.0, _safe_ratio(inter_area, float(gt_geom.area))))
        pred_counts_per_gt.append(count)

    pred_purities: list[float] = []
    gt_counts_per_pred: list[int] = []
    gt_sindex = gt.sindex if not gt.empty else None
    for pred_geom in pred.geometry:
        inter_area = 0.0
        count = 0
        if gt_sindex is not None:
            for gt_idx in gt_sindex.intersection(pred_geom.bounds):
                gt_geom = gt.iloc[gt_idx].geometry
                if not pred_geom.intersects(gt_geom):
                    continue
                ia = float(pred_geom.intersection(gt_geom).area)
                if ia > 0:
                    inter_area += ia
                    count += 1
        pred_purities.append(min(1.0, _safe_ratio(inter_area, float(pred_geom.area))))
        gt_counts_per_pred.append(count)

    return {
        "gt_union_area_m2": gt_area,
        "pred_union_area_m2": pred_area,
        "intersection_area_m2": intersection_area,
        "area_precision": area_precision,
        "area_recall": area_recall,
        "area_f1": area_f1,
        "pred_gt_area_ratio": _safe_ratio(pred_area, gt_area),
        "signed_area_error": _safe_ratio(pred_area - gt_area, gt_area),
        "abs_area_error": _safe_ratio(abs(pred_area - gt_area), gt_area),
        "gt_coverage_mean": sum(gt_coverages) / len(gt_coverages) if gt_coverages else 0.0,
        "gt_coverage_median": _median(gt_coverages),
        "gt_coverage_ge_05_rate": sum(v >= 0.5 for v in gt_coverages) / len(gt_coverages) if gt_coverages else 0.0,
        "gt_coverage_ge_08_rate": sum(v >= 0.8 for v in gt_coverages) / len(gt_coverages) if gt_coverages else 0.0,
        "pred_purity_mean": sum(pred_purities) / len(pred_purities) if pred_purities else 0.0,
        "pred_purity_median": _median(pred_purities),
        "pred_purity_ge_05_rate": sum(v >= 0.5 for v in pred_purities) / len(pred_purities) if pred_purities else 0.0,
        "pred_purity_ge_08_rate": sum(v >= 0.8 for v in pred_purities) / len(pred_purities) if pred_purities else 0.0,
        "gt_with_any_pred_rate": sum(v > 0 for v in gt_coverages) / len(gt_coverages) if gt_coverages else 0.0,
        "pred_with_any_gt_rate": sum(v > 0 for v in pred_purities) / len(pred_purities) if pred_purities else 0.0,
        "gt_with_multi_pred_intersections_rate": sum(c > 1 for c in pred_counts_per_gt) / len(pred_counts_per_gt) if pred_counts_per_gt else 0.0,
        "pred_with_multi_gt_intersections_rate": sum(c > 1 for c in gt_counts_per_pred) / len(gt_counts_per_pred) if gt_counts_per_pred else 0.0,
        "mean_pred_intersections_per_gt": sum(pred_counts_per_gt) / len(pred_counts_per_gt) if pred_counts_per_gt else 0.0,
        "mean_gt_intersections_per_pred": sum(gt_counts_per_pred) / len(gt_counts_per_pred) if gt_counts_per_pred else 0.0,
    }


def read_tracker_rows() -> list[dict[str, str]]:
    if not INPUT_TRACKER.exists():
        raise FileNotFoundError(f"Missing {INPUT_TRACKER}; run build_model_grid_metrics_tracker.py first")
    with INPUT_TRACKER.open() as fh:
        return list(csv.DictReader(fh))


def discover_prediction_rows() -> list[dict[str, str]]:
    """Walk results/ for every predictions_metric.gpkg and synthesize tracker-shaped rows.

    Many grids have predictions but no presence_metrics.csv (inference-only runs).
    Those are invisible to the grid-metrics tracker but still evaluable against Li GT.
    Rows produced here mimic the shape of model_grid_metrics_tracker.csv rows so the
    downstream evaluation loop is unchanged.
    """
    base = REPO_ROOT / "results"
    if not base.exists():
        return []
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pred_path in sorted(base.rglob("predictions_metric.gpkg")):
        result_dir = pred_path.parent
        rel = result_dir.relative_to(REPO_ROOT)
        parts = rel.parts
        # Skip non-grid-leaf dirs (predictions always live at the grid leaf)
        cfg = _read_config_json(result_dir / "config.json")
        model_run, ckpt = _infer_model_run(parts, cfg)
        region = _infer_region(parts)
        grid_id = _infer_grid_id(parts, "")
        if not (region and grid_id and model_run):
            continue
        key = (region, model_run, grid_id)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "region": region,
            "model_family": _family_of_run(model_run, ckpt),
            "model_run": model_run,
            "model_checkpoint": ckpt,
            "imagery_layer": _infer_imagery_layer(parts, cfg),
            "grid_id": grid_id,
            "result_dir": str(rel),
            "is_valid_eval": "False",
        })
    return out


def merge_row_sources(
    tracker: list[dict[str, str]],
    discovered: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Tracker rows take precedence for (region, model_run, grid_id) keys; discovered
    rows fill in the long tail of inference-only predictions."""
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in tracker:
        key = (r.get("region", ""), r.get("model_run", ""), r.get("grid_id", ""))
        by_key[key] = r
    for r in discovered:
        key = (r.get("region", ""), r.get("model_run", ""), r.get("grid_id", ""))
        by_key.setdefault(key, r)
    return list(by_key.values())


def _benchmark_alias(r: dict[str, str], gt_source_type: str) -> str:
    """Stable report alias for historical Li benchmark artifacts."""
    if gt_source_type != "li" or r.get("region") != "jhb":
        return ""
    run = r.get("model_run", "")
    if run == "v3c_geid_2024_02":
        return "v3c_vs_li_20260419"
    if run == "v4_aerial_2023":
        return "v4_vs_li_20260419"
    return ""


def _blank_out_row(r: dict[str, str], gt_source_type: str) -> dict[str, Any]:
    out = {k: "" for k in COLUMNS}
    for k in ["region", "model_family", "model_run", "imagery_layer", "grid_id", "result_dir"]:
        out[k] = r.get(k, "")
    out["gt_source_type"] = gt_source_type
    out["benchmark_alias"] = _benchmark_alias(r, gt_source_type)
    return out


def _evaluate_single_gt(
    r: dict[str, str],
    region: str,
    grid_id: str,
    pred_path: Path,
    gt_source_type: str,
    gt_path: Path,
    gt_layer: str | None,
) -> dict[str, Any]:
    out = _blank_out_row(r, gt_source_type)
    metric_crs = get_metric_crs(grid_id, region=region)
    try:
        gt_raw, chosen_layer = _read_layer_or_largest(gt_path, gt_layer)
        gt, n_drop_gt = _clean_metric_gdf(gt_raw, assumed_crs="EPSG:4326", metric_crs=metric_crs)
        pred_raw = gpd.read_file(str(pred_path))
        pred, n_drop_pred = _clean_metric_gdf(pred_raw, assumed_crs=metric_crs, metric_crs=metric_crs)
        metrics = evaluate_pair(gt, pred)
    except Exception as exc:
        out["eval_status"] = "error"
        out["skip_reason"] = repr(exc)
        out["gt_source"] = (
            str(gt_path.relative_to(REPO_ROOT)) if gt_path.is_relative_to(REPO_ROOT) else str(gt_path)
        )
        return out

    out.update(metrics)
    out.update({
        "gt_source": str(gt_path.relative_to(REPO_ROOT)) if gt_path.is_relative_to(REPO_ROOT) else str(gt_path),
        "gt_layer": chosen_layer or "",
        "gt_polygon_count": len(gt),
        "pred_polygon_count": len(pred),
        "n_dropped_gt": n_drop_gt,
        "n_dropped_pred": n_drop_pred,
        "eval_status": "ok",
        "skip_reason": "",
    })
    return out


def evaluate_tracker_row(r: dict[str, str]) -> list[dict[str, Any]]:
    """Emit one output row per available GT source (reviewed + Li).

    Li GT is evaluated independently of ``is_valid_eval`` — the whole point of
    the Li benchmark is that it does not rely on prior reviewed eval.
    """
    region = REGION_KEY.get(r.get("region", ""), "")
    grid_id = r.get("grid_id", "")
    if not region or not grid_id:
        out = _blank_out_row(r, "")
        out["eval_status"] = "skipped"
        out["skip_reason"] = "missing region/grid_id"
        return [out]

    pred_path = REPO_ROOT / r["result_dir"] / "predictions_metric.gpkg"
    if not pred_path.exists():
        out = _blank_out_row(r, "")
        out["eval_status"] = "skipped"
        out["skip_reason"] = f"missing predictions_metric.gpkg: {pred_path.relative_to(REPO_ROOT)}"
        return [out]

    rows: list[dict[str, Any]] = []

    reviewed = _reviewed_gt_spec(region, grid_id)
    if reviewed is not None:
        rows.append(_evaluate_single_gt(r, region, grid_id, pred_path, "reviewed", reviewed[0], reviewed[1]))
    # else: silently skip the reviewed row when GT is missing — we don't want
    # to double-count "no Li, no reviewed" as two skipped rows.

    li = _li_gt_spec(region, grid_id)
    if li is not None:
        rows.append(_evaluate_single_gt(r, region, grid_id, pred_path, "li", li[0], li[1]))

    if not rows:
        out = _blank_out_row(r, "")
        out["eval_status"] = "skipped"
        out["skip_reason"] = "no GT source (neither reviewed nor Li)"
        return [out]
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-run summary grouped by (region, model_family, model_run, gt_source_type).

    Micro-averaging: totals of intersection/pred/gt area across grids drive P/R/F1.
    This matches DeepSolar-style aggregate reporting and avoids macro bias from
    small grids.
    """
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("eval_status") == "ok":
            key = (r["region"], r["model_family"], r["model_run"], r.get("gt_source_type", ""))
            groups[key].append(r)

    summaries: list[dict[str, Any]] = []
    for (region, fam, run, gt_type), items in sorted(groups.items()):
        inter = sum(float(x["intersection_area_m2"]) for x in items)
        pred = sum(float(x["pred_union_area_m2"]) for x in items)
        gt = sum(float(x["gt_union_area_m2"]) for x in items)
        ap = _safe_ratio(inter, pred)
        ar = _safe_ratio(inter, gt)
        af1 = _safe_ratio(2 * ap * ar, ap + ar) if (ap + ar) else 0.0
        summaries.append({
            "region": region,
            "model_family": fam,
            "model_run": run,
            "gt_source_type": gt_type,
            "benchmark_alias": items[0].get("benchmark_alias", ""),
            "n_grids": len(items),
            "gt_count": sum(int(x["gt_polygon_count"]) for x in items),
            "pred_count": sum(int(x["pred_polygon_count"]) for x in items),
            "area_precision": ap,
            "area_recall": ar,
            "area_f1": af1,
            "pred_gt_area_ratio": _safe_ratio(pred, gt),
            "signed_area_error": _safe_ratio(pred - gt, gt),
            "mean_gt_coverage_ge_05_rate": sum(float(x["gt_coverage_ge_05_rate"]) for x in items) / len(items),
            "mean_gt_coverage_ge_08_rate": sum(float(x["gt_coverage_ge_08_rate"]) for x in items) / len(items),
            "mean_pred_purity_ge_05_rate": sum(float(x["pred_purity_ge_05_rate"]) for x in items) / len(items),
            "mean_pred_purity_ge_08_rate": sum(float(x["pred_purity_ge_08_rate"]) for x in items) / len(items),
        })
    return summaries


def write_outputs(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in COLUMNS})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ok = [r for r in rows if r.get("eval_status") == "ok"]
    skipped = [r for r in rows if r.get("eval_status") != "ok"]
    md: list[str] = []
    md.append("# Model Area/Coverage Tracker")
    md.append("")
    md.append(f"Generated: {now}")
    md.append(f"Source grid tracker: `{INPUT_TRACKER.relative_to(REPO_ROOT)}`")
    md.append(f"Rows evaluated: {len(ok)} ok, {len(skipped)} skipped/error")
    md.append("")
    md.append("**Metric frame** (DeepSolar-style geometry-overlap, complementary to installation TP/FP/FN):")
    md.append("- **area F1** = headline. Harmonic mean of area precision (pred ∩ GT / pred) and area recall (pred ∩ GT / GT).")
    md.append("- **pred/GT** = total predicted area / total GT area. >1 means model inflates total area, <1 deflates.")
    md.append("- **cov≥0.5** = fraction of GT polygons with ≥50% area covered by predictions (discovery).")
    md.append("- **purity≥0.5** = fraction of pred polygons with ≥50% area inside any GT (false-positive penalty).")
    md.append("- Read pred/GT and purity together — high area F1 with pred/GT ≫ 1 means model is over-predicting and getting lucky on overlap.")
    md.append("")
    md.append("Full CSV (`model_area_coverage_tracker.csv`) has all 36 columns including median variants and ≥0.8 thresholds; this MD shows the readable digest.")
    md.append("")

    def _render_narrow(subset: list[dict[str, Any]], best_metric: str = "area_f1", desc: bool = True) -> list[str]:
        """9-column narrow render, sorted by best_metric desc, ⭐ on top row."""
        out: list[str] = []
        out.append("| family | model_run | grids | **area F1** | area P | area R | pred/GT | cov≥0.5 | purity≥0.5 |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        sorted_subset = sorted(subset, key=lambda s: s.get(best_metric, 0.0), reverse=desc)
        for i, s in enumerate(sorted_subset):
            star = " ⭐" if i == 0 else ""
            out.append(
                f"| {s['model_family']} | `{s['model_run']}` | {s['n_grids']} | "
                f"**{_fmt(s['area_f1'])}**{star} | {_fmt(s['area_precision'])} | {_fmt(s['area_recall'])} | "
                f"{_fmt(s['pred_gt_area_ratio'])} | "
                f"{_fmt(s['mean_gt_coverage_ge_05_rate'])} | {_fmt(s['mean_pred_purity_ge_05_rate'])} |"
            )
        return out

    def _imagery_bucket(run: str) -> str:
        """Group key for splitting Li runs by imagery layer."""
        if "geid_2024_02" in run:
            return "geid_2024_02"
        if "aerial_2023" in run:
            return "aerial_2023"
        if "aerial_2025" in run:
            return "aerial_2025"
        return "other"

    li_summaries = [s for s in summaries if s.get("gt_source_type") == "li"]
    reviewed_summaries = [s for s in summaries if s.get("gt_source_type") == "reviewed"]

    # ---- TL;DR ----
    li_jhb_cbd = [s for s in li_summaries if _imagery_bucket(s["model_run"]) == "geid_2024_02"]
    if li_jhb_cbd:
        top3 = sorted(li_jhb_cbd, key=lambda s: s.get("area_f1", 0), reverse=True)[:3]
        md.append("## TL;DR — top runs on V1.4 main canon (25-grid JHB CBD GEID × Li GT)")
        md.append("")
        md.append("| rank | model_run | area F1 | area R | pred/GT | cov≥0.5 |")
        md.append("|---|---|---:|---:|---:|---:|")
        for i, s in enumerate(top3):
            md.append(
                f"| {i+1} | `{s['model_run']}` | **{_fmt(s['area_f1'])}** | {_fmt(s['area_recall'])} | "
                f"{_fmt(s['pred_gt_area_ratio'])} | {_fmt(s['mean_gt_coverage_ge_05_rate'])} |"
            )
        md.append("")

    # ---- Section: V1.4 main canon ----
    md.append("## V1.4 main canon: 25-grid JHB CBD GEID × independent Li human GT")
    md.append("")
    md.append("Headline area-eval frame. Li GT is independent of the model (no self-loop bias).")
    md.append("All runs use the same 25 grids, same chunked GEID 2024-02 imagery, same v4_canonical postproc.")
    md.append("Sorted by area F1 desc; ⭐ on the top row.")
    md.append("")
    if li_jhb_cbd:
        md.extend(_render_narrow(li_jhb_cbd))
    else:
        md.append("_(no geid_2024_02 Li-evaluated runs)_")
    md.append("")

    # ---- Section: Other Li runs (different imagery domains, not directly comparable) ----
    li_other = [s for s in li_summaries if _imagery_bucket(s["model_run"]) != "geid_2024_02"]
    if li_other:
        md.append("## Other Li GT runs (different imagery domains — not directly comparable to canon)")
        md.append("")
        md.append("Runs on `aerial_2023` / `aerial_2025` / other layers. Imagery vintage and resolution differ from")
        md.append("the GEID canon, so these area F1 values are not apples-to-apples with the canon table above.")
        md.append("")
        md.extend(_render_narrow(li_other))
        md.append("")

    # ---- Section: Reviewed GT (diagnostic only) ----
    md.append("## Reviewed-GT diagnostic (self-loop biased — for trend tracking only)")
    md.append("")
    md.append("Reviewed GT is derived from prior model predictions, so models trained on review-derived data")
    md.append("get inflated numbers here. Use as a within-family trend signal, not for cross-detector ranking.")
    md.append("Split by region and imagery bucket for clarity.")
    md.append("")
    if reviewed_summaries:
        # Group by (region, imagery_bucket) for readability
        rev_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for s in reviewed_summaries:
            rev_groups[(s["region"], _imagery_bucket(s["model_run"]))].append(s)
        for (region, bucket), items in sorted(rev_groups.items()):
            md.append(f"### {region} / {bucket} ({len(items)} runs)")
            md.append("")
            md.extend(_render_narrow(items))
            md.append("")
    else:
        md.append("_(no reviewed-GT-evaluated runs)_")
        md.append("")
    if skipped:
        counts: dict[str, int] = defaultdict(int)
        for r in skipped:
            counts[r.get("skip_reason", "unknown")] += 1
        md.append("## Skipped/error rows")
        md.append("")
        for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            md.append(f"- {n}: {reason}")
        md.append("")
    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    tracker_rows = read_tracker_rows()
    discovered_rows = discover_prediction_rows()
    merged = merge_row_sources(tracker_rows, discovered_rows)
    print(
        f"[discover] tracker={len(tracker_rows)} predictions_walk={len(discovered_rows)} "
        f"merged_unique={len(merged)}"
    )
    rows: list[dict[str, Any]] = []
    for r in merged:
        rows.extend(evaluate_tracker_row(r))
    summaries = summarize(rows)
    write_outputs(rows, summaries)
    print(f"[write] {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"[write] {OUTPUT_MD.relative_to(REPO_ROOT)}")
    ok_by_type: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("eval_status") == "ok":
            ok_by_type[r.get("gt_source_type", "")] += 1
    print(
        f"[summary] ok_reviewed={ok_by_type.get('reviewed', 0)} "
        f"ok_li={ok_by_type.get('li', 0)} total_rows={len(rows)} summaries={len(summaries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
