#!/usr/bin/env python3
"""Freeze SAM-refined inventory outputs by applying a policy config.

This is a lightweight finalize step for the V1.4 V3-C+SAM route. It does not
run Mask R-CNN or SAM. It reads an existing SAM-refined result tree and writes a
new result tree with fixed threshold policy, provenance config, and corrected
area columns.

Typical usage:

  python scripts/analysis/filter_sam_inventory.py \
    --src-root results/johannesburg/v3c_sam_maskbox_vexcel_2024 \
    --config configs/postproc/v4_agg.json \
    --output-root results/johannesburg/v3c_sam_maskbox_vexcel_2024_v4_agg \
    --force

Threshold sweep against Channel 2 clean GT:

  python scripts/analysis/filter_sam_inventory.py \
    --src-root results/johannesburg/v3c_sam_maskbox_vexcel_2024 \
    --config configs/postproc/v4_agg.json \
    --sweep \
    --sweep-output results/analysis/v3c_sam_threshold_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.polygon_validation import geometry_finite  # noqa: E402


@dataclass(frozen=True)
class InventoryPolicy:
    tag: str
    detector_confidence_min: float
    sam_score_min: float
    refined_area_m2_min: float
    refined_area_m2_max: float
    update_area_columns: bool = True


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _policy_from_config(config_path: Path) -> InventoryPolicy:
    cfg = _read_json(config_path)
    meta = cfg.get("_meta", {}) or {}
    policy = meta.get("inventory_policy", {}) or {}
    filt = policy.get("filter", {}) or {}

    tag = str(policy.get("tag") or meta.get("tag") or config_path.stem)
    detector_conf = float(
        filt.get("detector_confidence_min", cfg.get("post_conf_threshold", 0.85))
    )
    sam_score = float(filt.get("sam_score_min", 0.0))
    area_min = float(filt.get("refined_area_m2_min", 0.0))
    area_max = float(filt.get("refined_area_m2_max", 20_000.0))
    update_area = bool(filt.get("update_area_columns", True))
    return InventoryPolicy(
        tag=tag,
        detector_confidence_min=detector_conf,
        sam_score_min=sam_score,
        refined_area_m2_min=area_min,
        refined_area_m2_max=area_max,
        update_area_columns=update_area,
    )


def _csv_floats(raw: str | None, fallback: list[float]) -> list[float]:
    if not raw:
        return fallback
    vals: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals or fallback


def _geometry_area_m2(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    return gdf.geometry.area.astype(float)


def _filter_gdf(
    gdf: gpd.GeoDataFrame,
    policy: InventoryPolicy,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    stats: dict[str, int] = {"input": int(len(gdf))}
    if gdf.empty:
        stats.update(
            {
                "after_valid_geometry": 0,
                "after_detector_confidence": 0,
                "after_sam_score": 0,
                "after_refined_area": 0,
            }
        )
        return gdf.copy(), stats

    out = gdf.copy()
    valid = out.geometry.notna() & out.geometry.is_valid & ~out.geometry.is_empty
    if valid.any():
        valid &= out.geometry.apply(geometry_finite)
    out = out[valid].copy()
    stats["after_valid_geometry"] = int(len(out))

    if "confidence" not in out.columns:
        raise ValueError("source predictions are missing required column: confidence")
    out = out[out["confidence"].astype(float) >= policy.detector_confidence_min].copy()
    stats["after_detector_confidence"] = int(len(out))

    if policy.sam_score_min > 0:
        if "sam_score" not in out.columns:
            raise ValueError("source predictions are missing required column: sam_score")
        out = out[out["sam_score"].astype(float) >= policy.sam_score_min].copy()
    stats["after_sam_score"] = int(len(out))

    if not out.empty:
        refined_area = _geometry_area_m2(out)
        keep = (
            (refined_area >= policy.refined_area_m2_min)
            & (refined_area <= policy.refined_area_m2_max)
        )
        out = out[keep].copy()
    stats["after_refined_area"] = int(len(out))

    if policy.update_area_columns and not out.empty:
        refined_area = _geometry_area_m2(out)
        if "area_m2" in out.columns and "pre_inventory_area_m2" not in out.columns:
            out["pre_inventory_area_m2"] = out["area_m2"].astype(float)
        out["area_m2"] = refined_area
        out["refined_area_m2"] = refined_area
        out["sam_area_m2"] = refined_area
        out["length_m"] = out.geometry.length.astype(float)
        out["perimeter_m"] = out.geometry.length.astype(float)

    return out.reset_index(drop=True), stats


def _empty_like(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.iloc[0:0].copy()


def _write_grid(
    grid_id: str,
    src_path: Path,
    out_dir: Path,
    filtered: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    config_path: Path,
    policy: InventoryPolicy,
    stats: dict[str, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if filtered.empty and len(filtered.columns) == 0:
        filtered = _empty_like(source_gdf)

    metric_path = out_dir / "predictions_metric.gpkg"
    export_path = out_dir / "predictions.geojson"
    filtered.to_file(metric_path, driver="GPKG")
    if filtered.crs is not None:
        filtered.to_crs("EPSG:4326").to_file(export_path, driver="GeoJSON")
    else:
        filtered.to_file(export_path, driver="GeoJSON")

    cfg = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid_id": grid_id,
        "source": str(src_path.relative_to(REPO_ROOT) if src_path.is_relative_to(REPO_ROOT) else src_path),
        "policy_config": str(config_path.relative_to(REPO_ROOT) if config_path.is_relative_to(REPO_ROOT) else config_path),
        "policy": policy.__dict__,
        "filter_stats": stats,
        "result_count": int(len(filtered)),
        "artifacts": {
            "predictions_metric": "predictions_metric.gpkg",
            "predictions_export": "predictions.geojson",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _grid_pred_paths(src_root: Path, grids: list[str] | None) -> list[tuple[str, Path]]:
    if grids:
        pairs = [(g, src_root / g / "predictions_metric.gpkg") for g in grids]
    else:
        pairs = [
            (p.parent.name, p)
            for p in sorted(src_root.glob("G*/predictions_metric.gpkg"))
        ]
    return [(grid, path) for grid, path in pairs if path.exists()]


def apply_policy(
    src_root: Path,
    output_root: Path,
    config_path: Path,
    policy: InventoryPolicy,
    grids: list[str] | None,
    *,
    force: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if output_root.exists() and any(output_root.iterdir()) and force and not dry_run:
        # Only remove the target run root. Source roots and sibling results are untouched.
        shutil.rmtree(output_root)
    elif output_root.exists() and any(output_root.iterdir()) and not dry_run:
        raise FileExistsError(f"output root already exists and is not empty: {output_root}")

    rows: list[dict[str, Any]] = []
    for grid_id, src_path in _grid_pred_paths(src_root, grids):
        source = gpd.read_file(src_path)
        filtered, stats = _filter_gdf(source, policy)
        row = {
            "grid_id": grid_id,
            "n_in": stats["input"],
            "n_out": stats["after_refined_area"],
            "drop_total": stats["input"] - stats["after_refined_area"],
            **stats,
        }
        rows.append(row)
        print(
            f"{grid_id}: {stats['input']} -> {stats['after_refined_area']} "
            f"(conf>={policy.detector_confidence_min}, sam>={policy.sam_score_min})"
        )
        if not dry_run:
            _write_grid(
                grid_id=grid_id,
                src_path=src_path,
                out_dir=output_root / grid_id,
                filtered=filtered,
                source_gdf=source,
                config_path=config_path,
                policy=policy,
                stats=stats,
            )

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "src_root": str(src_root.relative_to(REPO_ROOT) if src_root.is_relative_to(REPO_ROOT) else src_root),
            "output_root": str(output_root.relative_to(REPO_ROOT) if output_root.is_relative_to(REPO_ROOT) else output_root),
            "policy_config": str(config_path.relative_to(REPO_ROOT) if config_path.is_relative_to(REPO_ROOT) else config_path),
            "policy": policy.__dict__,
            "grids": rows,
        }
        (output_root / "_filter_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        with (output_root / "_filter_summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["grid_id"])
            writer.writeheader()
            writer.writerows(rows)
    return rows


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clean_metric(gdf: gpd.GeoDataFrame, metric_crs: str | None = None) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf.copy()
    if metric_crs and out.crs is not None and str(out.crs) != metric_crs:
        out = out.to_crs(metric_crs)
    valid = out.geometry.notna() & out.geometry.is_valid & ~out.geometry.is_empty
    if valid.any():
        valid &= out.geometry.apply(geometry_finite)
    return out[valid].reset_index(drop=True)


def _area_metrics(gt: gpd.GeoDataFrame, pred: gpd.GeoDataFrame) -> dict[str, float]:
    gt_union = unary_union(gt.geometry.tolist()) if len(gt) else None
    pred_union = unary_union(pred.geometry.tolist()) if len(pred) else None
    gt_area = float(gt_union.area) if gt_union and not gt_union.is_empty else 0.0
    pred_area = float(pred_union.area) if pred_union and not pred_union.is_empty else 0.0
    inter = (
        float(gt_union.intersection(pred_union).area)
        if gt_area > 0 and pred_area > 0
        else 0.0
    )
    area_p = _safe_ratio(inter, pred_area)
    area_r = _safe_ratio(inter, gt_area)
    area_f1 = _safe_ratio(2 * area_p * area_r, area_p + area_r) if area_p + area_r else 0.0
    return {
        "gt_area_m2": gt_area,
        "pred_area_m2": pred_area,
        "intersection_area_m2": inter,
        "area_precision": area_p,
        "area_recall": area_r,
        "area_f1": area_f1,
        "pred_gt_area_ratio": _safe_ratio(pred_area, gt_area),
    }


def run_sweep(
    src_root: Path,
    clean_gt_root: Path,
    sweep_output: Path,
    base_policy: InventoryPolicy,
    grids: list[str] | None,
    detector_conf_grid: list[float],
    sam_score_grid: list[float],
) -> list[dict[str, Any]]:
    pred_paths = _grid_pred_paths(src_root, grids)
    cache: list[tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame]] = []
    for grid_id, pred_path in pred_paths:
        gt_path = clean_gt_root / grid_id / f"{grid_id}_clean_gt.gpkg"
        if not gt_path.exists():
            continue
        pred = gpd.read_file(pred_path)
        gt = gpd.read_file(gt_path)
        if pred.crs is not None and gt.crs is not None and pred.crs != gt.crs:
            pred = pred.to_crs(gt.crs)
        pred = _clean_metric(pred, str(gt.crs) if gt.crs else None)
        gt = _clean_metric(gt)
        cache.append((grid_id, gt, pred))

    rows: list[dict[str, Any]] = []
    for detector_conf in detector_conf_grid:
        for sam_score in sam_score_grid:
            policy = replace(
                base_policy,
                detector_confidence_min=detector_conf,
                sam_score_min=sam_score,
            )
            total = {
                "gt_area_m2": 0.0,
                "pred_area_m2": 0.0,
                "intersection_area_m2": 0.0,
                "gt_count": 0,
                "pred_count": 0,
                "n_grids": 0,
            }
            for _grid_id, gt, pred in cache:
                filtered, _stats = _filter_gdf(pred, policy)
                metrics = _area_metrics(gt, filtered)
                total["gt_area_m2"] += metrics["gt_area_m2"]
                total["pred_area_m2"] += metrics["pred_area_m2"]
                total["intersection_area_m2"] += metrics["intersection_area_m2"]
                total["gt_count"] += int(len(gt))
                total["pred_count"] += int(len(filtered))
                total["n_grids"] += 1
            area_p = _safe_ratio(total["intersection_area_m2"], total["pred_area_m2"])
            area_r = _safe_ratio(total["intersection_area_m2"], total["gt_area_m2"])
            area_f1 = _safe_ratio(2 * area_p * area_r, area_p + area_r) if area_p + area_r else 0.0
            rows.append(
                {
                    "detector_confidence_min": detector_conf,
                    "sam_score_min": sam_score,
                    "n_grids": total["n_grids"],
                    "gt_count": total["gt_count"],
                    "pred_count": total["pred_count"],
                    "gt_area_m2": total["gt_area_m2"],
                    "pred_area_m2": total["pred_area_m2"],
                    "intersection_area_m2": total["intersection_area_m2"],
                    "area_precision": area_p,
                    "area_recall": area_r,
                    "area_f1": area_f1,
                    "pred_gt_area_ratio": _safe_ratio(total["pred_area_m2"], total["gt_area_m2"]),
                    "signed_area_error": _safe_ratio(
                        total["pred_area_m2"] - total["gt_area_m2"],
                        total["gt_area_m2"],
                    ),
                }
            )

    sweep_output.parent.mkdir(parents=True, exist_ok=True)
    with sweep_output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["detector_confidence_min"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WRITE] {sweep_output}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--grids", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--clean-gt-root",
        type=Path,
        default=REPO_ROOT / "data" / "annotations_channel2_clean",
    )
    parser.add_argument(
        "--sweep-output",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "v3c_sam_threshold_sweep.csv",
    )
    parser.add_argument(
        "--detector-conf-grid",
        default="0.85,0.87,0.89,0.91,0.92,0.93,0.95",
    )
    parser.add_argument("--sam-score-grid", default="0.0,0.96,0.98")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_root = _resolve(args.src_root)
    config_path = _resolve(args.config)
    policy = _policy_from_config(config_path)

    if args.sweep:
        run_sweep(
            src_root=src_root,
            clean_gt_root=_resolve(args.clean_gt_root),
            sweep_output=_resolve(args.sweep_output),
            base_policy=policy,
            grids=args.grids,
            detector_conf_grid=_csv_floats(args.detector_conf_grid, [policy.detector_confidence_min]),
            sam_score_grid=_csv_floats(args.sam_score_grid, [policy.sam_score_min]),
        )
        return 0

    output_root = _resolve(args.output_root) if args.output_root else src_root.parent / f"{src_root.name}_{policy.tag}"
    apply_policy(
        src_root=src_root,
        output_root=output_root,
        config_path=config_path,
        policy=policy,
        grids=args.grids,
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
