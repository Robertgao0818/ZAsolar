#!/usr/bin/env python3
"""Stage 3 of the direct pipeline: raw_detections.pkl → predictions_metric.gpkg.

Reads a raw artifact written by `detect_direct.py` (or refined by Phase 1.5
SAM) and produces:

  - `<output-dir>/predictions_metric.gpkg`   (metric CRS, full schema)
  - `<output-dir>/predictions.geojson`       (EPSG:4326 export)
  - `<output-dir>/config.json`               (provenance)
  - `<output-dir>/diagnostics.md`            (count diagnostics)

Pipeline:
  1. (optional) pre-vector score filter  (legacy `confidence_threshold`)
  2. paste each detection's mask_crop back, threshold, vectorize
     → keep largest component (default) or other policy
  3. assemble per-chip GDFs → grid-level GDF
  4. compute_geometric_properties (area_m2, elongation, solidity)
  5. compute_mask_mean_confidence
  6. compute_rgb_zonal_means GROUPED BY source_tif
  7. set confidence column (Phase 1: confidence = score)
  8. apply_postproc_filters (tiered conf, area, tiered elong, RGB)
  9. grid-level spatial_nms (IoU 0.5)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import box

from core import region_registry
from core.grid_utils import get_metric_crs, normalize_grid_id
from core.inference.raw_artifact import (
    PIPELINE_VERSION,
    RawArtifact,
    read_artifact,
)
from core.postproc import (
    DEFAULT_CONF_TIERED,
    DEFAULT_ELONGATION_TIERED,
    apply_postproc_filters,
    compute_geometric_properties,
    compute_mask_mean_confidence,
    compute_rgb_zonal_means,
    load_postproc_config,
    paint_and_vectorize_pixel_or,
    spatial_nms,
    vectorize_chip_mask,
)


CANONICAL_PIPELINE_VERSION = PIPELINE_VERSION  # "direct_maskrcnn_v1"


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="finalize.py",
        description="Vectorize + post-process a direct raw_detections artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="raw_detections.pkl (or refined_detections.pkl)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="output directory; refuses to overwrite a non-direct config.json")
    p.add_argument("--postproc-config", type=Path, default=None,
                   help="JSON config; defaults applied when omitted")
    p.add_argument("--strict-postproc-config", action="store_true",
                   help="Unknown keys raise instead of warn")
    p.add_argument("--merge-mode", choices=["pixel-or", "per-detection"], default="pixel-or",
                   help="pixel-or (default, geoai-equivalent): paint all detection masks "
                        "to a chunk-sized raster, OR-merge on overlap, vectorize once. "
                        "per-detection (diagnostic): each detection produces its own polygon, "
                        "merged later only via spatial_nms.")
    p.add_argument("--vectorize-multi-component",
                   choices=["largest", "union", "explode"], default="explode",
                   help="Only used in per-detection mode and for the leftover "
                        "'multi-component within one detection' case in pixel-or.")
    p.add_argument("--simplify-tolerance-pixels", type=float, default=0.0,
                   help="Pixel-space tolerance for shapely.simplify (0 = no simplify)")
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="IoU threshold for grid-level spatial_nms after vectorize")
    p.add_argument("--no-pre-vector-filter", action="store_true",
                   help="Disable legacy confidence_threshold → pre_vector cutoff")
    p.add_argument("--allow-overwrite-canonical", action="store_true",
                   help="Permit writing into a non-direct existing run dir")
    p.add_argument("--confidence-source", choices=["score", "mask_mean_confidence"],
                   default="score",
                   help="Phase 1 default = score; mask_mean_confidence reserved for ablations")
    return p


# ─────────────────────────────────────────────────────────────────────────
# Vectorize one chip
# ─────────────────────────────────────────────────────────────────────────
def _vectorize_chip(
    chip,
    *,
    mask_threshold: float,
    multi_component: str,
    simplify_tolerance_pixels: float,
) -> gpd.GeoDataFrame:
    """Build a per-chip GeoDataFrame in source_crs."""
    rows = []
    chip_size = int(chip.chip_shape[0])
    for det in chip.detections:
        # Build the chip's affine transform from the stored 6-tuple.
        win_tr = Affine(*chip.window_transform)
        result = vectorize_chip_mask(
            det.mask_crop_uint8,
            tuple(det.mask_crop_offset),
            threshold=mask_threshold,
            window_transform=win_tr,
            source_crs=chip.source_crs,
            multi_component=multi_component,
            simplify_tolerance_pixels=simplify_tolerance_pixels,
        )

        # Clip to chip valid window (reject polygons outside real raster data).
        valid = chip.valid_window  # (col_off, row_off, w, h)
        # Convert to source-CRS clipping bounds via window_transform.
        # valid_window is in *source-TIF pixel space*: convert pixel→world
        src_tr = Affine(*chip.source_transform)
        cx0, cy0 = src_tr * (valid[0], valid[1])
        cx1, cy1 = src_tr * (valid[0] + valid[2], valid[1] + valid[3])
        x_min, x_max = sorted([cx0, cx1])
        y_min, y_max = sorted([cy0, cy1])
        clip_geom = box(x_min, y_min, x_max, y_max)

        for poly in result.geoms:
            if poly is None or poly.is_empty:
                continue
            try:
                clipped = poly.intersection(clip_geom)
            except Exception:
                clipped = poly
            if clipped.is_empty:
                continue
            rows.append({
                "geometry": clipped,
                "score": det.score,
                "label": det.label,
                "chip_index": chip.chip_index,
                "source_tile": chip.source_tile_id,
                "source_tif": chip.source_tif,
                "source_detection_index": det.source_detection_index,
                "n_components_dropped": result.n_components_dropped,
                # We need the soft-mask data for mask_mean_confidence later;
                # carry crop + offset along.
                "_mask_crop": det.mask_crop_uint8,
                "_mask_offset": tuple(det.mask_crop_offset),
            })

    if not rows:
        return gpd.GeoDataFrame(
            columns=[
                "geometry", "score", "label", "chip_index",
                "source_tile", "source_tif", "source_detection_index",
                "n_components_dropped", "_mask_crop", "_mask_offset",
            ],
            geometry="geometry",
            crs=chip.source_crs,
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=chip.source_crs)


def _vectorize_pixel_or(
    artifact,
    *,
    mask_threshold: float,
    multi_component: str,
    simplify_tolerance_pixels: float,
) -> list[gpd.GeoDataFrame]:
    """Pixel-OR path: per source TIF, paint all detections to a chunk-sized
    raster and vectorize once.

    Each source TIF emits its own GeoDataFrame in source_crs; caller concats
    them into a grid-level GDF and reprojects to metric.
    """
    # Build a lookup: source_tif -> SourceTile metadata
    src_meta_by_path = {s.path: s for s in artifact.source_tiles}

    # Group detections by source TIF
    grouped: dict[str, list[dict]] = {}
    for chip in artifact.chips:
        if not chip.detections:
            continue
        col_off, row_off, _w, _h = chip.window
        for det in chip.detections:
            grouped.setdefault(chip.source_tif, []).append({
                "mask_crop_uint8": det.mask_crop_uint8,
                "source_offset": (
                    int(col_off) + int(det.mask_crop_offset[0]),
                    int(row_off) + int(det.mask_crop_offset[1]),
                ),
                "score": float(det.score),
                "label": int(det.label),
                "chip_index": int(chip.chip_index),
                "source_detection_index": int(det.source_detection_index),
            })

    chip_gdfs: list[gpd.GeoDataFrame] = []
    for source_tif, detections in grouped.items():
        meta = src_meta_by_path.get(source_tif)
        if meta is None:
            continue
        from affine import Affine
        src_tr = Affine(*meta.transform)
        H, W = meta.shape

        painted = paint_and_vectorize_pixel_or(
            detections,
            raster_height=int(H),
            raster_width=int(W),
            source_transform=src_tr,
            source_crs=meta.crs,
            mask_threshold=mask_threshold,
            multi_component="explode",  # always explode in pixel-or; ORed mask is one connected blob per panel cluster
            simplify_tolerance_pixels=simplify_tolerance_pixels,
        )
        if not painted:
            continue
        rows = []
        from pathlib import Path as _P
        for p in painted:
            rows.append({
                "geometry": p.geom,
                "score": p.score,
                "label": p.label,
                "mask_mean_confidence_painted": p.mask_mean_confidence,
                "chip_index": -1,                 # multi-chip merge: unset
                "source_tile": _P(source_tif).stem,
                "source_tif": source_tif,
                "source_detection_index": -1,
                "n_components_dropped": 0,
                "_mask_crop": None,
                "_mask_offset": (0, 0),
            })
        if rows:
            chip_gdfs.append(gpd.GeoDataFrame(rows, geometry="geometry", crs=meta.crs))
    return chip_gdfs


# ─────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> int:
    # ── Load artifact ────────────────────────────────────────────────
    artifact: RawArtifact = read_artifact(args.input)
    if artifact.pipeline_version != CANONICAL_PIPELINE_VERSION:
        print(f"[WARN] artifact pipeline_version={artifact.pipeline_version!r} "
              f"(expected {CANONICAL_PIPELINE_VERSION!r}); proceeding anyway")

    grid_id = artifact.grid_id
    region_arg = artifact.region_arg
    region_key = artifact.region_key
    print(f"[finalize] grid={grid_id} region={region_arg} "
          f"chips={len(artifact.chips)} layer={artifact.imagery_layer_id}")

    # ── Output dir + canonical guard ─────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
            existing_pipeline = existing.get("pipeline_version", "")
            if existing_pipeline != CANONICAL_PIPELINE_VERSION and not args.allow_overwrite_canonical:
                sys.exit(
                    f"[finalize] refusing to overwrite {args.output_dir}: "
                    f"existing config.json has pipeline_version={existing_pipeline!r}, "
                    f"expected {CANONICAL_PIPELINE_VERSION!r}. "
                    "Use --allow-overwrite-canonical to override."
                )
        except Exception:
            pass

    # ── Postproc config ──────────────────────────────────────────────
    postproc_cfg: dict[str, Any] = {}
    if args.postproc_config is not None:
        postproc_cfg = load_postproc_config(args.postproc_config, strict=args.strict_postproc_config)
        print(f"[finalize] postproc config: {args.postproc_config.name} → {sorted(postproc_cfg.keys())}")
    # Defaults from V1.4 plan
    postproc_cfg.setdefault("elongation_tiered", DEFAULT_ELONGATION_TIERED)
    postproc_cfg.setdefault("conf_tiered", DEFAULT_CONF_TIERED)
    postproc_cfg.setdefault("min_object_area", 5.0)
    postproc_cfg.setdefault("shadow_rgb_thresh", 60)
    postproc_cfg.setdefault("over_bright_thresh", 250)
    mask_threshold = float(postproc_cfg.get("mask_threshold", artifact.mask_threshold_used))

    # ── Pre-vector score filter (legacy confidence_threshold → pre_vector) ──
    pre_vector_thresh = postproc_cfg.get("pre_vector_score_threshold")
    n_raw_total = sum(len(c.detections) for c in artifact.chips)
    n_after_pre_vector = n_raw_total
    if pre_vector_thresh is not None and not args.no_pre_vector_filter:
        cutoff = float(pre_vector_thresh)
        kept_chips = []
        for c in artifact.chips:
            kept = [d for d in c.detections if d.score >= cutoff]
            kept_chips.append(c.__class__(
                chip_index=c.chip_index,
                source_tif=c.source_tif,
                source_tile_id=c.source_tile_id,
                source_crs=c.source_crs,
                source_transform=c.source_transform,
                window=c.window,
                window_transform=c.window_transform,
                valid_window=c.valid_window,
                valid_shape=c.valid_shape,
                chip_shape=c.chip_shape,
                detections=kept,
            ))
        artifact.chips = kept_chips
        n_after_pre_vector = sum(len(c.detections) for c in artifact.chips)
        print(f"[finalize] pre-vector score filter (≥{cutoff}): "
              f"{n_raw_total} → {n_after_pre_vector}")

    # ── Vectorize ────────────────────────────────────────────────────
    if args.merge_mode == "pixel-or":
        chip_gdfs = _vectorize_pixel_or(
            artifact,
            mask_threshold=mask_threshold,
            multi_component=args.vectorize_multi_component,
            simplify_tolerance_pixels=args.simplify_tolerance_pixels,
        )
    else:
        chip_gdfs = []
        for chip in artifact.chips:
            if not chip.detections:
                continue
            g = _vectorize_chip(
                chip,
                mask_threshold=mask_threshold,
                multi_component=args.vectorize_multi_component,
                simplify_tolerance_pixels=args.simplify_tolerance_pixels,
            )
            if len(g) > 0:
                chip_gdfs.append(g)

    if not chip_gdfs:
        print("[finalize] no polygons after vectorization — writing empty outputs")
        _write_outputs(
            args=args,
            grid_id=grid_id,
            region_key=region_key,
            artifact=artifact,
            pred_metric=gpd.GeoDataFrame(columns=_min_schema_cols(),
                                         geometry="geometry",
                                         crs=get_metric_crs(grid_id, region=region_key)),
            postproc_cfg=postproc_cfg,
            n_raw_total=n_raw_total,
            n_after_pre_vector=n_after_pre_vector,
            n_after_postproc=0,
            n_after_nms=0,
        )
        return 0

    pred = gpd.GeoDataFrame(pd.concat(chip_gdfs, ignore_index=True),
                            geometry="geometry",
                            crs=chip_gdfs[0].crs)
    print(f"[finalize] vectorized → {len(pred)} polygons "
          f"(in {pred.crs})")

    # ── Reproject to metric CRS ──────────────────────────────────────
    metric_crs = get_metric_crs(grid_id, region=region_key)
    pred_metric = pred.to_crs(metric_crs)

    # ── Geometric properties ─────────────────────────────────────────
    pred_metric = compute_geometric_properties(pred_metric)

    # ── Mask mean confidence ─────────────────────────────────────────
    if "mask_mean_confidence_painted" in pred_metric.columns:
        # pixel-OR mode already computed it from the merged raster
        pred_metric["mask_mean_confidence"] = pred_metric["mask_mean_confidence_painted"].astype(float)
        pred_metric = pred_metric.drop(columns=["mask_mean_confidence_painted"])
    else:
        masks_by_index = {}
        for i, row in pred_metric.iterrows():
            if row["_mask_crop"] is not None:
                masks_by_index[int(i)] = (row["_mask_crop"], row["_mask_offset"])
        pred_metric["mask_mean_confidence"] = compute_mask_mean_confidence(
            list(pred_metric.index), masks_by_index, mask_threshold=mask_threshold,
        )

    # ── RGB zonal means GROUPED BY source_tif ────────────────────────
    pred_metric["mean_r"] = 0.0
    pred_metric["mean_g"] = 0.0
    pred_metric["mean_b"] = 0.0
    for src_tif, group in pred_metric.groupby("source_tif"):
        # Reproject the group's geometries to that source TIF's CRS for sampling.
        # Look up source CRS from the artifact's source_tiles list.
        src_meta = next((s for s in artifact.source_tiles if s.path == src_tif), None)
        if src_meta is None:
            continue
        try:
            sub = group.to_crs(src_meta.crs)
        except Exception:
            sub = group
        mr, mg, mb = compute_rgb_zonal_means(sub, src_tif)
        pred_metric.loc[group.index, "mean_r"] = mr
        pred_metric.loc[group.index, "mean_g"] = mg
        pred_metric.loc[group.index, "mean_b"] = mb

    # ── Confidence column (Phase 1: confidence = score) ─────────────
    pred_metric["confidence"] = pred_metric[args.confidence_source].astype(float)
    print(f"[finalize] confidence_source = {args.confidence_source}")

    # ── Post-proc filters ────────────────────────────────────────────
    pred_filtered, stats = apply_postproc_filters(pred_metric, postproc_cfg)
    n_after_postproc = len(pred_filtered)
    print(f"[finalize] postproc filters: {stats}")

    # ── Grid-level spatial NMS ───────────────────────────────────────
    pred_nms = spatial_nms(pred_filtered, iou_threshold=float(args.nms_iou))
    n_after_nms = len(pred_nms)
    print(f"[finalize] grid-level spatial_nms (IoU={args.nms_iou}): "
          f"{n_after_postproc} → {n_after_nms}")

    # ── Write outputs ────────────────────────────────────────────────
    _write_outputs(
        args=args,
        grid_id=grid_id,
        region_key=region_key,
        artifact=artifact,
        pred_metric=pred_nms,
        postproc_cfg=postproc_cfg,
        n_raw_total=n_raw_total,
        n_after_pre_vector=n_after_pre_vector,
        n_after_postproc=n_after_postproc,
        n_after_nms=n_after_nms,
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────
def _min_schema_cols() -> list[str]:
    return [
        "geometry", "score", "mask_mean_confidence", "confidence",
        "area_m2", "elongation", "solidity",
        "mean_r", "mean_g", "mean_b",
        "source_tile", "source_tif", "chip_index", "label",
    ]


def _write_outputs(
    *,
    args: argparse.Namespace,
    grid_id: str,
    region_key: str,
    artifact: RawArtifact,
    pred_metric: gpd.GeoDataFrame,
    postproc_cfg: dict[str, Any],
    n_raw_total: int,
    n_after_pre_vector: int,
    n_after_postproc: int,
    n_after_nms: int,
) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop internal helper columns; ensure minimum schema columns exist.
    drop_cols = [c for c in ("_mask_crop", "_mask_offset") if c in pred_metric.columns]
    pred_clean = pred_metric.drop(columns=drop_cols, errors="ignore").copy()
    for col in _min_schema_cols():
        if col not in pred_clean.columns:
            if col == "geometry":
                continue
            pred_clean[col] = np.nan

    gpkg_path = out_dir / "predictions_metric.gpkg"
    geojson_path = out_dir / "predictions.geojson"
    pred_clean.to_file(gpkg_path, driver="GPKG", layer="predictions")
    pred_4326 = pred_clean.to_crs("EPSG:4326")
    pred_4326.to_file(geojson_path, driver="GeoJSON")
    print(f"[finalize] wrote {gpkg_path}")
    print(f"[finalize] wrote {geojson_path}")

    # config.json
    artifact_hash = _file_sha256(args.input)
    postproc_hash = _file_sha256(args.postproc_config) if args.postproc_config else ""
    config = {
        "pipeline_version": CANONICAL_PIPELINE_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid_id": grid_id,
        "region_arg": artifact.region_arg,
        "region_key": region_key,
        "imagery_layer_id": artifact.imagery_layer_id,
        "model_run_id": artifact.model_run_id,
        "model_path": artifact.model_path,
        "model_sha256": artifact.model_sha256,
        "raw_artifact_path": str(args.input),
        "raw_artifact_sha256": artifact_hash,
        "postproc_config_path": str(args.postproc_config) if args.postproc_config else "",
        "postproc_config_sha256": postproc_hash,
        "postproc_config_resolved": _stringify_for_json(postproc_cfg),
        "confidence_source": args.confidence_source,
        "merge_mode": args.merge_mode,
        "nms_iou": float(args.nms_iou),
        "vectorize_multi_component": args.vectorize_multi_component,
        "simplify_tolerance_pixels": args.simplify_tolerance_pixels,
        "detector_score_threshold": artifact.detector_score_threshold,
        "detections_per_img": artifact.detections_per_img,
        "mask_threshold_used": artifact.mask_threshold_used,
        "result_count": int(n_after_nms),
        "stage_counts": {
            "raw_total": int(n_raw_total),
            "after_pre_vector_filter": int(n_after_pre_vector),
            "after_postproc": int(n_after_postproc),
            "after_nms": int(n_after_nms),
        },
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, default=str))
    print(f"[finalize] wrote {config_path}")

    # diagnostics.md
    diag = (
        f"# Direct pipeline diagnostics\n\n"
        f"- grid_id: {grid_id}\n"
        f"- region: {artifact.region_arg} ({region_key})\n"
        f"- imagery_layer: {artifact.imagery_layer_id}\n"
        f"- model_run: {artifact.model_run_id}\n"
        f"- model_path: {artifact.model_path}\n\n"
        f"## Stage counts\n\n"
        f"| Stage | Count |\n|---|---|\n"
        f"| Raw detector output (≥ score_thresh {artifact.detector_score_threshold}) | {n_raw_total} |\n"
        f"| After pre-vector score filter | {n_after_pre_vector} |\n"
        f"| After post-proc filters (area / elong / RGB / conf) | {n_after_postproc} |\n"
        f"| After grid-level spatial NMS | {n_after_nms} |\n"
    )
    (out_dir / "diagnostics.md").write_text(diag)
    print(f"[finalize] wrote {out_dir / 'diagnostics.md'}")


def _stringify_for_json(d: dict) -> dict:
    """JSON-safe copy: tuples → lists; numpy scalars → python scalars."""
    out = {}
    for k, v in d.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, list):
            out[k] = [list(x) if isinstance(x, tuple) else x for x in v]
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _file_sha256(path: str | Path | None) -> str:
    if path is None:
        return ""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
