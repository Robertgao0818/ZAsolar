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
    compute_rgb_zonal_means,
    affine_pixel_area,
    dissolve_hairline_gaps,
    load_postproc_config,
    paint_geoai_parity_mask,
    paint_and_vectorize_pixel_or,
    parse_threshold_area_tiers,
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
    p.add_argument("--merge-mode", choices=["pixel-or", "per-detection"], default=None,
                   help="per-detection (default): vectorize each detection mask "
                        "before grid-level NMS. pixel-or: paint all detection masks "
                        "to a chunk-sized raster, OR-merge on overlap, vectorize once. "
                        "Can also be set in postproc config.")
    p.add_argument("--parity-mode", choices=["direct", "geoai"], default="direct",
                   help="direct = current direct finalizer; geoai = rebuild geoai-style "
                        "two-band mask rasters and vectorize through geoai.orthogonalize")
    p.add_argument("--vectorize-multi-component",
                   choices=["largest", "union", "explode"], default=None,
                   help="For per-detection masks: largest is the default. "
                        "For pixel-or masks: explode is the default. "
                        "Can also be set in postproc config.")
    p.add_argument("--simplify-tolerance-pixels", type=float, default=0.0,
                   help="Pixel-space tolerance for shapely.simplify (0 = no simplify)")
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="IoU threshold for grid-level spatial_nms after vectorize")
    p.add_argument("--dissolve-hairline-tolerance-m", type=float, default=0.0,
                   help="If >0, after spatial_nms merge polygon pairs whose "
                        "boundary-to-boundary distance ≤ tolerance metres. "
                        "Targets cat-1 (single installation split across TIF chunk seam). "
                        "0.5 m is safe — smaller than typical inter-installation gap. "
                        "0.0 (default) disables.")
    p.add_argument("--no-pre-vector-filter", action="store_true",
                   help="Disable legacy confidence_threshold → pre_vector cutoff")
    p.add_argument("--allow-overwrite-canonical", action="store_true",
                   help="Permit writing into a non-direct existing run dir")
    p.add_argument("--confidence-source", choices=["score", "mask_mean_confidence", "existing"],
                   default="score",
                   help="Phase 1 default = score; existing preserves a confidence column "
                        "already produced by a parity/vectorization path")
    p.add_argument("--mask-threshold-area-m2-tiers", default=None,
                   help="Adaptive mask threshold tiers as JSON/list, e.g. "
                        "'[[200,0.55],[100,0.45]]'. Overrides config key of same name.")
    p.add_argument("--mask-threshold-area-px-tiers", default=None,
                   help="Adaptive mask threshold tiers in pixels, e.g. "
                        "'[[5000,0.55],[2000,0.45]]'. Mostly for tests/debug.")
    p.add_argument("--mask-hysteresis-high-threshold", type=float, default=None,
                   help="Keep only low-threshold mask components connected to "
                        "pixels above this high threshold.")
    p.add_argument("--mask-hysteresis-min-core-area-px", type=int, default=None,
                   help="Minimum high-threshold core pixels required when hysteresis is enabled.")
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
    mask_threshold_area_tiers: tuple[tuple[float, float], ...] | None,
    mask_threshold_area_units: str,
    mask_hysteresis_high_threshold: float | None,
    mask_hysteresis_min_core_area_px: int,
) -> gpd.GeoDataFrame:
    """Build a per-chip GeoDataFrame in source_crs."""
    rows = []
    for det in chip.detections:
        # Build the chip's affine transform from the stored 6-tuple.
        win_tr = Affine(*chip.window_transform)
        threshold_area_scale = _threshold_area_scale(
            win_tr,
            mask_threshold_area_units,
        )
        result = vectorize_chip_mask(
            det.mask_crop_uint8,
            tuple(det.mask_crop_offset),
            threshold=mask_threshold,
            window_transform=win_tr,
            source_crs=chip.source_crs,
            multi_component=multi_component,
            simplify_tolerance_pixels=simplify_tolerance_pixels,
            threshold_area_tiers=mask_threshold_area_tiers,
            threshold_area_scale=threshold_area_scale,
            hysteresis_high_threshold=mask_hysteresis_high_threshold,
            hysteresis_min_core_area_px=mask_hysteresis_min_core_area_px,
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
                "mask_threshold_effective": result.effective_threshold,
                "mask_hysteresis_high_threshold": result.high_threshold,
                "mask_hysteresis_core_pixels": result.core_pixel_count,
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
                "n_components_dropped", "mask_threshold_effective",
                "mask_hysteresis_high_threshold", "mask_hysteresis_core_pixels",
                "_mask_crop", "_mask_offset",
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
    mask_threshold_area_tiers: tuple[tuple[float, float], ...] | None,
    mask_threshold_area_units: str,
    mask_hysteresis_high_threshold: float | None,
    mask_hysteresis_min_core_area_px: int,
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
        threshold_area_scale = _threshold_area_scale(
            src_tr,
            mask_threshold_area_units,
        )

        painted = paint_and_vectorize_pixel_or(
            detections,
            raster_height=int(H),
            raster_width=int(W),
            source_transform=src_tr,
            source_crs=meta.crs,
            mask_threshold=mask_threshold,
            multi_component="explode",  # always explode in pixel-or; ORed mask is one connected blob per panel cluster
            simplify_tolerance_pixels=simplify_tolerance_pixels,
            threshold_area_tiers=mask_threshold_area_tiers,
            threshold_area_scale=threshold_area_scale,
            hysteresis_high_threshold=mask_hysteresis_high_threshold,
            hysteresis_min_core_area_px=mask_hysteresis_min_core_area_px,
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
                "mask_threshold_effective": p.effective_threshold,
                "mask_hysteresis_high_threshold": p.high_threshold,
                "mask_hysteresis_core_pixels": p.core_pixel_count,
                "_mask_crop": None,
                "_mask_offset": (0, 0),
            })
        if rows:
            chip_gdfs.append(gpd.GeoDataFrame(rows, geometry="geometry", crs=meta.crs))
    return chip_gdfs


def _geoai_parity_vectorize(
    artifact,
    *,
    output_dir: Path,
    postproc_cfg: dict[str, Any],
    mask_threshold: float,
) -> list[gpd.GeoDataFrame]:
    """Rebuild geoai-style two-band masks, then call geoai.orthogonalize.

    This mirrors the legacy path in detect_and_evaluate.py:
      SolarPanelDetector.generate_masks(...) -> mask GeoTIFF with
      band 1=binary mask and band 2=score confidence;
      geoai.orthogonalize(..., epsilon=0.2);
      zonal mean of band 2 becomes `confidence`.
    """
    try:
        import geoai
    except ImportError as exc:
        raise RuntimeError(
            "finalize --parity-mode geoai requires geoai-py in the environment"
        ) from exc

    import rasterio

    masks_dir = output_dir / "masks"
    vectors_dir = output_dir / "vectors"
    masks_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    src_meta_by_path = {s.path: s for s in artifact.source_tiles}
    grouped: dict[str, list[dict]] = {}
    for chip in artifact.chips:
        if not chip.detections:
            continue
        col_off, row_off, _w, _h = chip.window
        for det in chip.detections:
            grouped.setdefault(chip.source_tif, []).append({
                "mask_chip_uint8": det.mask_chip_uint8,
                "mask_crop_uint8": det.mask_crop_uint8,
                "chip_source_offset": (int(col_off), int(row_off)),
                "source_offset": (
                    int(col_off) + int(det.mask_crop_offset[0]),
                    int(row_off) + int(det.mask_crop_offset[1]),
                ),
                "score": float(det.score),
                "label": int(det.label),
            })

    min_object_area = float(postproc_cfg.get("min_object_area", 5.0))
    max_object_area = postproc_cfg.get("max_object_area")
    if max_object_area is not None:
        max_object_area = float(max_object_area)

    out_gdfs: list[gpd.GeoDataFrame] = []
    for source_tif, detections in grouped.items():
        meta = src_meta_by_path.get(source_tif)
        if meta is None:
            continue
        raster_height, raster_width = int(meta.shape[0]), int(meta.shape[1])
        mask_array, conf_array, n_painted = paint_geoai_parity_mask(
            detections,
            raster_height=raster_height,
            raster_width=raster_width,
            mask_threshold=mask_threshold,
            min_object_area=min_object_area,
            max_object_area=max_object_area,
        )
        if n_painted == 0:
            continue

        source_stem = Path(source_tif).stem
        mask_path = masks_dir / f"{source_stem}_mask.tif"
        vector_path = vectors_dir / f"{source_stem}_vectors.geojson"
        with rasterio.open(source_tif) as src:
            profile = src.profile.copy()
        for stale_key in (
            "photometric",
            "photometric_interpretation",
            "jpeg_quality",
            "jpegtables",
            "ycbcr_subsampling",
        ):
            profile.pop(stale_key, None)
        profile.update(dtype=rasterio.uint8, count=2, compress="lzw", nodata=0)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask_array, 1)
            dst.write(conf_array, 2)

        try:
            gdf_tile = geoai.orthogonalize(
                input_path=str(mask_path),
                output_path=str(vector_path),
                epsilon=0.2,
            )
        except Exception as exc:
            if "No valid polygons" in str(exc) or "No geometries" in str(exc):
                continue
            raise

        if gdf_tile is None or len(gdf_tile) == 0:
            continue
        gdf_tile = gdf_tile.copy()
        gdf_tile["confidence"] = _zonal_band_mean(gdf_tile, mask_path, band=2) / 255.0
        gdf_tile["mask_mean_confidence_painted"] = gdf_tile["confidence"].astype(float)
        # Legacy detect_and_evaluate.py computes geoai geometric properties
        # while each tile is still in its source CRS, then reprojects the
        # GeoDataFrame later. Preserve that attribute-ordering in parity mode.
        try:
            gdf_tile = geoai.add_geometric_properties(gdf_tile)
        except Exception:
            pass
        gdf_tile["source_tile"] = source_stem
        gdf_tile["source_tif"] = source_tif
        gdf_tile["chip_index"] = -1
        gdf_tile["label"] = 1
        out_gdfs.append(gdf_tile)

    return out_gdfs


def _zonal_band_mean(gdf: gpd.GeoDataFrame, raster_path: Path, *, band: int) -> np.ndarray:
    """Mean raster band value inside each geometry, matching nodata=0 behavior."""
    try:
        import rasterstats as rs
        stats = rs.zonal_stats(gdf, str(raster_path), band=band, stats=["mean"], nodata=0)
        return np.array([
            float(s["mean"]) if s.get("mean") is not None else 0.0
            for s in stats
        ], dtype=np.float64)
    except Exception:
        from rasterio import open as rio_open
        from rasterio.features import geometry_mask

        means = np.zeros(len(gdf), dtype=np.float64)
        with rio_open(raster_path) as src:
            arr = src.read(band)
            transform = src.transform
            out_shape = arr.shape
        for i, geom in enumerate(gdf.geometry.values):
            if geom is None or geom.is_empty:
                continue
            try:
                m = geometry_mask([geom], out_shape=out_shape, transform=transform,
                                  invert=True, all_touched=False)
            except Exception:
                continue
            vals = arr[m]
            vals = vals[vals != 0]
            if vals.size:
                means[i] = float(vals.mean())
        return means


def _threshold_area_scale(transform: Affine, area_units: str) -> float | None:
    if area_units == "m2":
        return affine_pixel_area(transform)
    if area_units == "px":
        return 1.0
    return None


def _json_arg(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON argument: {value!r}") from exc


def _resolve_mask_shape_options(args: argparse.Namespace, postproc_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve CLI/config controls for mask-to-polygon shaping."""
    cli_m2 = _json_arg(args.mask_threshold_area_m2_tiers)
    cli_px = _json_arg(args.mask_threshold_area_px_tiers)
    cfg_m2 = postproc_cfg.get("mask_threshold_area_m2_tiers")
    cfg_px = postproc_cfg.get("mask_threshold_area_px_tiers")

    if cli_m2 is not None:
        tiers = parse_threshold_area_tiers(cli_m2)
        units = "m2"
    elif cli_px is not None:
        tiers = parse_threshold_area_tiers(cli_px)
        units = "px"
    elif cfg_m2 is not None:
        tiers = parse_threshold_area_tiers(cfg_m2)
        units = "m2"
    elif cfg_px is not None:
        tiers = parse_threshold_area_tiers(cfg_px)
        units = "px"
    else:
        tiers = None
        units = "none"

    high = args.mask_hysteresis_high_threshold
    if high is None:
        high = postproc_cfg.get("mask_hysteresis_high_threshold")
    high = None if high is None else float(high)

    min_core = args.mask_hysteresis_min_core_area_px
    if min_core is None:
        min_core = postproc_cfg.get("mask_hysteresis_min_core_area_px", 1)
    min_core = max(1, int(min_core))

    return {
        "mask_threshold_area_tiers": tiers,
        "mask_threshold_area_units": units,
        "mask_hysteresis_high_threshold": high,
        "mask_hysteresis_min_core_area_px": min_core,
    }


def _resolve_vectorization_options(args: argparse.Namespace, postproc_cfg: dict[str, Any]) -> None:
    json_merge_mode = postproc_cfg.get("merge_mode")
    if args.merge_mode is None:
        # Only JSON gave a value (or neither); use JSON value or default.
        args.merge_mode = str(json_merge_mode) if json_merge_mode is not None else "per-detection"
    elif json_merge_mode is not None and str(json_merge_mode) != args.merge_mode:
        # Both CLI and JSON gave merge_mode, but they disagree — refuse to proceed
        # rather than silently discard the JSON value.
        raise ValueError(
            f"merge_mode conflict: --merge-mode CLI={args.merge_mode!r} "
            f"but postproc JSON merge_mode={json_merge_mode!r}. "
            "Remove merge_mode from the postproc JSON or omit --merge-mode from the CLI "
            "so that only one source specifies it."
        )
    # else: CLI gave a value and JSON is absent or agrees — keep CLI value as-is.
    if args.merge_mode not in {"pixel-or", "per-detection"}:
        raise ValueError(f"unsupported merge_mode: {args.merge_mode!r}")

    if args.vectorize_multi_component is None:
        default_multi = "largest" if args.merge_mode == "per-detection" else "explode"
        args.vectorize_multi_component = str(
            postproc_cfg.get("vectorize_multi_component", default_multi)
        )
    if args.vectorize_multi_component not in {"largest", "union", "explode"}:
        raise ValueError(
            f"unsupported vectorize_multi_component: {args.vectorize_multi_component!r}"
        )


def _compute_per_detection_mask_mean_confidence(
    pred_metric: gpd.GeoDataFrame,
    *,
    fallback_threshold: float,
) -> np.ndarray:
    out = np.full(len(pred_metric), np.nan, dtype=np.float64)
    has_threshold = "mask_threshold_effective" in pred_metric.columns
    for pos, (_idx, row) in enumerate(pred_metric.iterrows()):
        mask = row.get("_mask_crop")
        if mask is None:
            continue
        threshold = float(row["mask_threshold_effective"]) if has_threshold else fallback_threshold
        cutoff = int(round(max(0.0, min(1.0, threshold)) * 255))
        kept = mask[mask >= cutoff]
        if kept.size:
            out[pos] = float(kept.mean()) / 255.0
    return out


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
    _resolve_vectorization_options(args, postproc_cfg)
    mask_shape_opts = _resolve_mask_shape_options(args, postproc_cfg)
    print(
        "[finalize] vectorization: "
        f"merge_mode={args.merge_mode} "
        f"multi_component={args.vectorize_multi_component} "
        f"mask_threshold={mask_threshold}"
    )
    if mask_shape_opts["mask_threshold_area_tiers"] is not None:
        print(
            "[finalize] adaptive mask threshold tiers "
            f"({mask_shape_opts['mask_threshold_area_units']}): "
            f"{mask_shape_opts['mask_threshold_area_tiers']}"
        )
    if mask_shape_opts["mask_hysteresis_high_threshold"] is not None:
        print(
            "[finalize] mask hysteresis: "
            f"high={mask_shape_opts['mask_hysteresis_high_threshold']} "
            f"min_core_px={mask_shape_opts['mask_hysteresis_min_core_area_px']}"
        )

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

    if args.parity_mode == "geoai" and args.confidence_source == "score":
        args.confidence_source = "existing"
        print("[finalize] geoai parity mode: preserving mask-band confidence")

    # ── Vectorize ────────────────────────────────────────────────────
    if args.parity_mode == "geoai":
        chip_gdfs = _geoai_parity_vectorize(
            artifact,
            output_dir=args.output_dir,
            postproc_cfg=postproc_cfg,
            mask_threshold=mask_threshold,
        )
    elif args.merge_mode == "pixel-or":
        chip_gdfs = _vectorize_pixel_or(
            artifact,
            mask_threshold=mask_threshold,
            multi_component=args.vectorize_multi_component,
            simplify_tolerance_pixels=args.simplify_tolerance_pixels,
            **mask_shape_opts,
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
                **mask_shape_opts,
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
    if args.parity_mode == "geoai" and "area_m2" in pred_metric.columns:
        # Already computed pre-reprojection in _geoai_parity_vectorize to match
        # detect_and_evaluate.py legacy behavior.
        pass
    elif args.parity_mode == "geoai":
        try:
            import geoai
            pred_metric = geoai.add_geometric_properties(pred_metric)
        except Exception:
            pred_metric = compute_geometric_properties(pred_metric)
    else:
        pred_metric = compute_geometric_properties(pred_metric)

    # ── Mask mean confidence ─────────────────────────────────────────
    if "mask_mean_confidence" in pred_metric.columns:
        pred_metric["mask_mean_confidence"] = pred_metric["mask_mean_confidence"].astype(float)
    elif "mask_mean_confidence_painted" in pred_metric.columns:
        # pixel-OR mode already computed it from the merged raster
        pred_metric["mask_mean_confidence"] = pred_metric["mask_mean_confidence_painted"].astype(float)
        pred_metric = pred_metric.drop(columns=["mask_mean_confidence_painted"])
    else:
        pred_metric["mask_mean_confidence"] = _compute_per_detection_mask_mean_confidence(
            pred_metric,
            fallback_threshold=mask_threshold,
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
    if args.confidence_source != "existing":
        pred_metric["confidence"] = pred_metric[args.confidence_source].astype(float)
    elif "confidence" not in pred_metric.columns:
        raise ValueError("confidence_source=existing but no confidence column is present")
    else:
        pred_metric["confidence"] = pred_metric["confidence"].astype(float)
    print(f"[finalize] confidence_source = {args.confidence_source}")

    if args.parity_mode == "geoai":
        # Legacy detect_and_evaluate.py does grid-level spatial NMS before
        # area / elongation / confidence filters.
        pred_nms_first = spatial_nms(pred_metric, iou_threshold=float(args.nms_iou))
        n_after_nms_first = len(pred_nms_first)
        print(f"[finalize] geoai-parity spatial_nms before postproc "
              f"(IoU={args.nms_iou}): {len(pred_metric)} → {n_after_nms_first}")
        pred_nms, stats = apply_postproc_filters(pred_nms_first, postproc_cfg)
        n_after_postproc = len(pred_nms)
        n_after_nms = len(pred_nms)
        print(f"[finalize] postproc filters: {stats}")
    else:
        # ── Post-proc filters ────────────────────────────────────────────
        pred_filtered, stats = apply_postproc_filters(pred_metric, postproc_cfg)
        n_after_postproc = len(pred_filtered)
        print(f"[finalize] postproc filters: {stats}")

        # ── Grid-level spatial NMS ───────────────────────────────────────
        pred_nms = spatial_nms(pred_filtered, iou_threshold=float(args.nms_iou))
        n_after_nms = len(pred_nms)
        print(f"[finalize] grid-level spatial_nms (IoU={args.nms_iou}): "
              f"{n_after_postproc} → {n_after_nms}")

    # ── Cross-TIF hairline dissolve (cat-1 fix) ──────────────────────
    dissolve_tol = float(args.dissolve_hairline_tolerance_m)
    if dissolve_tol > 0:
        n_before_dissolve = len(pred_nms)
        pred_nms = dissolve_hairline_gaps(pred_nms, tolerance_m=dissolve_tol)
        n_after_nms = len(pred_nms)
        print(f"[finalize] dissolve_hairline_gaps (tol={dissolve_tol} m): "
              f"{n_before_dissolve} → {n_after_nms}")

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
        "parity_mode": args.parity_mode,
        "confidence_source": args.confidence_source,
        "merge_mode": args.merge_mode,
        "nms_iou": float(args.nms_iou),
        "vectorize_multi_component": args.vectorize_multi_component,
        "simplify_tolerance_pixels": args.simplify_tolerance_pixels,
        "mask_threshold_area_m2_tiers": _stringify_for_json({
            "tiers": parse_threshold_area_tiers(
                _json_arg(args.mask_threshold_area_m2_tiers)
                if args.mask_threshold_area_m2_tiers is not None
                else postproc_cfg.get("mask_threshold_area_m2_tiers")
            )
        })["tiers"],
        "mask_threshold_area_px_tiers": _stringify_for_json({
            "tiers": parse_threshold_area_tiers(
                _json_arg(args.mask_threshold_area_px_tiers)
                if args.mask_threshold_area_px_tiers is not None
                else postproc_cfg.get("mask_threshold_area_px_tiers")
            )
        })["tiers"],
        "mask_hysteresis_high_threshold": (
            args.mask_hysteresis_high_threshold
            if args.mask_hysteresis_high_threshold is not None
            else postproc_cfg.get("mask_hysteresis_high_threshold")
        ),
        "mask_hysteresis_min_core_area_px": (
            args.mask_hysteresis_min_core_area_px
            if args.mask_hysteresis_min_core_area_px is not None
            else postproc_cfg.get("mask_hysteresis_min_core_area_px", 1)
        ),
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
        f"- parity_mode: {args.parity_mode}\n"
        f"- confidence_source: {args.confidence_source}\n\n"
        f"- merge_mode: {args.merge_mode}\n"
        f"- vectorize_multi_component: {args.vectorize_multi_component}\n\n"
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
        if v is None:
            out[k] = None
        elif isinstance(v, tuple):
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
