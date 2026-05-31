#!/usr/bin/env python3
"""Cluster prediction detections into multi-target Gemini review chips.

This is the current-inventory analogue of the solar_backdating chip-group
bridge: nearby detections are packed into one fixed-size image chip, each
detection is marked as T01/T02/..., and downstream Gemini review returns one
target-level row per detection.  The generated ``chip_targets.csv`` is also a
candidate manifest for ``build_gemini_review_training_pool.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import Point, box
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core import region_registry  # noqa: E402


DEFAULT_METRIC_CRS = "EPSG:32735"
DEFAULT_CAPTURE_DATE = "2024-06-30"
DEFAULT_PREDICTIONS_FILENAME = "predictions_metric_merge01_c0925.gpkg"

GROUP_FIELDS = [
    "chip_id",
    "region_key",
    "grid_id",
    "source_tile",
    "tile_path",
    "image_path",
    "raw_image_path",
    "n_targets",
    "target_ids",
    "pred_ids",
    "predictions_paths",
    "chip_size_m",
    "output_px",
    "chip_minx",
    "chip_miny",
    "chip_maxx",
    "chip_maxy",
    "capture_date",
    "group_width_m",
    "group_height_m",
    "max_target_offset_m",
]

TARGET_FIELDS = [
    "candidate_id",
    "target_id",
    "chip_id",
    "target_index",
    "target_label",
    "region_key",
    "region",
    "grid_id",
    "pred_id",
    "predictions_path",
    "results_root",
    "model_run",
    "imagery_layer",
    "source_tile",
    "tile_path",
    "image_path",
    "raw_image_path",
    "capture_date",
    "score",
    "confidence",
    "sam_score",
    "n_merged",
    "area_m2",
    "target_offset_x_m",
    "target_offset_y_m",
    "search_radius_m",
    "chip_size_m",
]


@dataclass(frozen=True)
class Target:
    candidate_id: str
    region_key: str
    grid_id: str
    pred_id: int
    predictions_path: Path
    results_root: str
    model_run: str
    imagery_layer: str
    source_tile: str
    tile_path: Path
    geom_metric: Any
    centroid: Point
    pack_bounds: tuple[float, float, float, float]
    score: Any
    confidence: Any
    sam_score: Any
    n_merged: Any
    area_m2: float


@dataclass(frozen=True)
class ChipGroup:
    chip_id: str
    member_indices: tuple[int, ...]
    tile_path: Path
    source_tile: str
    center_x: float
    center_y: float
    chip_bounds: tuple[float, float, float, float]
    pack_bounds: tuple[float, float, float, float]


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _norm_region(value: Any) -> str:
    text = _norm_text(value)
    if text.lower() == "jnb":
        text = "johannesburg"
    return region_registry.normalize_region_key(text) or text


def _format_any(value: Any) -> str:
    text = _norm_text(value)
    return text


def _format_float(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    margin: float,
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    return minx - margin, miny - margin, maxx + margin, maxy + margin


def _union_bounds(
    bounds: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    vals = list(bounds)
    return (
        min(v[0] for v in vals),
        min(v[1] for v in vals),
        max(v[2] for v in vals),
        max(v[3] for v in vals),
    )


def _bounds_width(bounds: tuple[float, float, float, float]) -> float:
    return float(bounds[2] - bounds[0])


def _bounds_height(bounds: tuple[float, float, float, float]) -> float:
    return float(bounds[3] - bounds[1])


def _resolve_predictions_path(
    row: Mapping[str, Any],
    *,
    results_root: Path | None,
    predictions_filename: str,
) -> Path:
    explicit = _norm_text(row.get("predictions_path"))
    if explicit:
        return Path(explicit).expanduser()
    grid_id = _norm_text(row.get("grid_id")).upper()
    root_text = _norm_text(row.get("results_root"))
    root = Path(root_text).expanduser() if root_text else results_root
    if not root:
        raise ValueError(f"candidate {grid_id}: missing predictions_path/results_root")
    return root / grid_id / predictions_filename


def _resolve_tile_path(
    *,
    row: Mapping[str, Any],
    pred_row: Mapping[str, Any],
    grid_id: str,
    tiles_root: Path | None,
) -> tuple[str, Path]:
    explicit = _norm_text(row.get("tile_path"))
    if explicit:
        path = Path(explicit).expanduser()
        return path.stem, path

    source_tile = _norm_text(row.get("source_tile") or pred_row.get("source_tile"))
    source_tif = _norm_text(row.get("source_tif") or pred_row.get("source_tif"))
    if source_tif and Path(source_tif).exists():
        path = Path(source_tif)
        return source_tile or path.stem, path

    if tiles_root and source_tile:
        path = tiles_root / grid_id / f"{source_tile}.tif"
        return source_tile, path

    raise ValueError(
        f"{grid_id}: cannot resolve tile path; provide tile_path/source_tile + --tiles-root"
    )


def _load_prediction_cache(
    path: Path,
    *,
    metric_crs: str,
    cache: dict[Path, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    if path not in cache:
        if not path.exists():
            raise FileNotFoundError(path)
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            # Legacy reviewed/prediction exports are lon/lat; make the
            # assumption explicit rather than failing (mirrors
            # build_gemini_review_training_pool.load_prediction_geometry).
            print(f"[WARN] prediction CRS missing, assuming EPSG:4326: {path}", file=sys.stderr)
            gdf = gdf.set_crs(epsg=4326)
        cache[path] = gdf.to_crs(metric_crs)
    return cache[path]


def load_targets(
    candidate_manifest: Path,
    *,
    metric_crs: str,
    pack_margin_m: float,
    results_root: Path | None,
    predictions_filename: str,
    tiles_root: Path | None,
    limit: int | None,
) -> list[Target]:
    candidates = pd.read_csv(candidate_manifest)
    required = {"grid_id", "pred_id"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate manifest missing columns: {sorted(missing)}")
    if limit is not None:
        candidates = candidates.head(limit).copy()

    pred_cache: dict[Path, gpd.GeoDataFrame] = {}
    targets: list[Target] = []
    for idx, row in candidates.iterrows():
        grid_id = _norm_text(row.get("grid_id")).upper()
        pred_id = int(row["pred_id"])
        region = _norm_region(row.get("region_key") or row.get("region") or row.get("city"))
        if not region:
            region = "johannesburg" if grid_id.startswith("JNB") else ""
        predictions_path = _resolve_predictions_path(
            row,
            results_root=results_root,
            predictions_filename=predictions_filename,
        )
        gdf = _load_prediction_cache(predictions_path, metric_crs=metric_crs, cache=pred_cache)
        if pred_id < 0 or pred_id >= len(gdf):
            raise IndexError(f"{predictions_path}: pred_id {pred_id} out of range")
        pred_row = gdf.iloc[pred_id]
        geom = pred_row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            try:
                geom = geom.buffer(0)
            except Exception as exc:
                print(
                    f"[WARN] skipping {grid_id} pred_id {pred_id}: invalid geometry ({exc})",
                    file=sys.stderr,
                )
                continue
        if geom is None or geom.is_empty:
            continue
        source_tile, tile_path = _resolve_tile_path(
            row=row,
            pred_row=pred_row,
            grid_id=grid_id,
            tiles_root=tiles_root,
        )
        candidate_id = _norm_text(row.get("candidate_id") or row.get("anchor_id"))
        if not candidate_id:
            candidate_id = f"{grid_id}_pred{pred_id:06d}"
        targets.append(
            Target(
                candidate_id=candidate_id,
                region_key=region,
                grid_id=grid_id,
                pred_id=pred_id,
                predictions_path=predictions_path,
                results_root=_norm_text(row.get("results_root")),
                model_run=_norm_text(row.get("model_run")),
                imagery_layer=_norm_text(row.get("imagery_layer")),
                source_tile=source_tile,
                tile_path=tile_path,
                geom_metric=geom,
                centroid=geom.centroid,
                pack_bounds=_expand_bounds(tuple(float(v) for v in geom.bounds), pack_margin_m),
                score=pred_row.get("score", row.get("score", "")),
                confidence=pred_row.get("confidence", row.get("confidence", "")),
                sam_score=pred_row.get("sam_score", row.get("sam_score", "")),
                n_merged=pred_row.get("n_merged", row.get("n_merged", "")),
                area_m2=float(pred_row.get("area_m2", geom.area) or geom.area),
            )
        )
    if not targets:
        raise SystemExit("No valid targets loaded.")
    return targets


def _cluster_bucket(
    targets: Sequence[Target],
    *,
    bucket_indices: Sequence[int],
    chip_size_m: float,
    max_targets_per_chip: int,
    chip_prefix: str,
) -> list[ChipGroup]:
    half = chip_size_m / 2.0
    points = [targets[i].centroid for i in bucket_indices]
    tree = STRtree(points)
    assigned = np.zeros(len(bucket_indices), dtype=bool)
    seed_order_local = sorted(
        range(len(bucket_indices)),
        key=lambda local_i: (
            math.floor(float(points[local_i].y) / chip_size_m),
            math.floor(float(points[local_i].x) / chip_size_m),
            targets[bucket_indices[local_i]].grid_id,
            targets[bucket_indices[local_i]].pred_id,
        ),
    )
    groups: list[ChipGroup] = []
    for local_seed in seed_order_local:
        if assigned[local_seed]:
            continue
        seed_idx = bucket_indices[local_seed]
        seed = targets[seed_idx]
        members_local = [local_seed]
        group_bounds = seed.pack_bounds
        query = box(
            float(seed.centroid.x) - half,
            float(seed.centroid.y) - half,
            float(seed.centroid.x) + half,
            float(seed.centroid.y) + half,
        )
        candidate_locals = [int(i) for i in tree.query(query)]
        candidate_locals = [
            i
            for i in candidate_locals
            if i != local_seed and not assigned[i]
        ]
        candidate_locals.sort(
            key=lambda i: (
                (float(points[i].x) - float(seed.centroid.x)) ** 2
                + (float(points[i].y) - float(seed.centroid.y)) ** 2,
                targets[bucket_indices[i]].grid_id,
                targets[bucket_indices[i]].pred_id,
            )
        )
        for local_i in candidate_locals:
            if len(members_local) >= max_targets_per_chip:
                break
            target = targets[bucket_indices[local_i]]
            new_bounds = _union_bounds([group_bounds, target.pack_bounds])
            if (
                _bounds_width(new_bounds) <= chip_size_m
                and _bounds_height(new_bounds) <= chip_size_m
            ):
                members_local.append(local_i)
                group_bounds = new_bounds
        for local_i in members_local:
            assigned[local_i] = True

        center_x = (group_bounds[0] + group_bounds[2]) / 2.0
        center_y = (group_bounds[1] + group_bounds[3]) / 2.0
        members = tuple(
            sorted((bucket_indices[i] for i in members_local), key=lambda i: targets[i].pred_id)
        )
        representative = targets[members[0]]
        groups.append(
            ChipGroup(
                chip_id=f"{chip_prefix}_c{len(groups) + 1:07d}",
                member_indices=members,
                tile_path=representative.tile_path,
                source_tile=representative.source_tile,
                center_x=center_x,
                center_y=center_y,
                chip_bounds=(center_x - half, center_y - half, center_x + half, center_y + half),
                pack_bounds=group_bounds,
            )
        )
    return groups


def build_chip_groups(
    targets: Sequence[Target],
    *,
    chip_size_m: float,
    max_targets_per_chip: int,
    chip_prefix: str,
) -> list[ChipGroup]:
    buckets: dict[tuple[str, str, str], list[int]] = {}
    for idx, target in enumerate(targets):
        key = (target.grid_id, target.source_tile, str(target.tile_path))
        buckets.setdefault(key, []).append(idx)
    groups: list[ChipGroup] = []
    for bucket_no, key in enumerate(sorted(buckets), start=1):
        bucket_prefix = f"{chip_prefix}_b{bucket_no:05d}"
        groups.extend(
            _cluster_bucket(
                targets,
                bucket_indices=buckets[key],
                chip_size_m=chip_size_m,
                max_targets_per_chip=max_targets_per_chip,
                chip_prefix=bucket_prefix,
            )
        )
    return groups


def _metric_bounds_to_raster_bounds(
    bounds: tuple[float, float, float, float],
    *,
    metric_crs: str,
    raster_crs: Any,
) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs(metric_crs, raster_crs, always_xy=True)
    geom = shapely_transform(transformer.transform, box(*bounds))
    return tuple(float(v) for v in geom.bounds)


def _geometry_to_raster(geom: Any, *, metric_crs: str, raster_crs: Any) -> Any:
    transformer = Transformer.from_crs(metric_crs, raster_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def _read_rgb_chip(
    src: rasterio.io.DatasetReader,
    *,
    raster_bounds: tuple[float, float, float, float],
    output_px: int,
) -> np.ndarray:
    window = from_bounds(*raster_bounds, transform=src.transform)
    indexes = [1, 2, 3] if src.count >= 3 else [1]
    arr = src.read(
        indexes,
        window=window,
        out_shape=(len(indexes), output_px, output_px),
        boundless=True,
        fill_value=0,
        resampling=Resampling.bilinear,
    )
    if len(indexes) == 1:
        arr = np.repeat(arr, 3, axis=0)
    return np.moveaxis(arr[:3], 0, -1).astype(np.uint8)


def _coord_to_pixel(
    x: float,
    y: float,
    *,
    raster_bounds: tuple[float, float, float, float],
    output_px: int,
) -> tuple[float, float]:
    minx, miny, maxx, maxy = raster_bounds
    px = (float(x) - minx) / (maxx - minx) * output_px
    py = (maxy - float(y)) / (maxy - miny) * output_px
    return px, py


def _draw_label(
    draw: ImageDraw.ImageDraw,
    *,
    px: float,
    py: float,
    label: str,
    color: tuple[int, int, int],
    ring_r: int,
    output_px: int,
) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    label_w = bbox[2] - bbox[0] + 8
    label_h = bbox[3] - bbox[1] + 6
    label_x = int(px + ring_r + 4)
    label_y = int(py - ring_r - label_h - 3)
    if label_x + label_w >= output_px:
        label_x = int(px - ring_r - label_w - 4)
    if label_y < 0:
        label_y = int(py + ring_r + 3)
    label_x = max(0, min(label_x, max(0, output_px - label_w - 1)))
    label_y = max(0, min(label_y, max(0, output_px - label_h - 1)))
    draw.rectangle(
        [label_x, label_y, label_x + label_w, label_y + label_h],
        fill=(0, 0, 0),
        outline=color,
    )
    draw.text((label_x + 4, label_y + 3), label, fill=color, font=font)


def _draw_target(
    draw: ImageDraw.ImageDraw,
    *,
    geom_raster: Any,
    centroid_raster: Point,
    raster_bounds: tuple[float, float, float, float],
    output_px: int,
    chip_size_m: float,
    search_radius_m: float,
    label: str,
    color: tuple[int, int, int],
) -> None:
    geoms = [geom_raster] if geom_raster.geom_type == "Polygon" else list(geom_raster.geoms)
    for poly in geoms:
        coords = [
            _coord_to_pixel(x, y, raster_bounds=raster_bounds, output_px=output_px)
            for x, y in poly.exterior.coords
        ]
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=3, joint="curve")

    px, py = _coord_to_pixel(
        centroid_raster.x,
        centroid_raster.y,
        raster_bounds=raster_bounds,
        output_px=output_px,
    )
    ring_r = max(8, int(round(output_px * search_radius_m / chip_size_m)))
    cross_r = max(6, int(round(output_px * 0.025)))
    for offset in range(3, 0, -1):
        r = ring_r + offset
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 0, 0), width=1)
    draw.ellipse([px - ring_r, py - ring_r, px + ring_r, py + ring_r], outline=color, width=2)
    draw.line([px - cross_r, py, px + cross_r, py], fill=(0, 0, 0), width=5)
    draw.line([px, py - cross_r, px, py + cross_r], fill=(0, 0, 0), width=5)
    draw.line([px - cross_r, py, px + cross_r, py], fill=color, width=3)
    draw.line([px, py - cross_r, px, py + cross_r], fill=color, width=3)
    _draw_label(
        draw,
        px=px,
        py=py,
        label=label,
        color=color,
        ring_r=ring_r,
        output_px=output_px,
    )


def render_groups(
    groups: Sequence[ChipGroup],
    targets: Sequence[Target],
    *,
    output_dir: Path,
    metric_crs: str,
    chip_size_m: float,
    search_radius_m: float,
    output_px: int,
    capture_date: str,
    allow_missing_tiles: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chip_dir = output_dir / "review_chips"
    chip_dir.mkdir(parents=True, exist_ok=True)
    palette = (
        (255, 215, 0),
        (0, 180, 255),
        (255, 92, 92),
        (76, 217, 100),
        (255, 128, 0),
        (210, 120, 255),
    )
    group_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for group in groups:
        members = [targets[i] for i in group.member_indices]
        if not group.tile_path.exists():
            if allow_missing_tiles:
                continue
            raise FileNotFoundError(group.tile_path)

        with rasterio.open(group.tile_path) as src:
            raster_bounds = _metric_bounds_to_raster_bounds(
                group.chip_bounds,
                metric_crs=metric_crs,
                raster_crs=src.crs,
            )
            rgb = _read_rgb_chip(src, raster_bounds=raster_bounds, output_px=output_px)
            raw = Image.fromarray(rgb, mode="RGB")
            review = raw.copy()
            draw = ImageDraw.Draw(review)
            target_ids: list[str] = []
            pred_ids: list[str] = []
            prediction_paths: list[str] = []
            for target_index, target in enumerate(members, start=1):
                label = f"T{target_index:02d}"
                color = palette[(target_index - 1) % len(palette)]
                geom_raster = _geometry_to_raster(
                    target.geom_metric,
                    metric_crs=metric_crs,
                    raster_crs=src.crs,
                )
                centroid_raster = _geometry_to_raster(
                    target.centroid,
                    metric_crs=metric_crs,
                    raster_crs=src.crs,
                )
                _draw_target(
                    draw,
                    geom_raster=geom_raster,
                    centroid_raster=centroid_raster,
                    raster_bounds=raster_bounds,
                    output_px=output_px,
                    chip_size_m=chip_size_m,
                    search_radius_m=search_radius_m,
                    label=label,
                    color=color,
                )
                target_ids.append(target.candidate_id)
                pred_ids.append(str(target.pred_id))
                prediction_paths.append(str(target.predictions_path))
                target_rows.append(
                    {
                        "candidate_id": target.candidate_id,
                        "target_id": target.candidate_id,
                        "chip_id": group.chip_id,
                        "target_index": target_index,
                        "target_label": label,
                        "region_key": target.region_key,
                        "region": target.region_key,
                        "grid_id": target.grid_id,
                        "pred_id": target.pred_id,
                        "predictions_path": str(target.predictions_path),
                        "results_root": target.results_root,
                        "model_run": target.model_run,
                        "imagery_layer": target.imagery_layer,
                        "source_tile": target.source_tile,
                        "tile_path": str(target.tile_path),
                        "image_path": str(chip_dir / f"{group.chip_id}_review.png"),
                        "raw_image_path": str(chip_dir / f"{group.chip_id}_raw.png"),
                        "capture_date": capture_date,
                        "score": _format_any(target.score),
                        "confidence": _format_any(target.confidence),
                        "sam_score": _format_any(target.sam_score),
                        "n_merged": _format_any(target.n_merged),
                        "area_m2": _format_float(target.area_m2),
                        "target_offset_x_m": _format_float(target.centroid.x - group.center_x),
                        "target_offset_y_m": _format_float(target.centroid.y - group.center_y),
                        "search_radius_m": _format_float(search_radius_m, 2),
                        "chip_size_m": _format_float(chip_size_m, 2),
                    }
                )

            raw_path = chip_dir / f"{group.chip_id}_raw.png"
            review_path = chip_dir / f"{group.chip_id}_review.png"
            raw.save(raw_path)
            review.save(review_path)

        group_width = _bounds_width(group.pack_bounds)
        group_height = _bounds_height(group.pack_bounds)
        max_offset = max(
            math.hypot(float(t.centroid.x) - group.center_x, float(t.centroid.y) - group.center_y)
            for t in members
        )
        group_rows.append(
            {
                "chip_id": group.chip_id,
                "region_key": members[0].region_key,
                "grid_id": members[0].grid_id,
                "source_tile": group.source_tile,
                "tile_path": str(group.tile_path),
                "image_path": str(chip_dir / f"{group.chip_id}_review.png"),
                "raw_image_path": str(chip_dir / f"{group.chip_id}_raw.png"),
                "n_targets": len(members),
                "target_ids": ";".join(target_ids),
                "pred_ids": ";".join(pred_ids),
                "predictions_paths": ";".join(sorted(set(prediction_paths))),
                "chip_size_m": _format_float(chip_size_m, 2),
                "output_px": output_px,
                "chip_minx": _format_float(group.chip_bounds[0]),
                "chip_miny": _format_float(group.chip_bounds[1]),
                "chip_maxx": _format_float(group.chip_bounds[2]),
                "chip_maxy": _format_float(group.chip_bounds[3]),
                "capture_date": capture_date,
                "group_width_m": _format_float(group_width),
                "group_height_m": _format_float(group_height),
                "max_target_offset_m": _format_float(max_offset),
            }
        )

    return group_rows, target_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(
        args.candidate_manifest,
        metric_crs=args.metric_crs,
        pack_margin_m=args.pack_margin_m,
        results_root=args.results_root,
        predictions_filename=args.predictions_filename,
        tiles_root=args.tiles_root,
        limit=args.limit_candidates,
    )
    groups = build_chip_groups(
        targets,
        chip_size_m=args.chip_size_m,
        max_targets_per_chip=args.max_targets_per_chip,
        chip_prefix=args.chip_prefix,
    )
    group_rows, target_rows = render_groups(
        groups,
        targets,
        output_dir=output_dir,
        metric_crs=args.metric_crs,
        chip_size_m=args.chip_size_m,
        search_radius_m=args.search_radius_m,
        output_px=args.output_px,
        capture_date=args.capture_date,
        allow_missing_tiles=args.allow_missing_tiles,
    )
    write_csv(output_dir / "chip_groups.csv", group_rows, GROUP_FIELDS)
    write_csv(output_dir / "chip_targets.csv", target_rows, TARGET_FIELDS)
    counts = pd.Series([int(r["n_targets"]) for r in group_rows], dtype="int64")
    summary = {
        "candidate_manifest": str(args.candidate_manifest),
        "n_targets": len(target_rows),
        "n_chip_groups": len(group_rows),
        "mean_targets_per_chip": float(counts.mean()) if len(counts) else 0.0,
        "median_targets_per_chip": float(counts.median()) if len(counts) else 0.0,
        "max_targets_per_chip_observed": int(counts.max()) if len(counts) else 0,
        "gemini_call_reduction_factor": (
            round(len(target_rows) / len(group_rows), 3) if group_rows else 0.0
        ),
        "chip_size_m": args.chip_size_m,
        "output_px": args.output_px,
        "outputs": {
            "chip_groups_csv": str(output_dir / "chip_groups.csv"),
            "chip_targets_csv": str(output_dir / "chip_targets.csv"),
            "review_chips_dir": str(output_dir / "review_chips"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--predictions-filename", default=DEFAULT_PREDICTIONS_FILENAME)
    parser.add_argument("--tiles-root", type=Path)
    parser.add_argument("--metric-crs", default=DEFAULT_METRIC_CRS)
    parser.add_argument("--chip-prefix", default="gemini_det_review")
    parser.add_argument("--capture-date", default=DEFAULT_CAPTURE_DATE)
    parser.add_argument("--chip-size-m", type=float, default=64.0)
    parser.add_argument("--pack-margin-m", type=float, default=4.0)
    parser.add_argument("--search-radius-m", type=float, default=5.0)
    parser.add_argument("--output-px", type=int, default=768)
    parser.add_argument("--max-targets-per-chip", type=int, default=4)
    parser.add_argument("--limit-candidates", type=int)
    parser.add_argument("--allow-missing-tiles", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
