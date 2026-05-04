#!/usr/bin/env python3
"""Build buffered PV anchor chips from existing GT annotations.

Existing aerial/Vexcel GT masks are used as location anchors, not as exact
historical GEID masks. Each output row contains a centroid plus an expanded
chip bbox that tolerates source/viewpoint/georegistration offset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import region_registry
from core.annotation_loader import AnnotationEntry, discover_annotations, load_annotation_gdf
from scripts.temporal.geid_temporal_common import write_csv_rows

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "geid_temporal" / "anchors.csv"
ANCHOR_FIELDS = [
    "anchor_id",
    "region_key",
    "grid_id",
    "source_annotation_path",
    "source_feature_id",
    "quality_tier",
    "anchor_policy",
    "centroid_lon",
    "centroid_lat",
    "source_area_m2",
    "source_width_m",
    "source_height_m",
    "chip_half_m",
    "search_radius_m",
    "chip_lon_min",
    "chip_lat_min",
    "chip_lon_max",
    "chip_lat_max",
    "alignment_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", default="johannesburg", help="Canonical region key from regions.yaml. Default: johannesburg")
    parser.add_argument("--grid-ids", nargs="*", help="Optional grid IDs to include. Default: all discovered annotations in region.")
    parser.add_argument("--annotation-path", type=Path, help="Optional direct GPKG path for a single grid.")
    parser.add_argument("--annotation-layer", help="Optional layer name for --annotation-path.")
    parser.add_argument("--grid-id", help="Grid ID for --annotation-path mode.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chip-min-half-m", type=float, default=18.0, help="Minimum chip half-side in metres.")
    parser.add_argument("--chip-max-half-m", type=float, default=45.0, help="Maximum chip half-side in metres.")
    parser.add_argument("--mask-margin-m", type=float, default=8.0, help="Extra metres around source polygon bbox.")
    parser.add_argument("--search-radius-m", type=float, default=10.0, help="Expected scoring/search tolerance around anchor.")
    parser.add_argument("--max-anchors", type=int, help="Optional cap for smoke tests.")
    return parser.parse_args()


def _direct_entry(args: argparse.Namespace) -> AnnotationEntry:
    if not args.grid_id:
        raise SystemExit("--grid-id is required with --annotation-path")
    path = args.annotation_path
    if path is None or not path.exists():
        raise SystemExit(f"Annotation path not found: {path}")
    return AnnotationEntry(
        grid_id=str(args.grid_id).strip().upper(),
        region_key=args.region,
        path=path.resolve(),
        schema_type="direct",
        annotation_count=None,
        annotation_layer=args.annotation_layer,
        registered=False,
    )


def _entries_from_args(args: argparse.Namespace) -> list[AnnotationEntry]:
    if args.annotation_path:
        return [_direct_entry(args)]
    wanted = {g.strip().upper() for g in args.grid_ids} if args.grid_ids else None
    discovered = discover_annotations(regions=[args.region])
    entries = []
    for grid_id, entry in sorted(discovered.items()):
        if wanted is not None and grid_id not in wanted:
            continue
        entries.append(entry)
    if wanted is not None:
        found = {e.grid_id for e in entries}
        missing = sorted(wanted - found)
        if missing:
            print(f"[WARN] no annotation found for grids: {', '.join(missing)}", file=sys.stderr)
    return entries


def _bbox_to_wgs84(transformer: Transformer, cx: float, cy: float, half_m: float) -> tuple[float, float, float, float]:
    points = [
        transformer.transform(cx - half_m, cy - half_m),
        transformer.transform(cx - half_m, cy + half_m),
        transformer.transform(cx + half_m, cy - half_m),
        transformer.transform(cx + half_m, cy + half_m),
    ]
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return min(lons), min(lats), max(lons), max(lats)


def build_anchor_rows(
    entries: list[AnnotationEntry],
    *,
    chip_min_half_m: float,
    chip_max_half_m: float,
    mask_margin_m: float,
    search_radius_m: float,
    max_anchors: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in entries:
        try:
            region_cfg = region_registry.get_region_config(entry.region_key)
        except KeyError as exc:
            raise SystemExit(f"Unknown region {entry.region_key!r}: {exc}") from exc
        metric_crs = region_cfg.crs_metric
        grid_meta = region_cfg.grids.get(entry.grid_id, {}) if region_cfg.grids else {}
        quality_tier = grid_meta.get("quality_tier", "") if isinstance(grid_meta, dict) else ""

        gdf = load_annotation_gdf(entry)
        if gdf.empty:
            print(f"[WARN] empty annotation: {entry.path}", file=sys.stderr)
            continue
        gdf_metric = gdf.to_crs(metric_crs)
        to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

        for local_idx, geom in enumerate(gdf_metric.geometry):
            if geom is None or geom.is_empty or not geom.is_valid:
                continue
            minx, miny, maxx, maxy = geom.bounds
            width_m = float(maxx - minx)
            height_m = float(maxy - miny)
            # Use source bbox as a size hint, but do not use the original mask as
            # exact historical supervision. The chip is deliberately expanded.
            chip_half_m = max(chip_min_half_m, width_m / 2.0 + mask_margin_m, height_m / 2.0 + mask_margin_m)
            chip_half_m = min(chip_half_m, chip_max_half_m)
            centroid = geom.centroid
            centroid_lon, centroid_lat = to_wgs84.transform(float(centroid.x), float(centroid.y))
            lon_min, lat_min, lon_max, lat_max = _bbox_to_wgs84(to_wgs84, float(centroid.x), float(centroid.y), chip_half_m)
            # Stable within the source annotation file: do not use len(rows),
            # otherwise the same feature receives different anchor IDs when a
            # run includes a different set/order of grids.
            anchor_id = f"{entry.region_key}_{entry.grid_id}_a{local_idx + 1:06d}"
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "region_key": entry.region_key,
                    "grid_id": entry.grid_id,
                    "source_annotation_path": str(entry.path),
                    "source_feature_id": local_idx,
                    "quality_tier": quality_tier,
                    "anchor_policy": "gt_centroid_buffered_bbox",
                    "centroid_lon": f"{centroid_lon:.10f}",
                    "centroid_lat": f"{centroid_lat:.10f}",
                    "source_area_m2": f"{float(geom.area):.4f}",
                    "source_width_m": f"{width_m:.4f}",
                    "source_height_m": f"{height_m:.4f}",
                    "chip_half_m": f"{chip_half_m:.2f}",
                    "search_radius_m": f"{search_radius_m:.2f}",
                    "chip_lon_min": f"{lon_min:.10f}",
                    "chip_lat_min": f"{lat_min:.10f}",
                    "chip_lon_max": f"{lon_max:.10f}",
                    "chip_lat_max": f"{lat_max:.10f}",
                    "alignment_note": "source mask used as anchor only; exact mask overlap is not required for historical GEID scoring",
                }
            )
            if max_anchors and len(rows) >= max_anchors:
                return rows
    return rows


def main() -> None:
    args = parse_args()
    if args.chip_min_half_m <= 0 or args.chip_max_half_m <= 0:
        raise SystemExit("chip half sizes must be positive")
    if args.chip_min_half_m > args.chip_max_half_m:
        raise SystemExit("--chip-min-half-m must be <= --chip-max-half-m")

    entries = _entries_from_args(args)
    if not entries:
        raise SystemExit("No annotation entries found.")
    rows = build_anchor_rows(
        entries,
        chip_min_half_m=args.chip_min_half_m,
        chip_max_half_m=args.chip_max_half_m,
        mask_margin_m=args.mask_margin_m,
        search_radius_m=args.search_radius_m,
        max_anchors=args.max_anchors,
    )
    if not rows:
        raise SystemExit("No valid anchor geometries found.")
    write_csv_rows(args.output, rows, ANCHOR_FIELDS)
    grids = sorted({str(row["grid_id"]) for row in rows})
    print(f"Wrote {len(rows)} anchors across {len(grids)} grids -> {args.output}")
    print("Policy: source masks are anchors only; historical scoring should use buffered chips/search windows.")


if __name__ == "__main__":
    main()
