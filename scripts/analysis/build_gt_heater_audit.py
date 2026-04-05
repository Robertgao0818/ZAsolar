#!/usr/bin/env python3
"""
GT 加热器污染审计 — 构建审计队列 + 导出 chip。

扫描 data/annotations/cleaned/*_SAM2_*.gpkg，为每个 GT polygon 生成审计记录，
按几何特征分层，导出 400×400 chip 供人工复核。

兼容两种 schema：
  - 早期文件 (Mar 20-24): 有 annotation_id, grid_id, num_parts，无几何特征
  - Apr 3 文件: 有 area_m2, elongation, solidity 等预计算字段，无 annotation_id

用法：
    python scripts/analysis/build_gt_heater_audit.py

    # 指定 run-id
    python scripts/analysis/build_gt_heater_audit.py --run-id heater_v1

    # 跳过 chip 导出（仅生成 CSV）
    python scripts/analysis/build_gt_heater_audit.py --skip-chips

    # 限制 chip 数量
    python scripts/analysis/build_gt_heater_audit.py --max-chips 1000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEANED_DIR = BASE_DIR / "data" / "annotations" / "cleaned"

import os
TILES_ROOT = Path(os.environ.get("SOLAR_TILES_ROOT", BASE_DIR / "tiles"))

METRIC_CRS = "EPSG:32734"  # UTM 34S for Cape Town


# ── Geometry helpers ───────────────────────────────────────────────────

def compute_geometry_bucket(elongation: float, solidity: float) -> str:
    """Reuse exact rules from analyze_small_fp.py:283-288."""
    if elongation > 6 or solidity < 0.7:
        return "sliver_fragment"
    if elongation > 2.5:
        return "elongated"
    return "compact"


def compute_elongation(geom) -> float:
    """Compute elongation from geometry's minimum rotated rectangle."""
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        d1 = ((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)**0.5
        d2 = ((coords[1][0] - coords[2][0])**2 + (coords[1][1] - coords[2][1])**2)**0.5
        major = max(d1, d2)
        minor = min(d1, d2)
        return major / minor if minor > 0 else 999.0
    except Exception:
        return 1.0


def compute_solidity(geom) -> float:
    """Compute solidity = area / convex_hull_area."""
    try:
        ch_area = geom.convex_hull.area
        return geom.area / ch_area if ch_area > 0 else 1.0
    except Exception:
        return 1.0


# ── Risk tier ──────────────────────────────────────────────────────────

def assign_risk_tier(area_m2: float, geometry_bucket: str, solidity: float) -> str:
    """
    A: 5 <= area < 30, compact, solidity > 0.9 — heater/pool heater core shape
    B: 5 <= area < 30, elongated
    C: 5 <= area < 30, sliver_fragment
    D: everything else
    """
    if 5 <= area_m2 < 30:
        if geometry_bucket == "compact" and solidity > 0.9:
            return "A"
        if geometry_bucket == "elongated":
            return "B"
        if geometry_bucket == "sliver_fragment":
            return "C"
        # compact but solidity <= 0.9
        return "B"
    return "D"


# ── Discover & load ───────────────────────────────────────────────────

def discover_cleaned_sources() -> dict[str, dict]:
    """Auto-discover *_SAM2_*.gpkg from cleaned/ dir, same logic as export_coco."""
    sources = {}
    for f in sorted(CLEANED_DIR.glob("*_SAM2_*.gpkg")):
        grid_id = f.name.split("_SAM2_")[0]
        layers = fiona.listlayers(str(f))
        if layers:
            sources[grid_id] = {"file": f, "layer": layers[0]}
    return sources


def load_and_normalize(grid_id: str, src_info: dict) -> pd.DataFrame:
    """Load a single grid's GPKG and normalize to a flat DataFrame with
    consistent columns regardless of schema version.

    Returns DataFrame with columns:
        audit_id, grid_id, row_index, source_file,
        area_m2, elongation, solidity, geometry_bucket, risk_tier,
        confidence, review_status, source, geometry (in METRIC_CRS)
    """
    gdf = gpd.read_file(str(src_info["file"]), layer=src_info["layer"])
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    gdf = gdf.reset_index(drop=True)

    if len(gdf) == 0:
        return pd.DataFrame()

    # Set CRS if missing, then project to metric CRS
    if gdf.crs is None:
        # Assume EPSG:4326 for files without CRS
        gdf = gdf.set_crs("EPSG:4326")

    if str(gdf.crs) == METRIC_CRS or (gdf.crs and gdf.crs.to_epsg() == 32734):
        gdf_metric = gdf
    else:
        gdf_metric = gdf.to_crs(METRIC_CRS)

    rows = []
    has_precomputed = "area_m2" in gdf.columns and "elongation" in gdf.columns
    source_file = src_info["file"].name

    for idx, (_, row) in enumerate(gdf_metric.iterrows()):
        geom = row.geometry

        # Area
        if has_precomputed and pd.notna(gdf.iloc[idx].get("area_m2")):
            area_m2 = float(gdf.iloc[idx]["area_m2"])
        else:
            area_m2 = geom.area

        # Elongation
        if has_precomputed and pd.notna(gdf.iloc[idx].get("elongation")):
            elong = float(gdf.iloc[idx]["elongation"])
        else:
            elong = compute_elongation(geom)

        # Solidity
        if has_precomputed and pd.notna(gdf.iloc[idx].get("solidity")):
            solid = float(gdf.iloc[idx]["solidity"])
        else:
            solid = compute_solidity(geom)

        geo_bucket = compute_geometry_bucket(elong, solid)
        tier = assign_risk_tier(area_m2, geo_bucket, solid)

        # Optional fields
        conf = float(gdf.iloc[idx].get("confidence", 0)) if "confidence" in gdf.columns else None
        rev_status = str(gdf.iloc[idx].get("review_status", "")) if "review_status" in gdf.columns else ""
        source = str(gdf.iloc[idx].get("source", "")) if "source" in gdf.columns else ""

        rows.append({
            "audit_id": f"{grid_id}_{idx:04d}",
            "grid_id": grid_id,
            "row_index": idx,
            "source_file": source_file,
            "area_m2": round(area_m2, 2),
            "elongation": round(elong, 3),
            "solidity": round(solid, 3),
            "geometry_bucket": geo_bucket,
            "risk_tier": tier,
            "confidence": round(conf, 4) if conf is not None else "",
            "review_status": rev_status,
            "source": source,
            "audit_label": "",
            "audit_notes": "",
            "reviewed_at": "",
        })

    df = pd.DataFrame(rows)
    # Attach metric geometry for chip export
    df["geometry"] = gdf_metric.geometry.values
    return df


# ── Chip export ────────────────────────────────────────────────────────

def draw_polygon_outline(draw, geom, transform, color="red", width=2):
    """Draw polygon outline, handling Polygon and MultiPolygon."""
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        coords = []
        for x, y in poly.exterior.coords:
            px, py = ~transform * (x, y)
            coords.append((int(px), int(py)))
        if len(coords) >= 3:
            draw.polygon(coords, outline=color, width=width)


def get_tile_for_centroid(grid_id: str, centroid_x: float, centroid_y: float) -> Path | None:
    """Find the tile containing a centroid (in tile CRS, typically EPSG:4326)."""
    tiles_dir = TILES_ROOT / grid_id
    if not tiles_dir.exists():
        return None

    # Try _geo.tif first, then regular .tif
    tiles = sorted(tiles_dir.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted([
            f for f in tiles_dir.glob(f"{grid_id}_*_*.tif")
            if "_geo" not in f.stem and "mosaic" not in f.stem
        ])

    for tile_path in tiles:
        try:
            with rasterio.open(str(tile_path)) as src:
                if src.bounds.left <= centroid_x <= src.bounds.right and \
                   src.bounds.bottom <= centroid_y <= src.bounds.top:
                    return tile_path
        except Exception:
            continue
    return None


def export_chips(
    df: pd.DataFrame,
    output_dir: Path,
    chip_size: int = 400,
    max_chips: int = 2000,
) -> int:
    """Export RGB + overlay chips for audit records."""
    print("\n=== Exporting chips ===")

    if len(df) == 0:
        print("  No records to export")
        return 0

    if len(df) > max_chips:
        df = df.sample(max_chips, random_state=42)
        print(f"  Sampled {max_chips} chips")

    rgb_dir = output_dir / "chips" / "rgb"
    overlay_dir = output_dir / "chips" / "overlay"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    transformer = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    half = chip_size // 2
    count = 0
    chip_paths = {}  # audit_id → relative chip path

    for _, row in df.iterrows():
        grid_id = row["grid_id"]
        audit_id = row["audit_id"]
        geom_metric = row["geometry"]
        try:
            if geom_metric is None or geom_metric.is_empty:
                continue
            cx_m, cy_m = geom_metric.centroid.x, geom_metric.centroid.y
        except Exception:
            continue

        # Convert centroid to tile CRS (EPSG:4326) for tile lookup
        cx_4326, cy_4326 = transformer.transform(cx_m, cy_m)

        tile_path = get_tile_for_centroid(grid_id, cx_4326, cy_4326)
        if tile_path is None:
            continue

        try:
            with rasterio.open(str(tile_path)) as src:
                tile_crs = src.crs

                # Reproject geometry to tile CRS
                row_geom = gpd.GeoSeries(
                    [geom_metric], crs=METRIC_CRS,
                ).to_crs(tile_crs).iloc[0]

                cx, cy = row_geom.centroid.x, row_geom.centroid.y
                px, py = ~src.transform * (cx, cy)
                px, py = int(px), int(py)

                row_start = max(0, py - half)
                col_start = max(0, px - half)
                row_stop = min(src.height, py + half)
                col_stop = min(src.width, px + half)

                window = rasterio.windows.Window.from_slices(
                    (row_start, row_stop), (col_start, col_stop),
                )
                data = src.read(window=window)

                if data.shape[1] < 20 or data.shape[2] < 20:
                    continue

                # Pad to full chip_size if clipped at tile edge
                actual_h, actual_w = data.shape[1], data.shape[2]
                if actual_h < chip_size or actual_w < chip_size:
                    padded = np.zeros((data.shape[0], chip_size, chip_size),
                                     dtype=data.dtype)
                    y_off = half - (py - row_start)
                    x_off = half - (px - col_start)
                    padded[:, y_off:y_off + actual_h, x_off:x_off + actual_w] = data
                    data = padded
                    gt = src.transform
                    win_origin_x = gt.c + (col_start - x_off) * gt.a
                    win_origin_y = gt.f + (row_start - y_off) * gt.e
                    window_transform = Affine(gt.a, gt.b, win_origin_x,
                                              gt.d, gt.e, win_origin_y)
                else:
                    window_transform = rasterio.windows.transform(window, src.transform)

                # Pure RGB chip
                bands = min(3, data.shape[0])
                img_rgb = Image.fromarray(data[:bands].transpose(1, 2, 0))
                fname = f"{audit_id}.png"
                img_rgb.save(str(rgb_dir / fname))

                # Overlay version
                img_overlay = img_rgb.copy()
                draw = ImageDraw.Draw(img_overlay)

                # Red outline for target GT polygon
                draw_polygon_outline(draw, row_geom, window_transform, "red", 2)

                # Text annotation
                area = row.get("area_m2", 0)
                tier = row.get("risk_tier", "?")
                draw.text(
                    (4, 4),
                    f"GT audit | tier={tier} | {area:.1f}m²",
                    fill="yellow",
                )

                img_overlay.save(str(overlay_dir / fname))

                chip_paths[audit_id] = f"chips/overlay/{fname}"
                count += 1

        except Exception as e:
            print(f"  [WARN] chip failed for {audit_id}: {e}")

    print(f"  Exported {count} chips")
    return count, chip_paths


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GT 加热器污染审计 — 构建审计队列 + 导出 chip",
    )
    parser.add_argument("--run-id", default=None,
                        help="运行 ID（默认 heater_YYYYMMDD_HHMM）")
    parser.add_argument("--skip-chips", action="store_true",
                        help="跳过 chip 导出，仅生成 CSV")
    parser.add_argument("--max-chips", type=int, default=2000,
                        help="最大 chip 导出数量")
    parser.add_argument("--chip-size", type=int, default=400)
    args = parser.parse_args()

    run_id = args.run_id or f"heater_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir = BASE_DIR / "results" / "analysis" / "gt_heater_audit" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Discover & load ──────────────────────────────────────
    print("=== Phase 1: Loading GT annotations ===")
    sources = discover_cleaned_sources()
    print(f"  Found {len(sources)} cleaned SAM2 files")

    all_records = []
    skipped_no_tiles = 0

    for grid_id, src_info in sorted(sources.items()):
        tiles_dir = TILES_ROOT / grid_id
        if not tiles_dir.exists():
            skipped_no_tiles += 1
            continue

        df = load_and_normalize(grid_id, src_info)
        if len(df) > 0:
            all_records.append(df)
            print(f"  {grid_id}: {len(df)} GT polygons")

    if skipped_no_tiles:
        print(f"  Skipped {skipped_no_tiles} grids (no tiles)")

    if not all_records:
        print("ERROR: No GT records found")
        sys.exit(1)

    full_df = pd.concat(all_records, ignore_index=True)
    print(f"\n  Total: {len(full_df)} GT records")

    # ── Phase 2: Risk tier summary ────────────────────────────────────
    print("\n=== Phase 2: Risk tier summary ===")
    tier_counts = full_df["risk_tier"].value_counts().sort_index()
    for tier, cnt in tier_counts.items():
        print(f"  Tier {tier}: {cnt}")

    phase1_df = full_df[full_df["risk_tier"] == "A"].copy()
    print(f"\n  Phase 1 queue (tier A): {len(phase1_df)} records")

    # ── Phase 3: Export chips ─────────────────────────────────────────
    chip_paths = {}
    if not args.skip_chips and len(phase1_df) > 0:
        result = export_chips(
            phase1_df, output_dir,
            chip_size=args.chip_size, max_chips=args.max_chips,
        )
        if isinstance(result, tuple):
            _, chip_paths = result

    # Update chip paths in DataFrames
    full_df["chip_path"] = full_df["audit_id"].map(chip_paths).fillna("")
    full_df["overlay_path"] = full_df["chip_path"]  # same for now

    phase1_df = full_df[full_df["risk_tier"] == "A"].copy()

    # ── Phase 4: Save CSVs ────────────────────────────────────────────
    print("\n=== Phase 4: Saving CSVs ===")

    csv_cols = [
        "audit_id", "grid_id", "row_index", "source_file",
        "area_m2", "elongation", "solidity",
        "geometry_bucket", "risk_tier", "confidence", "review_status", "source",
        "chip_path", "overlay_path", "audit_label", "audit_notes", "reviewed_at",
    ]

    full_csv = output_dir / "audit_queue_full.csv"
    full_df[csv_cols].to_csv(full_csv, index=False)
    print(f"  Full queue: {full_csv} ({len(full_df)} rows)")

    phase1_csv = output_dir / "audit_queue_phase1.csv"
    phase1_df[csv_cols].to_csv(phase1_csv, index=False)
    print(f"  Phase 1 queue: {phase1_csv} ({len(phase1_df)} rows)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n=== Done ===")
    print(f"  Output: {output_dir}")
    print(f"  Full queue: {len(full_df)} GT records across {full_df['grid_id'].nunique()} grids")
    print(f"  Phase 1 (tier A): {len(phase1_df)} records to review")
    if chip_paths:
        print(f"  Chips exported: {len(chip_paths)}")

    # Area distribution for phase 1
    if len(phase1_df) > 0:
        print(f"\n  Phase 1 area_m2 stats:")
        stats = phase1_df["area_m2"].describe()
        print(f"    min={stats['min']:.1f}  median={stats['50%']:.1f}  "
              f"max={stats['max']:.1f}  mean={stats['mean']:.1f}")


if __name__ == "__main__":
    main()
