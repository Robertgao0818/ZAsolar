"""Boundary-ignore band rasterization for SAM-derived mask supervision.

For a polygon's binary fg mask, builds an ``ignore band`` of width ``band_px``
straddling the polygon edge. BCE loss should skip ignore pixels.

  ignore_band = dilate(fg, k=band_px) AND NOT erode(fg, k=band_px)

That gives a ring of ``2 * band_px`` total width centered on the edge.

Usage as a library:

    from scripts.training.jhb_phaseA.boundary_ignore import (
        rasterize_polygon_with_ignore,
        rasterize_target_masks,
    )

    fg_mask, ignore_mask = rasterize_polygon_with_ignore(
        polygon_xy_pixel,  # (N, 2) np array
        h=H, w=W,
        band_px=2,
    )

CLI: produce a side-by-side visualization of (image, fg, ignore band) for
a few real polygons from a JHB grid, so we can eyeball band width before
plumbing it into train.py.

    python scripts/training/jhb_phaseA/boundary_ignore.py \
        --grid G0922 --n-samples 8 --band-px 2 \
        --out results/analysis/jhb_phaseA_prep/band_dry_run/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box as shapely_box

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402

REVIEW_ROOT = PROJECT_ROOT / "results/johannesburg/v3c_vexcel_2024_ch1_sample"


def rasterize_polygon_with_ignore(
    pts_pixel: np.ndarray,
    h: int,
    w: int,
    band_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one polygon to (fg_mask, ignore_mask).

    Args:
        pts_pixel: (N, 2) int32 polygon vertices in pixel coords.
        h, w: output mask size.
        band_px: half-width of the ignore band on each side of the edge.
                 Total band width = 2 * band_px pixels.

    Returns:
        fg_mask: uint8 (h, w) with 1 inside polygon, 0 outside.
        ignore_mask: uint8 (h, w) with 1 in the band straddling the edge,
                     0 elsewhere. BCE should multiply by (1 - ignore_mask).
    """
    fg = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(pts_pixel).astype(np.int32).reshape(-1, 2)
    cv2.fillPoly(fg, [pts], 1)
    if band_px <= 0:
        return fg, np.zeros_like(fg)
    k = 2 * band_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(fg, kernel)
    ero = cv2.erode(fg, kernel)
    ignore = (dil & ~ero).astype(np.uint8)
    return fg, ignore


def rasterize_target_masks(
    polygons_xy_pixel: list[np.ndarray],
    h: int,
    w: int,
    band_px_per_inst: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized helper for an entire image: stack of per-instance fg + ignore.

    Returns:
        fg_stack:     uint8 (N, h, w)
        ignore_stack: uint8 (N, h, w)
    """
    n = len(polygons_xy_pixel)
    fg_stack = np.zeros((n, h, w), dtype=np.uint8)
    ig_stack = np.zeros((n, h, w), dtype=np.uint8)
    for i, (pts, bp) in enumerate(zip(polygons_xy_pixel, band_px_per_inst)):
        fg_stack[i], ig_stack[i] = rasterize_polygon_with_ignore(pts, h, w, bp)
    return fg_stack, ig_stack


def get_band_px_for_area(area_m2: float) -> int:
    """SAM mask boundary-ignore band width by area bucket (Phase A spec).

    <300 m²    -> 2 px (≈ 13.4 cm at Vexcel 6.7cm GSD; covers SAM wobble)
    300-600 m² -> 3 px (edge softer at larger scale)
    >=600 m²   -> 4 px (large arrays + step structure; user-verified content correct)
    """
    if area_m2 < 300:
        return 2
    if area_m2 < 600:
        return 3
    return 4


def _band_px_label(bp: int) -> str:
    return f"band_{bp}px"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="G0922",
                    help="JHB CBD grid id (default: G0922 — has 215 SAM polygons)")
    ap.add_argument("--source", choices=["sam_added", "v3c_correct"], default="sam_added")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--band-px", type=int, default=2,
                    help="Override fixed band; if 0, use area-bucket rule")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chip-size", type=int, default=400)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "results/analysis/jhb_phaseA_prep/band_dry_run")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    grid = args.grid
    if args.source == "sam_added":
        gpkg = REVIEW_ROOT / grid / "review" / f"{grid}_sam_added.gpkg"
    else:
        gpkg = REVIEW_ROOT / grid / "review" / f"{grid}_reviewed.gpkg"
    if not gpkg.exists():
        print(f"[ERR] {gpkg} not found", file=sys.stderr)
        sys.exit(1)

    gdf = gpd.read_file(gpkg)
    if args.source == "v3c_correct" and "review_status" in gdf.columns:
        gdf = gdf[gdf.review_status == "correct"].reset_index(drop=True)
    if len(gdf) == 0:
        print(f"[WARN] no polygons found for {grid}/{args.source}")
        sys.exit(0)

    if "area_m2" in gdf.columns:
        areas_m2 = gdf["area_m2"].astype(float).values
    else:
        gdf_metric = gdf.to_crs("EPSG:32735") if str(gdf.crs) != "EPSG:32735" else gdf
        areas_m2 = gdf_metric.geometry.area.values

    n = min(args.n_samples, len(gdf))
    idxs = rng.choice(len(gdf), size=n, replace=False)
    print(f"[GRID] {grid} {args.source}: total={len(gdf)}, sampling {n}")

    tiles_dir = resolve_tiles_dir(grid, region="johannesburg", imagery_layer="vexcel_2024")
    if tiles_dir.is_file():
        tiles = [tiles_dir]
    else:
        tiles = sorted(tiles_dir.glob(f"{grid}_*_*_geo.tif"))
    if not tiles:
        print(f"[ERR] no tiles for {grid}", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    written = 0
    chip = args.chip_size
    for poly_idx in idxs:
        g = gdf.geometry.iloc[poly_idx]
        a = float(areas_m2[poly_idx])
        bp = args.band_px if args.band_px > 0 else get_band_px_for_area(a)

        for tile_path in tiles:
            with rasterio.open(tile_path) as src:
                if str(gdf.crs) != str(src.crs):
                    g_local = gpd.GeoSeries([g], crs=gdf.crs).to_crs(src.crs).iloc[0]
                else:
                    g_local = g
                tb = src.bounds
                if not shapely_box(tb.left, tb.bottom, tb.right, tb.top).intersects(g_local):
                    continue
                cx, cy = g_local.centroid.x, g_local.centroid.y
                inv = ~src.transform
                col, row = inv * (cx, cy)
                x0 = max(0, int(round(col)) - chip // 2)
                y0 = max(0, int(round(row)) - chip // 2)
                x0 = min(x0, src.width - chip)
                y0 = min(y0, src.height - chip)
                if x0 < 0 or y0 < 0:
                    continue
                window = Window(x0, y0, chip, chip)
                rgb = src.read([1, 2, 3], window=window)
                rgb = np.transpose(rgb, (1, 2, 0))

                pts_world = np.array(g_local.exterior.coords)
                cols, rows = inv * (pts_world[:, 0], pts_world[:, 1])
                pts_pixel = np.stack([cols - x0, rows - y0], axis=1)

                fg, ignore = rasterize_polygon_with_ignore(pts_pixel, chip, chip, bp)

                vis = rgb.copy()
                if vis.dtype != np.uint8:
                    vis = np.clip(vis, 0, 255).astype(np.uint8)
                overlay = vis.copy()
                overlay[ignore == 1] = (255, 0, 0)
                overlay[fg == 1] = (
                    0.5 * overlay[fg == 1] + 0.5 * np.array([0, 200, 0])
                ).astype(np.uint8)
                blend = (0.5 * vis + 0.5 * overlay).astype(np.uint8)

                stem = (
                    f"{grid}_{args.source}_idx{poly_idx:04d}"
                    f"_a{a:.0f}m2_{_band_px_label(bp)}.png"
                )
                cv2.imwrite(str(args.out / stem), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
                written += 1
                break

    print(f"[DONE] wrote {written} preview PNGs -> {args.out}")


if __name__ == "__main__":
    main()
