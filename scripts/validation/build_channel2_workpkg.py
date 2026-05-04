#!/usr/bin/env python3
"""Build a Channel 2 (exhaustive recall) RA work package.

For each selected micro grid:
  1. Stitch Vexcel sub-tiles into a single mosaic GeoTIFF (6.7 cm GSD, JPEG-compressed).
  2. Copy Li GT (Joburg_CBD_Li_supp_v1) into the package, optionally shifted south
     by per-grid measured Δy to correct the systematic GEID/Vexcel offset.
  3. Copy V3-C predictions as a reference layer (NOT GT seed; styled differently in QGIS).
  4. Emit an empty annotation gpkg `{grid}_T1.gpkg` with the canonical schema.
  5. Write a per-grid README + a top-level QGIS project tip-sheet.

Output goes to data/annotations_channel2_micro/<grid>/ inside the project.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge
from shapely.affinity import translate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "annotations_channel2_micro"
DEFAULT_TILES_ROOT = Path("/home/gaosh/zasolar_data/tiles/johannesburg/vexcel_2024")
DEFAULT_LI_DIR = PROJECT_ROOT / "data" / "annotations" / "Joburg_CBD_Li_supp_v1"
DEFAULT_PRED_DIR = PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024"

# Per-grid measured Δy (Li − Vexcel pred) in metres in EPSG:32735 (UTM 35S).
# Source: 25-grid V3-C-vs-Li offset sweep (n_pairs IoU≥0.5 area-ratio≥0.5),
# captured 2026-04-29. Positive Δy = Li sits north of Vexcel-anchored truth,
# so shift Li south (negative Δy in UTM) to align with Vexcel.
GRID_OFFSETS_M = {
    # grid : (mean_dx, mean_dy) -- we apply opposite sign on shift
    "G0774": (0.17, 1.23),
    "G0816": (0.20, 1.27),
    "G0922": (0.03, 0.67),
}

DEFAULT_GRIDS = list(GRID_OFFSETS_M.keys())

# Annotation gpkg schema (matches data/annotations/Joburg_CBD_Li_supp_v1 layout
# extended with installation-spec fields).
T1_FIELDS = {
    "annotation_id": "TEXT",
    "label": "TEXT",          # 'pv'
    "axis_a": "TEXT",         # A1 / A2 / A3
    "axis_b": "TEXT",         # H / R / S / G
    "label_source": "TEXT",   # human_manual / human_manual_sam_assisted / ...
    "annotator": "TEXT",
    "review_notes": "TEXT",
    "merge_status": "TEXT",   # split / merged / unchanged (vs Li starting layer)
}


def stitch_mosaic(grid: str, tiles_root: Path, out_path: Path) -> dict:
    tiles = sorted((tiles_root / grid).glob(f"{grid}_*_geo.tif"))
    if not tiles:
        raise FileNotFoundError(f"No Vexcel tiles for {grid} under {tiles_root}")
    srcs = [rasterio.open(p) for p in tiles]
    try:
        mosaic, transform = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "jpeg",
            "tiled": True,
            "photometric": "ycbcr",
            "blockxsize": 512,
            "blockysize": 512,
        })
        meta.pop("nodata", None)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
        return {
            "tile_count": len(tiles),
            "width": int(mosaic.shape[2]),
            "height": int(mosaic.shape[1]),
            "size_mb": round(out_path.stat().st_size / 1e6, 1),
        }
    finally:
        for s in srcs:
            s.close()


def build_offset_corrected_li(grid: str, li_dir: Path, out_path: Path,
                              offset_m: tuple[float, float]) -> int:
    src = li_dir / f"{grid}.gpkg"
    if not src.exists():
        raise FileNotFoundError(src)
    gdf = gpd.read_file(src).to_crs(epsg=32735)
    dx, dy = offset_m
    # Li is +Δy north of pred → shift Li south by Δy to align with Vexcel pixels
    gdf["geometry"] = gdf.geometry.apply(lambda g: translate(g, xoff=-dx, yoff=-dy))
    gdf["source_layer"] = "Joburg_CBD_Li_supp_v1"
    gdf["offset_applied_dx"] = -dx
    gdf["offset_applied_dy"] = -dy
    gdf["axis_a"] = "A2"   # Li historical = A2 by default
    gdf["axis_b"] = "H"
    gdf.to_file(out_path, layer=f"li_starting_layer", driver="GPKG")
    return len(gdf)


def copy_pred_reference(grid: str, pred_dir: Path, out_path: Path) -> int:
    src = pred_dir / grid / "predictions_metric.gpkg"
    if not src.exists():
        raise FileNotFoundError(src)
    gdf = gpd.read_file(src).to_crs(epsg=32735)
    gdf["source_layer"] = "v3c_vexcel_2024"
    gdf.to_file(out_path, layer="v3c_pred_reference", driver="GPKG")
    return len(gdf)


def make_empty_t1(out_path: Path) -> None:
    schema_gdf = gpd.GeoDataFrame(
        {k: [] for k in T1_FIELDS}, geometry=gpd.GeoSeries([], crs="EPSG:32735")
    )
    schema_gdf.to_file(out_path, layer="annotations", driver="GPKG")


GRID_README_TEMPLATE = """\
# {grid} — Channel 2 Exhaustive Recall Annotation Package

**Status**: T1-target (gold evaluation GT)
**Source imagery**: Vexcel za-gp-johannesburg-2024 (6.7 cm GSD)
**Mosaic**: `{grid}_vexcel_mosaic.tif` ({mosaic_w}×{mosaic_h} px, {mosaic_mb} MB)
**Starting layer**: `{grid}_li_starting.gpkg` (Li_supp_v1 shifted Δx={dx:+.2f}m Δy={dy:+.2f}m to align with Vexcel)
**Reference (do NOT use as GT)**: `{grid}_v3c_pred.gpkg`
**Output**: `{grid}_T1.gpkg` ← RA edits this layer

## Task

Produce **exhaustive-recall installation-level GT** on this grid. Goal: every PV
installation visible on the Vexcel mosaic is captured exactly once, with
geometry tight to the installation footprint.

## Workflow in QGIS

1. Open `{grid}_T1.gpkg` (layer `annotations`) in editing mode.
2. Drag in `{grid}_li_starting.gpkg` and `{grid}_v3c_pred.gpkg` as reference.
3. For each Li starting polygon: confirm geometry against the Vexcel mosaic.
   - If correct → copy into `{grid}_T1.gpkg`, set `axis_a=A1`, `merge_status=unchanged`.
   - If geometry wrong → redraw; set `axis_a=A1`, `merge_status=unchanged`.
   - If Li merged multiple installations → split into separate polygons; set `merge_status=split`.
   - If Li split a single installation → merge; set `merge_status=merged`.
   - If Li polygon is **not** a PV installation (water heater, skylight, …) → drop it.
4. **Scan for missed panels**: pan the entire grid extent at zoom ≥ 1:500.
   Every visible PV installation that is NOT in the Li starting layer must be added.
   The V3-C reference layer can hint at candidates but **do not blindly trust it** —
   V3-C has ~30% FP rate on small dark targets (water heaters).
5. Set `axis_a` per polygon: A1 if installation-spec compliant, A2 if uncertain
   (note in `review_notes`).
6. Set `axis_b = H` for all (human-drawn). `label = 'pv'`.

## Annotation Rules (refer to data/annotations/ANNOTATION_SPEC.md)

- **One polygon per installation** — merge contiguous panels on same roof.
- **Tight outer envelope** — do not include surrounding roof / walkways unless
  they are <1 m gaps between panels.
- **Skip solar water heaters** (typically vertical-mounted dark cylinders or
  small flat horizontal heaters on rails — see project memory on PV-vs-thermal).
- **Skip if >50% obscured** by trees / shadow.
- **Edge tolerance** — boundary within 1–2 panel widths (~0.3–0.5 m) of
  installation edge.

## Provenance Fields (per polygon)

| field          | example value                  |
|----------------|--------------------------------|
| annotation_id  | `{grid}_T1_{{row}}` auto       |
| label          | pv                             |
| axis_a         | A1                             |
| axis_b         | H                              |
| label_source   | human_manual_sam_assisted      |
| annotator      | (RA name)                      |
| review_notes   | (free text, optional)          |
| merge_status   | unchanged / split / merged / new |

## Numerical Targets

- Existing Li polygons in starting layer: **{n_li}**
- V3-C reference polygons (for hints): **{n_pred}**
- Expected final T1 count: roughly **{n_pred}–{n_pred_plus} polygons** (V3-C
  often catches small panels Li missed; some V3-C are FP and will not become T1).
- Estimated time: **~{est_hours} hours**.
"""


def grid_workpkg(grid: str, args, top_root: Path) -> dict:
    out_dir = top_root / grid
    out_dir.mkdir(parents=True, exist_ok=True)

    mosaic_path = out_dir / f"{grid}_vexcel_mosaic.tif"
    if not mosaic_path.exists() or args.overwrite:
        m = stitch_mosaic(grid, args.tiles_root, mosaic_path)
    else:
        with rasterio.open(mosaic_path) as r:
            m = {"tile_count": -1, "width": r.width, "height": r.height,
                 "size_mb": round(mosaic_path.stat().st_size / 1e6, 1)}

    li_path = out_dir / f"{grid}_li_starting.gpkg"
    n_li = build_offset_corrected_li(grid, args.li_dir, li_path, GRID_OFFSETS_M[grid])

    pred_path = out_dir / f"{grid}_v3c_pred.gpkg"
    n_pred = copy_pred_reference(grid, args.pred_dir, pred_path)

    t1_path = out_dir / f"{grid}_T1.gpkg"
    if not t1_path.exists() or args.overwrite:
        make_empty_t1(t1_path)

    est_h = max(1.0, round(0.025 * n_li + 0.5, 1))
    readme = GRID_README_TEMPLATE.format(
        grid=grid,
        mosaic_w=m["width"], mosaic_h=m["height"], mosaic_mb=m["size_mb"],
        dx=-GRID_OFFSETS_M[grid][0], dy=-GRID_OFFSETS_M[grid][1],
        n_li=n_li, n_pred=n_pred, n_pred_plus=int(n_pred * 1.15),
        est_hours=est_h,
    )
    (out_dir / "README.md").write_text(readme)

    return {"grid": grid, "out_dir": out_dir, "n_li": n_li, "n_pred": n_pred,
            "mosaic": m, "est_hours": est_h}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grids", nargs="+", default=DEFAULT_GRIDS)
    p.add_argument("--tiles-root", type=Path, default=DEFAULT_TILES_ROOT)
    p.add_argument("--li-dir", type=Path, default=DEFAULT_LI_DIR)
    p.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for g in args.grids:
        if g not in GRID_OFFSETS_M:
            print(f"WARN: {g} has no measured offset; skipping. Add it to GRID_OFFSETS_M.",
                  file=sys.stderr)
            continue
        s = grid_workpkg(g, args, args.output_root)
        summaries.append(s)
        print(f"[{g}] mosaic {s['mosaic']['width']}x{s['mosaic']['height']}px "
              f"({s['mosaic']['size_mb']}MB)  Li={s['n_li']}  V3-C={s['n_pred']}  "
              f"~{s['est_hours']}h")

    total_h = sum(s["est_hours"] for s in summaries)
    total_li = sum(s["n_li"] for s in summaries)
    print(f"\nTotal: {len(summaries)} grids, ~{total_li} starting polygons, ~{total_h:.1f} RA hours")
    print(f"Output: {args.output_root}")


if __name__ == "__main__":
    main()
