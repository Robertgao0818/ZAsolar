"""Build a QGIS work package for clean boundary hand-refinement.

The package is intentionally separate from the reviewed-prediction training
pool.  It samples a few hundred representative polygons from the JHB Phase A
review outputs, creates an editable copy, and adds local Vexcel 2024 imagery as
the basemap.

Outputs:
    results/analysis/jhb_phaseA_boundary_refine_qgis/
      boundary_refine.qgs
      boundary_refine_workpkg.gpkg
      rasters/<grid>_vexcel_2024.vrt
      README.md
"""
from __future__ import annotations

import argparse
import html
import math
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402


REVIEW_ROOT = PROJECT_ROOT / "results/johannesburg/v3c_vexcel_2024_ch1_sample"
OUT_ROOT = PROJECT_ROOT / "results/analysis/jhb_phaseA_boundary_refine_qgis"

TARGET_CRS = "EPSG:3857"  # matches Vexcel 2024 tiles
METRIC_CRS = "EPSG:32735"  # Johannesburg UTM metric CRS

GRIDS = [
    "G0772", "G0773", "G0774", "G0775", "G0776",
    "G0814", "G0815", "G0816", "G0817", "G0818",
    "G0853", "G0854", "G0855", "G0856", "G0857",
    "G0888", "G0889", "G0890", "G0891", "G0892",
    "G0922", "G0923", "G0924", "G0925", "G0926",
]

AREA_BINS = [0, 10, 20, 40, 80, 150, 300, 600, float("inf")]
AREA_LABELS = ["<10", "10-20", "20-40", "40-80", "80-150", "150-300", "300-600", ">=600"]

SAM_TARGETS = {
    "<10": 25,
    "10-20": 35,
    "20-40": 35,
    "40-80": 35,
    "80-150": 25,
    "150-300": 25,
    "300-600": 20,
    ">=600": 999,  # keep all very large SAM arrays
}

V3C_TARGETS = {
    "<10": 10,
    "10-20": 10,
    "20-40": 12,
    "40-80": 12,
    "80-150": 12,
    "150-300": 12,
    "300-600": 10,
    ">=600": 10,
}


@dataclass(frozen=True)
class RasterLayer:
    layer_id: str
    name: str
    source: Path
    extent: tuple[float, float, float, float]


@dataclass(frozen=True)
class VectorLayer:
    layer_id: str
    name: str
    source: str
    extent: tuple[float, float, float, float]
    geometry: str
    color: str
    outline: str
    read_only: bool


def _area_bucket(area_m2: float) -> str:
    for label, lo, hi in zip(AREA_LABELS, AREA_BINS[:-1], AREA_BINS[1:]):
        if lo <= area_m2 < hi:
            return label
    return ">=600"


def _safe_wobble(geom, tol_m: float) -> float:
    if geom is None or geom.is_empty:
        return float("nan")
    perimeter = float(geom.length)
    if perimeter <= 0:
        return float("nan")
    simplified = geom.simplify(tol_m, preserve_topology=True)
    simp_perimeter = float(simplified.length)
    if simp_perimeter <= 0:
        return float("nan")
    return perimeter / simp_perimeter


def _read_source(grid: str, source_pool: str) -> gpd.GeoDataFrame:
    if source_pool == "sam_added":
        path = REVIEW_ROOT / grid / "review" / f"{grid}_sam_added.gpkg"
    elif source_pool == "v3c_correct":
        path = REVIEW_ROOT / grid / "review" / f"{grid}_reviewed.gpkg"
    else:
        raise ValueError(source_pool)

    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)

    gdf = gpd.read_file(path)
    if source_pool == "v3c_correct" and "review_status" in gdf.columns:
        gdf = gdf[gdf["review_status"] == "correct"].copy()
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)

    source_idx = gdf.index.to_numpy()
    gdf = gdf.reset_index(drop=True)
    gdf["grid"] = grid
    gdf["source_pool"] = source_pool
    gdf["source_idx"] = source_idx.astype(int)
    gdf["ref_id"] = [
        f"{source_pool}_{grid}_{idx:04d}" for idx in gdf["source_idx"].astype(int)
    ]
    return gdf


def _load_all_candidates() -> gpd.GeoDataFrame:
    frames = []
    for grid in GRIDS:
        for source_pool in ("sam_added", "v3c_correct"):
            gdf = _read_source(grid, source_pool)
            if not gdf.empty:
                frames.append(gdf)
    if not frames:
        raise RuntimeError(f"No review polygons found under {REVIEW_ROOT}")

    raw = pd.concat(frames, ignore_index=True)
    raw = gpd.GeoDataFrame(raw, geometry="geometry", crs=frames[0].crs)
    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    raw["geometry"] = raw.geometry.buffer(0)
    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()

    target = raw.to_crs(TARGET_CRS)
    metric = raw.to_crs(METRIC_CRS)

    target["area_m2"] = metric.geometry.area.astype(float)
    target["perimeter_m"] = metric.geometry.length.astype(float)
    target["area_bucket"] = target["area_m2"].map(_area_bucket)
    target["wobble_02m"] = metric.geometry.map(lambda g: _safe_wobble(g, 0.2))
    target["wobble_05m"] = metric.geometry.map(lambda g: _safe_wobble(g, 0.5))
    target["wobble_10m"] = metric.geometry.map(lambda g: _safe_wobble(g, 1.0))
    target["bbox_fill"] = [
        float(g.area / g.envelope.area) if g.envelope.area > 0 else float("nan")
        for g in metric.geometry
    ]
    target["centroid_x"] = target.geometry.centroid.x.astype(float)
    target["centroid_y"] = target.geometry.centroid.y.astype(float)

    target = _add_density(target)
    target = _add_imagery_metrics(target)
    target = _assign_focus(target)
    target = _add_priority(target)
    return target


def _add_density(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    metric = gdf.to_crs(METRIC_CRS)
    centroids = metric.geometry.centroid
    sindex = gpd.GeoSeries(centroids, crs=METRIC_CRS).sindex
    counts: list[int] = []
    for i, pt in enumerate(centroids):
        candidate_idx = list(sindex.query(pt.buffer(40.0), predicate="intersects"))
        counts.append(max(0, len(candidate_idx) - 1))
    gdf["density_40m"] = counts
    return gdf


def _tile_paths_for_grid(grid: str) -> list[Path]:
    tiles_dir = resolve_tiles_dir(grid, region="johannesburg", imagery_layer="vexcel_2024")
    if tiles_dir.is_file():
        return [tiles_dir]
    return sorted(tiles_dir.glob(f"{grid}_*_*_geo.tif"))


def _add_imagery_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    brightness = pd.Series(np.nan, index=gdf.index, dtype=float)
    contrast = pd.Series(np.nan, index=gdf.index, dtype=float)

    for grid, sub in gdf.groupby("grid"):
        tile_paths = _tile_paths_for_grid(grid)
        if not tile_paths:
            continue
        datasets = []
        try:
            for path in tile_paths:
                src = rasterio.open(path)
                datasets.append(src)
            for idx, row in sub.iterrows():
                x, y = float(row["centroid_x"]), float(row["centroid_y"])
                src = next(
                    (
                        ds for ds in datasets
                        if ds.bounds.left <= x <= ds.bounds.right
                        and ds.bounds.bottom <= y <= ds.bounds.top
                    ),
                    None,
                )
                if src is None:
                    continue
                col, row_px = ~src.transform * (x, y)
                chip = 96
                window = Window(
                    int(round(col)) - chip // 2,
                    int(round(row_px)) - chip // 2,
                    chip,
                    chip,
                )
                arr = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                valid = arr.sum(axis=0) > 0
                if not valid.any():
                    continue
                vals = arr[:, valid].astype(np.float32)
                brightness.loc[idx] = float(vals.mean())
                contrast.loc[idx] = float(vals.std())
        finally:
            for src in datasets:
                src.close()

    gdf["brightness_mean"] = brightness
    gdf["contrast_std"] = contrast
    return gdf


def _assign_focus(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    density_cut = float(np.nanpercentile(gdf["density_40m"], 75))
    bright_cut = float(np.nanpercentile(gdf["brightness_mean"], 20))
    contrast_cut = float(np.nanpercentile(gdf["contrast_std"], 20))
    sam_wobble_cut = float(
        np.nanpercentile(gdf.loc[gdf.source_pool == "sam_added", "wobble_10m"], 75)
    )
    v3c_smooth_cut = float(
        np.nanpercentile(gdf.loc[gdf.source_pool == "v3c_correct", "wobble_10m"], 35)
    )

    focus = []
    for _, row in gdf.iterrows():
        area = float(row["area_m2"])
        density = float(row["density_40m"])
        bright = float(row["brightness_mean"]) if not pd.isna(row["brightness_mean"]) else math.inf
        con = float(row["contrast_std"]) if not pd.isna(row["contrast_std"]) else math.inf
        wobble = float(row["wobble_10m"]) if not pd.isna(row["wobble_10m"]) else 1.0
        source = row["source_pool"]

        if source == "sam_added" and area >= 600:
            label = "large_step_array"
        elif source == "sam_added" and area >= 150:
            label = "large_multistep"
        elif density >= density_cut and area < 150:
            label = "dense_roof"
        elif bright <= bright_cut or con <= contrast_cut:
            label = "shadow_low_contrast"
        elif source == "sam_added" and wobble >= sam_wobble_cut:
            label = "sam_wobble"
        elif source == "v3c_correct" and area >= 80 and wobble <= v3c_smooth_cut:
            label = "v3c_halo_probe"
        elif area < 20:
            label = "small_residential_single"
        elif area < 80:
            label = "medium_residential"
        else:
            label = "representative"
        focus.append(label)

    gdf["focus"] = focus
    return gdf


def _rank01(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=ascending).fillna(0.5)


def _add_priority(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    high_wobble = _rank01(gdf["wobble_10m"], ascending=True)
    high_density = _rank01(gdf["density_40m"], ascending=True)
    low_brightness = _rank01(gdf["brightness_mean"], ascending=False)
    low_contrast = _rank01(gdf["contrast_std"], ascending=False)
    large_area = _rank01(gdf["area_m2"], ascending=True)
    gdf["sample_priority"] = (
        0.30 * high_wobble
        + 0.25 * high_density
        + 0.20 * low_brightness
        + 0.15 * low_contrast
        + 0.10 * large_area
    )
    return gdf


def _select_candidates(gdf: gpd.GeoDataFrame, seed: int) -> gpd.GeoDataFrame:
    selected_parts = []
    for source_pool, targets in (("sam_added", SAM_TARGETS), ("v3c_correct", V3C_TARGETS)):
        src = gdf[gdf["source_pool"] == source_pool]
        for bucket, target_n in targets.items():
            group = src[src["area_bucket"] == bucket].copy()
            if group.empty:
                continue
            if len(group) <= target_n:
                selected_parts.append(group)
                continue

            top_n = int(math.ceil(target_n * 0.60))
            priority = group.nlargest(top_n, "sample_priority")
            remainder = group.drop(priority.index)
            random_n = target_n - len(priority)
            if random_n > 0 and not remainder.empty:
                random_part = remainder.sample(
                    n=min(random_n, len(remainder)),
                    random_state=seed + len(selected_parts),
                )
                selected_parts.append(pd.concat([priority, random_part]))
            else:
                selected_parts.append(priority)

    selected = pd.concat(selected_parts, ignore_index=False)
    selected = selected.drop_duplicates(subset=["ref_id"]).copy()
    selected = selected.sort_values(["grid", "source_pool", "source_idx"]).reset_index(drop=True)
    selected["sample_rank"] = np.arange(1, len(selected) + 1)
    return gpd.GeoDataFrame(selected, geometry="geometry", crs=gdf.crs)


def _write_workpkg(selected: gpd.GeoDataFrame, all_candidates: gpd.GeoDataFrame, out_dir: Path) -> Path:
    gpkg = out_dir / "boundary_refine_workpkg.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    keep_cols = [
        "ref_id", "sample_rank", "grid", "source_pool", "source_idx", "focus",
        "area_bucket", "area_m2", "perimeter_m", "bbox_fill",
        "wobble_02m", "wobble_05m", "wobble_10m", "density_40m",
        "brightness_mean", "contrast_std", "confidence", "review_status",
        "source_tile", "sample_priority", "geometry",
    ]
    cols = [c for c in keep_cols if c in selected.columns]
    reference = selected[cols].copy()
    reference.to_file(gpkg, layer="reference_polygons", driver="GPKG")

    edit = reference.copy()
    edit["clean_id"] = edit["ref_id"]
    edit["edit_status"] = "pending"
    edit["use_for_training"] = 1
    edit["edited_by"] = ""
    edit["edit_notes"] = ""
    edit["boundary_source"] = "manual_qgis"
    edit.to_file(gpkg, layer="clean_boundary_edit", driver="GPKG", mode="a")

    centroids = reference.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids[[
        "ref_id", "sample_rank", "grid", "source_pool", "focus", "area_bucket",
        "area_m2", "geometry",
    ]].to_file(gpkg, layer="ref_centroids", driver="GPKG", mode="a")

    tile_index = _build_tile_index(sorted(selected["grid"].unique()))
    tile_index.to_file(gpkg, layer="vexcel_tile_index", driver="GPKG", mode="a")

    selected.drop(columns="geometry").to_csv(out_dir / "selected_candidates.csv", index=False)
    summary = (
        selected.groupby(["source_pool", "area_bucket", "focus"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["source_pool", "area_bucket", "focus"])
    )
    summary.to_csv(out_dir / "selection_summary.csv", index=False)

    all_summary = (
        all_candidates.groupby(["source_pool", "area_bucket"], observed=True)
        .size()
        .reset_index(name="available_n")
        .sort_values(["source_pool", "area_bucket"])
    )
    all_summary.to_csv(out_dir / "available_pool_summary.csv", index=False)
    return gpkg


def _build_tile_index(grids: list[str]) -> gpd.GeoDataFrame:
    rows = []
    for grid in grids:
        for path in _tile_paths_for_grid(grid):
            with rasterio.open(path) as src:
                b = src.bounds
                rows.append({
                    "grid": grid,
                    "tile_name": path.stem,
                    "path": str(path),
                    "geometry": box(b.left, b.bottom, b.right, b.top),
                })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS)


def _write_vrts(grids: list[str], out_dir: Path) -> list[RasterLayer]:
    raster_dir = out_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    layers: list[RasterLayer] = []
    for grid in grids:
        paths = _tile_paths_for_grid(grid)
        if not paths:
            continue
        vrt_path, extent = _write_grid_vrt(grid, paths, raster_dir)
        layers.append(
            RasterLayer(
                layer_id=_layer_id(f"{grid}_vexcel_2024"),
                name=f"{grid} Vexcel 2024",
                source=vrt_path,
                extent=extent,
            )
        )
    return layers


def _write_grid_vrt(grid: str, paths: list[Path], raster_dir: Path) -> tuple[Path, tuple[float, float, float, float]]:
    infos = []
    for path in paths:
        with rasterio.open(path) as src:
            infos.append({
                "path": path,
                "width": src.width,
                "height": src.height,
                "bounds": src.bounds,
                "transform": src.transform,
                "dtype": src.dtypes[0],
            })

    xres = float(abs(infos[0]["transform"].a))
    yres = float(abs(infos[0]["transform"].e))
    minx = min(info["bounds"].left for info in infos)
    miny = min(info["bounds"].bottom for info in infos)
    maxx = max(info["bounds"].right for info in infos)
    maxy = max(info["bounds"].top for info in infos)
    width = int(round((maxx - minx) / xres))
    height = int(round((maxy - miny) / yres))
    dtype = _vrt_dtype(str(infos[0]["dtype"]))

    bands = []
    for band_no, color in [(1, "Red"), (2, "Green"), (3, "Blue")]:
        sources = []
        for info in infos:
            b = info["bounds"]
            xoff = int(round((b.left - minx) / xres))
            yoff = int(round((maxy - b.top) / yres))
            rel = html.escape(Path(_relpath(info["path"], raster_dir)).as_posix())
            sources.append(f"""
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{rel}</SourceFilename>
      <SourceBand>{band_no}</SourceBand>
      <SourceProperties RasterXSize="{info['width']}" RasterYSize="{info['height']}" DataType="{dtype}" BlockXSize="512" BlockYSize="512"/>
      <SrcRect xOff="0" yOff="0" xSize="{info['width']}" ySize="{info['height']}"/>
      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{info['width']}" ySize="{info['height']}"/>
    </SimpleSource>""")
        bands.append(f"""
  <VRTRasterBand dataType="{dtype}" band="{band_no}">
    <ColorInterp>{color}</ColorInterp>{''.join(sources)}
  </VRTRasterBand>""")

    vrt = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>EPSG:3857</SRS>
  <GeoTransform>{minx:.12f}, {xres:.12f}, 0.0, {maxy:.12f}, 0.0, {-yres:.12f}</GeoTransform>{''.join(bands)}
</VRTDataset>
"""
    vrt_path = raster_dir / f"{grid}_vexcel_2024.vrt"
    vrt_path.write_text(vrt, encoding="utf-8")
    return vrt_path, (minx, miny, maxx, maxy)


def _relpath(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path.resolve(), start.resolve())


def _vrt_dtype(dtype: str) -> str:
    mapping = {
        "uint8": "Byte",
        "uint16": "UInt16",
        "int16": "Int16",
        "uint32": "UInt32",
        "int32": "Int32",
        "float32": "Float32",
        "float64": "Float64",
    }
    return mapping.get(dtype, "Byte")


def _layer_id(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return f"{safe}_{uuid.uuid4().hex[:12]}"


def _write_styles(out_dir: Path) -> None:
    (out_dir / "reference_polygons.qml").write_text(_reference_qml(), encoding="utf-8")
    (out_dir / "clean_boundary_edit.qml").write_text(_edit_qml(), encoding="utf-8")


def _reference_qml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="source_pool" enableorderby="0">
    <categories>
      <category symbol="0" value="sam_added" label="SAM added" render="true"/>
      <category symbol="1" value="v3c_correct" label="V3C correct" render="true"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" alpha="0.8">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option name="color" value="255,170,0,30" type="QString"/>
            <Option name="outline_color" value="255,128,0,255" type="QString"/>
            <Option name="outline_width" value="0.35" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="0.8">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option name="color" value="168,85,247,25" type="QString"/>
            <Option name="outline_color" value="126,34,206,255" type="QString"/>
            <Option name="outline_width" value="0.35" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""


def _edit_qml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="edit_status" enableorderby="0">
    <categories>
      <category symbol="0" value="pending" label="Pending" render="true"/>
      <category symbol="1" value="done" label="Done" render="true"/>
      <category symbol="2" value="skip" label="Skip" render="true"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" alpha="0.8">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option name="color" value="6,182,212,45" type="QString"/>
            <Option name="outline_color" value="8,145,178,255" type="QString"/>
            <Option name="outline_width" value="0.45" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="0.8">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option name="color" value="34,197,94,55" type="QString"/>
            <Option name="outline_color" value="21,128,61,255" type="QString"/>
            <Option name="outline_width" value="0.45" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="2" alpha="0.8">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option name="color" value="239,68,68,45" type="QString"/>
            <Option name="outline_color" value="185,28,28,255" type="QString"/>
            <Option name="outline_width" value="0.45" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""


def _write_qgis_project(
    out_dir: Path,
    raster_layers: list[RasterLayer],
    gpkg: Path,
    vector_extent: tuple[float, float, float, float],
) -> Path:
    vector_layers = [
        VectorLayer(
            layer_id=_layer_id("clean_boundary_edit"),
            name="clean_boundary_edit",
            source="boundary_refine_workpkg.gpkg|layername=clean_boundary_edit",
            extent=vector_extent,
            geometry="Polygon",
            color="6,182,212,45",
            outline="8,145,178,255",
            read_only=False,
        ),
        VectorLayer(
            layer_id=_layer_id("reference_polygons"),
            name="reference_polygons",
            source="boundary_refine_workpkg.gpkg|layername=reference_polygons",
            extent=vector_extent,
            geometry="Polygon",
            color="255,170,0,25",
            outline="255,128,0,255",
            read_only=True,
        ),
        VectorLayer(
            layer_id=_layer_id("ref_centroids"),
            name="ref_centroids",
            source="boundary_refine_workpkg.gpkg|layername=ref_centroids",
            extent=vector_extent,
            geometry="Point",
            color="255,255,255,180",
            outline="31,41,55,255",
            read_only=True,
        ),
        VectorLayer(
            layer_id=_layer_id("vexcel_tile_index"),
            name="vexcel_tile_index",
            source="boundary_refine_workpkg.gpkg|layername=vexcel_tile_index",
            extent=vector_extent,
            geometry="Polygon",
            color="0,0,0,0",
            outline="107,114,128,180",
            read_only=True,
        ),
    ]

    all_extents = [vector_extent] + [r.extent for r in raster_layers]
    project_extent = (
        min(e[0] for e in all_extents),
        min(e[1] for e in all_extents),
        max(e[2] for e in all_extents),
        max(e[3] for e in all_extents),
    )

    layer_tree = []
    layer_tree.append('    <layer-tree-group name="Boundary refinement" checked="Qt::Checked" expanded="1">')
    for vl in vector_layers:
        checked = "Qt::Checked" if vl.name != "vexcel_tile_index" else "Qt::Unchecked"
        layer_tree.append(_layer_tree_layer(vl.layer_id, vl.name, vl.source, "ogr", checked=checked))
    layer_tree.append("    </layer-tree-group>")
    layer_tree.append('    <layer-tree-group name="Vexcel 2024 basemap" checked="Qt::Checked" expanded="0">')
    for rl in raster_layers:
        source = Path(_relpath(rl.source, out_dir)).as_posix()
        layer_tree.append(_layer_tree_layer(rl.layer_id, rl.name, source, "gdal", checked="Qt::Checked"))
    layer_tree.append("    </layer-tree-group>")

    project_layers = []
    for rl in raster_layers:
        source = Path(_relpath(rl.source, out_dir)).as_posix()
        project_layers.append(_raster_maplayer(rl, source))
    for vl in vector_layers:
        project_layers.append(_vector_maplayer(vl))

    qgs = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname="jhb_phaseA_boundary_refine" version="3.34">
  <homePath path="."/>
  <title>JHB Phase A Boundary Refinement</title>
  <projectCrs>
{_srs_3857(indent=4)}
  </projectCrs>
  <layer-tree-group name="" checked="Qt::Checked" expanded="1">
{chr(10).join(layer_tree)}
  </layer-tree-group>
  <projectlayers>
{chr(10).join(project_layers)}
  </projectlayers>
  <mapcanvas name="theMapCanvas">
    <extent>
      <xmin>{project_extent[0]:.8f}</xmin>
      <ymin>{project_extent[1]:.8f}</ymin>
      <xmax>{project_extent[2]:.8f}</xmax>
      <ymax>{project_extent[3]:.8f}</ymax>
    </extent>
    <rotation>0</rotation>
    <destinationsrs>
{_srs_3857(indent=6)}
    </destinationsrs>
  </mapcanvas>
</qgis>
"""
    qgs_path = out_dir / "boundary_refine.qgs"
    qgs_path.write_text(qgs, encoding="utf-8")
    return qgs_path


def _layer_tree_layer(layer_id: str, name: str, source: str, provider: str, *, checked: str) -> str:
    return (
        f'      <layer-tree-layer id="{html.escape(layer_id)}" name="{html.escape(name)}" '
        f'source="{html.escape(source)}" providerKey="{provider}" checked="{checked}" '
        'expanded="1" legend_split_behavior="0" patch_size="-1,-1">'
        "<customproperties><Option/></customproperties></layer-tree-layer>"
    )


def _raster_maplayer(layer: RasterLayer, source: str) -> str:
    xmin, ymin, xmax, ymax = layer.extent
    return f"""    <maplayer type="raster" hasScaleBasedVisibilityFlag="0" minScale="100000000" maxScale="0">
      <extent><xmin>{xmin:.8f}</xmin><ymin>{ymin:.8f}</ymin><xmax>{xmax:.8f}</xmax><ymax>{ymax:.8f}</ymax></extent>
      <id>{html.escape(layer.layer_id)}</id>
      <datasource>{html.escape(source)}</datasource>
      <layername>{html.escape(layer.name)}</layername>
      <srs>
{_srs_3857(indent=8)}
      </srs>
      <provider>gdal</provider>
      <pipe>
        <rasterrenderer opacity="1" type="multibandcolor" redBand="1" greenBand="2" blueBand="3" alphaBand="-1">
          <rasterTransparency/>
        </rasterrenderer>
        <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
        <huesaturation colorizeOn="0" grayscaleMode="0" saturation="0"/>
      </pipe>
      <blendMode>0</blendMode>
    </maplayer>"""


def _vector_maplayer(layer: VectorLayer) -> str:
    xmin, ymin, xmax, ymax = layer.extent
    readonly = "1" if layer.read_only else "0"
    wkb = "Point" if layer.geometry == "Point" else "MultiPolygon"
    renderer = _point_renderer(layer) if layer.geometry == "Point" else _polygon_renderer(layer)
    return f"""    <maplayer type="vector" geometry="{layer.geometry}" wkbType="{wkb}" readOnly="{readonly}" hasScaleBasedVisibilityFlag="0" minScale="100000000" maxScale="0">
      <extent><xmin>{xmin:.8f}</xmin><ymin>{ymin:.8f}</ymin><xmax>{xmax:.8f}</xmax><ymax>{ymax:.8f}</ymax></extent>
      <id>{html.escape(layer.layer_id)}</id>
      <datasource>{html.escape(layer.source)}</datasource>
      <layername>{html.escape(layer.name)}</layername>
      <srs>
{_srs_3857(indent=8)}
      </srs>
      <provider encoding="UTF-8">ogr</provider>
{renderer}
    </maplayer>"""


def _polygon_renderer(layer: VectorLayer) -> str:
    return f"""      <renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0">
        <symbols>
          <symbol type="fill" name="0" alpha="1" clip_to_extent="1">
            <layer class="SimpleFill" enabled="1" locked="0" pass="0">
              <Option type="Map">
                <Option name="color" value="{layer.color}" type="QString"/>
                <Option name="outline_color" value="{layer.outline}" type="QString"/>
                <Option name="outline_width" value="0.45" type="QString"/>
                <Option name="outline_width_unit" value="MM" type="QString"/>
                <Option name="style" value="solid" type="QString"/>
              </Option>
            </layer>
          </symbol>
        </symbols>
      </renderer-v2>"""


def _point_renderer(layer: VectorLayer) -> str:
    return f"""      <renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0">
        <symbols>
          <symbol type="marker" name="0" alpha="1" clip_to_extent="1">
            <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
              <Option type="Map">
                <Option name="color" value="{layer.color}" type="QString"/>
                <Option name="outline_color" value="{layer.outline}" type="QString"/>
                <Option name="outline_width" value="0.2" type="QString"/>
                <Option name="size" value="1.8" type="QString"/>
                <Option name="name" value="circle" type="QString"/>
              </Option>
            </layer>
          </symbol>
        </symbols>
      </renderer-v2>"""


def _srs_3857(indent: int = 0) -> str:
    sp = " " * indent
    return f"""{sp}<spatialrefsys nativeFormat="Wkt">
{sp}  <wkt>PROJCRS["WGS 84 / Pseudo-Mercator",BASEGEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]]],CONVERSION["Popular Visualisation Pseudo-Mercator",METHOD["Popular Visualisation Pseudo Mercator"]],CS[Cartesian,2],AXIS["easting (X)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["northing (Y)",north,ORDER[2],LENGTHUNIT["metre",1]],ID["EPSG",3857]]</wkt>
{sp}  <proj4>+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs</proj4>
{sp}  <srsid>3857</srsid>
{sp}  <srid>3857</srid>
{sp}  <authid>EPSG:3857</authid>
{sp}  <description>WGS 84 / Pseudo-Mercator</description>
{sp}  <projectionacronym>merc</projectionacronym>
{sp}  <ellipsoidacronym>EPSG:7030</ellipsoidacronym>
{sp}  <geographicflag>false</geographicflag>
{sp}</spatialrefsys>"""


def _write_readme(out_dir: Path, selected: gpd.GeoDataFrame) -> None:
    source_counts = selected["source_pool"].value_counts().to_dict()
    focus_counts = selected["focus"].value_counts().sort_index().to_dict()
    lines = [
        "# JHB Phase A boundary refinement QGIS package",
        "",
        "Open `boundary_refine.qgs` in QGIS.",
        "",
        "Primary edit layer: `clean_boundary_edit` in `boundary_refine_workpkg.gpkg`.",
        "Reference-only layer: `reference_polygons`.",
        "Basemap: local Vexcel 2024 VRTs under `rasters/`.",
        "",
        "Workflow:",
        "1. Turn on editing for `clean_boundary_edit`.",
        "2. Reshape each polygon to the visible PV installation footprint.",
        "3. Preserve real stepped/trapezoid structure; remove only roof halo and SAM edge wobble.",
        "4. Set `edit_status=done` when refined, or `skip` if the image is ambiguous.",
        "5. Keep `use_for_training=1` only for polygons you trust as clean mask supervision.",
        "",
        "Counts:",
        f"- selected_total: {len(selected)}",
        f"- by_source: {source_counts}",
        f"- by_focus: {focus_counts}",
        "",
        "Notes:",
        "- Geometry CRS is EPSG:3857 to match the Vexcel raster tiles.",
        "- `area_m2` and wobble fields were computed in EPSG:32735.",
        "- `v3c_halo_probe` is a proxy bucket, not a confirmed error label.",
        "- Do not edit `reference_polygons`; edit `clean_boundary_edit`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--seed", type=int, default=20260508)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    all_candidates = _load_all_candidates()
    selected = _select_candidates(all_candidates, seed=args.seed)
    gpkg = _write_workpkg(selected, all_candidates, args.out)
    raster_layers = _write_vrts(sorted(selected["grid"].unique()), args.out)
    _write_styles(args.out)
    qgs = _write_qgis_project(
        args.out,
        raster_layers,
        gpkg,
        tuple(float(v) for v in selected.total_bounds),
    )
    _write_readme(args.out, selected)

    print(f"[DONE] selected {len(selected)} polygons")
    print(f"[SAVE] {gpkg}")
    print(f"[SAVE] {qgs}")
    print(f"[SAVE] {args.out / 'README.md'}")


if __name__ == "__main__":
    main()
