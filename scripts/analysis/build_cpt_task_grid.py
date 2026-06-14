#!/usr/bin/env python3
"""Build the Cape Town CPT task grid + G->CPT crosswalk (ADR-0002 decision #5).

Takes the legacy 2214-cell Cape Town task grid (Gao ``G\\d{4}`` scheme) plus a
per-cell WMS imagery-coverage probe CSV, keeps every cell whose coverage clears
``--threshold`` (default 0.05), renames the kept cells ``CPT####``
**digit-preserving** (``G1240`` -> ``CPT1240``), and emits:

  1. ``data/task_grid_cpt.gpkg`` (+ ``.geojson`` twin) — kept cells only, in the
     7-city Vexcel column convention (see
     ``data/vexcel_task_grids/pretoria_task_grid.gpkg``) plus one extra
     ``legacy_gao_id`` provenance column. Geometry is the **exact** source cell
     geometry (no re-cut, Z dropped to match the 2D Vexcel grids).
  2. ``data/ct_grid_crosswalk_g_to_cpt.csv`` — ALL 2214 source rows with
     ``g_id, cpt_id, kept, coverage_fraction, reason``. The permanent provenance
     crosswalk required by ADR-0002 section 5.

The build is deterministic: identical inputs produce byte-stable outputs (cells
are processed in sorted source order).

Hard asserts (fail loudly):
  - Every G-prefixed anchor from ``cape_town -> aerial_2025 -> coverage_grids``
    in regions.yaml is kept.
  - Digit-preserving bijection: stripping ``CPT``/``G`` prefixes maps kept rows
    1:1, no duplicate ``gridcell_id``.
  - Per kept cell, output geometry equals the source geometry exactly
    (shapely ``equals_exact`` tolerance 0).
  - kept + dropped == 2214; crosswalk has exactly 2214 rows.

Usage:
  python scripts/analysis/build_cpt_task_grid.py
  python scripts/analysis/build_cpt_task_grid.py --threshold 0.05
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from shapely import force_2d

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_GRID = PROJECT_ROOT / "data" / "task_grid.gpkg"
DEFAULT_PROBE_CSV = (
    PROJECT_ROOT / "results" / "analysis" / "ct_wms_coverage_probe" / "probe.csv"
)
DEFAULT_REGIONS_YAML = PROJECT_ROOT / "configs" / "datasets" / "regions.yaml"
DEFAULT_OUT_GPKG = PROJECT_ROOT / "data" / "task_grid_cpt.gpkg"
DEFAULT_OUT_CROSSWALK = PROJECT_ROOT / "data" / "ct_grid_crosswalk_g_to_cpt.csv"

WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:32734"
REGION_KEY = "cape_town"
CITY = "Cape Town"
PROVINCE = "Western Cape"
# WMS layer that backs the probe (scripts/imagery/download_tiles.py WMS_LAYER).
COLLECTION_ID = "Aerial Imagery_Aerial Imagery 2025Jan"
PRODUCT = "wms_aerial_ortho"
COVERAGE_SOURCE = "ct_wms_blank_probe_2026-06-12"
# Layer is "Aerial Imagery 2025Jan"; no finer capture dates are published.
CAPTURE_START = "2025-01-01"
CAPTURE_END = "2025-01-31"
# Native GSD is not declared by the WMS server or download_tiles.py; leave null
# rather than fabricate a value (the Vexcel schema allows null avg_gsd_m).
AVG_GSD_M: float | None = None
GRID_SIZE_M = 1000
GRID_ID_RE = re.compile(r"^G(\d{4})$")
FULLISH_THRESHOLD = 0.999  # is_edge = coverage_fraction < this (Vexcel convention)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def g_to_cpt(g_id: str) -> str:
    """Digit-preserving rename: ``G1240`` -> ``CPT1240``."""
    m = GRID_ID_RE.match(g_id)
    if not m:
        raise ValueError(f"source grid id is not a 4-digit G-ID: {g_id!r}")
    return f"CPT{m.group(1)}"


def load_g_anchors(regions_yaml: Path) -> list[str]:
    """G-prefixed coverage_grids of cape_town.aerial_2025 (Li L-IDs excluded)."""
    with regions_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    coverage = (
        cfg["regions"][REGION_KEY]["imagery_layers"]["aerial_2025"]["coverage_grids"]
    )
    return sorted(str(x) for x in coverage if str(x).startswith("G"))


def build(
    *,
    source_grid: Path,
    probe_csv: Path,
    regions_yaml: Path,
    threshold: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, list[str]]:
    src = gpd.read_file(source_grid)
    if src.crs is None or src.crs.to_string() != WGS84:
        src = src.to_crs(WGS84)
    # Deterministic source ordering.
    src = src.sort_values("gridcell_id").reset_index(drop=True)

    probe = pd.read_csv(probe_csv)
    cf_by_id = dict(zip(probe["gridcell_id"].astype(str), probe["coverage_fraction"]))
    status_by_id = dict(zip(probe["gridcell_id"].astype(str), probe["status"]))

    anchors = load_g_anchors(regions_yaml)

    # Precompute metric reprojection once (shared transform) for centroid_x/y.
    src_metric = src.to_crs(CRS_METRIC)

    crosswalk_rows: list[dict[str, Any]] = []
    kept_records: list[dict[str, Any]] = []

    for i, row in src.iterrows():
        g_id = str(row["gridcell_id"])
        src_geom = row.geometry
        geom_2d = force_2d(src_geom)

        cf = cf_by_id.get(g_id)
        status = status_by_id.get(g_id)

        if cf is None:
            # Cell missing from probe — defensive; should never happen.
            crosswalk_rows.append(
                {
                    "g_id": g_id,
                    "cpt_id": "",
                    "kept": False,
                    "coverage_fraction": "",
                    "reason": "probe_missing",
                }
            )
            continue

        cf = float(cf)
        keep = cf >= threshold
        if keep:
            cpt_id = g_to_cpt(g_id)
            reason = "covered"
        else:
            cpt_id = ""
            reason = "no_imagery" if cf == 0.0 else "below_threshold"

        crosswalk_rows.append(
            {
                "g_id": g_id,
                "cpt_id": cpt_id,
                "kept": keep,
                "coverage_fraction": cf,
                "reason": reason,
            }
        )

        if not keep:
            continue

        metric_geom = src_metric.geometry.iloc[i]
        centroid_m = metric_geom.centroid
        centroid_wgs = geom_2d.centroid  # EPSG:4326 centroid for lon/lat
        area_sqm = float(metric_geom.area)
        minx, miny, maxx, maxy = geom_2d.bounds

        kept_records.append(
            {
                "gridcell_id": cpt_id,
                "Name": cpt_id,
                "region_key": REGION_KEY,
                "city": CITY,
                "province": PROVINCE,
                "collection_id": COLLECTION_ID,
                "product": PRODUCT,
                "capture_start": CAPTURE_START,
                "capture_end": CAPTURE_END,
                "avg_gsd_m": AVG_GSD_M,
                "coverage_area_km2": None,
                "source_bbox": f"{minx},{miny},{maxx},{maxy}",
                "coverage_source": COVERAGE_SOURCE,
                "crs_metric": CRS_METRIC,
                "grid_size_m": GRID_SIZE_M,
                "area_sqm": area_sqm,
                "coverage_fraction": cf,
                "is_edge": cf < FULLISH_THRESHOLD,
                "centroid_x": float(centroid_m.x),
                "centroid_y": float(centroid_m.y),
                "lon": float(centroid_wgs.x),
                "lat": float(centroid_wgs.y),
                "legacy_gao_id": g_id,
                "geometry": geom_2d,
            }
        )

    column_order = [
        "gridcell_id",
        "Name",
        "region_key",
        "city",
        "province",
        "collection_id",
        "product",
        "capture_start",
        "capture_end",
        "avg_gsd_m",
        "coverage_area_km2",
        "source_bbox",
        "coverage_source",
        "crs_metric",
        "grid_size_m",
        "area_sqm",
        "coverage_fraction",
        "is_edge",
        "centroid_x",
        "centroid_y",
        "lon",
        "lat",
        "legacy_gao_id",
        "geometry",
    ]
    cpt = gpd.GeoDataFrame(kept_records, columns=column_order, crs=WGS84)
    crosswalk = pd.DataFrame(
        crosswalk_rows,
        columns=["g_id", "cpt_id", "kept", "coverage_fraction", "reason"],
    )
    return cpt, crosswalk, anchors


def run_asserts(
    cpt: gpd.GeoDataFrame,
    crosswalk: pd.DataFrame,
    anchors: list[str],
    src: gpd.GeoDataFrame,
    *,
    n_source: int,
) -> None:
    # 1. Row-count invariant.
    n_kept = int(crosswalk["kept"].sum())
    n_dropped = int((~crosswalk["kept"]).sum())
    assert n_kept + n_dropped == n_source, (
        f"kept ({n_kept}) + dropped ({n_dropped}) != source ({n_source})"
    )
    assert len(crosswalk) == n_source, (
        f"crosswalk rows ({len(crosswalk)}) != source ({n_source})"
    )
    assert len(cpt) == n_kept, f"cpt rows ({len(cpt)}) != kept ({n_kept})"

    # 2. Every anchor kept.
    kept_g = set(crosswalk.loc[crosswalk["kept"], "g_id"])
    missing_anchors = [a for a in anchors if a not in kept_g]
    assert not missing_anchors, f"anchors dropped: {missing_anchors}"

    # 3. Digit-preserving bijection + no duplicate gridcell_id.
    assert cpt["gridcell_id"].is_unique, "duplicate CPT gridcell_id"
    cpt_digits = {gid[3:] for gid in cpt["gridcell_id"]}
    g_digits = {gid[1:] for gid in kept_g}
    assert cpt_digits == g_digits, "digit-preserving bijection broken (CPT<->G)"
    assert len(cpt_digits) == n_kept, "non-injective digit mapping"
    # Every CPT row's legacy_gao_id maps back digit-for-digit.
    for _, r in cpt.iterrows():
        assert r["gridcell_id"] == g_to_cpt(r["legacy_gao_id"]), (
            f"row {r['gridcell_id']} != g_to_cpt({r['legacy_gao_id']})"
        )

    # 4. Geometry identity (output == source, exact, after Z drop).
    src_geom_by_id = dict(zip(src["gridcell_id"].astype(str), src.geometry))
    for _, r in cpt.iterrows():
        src_geom = force_2d(src_geom_by_id[r["legacy_gao_id"]])
        assert r.geometry.equals_exact(src_geom, tolerance=0.0), (
            f"geometry mismatch for {r['legacy_gao_id']} -> {r['gridcell_id']}"
        )


def write_outputs(
    cpt: gpd.GeoDataFrame,
    crosswalk: pd.DataFrame,
    *,
    out_gpkg: Path,
    out_crosswalk: Path,
    write_geojson: bool,
) -> Path | None:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    out_crosswalk.parent.mkdir(parents=True, exist_ok=True)

    if out_gpkg.exists():
        out_gpkg.unlink()
    cpt.to_file(out_gpkg, driver="GPKG")

    geojson_path: Path | None = None
    if write_geojson:
        geojson_path = out_gpkg.with_suffix(".geojson")
        if geojson_path.exists():
            geojson_path.unlink()
        cpt.to_file(geojson_path, driver="GeoJSON")

    crosswalk.to_csv(out_crosswalk, index=False)
    return geojson_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-grid", type=Path, default=DEFAULT_SOURCE_GRID)
    parser.add_argument("--probe-csv", type=Path, default=DEFAULT_PROBE_CSV)
    parser.add_argument("--regions-yaml", type=Path, default=DEFAULT_REGIONS_YAML)
    parser.add_argument("--out-gpkg", type=Path, default=DEFAULT_OUT_GPKG)
    parser.add_argument("--out-crosswalk", type=Path, default=DEFAULT_OUT_CROSSWALK)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Keep cells with coverage_fraction >= this (default 0.05).",
    )
    parser.add_argument(
        "--no-geojson",
        action="store_true",
        help="Skip the .geojson twin (Vexcel grids ship one by default).",
    )
    args = parser.parse_args()

    src = gpd.read_file(args.source_grid)
    if src.crs is None or src.crs.to_string() != WGS84:
        src = src.to_crs(WGS84)
    src = src.sort_values("gridcell_id").reset_index(drop=True)
    n_source = len(src)

    cpt, crosswalk, anchors = build(
        source_grid=args.source_grid,
        probe_csv=args.probe_csv,
        regions_yaml=args.regions_yaml,
        threshold=args.threshold,
    )

    run_asserts(cpt, crosswalk, anchors, src, n_source=n_source)

    geojson_path = write_outputs(
        cpt,
        crosswalk,
        out_gpkg=args.out_gpkg,
        out_crosswalk=args.out_crosswalk,
        write_geojson=not args.no_geojson,
    )

    n_kept = int(crosswalk["kept"].sum())
    n_dropped = int((~crosswalk["kept"]).sum())
    print(f"threshold:         {args.threshold}")
    print(f"source cells:      {n_source}")
    print(f"kept (CPT):        {n_kept}")
    print(f"dropped:           {n_dropped}")
    print(f"anchors kept:      {len(anchors)}/{len(anchors)} (all)")
    print(f"gpkg:              {display_path(args.out_gpkg)}")
    if geojson_path is not None:
        print(f"geojson:           {display_path(geojson_path)}")
    print(f"crosswalk:         {display_path(args.out_crosswalk)}")
    print("asserts:           PASS")


if __name__ == "__main__":
    main()
