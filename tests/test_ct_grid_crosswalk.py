"""CT regrid crosswalk integrity (ADR-0002 decision #5, 2026-06-12).

The Cape Town census grid was renumbered G\\d{4} -> CPT\\d{4}, DIGIT-PRESERVING
(G1240 -> CPT1240), dropping ocean cells with no City-of-Cape-Town WMS aerial
coverage. These tests pin the crosswalk + new task grid against the source grid:

  - data/ct_grid_crosswalk_g_to_cpt.csv  (2214 rows, kept flag + reason)
  - data/task_grid_cpt.gpkg              (1103 kept CPT cells)
  - data/task_grid.gpkg                  (2214 source Gao G cells, geometry source)

Geometry is NOT re-cut: each CPT cell is the identical source G cell.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from core import region_registry

BASE_DIR = Path(__file__).resolve().parent.parent
CROSSWALK = BASE_DIR / "data" / "ct_grid_crosswalk_g_to_cpt.csv"
CPT_GRID = BASE_DIR / "data" / "task_grid_cpt.gpkg"
SOURCE_GRID = BASE_DIR / "data" / "task_grid.gpkg"

SOURCE_CELL_COUNT = 2214
KEPT_CELL_COUNT = 1103
DROPPED_CELL_COUNT = SOURCE_CELL_COUNT - KEPT_CELL_COUNT  # 1111


@pytest.fixture(scope="module")
def crosswalk() -> pd.DataFrame:
    if not CROSSWALK.exists():
        pytest.skip(f"crosswalk not built: {CROSSWALK}")
    return pd.read_csv(CROSSWALK)


@pytest.fixture(scope="module")
def cpt_grid() -> gpd.GeoDataFrame:
    if not CPT_GRID.exists():
        pytest.skip(f"CPT task grid not built: {CPT_GRID}")
    return gpd.read_file(CPT_GRID)


def test_crosswalk_row_count(crosswalk):
    assert len(crosswalk) == SOURCE_CELL_COUNT


def test_crosswalk_kept_dropped_split(crosswalk):
    kept = crosswalk[crosswalk["kept"] == True]  # noqa: E712
    dropped = crosswalk[crosswalk["kept"] == False]  # noqa: E712
    assert len(kept) == KEPT_CELL_COUNT
    assert len(dropped) == DROPPED_CELL_COUNT
    # Kept rows carry a CPT id; dropped rows do not.
    assert kept["cpt_id"].notna().all()
    assert dropped["cpt_id"].isna().all()


def test_crosswalk_digit_preserving(crosswalk):
    """Every kept row maps G#### -> CPT#### preserving the 4-digit number."""
    kept = crosswalk[crosswalk["kept"] == True].copy()  # noqa: E712
    g_digits = kept["g_id"].str.removeprefix("G")
    cpt_digits = kept["cpt_id"].astype(str).str.removeprefix("CPT")
    assert (g_digits == cpt_digits).all()
    # G/CPT ids well-formed.
    assert kept["g_id"].str.fullmatch(r"G\d{4}").all()
    assert kept["cpt_id"].str.fullmatch(r"CPT\d{4}").all()


def test_crosswalk_cpt_ids_unique(crosswalk):
    kept = crosswalk[crosswalk["kept"] == True]  # noqa: E712
    assert kept["cpt_id"].is_unique
    assert kept["g_id"].is_unique


def test_all_aerial_2025_anchors_kept(crosswalk):
    """Every G-prefixed aerial_2025 coverage anchor MUST survive the regrid —
    these are the grids with downloaded imagery / GT / training provenance."""
    ct = region_registry.get_region_config("cape_town")
    anchors = sorted(
        {
            g
            for layer in ct.imagery_layers.values()
            for g in layer.coverage_grids
            if g.startswith("G")
        }
    )
    assert anchors, "expected G-prefixed anchors in aerial_2025 coverage_grids"
    kept_g = set(crosswalk[crosswalk["kept"] == True]["g_id"])  # noqa: E712
    missing = [a for a in anchors if a not in kept_g]
    assert not missing, f"aerial_2025 anchors dropped by regrid: {missing}"


def test_cpt_grid_cell_count_and_namespace(cpt_grid):
    assert len(cpt_grid) == KEPT_CELL_COUNT
    ids = cpt_grid["gridcell_id"].astype(str)
    assert ids.str.fullmatch(r"CPT\d{4}").all()
    assert ids.is_unique


def test_cpt_grid_matches_crosswalk_kept_set(cpt_grid, crosswalk):
    kept_cpt = set(crosswalk[crosswalk["kept"] == True]["cpt_id"])  # noqa: E712
    grid_cpt = set(cpt_grid["gridcell_id"].astype(str))
    assert kept_cpt == grid_cpt


def test_cpt_geometry_equals_source_g_cell(cpt_grid):
    """CPT cell geometry == its source G cell geometry (no re-cut)."""
    if not SOURCE_GRID.exists():
        pytest.skip(f"source grid not present: {SOURCE_GRID}")
    src = gpd.read_file(SOURCE_GRID)
    src_by_g = {str(r.gridcell_id): r.geometry for r in src.itertuples()}
    # Use the back-reference column carried by the CPT grid.
    legacy_col = next(
        (c for c in ("legacy_gao_id", "g_id_source") if c in cpt_grid.columns),
        None,
    )
    assert legacy_col is not None, "CPT grid must carry a legacy G-ID back-reference"
    # Sample across the id range (cheap + deterministic).
    sample = cpt_grid.iloc[:: max(1, len(cpt_grid) // 20)]
    checked = 0
    for row in sample.itertuples():
        gid = str(getattr(row, legacy_col))
        assert gid in src_by_g, f"{row.gridcell_id} back-ref {gid} missing from source grid"
        assert row.geometry.equals(src_by_g[gid]), (
            f"{row.gridcell_id} geometry differs from source {gid}"
        )
        checked += 1
    assert checked >= 10
