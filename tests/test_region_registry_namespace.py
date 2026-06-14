"""ADR-0002 grid-namespace tests: retired namespaces + two-tier lookup.

JHB's canonical grid scheme is JNB (Vexcel-382, 2026-06-03 decision); its
Gxxxx / JHBnn namespaces are retired. CT's canonical census scheme is CPT
(2026-06-12 regrid, ADR-0002 decision #5); its G\\d{4} namespace is likewise
retired. Both stay resolvable for historical artifacts but are excluded from
the active lookup tier — so a bare CT/JHB G-overlap ID (G1189 etc.) now has
NO active owner and the retired-tier fallback returns BOTH regions (this is
honest ambiguity: post-regrid neither G\\d{4} namespace is canonical).
Singular lookup_region() still picks the first registry hit (cape_town),
preserving backward compatibility.
"""

from __future__ import annotations

import pytest

from core import region_registry

# rule 06-multi-city: these G-IDs exist in BOTH CT and JHB task grids but
# cover different physical areas. Post CT regrid (2026-06-12) G\d{4} is
# retired in both regions, so these resolve to both via the retired tier.
OVERLAP_GIDS = ["G1189", "G1190", "G1293", "G1513", "G1570", "G1630"]


def test_retired_patterns_loaded():
    jhb = region_registry.get_region_config("johannesburg")
    assert jhb.retired_grid_id_patterns == ("G\\d{4}", "JHB\\d{2}")
    # CT regrid (ADR-0002 #5; full-metro decision A, 2026-06-14): G\d{4} AND
    # L\d{4} retired, CPT\d{4} the sole active CT census namespace (CPT统一).
    ct = region_registry.get_region_config("cape_town")
    assert ct.retired_grid_id_patterns == ("G\\d{4}", "L\\d{4}")


@pytest.mark.parametrize("gid", OVERLAP_GIDS)
def test_overlap_gid_resolves_to_both_via_retired_tier(gid):
    # Post CT regrid neither region holds G\d{4} as active, so the overlap
    # IDs have no active owner and fall back to the retired tier in both.
    assert region_registry.lookup_regions(gid) == ["cape_town", "johannesburg"]


def test_overlap_gid_include_retired_returns_both():
    hits = region_registry.lookup_regions("G1189", include_retired=True)
    assert hits == ["cape_town", "johannesburg"]


def test_jhb_only_gid_falls_back_to_retired_namespace():
    # G0816 (JHB CBD 25-grid eval suite) exists only in JHB: no active
    # region claims it, so the retired-tier fallback must keep it resolvable.
    assert region_registry.lookup_regions("G0816") == ["johannesburg"]
    assert region_registry.lookup_region("G0816") == "johannesburg"


def test_jhb_legacy_jhbnn_falls_back_to_retired_namespace():
    assert region_registry.lookup_regions("JHB01") == ["johannesburg"]


def test_jnb_canonical_is_active():
    assert region_registry.lookup_regions("JNB0001") == ["johannesburg"]


def test_li_lprefix_resolves_to_cape_town_via_retired_tier():
    # Full-metro CPT统一 (decision A, 2026-06-14): L\d{4} is now RETIRED (folded
    # into CPT via data/ct_grid_crosswalk_l_to_cpt.csv). It stays resolvable —
    # no active region claims it, so the retired-tier fallback returns cape_town.
    assert region_registry.lookup_regions("L1787") == ["cape_town"]
    assert region_registry.lookup_region("L1787") == "cape_town"


# --- CPT canonical census namespace (ADR-0002 decision #5, 2026-06-12) -------

def test_cpt_namespace_is_active_cape_town():
    """The CPT census grid is the ACTIVE cape_town namespace post-regrid."""
    assert region_registry.lookup_regions("CPT1240") == ["cape_town"]
    assert region_registry.lookup_region("CPT1240") == "cape_town"
    # A dropped (ocean) source G-cell has no CPT counterpart; an arbitrary CPT
    # ID still resolves only to cape_town via the active CPT task grid.
    assert region_registry.lookup_regions("CPT0288") == ["cape_town"]


def test_cpt_pattern_in_grid_id_pattern():
    ct = region_registry.get_region_config("cape_town")
    assert "CPT" in ct.grid_id_pattern
    # grid_id_pattern stays the FULL resolvable set (ADR-0002): CPT (active) +
    # G (retired) + L (retired, folded into CPT — decision A 2026-06-14).
    import re

    assert re.fullmatch(ct.grid_id_pattern, "CPT1240")
    assert re.fullmatch(ct.grid_id_pattern, "G1240")
    assert re.fullmatch(ct.grid_id_pattern, "L1787")


def test_ct_g_id_resolves_via_gao_scheme_fallback():
    """A CT-only legacy G-ID (no JHB overlap, e.g. dropped/kept CT cell) still
    resolves to cape_town via the gao annotation scheme task grid even though
    the region's primary task grid is now the CPT census grid."""
    # G1240 is kept in CT (CPT1240) and is not in JHB's coverage -> CT only.
    assert region_registry.lookup_regions("G1240") == ["cape_town"]
    assert region_registry.lookup_region("G1240") == "cape_town"


def test_unknown_grid_resolves_nowhere():
    assert region_registry.lookup_regions("ZZZ9999") == []
    assert region_registry.lookup_region("ZZZ9999") is None


def test_lookup_region_singular_unchanged_for_overlap_ids():
    # Pre-ADR-0002, lookup_region() returned the arbitrary first hit, which
    # was already cape_town. The two-tier change must not flip it.
    assert region_registry.lookup_region("G1189") == "cape_town"


def _active_grid_ids(key: str) -> set[str]:
    config = region_registry.get_region_config(key)
    ids: set[str] = set()
    for layer in config.imagery_layers.values():
        ids.update(layer.coverage_grids)
    ids.update(config.grids)
    ids.update(region_registry._task_grid_ids(key))
    ids.update(region_registry._scheme_task_grid_ids(key))
    return {g for g in ids if not region_registry._is_retired_grid(key, g)}


def test_active_namespaces_pairwise_disjoint():
    """ADR-0002 invariant: no grid ID may live in two ACTIVE namespaces.

    Enumerates every registered ID per region (coverage_grids, grids
    section, task grids), drops retired-namespace IDs, and asserts the
    active sets never intersect. Adding a region/scheme that collides must
    fail here, not silently mis-resolve at runtime.
    """
    regions = region_registry.list_regions()
    active = {key: _active_grid_ids(key) for key in regions}
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            clash = active[a] & active[b]
            assert not clash, (
                f"Active grid namespaces of '{a}' and '{b}' overlap: "
                f"{sorted(clash)[:10]} — retire one side in regions.yaml"
            )


def test_multiple_active_hits_warn(monkeypatch):
    fake = {
        "fake_region_x": region_registry.RegionConfig(
            key="fake_region_x",
            description="",
            crs_metric="EPSG:32734",
            crs_exchange="EPSG:4326",
            paths=region_registry.RegionPaths("t", "r", "a", "missing.gpkg"),
            grid_id_pattern="X\\d{4}",
            grids={"X0001": {}},
        ),
        "fake_region_y": region_registry.RegionConfig(
            key="fake_region_y",
            description="",
            crs_metric="EPSG:32735",
            crs_exchange="EPSG:4326",
            paths=region_registry.RegionPaths("t", "r", "a", "missing.gpkg"),
            grid_id_pattern="X\\d{4}",
            grids={"X0001": {}},
        ),
    }
    monkeypatch.setattr(region_registry, "_get_registry", lambda: fake)
    with pytest.warns(UserWarning, match="multiple ACTIVE namespaces"):
        hits = region_registry.lookup_regions("X0001")
    assert hits == ["fake_region_x", "fake_region_y"]


# ---------------------------------------------------------------------------
# grid_utils geometry resolution (TRAP A — CPT regrid fall-through)
# ---------------------------------------------------------------------------
# After CT's primary task grid became the CPT census grid (no G-cells), bare
# G-ID geometry resolution must fall through to the gao annotation scheme task
# grid (data/task_grid.gpkg). For the 6 G-IDs that overlap JHB this must still
# return the CAPE TOWN cell (lon < 18.7), never JHB's same-named cell (lon ~28).

from core import grid_utils  # noqa: E402


def test_get_grid_spec_cpt_via_primary_task_grid():
    spec = grid_utils.get_grid_spec("CPT1240")
    # Cape Town longitudes are ~18.x; this exercises the active CPT task grid.
    assert 18.0 < spec.xmin < 18.7
    assert spec.xmax > spec.xmin


def test_get_grid_spec_g_id_via_gao_scheme_fallback():
    """get_grid_spec('G1240') must still work post-regrid: the gao scheme's
    data/task_grid.gpkg carries the retired G-cell geometry, and it is the
    DIGIT-PRESERVING source of CPT1240 — so both resolve to the same cell."""
    g_spec = grid_utils.get_grid_spec("G1240")
    cpt_spec = grid_utils.get_grid_spec("CPT1240")
    assert g_spec.xmin == pytest.approx(cpt_spec.xmin)
    assert g_spec.ymin == pytest.approx(cpt_spec.ymin)
    assert g_spec.xmax == pytest.approx(cpt_spec.xmax)
    assert g_spec.ymax == pytest.approx(cpt_spec.ymax)


@pytest.mark.parametrize("gid", OVERLAP_GIDS)
def test_trap_a_bare_overlap_gid_resolves_to_cape_town_cell(gid):
    """Regression for TRAP A: a BARE overlapping G-ID (no region passed) must
    resolve to the Cape Town cell (lon < 18.7), not JHB's (lon ~28)."""
    rec = grid_utils.get_grid_record(gid)
    lon = float(rec.geometry.centroid.x)
    assert lon < 18.7, f"{gid} mis-resolved to a non-CT cell at lon={lon}"
    # And the metric CRS must be CT's UTM 34S, not JHB's 35S.
    assert grid_utils.get_metric_crs(gid) == "EPSG:32734"


@pytest.mark.parametrize("gid", OVERLAP_GIDS)
def test_explicit_jhb_region_still_reaches_jhb_cell(gid):
    """The fix must not break explicit JHB resolution: region='jhb' still gets
    the Johannesburg cell (lon ~28) for the overlapping IDs."""
    rec = grid_utils.get_grid_record(gid, region="jhb")
    lon = float(rec.geometry.centroid.x)
    assert lon > 20.0, f"{gid} (region=jhb) mis-resolved to lon={lon}"
    assert grid_utils.get_metric_crs(gid, region="jhb") == "EPSG:32735"
