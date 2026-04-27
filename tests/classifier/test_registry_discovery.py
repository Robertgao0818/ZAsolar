"""Registry-driven discovery coverage test for build_cls_dataset.

Verifies that `discover_grid_sources()` picks up all three expected source
buckets — CT batch003 (legacy flat), CT batch004 (aerial_2025), JHB Sandton
(v4_aerial_2023) — on the current working tree.

This test is observational: it reads live `results/` state. Run it after
any `regions.yaml` or results-layout change to catch silent regressions in
discovery coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.classifier import build_cls_dataset as builder  # noqa: E402


EXPECTED_BUCKETS = {
    "cape_town:legacy_flat_batch003",
    "cape_town:v3c_targeted_hn_aerial_2025",
    "johannesburg:v4_aerial_2023",
}


@pytest.fixture(scope="module")
def sources():
    return builder.discover_grid_sources()


def test_three_expected_source_buckets_present(sources):
    found = {s.source_bucket for s in sources}
    missing = EXPECTED_BUCKETS - found
    assert not missing, (
        f"Registry discovery missing buckets: {missing}. "
        f"Found: {sorted(found)}"
    )


def test_deprecated_model_runs_excluded_by_default(sources):
    # v3c_geid_2024_02 is marked deprecated: true in regions.yaml
    deprecated_hits = [s for s in sources if s.deprecated]
    assert not deprecated_hits, (
        f"Deprecated model_runs should be excluded by default: "
        f"{[s.source_bucket for s in deprecated_hits]}"
    )


def test_each_source_has_review_data(sources):
    for src in sources:
        assert src.reviewed_gpkg is not None or src.review_csv is not None, (
            f"{src.source_bucket}/{src.grid_id} has neither gpkg nor csv"
        )


def test_grid_count_reasonable(sources):
    """Sanity check: we expect on the order of ~100 grids with review data."""
    assert len(sources) >= 80, f"Only {len(sources)} grid sources discovered"
    assert len(sources) < 500, f"{len(sources)} is suspiciously high"


def test_bucket_grid_counts_within_plan(sources):
    """Plan's 已核实的关键事实 lists approximate counts per bucket; fail loudly
    if any bucket drops below a conservative floor."""
    floors = {
        "cape_town:legacy_flat_batch003": 15,        # plan: 21
        "cape_town:v3c_targeted_hn_aerial_2025": 30,  # plan: 36
        "johannesburg:v4_aerial_2023": 40,            # plan: 50
    }
    counts = {b: 0 for b in floors}
    for s in sources:
        if s.source_bucket in counts:
            counts[s.source_bucket] += 1
    for bucket, floor in floors.items():
        assert counts[bucket] >= floor, (
            f"{bucket} has only {counts[bucket]} grids; expected at least {floor}"
        )
