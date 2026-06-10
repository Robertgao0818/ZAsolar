"""ImageryLayerConfig.census_imagery_mid_date wiring test.

Verifies the field declared in regions.yaml is exposed via the typed registry
API. Subrepo (solar_backdating) reads this through ImageryLayerConfig, so a
silent drop at the dataclass boundary would break Task G consumers.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.region_registry import (  # noqa: E402
    ImageryLayerConfig,
    get_imagery_layer,
    get_region_config,
)


def test_imagery_layer_config_has_census_field_default_none() -> None:
    layer = ImageryLayerConfig(
        region_key="x",
        layer_id="y",
        path="p",
        source="aerial",
        vintage="2024",
        file_layout="chunked",
        file_pattern=None,
        crs="EPSG:4326",
        coverage_grids=(),
    )
    assert layer.census_imagery_mid_date is None


def test_jhb_geid_2024_02_exposes_census_mid_date() -> None:
    layer = get_imagery_layer("johannesburg", "geid_2024_02")
    assert layer.census_imagery_mid_date == "2024-02-15"


def test_jhb_vexcel_2024_exposes_census_mid_date() -> None:
    layer = get_imagery_layer("johannesburg", "vexcel_2024")
    # Corrected 2026-06-04 from the nominal 2024-06-30 to the true flight-window
    # midpoint (2024-02-17..2024-04-19) after /ortho/dates verification. This is a
    # fallback-only aggregate; per-grid present-side clamping uses real flight dates.
    assert layer.census_imagery_mid_date == "2024-03-18"


def test_jhb_aerial_2023_exposes_census_mid_date() -> None:
    layer = get_imagery_layer("johannesburg", "aerial_2023")
    assert layer.census_imagery_mid_date == "2023-06-30"


def test_ct_aerial_2025_exposes_census_mid_date() -> None:
    layer = get_imagery_layer("cape_town", "aerial_2025")
    assert layer.census_imagery_mid_date == "2025-06-30"


def test_layers_without_field_are_none() -> None:
    """Vexcel placeholder regions added later don't yet declare the field; must be None, not raise."""
    for region_key in ("durban", "pretoria", "stellenbosch", "george", "port_elizabeth"):
        try:
            config = get_region_config(region_key)
        except KeyError:
            continue
        for layer_id, layer in config.imagery_layers.items():
            assert layer.census_imagery_mid_date is None or isinstance(layer.census_imagery_mid_date, str), (
                f"{region_key}/{layer_id} census_imagery_mid_date must be str|None"
            )
