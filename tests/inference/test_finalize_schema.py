"""End-to-end finalize smoke test: synthetic raw artifact → predictions_metric.gpkg.

Verifies the V1.4 minimum gpkg schema contract (every required column present;
metric CRS; geojson is EPSG:4326). Does NOT test detector forward — just the
finalize half of the pipeline.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from core.inference.raw_artifact import (
    Chip,
    Detection,
    PIPELINE_VERSION,
    RawArtifact,
    SCHEMA_VERSION,
    SourceTile,
    utc_now_iso,
    write_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_synthetic_jhb_tif(path: Path, width: int = 400, height: int = 400) -> dict:
    """Synthetic JHB-CRS GeoTIFF; returns metadata dict.

    Uses EPSG:32735 (JHB metric) as both source and metric CRS to keep the
    test simple (no reprojection-during-zonal-stats edge cases).
    """
    arr = np.full((3, height, width), 100, dtype=np.uint8)
    transform = from_origin(500_000.0, 7_200_000.0, 0.5, 0.5)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=3, dtype="uint8",
        crs="EPSG:32735", transform=transform,
    ) as dst:
        dst.write(arr)
    return {
        "path": str(path), "size_bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "crs": "EPSG:32735",
        "transform": tuple(transform)[:6],
        "bounds": (500_000.0, 7_200_000.0 - 0.5 * height, 500_000.0 + 0.5 * width, 7_200_000.0),
        "shape": (height, width),
    }


def _make_synthetic_artifact(tif_meta: dict) -> RawArtifact:
    """Build a single-chip artifact with one synthetic detection."""
    # Make a 50×50 mask block in the middle of the chip → ~25 m² panel
    # at 0.5 m GSD (50px × 0.5 = 25 m wide; 25×25 = 625m²... let's shrink).
    # Use 14×14 = 7m × 7m = 49 m² (residential).
    box_x1, box_y1, box_x2, box_y2 = 100, 100, 114, 114
    crop = np.zeros((14, 14), dtype=np.uint8)
    crop[1:13, 1:13] = 240  # well above default mask_threshold * 255 = 76.5

    det = Detection(
        box_chip_xyxy=(float(box_x1), float(box_y1), float(box_x2), float(box_y2)),
        box_source_xyxy=(float(box_x1), float(box_y1), float(box_x2), float(box_y2)),
        score=0.9,  # above pre_vector_score_threshold(0.3) and tiered conf(0.85 residential)
        label=1,
        mask_crop_uint8=crop,
        mask_crop_offset=(box_x1, box_y1),
        mask_crop_shape=(14, 14),
        source_detection_index=0,
    )
    chip = Chip(
        chip_index=0,
        source_tif=tif_meta["path"],
        source_tile_id=Path(tif_meta["path"]).stem,
        source_crs=tif_meta["crs"],
        source_transform=tif_meta["transform"],
        window=(0, 0, 400, 400),
        window_transform=tif_meta["transform"],
        valid_window=(0, 0, 400, 400),
        valid_shape=(400, 400),
        chip_shape=(400, 400),
        detections=[det],
    )
    src = SourceTile(
        path=tif_meta["path"], size_bytes=tif_meta["size_bytes"],
        mtime=tif_meta["mtime"], crs=tif_meta["crs"],
        transform=tif_meta["transform"], bounds=tif_meta["bounds"],
        shape=tif_meta["shape"],
    )
    return RawArtifact(
        schema_version=SCHEMA_VERSION,
        pipeline_version=PIPELINE_VERSION,
        created_at_utc=utc_now_iso(),
        git_commit="", script_sha256="",
        torch_version="", torchvision_version="", rasterio_version="",
        grid_id="G1110",   # any grid registered in regions.yaml under jhb
        region_arg="jhb", region_key="johannesburg",
        imagery_layer_id="aerial_2023",
        model_run_id="test_run",
        model_path="/tmp/fake.pth", model_sha256="0" * 64,
        model_builder="core.models.build_solar_maskrcnn",
        detector_score_threshold=0.05,
        detections_per_img=300,
        nms_thresh=0.5,
        mask_threshold_used=0.3,
        raw_mask_storage="crop",
        chip_size=(400, 400), overlap=0.0, edge_pad=True,
        source_tiles=[src],
        chips=[chip],
    )


def test_finalize_writes_min_schema(tmp_path):
    """Round-trip: synthetic artifact → finalize → gpkg has minimum schema."""
    tif = tmp_path / "G1110_mosaic.tif"
    tif_meta = _write_synthetic_jhb_tif(tif)
    artifact = _make_synthetic_artifact(tif_meta)

    raw_path = tmp_path / "raw_detections.pkl"
    write_artifact(artifact, raw_path)

    out_dir = tmp_path / "out"

    # Run finalize via subprocess so we exercise the CLI path.
    cmd = [
        sys.executable, str(REPO_ROOT / "finalize.py"),
        "--input", str(raw_path),
        "--output-dir", str(out_dir),
    ]
    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(f"finalize.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    gpkg = out_dir / "predictions_metric.gpkg"
    geojson = out_dir / "predictions.geojson"
    config = out_dir / "config.json"
    diag = out_dir / "diagnostics.md"
    assert gpkg.exists()
    assert geojson.exists()
    assert config.exists()
    assert diag.exists()

    g = gpd.read_file(gpkg)
    # Minimum schema (V1.4 plan)
    required = {
        "geometry", "score", "mask_mean_confidence", "confidence",
        "area_m2", "elongation", "solidity",
        "mean_r", "mean_g", "mean_b",
        "source_tile", "source_tif", "chip_index", "label",
    }
    assert required.issubset(set(g.columns)), f"missing: {required - set(g.columns)}"
    # Metric CRS for JHB is EPSG:32735
    assert str(g.crs) == "EPSG:32735"
    # We have one polygon (filtered through everything)
    assert len(g) == 1
    # Confidence == score (Phase 1)
    assert abs(g.iloc[0]["confidence"] - g.iloc[0]["score"]) < 1e-9

    # Geojson is 4326
    gj = gpd.read_file(geojson)
    assert str(gj.crs) == "EPSG:4326"

    cfg = json.loads(config.read_text())
    assert cfg["pipeline_version"] == "direct_maskrcnn_v1"
    assert cfg["confidence_source"] == "score"
    assert cfg["result_count"] == 1
    assert cfg["stage_counts"]["raw_total"] == 1


def test_canonical_overwrite_guard(tmp_path):
    """finalize.py refuses to overwrite a non-direct config.json."""
    tif = tmp_path / "G1110_mosaic.tif"
    tif_meta = _write_synthetic_jhb_tif(tif)
    artifact = _make_synthetic_artifact(tif_meta)
    raw_path = tmp_path / "raw_detections.pkl"
    write_artifact(artifact, raw_path)

    # Pre-create a non-direct config.json in the output dir.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "config.json").write_text(json.dumps({"pipeline_version": "old_geoai_v1"}))

    cmd = [
        sys.executable, str(REPO_ROOT / "finalize.py"),
        "--input", str(raw_path),
        "--output-dir", str(out_dir),
    ]
    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "refusing to overwrite" in (result.stdout + result.stderr)
