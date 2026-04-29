"""Round-trip + schema-version tests for raw_artifact."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from core.inference.raw_artifact import (
    Chip,
    Detection,
    RawArtifact,
    SCHEMA_VERSION,
    SchemaVersionError,
    SourceTile,
    read_artifact,
    utc_now_iso,
    write_artifact,
)


def _build_artifact() -> RawArtifact:
    mask = np.random.randint(0, 256, size=(20, 20), dtype=np.uint8)
    det = Detection(
        box_chip_xyxy=(10.5, 20.5, 30.5, 40.5),
        box_source_xyxy=(110.5, 220.5, 130.5, 240.5),
        score=0.87,
        label=1,
        mask_crop_uint8=mask,
        mask_crop_offset=(10, 20),
        mask_crop_shape=(20, 20),
        source_detection_index=3,
    )
    chip = Chip(
        chip_index=0,
        source_tif="/tmp/fake.tif",
        source_tile_id="G1234_0_0_geo",
        source_crs="EPSG:4326",
        source_transform=(0.001, 0.0, 18.0, 0.0, -0.001, -34.0),
        window=(0, 0, 400, 400),
        window_transform=(0.001, 0.0, 18.0, 0.0, -0.001, -34.0),
        valid_window=(0, 0, 400, 400),
        valid_shape=(400, 400),
        chip_shape=(400, 400),
        detections=[det],
    )
    src = SourceTile(
        path="/tmp/fake.tif", size_bytes=12345, mtime=1.0,
        crs="EPSG:4326",
        transform=(0.001, 0.0, 18.0, 0.0, -0.001, -34.0),
        bounds=(18.0, -34.4, 18.4, -34.0),
        shape=(4096, 4096),
    )
    return RawArtifact(
        schema_version=SCHEMA_VERSION,
        pipeline_version="direct_maskrcnn_v1",
        created_at_utc=utc_now_iso(),
        git_commit="abc123",
        script_sha256="def456",
        torch_version="2.0.0",
        torchvision_version="0.15.0",
        rasterio_version="1.3.0",
        grid_id="G1234",
        region_arg="ct",
        region_key="cape_town",
        imagery_layer_id="aerial_2025",
        model_run_id="v3c_targeted_hn_aerial_2025",
        model_path="/tmp/best.pth",
        model_sha256="0" * 64,
        model_builder="core.models.build_solar_maskrcnn",
        detector_score_threshold=0.05,
        detections_per_img=300,
        nms_thresh=0.5,
        mask_threshold_used=0.3,
        raw_mask_storage="crop",
        chip_size=(400, 400),
        overlap=0.25,
        edge_pad=True,
        source_tiles=[src],
        chips=[chip],
    )


def test_round_trip(tmp_path):
    a = _build_artifact()
    p = tmp_path / "raw.pkl"
    write_artifact(a, p)
    b = read_artifact(p)

    # Top-level metadata exact
    assert b.schema_version == a.schema_version
    assert b.pipeline_version == a.pipeline_version
    assert b.grid_id == a.grid_id
    assert b.region_key == a.region_key
    assert b.detector_score_threshold == a.detector_score_threshold
    assert b.chip_size == a.chip_size
    assert b.overlap == a.overlap

    # Source tiles
    assert len(b.source_tiles) == len(a.source_tiles)
    assert b.source_tiles[0].path == a.source_tiles[0].path

    # Chip + detection structure preserved
    assert len(b.chips) == 1
    assert len(b.chips[0].detections) == 1
    bd = b.chips[0].detections[0]
    ad = a.chips[0].detections[0]
    assert bd.box_chip_xyxy == ad.box_chip_xyxy
    assert bd.score == ad.score
    assert bd.label == ad.label
    assert bd.mask_crop_offset == ad.mask_crop_offset
    assert bd.mask_crop_shape == ad.mask_crop_shape

    # Numpy mask exact
    assert np.array_equal(bd.mask_crop_uint8, ad.mask_crop_uint8)


def test_schema_version_mismatch_raises(tmp_path):
    p = tmp_path / "old.pkl"
    payload = {"schema_version": 999, "junk": "data"}
    with open(p, "wb") as f:
        pickle.dump(payload, f, protocol=5)
    with pytest.raises(SchemaVersionError, match="schema_version=999"):
        read_artifact(p)


def test_not_artifact_raises(tmp_path):
    p = tmp_path / "garbage.pkl"
    with open(p, "wb") as f:
        pickle.dump([1, 2, 3], f, protocol=5)
    with pytest.raises(SchemaVersionError, match="not a raw-detections artifact"):
        read_artifact(p)
