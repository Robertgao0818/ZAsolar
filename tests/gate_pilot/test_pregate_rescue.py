"""Pregate rescue: neighbour boxes that cross a rejected window, one pass."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.inference.pregate_rescue import rescue_candidates  # noqa: E402
from core.inference.presence_pregate import StageClock, build_timing_payload  # noqa: E402


def _window(chip_index: int, tif_index, origin, size=(100, 80)):
    return {
        "chip_index": chip_index,
        "tif_index": tif_index,
        "window": (origin[0], origin[1], size[0], size[1]),
    }


def _det(chip_index: int, score: float, box):
    return {"chip_index": chip_index, "score": score, "box_chip_xyxy": box}


def test_overlap_into_the_rejected_window_triggers():
    windows = [
        _window(0, "tif-a", (0, 0)),
        _window(1, "tif-a", (90, 0)),
    ]
    # Chip 0 box runs from x=80 to x=120 in chip pixels, so tile x=80..120.
    # Chip 1's window starts at tile x=90. Positive overlap.
    chosen = rescue_candidates(
        windows,
        [_det(0, 0.4, (80, 10, 120, 40))],
        rejected=[1],
        min_score=0.3,
    )
    assert chosen == {1}


def test_box_inside_the_neighbour_does_not_trigger():
    windows = [
        _window(0, "tif-a", (0, 0)),
        _window(1, "tif-a", (90, 0)),
    ]
    chosen = rescue_candidates(
        windows,
        [_det(0, 0.9, (10, 10, 40, 40))],
        rejected=[1],
        min_score=0.3,
    )
    assert chosen == set()


def test_edge_touch_without_area_does_not_trigger():
    windows = [
        _window(0, "tif-a", (0, 0), size=(100, 100)),
        _window(1, "tif-a", (100, 0), size=(100, 100)),
    ]
    # Tile box ends exactly on chip 1's left edge. Area is zero.
    chosen = rescue_candidates(
        windows,
        [_det(0, 0.9, (80, 10, 100, 40))],
        rejected=[1],
        min_score=0.3,
    )
    assert chosen == set()


def test_low_score_does_not_trigger():
    windows = [
        _window(0, "tif-a", (0, 0)),
        _window(1, "tif-a", (90, 0)),
    ]
    chosen = rescue_candidates(
        windows,
        [_det(0, 0.299, (80, 10, 120, 40))],
        rejected=[1],
        min_score=0.3,
    )
    assert chosen == set()
    chosen = rescue_candidates(
        windows,
        [_det(0, 0.3, (80, 10, 120, 40))],
        rejected=[1],
        min_score=0.3,
    )
    assert chosen == {1}


def test_different_tif_does_not_trigger():
    windows = [
        _window(0, "tif-a", (0, 0)),
        _window(1, "tif-b", (90, 0)),
    ]
    chosen = rescue_candidates(
        windows,
        [_det(0, 0.9, (80, 10, 120, 40))],
        rejected=[1],
        min_score=0.3,
    )
    assert chosen == set()


def test_no_cascade_from_a_rejected_chip():
    """Chip 1 would bridge chip 0 and chip 2, but chip 1 was itself rejected."""
    windows = [
        _window(0, "tif-a", (0, 0), size=(100, 100)),
        _window(1, "tif-a", (90, 0), size=(100, 100)),
        _window(2, "tif-a", (180, 0), size=(100, 100)),
    ]
    # Only chip 0 was kept. Its box reaches chip 1 and stops short of chip 2.
    chosen = rescue_candidates(
        windows,
        [_det(0, 0.8, (80, 10, 110, 40))],
        rejected=[1, 2],
        min_score=0.3,
    )
    assert chosen == {1}


def test_irregular_window_offsets():
    windows = [
        _window(0, "tif-a", (10, 20), size=(50, 40)),
        _window(1, "tif-a", (55, 18), size=(30, 40)),
    ]
    # Chip-0 box (40, 5)-(50, 15) -> tile (50, 25)-(60, 35).
    # Chip 1 occupies tile x=55..85, y=18..58. Overlap is real and not on a stride.
    chosen = rescue_candidates(
        windows,
        [_det(0, 0.5, (40, 5, 50, 15))],
        rejected=[1],
        min_score=0.3,
    )
    assert chosen == {1}
    missed = rescue_candidates(
        windows,
        [_det(0, 0.5, (0, 0, 10, 10))],
        rejected=[1],
        min_score=0.3,
    )
    assert missed == set()


def test_timing_payload_gains_rescue_fields_only_when_asked():
    clock = StageClock(n_chips=4, n_kept=3, n_rejected=1, n_detections=2, n_rescued=1, rescue_s=0.25)
    plain = build_timing_payload(clock=clock, settings=None)
    assert "n_rescued" not in plain
    assert "rescue_s" not in plain
    rescued = build_timing_payload(clock=clock, settings=None, include_rescue=True)
    assert rescued["n_rescued"] == 1
    assert rescued["rescue_s"] == pytest.approx(0.25)
    assert {key: plain[key] for key in plain} == {
        key: rescued[key] for key in plain
    }


def _write_two_chip_tif(path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    array = np.full((3, 400, 800), 80, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=400, width=800, count=3,
        dtype="uint8", crs="EPSG:3857",
        transform=from_origin(0, 0, 0.1, 0.1),
    ) as handle:
        handle.write(array)


def test_stub_model_replaces_the_empty_chip_and_marks_rescued(tmp_path, monkeypatch, capsys):
    import detect_direct
    from core.inference.raw_artifact import read_artifact
    from core.models.maskrcnn import LoadInfo

    tile = tmp_path / "DBN0001_0_0_geo.tif"
    _write_two_chip_tif(tile)
    head = tmp_path / "vit30_fold0" / "head.pt"
    head.parent.mkdir()
    head.write_bytes(b"unused")

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.roi_heads = type("Heads", (), {})()
            self.seen: list[int] = []

        def forward(self, images):
            self.seen.append(len(images))
            outputs = []
            for _image in images:
                # Box starts left of this chip, so a detection on the right
                # chip (origin x=400) reaches back into the left chip.
                outputs.append({
                    "boxes": torch.tensor([[-30.0, 10.0, 20.0, 40.0]]),
                    "scores": torch.tensor([0.8]),
                    "labels": torch.tensor([1]),
                    "masks": torch.ones(1, 1, 400, 400),
                })
            return outputs

    model = _Model()

    def _builder(*_args, **_kwargs):
        return model, LoadInfo(missing=[], unexpected=[], pretrained_path="fake")

    class _Gate:
        def __init__(self, head_paths, device):
            self.head_paths = head_paths

        def score(self, chips):
            assert len(chips) == 2
            return np.array([[0.10], [0.90]])

    monkeypatch.setattr(detect_direct, "resolve_tile_paths", lambda _args, _key: [tile])
    monkeypatch.setattr(detect_direct, "build_solar_maskrcnn", _builder)
    monkeypatch.setattr(detect_direct, "PresencePregate", _Gate)
    out = tmp_path / "rescued"
    code = detect_direct.run(detect_direct.build_parser().parse_args([
        "--grid-id", "DBN0001",
        "--region", "durban",
        "--imagery-layer", "vexcel_ortho_2025_12_2026_01",
        "--model-run", "rescue_probe",
        "--model-path", str(head),
        "--output-dir", str(out),
        "--chip-size", "400",
        "--overlap", "0",
        "--batch-size", "2",
        "--num-workers", "0",
        "--device", "cpu",
        "--pregate-head", str(head),
        "--pregate-threshold", "0.45",
        "--pregate-rescue",
        "--pregate-rescue-min-score", "0.3",
        "--stage-timing",
    ]))
    assert code == 0
    # Main loop detects the one kept chip, then the rescue pass detects the other.
    assert model.seen == [1, 1]
    artifact = read_artifact(out / "raw_detections.pkl")
    assert [len(chip.detections) for chip in artifact.chips] == [1, 1]
    assert artifact.chips[0].chip_index == 0
    timing = json.loads((out / "stage_timing.json").read_text(encoding="utf-8"))
    assert timing["n_rescued"] == 1
    assert timing["n_rejected"] == 1
    assert "rescue_s" in timing
    text = capsys.readouterr().out
    assert "pregate_rescue n_candidates=1 n_rescued=1" in text
    with (out / "pregate_decisions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["chip_index"] for row in rows] == ["0", "1"]
    assert [row["kept"] for row in rows] == ["0", "1"]
    assert [row["rescued"] for row in rows] == ["1", "0"]
