"""The presence pregate stays off unless a head is supplied."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.inference.presence_pregate import (  # noqa: E402
    TRIAL_THRESHOLD,
    VIT_DIM,
    VIT_IMG,
    VIT_MEAN,
    VIT_PATCHES,
    VIT_STD,
    PresencePregate,
    StageClock,
    build_timing_payload,
    default_head_paths,
    keep_from_scores,
    prepare_gate_batch,
    resolve_pregate_settings,
    write_decisions,
)
from scripts.gate_pilot.learning_curve_train import (  # noqa: E402
    VIT_DIM as TRAIN_DIM,
    VIT_IMG as TRAIN_IMG,
    VIT_MEAN as TRAIN_MEAN,
    VIT_STD as TRAIN_STD,
)


def test_preprocessing_matches_the_fold_trainer():
    assert VIT_IMG == TRAIN_IMG
    assert VIT_DIM == TRAIN_DIM
    assert VIT_MEAN == TRAIN_MEAN
    assert VIT_STD == TRAIN_STD


def test_default_heads_are_the_four_trained_folds():
    paths = default_head_paths()
    assert len(paths) == 4
    assert [path.parent.name for path in paths] == [
        "vit30_fold0", "vit30_fold1", "vit30_fold2", "vit30_fold3",
    ]
    assert all(path.is_file() for path in paths)
    assert not any("fold4" in path.parent.name for path in paths)


def test_gate_is_off_without_a_head():
    assert resolve_pregate_settings(
        heads=None, threshold=None, chip_size=400, allow_non400=False,
    ) is None


def test_threshold_without_a_head_is_refused():
    with pytest.raises(ValueError, match="without --pregate-head"):
        resolve_pregate_settings(
            heads=None, threshold=0.45, chip_size=400, allow_non400=False,
        )


def test_non_400_chip_is_refused(tmp_path):
    head = tmp_path / "vit30_fold0" / "head.pt"
    head.parent.mkdir()
    head.write_bytes(b"not read")
    with pytest.raises(ValueError, match="400px"):
        resolve_pregate_settings(
            heads=[str(head)], threshold=None, chip_size=512, allow_non400=False,
        )


def test_missing_head_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_pregate_settings(
            heads=[str(tmp_path / "vit30_fold0" / "head.pt")],
            threshold=None,
            chip_size=400,
            allow_non400=False,
        )


def test_omitted_threshold_uses_the_cape_town_trial_value(tmp_path):
    head = tmp_path / "vit30_fold0" / "head.pt"
    head.parent.mkdir()
    head.write_bytes(b"x")
    settings = resolve_pregate_settings(
        heads=[str(head)], threshold=None, chip_size=400, allow_non400=False,
    )
    assert settings is not None
    assert settings.threshold == TRIAL_THRESHOLD == 0.45


def test_keep_uses_the_max_score_and_includes_the_threshold():
    scores = np.array([
        [0.44, 0.45],
        [0.10, 0.20],
        [0.90, 0.01],
    ])
    kept = keep_from_scores(scores, 0.45)
    assert kept.tolist() == [True, False, True]
    assert keep_from_scores(np.array([0.45, 0.449]), 0.45).tolist() == [True, False]


def test_unnormalized_chips_are_refused():
    chip = torch.full((3, 8, 8), 255.0)
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        prepare_gate_batch([chip])


def test_prepare_resizes_and_normalizes_a_constant_chip():
    chip = torch.full((3, 4, 4), 0.5)
    batch = prepare_gate_batch([chip])
    assert tuple(batch.shape) == (1, 3, VIT_IMG, VIT_IMG)
    expected = (0.5 - VIT_MEAN[0]) / VIT_STD[0]
    assert batch[0, 0, 0, 0].item() == pytest.approx(expected, rel=1e-5)


def _save_head(path: Path, bias: float) -> None:
    head = torch.nn.Linear(VIT_DIM, 1)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.constant_(head.bias, bias)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict()}, path)


class _FlatTokens(torch.nn.Module):
    def forward_features(self, batch):
        return torch.zeros(batch.shape[0], VIT_PATCHES + 1, VIT_DIM)


def test_score_keeps_a_chip_only_when_some_head_clears_the_threshold(tmp_path):
    low = tmp_path / "vit_low" / "head.pt"
    high = tmp_path / "vit_high" / "head.pt"
    _save_head(low, bias=-2.0)  # sigmoid(-2) ≈ 0.119
    _save_head(high, bias=0.0)   # sigmoid(0) = 0.5
    gate = PresencePregate([low, high], device="cpu", backbone=_FlatTokens())
    scores = gate.score([torch.zeros(3, 8, 8)])
    assert scores.shape == (1, 2)
    assert scores[0, 0] == pytest.approx(torch.sigmoid(torch.tensor(-2.0)).item(), abs=1e-5)
    assert scores[0, 1] == pytest.approx(0.5, abs=1e-5)
    assert keep_from_scores(scores, 0.45).tolist() == [True]

    both_low = tmp_path / "vit_lower" / "head.pt"
    _save_head(both_low, bias=-2.0)
    gate = PresencePregate([low, both_low], device="cpu", backbone=_FlatTokens())
    rejected = gate.score([torch.zeros(3, 8, 8)])
    assert keep_from_scores(rejected, 0.45).tolist() == [False]


def test_timing_payload_does_not_treat_skip_fraction_as_time_saved():
    clock = StageClock(batch_wait_s=1, pregate_s=2, detect_s=3, loop_wall_s=10, n_chips=100, n_kept=17, n_rejected=83)
    payload = build_timing_payload(clock=clock, settings=None)
    assert payload["gate_enabled"] is False
    assert payload["chip_skip_fraction"] == pytest.approx(0.83)
    assert payload["chip_skip_fraction"] != pytest.approx(1.0 - payload["detect_s"] / payload["loop_wall_s"])
    assert "not the share of wall-clock time saved" in payload["note"]


def test_decisions_round_trip(tmp_path):
    path = tmp_path / "pregate_decisions.csv"
    write_decisions(path, [{
        "chip_index": 3,
        "kept": 1,
        "score_max": "0.45000000",
        "vit30_fold0": "0.45000000",
    }], ["vit30_fold0"])
    text = path.read_text(encoding="utf-8")
    assert "chip_index,kept,score_max,vit30_fold0" in text
    assert "3,1,0.45000000,0.45000000" in text


def test_detect_direct_defaults_leave_the_gate_off():
    import detect_direct

    args = detect_direct.build_parser().parse_args([
        "--grid-id", "DBN0001",
        "--region", "durban",
        "--imagery-layer", "vexcel_ortho_2025_12_2026_01",
        "--model-run", "probe",
        "--model-path", "missing.pth",
    ])
    assert args.pregate_head is None
    assert args.pregate_threshold is None
    assert args.stage_timing is False


def test_citywide_template_does_not_call_the_pregate():
    text = (ROOT / "scripts" / "runpod_detect_direct_template.sh").read_text(encoding="utf-8")
    assert "pregate" not in text


def _write_two_chip_tif(path: Path) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    array = np.full((3, 400, 800), 80, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=400, width=800, count=3,
        dtype="uint8", crs="EPSG:3857",
        transform=from_origin(0, 0, 0.1, 0.1),
    ) as handle:
        handle.write(array)


def test_detect_direct_drops_only_the_rejected_chip(tmp_path, monkeypatch):
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
                outputs.append({
                    "boxes": torch.tensor([[1.0, 1.0, 6.0, 6.0]]),
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
    out = tmp_path / "on"
    code = detect_direct.run(detect_direct.build_parser().parse_args([
        "--grid-id", "DBN0001",
        "--region", "durban",
        "--imagery-layer", "vexcel_ortho_2025_12_2026_01",
        "--model-run", "durban_pregate_pair_on",
        "--model-path", str(head),
        "--output-dir", str(out),
        "--chip-size", "400",
        "--overlap", "0",
        "--batch-size", "2",
        "--num-workers", "0",
        "--device", "cpu",
        "--pregate-head", str(head),
        "--pregate-threshold", "0.45",
        "--stage-timing",
    ]))
    assert code == 0
    assert model.seen == [1]
    artifact = read_artifact(out / "raw_detections.pkl")
    assert len(artifact.chips) == 2
    assert [len(chip.detections) for chip in artifact.chips] == [0, 1]
    timing = (out / "stage_timing.json").read_text(encoding="utf-8")
    assert '"n_rejected": 1' in timing
    assert "not the share of wall-clock time saved" in timing
    decisions = (out / "pregate_decisions.csv").read_text(encoding="utf-8")
    assert "0,0," in decisions
    assert "1,1," in decisions
