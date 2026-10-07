"""The Durban pair is a small explicit batch, not a citywide switch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import shapely
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_pilot import pregate_pair  # noqa: E402
from scripts.gate_pilot.pregate_pair import (  # noqa: E402
    assemble_pair_report,
    build_plan,
    classify_coverage,
    compare_timing,
    coverage_from_gpkgs,
    execute,
    normalize_grids,
    summarize_coverage,
)


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    head = tmp_path / "heads" / "vit30_fold0" / "head.pt"
    head.parent.mkdir(parents=True, exist_ok=True)
    head.write_bytes(b"head")
    model = tmp_path / "detector.pth"
    model.write_bytes(b"detector")
    values = {
        "grids": ["DBN0001", "dbn0002"],
        "model_path": model,
        "imagery_layer": "vexcel_ortho_2025_12_2026_01",
        "pregate_head": [str(head)],
        "pregate_threshold": 0.45,
        "output_root": tmp_path / "out",
        "batch_size": 8,
        "num_workers": 4,
        "device": "cpu",
        "detector_score_threshold": 0.05,
        "postproc_config": ROOT / "configs" / "postproc" / "v4_canonical.json",
        "allow_larger_batch": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_eleven_grids_are_refused_and_there_is_no_citywide_switch():
    with pytest.raises(ValueError, match="no citywide switch"):
        normalize_grids([f"DBN{i:04d}" for i in range(1, 12)], allow_larger_batch=False)


def test_duplicate_and_non_durban_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_grids(["DBN0001", "dbn0001"], allow_larger_batch=False)
    with pytest.raises(ValueError, match="not a Durban"):
        normalize_grids(["CPT1238"], allow_larger_batch=False)


def test_plan_runs_each_grid_twice_and_does_not_download(tmp_path):
    plan = build_plan(_args(tmp_path))
    assert plan["citywide"] is False
    assert plan["downloads_imagery"] is False
    assert plan["deletes_tiles"] is False
    assert plan["grids"] == ["DBN0001", "DBN0002"]
    assert plan["threshold"] == 0.45
    assert len(plan["steps"]) == 8
    kinds = [(step["grid_id"], step["arm"], step["kind"]) for step in plan["steps"]]
    assert kinds == [
        ("DBN0001", "off", "detect"),
        ("DBN0001", "off", "finalize"),
        ("DBN0001", "on", "detect"),
        ("DBN0001", "on", "finalize"),
        ("DBN0002", "off", "detect"),
        ("DBN0002", "off", "finalize"),
        ("DBN0002", "on", "detect"),
        ("DBN0002", "on", "finalize"),
    ]
    for step in plan["steps"]:
        assert "download" not in " ".join(step["argv"])
        assert step["cwd"] == str(ROOT)
    off = plan["steps"][0]["argv"]
    on = plan["steps"][2]["argv"]
    assert "--pregate-head" not in off
    assert "--stage-timing" in off and "--stage-timing" in on
    assert "--pregate-head" in on
    assert "--pregate-threshold" in on
    assert on[on.index("--pregate-threshold") + 1] == "0.45"
    assert "--merge-mode" in plan["steps"][1]["argv"]
    assert "per-detection" in plan["steps"][1]["argv"]
    assert str(plan["steps"][0]["output_dir"]).endswith("/off/DBN0001")
    assert str(plan["steps"][2]["output_dir"]).endswith("/on/DBN0001")
    assert plan["postproc_config"].endswith("v4_canonical.json")


def test_execute_refuses_citywide_download_or_an_existing_raw(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="citywide"):
        execute({"citywide": True, "downloads_imagery": False, "deletes_tiles": False})
    plan = build_plan(_args(tmp_path))
    raw = Path(plan["steps"][0]["raw_path"])
    raw.parent.mkdir(parents=True)
    raw.write_text("already", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        execute(plan)
    monkeypatch.setattr(pregate_pair, "existing_raw_paths", lambda _plan: [])
    monkeypatch.setattr(pregate_pair, "DINO_CACHE", tmp_path)
    monkeypatch.setattr(pregate_pair, "missing_tiles", lambda _grids, _layer: ["DBN0001: absent"])
    with pytest.raises(FileNotFoundError, match="does not download"):
        execute({
            "citywide": False,
            "downloads_imagery": False,
            "deletes_tiles": False,
            "grids": ["DBN0001"],
            "imagery_layer": "vexcel_ortho_2025_12_2026_01",
            "steps": [],
        })


def test_coverage_separates_missed_reduced_and_retained():
    reference = [box(0, 0, 10, 10), box(100, 100, 110, 110), box(200, 200, 210, 210)]
    other = [box(0, 0, 10, 5), box(200, 200, 210, 210)]
    rows = classify_coverage(reference, other)
    assert [row["status"] for row in rows] == ["reduced", "missed", "retained"]
    summary = summarize_coverage(rows, gated_only_area_m2=0.0)
    assert summary["n_missed"] == 1
    assert summary["n_reduced"] == 1
    assert summary["n_retained"] == 1
    assert summary["missed_area_m2"] == pytest.approx(100.0)
    assert summary["reduced_uncovered_m2"] == pytest.approx(50.0)
    assert "citywide" in summary["limitation"]


def test_gpkg_pair_uses_the_ungated_polygons_as_the_reference(tmp_path):
    import geopandas as gpd

    off = gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10), box(100, 100, 110, 110)],
        crs="EPSG:32736",
    )
    on = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 5), box(300, 300, 301, 301)], crs="EPSG:32736")
    off_path = tmp_path / "off.gpkg"
    on_path = tmp_path / "on.gpkg"
    off.to_file(off_path, layer="predictions", driver="GPKG")
    on.to_file(on_path, layer="predictions", driver="GPKG")
    _rows, summary = coverage_from_gpkgs(off_path, on_path)
    assert summary["n_reduced"] == 1
    assert summary["n_missed"] == 1
    assert summary["gated_only_area_m2"] == pytest.approx(1.0)


def test_skip_fraction_is_not_reported_as_the_time_ratio():
    off = {
        "loop_wall_s": 100.0,
        "finalize_process_s": 10.0,
        "detect_process_s": 120.0,
        "batch_wait_s": 30.0,
        "pregate_s": 0.0,
        "detect_s": 70.0,
    }
    on = {
        "loop_wall_s": 40.0,
        "finalize_process_s": 10.0,
        "detect_process_s": 55.0,
        "batch_wait_s": 12.0,
        "pregate_s": 15.0,
        "detect_s": 10.0,
        "chip_skip_fraction": 0.8339,
    }
    timing = compare_timing(off, on)
    assert timing["work_s_ratio_on_over_off"] == pytest.approx(50.0 / 110.0)
    assert timing["process_s_ratio_on_over_off"] == pytest.approx(65.0 / 130.0)
    assert timing["chip_skip_fraction"] == pytest.approx(0.8339)
    assert timing["chip_skip_fraction"] != pytest.approx(1.0 - timing["work_s_ratio_on_over_off"])
    assert timing["chip_skip_fraction_is_not_time_saved"] is True
    report = assemble_pair_report([{
        "grid_id": "DBN0001",
        "coverage": {
            "n_missed": 2,
            "n_reduced": 1,
            "n_retained": 4,
            "reference_area_m2": 70.0,
            "covered_area_m2": 40.0,
        },
        "timing": timing,
        "polygons": [],
    }])
    assert report["n_missed"] == 2
    assert report["n_reduced"] == 1
    assert report["citywide"] is False
    assert report["work_s_ratio_on_over_off"] == pytest.approx(50.0 / 110.0)
    assert "not the share of time saved" in report["reading"]


def test_empty_union_does_not_crash():
    rows = classify_coverage([box(0, 0, 1, 1)], [shapely.geometry.Polygon()])
    assert rows[0]["status"] == "missed"
