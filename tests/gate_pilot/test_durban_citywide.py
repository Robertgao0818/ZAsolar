"""Citywide Durban runner: one GPU slot, CPU staging and finalize, no solar_cls."""

from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_pilot import durban_citywide as citywide  # noqa: E402
from scripts.gate_pilot.durban_citywide import (  # noqa: E402
    assign_actions,
    build_plan,
    check_engine,
    classify_finalize_result,
    discover_grids,
    execute,
    finalize_one,
    gpu_process_main,
    main,
    reap_intermediates,
    release_staged,
    run_pipeline,
    stage_grid,
    write_sentinel,
)


def _touch_grid(root: Path, grid_id: str, n_tiles: int, size: int = 16) -> None:
    folder = root / grid_id
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(n_tiles):
        (folder / f"{grid_id}_0_{index}_geo.tif").write_bytes(b"x" * size)


def _heads(tmp_path: Path, n: int = 4) -> list[Path]:
    paths = []
    for index in range(n):
        path = tmp_path / "heads" / f"vit30_fold{index}" / "head.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"head")
        paths.append(path)
    return paths


def _args(tmp_path: Path, tile_root: Path, **overrides) -> Namespace:
    model = tmp_path / "detector.pth"
    model.write_bytes(b"detector")
    values = {
        "tile_root": tile_root,
        "grids": None,
        "model_path": model,
        "imagery_layer": "vexcel_ortho_2025_12_2026_01",
        "pregate_head": [str(path) for path in _heads(tmp_path)],
        "pregate_threshold": 0.45,
        "output_root": tmp_path / "out",
        "postproc_config": citywide.DEFAULT_POSTPROC,
        "batch_size": 8,
        "num_workers": 4,
        "prefetch_factor": 2,
        "device": "cpu",
        "detector_score_threshold": 0.05,
        "no_stage": False,
        "shm_root": tmp_path / "shm",
        "shm_budget_gib": 20.0,
        "stage_ahead": 3,
        "stage_workers": 2,
        "finalize_workers": 2,
        "reap_raw": False,
        "pregate_rescue": False,
        "pregate_rescue_min_score": 0.3,
    }
    values.update(overrides)
    return Namespace(**values)


def _quiet(_message: str) -> None:
    return None


def _grid(grid_id: str, action: str = "detect", nbytes: int = 100) -> dict:
    return {
        "grid_id": grid_id,
        "tile_count": 4,
        "complete": True,
        "bytes": nbytes,
        "source_dir": f"/tiles/{grid_id}",
        "action": action,
    }


def _sched_plan(tmp_path: Path, grids: list[dict], **overrides) -> dict:
    plan = {
        "state_dir": str(tmp_path / "state"),
        "grids": grids,
        "stage_ahead": 3,
        "shm_budget_bytes": 20 * 1024**3,
        "stage_tiles": True,
        "stage_workers": 2,
        "finalize_workers": 2,
        "n_skip": sum(1 for grid in grids if grid["action"] == "skip"),
        "n_grids": len(grids),
        "shm_root": str(tmp_path / "shm"),
    }
    plan.update(overrides)
    return plan


def _ok_detect(_grid_row: dict) -> dict:
    return {"rc": 0, "error": ""}


def _ok_finalize(_grid_row: dict) -> dict:
    return {"rc": 0, "empty": False}


def _ok_stage(grid: dict) -> int:
    return int(grid["bytes"])


def _ok_release(_grid_row: dict) -> None:
    return None


def test_discovery_keeps_a_partial_grid_and_ignores_junk(tmp_path: Path):
    root = tmp_path / "tiles"
    _touch_grid(root, "DBN0001", 4, size=8)
    _touch_grid(root, "DBN0002", 3, size=8)
    (root / "DBN0003").mkdir(parents=True)
    (root / "notes").mkdir()
    (root / "DBN12").mkdir()
    found = discover_grids(root)
    assert [grid["grid_id"] for grid in found] == ["DBN0001", "DBN0002"]
    assert found[0]["complete"] is True
    assert found[0]["bytes"] == 32
    assert found[1]["complete"] is False
    assert found[1]["tile_count"] == 3


def test_more_than_ten_grids_stay_gate_on_only(tmp_path: Path):
    root = tmp_path / "tiles"
    for index in range(1, 12):
        _touch_grid(root, f"DBN{index:04d}", 4)
    _touch_grid(root, "DBN0012", 3)
    plan = build_plan(_args(tmp_path, root))
    assert plan["n_grids"] == 12
    assert plan["citywide"] is True
    assert plan["solar_cls"] is False
    assert plan["off_arm"] is False
    assert plan["downloads_imagery"] is False
    assert plan["deletes_source_tiles"] is False
    assert plan["gpu_slots"] == 1
    assert plan["threshold"] == 0.45
    assert plan["keep_rule"] == "max head score >= threshold"
    assert plan["merge_mode"] == "per-detection"
    assert plan["n_partial"] == 1
    assert plan["partial_grid_ids"] == ["DBN0012"]
    assert plan["n_detect"] == 12
    detect_args = plan["detect_args"]
    assert detect_args["chip_size"] == 400
    assert detect_args["overlap"] == 0.25
    assert detect_args["detections_per_img"] == 300
    assert detect_args["nms_iou_threshold"] == 0.5
    assert detect_args["detector_score_threshold"] == 0.05
    assert detect_args["mask_threshold"] == 0.3
    assert detect_args["raw_mask_storage"] == "crop"
    assert detect_args["pregate_threshold"] == 0.45
    assert len(detect_args["pregate_head"]) == 4
    assert plan["engine"]["solar_cls"] is False
    assert plan["reap_raw"] is False
    assert plan["pregate_rescue"] is False
    assert plan["detect_args"]["pregate_rescue"] is False
    assert plan["detect_args"]["pregate_rescue_min_score"] == 0.3
    assert "pregate_rescue" not in plan["engine"]
    assert "not the share of time saved" in plan["reading"]
    assert "source tiles stay" in plan["reading"]


def test_default_heads_come_from_the_four_folds(tmp_path: Path, monkeypatch):
    root = tmp_path / "tiles"
    _touch_grid(root, "DBN0001", 4)
    heads = _heads(tmp_path)
    monkeypatch.setattr(citywide, "default_head_paths", lambda: tuple(heads))
    plan = build_plan(_args(tmp_path, root, pregate_head=None))
    assert plan["heads"] == [str(path.resolve()) for path in heads]
    assert plan["threshold"] == citywide.TRIAL_THRESHOLD


def test_bad_ids_duplicates_and_missing_tiles_are_refused(tmp_path: Path):
    root = tmp_path / "tiles"
    _touch_grid(root, "DBN0001", 4)
    with pytest.raises(ValueError, match="not a Durban"):
        build_plan(_args(tmp_path, root, grids=["CPT0001"]))
    with pytest.raises(ValueError, match="duplicate"):
        build_plan(_args(tmp_path, root, grids=["DBN0001", "dbn0001"]))
    with pytest.raises(FileNotFoundError, match="DBN0099"):
        build_plan(_args(tmp_path, root, grids=["DBN0099"]))
    with pytest.raises(ValueError, match="threshold"):
        build_plan(_args(tmp_path, root, pregate_threshold=1.5))


def test_resume_skips_ok_retries_fail_and_finalizes_a_raw_pickle(tmp_path: Path):
    state = tmp_path / "state"
    out = tmp_path / "out"
    grids = [_grid(f"DBN000{index}", "detect") for index in range(1, 6)]
    write_sentinel(state, "DBN0001", "ok")
    write_sentinel(state, "DBN0002", "fail")
    (out / "DBN0003").mkdir(parents=True)
    (out / "DBN0003" / "raw_detections.pkl").write_bytes(b"pkl")
    write_sentinel(state, "DBN0004", "empty")
    (out / "DBN0005").mkdir(parents=True)
    (out / "DBN0005" / "raw_detections.pkl").write_bytes(b"pkl")
    write_sentinel(state, "DBN0005", "fail")
    # A finished grid keeps its raw pickle under the default. The sentinel
    # is checked first, so the pickle must not pull it back into finalize_only.
    grids.append(_grid("DBN0006", "detect"))
    (out / "DBN0006").mkdir(parents=True)
    (out / "DBN0006" / "raw_detections.pkl").write_bytes(b"pkl")
    write_sentinel(state, "DBN0006", "ok")
    actions = {row["grid_id"]: row["action"] for row in assign_actions(grids, state, out)}
    assert actions == {
        "DBN0001": "skip",
        "DBN0002": "detect",
        "DBN0003": "finalize_only",
        "DBN0004": "skip",
        "DBN0005": "finalize_only",
        "DBN0006": "skip",
    }


def test_engine_mismatch_and_unmarked_sentinels_are_refused(tmp_path: Path):
    state = tmp_path / "state"
    state.mkdir()
    engine = {"schema": "durban_citywide_pregate_v1", "threshold": 0.45}
    (state / "infer_DBN0001.ok").write_text("")
    with pytest.raises(RuntimeError, match="no ENGINE"):
        check_engine(state, engine)
    (state / "ENGINE.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="different engine"):
        check_engine(state, engine)
    (state / "ENGINE.json").write_text(json.dumps(engine) + "\n")
    check_engine(state, engine)


def test_stage_copies_under_shm_and_release_cannot_leave_it(tmp_path: Path):
    source = tmp_path / "tiles" / "DBN0001"
    source.mkdir(parents=True)
    payload = b"tile-bytes"
    (source / "DBN0001_0_0_geo.tif").write_bytes(payload)
    shm = tmp_path / "shm"
    dest = shm / "DBN0001"
    assert stage_grid(source, dest, shm) == len(payload)
    assert (dest / "DBN0001_0_0_geo.tif").read_bytes() == payload
    assert (source / "DBN0001_0_0_geo.tif").read_bytes() == payload
    release_staged(dest, shm)
    assert not dest.exists()
    assert (source / "DBN0001_0_0_geo.tif").is_file()
    with pytest.raises(RuntimeError, match="outside"):
        release_staged(source, shm)
    with pytest.raises(RuntimeError, match="outside"):
        release_staged(shm, shm)
    assert source.is_dir()
    assert (source / "DBN0001_0_0_geo.tif").read_bytes() == payload
    link = shm / "DBN0001"
    link.symlink_to(source)
    with pytest.raises(RuntimeError, match="outside"):
        stage_grid(source, link, shm)
    assert (source / "DBN0001_0_0_geo.tif").read_bytes() == payload


def test_finalize_classes_and_reap_keeps_the_gate_table(tmp_path: Path):
    assert classify_finalize_result(0, "no polygons after vectorization\n", 0) == {
        "rc": 0, "empty": True,
    }
    assert classify_finalize_result(0, "wrote polygons", 20) == {"rc": 0, "empty": False}
    missing = classify_finalize_result(0, "wrote polygons", 0)
    assert missing["rc"] == 1
    assert missing["error"] == "finalize wrote no geopackage"
    crashed = classify_finalize_result(2, "no polygons after vectorization", 20)
    assert crashed["rc"] == 2
    assert crashed["error"] == "finalize exit 2"
    grid = tmp_path / "DBN0001"
    (grid / "masks").mkdir(parents=True)
    (grid / "vectors").mkdir()
    (grid / "raw_detections.pkl").write_bytes(b"pkl")
    (grid / "pregate_decisions.csv").write_text("chip\n")
    (grid / "predictions_metric.gpkg").write_bytes(b"gpkg")
    reap_intermediates(grid)
    assert (grid / "raw_detections.pkl").is_file()
    assert not (grid / "masks").exists()
    assert not (grid / "vectors").exists()
    assert (grid / "pregate_decisions.csv").is_file()
    assert (grid / "predictions_metric.gpkg").is_file()
    reap_intermediates(grid, keep_raw=False)
    assert not (grid / "raw_detections.pkl").exists()
    assert (grid / "pregate_decisions.csv").is_file()


def test_plan_reap_raw_reaches_finalize(tmp_path: Path, monkeypatch):
    root = tmp_path / "tiles"
    _touch_grid(root, "DBN0001", 4)
    kept = build_plan(_args(tmp_path, root))
    assert kept["reap_raw"] is False
    reaped = build_plan(_args(tmp_path, root, reap_raw=True))
    assert reaped["reap_raw"] is True
    rescued = build_plan(_args(
        tmp_path, root, pregate_rescue=True, pregate_rescue_min_score=0.35,
    ))
    assert rescued["detect_args"]["pregate_rescue"] is True
    assert rescued["detect_args"]["pregate_rescue_min_score"] == 0.35
    assert rescued["engine"]["pregate_rescue"] is True
    assert rescued["engine"]["pregate_rescue_min_score"] == 0.35
    with pytest.raises(ValueError, match="rescue min score"):
        build_plan(_args(tmp_path, root, pregate_rescue=True, pregate_rescue_min_score=1.5))

    def _finish(plan: dict) -> None:
        grid = Path(plan["output_root"]) / "DBN0001"
        grid.mkdir(parents=True, exist_ok=True)
        (grid / "raw_detections.pkl").write_bytes(b"pkl")
        (grid / "masks").mkdir(exist_ok=True)
        (grid / "vectors").mkdir(exist_ok=True)
        (grid / "pregate_decisions.csv").write_text("chip\n", encoding="utf-8")

        def fake_run(argv, **kwargs):
            (grid / "predictions_metric.gpkg").write_bytes(b"gpkg-bytes")
            return __import__("subprocess").CompletedProcess(argv, 0)

        monkeypatch.setattr(citywide.subprocess, "run", fake_run)
        result = finalize_one(plan, "DBN0001")
        assert result["rc"] == 0
        assert result["empty"] is False
        assert not (grid / "masks").exists()
        assert not (grid / "vectors").exists()
        assert (grid / "pregate_decisions.csv").is_file()

    _finish(kept)
    assert (Path(kept["output_root"]) / "DBN0001" / "raw_detections.pkl").is_file()
    _finish(reaped)
    assert not (Path(reaped["output_root"]) / "DBN0001" / "raw_detections.pkl").exists()


def test_stage_failure_releases_and_does_not_detect(tmp_path: Path):
    grids = [_grid("DBN0001"), _grid("DBN0002")]
    detected = []
    released = []

    def stage(grid):
        if grid["grid_id"] == "DBN0002":
            raise RuntimeError("disk full")
        return 1

    def detect(grid):
        detected.append(grid["grid_id"])
        return {"rc": 0, "error": ""}

    counts = run_pipeline(
        _sched_plan(tmp_path, grids),
        detect=detect,
        finalize=_ok_finalize,
        stage=stage,
        release=lambda grid: released.append(grid["grid_id"]),
        log=_quiet,
    )
    assert detected == ["DBN0001"]
    assert "DBN0002" in released
    assert "DBN0001" in released
    assert counts == {"ok": 1, "empty": 0, "fail": 1, "skip": 0}
    state = tmp_path / "state"
    assert (state / "infer_DBN0002.fail").is_file()
    assert (state / "infer_DBN0001.ok").is_file()


def test_detect_failure_releases_immediately_and_does_not_finalize(tmp_path: Path):
    finalized = []
    released = []
    counts = run_pipeline(
        _sched_plan(tmp_path, [_grid("DBN0001")]),
        detect=lambda _grid_row: {"rc": 1, "error": "boom"},
        finalize=lambda grid: finalized.append(grid["grid_id"]) or {"rc": 0, "empty": False},
        stage=_ok_stage,
        release=lambda grid: released.append(grid["grid_id"]),
        log=_quiet,
    )
    assert counts["fail"] == 1
    assert finalized == []
    assert released == ["DBN0001"]
    assert (tmp_path / "state" / "infer_DBN0001.fail").is_file()


def test_release_error_after_detect_failure_still_fails(tmp_path: Path):
    finalized = []

    def release(_grid_row):
        raise RuntimeError("busy")

    counts = run_pipeline(
        _sched_plan(tmp_path, [_grid("DBN0001")]),
        detect=lambda _grid_row: {"rc": 1, "error": "boom"},
        finalize=lambda grid: finalized.append(grid["grid_id"]) or {"rc": 0, "empty": False},
        stage=_ok_stage,
        release=release,
        log=_quiet,
    )
    assert counts["fail"] == 1
    assert finalized == []


def test_release_error_after_successful_finalize_keeps_the_sentinel(tmp_path: Path):
    """Polygons are already written. A late delete error must not flip ok to fail."""
    notes = []

    def release(_grid_row):
        raise RuntimeError("busy")

    counts = run_pipeline(
        _sched_plan(tmp_path, [_grid("DBN0001")]),
        detect=_ok_detect,
        finalize=_ok_finalize,
        stage=_ok_stage,
        release=release,
        log=notes.append,
    )
    assert counts == {"ok": 1, "empty": 0, "fail": 0, "skip": 0}
    assert (tmp_path / "state" / "infer_DBN0001.ok").is_file()
    assert any("release DBN0001 after finalize" in note and "busy" in note for note in notes)


def test_finalize_only_restages_then_releases_after(tmp_path: Path):
    """A resume grid is copied again. The pkl still names the staged tile paths."""
    order = []
    counts = run_pipeline(
        _sched_plan(tmp_path, [_grid("DBN0003", "finalize_only")]),
        detect=lambda _grid_row: (_ for _ in ()).throw(AssertionError("detect")),
        finalize=lambda grid: order.append(("fin", grid["grid_id"])) or {"rc": 0, "empty": True},
        stage=lambda grid: order.append(("stage", grid["grid_id"])) or 1,
        release=lambda grid: order.append(("release", grid["grid_id"])),
        log=_quiet,
    )
    assert order == [("stage", "DBN0003"), ("fin", "DBN0003"), ("release", "DBN0003")]
    assert counts == {"ok": 0, "empty": 1, "fail": 0, "skip": 0}
    assert (tmp_path / "state" / "infer_DBN0003.empty").is_file()


def test_staged_copy_is_present_during_finalize_and_gone_after(tmp_path: Path):
    shm = tmp_path / "shm"
    present = {}

    def stage(grid):
        dest = shm / grid["grid_id"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "tile.tif").write_bytes(b"tif")
        return 3

    def finalize(grid):
        present[grid["grid_id"]] = (shm / grid["grid_id"] / "tile.tif").is_file()
        return {"rc": 0, "empty": False}

    def release(grid):
        dest = shm / grid["grid_id"]
        if dest.exists():
            shutil.rmtree(dest)

    counts = run_pipeline(
        _sched_plan(tmp_path, [_grid("DBN0002")]),
        detect=_ok_detect,
        finalize=finalize,
        stage=stage,
        release=release,
        log=_quiet,
    )
    assert present == {"DBN0002": True}
    assert not (shm / "DBN0002").exists()
    assert counts["ok"] == 1


def test_no_stage_never_copies(tmp_path: Path):
    seen = []
    counts = run_pipeline(
        _sched_plan(tmp_path, [_grid("DBN0001")], stage_tiles=False),
        detect=lambda grid: seen.append(grid["grid_id"]) or {"rc": 0, "error": ""},
        finalize=_ok_finalize,
        stage=lambda _grid_row: (_ for _ in ()).throw(AssertionError("stage")),
        release=lambda grid: seen.append(f"release:{grid['grid_id']}"),
        log=_quiet,
    )
    assert counts["ok"] == 1
    assert seen == ["DBN0001", "release:DBN0001"]


def test_budget_stages_one_oversized_neighbor_at_a_time(tmp_path: Path):
    inside = {"n": 0, "max": 0}
    lock = threading.Lock()

    def stage(_grid_row):
        with lock:
            inside["n"] += 1
            inside["max"] = max(inside["max"], inside["n"])
        time.sleep(0.4)
        with lock:
            inside["n"] -= 1
        return 1

    grids = [_grid("DBN0001", "detect", nbytes=100), _grid("DBN0002", "detect", nbytes=100)]
    counts = run_pipeline(
        _sched_plan(tmp_path, grids, shm_budget_bytes=150, stage_ahead=4),
        detect=_ok_detect,
        finalize=_ok_finalize,
        stage=stage,
        release=_ok_release,
        log=_quiet,
    )
    assert counts["ok"] == 2
    assert inside["max"] == 1


def test_stage_ahead_overlaps_copies(tmp_path: Path):
    inside = {"n": 0, "max": 0}
    lock = threading.Lock()

    def stage(_grid_row):
        with lock:
            inside["n"] += 1
            inside["max"] = max(inside["max"], inside["n"])
        time.sleep(0.4)
        with lock:
            inside["n"] -= 1
        return 1

    grids = [_grid("DBN0001"), _grid("DBN0002")]
    counts = run_pipeline(
        _sched_plan(tmp_path, grids, shm_budget_bytes=10**12, stage_ahead=2),
        detect=_ok_detect,
        finalize=_ok_finalize,
        stage=stage,
        release=_ok_release,
        log=_quiet,
    )
    assert counts["ok"] == 2
    assert inside["max"] == 2


def test_one_detect_slot_overlaps_the_previous_finalize(tmp_path: Path):
    active = {"n": 0, "max": 0}
    lock = threading.Lock()
    finalize_entered = threading.Event()
    second_detect_entered = threading.Event()

    def detect(grid):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        try:
            if grid["grid_id"] == "DBN0002":
                second_detect_entered.set()
                assert finalize_entered.wait(2)
            return {"rc": 0, "error": ""}
        finally:
            with lock:
                active["n"] -= 1

    def finalize(grid):
        if grid["grid_id"] == "DBN0001":
            finalize_entered.set()
            assert second_detect_entered.wait(2)
        return {"rc": 0, "empty": False}

    counts = run_pipeline(
        _sched_plan(tmp_path, [_grid("DBN0001"), _grid("DBN0002")], stage_ahead=2),
        detect=detect,
        finalize=finalize,
        stage=_ok_stage,
        release=_ok_release,
        log=_quiet,
    )
    assert counts == {"ok": 2, "empty": 0, "fail": 0, "skip": 0}
    assert active["max"] == 1
    assert finalize_entered.is_set()
    assert second_detect_entered.is_set()


def test_execute_writes_the_engine_and_does_not_open_a_gpu_process(tmp_path: Path, monkeypatch):
    root = tmp_path / "tiles"
    _touch_grid(root, "DBN0001", 4)
    _touch_grid(root, "DBN0002", 3)
    monkeypatch.setattr(citywide, "DINO_CACHE", tmp_path / "dino")
    (tmp_path / "dino").mkdir()

    def forbid_gpu(_plan):
        raise AssertionError("GPU process started")

    monkeypatch.setattr(citywide, "GpuClient", forbid_gpu)
    plan = build_plan(_args(tmp_path, root, grids=["DBN0001"]))
    plan["solar_cls"] = True
    with pytest.raises(RuntimeError, match="solar_cls"):
        execute(plan)
    plan["solar_cls"] = False
    counts = execute(
        plan,
        detect=_ok_detect,
        finalize=_ok_finalize,
        stage=_ok_stage,
        release=_ok_release,
    )
    assert counts["ok"] == 1
    state = Path(plan["state_dir"])
    assert (state / "infer_DBN0001.ok").is_file()
    assert json.loads((state / "ENGINE.json").read_text()) == plan["engine"]
    summary = json.loads((Path(plan["output_root"]) / "citywide_plan.json").read_text())
    assert "grids" not in summary
    assert summary["partial_grid_ids"] == []
    assert summary["gpu_slots"] == 1


def test_execute_refuses_sentinels_from_another_engine(tmp_path: Path, monkeypatch):
    root = tmp_path / "tiles"
    _touch_grid(root, "DBN0001", 4)
    monkeypatch.setattr(citywide, "DINO_CACHE", tmp_path / "dino")
    (tmp_path / "dino").mkdir()
    plan = build_plan(_args(tmp_path, root))
    state = Path(plan["state_dir"])
    state.mkdir(parents=True)
    (state / "infer_DBN0001.ok").write_text("")
    (state / "ENGINE.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="different engine"):
        execute(plan, detect=_ok_detect, finalize=_ok_finalize, stage=_ok_stage, release=_ok_release)


def test_print_plan_does_not_run_and_a_failed_grid_returns_1(tmp_path: Path, capsys, monkeypatch):
    root = tmp_path / "tiles"
    _touch_grid(root, "DBN0001", 4)
    args = _args(tmp_path, root)
    code = main([
        "print-plan",
        "--tile-root", str(root),
        "--model-path", str(args.model_path),
        "--pregate-head", args.pregate_head[0],
        "--output-root", str(tmp_path / "out"),
        "--shm-root", str(tmp_path / "shm"),
    ])
    assert code == 0
    shown = json.loads(capsys.readouterr().out)
    assert "detect_args" not in shown
    assert shown["n_grids"] == 1
    assert shown["solar_cls"] is False
    assert not (tmp_path / "out" / "citywide_counts.json").exists()

    missing = main([
        "print-plan",
        "--tile-root", str(root),
        "--model-path", str(tmp_path / "missing.pth"),
    ])
    assert missing == 2

    monkeypatch.setattr(citywide, "execute", lambda _plan: {"ok": 0, "empty": 0, "fail": 1, "skip": 0})
    failed = main([
        "run",
        "--tile-root", str(root),
        "--model-path", str(args.model_path),
        "--pregate-head", args.pregate_head[0],
        "--output-root", str(tmp_path / "run"),
        "--shm-root", str(tmp_path / "shm"),
    ])
    assert failed == 1


def test_gpu_env_points_tiles_at_shm_and_forkserver_precedes_the_load(monkeypatch):
    monkeypatch.delenv("SOLAR_TILES_ROOT", raising=False)
    monkeypatch.delenv("SOLAR_DATALOADER_START", raising=False)
    citywide._prepare_gpu_env("/dev/shm/durban_pregate_tiles")
    assert os.environ["SOLAR_TILES_ROOT"] == "/dev/shm/durban_pregate_tiles"
    assert os.environ["SOLAR_DATALOADER_START"] == "forkserver"
    citywide._prepare_gpu_env("")
    assert "SOLAR_TILES_ROOT" not in os.environ

    import detect_direct

    order = []
    monkeypatch.setattr(citywide, "_start_dataloader_forkserver", lambda: order.append("server"))

    def _load(_args):
        order.append("load")
        return "loaded"

    monkeypatch.setattr(detect_direct, "load_runtime", _load)
    module, runtime = citywide._boot_gpu_runtime({"device": "cpu"})
    assert module is detect_direct
    assert runtime == "loaded"
    assert order == ["server", "load"]
    monkeypatch.delenv("SOLAR_DATALOADER_START", raising=False)
    assert detect_direct.dataloader_context() is None
    monkeypatch.setenv("SOLAR_DATALOADER_START", "forkserver")
    assert detect_direct.dataloader_context().get_start_method() == "forkserver"


def test_gpu_process_reports_a_grid_error_and_still_closes(monkeypatch):
    order = []
    runtime = object()

    def infer(args, got_runtime):
        order.append(args.grid_id)
        assert got_runtime is runtime
        assert args.output_dir == Path("/out") / args.grid_id
        assert args.output_raw is None
        if args.grid_id == "DBN0002":
            raise RuntimeError("tile missing")
        return 0

    def close(got_runtime):
        assert got_runtime is runtime
        order.append("close")

    fake = type("Detect", (), {"infer_grid": staticmethod(infer), "close_runtime": staticmethod(close)})
    monkeypatch.setattr(citywide, "_die_with_parent", lambda: order.append("pdeath"))
    monkeypatch.setattr(
        citywide, "_raise_nofile_limit",
        lambda: order.append("nofile") or (1024, 1048576),
    )
    monkeypatch.setattr(citywide, "_prepare_gpu_env", lambda shm: order.append(("env", shm)))
    monkeypatch.setattr(citywide, "_boot_gpu_runtime", lambda _args: (fake, runtime))
    jobs = queue.Queue()
    jobs.put("DBN0001")
    jobs.put("DBN0002")
    jobs.put(None)
    results = queue.Queue()
    gpu_process_main({"output_dir": "/out"}, "/dev/shm/tiles", jobs, results)
    assert results.get_nowait() == {"grid_id": "DBN0001", "rc": 0, "error": ""}
    failed = results.get_nowait()
    assert failed["grid_id"] == "DBN0002"
    assert failed["rc"] == 1
    assert "RuntimeError: tile missing" in failed["error"]
    assert order == ["pdeath", "nofile", ("env", "/dev/shm/tiles"), "DBN0001", "DBN0002", "close"]


def test_nofile_soft_limit_rises_to_the_hard_limit(monkeypatch):
    import resource

    state = {"soft": 1024, "hard": 4096}

    def getrlimit(res):
        assert res == resource.RLIMIT_NOFILE
        return state["soft"], state["hard"]

    def setrlimit(res, limits):
        assert res == resource.RLIMIT_NOFILE
        state["soft"], state["hard"] = limits

    monkeypatch.setattr(resource, "getrlimit", getrlimit)
    monkeypatch.setattr(resource, "setrlimit", setrlimit)
    assert citywide._raise_nofile_limit() == (4096, 4096)


def test_five_consecutive_failures_stop_without_marking_the_rest(tmp_path: Path):
    grids = [_grid(f"DBN{index:04d}") for index in range(1, 8)]
    detected = []
    notes = []

    def detect(grid):
        detected.append(grid["grid_id"])
        return {"rc": 1, "error": "boom"}

    counts = run_pipeline(
        _sched_plan(tmp_path, grids),
        detect=detect,
        finalize=_ok_finalize,
        stage=_ok_stage,
        release=_ok_release,
        log=notes.append,
    )
    assert counts["fail"] == 5
    assert counts["ok"] == 0
    state = tmp_path / "state"
    assert len(list(state.glob("infer_*.fail"))) == 5
    assert len(list(state.glob("infer_*.*"))) == 5
    assert detected == [f"DBN{index:04d}" for index in range(1, 6)]
    assert any(note == "stop: 5 consecutive failures" for note in notes)
    assert "DBN0006" not in detected
    assert "DBN0007" not in detected


def test_a_success_resets_the_consecutive_failure_count(tmp_path: Path):
    # The fifth detect call succeeds. Staging can reorder grid ids, so the
    # reset is defined on call order. One detect slot makes that order serial.
    seen = {"n": 0}

    def detect(_grid_row):
        seen["n"] += 1
        if seen["n"] == 5:
            return {"rc": 0, "error": ""}
        return {"rc": 1, "error": "boom"}

    grids = [_grid(f"DBN{index:04d}") for index in range(1, 10)]
    notes = []
    counts = run_pipeline(
        _sched_plan(tmp_path, grids),
        detect=detect,
        finalize=_ok_finalize,
        stage=_ok_stage,
        release=_ok_release,
        log=notes.append,
    )
    assert counts == {"ok": 1, "empty": 0, "fail": 8, "skip": 0}
    assert seen["n"] == 9
    assert not any(note.startswith("stop:") for note in notes)
    assert len(list((tmp_path / "state").glob("infer_*.*"))) == 9


def test_shutdown_loader_does_not_grow_worker_pipes(monkeypatch):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import detect_direct

    monkeypatch.setenv("SOLAR_DATALOADER_START", "forkserver")
    context = detect_direct.dataloader_context()
    counts = []
    for _ in range(5):
        loader = DataLoader(
            TensorDataset(torch.arange(4)),
            batch_size=2,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True,
            shuffle=False,
            multiprocessing_context=context,
        )
        for _batch in loader:
            pass
        detect_direct._shutdown_loader(loader)
        counts.append(len(os.listdir("/proc/self/fd")))
    assert counts[-1] == counts[0], counts


def test_forkserver_reads_one_chip_without_starting_cuda(tmp_path: Path, monkeypatch):
    import numpy as np
    import rasterio
    import torch
    from rasterio.transform import from_origin
    from torch.utils.data import DataLoader

    import detect_direct
    from core.inference.tile_dataset import SlidingWindowDataset, list_collate

    cuda_before = torch.cuda.is_initialized()
    monkeypatch.setenv("SOLAR_DATALOADER_START", "forkserver")
    tile = tmp_path / "DBN0001_0_0_geo.tif"
    array = np.full((3, 64, 64), 40, dtype=np.uint8)
    with rasterio.open(
        tile, "w", driver="GTiff", height=64, width=64, count=3,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 0, 0.1, 0.1),
    ) as handle:
        handle.write(array)
    citywide._start_dataloader_forkserver()
    dataset = SlidingWindowDataset([tile], chip_size=64, overlap=0, edge_pad=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=list_collate,
        shuffle=False,
        multiprocessing_context=detect_direct.dataloader_context(),
    )
    try:
        tensors, metas = next(iter(loader))
    finally:
        detect_direct._shutdown_loader(loader)
        dataset.close()
    assert tuple(tensors[0].shape) == (3, 64, 64)
    assert metas[0].chip_index == 0
    assert torch.cuda.is_initialized() is cuda_before
