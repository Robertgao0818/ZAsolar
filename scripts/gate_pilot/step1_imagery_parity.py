#!/usr/bin/env python3
"""Step 1: download the six grids, check the chip grid, rerun Mask R-CNN.

The holdout grids may be downloaded and may be re-detected. This script
does not load a gate model and does not score chips with one.

Exit 0: imagery matches and the four reruns pass the IoU rule.
Exit 2: pre-registered stop (chip grid disagrees and the real grid trips
the step-0 rule, or a rerun fails the 95% / 5% imagery check).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import geopandas as gpd
import pandas as pd

from core import region_registry
from core.eval_matching import iou_matching
from scripts.gate_pilot.chip_windows import chip_windows_from_tifs
from scripts.gate_pilot.common import (
    HOLDOUT_PV_GRID,
    OUT_ROOT,
    REPO_ROOT,
    RESULT_MD,
    STEP0_DIR,
    TRAIN_PV_GRIDS,
    VECTOR_DIR,
    setup_logging,
)
from scripts.gate_pilot.feasibility import g2_minimum_pass
from scripts.gate_pilot.intersect import chips_hit_mask

TILES_ROOT = Path.home() / "zasolar_data" / "tiles" / "cape_town" / "aerial_2025"
PARITY_DIR = OUT_ROOT / "parity"
STEP1_DIR = OUT_ROOT / "step1"
CHECKPOINT = REPO_ROOT / "checkpoints" / "exp_unified_reviewall_A" / "best_model.pth"
POSTPROC = REPO_ROOT / "configs" / "postproc" / "v4_canonical.json"
BOUND_TOL_DEG = 1e-6
CHIP_RATE_RE = re.compile(
    r"\[detect_direct\] (\d+) chips → (\d+) detections "
    r"in ([0-9.]+)s \(([0-9.]+) chips/s"
)


def parity_decision(n_backup: int, n_rerun: int, backup_recall: float, rerun_precision: float) -> dict:
    """Both match rates are from one strict 1:1 assignment at IoU >= 0.5."""
    if n_backup <= 0:
        raise ValueError("backup polygon count must be positive")
    count_rel = abs(n_rerun - n_backup) / n_backup
    passed = backup_recall >= 0.95 and rerun_precision >= 0.95 and count_rel <= 0.05
    return {
        "n_backup": n_backup,
        "n_rerun": n_rerun,
        "backup_match_rate": backup_recall,
        "rerun_match_rate": rerun_precision,
        "count_rel_diff": count_rel,
        "pass": passed,
    }


def parse_detect_rate(log_text: str) -> dict | None:
    found = CHIP_RATE_RE.findall(log_text)
    if not found:
        return None
    chips, detections, seconds, rate = found[-1]
    return {
        "n_chips": int(chips),
        "n_detections": int(detections),
        "seconds": float(seconds),
        "chips_per_s": float(rate),
    }


def _child_env() -> dict:
    env = os.environ.copy()
    env["SOLAR_TILES_ROOT"] = str(TILES_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(cmd: list[str], log_path: Path, log) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("run %s", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=_child_env(), stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{cmd[1]} exited {completed.returncode}\n{tail}")


def _tile_dir(grid_id: str) -> Path:
    source = region_registry.resolve_source_grid_id(grid_id, "cape_town")
    return TILES_ROOT / source


def _tile_paths(grid_id: str) -> list[Path]:
    directory = _tile_dir(grid_id)
    return sorted(directory.glob(f"{grid_id}_*_*_geo.tif"))


def download_grid(grid_id: str, log) -> list[Path]:
    paths = _tile_paths(grid_id)
    if len(paths) == 42:
        log.info("%s already has 42 tiles", grid_id)
        return paths
    _run(
        [sys.executable, "scripts/imagery/download_tiles.py", "--grid-id", grid_id, "--region", "ct"],
        STEP1_DIR / f"download_{grid_id}.log",
        log,
    )
    paths = _tile_paths(grid_id)
    if len(paths) != 42:
        raise RuntimeError(f"{grid_id} has {len(paths)} tiles after download, expected 42")
    return paths


def _compare_chips(grid_id: str, real: gpd.GeoDataFrame) -> dict:
    ref = pd.read_csv(STEP0_DIR / "chip_labels" / f"{grid_id}.csv")
    got = real.drop(columns="geometry")
    merged = ref.merge(got, on=["chip_index", "source_tile"], suffixes=("_ref", "_real"))
    pixel_match = False
    max_delta = None
    if len(merged) == len(ref) == len(got):
        pixel_match = bool(
            (merged["col_off_ref"] == merged["col_off_real"]).all()
            and (merged["row_off_ref"] == merged["row_off_real"]).all()
            and (merged["valid_w_ref"] == merged["valid_w_real"]).all()
            and (merged["valid_h_ref"] == merged["valid_h_real"]).all()
        )
        deltas = []
        for column in ("minx", "miny", "maxx", "maxy"):
            deltas.append((merged[f"{column}_ref"] - merged[f"{column}_real"]).abs().max())
        max_delta = float(max(deltas))
    return {
        "grid_id": grid_id,
        "n_ref": int(len(ref)),
        "n_real": int(len(got)),
        "n_joined": int(len(merged)),
        "pixel_match": pixel_match,
        "max_bound_delta_deg": max_delta,
        "match": bool(pixel_match and max_delta is not None and max_delta <= BOUND_TOL_DEG),
    }


def _real_holdout_rule(chips: gpd.GeoDataFrame) -> dict:
    pre = gpd.read_file(VECTOR_DIR / HOLDOUT_PV_GRID / "predictions_metric.gpkg")
    n_pos = int(chips_hit_mask(chips, pre).sum())
    return g2_minimum_pass(len(chips), n_pos)


def detect_and_finalize(grid_id: str, log) -> dict:
    out = PARITY_DIR / grid_id
    out.mkdir(parents=True, exist_ok=True)
    done = out / "detect.done"
    detect_log = out / "detect_direct.log"
    if not done.exists():
        _run(
            [
                sys.executable, "detect_direct.py",
                "--grid-id", grid_id, "--region", "ct", "--imagery-layer", "aerial_2025",
                "--model-run", "gate_pilot_parity",
                "--model-path", str(CHECKPOINT),
                "--output-dir", str(out),
                "--detections-per-img", "300",
                "--chip-size", "400",
                "--overlap", "0.25",
                "--mask-threshold", "0.3",
                "--detector-score-threshold", "0.05",
                "--raw-mask-storage", "crop",
                "--batch-size", "4",
                "--num-workers", "2",
                "--prefetch-factor", "2",
                "--device", "cuda",
            ],
            detect_log,
            log,
        )
        _run(
            [
                sys.executable, "finalize.py",
                "--input", str(out / "raw_detections.pkl"),
                "--output-dir", str(out),
                "--postproc-config", str(POSTPROC),
                "--merge-mode", "per-detection",
                "--allow-overwrite-canonical",
            ],
            out / "finalize.log",
            log,
        )
        gpkg = out / "predictions_metric.gpkg"
        if not gpkg.exists():
            raise RuntimeError(f"finalize produced no {gpkg}")
        done.write_text("ok\n", encoding="utf-8")
    text = detect_log.read_text(encoding="utf-8", errors="replace") if detect_log.exists() else ""
    rate = parse_detect_rate(text)
    backup = gpd.read_file(VECTOR_DIR / grid_id / "predictions_metric.gpkg").reset_index(drop=True)
    rerun = gpd.read_file(out / "predictions_metric.gpkg").reset_index(drop=True)
    if backup.crs is None or rerun.crs is None or not backup.crs.equals(rerun.crs):
        raise RuntimeError(f"{grid_id} backup/rerun CRS differ: {backup.crs} vs {rerun.crs}")
    matched = iou_matching(backup, rerun, iou_threshold=0.5, merge_preds=False)
    decision = parity_decision(len(backup), len(rerun), matched["recall"], matched["precision"])
    decision["grid_id"] = grid_id
    decision["chips_per_s"] = None if rate is None else rate["chips_per_s"]
    decision["detect_seconds"] = None if rate is None else rate["seconds"]
    decision["mean_matched_iou"] = (
        float(sum(matched["iou_scores"]) / len(matched["iou_scores"])) if matched["iou_scores"] else None
    )
    return decision


def _rewrite_result(chip_rows: list[dict], parity_rows: list[dict], stopped_reason: str | None) -> None:
    text = RESULT_MD.read_text(encoding="utf-8")
    chip_lines = [
        "| 网格 | 像素窗口一致 | 地理范围最大差（度） |",
        "|---|---|---:|",
    ]
    for row in chip_rows:
        delta = "—" if row["max_bound_delta_deg"] is None else f"{row['max_bound_delta_deg']:.3e}"
        chip_lines.append(
            f"| {row['grid_id']} | {'是' if row['match'] else '否'} | {delta} |"
        )
    parity_lines = [
        "| 网格 | 备份多边形 | 重跑多边形 | 备份匹配率 | 重跑匹配率 | 数量相对差 | chip/秒 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in parity_rows:
        rate = "—" if row["chips_per_s"] is None else f"{row['chips_per_s']:.2f}"
        parity_lines.append(
            "| {grid_id} | {n_backup} | {n_rerun} | {backup_match_rate:.1%} | "
            "{rerun_match_rate:.1%} | {count_rel_diff:.1%} | {rate} |".format(rate=rate, **row)
        )
    if stopped_reason:
        body = (
            "步骤 1 提前止损。" + stopped_reason
            + " 检测器没有被接进生产脚本，门也没有训练。"
        )
        status = "步骤 1 提前止损"
    elif parity_rows and all(row["pass"] for row in parity_rows):
        body = (
            "六个格的真瓦片 chip 窗口和步骤 0 的几何载体一致。"
            "四个有光伏的格上，重跑的过滤前多边形和 6 月备份按 IoU≥0.5 一对一匹配，"
            "两个方向都不低于 95%，数量差也不超过 5%。影像没有换到对不上标签的程度，可以进入训练。"
        )
        status = "步骤 1 通过，训练尚未开始"
    else:
        body = "步骤 1 还没跑完。"
        status = "步骤 1 进行中"
    section = (
        "## 影像一致性\n\n"
        + body
        + "\n\n"
        + "\n".join(chip_lines)
        + "\n\n"
        + ("\n".join(parity_lines) + "\n\n" if parity_rows else "")
        + "重跑输出在 `~/zasolar_data/gate_pilot_2026-09/parity/`，没有覆盖普查结果。\n"
    )
    text = re.sub(r"状态：.*", f"状态：{status}", text, count=1)
    text = re.sub(r"## 影像一致性\n\n.*?(?=\n## 单格计数)", section, text, count=1, flags=re.S)
    # The conclusion stays the step-0 sentence until the pilot is finished.
    # Put the step-1 sentence in the imagery section, which is what this rewrite owns.
    RESULT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    log = setup_logging("step1_imagery_parity")
    STEP1_DIR.mkdir(parents=True, exist_ok=True)
    feasibility = json.loads((STEP0_DIR / "feasibility.json").read_text(encoding="utf-8"))
    if feasibility["verdict"] != "pass":
        raise RuntimeError("step 0 verdict is not pass; refusing to download imagery")
    selection = feasibility["selection"]
    grids = [
        TRAIN_PV_GRIDS[0], TRAIN_PV_GRIDS[1], TRAIN_PV_GRIDS[2],
        selection["e_train"], HOLDOUT_PV_GRID, selection["e_test"],
    ]
    chip_rows = []
    real_holdout = None
    for grid_id in grids:
        log.info("download and chip check %s", grid_id)
        paths = download_grid(grid_id, log)
        real = chip_windows_from_tifs(paths, grid_id)
        comparison = _compare_chips(grid_id, real)
        chip_rows.append(comparison)
        log.info("chip compare %s", comparison)
        if grid_id == HOLDOUT_PV_GRID:
            real_holdout = _real_holdout_rule(real)
            log.info("real-tile holdout rule %s", real_holdout)
    (STEP1_DIR / "chip_compare.json").write_text(json.dumps(chip_rows, indent=2) + "\n", encoding="utf-8")
    if real_holdout is None:
        raise RuntimeError("holdout chips were not rebuilt")
    (STEP1_DIR / "holdout_rule_real_tiles.json").write_text(
        json.dumps(real_holdout, indent=2) + "\n", encoding="utf-8"
    )
    chips_ok = all(row["match"] for row in chip_rows)
    if real_holdout["stop"]:
        reason = (
            f"真瓦片上 CPT1238 的 0.99×P_pre = {real_holdout['continuous_min_pass']:.2%}，"
            "高于 20%。步骤 0 的几何载体和真瓦片不是同一套窗口，以真瓦片为准停。"
        )
        _rewrite_result(chip_rows, [], reason)
        log.info(reason)
        return 2
    if not chips_ok:
        reason = "真瓦片的 chip 窗口和步骤 0 的几何载体不一致，先停下，不在错位的窗口上训练。"
        _rewrite_result(chip_rows, [], reason)
        log.info(reason)
        return 2

    parity_rows = []
    for grid_id in (*TRAIN_PV_GRIDS, HOLDOUT_PV_GRID):
        log.info("parity detect %s", grid_id)
        parity_rows.append(detect_and_finalize(grid_id, log))
        log.info("parity %s", parity_rows[-1])
    (STEP1_DIR / "parity.json").write_text(json.dumps(parity_rows, indent=2) + "\n", encoding="utf-8")
    if not all(row["pass"] for row in parity_rows):
        failed = [row["grid_id"] for row in parity_rows if not row["pass"]]
        reason = (
            "重跑和 6 月备份对不上（IoU≥0.5 的一对一匹配率低于 95%，或多边形数量差超过 5%）："
            + ", ".join(failed)
            + "。标签先不要换成这次重跑，等负责人决定。"
        )
        _rewrite_result(chip_rows, parity_rows, reason)
        log.info(reason)
        return 2
    _rewrite_result(chip_rows, parity_rows, None)
    log.info("step 1 passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import logging
        logging.getLogger("step1_imagery_parity").exception("step 1 failed")
        raise
