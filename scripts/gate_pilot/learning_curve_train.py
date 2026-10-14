#!/usr/bin/env python3
"""Train the 2026-09-28 presence-gate learning curve and freeze rule-A taus.

Three arms, nothing else:

- yolo10: YOLOv11n on the nested 10 grids, folds 0–2
- yolo30: YOLOv11n on all 30 grids, folds 0–4
- vit30: frozen DINOv2 ViT-S/14 plus a linear patch head on the same 30 grids

Each training grid is scored by the fold that did not train on it. Tau is the
largest threshold with zero whole-install loss and zero partial-loss area on
filtered installs of at least 30 m² (posthoc rule A, pass-through only). If no
positive tau qualifies, tau is written as 0 and that arm's loss condition
fails. Holdout grids are predicted once, after that arm's tau file exists.

On 2026-09-30 the owner stopped the queue after yolo30 fold 3. This entry
point scores YOLO and the ViT on folds 0–3 only (26 grids), leaves fold 4
untrained, and does not write the official yolo30 or vit30 tau files, the
final refits, or the holdout report.

Does not call step3_train or step5_holdout, does not write holdout labels,
and does not read or write gate_pilot_2026-09/step3.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import shapely

from scripts.gate_pilot.common import RESULT_MD
from scripts.gate_pilot.learning_curve_select import (
    FORCED_TRAIN,
    SELECTION_DIR,
    SELECTION_PATH,
    STRATA,
)

VENV_PYTHON = Path.home() / "venvs" / "yolo_gate" / "bin" / "python"
RUN_MD = _ROOT / "docs" / "experiments" / "2026-09-yolo-gate-pilot" / "RUN-learning-curve-2026-09-28.md"
TRAIN_DIR = SELECTION_DIR / "train"
CHIP_ROOT = SELECTION_DIR / "chips"
VECTOR_ROOT = SELECTION_DIR / "vectors"
LOG_PATH = SELECTION_DIR / "logs" / "train.log"
YOLO_WEIGHTS = Path.home() / "zasolar_data" / "models" / "yolo11n.pt"
DINO_CACHE = Path.home() / "zasolar_data" / "models" / "dinov2_floor" / "hf_cache"
TOKEN_DIR = TRAIN_DIR / "vit_tokens"
N_CHIPS = 2058
N_FOLDS = 5
SMALL_AREA_M2 = 30.0
MISS_TOL_M2 = 1e-6
YOLO_SEED = 20260923
VIT_IMG = 392
VIT_PATCHES = (VIT_IMG // 14) ** 2  # 28 * 28
VIT_DIM = 384
VIT_MEAN = (0.485, 0.456, 0.406)
VIT_STD = (0.229, 0.224, 0.225)
LOCAL_MASKRCNN_CPS = 20.24
EXCLUDED = ("CPT1238", "CPT2172")
HOLDOUT = (
    "CPT3247", "CPT2119", "CPT2714",
    "CPT2308", "CPT2957", "CPT4369",
    "CPT2035", "CPT3577", "CPT3758",
)
# 2026-09-29 handoff table. Derived folds must match this before any training.
HANDOFF_FOLDS = {
    "CPT2932": 0, "CPT3695": 0, "CPT2373": 0, "CPT4235": 0,
    "CPT2216": 0, "CPT2729": 0, "CPT2330": 0, "CPT3865": 0,
    "CPT3468": 1, "CPT3734": 1, "CPT2182": 1, "CPT2776": 1,
    "CPT2332": 1, "CPT2406": 1, "CPT3802": 1,
    "CPT3635": 2, "CPT2174": 2, "CPT2718": 2, "CPT2997": 2,
    "CPT2040": 2, "CPT3021": 2, "CPT2207": 2,
    "CPT3246": 3, "CPT1506": 3, "CPT2390": 3, "CPT2616": 3,
    "CPT3638": 4, "CPT2496": 4, "CPT2167": 4, "CPT3194": 4,
}
TEN_IDS = (
    "CPT2932", "CPT3468", "CPT3635", "CPT2373", "CPT2182",
    "CPT2718", "CPT2216", "CPT2332", "CPT2330", "CPT3802",
)
STATUS_START = "<!--LC_STATUS-->"
STATUS_END = "<!--/LC_STATUS-->"
INTERIM_START = "<!--LC_INTERIM-->"
INTERIM_END = "<!--/LC_INTERIM-->"
DEFERRED_FOLD = 4
RESULT_START = "<!--LC_RESULT-->"
RESULT_END = "<!--/LC_RESULT-->"


@dataclass
class Install:
    grid_id: str
    poly_id: int
    area_m2: float
    scores: np.ndarray
    geoms: tuple
    polygon: object


def stratum_members(selection: dict) -> dict[str, list[str]]:
    """Grids in each stratum, in the selection.json order the handoff rotates."""
    extra = selection["train_extra_by_stratum"]
    members = {
        name: list(FORCED_TRAIN[name]) + list(extra[name])
        for name in STRATA
    }
    flat = [grid_id for name in STRATA for grid_id in members[name]]
    if flat != list(selection["train_30"]):
        raise RuntimeError("stratum order does not reproduce selection.train_30")
    return members


def folds_from_selection(selection: dict) -> dict[str, int]:
    """Fold index is the position inside the stratum, modulo 5."""
    folds: dict[str, int] = {}
    for grids in stratum_members(selection).values():
        for index, grid_id in enumerate(grids):
            folds[grid_id] = index % N_FOLDS
    if folds != HANDOFF_FOLDS:
        raise RuntimeError("derived folds do not match the 2026-09-29 handoff table")
    return folds


def arm_grids(selection: dict, arm: str) -> tuple[str, ...]:
    if arm == "yolo10":
        grids = tuple(selection["train_10"])
        if grids != TEN_IDS:
            raise RuntimeError("train_10 does not match the handoff list")
        return grids
    if arm in ("yolo30", "vit30"):
        return tuple(selection["train_30"])
    raise ValueError(arm)


def active_folds(folds: dict[int, tuple[str, ...]], deferred: int = DEFERRED_FOLD) -> dict[int, tuple[str, ...]]:
    """Folds whose out-of-fold scores exist before the deferred fold is run."""
    if deferred not in folds:
        raise RuntimeError(f"missing deferred fold {deferred}")
    kept = {fold: ids for fold, ids in folds.items() if fold != deferred}
    if set(kept) != {0, 1, 2, 3}:
        raise RuntimeError(f"interim folds are {sorted(kept)}")
    return kept


def deferred_grids(folds: dict[int, tuple[str, ...]], deferred: int = DEFERRED_FOLD) -> tuple[str, ...]:
    return tuple(folds[deferred])


def interim_reading(yolo_met: bool, vit_met: bool) -> str:
    """YOLO versus ViT on the grids scored before the deferred fold."""
    if yolo_met and vit_met:
        return (
            "在已评分的 26 个格子上，YOLO 和冻住的 DINOv2 都让丢失条件成立。"
            "第五折的 4 个格子还没参加，阈值仍可能被它们拉低。"
        )
    if vit_met:
        return (
            "在已评分的 26 个格子上，冻住的 DINOv2 让丢失条件成立，YOLO 没有。"
            "这还不是 30 个格子全部算完的结果。"
        )
    if yolo_met:
        return (
            "在已评分的 26 个格子上，YOLO 让丢失条件成立，冻住的 DINOv2 没有。"
            "这还不是 30 个格子全部算完的结果。"
        )
    return (
        "在已评分的 26 个格子上，YOLO 和冻住的 DINOv2 都没有让丢失条件成立。"
        "第五折还没跑，所以这还不是 30 格对照的最终数。"
    )


def render_interim_section(report: dict) -> str:
    """Chinese note for the 26-grid comparison. Not a holdout verdict."""
    yolo10 = report["yolo10"]
    yolo = report["yolo30_folds_0_to_3"]
    vit = report["vit30_folds_0_to_3"]
    deferred = "、".join(report["deferred_grids"])
    return "\n".join([
        "",
        "## 前四折对照（第五折未跑）",
        "",
        "这一节只比较已经有折外分数的 26 个格子。第五折的 "
        f"{deferred} 仍在训练集里，但没有被留出评分，所以没有约束阈值。",
        "留出的 9 个格子没有评分。全量重训没有做。S1 未测，S2–S5 不写。",
        "",
        f"10 格 YOLO 是完整的那一臂：阈值 {yolo10['tau']:.4f}，"
        f"丢失条件{'成立' if yolo10['loss_met'] else '不成立'}。",
        f"30 格 YOLO 在这 26 个格子上：阈值 {yolo['tau']:.4f}，"
        f"丢失条件{'成立' if yolo['loss_met'] else '不成立'}，"
        f"整块丢失 {yolo['whole_loss_at_tau']}，"
        f"≥30 m² 部分丢失面积 {yolo['partial_area_ge_30_m2_at_tau']:.2f} m²，"
        f"折外放行率 {yolo['oof_pass_rate']:.3f}。",
        f"冻住的 DINOv2 在同样的 26 个格子上：阈值 {vit['tau']:.4f}，"
        f"丢失条件{'成立' if vit['loss_met'] else '不成立'}，"
        f"整块丢失 {vit['whole_loss_at_tau']}，"
        f"≥30 m² 部分丢失面积 {vit['partial_area_ge_30_m2_at_tau']:.2f} m²，"
        f"折外放行率 {vit['oof_pass_rate']:.3f}。",
        "",
        report["reading"],
        "数字说的是折外分数，不是留出集。早停用的验证集仍是被留出的那一折。",
        "ViT 没有做翻转，因为特征只抽了一次。",
        "",
    ])


def folds_for_arm(selection: dict, arm: str) -> dict[int, tuple[str, ...]]:
    fold_of = folds_from_selection(selection)
    grouped: dict[int, list[str]] = {fold: [] for fold in range(N_FOLDS)}
    for grid_id in arm_grids(selection, arm):
        grouped[fold_of[grid_id]].append(grid_id)
    present = {fold: tuple(grids) for fold, grids in grouped.items() if grids}
    if arm == "yolo10" and set(present) != {0, 1, 2}:
        raise RuntimeError(f"yolo10 folds are {sorted(present)}, expected 0, 1, 2")
    if arm in ("yolo30", "vit30") and set(present) != set(range(N_FOLDS)):
        raise RuntimeError(f"{arm} is missing a fold: {sorted(present)}")
    return present


def binding_tau_for_polygon(scores, geoms, poly, area_m2: float, tol: float = MISS_TOL_M2) -> float:
    """Largest score threshold that keeps this one filtered install under rule A.

    Installs under 30 m² only have to touch one passing chip. Larger installs
    must also keep partial-loss area at or under ``tol``. When the chip union
    never covers the polygon, rule A still counts zero partial loss once every
    intersecting chip passes, so the binding score is the lowest chip score.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("filtered install has no intersecting chip")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("chip scores must be finite and >= 0")
    if area_m2 < SMALL_AREA_M2:
        return float(scores.max())
    order = np.argsort(-scores, kind="mergesort")
    geoms = list(geoms)
    polygon = shapely.make_valid(poly)
    accumulated = []
    cursor = 0
    while cursor < len(order):
        score = float(scores[order[cursor]])
        while cursor < len(order) and float(scores[order[cursor]]) == score:
            accumulated.append(shapely.make_valid(geoms[int(order[cursor])]))
            cursor += 1
        miss = polygon.difference(shapely.union_all(accumulated)).area
        if miss <= tol:
            return score
    return float(scores.min())


def losses_for_polygon(scores, geoms, poly, area_m2: float, tau: float, tol: float = MISS_TOL_M2) -> tuple[int, float]:
    """Whole-loss flag, and partial-loss area counted only for installs >= 30 m²."""
    scores = np.asarray(scores, dtype=float)
    passed = scores >= tau
    if not bool(passed.any()):
        partial = float(area_m2) if area_m2 >= SMALL_AREA_M2 else 0.0
        return 1, partial
    if area_m2 < SMALL_AREA_M2 or bool(passed.all()):
        return 0, 0.0
    keep = [shapely.make_valid(geoms[index]) for index, flag in enumerate(passed) if flag]
    miss = shapely.make_valid(poly).difference(shapely.union_all(keep)).area
    if miss <= tol:
        return 0, 0.0
    return 0, float(miss)


def select_tau_from_bindings(bindings: list[float]) -> dict:
    """Min binding score is the max legal tau. Non-positive bindings fail the arm."""
    if not bindings:
        return {"tau": 1.0, "loss_met": True, "n_polygons": 0}
    tau = float(min(bindings))
    if (not math.isfinite(tau)) or tau <= 0.0:
        return {"tau": 0.0, "loss_met": False, "n_polygons": len(bindings), "binding_min": tau}
    return {"tau": tau, "loss_met": True, "n_polygons": len(bindings), "binding_min": tau}


def s4_reading(loss_met_10: bool, loss_met_30: bool) -> str:
    """How to read the 10-grid versus 30-grid out-of-fold loss condition.

    Both zeros means the nested 10 grids already met the rule. It does not
    mean training size never mattered: the earlier 179-chip fold is a
    different sample.
    """
    if loss_met_10 and loss_met_30:
        return (
            "两臂都找到了正的折外阈值，折外整块丢失都是 0，从 10 格加到 30 格没有下降。"
            "按交接的读法，分层后的这 10 格已经够，不是说数据量从来无关。"
        )
    if (not loss_met_10) and loss_met_30:
        return "10 格找不到正阈值，30 格找得到。折外丢失条件从失败变成成立，这一次加数据有用。"
    if (not loss_met_10) and (not loss_met_30):
        return "10 格和 30 格都找不到正阈值。加到 30 格没有把丢失条件救回来。"
    return (
        "10 格找得到正阈值，30 格找不到。30 格要同时守住更多安装，约束更严，"
        "不能把这说成加数据降低了丢失。"
    )


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _log(message: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + message
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _replace_span(path: Path, start: str, end: str, text: str) -> None:
    body = path.read_text(encoding="utf-8")
    if start not in body or end not in body:
        raise RuntimeError(f"{path} is missing the marker {start}")
    left = body.index(start)
    right = body.index(end, left) + len(end)
    path.write_text(body[:left] + start + text + end + body[right:], encoding="utf-8")


def set_run_status(text: str, title: str | None = None) -> None:
    _replace_span(RUN_MD, STATUS_START, STATUS_END, text)
    if title is None:
        return
    lines = RUN_MD.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise RuntimeError("RUN doc has no title")
    lines[0] = "# " + title
    RUN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_selection() -> dict:
    selection = _read_json(SELECTION_PATH)
    if selection.get("schema") != "learning_curve_selection_v1":
        raise RuntimeError("selection.json schema is not the locked draw")
    if int(selection["seed"]) != 20260928:
        raise RuntimeError("selection seed is not 20260928")
    if float(selection["empty_chip_keep_ratio"]) != 1.0:
        raise RuntimeError("empty chips must be kept at ratio 1")
    if set(selection["excluded_everywhere"]) != set(EXCLUDED):
        raise RuntimeError("excluded grids changed")
    if list(selection["holdout"]) != list(HOLDOUT):
        raise RuntimeError("holdout list changed")
    if set(selection["train_30"]) & (set(HOLDOUT) | set(EXCLUDED)):
        raise RuntimeError("a train grid is a holdout or an excluded grid")
    folds_from_selection(selection)
    return selection


def preflight() -> None:
    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        raise RuntimeError(f"refusing to train with {sys.executable}; use {VENV_PYTHON}")
    chips_done = (SELECTION_DIR / "chips.done").read_text(encoding="utf-8").strip()
    fetch_done = (SELECTION_DIR / "fetch.done").read_text(encoding="utf-8").strip()
    if chips_done != "ok 39" or fetch_done != "ok 39":
        raise RuntimeError(f"refusing to train: fetch={fetch_done!r} chips={chips_done!r}")
    for grid_id in HOLDOUT:
        label_dir = CHIP_ROOT / grid_id / "labels"
        if label_dir.exists():
            raise RuntimeError(f"holdout {grid_id} has a labels directory")
        images = list((CHIP_ROOT / grid_id / "images").glob("*.png"))
        if len(images) != N_CHIPS:
            raise RuntimeError(f"holdout {grid_id} has {len(images)} pngs")
    for grid_id in HANDOFF_FOLDS:
        if len(list((CHIP_ROOT / grid_id / "labels").glob("*.txt"))) != N_CHIPS:
            raise RuntimeError(f"{grid_id} does not have {N_CHIPS} label files")
    _log("PREFLIGHT ok chips.done=ok 39 holdout labels absent")


def _gpu_is_free() -> None:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    )
    busy = []
    for line in output.splitlines():
        if not line.strip() or "kwin" in line:
            continue
        busy.append(line.strip())
    if busy:
        raise RuntimeError("GPU is busy: " + " | ".join(busy))


def _assert_train_grids(grid_ids) -> None:
    bad = set(grid_ids) & (set(HOLDOUT) | set(EXCLUDED))
    if bad:
        raise RuntimeError(f"training path contains a held-out grid: {sorted(bad)}")


def _scores_are_complete(path: Path, grid_ids: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    if list(frame.columns) != ["grid_id", "chip_index", "score"]:
        return False
    if set(frame["grid_id"]) != set(grid_ids):
        return False
    if frame.duplicated(["grid_id", "chip_index"]).any():
        return False
    for grid_id, part in frame.groupby("grid_id"):
        indexes = set(part["chip_index"].astype(int))
        if indexes != set(range(N_CHIPS)):
            return False
    return bool(np.isfinite(frame["score"]).all() and (frame["score"] >= 0).all())


def _write_scores(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows, columns=["grid_id", "chip_index", "score"])
    frame = frame.sort_values(["grid_id", "chip_index"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _link_split(grid_ids: tuple[str, ...], split: str, root: Path) -> None:
    _assert_train_grids(grid_ids)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for grid_id in grid_ids:
        for index in range(N_CHIPS):
            stem = f"{grid_id}_{index:05d}"
            image_src = CHIP_ROOT / grid_id / "images" / f"{index:05d}.png"
            label_src = CHIP_ROOT / grid_id / "labels" / f"{index:05d}.txt"
            if not image_src.exists() or not label_src.exists():
                raise RuntimeError(f"missing chip or label {grid_id} {index}")
            image_dest = image_dir / f"{stem}.png"
            label_dest = label_dir / f"{stem}.txt"
            if not image_dest.exists():
                image_dest.symlink_to(image_src.resolve())
            if not label_dest.exists():
                label_dest.symlink_to(label_src.resolve())
    for directory in (image_dir, label_dir):
        for path in directory.iterdir():
            for grid_id in list(HOLDOUT) + list(EXCLUDED):
                if grid_id in path.name:
                    raise RuntimeError(f"held-out id leaked into {path}")


def _write_yaml(root: Path) -> Path:
    path = root / "data.yaml"
    path.write_text(
        "\n".join([
            f"path: {root}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: pv",
            "",
        ]),
        encoding="utf-8",
    )
    return path


def _ensure_yolo11n() -> Path:
    YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = Path(attempt_download_asset(str(YOLO_WEIGHTS)))
    if downloaded.resolve() != YOLO_WEIGHTS.resolve() or YOLO_WEIGHTS.stat().st_size < 1_000_000:
        raise RuntimeError(f"yolo11n.pt did not land at {YOLO_WEIGHTS} ({downloaded})")
    return YOLO_WEIGHTS


def _yolo_config(config: dict) -> dict:
    return config["yolo"]


def _train_yolo(arm: str, fold_name: str, data_root: Path, train_ids, val_ids, config: dict) -> Path:
    run_dir = TRAIN_DIR / "runs" / f"{arm}_{fold_name}"
    weights = run_dir / "weights" / "best.pt"
    sentinel = run_dir / "train.done"
    if sentinel.exists() and weights.exists():
        _log(f"REUSE {weights}")
        return weights
    if run_dir.exists():
        shutil.rmtree(run_dir)
    _link_split(tuple(train_ids), "train", data_root)
    _link_split(tuple(val_ids), "val", data_root)
    yaml_path = _write_yaml(data_root)
    yolo_cfg = _yolo_config(config)
    base = _ensure_yolo11n()
    from ultralytics import YOLO

    model = YOLO(str(base))
    epoch_mark = {"t": time.perf_counter()}

    def on_fit_epoch_end(trainer) -> None:
        now = time.perf_counter()
        _log(
            f"EPOCH arm={arm} fold={fold_name} epoch={int(trainer.epoch) + 1} "
            f"seconds={now - epoch_mark['t']:.1f}"
        )
        epoch_mark["t"] = now

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    started = time.perf_counter()
    model.train(
        data=str(yaml_path),
        epochs=int(yolo_cfg["epochs"]),
        patience=int(yolo_cfg["patience"]),
        batch=int(yolo_cfg["batch"]),
        imgsz=int(yolo_cfg["imgsz"]),
        flipud=float(yolo_cfg["flipud"]),
        fliplr=float(yolo_cfg["fliplr"]),
        device=0,
        seed=int(yolo_cfg["seed"]),
        project=str(TRAIN_DIR / "runs"),
        name=f"{arm}_{fold_name}",
        exist_ok=True,
        workers=int(yolo_cfg["workers"]),
        single_cls=True,
        cache=False,
        plots=False,
    )
    if not weights.exists():
        raise RuntimeError(f"{arm} {fold_name} did not write {weights}")
    sentinel.write_text("ok\n", encoding="utf-8")
    _log(f"TRAIN_OK arm={arm} fold={fold_name} seconds={time.perf_counter() - started:.1f}")
    del model
    import torch

    torch.cuda.empty_cache()
    return weights


def _parse_stem(stem: str, default_grid: str | None) -> tuple[str, int]:
    if "_" in stem:
        grid_id, index_s = stem.rsplit("_", 1)
        if grid_id.startswith("CPT") and index_s.isdigit():
            return grid_id, int(index_s)
    if default_grid is not None and stem.isdigit():
        return default_grid, int(stem)
    raise RuntimeError(f"cannot parse chip name {stem}")


def _score_yolo_dir(weights: Path, image_dir: Path, config: dict, default_grid: str | None) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    yolo_cfg = _yolo_config(config)
    lookup = {}
    for path in sorted(image_dir.glob("*.png")):
        grid_id, chip_index = _parse_stem(path.stem, default_grid)
        lookup[str(path)] = (grid_id, chip_index)
        lookup[str(path.resolve())] = (grid_id, chip_index)
    rows = []
    seen = set()
    for result in model.predict(
        source=str(image_dir),
        conf=float(yolo_cfg["conf"]),
        imgsz=int(yolo_cfg["imgsz"]),
        stream=True,
        verbose=False,
        device=0,
    ):
        raw = str(Path(result.path))
        resolved = str(Path(result.path).resolve())
        if raw in lookup:
            grid_id, chip_index = lookup[raw]
        elif resolved in lookup:
            grid_id, chip_index = lookup[resolved]
        else:
            grid_id, chip_index = _parse_stem(Path(result.path).stem, default_grid)
        boxes = result.boxes
        score = 0.0
        if boxes is not None and len(boxes):
            score = float(boxes.conf.max().item())
        rows.append({"grid_id": grid_id, "chip_index": chip_index, "score": score})
        seen.add((grid_id, chip_index))
    del model
    import torch

    torch.cuda.empty_cache()
    if len(seen) != len(rows):
        raise RuntimeError(f"duplicate predictions under {image_dir}")
    return rows


def _run_yolo_fold(selection: dict, arm: str, fold: int, val_ids: tuple[str, ...], config: dict) -> None:
    train_ids = tuple(grid_id for grid_id in arm_grids(selection, arm) if grid_id not in val_ids)
    if not train_ids or not val_ids:
        raise RuntimeError(f"{arm} fold {fold} has an empty split")
    _assert_train_grids(train_ids)
    _assert_train_grids(val_ids)
    score_path = TRAIN_DIR / "scores" / f"{arm}_fold{fold}.csv"
    data_root = TRAIN_DIR / "data" / arm / f"fold{fold}"
    started = time.perf_counter()
    weights = _train_yolo(arm, f"fold{fold}", data_root, train_ids, val_ids, config)
    if not _scores_are_complete(score_path, val_ids):
        score_started = time.perf_counter()
        rows = _score_yolo_dir(weights, data_root / "images" / "val", config, None)
        _write_scores(score_path, rows)
        if not _scores_are_complete(score_path, val_ids):
            raise RuntimeError(f"{score_path} failed validation")
        _log(f"SCORE_OK arm={arm} fold={fold} seconds={time.perf_counter() - score_started:.1f}")
    _log(f"FOLD_DONE arm={arm} fold={fold} seconds={time.perf_counter() - started:.1f}")


def _run_yolo_final(selection: dict, arm: str, config: dict) -> Path:
    grids = arm_grids(selection, arm)
    # Early stopping needs a val set. The first two training grids are also
    # in train. Tau does not use this fit.
    val_ids = grids[:2]
    return _train_yolo(arm, "final", TRAIN_DIR / "data" / arm / "final", grids, val_ids, config)


def _score_array(frame: pd.DataFrame, grid_id: str) -> np.ndarray:
    part = frame[frame["grid_id"] == grid_id]
    scores = np.full(N_CHIPS, np.nan)
    scores[part["chip_index"].astype(int).to_numpy()] = part["score"].to_numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{grid_id} is missing scores")
    return scores


_GRID_CACHE: dict[str, list[tuple]] = {}


def _static_installs(grid_id: str) -> list[tuple]:
    """Chip geometry and filtered polygons for one grid, computed once per process."""
    if grid_id in _GRID_CACHE:
        return _GRID_CACHE[grid_id]
    from scripts.gate_pilot.chip_windows import chip_windows_from_tifs
    from scripts.gate_pilot.step2_chips_and_labels import _read_layer, _tile_paths

    chips = chip_windows_from_tifs(_tile_paths(grid_id), grid_id).to_crs(32734)
    if list(chips["chip_index"]) != list(range(N_CHIPS)):
        raise RuntimeError(f"{grid_id} chip windows are not indexed 0..{N_CHIPS - 1}")
    chip_geoms = [shapely.make_valid(geom) for geom in chips.geometry]
    post = _read_layer(VECTOR_ROOT / grid_id / "predictions_metric_cls_filtered.gpkg")
    hits = pd.read_csv(CHIP_ROOT / grid_id / "intersections.csv")
    hits = hits[hits["layer"] == "post"]
    records = []
    if not hits.empty:
        missing = set(hits["poly_id"].astype(int)) - set(post["poly_id"].astype(int))
        if missing:
            raise RuntimeError(f"{grid_id} intersection poly_id missing from the filtered gpkg")
        posted = post.set_index("poly_id")
        for poly_id, group in hits.groupby("poly_id"):
            poly_id = int(poly_id)
            indexes = group["chip_index"].astype(int).to_numpy()
            if (indexes < 0).any() or (indexes >= N_CHIPS).any():
                raise RuntimeError(f"{grid_id} poly {poly_id} chip index out of range")
            area = float(posted.geometry.loc[poly_id].area)
            recorded = float(group["poly_area_m2"].iloc[0])
            if not math.isclose(area, recorded, rel_tol=1e-3, abs_tol=1e-3):
                raise RuntimeError(f"{grid_id} poly {poly_id} area {area} != intersection {recorded}")
            records.append((
                poly_id,
                area,
                shapely.make_valid(posted.geometry.loc[poly_id]),
                indexes,
                tuple(chip_geoms[int(index)] for index in indexes),
            ))
    _GRID_CACHE[grid_id] = records
    return records


def _load_installs(grid_id: str, scores: np.ndarray) -> list[Install]:
    if len(scores) != N_CHIPS:
        raise RuntimeError(f"{grid_id} score length {len(scores)}")
    installs = []
    for poly_id, area, polygon, indexes, geoms in _static_installs(grid_id):
        installs.append(Install(
            grid_id=grid_id,
            poly_id=poly_id,
            area_m2=area,
            scores=scores[indexes],
            geoms=geoms,
            polygon=polygon,
        ))
    return installs


def evaluate_scores(
    arm: str,
    score_frame: pd.DataFrame,
    grid_ids: tuple[str, ...],
    dest: Path | None = None,
) -> dict:
    """Freeze tau from rule A. Recompute losses at tau and just above it."""
    paired = []
    for grid_id in grid_ids:
        _log(f"TAU_GRID arm={arm} grid={grid_id}")
        for install in _load_installs(grid_id, _score_array(score_frame, grid_id)):
            binding = binding_tau_for_polygon(
                install.scores, install.geoms, install.polygon, install.area_m2,
            )
            paired.append((install, binding))
    chosen = select_tau_from_bindings([binding for _, binding in paired])
    tau = float(chosen["tau"])
    installs = [install for install, _binding in paired]
    whole, partial = _losses_at(installs, tau)
    if whole != 0 or partial > 0.0:
        raise RuntimeError(f"{arm} tau {tau} does not clear rule A ({whole}, {partial})")
    if chosen["loss_met"] and chosen["n_polygons"]:
        higher = math.nextafter(tau, math.inf)
        whole_hi, partial_hi = _losses_at(installs, higher)
        if whole_hi == 0 and partial_hi == 0.0:
            raise RuntimeError(f"{arm} tau {tau} is not maximal")
    if not chosen["loss_met"] and chosen["n_polygons"]:
        positive = score_frame.loc[score_frame["score"] > 0, "score"]
        probe = float(positive.min()) if len(positive) else 1e-6
        whole_pos, partial_pos = _losses_at(installs, probe)
        if whole_pos == 0 and partial_pos == 0.0:
            raise RuntimeError(f"{arm} has a working positive tau but was marked failed")
    passed = score_frame["score"].to_numpy() >= tau
    limiters = [
        {"grid_id": install.grid_id, "poly_id": install.poly_id, "area_m2": install.area_m2, "binding": binding}
        for install, binding in paired
        if chosen["n_polygons"] and binding == float(chosen.get("binding_min", tau))
    ]
    per_grid = {}
    for grid_id, part in score_frame.groupby("grid_id"):
        per_grid[str(grid_id)] = float((part["score"].to_numpy() >= tau).mean())
    payload = {
        "arm": arm,
        "rule": "A",
        "tau": tau,
        "loss_met": bool(chosen["loss_met"]),
        "n_polygons": int(chosen["n_polygons"]),
        "whole_loss_at_tau": int(whole),
        "partial_area_ge_30_m2_at_tau": float(partial),
        "oof_pass_rate": float(passed.mean()),
        "pass_rate_by_grid": per_grid,
        "limiters": limiters[:10],
        "conf_floor": 0.001 if arm.startswith("yolo") else 0.0,
        "note": (
            "Scores below the YOLO confidence floor are stored as 0. "
            "Tau 0 means no positive threshold met rule A; the arm's loss condition fails."
        ),
    }
    path = dest if dest is not None else TRAIN_DIR / f"frozen_tau_{arm}.json"
    label = "TAU_INTERIM" if dest is not None else "TAU_FROZEN"
    _json_dump(path, payload)
    _log(
        f"{label} arm={arm} path={path.name} tau={tau:.6f} "
        f"loss_met={payload['loss_met']} pass_rate={payload['oof_pass_rate']:.4f}"
    )
    return payload


def _losses_at(installs: list[Install], tau: float) -> tuple[int, float]:
    whole = 0
    partial = 0.0
    for install in installs:
        lost, miss = losses_for_polygon(
            install.scores, install.geoms, install.polygon, install.area_m2, tau,
        )
        whole += lost
        partial += miss
    return whole, partial


def _concat_oof(arm: str, folds: dict[int, tuple[str, ...]]) -> pd.DataFrame:
    frames = []
    expected = []
    for fold, grid_ids in folds.items():
        path = TRAIN_DIR / "scores" / f"{arm}_fold{fold}.csv"
        if not _scores_are_complete(path, grid_ids):
            raise RuntimeError(f"missing out-of-fold scores {path}")
        frames.append(pd.read_csv(path))
        expected.extend(grid_ids)
    frame = pd.concat(frames, ignore_index=True)
    if set(frame["grid_id"]) != set(expected) or frame.duplicated(["grid_id", "chip_index"]).any():
        raise RuntimeError(f"{arm} out-of-fold scores do not cover each training grid once")
    return frame


def _probe_vit_batch() -> int:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import timm

    os.environ["HF_HUB_CACHE"] = str(DINO_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=VIT_IMG,
    ).eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = torch.nn.Linear(model.num_features, 1).cuda()
    image_path = next((CHIP_ROOT / "CPT2332" / "images").glob("*.png"))
    array = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    base = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
    base = F.interpolate(base, size=(VIT_IMG, VIT_IMG), mode="bilinear", align_corners=False).squeeze(0).cuda()
    chosen = None
    peak = None
    for batch in (32, 16, 8, 4):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch_x = base.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
            with torch.no_grad():
                tokens = model.forward_features(batch_x)[:, 1:, :]
            loss = head(tokens).squeeze(-1).max(dim=1).values.sum()
            loss.backward()
            torch.cuda.synchronize()
            chosen = batch
            peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            _log(f"VIT_BATCH {batch} fits peak_mib={peak}")
            break
        except RuntimeError as exc:
            _log(f"VIT_BATCH {batch} did not fit: {exc.__class__.__name__}")
            torch.cuda.empty_cache()
    del model, head
    torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError("ViT patch head did not fit at batch 4")
    return chosen if peak is None else chosen


def _write_config() -> dict:
    path = TRAIN_DIR / "train_config.json"
    if path.exists():
        config = _read_json(path)
        _log(f"REUSE train config yolo_batch={config['yolo']['batch']} vit_batch={config['vit']['forward_batch']}")
        return config
    import torch
    import timm
    import ultralytics

    vit_batch = _probe_vit_batch()
    config = {
        "schema": "learning_curve_train_v1",
        "yolo": {
            "model": "yolo11n.pt",
            "weights": str(YOLO_WEIGHTS),
            "imgsz": 416,
            "epochs": 100,
            "patience": 20,
            "batch": 32,
            "flipud": 0.5,
            "fliplr": 0.5,
            "workers": 2,
            "single_cls": True,
            "seed": YOLO_SEED,
            "conf": 0.001,
            "cache": False,
            "device": 0,
            "plots": False,
            "oof_val": "held-out fold, same protocol as the 2026-09-23 pilot",
            "final_val": "first two training grids of the arm; tau does not use the final fit",
        },
        "vit": {
            "timm_name": "vit_small_patch14_dinov2.lvd142m",
            "pretrained_cache": str(DINO_CACHE),
            "img_size": VIT_IMG,
            "resize": "bilinear",
            "align_corners": False,
            "crop": False,
            "mean": list(VIT_MEAN),
            "std": list(VIT_STD),
            "forward_batch": vit_batch,
            "head_batch": 256,
            "epochs": 10,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "AdamW",
            "label": "y_pre",
            "score": "sigmoid(max patch logit)",
            "drop_cls": True,
            "augmentation": "none",
            "seed": YOLO_SEED,
            "workers": 2,
            "note": "Patch tokens are cached once. Flips are not applied, because a flip would require another backbone pass.",
        },
        "tau": {
            "rule": "A",
            "whole_loss": 0,
            "partial_area_m2_on_installs_ge_30": 0,
            "area_m2": SMALL_AREA_M2,
            "miss_tolerance_m2": MISS_TOL_M2,
        },
        "empty_chip_keep_ratio": 1.0,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "timm": timm.__version__,
    }
    _json_dump(path, config)
    _log(f"WROTE train_config vit_forward_batch={vit_batch}")
    return config


class _PngChips:
    def __init__(self, grid_id: str):
        import torch

        self.grid_id = grid_id
        self.mean = torch.tensor(VIT_MEAN).view(3, 1, 1)
        self.std = torch.tensor(VIT_STD).view(3, 1, 1)
        self.paths = [CHIP_ROOT / grid_id / "images" / f"{index:05d}.png" for index in range(N_CHIPS)]
        missing = [path for path in self.paths if not path.exists()]
        if missing:
            raise RuntimeError(f"{grid_id} missing {len(missing)} pngs")

    def __len__(self) -> int:
        return N_CHIPS

    def __getitem__(self, index: int):
        import torch
        import torch.nn.functional as F
        from PIL import Image

        array = np.asarray(Image.open(self.paths[index]).convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (400, 400, 3):
            raise RuntimeError(f"{self.paths[index]} shape {array.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(VIT_IMG, VIT_IMG), mode="bilinear", align_corners=False).squeeze(0)
        tensor = (tensor - self.mean) / self.std
        return tensor, index


def _load_backbone():
    import timm

    os.environ["HF_HUB_CACHE"] = str(DINO_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=VIT_IMG,
    ).eval().cuda()
    if int(model.num_features) != VIT_DIM:
        raise RuntimeError(f"expected embed dim {VIT_DIM}, got {model.num_features}")
    config = timm.data.resolve_data_config({}, model=model)
    if tuple(config["mean"]) != VIT_MEAN or tuple(config["std"]) != VIT_STD:
        raise RuntimeError(f"DINOv2 normalization changed: {config['mean']} {config['std']}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _token_path(grid_id: str) -> tuple[Path, Path]:
    return TOKEN_DIR / f"{grid_id}.f16", TOKEN_DIR / f"{grid_id}.json"


def _tokens_ready(grid_id: str) -> bool:
    data_path, sidecar = _token_path(grid_id)
    if not sidecar.exists() or not data_path.exists():
        return False
    meta = _read_json(sidecar)
    expected = N_CHIPS * VIT_PATCHES * VIT_DIM * 2
    return bool(meta.get("complete") and data_path.stat().st_size == expected and meta.get("img_size") == VIT_IMG)


def _open_tokens(grid_id: str, mode: str):
    path, _sidecar = _token_path(grid_id)
    return np.memmap(path, dtype=np.float16, mode=mode, shape=(N_CHIPS, VIT_PATCHES, VIT_DIM))


def _extract_tokens(grid_ids: tuple[str, ...], batch_size: int) -> None:
    _assert_train_grids(grid_ids)
    import torch
    from torch.utils.data import DataLoader

    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    need_bytes = len(grid_ids) * N_CHIPS * VIT_PATCHES * VIT_DIM * 2
    free = shutil.disk_usage(TOKEN_DIR).free
    pending = [grid_id for grid_id in grid_ids if not _tokens_ready(grid_id)]
    if pending and free < need_bytes + 2 * 1024 ** 3:
        raise RuntimeError(
            f"ViT token cache needs {need_bytes / 1e9:.1f} GB plus 2 GB, free {free / 1e9:.1f} GB"
        )
    if not pending:
        _log("VIT_TOKENS reuse cached patch tokens")
        return
    model = _load_backbone()
    for grid_id in pending:
        data_path, sidecar = _token_path(grid_id)
        if data_path.exists():
            data_path.unlink()
        tokens = _open_tokens(grid_id, "w+")
        loader = DataLoader(
            _PngChips(grid_id), batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        started = time.perf_counter()
        seen = 0
        for images, indexes in loader:
            with torch.inference_mode():
                features = model.forward_features(images.cuda(non_blocking=True))
            if features.shape[1] != VIT_PATCHES + 1:
                raise RuntimeError(f"expected {VIT_PATCHES + 1} tokens, got {tuple(features.shape)}")
            patch = features[:, 1:, :].half().cpu().numpy()
            tokens[indexes.numpy()] = patch
            seen += len(indexes)
        if seen != N_CHIPS:
            raise RuntimeError(f"{grid_id} extracted {seen} chips")
        tokens.flush()
        del tokens
        _json_dump(sidecar, {"complete": True, "img_size": VIT_IMG, "grid_id": grid_id, "n": N_CHIPS})
        _log(f"VIT_TOKENS grid={grid_id} seconds={time.perf_counter() - started:.1f}")
    del model
    torch.cuda.empty_cache()


def _labels(grid_id: str) -> np.ndarray:
    manifest = pd.read_csv(CHIP_ROOT / grid_id / "manifest.csv", usecols=["chip_index", "y_pre"])
    if list(manifest["chip_index"]) != list(range(N_CHIPS)):
        raise RuntimeError(f"{grid_id} manifest is not in chip order")
    labels = manifest["y_pre"].astype(bool).to_numpy()
    return labels.astype(np.float32)


class _TokenChips:
    def __init__(self, grid_ids: tuple[str, ...]):
        self.rows = []
        self.maps = {}
        self.labels = {}
        for grid_id in grid_ids:
            if not _tokens_ready(grid_id):
                raise RuntimeError(f"missing tokens for {grid_id}")
            self.maps[grid_id] = _open_tokens(grid_id, "r")
            self.labels[grid_id] = _labels(grid_id)
            self.rows.extend((grid_id, index) for index in range(N_CHIPS))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        import torch

        grid_id, index = self.rows[item]
        tokens = torch.from_numpy(np.array(self.maps[grid_id][index], dtype=np.float32, copy=True))
        return tokens, float(self.labels[grid_id][index]), grid_id, index


def _train_head(name: str, train_ids: tuple[str, ...], config: dict) -> Path:
    import torch
    from torch.utils.data import DataLoader

    run_dir = TRAIN_DIR / "runs" / name
    weights = run_dir / "head.pt"
    sentinel = run_dir / "train.done"
    if sentinel.exists() and weights.exists():
        _log(f"REUSE {weights}")
        return weights
    vit_cfg = config["vit"]
    labels = np.concatenate([_labels(grid_id) for grid_id in train_ids])
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        raise RuntimeError(f"{name} positives={n_pos} negatives={n_neg}")
    pos_weight = n_neg / n_pos
    dataset = _TokenChips(train_ids)
    generator = torch.Generator()
    generator.manual_seed(int(vit_cfg["seed"]) + len(train_ids))
    loader = DataLoader(
        dataset, batch_size=int(vit_cfg["head_batch"]), shuffle=True,
        num_workers=0, generator=generator,
    )
    torch.manual_seed(int(vit_cfg["seed"]))
    head = torch.nn.Linear(VIT_DIM, 1).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(vit_cfg["lr"]), weight_decay=float(vit_cfg["weight_decay"]))
    weight = torch.tensor([pos_weight], device="cuda")
    best_loss = math.inf
    best_state = None
    run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(vit_cfg["epochs"]) + 1):
        total = 0.0
        seen = 0
        head.train()
        for tokens, target, _grid, _index in loader:
            tokens = tokens.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            chip_logit = head(tokens).squeeze(-1).max(dim=1).values
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                chip_logit, target, pos_weight=weight.expand_as(chip_logit),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} loss became non-finite at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        mean_loss = total / seen
        _log(f"VIT_EPOCH name={name} epoch={epoch} loss={mean_loss:.4f} pos_weight={pos_weight:.2f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"{name} did not train")
    torch.save({
        "state_dict": best_state,
        "pos_weight": pos_weight,
        "train_loss": best_loss,
        "label": "y_pre",
        "n_train_grids": len(train_ids),
    }, weights)
    sentinel.write_text("ok\n", encoding="utf-8")
    return weights


def _score_tokens(weights: Path, grid_ids: tuple[str, ...]) -> list[dict]:
    import torch

    _assert_train_grids(grid_ids)
    checkpoint = torch.load(weights, map_location="cuda", weights_only=False)
    head = torch.nn.Linear(VIT_DIM, 1).cuda()
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    rows = []
    with torch.inference_mode():
        for grid_id in grid_ids:
            tokens = _open_tokens(grid_id, "r")
            for start in range(0, N_CHIPS, 64):
                stop = min(N_CHIPS, start + 64)
                batch = torch.from_numpy(np.array(tokens[start:stop], dtype=np.float32, copy=True)).cuda()
                chip_logit = head(batch).squeeze(-1).max(dim=1).values
                scores = torch.sigmoid(chip_logit).detach().cpu().numpy()
                for offset, score in enumerate(scores):
                    rows.append({"grid_id": grid_id, "chip_index": start + offset, "score": float(score)})
    return rows


def _run_vit_fold(selection: dict, fold: int, val_ids: tuple[str, ...], config: dict) -> None:
    train_ids = tuple(grid_id for grid_id in arm_grids(selection, "vit30") if grid_id not in val_ids)
    score_path = TRAIN_DIR / "scores" / f"vit30_fold{fold}.csv"
    started = time.perf_counter()
    weights = _train_head(f"vit30_fold{fold}", train_ids, config)
    if not _scores_are_complete(score_path, val_ids):
        _write_scores(score_path, _score_tokens(weights, val_ids))
        if not _scores_are_complete(score_path, val_ids):
            raise RuntimeError(f"{score_path} failed validation")
    _log(f"FOLD_DONE arm=vit30 fold={fold} seconds={time.perf_counter() - started:.1f}")


def _holdout_guard(arm: str, grid_ids) -> dict:
    tau_path = TRAIN_DIR / f"frozen_tau_{arm}.json"
    if not tau_path.exists():
        raise RuntimeError(f"refusing to predict holdout for {arm}: {tau_path.name} is missing")
    if set(grid_ids) & set(EXCLUDED):
        raise RuntimeError("CPT1238 or CPT2172 reached holdout predict")
    if set(grid_ids) - set(HOLDOUT):
        raise RuntimeError("holdout predict received a grid that is not in the locked holdout")
    for grid_id in grid_ids:
        if (CHIP_ROOT / grid_id / "labels").exists():
            raise RuntimeError(f"holdout {grid_id} gained a labels directory")
    return _read_json(tau_path)


def _read_elapsed(score_path: Path) -> float:
    marker = score_path.with_suffix(".seconds")
    if not marker.exists():
        return float("nan")
    return float(marker.read_text(encoding="utf-8").strip())


def _write_elapsed(score_path: Path, elapsed: float) -> None:
    score_path.with_suffix(".seconds").write_text(f"{elapsed:.4f}\n", encoding="utf-8")


def _score_holdout_yolo(arm: str, config: dict) -> tuple[pd.DataFrame, float]:
    _holdout_guard(arm, HOLDOUT)
    weights = TRAIN_DIR / "runs" / f"{arm}_final" / "weights" / "best.pt"
    if not weights.exists():
        raise RuntimeError(f"missing final weights {weights}")
    score_path = TRAIN_DIR / "holdout" / f"{arm}_scores.csv"
    if _scores_are_complete(score_path, HOLDOUT):
        return pd.read_csv(score_path), _read_elapsed(score_path)
    started = time.perf_counter()
    rows = []
    for grid_id in HOLDOUT:
        rows.extend(_score_yolo_dir(
            weights, CHIP_ROOT / grid_id / "images", config, grid_id,
        ))
    _write_scores(score_path, rows)
    if not _scores_are_complete(score_path, HOLDOUT):
        raise RuntimeError(f"{score_path} failed validation")
    elapsed = time.perf_counter() - started
    _write_elapsed(score_path, elapsed)
    return pd.read_csv(score_path), elapsed


def _score_holdout_vit(config: dict) -> tuple[pd.DataFrame, float]:
    import torch

    _holdout_guard("vit30", HOLDOUT)
    score_path = TRAIN_DIR / "holdout" / "vit30_scores.csv"
    weights = TRAIN_DIR / "runs" / "vit30_final" / "head.pt"
    if not weights.exists():
        raise RuntimeError(f"missing {weights}")
    if _scores_are_complete(score_path, HOLDOUT):
        return pd.read_csv(score_path), _read_elapsed(score_path)
    started = time.perf_counter()
    from torch.utils.data import DataLoader

    checkpoint = torch.load(weights, map_location="cuda", weights_only=False)
    model = _load_backbone()
    head = torch.nn.Linear(VIT_DIM, 1).cuda()
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    rows = []
    for grid_id in HOLDOUT:
        loader = DataLoader(
            _PngChips(grid_id), batch_size=int(config["vit"]["forward_batch"]),
            shuffle=False, num_workers=2, pin_memory=True,
        )
        with torch.inference_mode():
            for images, indexes in loader:
                features = model.forward_features(images.cuda(non_blocking=True))
                chip_logit = head(features[:, 1:, :]).squeeze(-1).max(dim=1).values
                scores = torch.sigmoid(chip_logit).detach().cpu().numpy()
                for offset, score in enumerate(scores):
                    rows.append({
                        "grid_id": grid_id,
                        "chip_index": int(indexes[offset]),
                        "score": float(score),
                    })
    _write_scores(score_path, rows)
    del model, head
    torch.cuda.empty_cache()
    if not _scores_are_complete(score_path, HOLDOUT):
        raise RuntimeError(f"{score_path} failed validation")
    elapsed = time.perf_counter() - started
    _write_elapsed(score_path, elapsed)
    return pd.read_csv(score_path), elapsed


def _holdout_metrics(arm: str, frame: pd.DataFrame, elapsed: float, selection: dict) -> dict:
    tau_payload = _holdout_guard(arm, tuple(frame["grid_id"].unique()))
    tau = float(tau_payload["tau"])
    stratum = {}
    for name, grids in selection["holdout_by_stratum"].items():
        for grid_id in grids:
            stratum[grid_id] = name
    per_grid = {}
    whole = 0
    partial = 0.0
    for grid_id in HOLDOUT:
        scores = _score_array(frame, grid_id)
        installs = _load_installs(grid_id, scores)
        grid_whole, grid_partial = _losses_at(installs, tau)
        whole += grid_whole
        partial += grid_partial
        per_grid[grid_id] = {
            "stratum": stratum[grid_id],
            "pass_rate": float((scores >= tau).mean()),
            "whole_loss": int(grid_whole),
            "partial_area_ge_30_m2": float(grid_partial),
        }
    n_chips = N_CHIPS * len(HOLDOUT)
    chips_per_s = None if not math.isfinite(elapsed) or elapsed <= 0 else n_chips / elapsed
    return {
        "tau": tau,
        "oof_loss_met": bool(tau_payload["loss_met"]),
        "oof_pass_rate": float(tau_payload["oof_pass_rate"]),
        "whole_loss": int(whole),
        "partial_area_ge_30_m2": float(partial),
        "chips_per_s": None if chips_per_s is None else float(chips_per_s),
        "seconds": None if not math.isfinite(elapsed) else float(elapsed),
        "n_chips": n_chips,
        "grids": per_grid,
        "local_relative_time_vs_20_24_cps": (
            None if chips_per_s is None else float(LOCAL_MASKRCNN_CPS / chips_per_s)
        ),
    }


def render_result_section(report: dict) -> str:
    """Chinese result section. Speed stays unjudged because S1 was not measured."""
    lines = [
        "## 学习曲线（2026-09-28 抽样，2026-09-29 训练）",
        "",
        "这一节是 30 个训练格的学习曲线，加上 9 个新留出格各评一次。试点那一节的「失败」不变。",
        "S1（生产机只读图的时间）没有测，所以下面不判简报 S2 的速度条款，也不把门接进 `detect_direct.py`。",
        "本机 Mask R-CNN 大约 20.24 chip/秒，只用来换算本机相对耗时，不能拿去和生产机大约 16.7 秒/格比。",
        "",
        s4_reading(report["s4"]["yolo10_loss_met"], report["s4"]["yolo30_loss_met"]),
        "",
        "| 臂 | 折外阈值 | 折外丢失条件 | 折外放行率 | 留出整块丢失 | 留出 ≥30 m² 部分丢失面积 | 留出 chip/秒 | 本机相对耗时 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("yolo10", "yolo30", "vit30"):
        row = report["arms"][arm]
        rate = "—" if row["chips_per_s"] is None else f"{row['chips_per_s']:.1f}"
        relative = "—" if row["local_relative_time_vs_20_24_cps"] is None else f"{row['local_relative_time_vs_20_24_cps']:.3f}"
        lines.append(
            f"| {arm} | {row['tau']:.4f} | {'成立' if row['oof_loss_met'] else '不成立'} | "
            f"{row['oof_pass_rate']:.1%} | {row['whole_loss']} | {row['partial_area_ge_30_m2']:.2f} | "
            f"{rate} | {relative} |"
        )
    lines.extend([
        "",
        "折外丢失条件成立，是指存在一个大于 0 的阈值，折外安装整块丢失为 0，并且面积至少 30 m² 的安装上部分丢失面积合计为 0。",
        "阈值记 0 就是没有这样的正阈值。这时全放行也能让丢失变成 0，但这一臂的丢失条件仍然算失败。",
        "留出整块丢失和部分丢失面积用的是上表这个已经冻结的阈值，不是在留出格上再挑一次。",
        "",
        "留出格放行率：",
        "",
        "| 网格 | 档 | yolo10 | yolo30 | vit30 |",
        "|---|---|---:|---:|---:|",
    ])
    for grid_id in HOLDOUT:
        cells = []
        stratum = report["arms"]["yolo30"]["grids"][grid_id]["stratum"]
        for arm in ("yolo10", "yolo30", "vit30"):
            cells.append(f"{report['arms'][arm]['grids'][grid_id]['pass_rate']:.1%}")
        lines.append(f"| {grid_id} | {stratum} | " + " | ".join(cells) + " |")
    yolo30 = report["arms"]["yolo30"]
    vit = report["arms"]["vit30"]
    yolo_loss_ok = yolo30["whole_loss"] == 0 and yolo30["partial_area_ge_30_m2"] <= MISS_TOL_M2
    vit_loss_ok = vit["whole_loss"] == 0 and vit["partial_area_ge_30_m2"] <= MISS_TOL_M2
    lines.extend([
        "",
        f"YOLO 30 格在留出格上的丢失条件{'成立' if yolo_loss_ok else '不成立'}："
        f"整块丢失 {yolo30['whole_loss']}，≥30 m² 部分丢失面积 {yolo30['partial_area_ge_30_m2']:.2f} m²。",
        f"ViT 30 格在留出格上的丢失条件{'成立' if vit_loss_ok else '不成立'}："
        f"整块丢失 {vit['whole_loss']}，≥30 m² 部分丢失面积 {vit['partial_area_ge_30_m2']:.2f} m²。",
        "S3 还要求 ViT 的相对耗时比 YOLO 30 格低至少 0.05。这个差用的是生产机时间，S1 没测，所以速度这一半不判。",
        "",
        "数字说的是这 9 个留出格、这一套冻结阈值上的丢失和放行率。",
        "它要紧，是因为上一轮把阈值压到 0 的直接原因是训练格子太少；这一轮就是在回答加格子能不能把正阈值找回来。",
        "如果不管 S1 还没测、只凭本机 chip/秒宣布加速通过，就会把一台 4070 笔记本的读图时间当成生产机的读图天花板。",
        "",
    ])
    return "\n".join(lines)


def _write_result(report: dict) -> None:
    section = RESULT_START + "\n" + render_result_section(report) + RESULT_END
    body = RESULT_MD.read_text(encoding="utf-8")
    if RESULT_START in body:
        left = body.index(RESULT_START)
        right = body.index(RESULT_END, left) + len(RESULT_END)
        body = body[:left] + section + body[right:]
    else:
        body = body.rstrip() + "\n\n" + section + "\n"
    RESULT_MD.write_text(body, encoding="utf-8")


def _write_interim(report: dict) -> None:
    _replace_span(RUN_MD, INTERIM_START, INTERIM_END, "\n" + render_interim_section(report).strip() + "\n")


def _run_interim_comparison(selection: dict, config: dict, yolo10_folds: dict, yolo30_folds: dict) -> None:
    """ViT versus YOLO on folds 0–3. Fold 4, finals, and holdout stay unrun."""
    kept = active_folds(yolo30_folds)
    held_back = deferred_grids(yolo30_folds)
    scored = tuple(grid_id for fold in sorted(kept) for grid_id in kept[fold])
    if len(scored) != 26 or len(held_back) != 4:
        raise RuntimeError(f"interim coverage is {len(scored)} scored and {len(held_back)} deferred")
    if set(scored) & set(held_back):
        raise RuntimeError("a deferred grid is also scored")
    set_run_status(
        "第四折已完成。第五折先不训。正在用前四折的折外分数对照 YOLO 和冻住的 DINOv2。"
        "留出未评，S2–S5 不写。S1 未测。",
        title="存在性门学习曲线 — 第四折后改跑 DINO",
    )
    _extract_tokens(arm_grids(selection, "vit30"), int(config["vit"]["forward_batch"]))
    vit_folds = active_folds(folds_for_arm(selection, "vit30"))
    for fold, val_ids in vit_folds.items():
        _run_vit_fold(selection, fold, val_ids, config)
    yolo10_tau = _freeze_if_missing(
        "yolo10", _concat_oof("yolo10", yolo10_folds), arm_grids(selection, "yolo10"),
    )
    yolo30_tau = evaluate_scores(
        "yolo30",
        _concat_oof("yolo30", kept),
        scored,
        dest=TRAIN_DIR / "interim_tau_yolo30_folds0to3.json",
    )
    vit_tau = evaluate_scores(
        "vit30",
        _concat_oof("vit30", vit_folds),
        scored,
        dest=TRAIN_DIR / "interim_tau_vit30_folds0to3.json",
    )
    report = {
        "schema": "learning_curve_interim_folds0to3_v1",
        "deferred_fold": DEFERRED_FOLD,
        "deferred_grids": list(held_back),
        "scored_grids": list(scored),
        "holdout_scored": False,
        "finals_trained": False,
        "s1_measured": False,
        "s2_speed_verdict": None,
        "yolo10": yolo10_tau,
        "yolo30_folds_0_to_3": yolo30_tau,
        "vit30_folds_0_to_3": vit_tau,
        "reading": interim_reading(bool(yolo30_tau["loss_met"]), bool(vit_tau["loss_met"])),
    }
    _json_dump(TRAIN_DIR / "interim_folds0to3.json", report)
    _write_interim(report)
    set_run_status(
        "前四折的折外对照已写进运行记录的 interim 一节，原始表在 "
        "`learning_curve_2026-09-28/train/interim_folds0to3.json`。"
        "第五折、全量重训和留出评分都还没做。S1 未测，S2–S5 不写。",
        title="存在性门学习曲线 — 前四折对照已写，第五折未跑",
    )
    _log("TOKEN_CACHE_KEPT for a possible fifth fold")


def _delete_token_cache() -> None:
    if TOKEN_DIR.exists():
        shutil.rmtree(TOKEN_DIR)
        _log("TOKEN_CACHE_DELETED")


def _freeze_if_missing(arm: str, frame: pd.DataFrame, grid_ids: tuple[str, ...]) -> dict:
    path = TRAIN_DIR / f"frozen_tau_{arm}.json"
    if path.exists():
        payload = _read_json(path)
        _log(f"REUSE frozen tau {arm}={payload['tau']}")
        return payload
    return evaluate_scores(arm, frame, grid_ids)


def main() -> int:
    try:
        preflight()
        _gpu_is_free()
        selection = load_selection()
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        set_run_status(
            "训练进行中。tmux 会话 `yolo_gate_lc_train`。留出还没有评分，S2–S5 先不写。S1 未测。",
            title="存在性门学习曲线 — 训练进行中",
        )
        _log("TRAIN_START learning_curve_train_v1")
        config = _write_config()
        yolo10_folds = folds_for_arm(selection, "yolo10")
        yolo30_folds = folds_for_arm(selection, "yolo30")
        for fold, val_ids in yolo10_folds.items():
            _run_yolo_fold(selection, "yolo10", fold, val_ids, config)
        for fold, val_ids in active_folds(yolo30_folds).items():
            _run_yolo_fold(selection, "yolo30", fold, val_ids, config)
        _run_interim_comparison(selection, config, yolo10_folds, yolo30_folds)
        _log("INTERIM_DONE folds=0-3 deferred_fold=4")
        return 0
    except Exception:
        import logging
        logging.basicConfig()
        logging.getLogger("learning_curve_train").exception("learning curve training failed")
        _log("TRAIN_FAILED")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise
