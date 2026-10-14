#!/usr/bin/env python3
"""Step 5: score the two holdout grids once with the frozen gate.

The threshold is read from frozen_gate.json. This script does not train
and does not write a new threshold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import region_registry
from core.inference.tile_dataset import SlidingWindowDataset
from scripts.gate_pilot.chip_windows import CHIP_SIZE, OVERLAP, chip_windows_from_tifs
from scripts.gate_pilot.common import (
    HOLDOUT_PV_GRID,
    INVENTORY_GPKG,
    OUT_ROOT,
    RESULT_MD,
    SMALL_AREA_M2,
    setup_logging,
)
from scripts.gate_pilot.intersect import chips_hit_mask

FROZEN_PATH = OUT_ROOT / "frozen_gate.json"
STEP5_DIR = OUT_ROOT / "step5"
CHIP_ROOT = OUT_ROOT / "chips"
TILES_ROOT = Path.home() / "zasolar_data" / "tiles" / "cape_town" / "aerial_2025"
MASKRCNN_CPT1238_CHIPS_PER_S = 20.24
CURVE_PNG = RESULT_MD.parent / "recall_pass_curve.png"


def _tile_paths(grid_id: str) -> list[Path]:
    source = region_registry.resolve_source_grid_id(grid_id, "cape_town")
    paths = sorted((TILES_ROOT / source).glob(f"{grid_id}_*_*_geo.tif"))
    if len(paths) != 42:
        raise RuntimeError(f"{grid_id} has {len(paths)} tiles")
    return paths


def _as_bgr_uint8(tensor) -> np.ndarray:
    """Rasterio chips are RGB. Ultralytics treats numpy input as BGR and flips it back."""
    array = np.rint(tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.transpose(array, (1, 2, 0))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def score_grid(grid_id: str, model, batch_size: int, conf: float, imgsz: int, log) -> tuple[np.ndarray, float]:
    paths = _tile_paths(grid_id)
    dataset = SlidingWindowDataset(
        paths, chip_size=CHIP_SIZE, overlap=OVERLAP, window_origin_mode="anchored",
    )
    scores = np.full(len(dataset), np.nan)
    # One batch warms the kernels. It is not part of the timed chips/s.
    warm, _ = dataset[0]
    model.predict([_as_bgr_uint8(warm)], conf=conf, imgsz=imgsz, verbose=False, device=0)
    started = time.perf_counter()
    batch_images: list[np.ndarray] = []
    batch_index: list[int] = []

    def flush() -> None:
        if not batch_images:
            return
        results = model.predict(batch_images, conf=conf, imgsz=imgsz, verbose=False, device=0)
        if len(results) != len(batch_index):
            raise RuntimeError(f"{grid_id} predict returned {len(results)} for {len(batch_index)} chips")
        for result, index in zip(results, batch_index):
            boxes = result.boxes
            scores[index] = float(boxes.conf.max().item()) if boxes is not None and len(boxes) else 0.0
        batch_images.clear()
        batch_index.clear()

    try:
        for index in range(len(dataset)):
            tensor, meta = dataset[index]
            if meta.chip_index != index:
                raise RuntimeError(f"{grid_id} chip_index drifted at {index}")
            batch_images.append(_as_bgr_uint8(tensor))
            batch_index.append(index)
            if len(batch_images) == batch_size:
                flush()
        flush()
    finally:
        dataset.close()
    elapsed = time.perf_counter() - started
    if np.isnan(scores).any():
        raise RuntimeError(f"{grid_id} has unscored chips")
    rate = len(scores) / elapsed
    log.info("%s scored %d chips in %.1fs (%.2f chips/s)", grid_id, len(scores), elapsed, rate)
    return scores, rate


def _rates(y_pre: np.ndarray, y_post: np.ndarray, passed: np.ndarray) -> dict:
    def recall(mask: np.ndarray) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(passed[mask].mean())
    return {
        "n_chips": int(len(passed)),
        "n_pre": int(y_pre.sum()),
        "n_post": int(y_post.sum()),
        "g1_post_recall": recall(y_post),
        "g2_pre_recall": recall(y_pre),
        "g3_pass_rate": float(passed.mean()),
        "n_post_missed": int((y_post & ~passed).sum()),
        "n_pre_missed": int((y_pre & ~passed).sum()),
    }


def _polygon_coverage(grid_id: str, scores: np.ndarray, tau: float) -> dict:
    hits = pd.read_csv(OUT_ROOT / "step2" / f"intersections_{grid_id}.csv")
    post = hits[hits["layer"] == "post"]
    out = {}
    for name, part in {
        "lt_30": post[post["poly_area_m2"] < SMALL_AREA_M2],
        "ge_30": post[post["poly_area_m2"] >= SMALL_AREA_M2],
    }.items():
        all_kept = 0
        any_kept = 0
        none_kept = 0
        for _, group in part.groupby("poly_id"):
            indexes = group["chip_index"].astype(int).to_numpy()
            kept = scores[indexes] >= tau
            if bool(kept.all()):
                all_kept += 1
            if bool(kept.any()):
                any_kept += 1
            else:
                none_kept += 1
        out[name] = {
            "n_polygons": int(part["poly_id"].nunique()) if len(part) else 0,
            "all_chips_passed": all_kept,
            "at_least_one_chip_passed": any_kept,
            "no_chip_passed": none_kept,
        }
    return out


def _inventory_recall(grid_id: str, scores: np.ndarray, tau: float) -> dict:
    inventory = gpd.read_file(INVENTORY_GPKG, columns=["source_grid", "area_m2"])
    inventory = inventory[inventory["source_grid"].astype(str) == grid_id]
    chips = chip_windows_from_tifs(_tile_paths(grid_id), grid_id)
    if len(chips) != len(scores):
        raise RuntimeError("chip window count does not match scores")
    hit = chips_hit_mask(chips, inventory)
    passed = scores >= tau
    n_hit = int(hit.sum())
    return {
        "n_inventory_polygons": int(len(inventory)),
        "n_inventory_chips": n_hit,
        "g1_inventory_recall": None if n_hit == 0 else float(passed[hit].mean()),
    }


def _curve(y_pre: np.ndarray, y_post: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for tau in np.linspace(0.0, 1.0, 1001):
        passed = scores >= tau
        row = _rates(y_pre, y_post, passed)
        row["tau"] = float(tau)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(curves: dict[str, pd.DataFrame]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    holdout = curves[HOLDOUT_PV_GRID]
    axis.plot(holdout["tau"], holdout["g1_post_recall"], label="CPT1238 post-filter recall")
    axis.plot(holdout["tau"], holdout["g2_pre_recall"], label="CPT1238 pre-filter recall")
    axis.plot(holdout["tau"], holdout["g3_pass_rate"], label="CPT1238 pass rate")
    empty_name = [name for name in curves if name != HOLDOUT_PV_GRID][0]
    axis.plot(curves[empty_name]["tau"], curves[empty_name]["g3_pass_rate"], label=f"{empty_name} pass rate")
    axis.axvline(0.0, color="black", linewidth=0.8, label="frozen tau = 0")
    axis.axhline(0.20, color="gray", linewidth=0.8, linestyle="--", label="20% pass cap")
    axis.set_xlabel("threshold")
    axis.set_ylabel("share of chips")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(CURVE_PNG, dpi=140)
    plt.close(figure)


def _rewrite_result(metrics: dict) -> None:
    text = RESULT_MD.read_text(encoding="utf-8")
    holdout = metrics["grids"][HOLDOUT_PV_GRID]
    empty_name = metrics["e_test"]
    empty = metrics["grids"][empty_name]
    g1 = holdout["g1_post_recall"]
    g2 = holdout["g2_pre_recall"]
    g3_h = holdout["g3_pass_rate"]
    g3_e = empty["g3_pass_rate"]
    passed = g1 >= 1.0 - 1e-9 and g2 >= 0.99 and g3_h <= 0.20 and g3_e <= 0.20
    if passed:
        conclusion = (
            "试点通过。留出格上过滤后 chip 召回是 100%，过滤前 chip 召回不低于 99%，"
            "两个留出格的放行率都不高于 20%。下一轮才是把门接进 detect_direct.py。"
        )
        status = "试点通过"
    else:
        misses = []
        if g1 < 1.0 - 1e-9:
            misses.append(f"G1 是 {g1:.2%}，差 {1-g1:.2%} 才到 100%")
        if g2 < 0.99:
            misses.append(f"G2 是 {g2:.2%}，低于 99%")
        if g3_h > 0.20:
            misses.append(f"CPT1238 的 G3 是 {g3_h:.2%}，比 20% 高 {g3_h-0.20:.2%}")
        if g3_e > 0.20:
            misses.append(f"{empty_name} 的 G3 是 {g3_e:.2%}，比 20% 高 {g3_e-0.20:.2%}")
        conclusion = (
            "试点失败。锁定阈值是 0，门放行了留出格上的每一张 chip。"
            + "；".join(misses)
            + "。G1 和 G2 因为全部放行而到了 100%，失败的是放行率。"
            "这次没有改阈值再测。CPT1238 已经被这次打分看过，下一轮要另选留出格。"
        )
        status = "试点失败"
    text = text.replace("状态：阈值已锁定为 0，留出测试尚未跑", f"状态：{status}", 1)
    # Replace the opening conclusion with the verdict, keeping the mechanism in the fold section.
    old_conclusion_start = "## 结论\n\n"
    old_conclusion_end = "\n\n## G1 / G2 / G3"
    start = text.index(old_conclusion_start)
    end = text.index(old_conclusion_end)
    why = (
        "忽略这个失败、改用曲线上另一个阈值再报通过，就把预注册的规则换掉了，"
        "而且用的是已经看过的 CPT1238。放行率超了多少，就说明这个锁定的门省不下 Mask R-CNN 的空 chip。"
    )
    text = text[:start] + "## 结论\n\n" + conclusion + "\n\n" + why + text[end:]
    table = (
        "## G1 / G2 / G3\n\n"
        "判定用冻结阈值 0，只测了这一次。\n\n"
        "| 网格 | G1 过滤后 chip 召回 | G2 过滤前 chip 召回 | G3 放行率 | 交付库存口径 G1 |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| CPT1238 | {g1:.1%} | {g2:.1%} | {g3_h:.1%} | {holdout['inventory']['g1_inventory_recall']:.1%} |\n"
        f"| {empty_name} | 空格无正样本 | 空格无正样本 | {g3_e:.1%} | 空格无正样本 |\n"
    )
    text = text[:text.index("## G1 / G2 / G3")] + table + text[text.index("## <30 m²"):]
    small = holdout["polygons"]["lt_30"]
    large = holdout["polygons"]["ge_30"]
    poly_section = (
        "## <30 m² 与 ≥30 m²\n\n"
        "口径是 CPT1238 单格过滤后文件。一块安装算「整块保住」是它碰到的每一张 chip 都放行；"
        "「至少一张放行」做不到的，整块安装会从普查里消失。阈值是 0，所以两种口径都是全部保住。\n\n"
        "| 面积 | 多边形数 | 全部相交 chip 都放行 | 至少一张放行 | 一张都没放行 |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| <30 m² | {small['n_polygons']} | {small['all_chips_passed']} | {small['at_least_one_chip_passed']} | {small['no_chip_passed']} |\n"
        f"| ≥30 m² | {large['n_polygons']} | {large['all_chips_passed']} | {large['at_least_one_chip_passed']} | {large['no_chip_passed']} |\n\n"
        "小于 30 m² 的过滤后阳性 chip 漏掉 "
        f"{holdout['n_post_small_missed']} 张，大于等于 30 m² 的漏掉 {holdout['n_post_large_missed']} 张。\n"
    )
    text = text[:text.index("## <30 m²")] + poly_section + text[text.index("## 步骤 0"):]
    gate_rate = holdout["chips_per_s"]
    relative = (MASKRCNN_CPT1238_CHIPS_PER_S / gate_rate) + g3_h
    speed = (
        "## 速度\n\n"
        "这台 RTX 4070 Laptop 上，重跑的 Mask R-CNN（`detect_direct.py`，batch 4，含读图）是每秒 19.9 到 21.3 张 chip，"
        "四格分别是 CPT2932 19.93、CPT2373 21.30、CPT2182 20.48、CPT1238 20.24，一格大约 100 秒。"
        "配对上的多边形平均 IoU 是 0.999。\n\n"
        f"门的单价在同一台机器上、从读 GeoTIFF 窗口开始计，CPT1238 是每秒 {gate_rate:.2f} 张 chip，"
        f"{empty_name} 是每秒 {empty['chips_per_s']:.2f} 张。模型加载和第一张的预热不计入。"
        f"和 CPT1238 的 Mask R-CNN 单价比，相对耗时 = {MASKRCNN_CPT1238_CHIPS_PER_S:.2f}/{gate_rate:.2f} + {g3_h:.2f} = {relative:.2f}。"
        "相对耗时大于 1，表示加上这道门之后检测阶段比现在更慢：门自己要付钱，又没有 chip 被挡在外面。\n"
    )
    text = text[:text.index("## 速度")] + speed + text[text.index("## G1 漏掉的 chip"):]
    miss = (
        "## G1 漏掉的 chip\n\n"
        + ("锁定阈值下没有漏掉过滤后阳性 chip。\n" if holdout["n_post_missed"] == 0
           else f"漏掉 {holdout['n_post_missed']} 张，清单在 `step5/missed_chips.csv`。\n")
        + "\n召回–放行率曲线按阈值从 0 扫到 1，图在 `docs/experiments/2026-09-yolo-gate-pilot/recall_pass_curve.png`。"
        "判定没有用这张图挑阈值。\n"
    )
    text = text[:text.index("## G1 漏掉的 chip")] + miss + text[text.index("## 空格、版本、冻结文件"):]
    RESULT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    log = setup_logging("step5_holdout")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    tau = float(frozen["tau"])
    from ultralytics import YOLO
    model = YOLO(frozen["weights_path"])
    STEP5_DIR.mkdir(parents=True, exist_ok=True)
    grids = [HOLDOUT_PV_GRID, frozen["e_test"]]
    payload = {"tau": tau, "e_test": frozen["e_test"], "grids": {}}
    curves = {}
    for grid_id in grids:
        scores, rate = score_grid(grid_id, model, batch_size=32, conf=float(frozen["conf"]), imgsz=int(frozen["imgsz"]), log=log)
        manifest = pd.read_csv(CHIP_ROOT / grid_id / "manifest.csv").sort_values("chip_index")
        if manifest["chip_index"].tolist() != list(range(len(scores))):
            raise RuntimeError(f"{grid_id} manifest chip_index is not 0..n-1")
        y_pre = manifest["y_pre"].to_numpy().astype(bool)
        y_post = manifest["y_post"].to_numpy().astype(bool)
        y_small = manifest["y_post_small"].to_numpy().astype(bool)
        hits = pd.read_csv(OUT_ROOT / "step2" / f"intersections_{grid_id}.csv")
        large_index = hits.loc[
            (hits["layer"] == "post") & (hits["poly_area_m2"] >= SMALL_AREA_M2), "chip_index"
        ].astype(int)
        y_large = np.zeros(len(scores), dtype=bool)
        if len(large_index):
            y_large[large_index.to_numpy()] = True
        passed = scores >= tau
        row = _rates(y_pre, y_post, passed)
        row["chips_per_s"] = rate
        row["polygons"] = _polygon_coverage(grid_id, scores, tau)
        row["inventory"] = _inventory_recall(grid_id, scores, tau)
        row["n_post_small_missed"] = int((y_small & ~passed).sum())
        row["n_post_large_missed"] = int((y_large & ~passed).sum())
        row["n_post_missed"] = row["n_post_missed"]
        pd.DataFrame({"chip_index": np.arange(len(scores)), "score": scores, "passed": passed}).to_csv(
            STEP5_DIR / f"scores_{grid_id}.csv", index=False
        )
        curve = _curve(y_pre, y_post, scores)
        curve.to_csv(STEP5_DIR / f"curve_{grid_id}.csv", index=False)
        curves[grid_id] = curve
        payload["grids"][grid_id] = row
        log.info("%s %s", grid_id, {k: row[k] for k in ("g1_post_recall", "g2_pre_recall", "g3_pass_rate")})
    _plot(curves)
    (STEP5_DIR / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _rewrite_result(payload)
    log.info("wrote %s", RESULT_MD)
    holdout = payload["grids"][HOLDOUT_PV_GRID]
    empty = payload["grids"][frozen["e_test"]]
    ok = (
        holdout["g1_post_recall"] >= 1.0 - 1e-9
        and holdout["g2_pre_recall"] >= 0.99
        and holdout["g3_pass_rate"] <= 0.20
        and empty["g3_pass_rate"] <= 0.20
    )
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import logging
        logging.getLogger("step5_holdout").exception("step 5 failed")
        raise
