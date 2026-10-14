#!/usr/bin/env python3
"""Step 0 of the YOLO presence-gate pilot: vector-only feasibility.

No imagery is downloaded. The script pulls census vectors, draws the two
empty grids, enumerates chips with SlidingWindowDataset on geometry-carrier
GeoTIFFs, and applies the pre-registered stop rule on CPT1238.

Exit codes: 0 feasibility passed, 2 pre-registered stop, 1 usage or runtime error.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import geopandas as gpd
import pandas as pd
import pyogrio

from scripts.gate_pilot.chip_windows import build_grid_chip_windows
from scripts.gate_pilot.common import (
    CANDIDATE_DIR,
    CROSSWALK_CSV,
    DROPBOX_PERDET,
    EMPTY_SEED,
    G2_RECALL,
    G3_PASS_RATE,
    HOLDOUT_PV_GRID,
    INVENTORY_CSV,
    INVENTORY_GPKG,
    LOG_DIR,
    OUT_ROOT,
    REPO_ROOT,
    RESULT_MD,
    SMALL_AREA_M2,
    STEP0_DIR,
    TRAIN_PV_GRIDS,
    VECTOR_DIR,
    VECTOR_FILES,
    setup_logging,
    sha256_file,
)
from scripts.gate_pilot.empty_select import select_empty_grids
from scripts.gate_pilot.feasibility import g2_minimum_pass
from scripts.gate_pilot.intersect import (
    best_partner_grid,
    chips_hit_mask,
    count_with_partner,
    iou_component_count,
)

SELECTION_PATH = STEP0_DIR / "empty_grid_selection.json"
CHECKPOINT = REPO_ROOT / "checkpoints" / "exp_unified_reviewall_A" / "best_model.pth"


def _rclone(relative_paths: list[str], dest: Path, log, *, required: bool) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    missing = [
        rel for rel in relative_paths
        if not (dest / rel).exists() or (dest / rel).stat().st_size == 0
    ]
    if not missing:
        log.info("already on disk: %d files under %s", len(relative_paths), dest)
        return []
    STEP0_DIR.mkdir(parents=True, exist_ok=True)
    list_path = STEP0_DIR / "rclone_files_from.txt"
    list_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
    cmd = [
        "rclone", "copy", DROPBOX_PERDET, str(dest),
        "--files-from", str(list_path),
        "--transfers", "8",
        "--checkers", "8",
        "--retries", "5",
        "--low-level-retries", "10",
        "--stats", "20s",
        "--stats-one-line",
    ]
    log.info("rclone copy %d missing files into %s", len(missing), dest)
    completed = subprocess.run(cmd, check=False)
    still = [rel for rel in missing if not (dest / rel).exists() or (dest / rel).stat().st_size == 0]
    if completed.returncode != 0 or still:
        message = f"rclone exited {completed.returncode} with {len(still)} files still missing, first: {still[:5]}"
        if required:
            raise RuntimeError(message)
        log.warning(message)
    return still


def census_empty_grid_ids() -> list[str]:
    crosswalk = pd.read_csv(CROSSWALK_CSV)
    kept = set(crosswalk.loc[crosswalk["kept"] == True, "cpt_id"].dropna().astype(str))  # noqa: E712
    task = gpd.read_file(REPO_ROOT / "data" / "task_grid_cpt.gpkg", columns=["gridcell_id"])
    in_task = set(task["gridcell_id"].astype(str))
    inventory = pd.read_csv(INVENTORY_CSV, usecols=["source_grid"])
    occupied = set(inventory["source_grid"].dropna().astype(str))
    return sorted((kept & in_task) - occupied)


def _feature_count(path: Path) -> int:
    return int(pyogrio.read_info(path)["features"])


def _read_polygons(path: Path) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.crs is None or frame.crs.to_epsg() != 32734:
        raise RuntimeError(f"{path} CRS is {frame.crs}, expected EPSG:32734")
    if "area_m2" not in frame.columns:
        frame["area_m2"] = frame.geometry.area
    return frame


def _draw_empty_grids(empty_ids: list[str], log) -> dict:
    metric_rels = [f"{grid_id}/predictions_metric.gpkg" for grid_id in empty_ids]
    _rclone(metric_rels, CANDIDATE_DIR, log, required=True)
    records = []
    for grid_id in empty_ids:
        path = CANDIDATE_DIR / grid_id / "predictions_metric.gpkg"
        records.append((grid_id, _feature_count(path)))
    counts = pd.DataFrame(records, columns=["grid_id", "pre_count"]).sort_values(
        ["pre_count", "grid_id"]
    )
    drawn = select_empty_grids(records, seed=EMPTY_SEED)
    payload = {
        "seed": EMPTY_SEED,
        "python": sys.version,
        "n_empty_inventory_grids": len(empty_ids),
        "n_pre_ge_1": drawn.n_candidates_pre_ge_1,
        "used_fallback": drawn.used_fallback,
        "cutoff_pre_count": drawn.cutoff_pre_count,
        "upper_half_size": len(drawn.upper_half),
        "e_train": drawn.e_train,
        "e_test": drawn.e_test,
        "e_train_pre_count": int(counts.loc[counts.grid_id == drawn.e_train, "pre_count"].iloc[0]),
        "e_test_pre_count": int(counts.loc[counts.grid_id == drawn.e_test, "pre_count"].iloc[0]),
        "rule": (
            "Keep pre_count>=1. Sort by (pre_count, grid_id). "
            "Upper half is the slice from n//2 to the end. "
            "If that slice has fewer than 2 grids, draw from all kept grids. "
            "random.Random(seed).sample on the lexicographically sorted ids; "
            "first draw is E_train, second is E_test."
        ),
    }
    if SELECTION_PATH.exists():
        locked = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        if (locked["e_train"], locked["e_test"]) != (drawn.e_train, drawn.e_test):
            raise RuntimeError(
                "empty-grid draw changed relative to "
                f"{SELECTION_PATH}: locked {(locked['e_train'], locked['e_test'])} "
                f"vs recomputed {(drawn.e_train, drawn.e_test)}"
            )
        log.info("empty-grid draw matches the locked file: %s %s", drawn.e_train, drawn.e_test)
    else:
        SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        SELECTION_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log.info("locked empty grids E_train=%s E_test=%s", drawn.e_train, drawn.e_test)
    upper = set(drawn.upper_half)
    counts["in_upper_half"] = counts["grid_id"].isin(upper)
    counts["role"] = ""
    counts.loc[counts.grid_id == drawn.e_train, "role"] = "E_train"
    counts.loc[counts.grid_id == drawn.e_test, "role"] = "E_test"
    counts.to_csv(STEP0_DIR / "empty_grid_pre_counts.csv", index=False)
    return payload


def _pilot_grids(selection: dict) -> list[tuple[str, str]]:
    return [
        (TRAIN_PV_GRIDS[0], "train_pv"),
        (TRAIN_PV_GRIDS[1], "train_pv"),
        (TRAIN_PV_GRIDS[2], "train_pv"),
        (selection["e_train"], "train_empty"),
        (HOLDOUT_PV_GRID, "holdout_pv"),
        (selection["e_test"], "holdout_empty"),
    ]


def _hash_vectors(grids: list[str], log) -> pd.DataFrame:
    rows = []
    for grid_id in grids:
        for name in VECTOR_FILES:
            path = VECTOR_DIR / grid_id / name
            if not path.exists():
                rows.append({"grid_id": grid_id, "filename": name, "nbytes": 0, "sha256": ""})
                log.warning("missing vector file %s", path)
                continue
            rows.append(
                {
                    "grid_id": grid_id,
                    "filename": name,
                    "nbytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(STEP0_DIR / "vector_sha256.csv", index=False)
    return frame


def _checkpoint_note(grids: list[str]) -> dict:
    local_sha = sha256_file(CHECKPOINT) if CHECKPOINT.exists() else ""
    recorded = {}
    for grid_id in grids:
        path = VECTOR_DIR / grid_id / "config.json"
        if not path.exists():
            recorded[grid_id] = None
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded[grid_id] = payload.get("model_sha256")
    unique = sorted({value for value in recorded.values() if value})
    return {
        "local_checkpoint": str(CHECKPOINT),
        "local_sha256": local_sha,
        "backup_config_sha256": recorded,
        "backup_sha256_unique": unique,
        "local_matches_backup": bool(unique) and unique == [local_sha],
        "plan_abbreviation": "c32ebf26…696e",
        "plan_abbreviation_note": (
            "The plan abbreviates the backup hash as c32ebf26…696e. "
            "The comparison that matters is the full model_sha256 inside "
            "the pulled config.json against the local checkpoint."
        ),
    }


def _count_gap(grid_id: str, pre: gpd.GeoDataFrame, post: gpd.GeoDataFrame, inventory: gpd.GeoDataFrame) -> dict:
    same = inventory[inventory["source_grid"].astype(str) == grid_id]
    n_post = len(post)
    matched_pre = count_with_partner(post, pre, 0.5) if n_post else 0
    components = iou_component_count(post, 0.10) if n_post else 0
    if n_post and len(same):
        same_partner = best_partner_grid(
            post,
            same,
            same["source_grid"].astype(str).to_numpy(),
            0.10,
        )
    else:
        same_partner = [None] * n_post
    hit_same = sum(partner is not None for partner in same_partner)
    remainder_pos = [i for i, partner in enumerate(same_partner) if partner is None]
    hit_other = 0
    if remainder_pos and n_post:
        other = inventory[inventory["source_grid"].astype(str) != grid_id]
        remainder = post.iloc[remainder_pos]
        other_partner = best_partner_grid(
            remainder,
            other,
            other["source_grid"].astype(str).to_numpy(),
            0.10,
        )
        hit_other = sum(partner is not None for partner in other_partner)
    hit_none = len(remainder_pos) - hit_other
    return {
        "grid_id": grid_id,
        "n_pre": int(len(pre)),
        "n_post": int(n_post),
        "n_inventory": int(len(same)),
        "n_inventory_small": int((same["area_m2"] < SMALL_AREA_M2).sum()) if len(same) else 0,
        "post_matched_to_pre_iou50": int(matched_pre),
        "post_iou010_components": int(components),
        "post_hit_same_grid_inventory": int(hit_same),
        "post_hit_other_grid_only": int(hit_other),
        "post_hit_no_inventory": int(hit_none),
    }


def _gap_sentence(row: dict) -> str:
    n_post = row["n_post"]
    matched = row["post_matched_to_pre_iou50"]
    if n_post == 0:
        cls_clause = (
            f"过滤前 {row['n_pre']} 条，过滤后文件是空的，交付库存 {row['n_inventory']} 条"
        )
    else:
        cls_clause = (
            f"过滤前 {row['n_pre']} 条到过滤后 {n_post} 条，"
            f"其中 {matched} 条过滤后多边形能以 IoU≥0.5 对上一条过滤前多边形"
        )
        if matched == n_post:
            cls_clause += "，这一段是分类器删行，剩下的形状还在过滤前文件里"
    merge_clause = (
        f"过滤后 {n_post} 条到交付库存 {row['n_inventory']} 条："
        f"过滤后多边形按 IoU≥0.10 在本格内并成 {row['post_iou010_components']} 组；"
        f"{row['post_hit_same_grid_inventory']} 条对得上本格库存，"
        f"{row['post_hit_other_grid_only']} 条只对得上别的格，"
        f"{row['post_hit_no_inventory']} 条对不上任何库存多边形"
    )
    return f"{row['grid_id']}：{cls_clause}。{merge_clause}。"


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _write_result(
    rows: list[dict],
    selection: dict,
    checkpoint: dict,
    gap_rows: list[dict],
    holdout_rule: dict,
) -> None:
    by_id = {row["grid_id"]: row for row in rows}
    holdout = by_id[HOLDOUT_PV_GRID]
    stopped = bool(holdout_rule["stop"])
    if stopped:
        conclusion = (
            f"步骤 0 提前止损。CPT1238 上 0.99×P_pre = "
            f"{_fmt_pct(holdout_rule['continuous_min_pass'])}，高于 20% 的放行上限，"
            f"所以过滤前召回和放行率不能同时过。这一步没有下载影像，也没有训练。"
        )
    else:
        integer_note = ""
        if holdout_rule["integer_exceeds_limit"]:
            integer_note = (
                f" 整数口径下至少要放行 {holdout_rule['integer_k_min']} 张，"
                f"占 {_fmt_pct(holdout_rule['integer_min_pass'])}，已经超过 20%；"
                f"预注册的停止条件用的是连续乘积，那个数没有过线，所以按文字继续。"
            )
        conclusion = (
            f"步骤 0 通过，试点还没做完。CPT1238 上 0.99×P_pre = "
            f"{_fmt_pct(holdout_rule['continuous_min_pass'])}，没有高于 20%。"
            f"G2 和 G3 在数学上还留得下空间，值得下载影像并训练。"
            f"{integer_note}"
        )
    rate_lines = [
        "| 网格 | 角色 | chip 数 | 过滤前多边形 | 过滤前阳性 chip | P_pre | 过滤后阳性 chip | P_post | <30 m² 阳性 chip | ≥30 m² 阳性 chip |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rate_lines.append(
            "| {grid_id} | {role} | {n_chips} | {n_pre_polygons} | {n_chips_pre} | {p_pre} | "
            "{n_chips_post} | {p_post} | {n_chips_post_small} | {n_chips_post_large} |".format(
                grid_id=row["grid_id"],
                role=row["role"],
                n_chips=row["n_chips"],
                n_pre_polygons=row["n_pre_polygons"],
                n_chips_pre=row["n_chips_pre"],
                p_pre=_fmt_pct(row["p_pre"]),
                n_chips_post=row["n_chips_post"],
                p_post=_fmt_pct(row["p_post"]),
                n_chips_post_small=row["n_chips_post_small"],
                n_chips_post_large=row["n_chips_post_large"],
            )
        )
    gap_lines = "\n".join(f"- {_gap_sentence(row)}" for row in gap_rows)
    match = "一致" if checkpoint["local_matches_backup"] else "不一致"
    body = f"""# YOLO 存在性门试点 — 结果

日期：2026-09-23
状态：{"步骤 0 提前止损" if stopped else "步骤 0 通过，步骤 1 尚未跑完"}
方案：`docs/plans/2026-09-23-yolo-presence-gate-pilot.md`

## 结论

{conclusion}

忽略这一步的话，会在一个数学上过不了的门上花掉下载和训练，或者反过来，会把一个其实还有空间的门提前停掉。上面这一句就是为了避免这两种走法。

## G1 / G2 / G3

留出测试还没有做。门没有在 CPT1238 或留出空格的 chip 上跑过。

| 网格 | G1 过滤后 chip 召回 | G2 过滤前 chip 召回 | G3 放行率 | 交付库存口径 G1 |
|---|---|---|---|---|
| CPT1238 | 未测 | 未测 | 未测 | 未测 |
| {selection["e_test"]} | 空格无正样本 | 空格无正样本 | 未测 | 空格无正样本 |

## <30 m² 与 ≥30 m²

这一档要等留出测试。步骤 0 只知道有多少张 chip 碰到这类多边形，见下面的 P_pre 表，还不知道门会放行哪些。

## 步骤 0 的 P_pre / P_post

P_pre 是与过滤前多边形相交面积 > 0 的 chip 占该格全部 chip 的比例。相交在瓦片的 EPSG:4326 里算：多边形从 EPSG:32734 投过来，chip 用有效像素矩形，不含零填充。没有面积下限，贴边但面积为 0 的不算。

{chr(10).join(rate_lines)}

CPT1238 的停止条件：0.99×P_pre = {_fmt_pct(holdout_rule["continuous_min_pass"])}，20% 上限是放行率。整数口径要放行至少 {holdout_rule["integer_k_min"]} 张阳性 chip，占 {_fmt_pct(holdout_rule["integer_min_pass"])}。预注册触发的是连续乘积严格大于 20%，这次{"触发了" if stopped else "没有触发"}。

chip 窗口来自 `SlidingWindowDataset`（400 像素、overlap 0.25、anchored），套在几何载体 GeoTIFF 上。载体的仿射用的是 `download_tiles.py` 同一对函数：`get_tile_bounds` 和 `from_bounds`，每张瓦片 2000×2000 像素。这不是市政影像。步骤 1 下载真瓦片后必须用真 GeoTIFF 重算一遍，核对窗口完全一致。

## 影像一致性

还没下载影像，也还没重跑 Mask R-CNN。

## 单格计数和交付库存为什么不一样

交付库存是 `ct_full_inventory_2026-06-21_merged`，城市尺度上把 IoU≥0.10 的检测并成一块安装。下面是这 6 个格上量出来的，不是从那句话直接套过来的。

{gap_lines}

## 速度

还没有门的单价，也没有这次重跑的 Mask R-CNN 单价。

## G1 漏掉的 chip

留出测试还没有做。

## 空格、版本、冻结文件

- 种子 {selection["seed"]}。交付库存里 0 块、且过滤前至少 1 块的格有 {selection["n_pre_ge_1"]} 个。上半区 {selection["upper_half_size"]} 个，切点是过滤前 {selection["cutoff_pre_count"]} 块。{"上半区不足 2 个，退回了全部候选。" if selection["used_fallback"] else "没有触发回退。"}
- E_train = {selection["e_train"]}（过滤前 {selection["e_train_pre_count"]} 块），E_test = {selection["e_test"]}（过滤前 {selection["e_test_pre_count"]} 块）。
- 训练格：CPT2932、CPT2373、CPT2182、{selection["e_train"]}。留出格：CPT1238、{selection["e_test"]}。
- 本机检测器 `{checkpoint["local_checkpoint"]}` 的 sha256 是 `{checkpoint["local_sha256"]}`。六个备份 config 里的 `model_sha256` 与它{match}（备份里出现的值：{", ".join(checkpoint["backup_sha256_unique"]) or "无"}）。方案正文里的缩写 `c32ebf26…696e` 尾段和这个完整哈希对不上；以备份 config 里的完整字段为准。
- ultralytics 还没装。`frozen_gate.json` 还没写。留出格没有跑过门。

## 这一步的产出

- 矢量：`~/zasolar_data/gate_pilot_2026-09/vectors/`
- 空格候选的过滤前 gpkg：`~/zasolar_data/gate_pilot_2026-09/step0/empty_candidates/`
- 表：`step0/chip_rates.csv`、`step0/empty_grid_pre_counts.csv`、`step0/vector_sha256.csv`、`step0/count_gap.json`、`step0/feasibility.json`
"""
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.write_text(body, encoding="utf-8")


def main() -> int:
    log = setup_logging("step0_vector_feasibility")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STEP0_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("step 0 start, output %s", OUT_ROOT)

    empty_ids = census_empty_grid_ids()
    log.info("empty inventory grids in the census footprint: %d", len(empty_ids))
    selection = _draw_empty_grids(empty_ids, log)
    pilot = _pilot_grids(selection)
    required_rels = []
    optional_rels = []
    for grid_id, role in pilot:
        for name in VECTOR_FILES:
            rel = f"{grid_id}/{name}"
            if name == "predictions_metric_cls_filtered.gpkg" and role.endswith("empty"):
                optional_rels.append(rel)
            else:
                required_rels.append(rel)
    _rclone(required_rels, VECTOR_DIR, log, required=True)
    _rclone(optional_rels, VECTOR_DIR, log, required=False)
    hashes = _hash_vectors([grid_id for grid_id, _role in pilot], log)
    checkpoint = _checkpoint_note([grid_id for grid_id, _role in pilot])
    log.info("checkpoint match=%s", checkpoint["local_matches_backup"])

    inventory = gpd.read_file(INVENTORY_GPKG, columns=["source_grid", "area_m2"])
    if inventory.crs is None or inventory.crs.to_epsg() != 32734:
        raise RuntimeError(f"inventory CRS is {inventory.crs}")

    rows = []
    gap_rows = []
    for grid_id, role in pilot:
        log.info("chips and intersections for %s (%s)", grid_id, role)
        pre_path = VECTOR_DIR / grid_id / "predictions_metric.gpkg"
        post_path = VECTOR_DIR / grid_id / "predictions_metric_cls_filtered.gpkg"
        if not pre_path.exists():
            raise RuntimeError(f"missing {pre_path}")
        pre = _read_polygons(pre_path)
        if post_path.exists():
            post = _read_polygons(post_path)
            post_missing = False
        elif role.endswith("empty"):
            post = pre.iloc[0:0].copy()
            post_missing = True
            log.warning("no filtered gpkg for empty grid %s; treating post as empty", grid_id)
        else:
            raise RuntimeError(f"missing filtered polygons for {grid_id}: {post_path}")
        chips = build_grid_chip_windows(grid_id)
        y_pre = chips_hit_mask(chips, pre)
        y_post = chips_hit_mask(chips, post)
        post_small = post[post["area_m2"] < SMALL_AREA_M2]
        post_large = post[post["area_m2"] >= SMALL_AREA_M2]
        y_small = chips_hit_mask(chips, post_small)
        y_large = chips_hit_mask(chips, post_large)
        n_chips = int(len(chips))
        n_pre_chips = int(y_pre.sum())
        rule = g2_minimum_pass(n_chips, n_pre_chips, recall=G2_RECALL, pass_limit=G3_PASS_RATE)
        row = {
            "grid_id": grid_id,
            "role": role,
            "n_tiles": int(chips["source_tile"].nunique()),
            "n_chips": n_chips,
            "n_pre_polygons": int(len(pre)),
            "n_post_polygons": int(len(post)),
            "n_post_small_polygons": int(len(post_small)),
            "n_post_large_polygons": int(len(post_large)),
            "filtered_gpkg_missing": post_missing,
            "n_chips_pre": n_pre_chips,
            "p_pre": n_pre_chips / n_chips,
            "n_chips_post": int(y_post.sum()),
            "p_post": float(y_post.sum()) / n_chips,
            "n_chips_post_small": int(y_small.sum()),
            "p_post_small": float(y_small.sum()) / n_chips,
            "n_chips_post_large": int(y_large.sum()),
            "p_post_large": float(y_large.sum()) / n_chips,
            "n_chips_post_not_pre": int((y_post & ~y_pre).sum()),
            "continuous_min_pass": rule["continuous_min_pass"],
            "integer_k_min": rule["integer_k_min"],
            "integer_min_pass": rule["integer_min_pass"],
            "stop_rule_triggered": bool(rule["stop"]),
        }
        rows.append(row)
        labels = chips.drop(columns="geometry").copy()
        labels["y_pre"] = y_pre
        labels["y_post"] = y_post
        labels["y_post_small"] = y_small
        labels["y_post_large"] = y_large
        label_dir = STEP0_DIR / "chip_labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        labels.to_csv(label_dir / f"{grid_id}.csv", index=False)
        gap = _count_gap(grid_id, pre, post, inventory)
        gap["filtered_gpkg_missing"] = post_missing
        gap_rows.append(gap)
        log.info(
            "%s chips=%d P_pre=%.4f P_post=%.4f post_not_subset=%d",
            grid_id,
            n_chips,
            row["p_pre"],
            row["p_post"],
            row["n_chips_post_not_pre"],
        )

    rates = pd.DataFrame(rows)
    rates.to_csv(STEP0_DIR / "chip_rates.csv", index=False)
    (STEP0_DIR / "count_gap.json").write_text(
        json.dumps(gap_rows, indent=2) + "\n", encoding="utf-8"
    )
    holdout_rule = g2_minimum_pass(
        int(rates.loc[rates.grid_id == HOLDOUT_PV_GRID, "n_chips"].iloc[0]),
        int(rates.loc[rates.grid_id == HOLDOUT_PV_GRID, "n_chips_pre"].iloc[0]),
        recall=G2_RECALL,
        pass_limit=G3_PASS_RATE,
    )
    feasibility = {
        "verdict": "stop" if holdout_rule["stop"] else "pass",
        "holdout_grid": HOLDOUT_PV_GRID,
        "rule": holdout_rule,
        "selection": selection,
        "checkpoint": checkpoint,
        "chip_enumeration": (
            "SlidingWindowDataset chip_size=400 overlap=0.25 window_origin_mode=anchored "
            "on 2000px geometry-carrier GeoTIFFs. Affine = from_bounds(get_tile_bounds). "
            "Not municipal imagery."
        ),
        "intersection": (
            "Polygons reprojected EPSG:32734 -> tile CRS EPSG:4326. "
            "Hit = valid-window intersection area > 0. No area floor."
        ),
        "vector_files_hashed": int((hashes["sha256"] != "").sum()) if len(hashes) else 0,
    }
    (STEP0_DIR / "feasibility.json").write_text(
        json.dumps(feasibility, indent=2) + "\n", encoding="utf-8"
    )
    _write_result(rows, selection, checkpoint, gap_rows, holdout_rule)
    log.info("wrote %s verdict=%s", RESULT_MD, feasibility["verdict"])
    return 2 if holdout_rule["stop"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.getLogger("step0_vector_feasibility").exception("step 0 failed")
        raise
