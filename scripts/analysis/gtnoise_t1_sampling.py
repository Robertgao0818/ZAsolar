"""gtnoise_t1_ceiling 窗口抽样 (F1-gap Tier A4 / C9).

为 T1 gold paired re-score 战役抽取标注窗口:
  ~60% 代表性核心 — 按检测密度比例采样(JHB CBD25 + CT independent_26 范围)
                    → 全域 GT 噪声天花板 headline 只从这一池计算。
  ~40% failure-archetype 过采样 — 仅作分层诊断,绝不进 headline
                    (防止过采样池抬高天花板估计)。

输出窗口带显式 `stratum` 字段;所有下游输出带 `gt_source=t1_gold`。
命名约束:独立诊断 channel,不叫 ch2_*,不触 compute_ch2_recall /
area_aggregate_eval 默认(feedback_eval_gt_lock_clean)。

Archetype 操作化(向量可计算的三类自动打标;阴影/浅色屋顶需要影像,
留 MANUAL 占位由 RA 按 docs/handoffs/2026-06-10-gtnoise-t1-ceiling-ra-protocol.md
从候选 gallery 人工选):
  arch_large   窗口含 ≥1 个 ≥500 m² 预测
  arch_dense   窗口预测数 ≥ 全体窗口 P90(稠密 multi-array 屋顶)
  arch_wobble  窗口预测平均边界复杂度(perimeter / 2√(π·area))≥ P90
               (SAM-wobble / V3C-halo 几何噪声代理)
  arch_shadow_MANUAL / arch_lightroof_MANUAL  人工选,脚本只出候选 gallery

冻结预测源(pin 在 configs/eval/gtnoise_t1_ceiling.yaml,评分 harness 共用):
  JHB: unified_reviewall_A full382 per-det+SAM @c=0.925 (nms01_c0925 production)
  CT : unifiedA_wave1_perdet (2026-06-08 CLS-only 锁定链的 detector 侧)

用法:
  python scripts/analysis/gtnoise_t1_sampling.py \
      --n-representative 36 --n-archetype 24 --window-m 200 --seed 20260610
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

JHB_TASK_GRID = REPO / "data" / "jhb_task_grid_unified.gpkg"
CT_TASK_GRID = REPO / "data" / "task_grid.gpkg"

JHB_CBD25 = ["G0772", "G0773", "G0774", "G0775", "G0776", "G0814", "G0815",
             "G0816", "G0817", "G0818", "G0853", "G0854", "G0855", "G0856",
             "G0857", "G0888", "G0889", "G0890", "G0891", "G0892", "G0922",
             "G0923", "G0924", "G0925", "G0926"]
CT26 = ["G1240", "G1243", "G1244", "G1245", "G1293", "G1294", "G1297",
        "G1298", "G1299", "G1300", "G1349", "G1354", "G1410", "G1411",
        "G1466", "G1467", "G1516", "G1520", "G1521", "G1522", "G1523",
        "G1524", "G1569", "G1570", "G1571", "G1572"]

JHB_PRED_ROOT = (REPO / "results" / "johannesburg" /
                 "unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox")
JHB_PRED_FILE = "predictions_metric_nms01_c0925.gpkg"   # production variant
CT_PRED_ROOT = REPO / "results" / "cape_town" / "unifiedA_wave1_perdet"
CT_PRED_FILE = "predictions_metric.gpkg"

JHB_CRS = "EPSG:32735"
CT_CRS = "EPSG:32734"

LARGE_M2 = 500.0


def _load_cells(task_grid: Path, grid_ids: list[str], metric_crs: str):
    tg = gpd.read_file(task_grid)
    id_col = next(c for c in ("gridcell_id", "grid_id", "id", "name")
                  if c in tg.columns)
    tg = tg[tg[id_col].isin(grid_ids)].to_crs(metric_crs)
    return tg, id_col


def _jhb_pred_frames(scope_union, metric_crs):
    """full382 JNB 格网中与 CBD25 范围相交的 cell 的 production 预测。"""
    frames = []
    for gdir in sorted(JHB_PRED_ROOT.iterdir()):
        if not gdir.is_dir():
            continue
        p = gdir / JHB_PRED_FILE
        if not p.exists():
            continue
        import pyogrio
        info = pyogrio.read_info(p)
        # cheap bbox prefilter before full read
        b = info.get("total_bounds")
        if b is not None and not scope_union.intersects(box(*b)):
            continue
        g = gpd.read_file(p)
        if g.empty:
            continue
        if g.crs is None or str(g.crs) != metric_crs:
            g = g.to_crs(metric_crs)
        g = g[g.geometry.representative_point().within(scope_union)]
        if len(g):
            g["src_grid"] = gdir.name
            frames.append(g)
    return frames


def _ct_pred_frames(metric_crs):
    frames = []
    for gid in CT26:
        p = CT_PRED_ROOT / gid / CT_PRED_FILE
        if not p.exists():
            continue
        g = gpd.read_file(p)
        if g.empty:
            continue
        if g.crs is None or str(g.crs) != metric_crs:
            g = g.to_crs(metric_crs)
        g["src_grid"] = gid
        frames.append(g)
    return frames


def build_windows(cells: gpd.GeoDataFrame, id_col: str, window_m: float,
                  metric_crs: str, region: str) -> gpd.GeoDataFrame:
    rows = []
    for _, cell in cells.iterrows():
        minx, miny, maxx, maxy = cell.geometry.bounds
        nx = max(1, int(round((maxx - minx) / window_m)))
        ny = max(1, int(round((maxy - miny) / window_m)))
        wx = (maxx - minx) / nx
        wy = (maxy - miny) / ny
        for i in range(nx):
            for j in range(ny):
                geom = box(minx + i * wx, miny + j * wy,
                           minx + (i + 1) * wx, miny + (j + 1) * wy)
                geom = geom.intersection(cell.geometry)
                if geom.is_empty or geom.area < 0.5 * wx * wy:
                    continue
                rows.append({
                    "window_id": f"{region}_{cell[id_col]}_{i}_{j}",
                    "region": region,
                    "grid_id": cell[id_col],
                    "geometry": geom,
                })
    return gpd.GeoDataFrame(rows, crs=metric_crs)


def annotate_windows(windows: gpd.GeoDataFrame, preds: gpd.GeoDataFrame):
    """per-window pred 计数 + archetype 自动打标字段。"""
    pts = preds.copy()
    pts["geometry"] = pts.geometry.representative_point()
    pts["pred_area_m2"] = preds.geometry.area
    per = preds.geometry.length
    pts["shape_cx"] = per / (2 * np.sqrt(np.pi * preds.geometry.area))
    joined = gpd.sjoin(pts, windows[["window_id", "geometry"]],
                       how="inner", predicate="within")
    agg = joined.groupby("window_id").agg(
        n_pred=("pred_area_m2", "size"),
        max_pred_m2=("pred_area_m2", "max"),
        mean_pred_m2=("pred_area_m2", "mean"),
        mean_shape_cx=("shape_cx", "mean"),
        mean_conf=("confidence", "mean")
        if "confidence" in joined.columns else ("pred_area_m2", "size"),
    )
    out = windows.merge(agg, on="window_id", how="left")
    out["n_pred"] = out["n_pred"].fillna(0).astype(int)
    dense_thr = float(np.percentile(out.loc[out["n_pred"] > 0, "n_pred"], 90)) \
        if (out["n_pred"] > 0).any() else np.inf
    wob_thr = float(np.nanpercentile(out["mean_shape_cx"], 90)) \
        if out["mean_shape_cx"].notna().any() else np.inf
    out["arch_large"] = out["max_pred_m2"].fillna(0) >= LARGE_M2
    out["arch_dense"] = out["n_pred"] >= max(dense_thr, 2)
    out["arch_wobble"] = out["mean_shape_cx"].fillna(0) >= wob_thr
    return out


def sample(windows: gpd.GeoDataFrame, n_rep: int, n_arch: int, seed: int):
    rng = np.random.default_rng(seed)
    w = windows.copy()
    w["stratum"] = ""

    # 代表性核心:按检测密度比例(无放回加权)
    cand = w[w["n_pred"] > 0]
    weights = cand["n_pred"] / cand["n_pred"].sum()
    n_rep = min(n_rep, len(cand))
    rep_idx = rng.choice(cand.index.to_numpy(), size=n_rep, replace=False,
                         p=weights.to_numpy())
    w.loc[rep_idx, "stratum"] = "representative"

    # archetype 过采样:三类自动 stratum 平分名额,排除已选
    auto_strata = ["arch_large", "arch_dense", "arch_wobble"]
    per_stratum = max(1, n_arch // len(auto_strata))
    for st in auto_strata:
        pool = w[(w[st]) & (w["stratum"] == "")]
        take = min(per_stratum, len(pool))
        if take == 0:
            continue
        idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        w.loc[idx, "stratum"] = st
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-m", type=float, default=200.0)
    ap.add_argument("--n-representative", type=int, default=36,
                    help="代表性核心窗口数(两区合计,~60%%)")
    ap.add_argument("--n-archetype", type=int, default=24,
                    help="failure-archetype 窗口数(两区合计,~40%%)")
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "results" / "analysis" / "gtnoise_t1_ceiling"
                    / "sampling")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_w = []
    for region, task_grid, grid_ids, crs, pred_loader in (
        ("jhb", JHB_TASK_GRID, JHB_CBD25, JHB_CRS, None),
        ("ct", CT_TASK_GRID, CT26, CT_CRS, None),
    ):
        cells, id_col = _load_cells(task_grid, grid_ids, crs)
        if cells.empty:
            print(f"[warn] {region}: no task-grid cells found for scope")
            continue
        print(f"[{region}] {len(cells)} cells")
        windows = build_windows(cells, id_col, args.window_m, crs, region)
        scope = cells.geometry.union_all() if hasattr(cells.geometry, "union_all") \
            else cells.unary_union
        frames = (_jhb_pred_frames(scope, crs) if region == "jhb"
                  else _ct_pred_frames(crs))
        if not frames:
            print(f"[warn] {region}: no predictions found — windows kept w/o density")
            preds = gpd.GeoDataFrame(geometry=[], crs=crs)
        else:
            preds = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)
        print(f"[{region}] {len(windows)} candidate windows, {len(preds)} preds")
        windows = annotate_windows(windows, preds)
        # 统一 4326 以便跨区 pool(密度字段已在 metric CRS 下算好)
        all_w.append(windows.to_crs("EPSG:4326"))

    # 跨区合并后统一抽样(代表性权重 = 检测数,两区一体按密度比例)
    pooled = gpd.GeoDataFrame(pd.concat(all_w, ignore_index=True))
    sampled = sample(pooled, args.n_representative, args.n_archetype, args.seed)
    sampled["gt_source"] = "t1_gold"
    sampled["sampling_seed"] = args.seed
    sampled["frozen_pred_jhb"] = f"{JHB_PRED_ROOT.name}/{JHB_PRED_FILE}"
    sampled["frozen_pred_ct"] = f"{CT_PRED_ROOT.name}/{CT_PRED_FILE}"

    picked = sampled[sampled["stratum"] != ""]
    # 全候选 + 选中两份(均 EPSG:4326,QGIS 直接加载)
    for region in ("jhb", "ct"):
        sub_all = sampled[sampled["region"] == region]
        sub = picked[picked["region"] == region]
        if sub_all.empty:
            continue
        sub_all.to_file(out / f"windows_candidates_{region}.gpkg", driver="GPKG")
        if not sub.empty:
            sub.to_file(out / f"windows_selected_{region}.geojson",
                        driver="GeoJSON")
    picked.drop(columns="geometry").to_csv(out / "windows_selected.csv",
                                           index=False)

    print("\n[selected]")
    print(picked.groupby(["region", "stratum"]).size().to_string())
    print(f"\n[out] {out}/windows_selected.csv (+ per-region gpkg/geojson)")
    print("[note] arch_shadow_MANUAL / arch_lightroof_MANUAL 由 RA 按协议文档"
          "从 windows_candidates_*.gpkg 人工补选(脚本不自动判定阴影/屋顶色)。")


if __name__ == "__main__":
    main()
