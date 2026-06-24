"""gtnoise_t1_ceiling 评分 harness (F1-gap Tier A4 / C9).

同一份冻结预测、换 GT 重打分:对每个标注窗口,把冻结预测与(a)新 T1 gold GT、
(b)旧 A2 GT 分别裁剪到窗口内打分,报 per-window paired delta 的 bootstrap CI。

指标(每窗口 × 每 GT 侧):
  strict 1:1 polygon F1@0.5、installation merge profile F1@0.5、
  area Tier-1(area_R/P/F1、pred/gt union m²)。
冻结预测按 configs/eval/gtnoise_t1_ceiling.yaml pin 死;merge_mode 字段透传。

headline 规则:GT 噪声天花板估计只从 stratum=representative 池计算;
archetype 池只出分层诊断行。绝不报独立 F1 的 CI —— 只报 paired delta CI。

种子先行版(--seed-pilot):G1238 上用「human 124 GT(T1 候选)vs SAM2 242
GT(A2)」整 grid paired 重打分,在战役标注完成之前给出首个 ceiling 粗估。
该数受两点污染须随行标注:(i) manifest 248≠盘上 124 的对账未完成;
(ii) human GT 未经 A1 checklist 复核,只是 T1 候选。

用法:
  python scripts/analysis/gtnoise_t1_score.py --seed-pilot
  python scripts/analysis/gtnoise_t1_score.py \
      --t1-gpkg /mnt/d/ZAsolar/annotations_inbox/gtnoise_t1_ceiling/t1_windows.gpkg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import detect_and_evaluate as dae  # noqa: E402
from core.polygon_validation import MAX_PLAUSIBLE_POLY_M2  # noqa: E402
from core.region_registry import get_region_config  # noqa: E402
from scripts.analysis.area_aggregate_eval import _gt_spec_for  # noqa: E402

CFG_PATH = REPO / "configs" / "eval" / "gtnoise_t1_ceiling.yaml"
OUT_ROOT = REPO / "results" / "analysis" / "gtnoise_t1_ceiling"
IOU = 0.5


def _clean(g: gpd.GeoDataFrame, metric_crs: str) -> gpd.GeoDataFrame:
    if g.empty:
        return g
    if g.crs is None:
        g = g.set_crs(metric_crs)
    g = g.to_crs(metric_crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g.geometry = g.geometry.buffer(0)
    g = g[(g.geometry.area > 0) & (g.geometry.area <= MAX_PLAUSIBLE_POLY_M2)]
    return g.reset_index(drop=True)


def _clip(g: gpd.GeoDataFrame, window) -> gpd.GeoDataFrame:
    """窗口裁剪:保留形心在窗口内的 polygon(整体保留,不切几何 ——
    切几何会人为制造碎片改变 F1 语义)。"""
    if g.empty:
        return g
    keep = g.geometry.representative_point().within(window)
    return g[keep].reset_index(drop=True)


def _score_pair(gt: gpd.GeoDataFrame, pred: gpd.GeoDataFrame) -> dict:
    out = {"n_gt": len(gt), "n_pred": len(pred)}
    for label, merge in (("strict", False), ("merge", True)):
        if len(gt) == 0 and len(pred) == 0:
            tp = fp = fn = 0
        elif len(pred) == 0:
            tp, fp, fn = 0, 0, len(gt)
        elif len(gt) == 0:
            tp, fp, fn = 0, len(pred), 0
        else:
            r = dae.iou_matching(gt, pred, iou_threshold=IOU, merge_preds=merge)
            tp, fp, fn = r["tp"], r["fp"], r["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        out.update({f"{label}_tp": tp, f"{label}_fp": fp, f"{label}_fn": fn,
                    f"{label}_f1_iou05": round(f1, 4)})
    # area Tier-1(窗口内 set-theoretic)
    gu = gt.geometry.union_all() if len(gt) else None
    pu = pred.geometry.union_all() if len(pred) else None
    ga = float(gu.area) if gu is not None else 0.0
    pa = float(pu.area) if pu is not None else 0.0
    inter = float(pu.intersection(gu).area) if (gu is not None and pu is not None) else 0.0
    ar = inter / ga if ga else None
    ap = inter / pa if pa else None
    af = (2 * ar * ap / (ar + ap)) if ar and ap and (ar + ap) else (0.0 if ga or pa else None)
    out.update({"gt_union_m2": round(ga, 1), "pred_union_m2": round(pa, 1),
                "area_R": None if ar is None else round(ar, 4),
                "area_P": None if ap is None else round(ap, 4),
                "area_F1": None if af is None else round(af, 4)})
    return out


def _paired_bootstrap_ci(deltas: np.ndarray, n_boot: int = 10_000,
                         seed: int = 20260610) -> tuple[float, float]:
    """per-window delta 的 paired bootstrap 95% CI(重采样窗口)。"""
    rng = np.random.default_rng(seed)
    if len(deltas) == 0:
        return float("nan"), float("nan")
    means = [float(np.mean(rng.choice(deltas, size=len(deltas), replace=True)))
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_frozen_preds(cfg: dict, region: str, metric_crs: str) -> gpd.GeoDataFrame:
    spec = cfg["frozen_predictions"][region]
    root = REPO / spec["root"]
    frames = []
    for gdir in sorted(p for p in root.iterdir() if p.is_dir()):
        p = gdir / spec["file"]
        if p.exists():
            g = gpd.read_file(p)
            if len(g):
                frames.append(g.to_crs(metric_crs) if g.crs else g.set_crs(metric_crs))
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=metric_crs)


def _old_gt_for_window(cfg: dict, region: str, grid_id: str, metric_crs: str):
    spec = cfg["old_gt"][region]
    if spec["pattern"] == "auto":
        gspec = _gt_spec_for(get_region_config("cape_town"), grid_id)
        if gspec is None:
            return None
        g = gpd.read_file(gspec[0], layer=gspec[1]) if gspec[1] else gpd.read_file(gspec[0])
    else:
        p = REPO / spec["root"] / spec["pattern"].format(grid=grid_id)
        if not p.exists():
            return None
        g = gpd.read_file(p)
    return _clean(g, metric_crs)


def cmd_windows(args) -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    t1 = gpd.read_file(args.t1_gpkg)
    req = set(cfg["t1_annotations"]["schema"]["required_fields"]) - {"geometry"}
    missing = req - set(t1.columns)
    if missing:
        raise SystemExit(f"[SCHEMA] T1 标注缺字段: {sorted(missing)} "
                         f"(见 configs/eval/gtnoise_t1_ceiling.yaml)")

    windows = {}
    for region in ("jhb", "ct"):
        p = REPO / cfg["sampling"]["windows_gpkg"][region]
        if p.exists():
            w = gpd.read_file(p)
            windows[region] = w[w["stratum"] != ""] if "stratum" in w.columns else w

    out_rows = []
    for region, wdf in windows.items():
        metric_crs = cfg["frozen_predictions"][region]["metric_crs"]
        preds = _load_frozen_preds(cfg, region, metric_crs)
        t1r = _clean(t1, metric_crs)
        for _, w in wdf.iterrows():
            wid = w["window_id"]
            geom = w.geometry
            pred_w = _clip(preds, geom)
            t1_w = t1r[t1r["window_id"] == wid] if "window_id" in t1r.columns \
                else _clip(t1r, geom)
            old = _old_gt_for_window(cfg, region, w["grid_id"], metric_crs)
            old_w = _clip(old, geom) if old is not None else None

            row = {"window_id": wid, "region": region, "grid_id": w["grid_id"],
                   "stratum": w.get("stratum", ""),
                   "gt_source": "t1_gold",
                   "merge_mode": cfg["frozen_predictions"][region]["merge_mode"],
                   "iou_caliber": IOU}
            row.update({f"t1_{k}": v for k, v in
                        _score_pair(_clip(t1_w, geom), pred_w).items()})
            if old_w is not None:
                row.update({f"old_{k}": v for k, v in
                            _score_pair(old_w, pred_w).items()})
                for m in ("strict_f1_iou05", "merge_f1_iou05", "area_F1"):
                    a, b = row.get(f"t1_{m}"), row.get(f"old_{m}")
                    row[f"delta_{m}"] = (round(a - b, 4)
                                         if a is not None and b is not None else None)
            out_rows.append(row)

    out_dir = OUT_ROOT / "score"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(out_dir / "per_window.csv", index=False)

    # headline:representative 池 paired delta + bootstrap CI
    summ = []
    for strata, tag in ((cfg["sampling"]["headline_strata"], "HEADLINE_representative"),
                        (cfg["sampling"]["diagnostic_strata"], "diagnostic_archetype")):
        sub = df[df["stratum"].isin(strata)]
        if sub.empty:
            continue
        line = {"pool": tag, "n_windows": len(sub)}
        for m in ("strict_f1_iou05", "merge_f1_iou05", "area_F1"):
            d = sub[f"delta_{m}"].dropna().to_numpy(float) \
                if f"delta_{m}" in sub.columns else np.array([])
            if len(d):
                lo, hi = _paired_bootstrap_ci(d)
                line[f"mean_delta_{m}"] = round(float(d.mean()), 4)
                line[f"delta_{m}_CI95"] = f"[{lo:.4f},{hi:.4f}]"
        summ.append(line)
    pd.DataFrame(summ).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    print(f"[out] {out_dir}/per_window.csv, summary.csv")


def cmd_seed_pilot(args) -> None:
    """G1238 整 grid paired 粗估(human T1 候选 vs SAM2 A2,同一冻结预测)。"""
    cfg = yaml.safe_load(CFG_PATH.read_text())
    metric_crs = cfg["frozen_predictions"]["ct"]["metric_crs"]
    grid = cfg["seed_t1"]["grid"]
    pred_path = (REPO / cfg["frozen_predictions"]["ct"]["root"] / grid /
                 cfg["frozen_predictions"]["ct"]["file"])
    if not pred_path.exists():
        # wave1 只含 independent_26;种子 G1238 用同 lineage 同 merge-mode 的
        # 本地补跑(unified_A per-det,v4_canonical,2026-06-10)
        alt = (REPO / "results/cape_town/gtnoise_seed_unifiedA_perdet" / grid /
               "predictions_metric.gpkg")
        print(f"[note] {pred_path.relative_to(REPO)} 不存在(G1238 ∉ wave1);"
              f"改用 {alt.relative_to(REPO)}(unified_A per-det 本地补跑,"
              "同 lineage 同 mode,与战役 pin 的差异仅 run 批次)。")
        pred_path = alt
    pred = _clean(gpd.read_file(pred_path), metric_crs)
    cands = cfg["seed_t1"]["on_disk_candidates"]
    gts = {Path(c["path"]).name: _clean(gpd.read_file(REPO / c["path"]), metric_crs)
           for c in cands}

    rows = []
    for name, gt in gts.items():
        r = _score_pair(gt, pred)
        r["gt_file"] = name
        r["grid_id"] = grid
        r["gt_source"] = "t1_gold_candidate" if "SAM2" not in name else "a2_sam2"
        rows.append(r)
        print(f"[seed] {name}: n_gt={r['n_gt']} strict F1={r['strict_f1_iou05']} "
              f"merge F1={r['merge_f1_iou05']} area_F1={r['area_F1']}")
    t1r, a2r = rows[0], rows[1]
    if "SAM2" in t1r["gt_file"]:
        t1r, a2r = a2r, t1r
    print(f"[seed-pilot] G1238 ceiling 粗估 (merge F1): A2={a2r['merge_f1_iou05']} "
          f"→ T1cand={t1r['merge_f1_iou05']} "
          f"(delta {t1r['merge_f1_iou05'] - a2r['merge_f1_iou05']:+.4f}); "
          "n=1 grid、预测源非战役 pin、T1 候选未过 A1 复核 — 仅作先行参考。")
    out_dir = OUT_ROOT / "seed_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "g1238_paired.csv", index=False)
    print(f"[out] {out_dir}/g1238_paired.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t1-gpkg", type=Path, default=None,
                    help="RA 交付的 T1 窗口标注 gpkg(战役主评分)")
    ap.add_argument("--seed-pilot", action="store_true",
                    help="G1238 种子 paired 粗估(战役前先行版)")
    args = ap.parse_args()
    if args.seed_pilot:
        cmd_seed_pilot(args)
    elif args.t1_gpkg:
        cmd_windows(args)
    else:
        raise SystemExit("需要 --seed-pilot 或 --t1-gpkg 之一")


if __name__ == "__main__":
    main()
