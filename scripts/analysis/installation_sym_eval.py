"""installation_sym 诊断 profile — GT 侧兄弟碎片 dissolve 后重匹配 (F1-gap Tier A3 / C3).

三个子命令:

  step0   零新语义:用现有 installation merge profile(pred 侧 many-to-one,
          dae.iou_matching merge_preds=True)@IoU0.5 与 strict 1:1 对照重打
          xdomain60 + CT independent_26,量化「strict 0.36 → area 0.763 的
          差距已被 pred 侧 merge 回收多少」。
  sweep   实测 fragments-per-cluster 随 dissolve gap 的曲线
          (0.5/1/2/3 m;SolarMapper 3 m 为上锚 — 注意该引用是对**预测**像素
          的 proximity 分组先例,不是 GT-side merge 先例)于三个 GT 面:
          CT SAM2 / clean_gt 25 / xdomain60 Li。附 <1 m gap 的 Li module 级
          over-merge audit(PTA 类 grid)。
  sym     installation_sym 诊断 profile:GT 兄弟碎片按 --gap dissolve 后
          重匹配;输出 installation_sym F1@0.5 + 两个 flip counter:
            fn_cluster_to_tp  — baseline 下含 FN 成员的 cluster 在 sym 下 TP
                                (= 切分 artifact 回收)
            tp_to_fn          — baseline 下含 TP 成员的 cluster 在 sym 下 FN
                                (= 部分检出 installation 暴露)

定位(docs/evaluation_protocol.md 全局规则 3):仅诊断 channel,禁入模型
排名主表。GT-side dissolve 是 train20 否决记录明文保留的「允许例外」
(exp_train20_val5_hn_negative_result.md:141 — split_within_gt siblings 的
评估侧 spatial merge),与被否决的训练侧 pre-merge 无关。

与 cluster_level_eval.py 的关系:那是 prediction-bridged clustering(GT 与
pred 互相桥接成分量,会因桥接奖励过涂);本 profile 是 pred-independent 的
GT-side dissolve——评估单元不随预测变化,直接回答「GT 切分 artifact 吃掉了
多少 F1」。两套语义并存且各有引用,见 docs/evaluation_protocol.md §5。

GT-merge 不动 precision:dissolve 只改 GT 侧;unmatched pred(lookalike FP)
计数与 strict/merge baseline 完全相同(xdomain60 P=0.291 / 2,969 FP 不受影响)。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import detect_and_evaluate as dae  # noqa: E402
from core.region_registry import get_region_config  # noqa: E402
from scripts.analysis.area_aggregate_eval import _gt_spec_for  # noqa: E402

XD_CITIES = ["pretoria", "bloemfontein", "durban",
             "east_london", "gqeberha", "pietermaritzburg"]
XD_RUN = "unified_reviewall_A_perdet_sam_maskbox_xdomain_c0925"
XD_GT_ROOT = REPO / "data" / "annotations" / "Vexcel"
XD_PRED_ROOT = REPO / "results" / "vexcel"

CT26_GRIDS = ["G1240", "G1243", "G1244", "G1245", "G1293", "G1294", "G1297",
              "G1298", "G1299", "G1300", "G1349", "G1354", "G1410", "G1411",
              "G1466", "G1467", "G1516", "G1520", "G1521", "G1522", "G1523",
              "G1524", "G1569", "G1570", "G1571", "G1572"]
CT26_RUNS = ["v3c_wave1_perdet", "v3c_wave1_pixelor",
             "unifiedA_wave1_perdet", "unifiedA_wave1_pixelor"]

CLEAN_GT_ROOT = REPO / "data" / "annotations_channel2_clean"
CT_GT_ROOT = REPO / "data" / "annotations" / "Capetown"

MAX_PLAUSIBLE_POLY_M2 = 20_000.0
IOU = 0.5


# ─────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────
def _load_clean(path: Path, metric_crs: str, layer=None) -> gpd.GeoDataFrame:
    g = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if g.empty:
        return g
    if g.crs is None:
        g = g.set_crs(metric_crs)
    g = g.to_crs(metric_crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g = g[g.geometry.is_valid | g.geometry.buffer(0).is_valid]
    g.geometry = g.geometry.buffer(0)
    g = g[g.geometry.area > 0]
    g = g[g.geometry.area <= MAX_PLAUSIBLE_POLY_M2]
    return g.reset_index(drop=True)


def _xd_grid_pairs():
    for city in XD_CITIES:
        metric_crs = get_region_config(city).crs_metric
        pred_root = XD_PRED_ROOT / city / XD_RUN
        gt_root = XD_GT_ROOT / city
        if not pred_root.exists():
            continue
        for gdir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
            pred_path = gdir / "predictions_metric.gpkg"
            gt_path = gt_root / f"{gdir.name}.gpkg"
            if pred_path.exists() and gt_path.exists():
                yield ("xdomain60", city, gdir.name, pred_path, gt_path, None,
                       metric_crs)


def _ct26_grid_pairs(run_id: str):
    region_cfg = get_region_config("cape_town")
    metric_crs = region_cfg.crs_metric
    for g in CT26_GRIDS:
        pred_path = REPO / "results" / "cape_town" / run_id / g / "predictions_metric.gpkg"
        gt_spec = _gt_spec_for(region_cfg, g)
        if pred_path.exists() and gt_spec is not None:
            yield (f"ct26_{run_id}", "cape_town", g, pred_path, gt_spec[0],
                   gt_spec[1], metric_crs)


# ─────────────────────────────────────────────────────────────────────────
# GT-side dissolve
# ─────────────────────────────────────────────────────────────────────────
def dissolve_gt(gt: gpd.GeoDataFrame, gap_m: float):
    """Buffer(+gap/2) → union → buffer(−gap/2) → explode.

    Returns (dissolved_gdf, member_lists):dissolved_gdf 每行一个 cluster
    polygon;member_lists[i] = 原 GT 行号列表(按原 gt 顺序)。
    """
    if len(gt) == 0:
        return gt, []
    buffered = gt.geometry.buffer(gap_m / 2.0)
    merged = unary_union(list(buffered))
    if isinstance(merged, Polygon):
        parts = [merged]
    elif isinstance(merged, MultiPolygon):
        parts = list(merged.geoms)
    else:
        parts = [p for p in getattr(merged, "geoms", []) if isinstance(p, Polygon)]
    # cluster membership:原 polygon 归属于包含其 buffered 形心的 part
    import shapely
    tree = shapely.STRtree(parts)
    members: list[list[int]] = [[] for _ in parts]
    for i, geom in enumerate(gt.geometry):
        hit = tree.query(geom.representative_point())
        assigned = None
        for j in hit:
            if parts[int(j)].intersects(geom):
                assigned = int(j)
                break
        if assigned is None:
            dists = [(p.distance(geom), j) for j, p in enumerate(parts)]
            assigned = min(dists)[1]
        members[assigned].append(i)
    shrunk = []
    keep_members = []
    for part, mem in zip(parts, members):
        if not mem:
            continue
        core = part.buffer(-gap_m / 2.0)
        if core.is_empty or core.area <= 0:
            # 反向 buffer 吃没了(细碎 GT):退回成员原геометрии的 union
            core = unary_union([gt.geometry.iloc[i] for i in mem])
        shrunk.append(core)
        keep_members.append(mem)
    out = gpd.GeoDataFrame(geometry=shrunk, crs=gt.crs)
    return out, keep_members


# ─────────────────────────────────────────────────────────────────────────
# step0
# ─────────────────────────────────────────────────────────────────────────
def _match(gt, pred, merge_preds: bool):
    if len(gt) == 0 or len(pred) == 0:
        return {"tp": 0, "fp": len(pred), "fn": len(gt)}
    r = dae.iou_matching(gt, pred, iou_threshold=IOU, merge_preds=merge_preds)
    return {"tp": r["tp"], "fp": r["fp"], "fn": r["fn"]}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def cmd_step0(args) -> None:
    out_dir = args.output_dir / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = list(_xd_grid_pairs())
    for run in CT26_RUNS:
        suites += list(_ct26_grid_pairs(run))

    rows = []
    for suite, region, grid, pred_path, gt_path, gt_layer, crs in suites:
        try:
            pred = _load_clean(pred_path, crs)
            gt = _load_clean(gt_path, crs, layer=gt_layer)
        except Exception as exc:
            print(f"[warn] {suite}/{grid}: {exc}")
            continue
        strict = _match(gt, pred, merge_preds=False)
        merge = _match(gt, pred, merge_preds=True)
        rows.append({"suite": suite, "region": region, "grid_id": grid,
                     "n_gt": len(gt), "n_pred": len(pred),
                     **{f"strict_{k}": v for k, v in strict.items()},
                     **{f"merge_{k}": v for k, v in merge.items()}})
        print(f"  {suite}/{grid}: strict tp={strict['tp']} fn={strict['fn']} "
              f"| merge tp={merge['tp']} fn={merge['fn']}")

    _write_csv(out_dir / "step0_per_grid.csv", rows)

    # pooled per suite
    summ = []
    df = pd.DataFrame(rows)
    for suite, sub in df.groupby("suite"):
        line = {"suite": suite, "n_grids": len(sub),
                "n_gt": int(sub["n_gt"].sum()), "n_pred": int(sub["n_pred"].sum())}
        for mode in ("strict", "merge"):
            tp = int(sub[f"{mode}_tp"].sum())
            fp = int(sub[f"{mode}_fp"].sum())
            fn = int(sub[f"{mode}_fn"].sum())
            p, r, f = _prf(tp, fp, fn)
            line.update({f"{mode}_tp": tp, f"{mode}_fp": fp, f"{mode}_fn": fn,
                         f"{mode}_P": round(p, 4), f"{mode}_R": round(r, 4),
                         f"{mode}_F1": round(f, 4)})
        line["merge_recovery_F1_pp"] = round(
            (line["merge_F1"] - line["strict_F1"]) * 100, 2)
        summ.append(line)
        print(f"[step0] {suite}: strict F1={line['strict_F1']} → "
              f"merge F1={line['merge_F1']} (回收 {line['merge_recovery_F1_pp']}pp); "
              f"P {line['strict_P']}→{line['merge_P']}, FP {line['strict_fp']}→{line['merge_fp']}")
    _write_csv(out_dir / "step0_summary.csv", summ)
    print(f"[out] {out_dir}/step0_summary.csv")


# ─────────────────────────────────────────────────────────────────────────
# sweep
# ─────────────────────────────────────────────────────────────────────────
def _gt_faces():
    """三个 GT 面:(face, grid, gt_path, layer, metric_crs)。"""
    faces = []
    # CT SAM2(全部 Capetown 标注 grid,经 area_aggregate 同款 auto-discovery)
    region_cfg = get_region_config("cape_town")
    seen = set()
    for f in sorted(CT_GT_ROOT.glob("G*.gpkg")):
        gid = f.stem.split("_")[0]
        if gid in seen:
            continue
        seen.add(gid)
        spec = _gt_spec_for(region_cfg, gid)
        if spec is not None:
            faces.append(("ct_sam2", gid, spec[0], spec[1], region_cfg.crs_metric))
    # clean_gt 25
    for d in sorted(CLEAN_GT_ROOT.iterdir()):
        p = d / f"{d.name}_clean_gt.gpkg"
        if p.exists():
            faces.append(("clean_gt25", d.name, p, None, "EPSG:32735"))
    # xdomain60 Li
    for city in XD_CITIES:
        crs = get_region_config(city).crs_metric
        groot = XD_GT_ROOT / city
        if not groot.exists():
            continue
        for f in sorted(groot.glob("*.gpkg")):
            faces.append((f"xdomain_li_{city}", f.stem, f, None, crs))
    return faces


def cmd_sweep(args) -> None:
    out_dir = args.output_dir / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    gaps = [float(g) for g in args.gaps]
    rows = []
    for face, gid, path, layer, crs in _gt_faces():
        try:
            gt = _load_clean(path, crs, layer=layer)
        except Exception as exc:
            print(f"[warn] {face}/{gid}: {exc}")
            continue
        if len(gt) == 0:
            continue
        for gap in gaps:
            dissolved, members = dissolve_gt(gt, gap)
            n_cluster = len(dissolved)
            sizes = [len(m) for m in members]
            rows.append({
                "face": face, "grid_id": gid, "gap_m": gap,
                "n_gt": len(gt), "n_clusters": n_cluster,
                "fragments_per_cluster": round(len(gt) / n_cluster, 4)
                if n_cluster else None,
                "max_cluster_members": max(sizes) if sizes else 0,
                "n_clusters_ge2": sum(1 for s in sizes if s >= 2),
                "n_clusters_ge5": sum(1 for s in sizes if s >= 5),
                "max_cluster_area_m2": round(
                    max((g.area for g in dissolved.geometry), default=0), 1),
            })
    _write_csv(out_dir / "sweep_per_grid.csv", rows)

    df = pd.DataFrame(rows)
    df["face_group"] = df["face"].str.replace(r"xdomain_li_.*", "xdomain_li",
                                              regex=True)
    summ = (df.groupby(["face_group", "gap_m"])
            .apply(lambda s: pd.Series({
                "n_grids": len(s),
                "total_gt": int(s["n_gt"].sum()),
                "total_clusters": int(s["n_clusters"].sum()),
                "fragments_per_cluster": round(
                    s["n_gt"].sum() / s["n_clusters"].sum(), 4),
                "share_clusters_ge2": round(
                    s["n_clusters_ge2"].sum() / s["n_clusters"].sum(), 4),
                "max_cluster_members": int(s["max_cluster_members"].max()),
            }), include_groups=False)
            .reset_index())
    summ.to_csv(out_dir / "sweep_summary.csv", index=False)
    print(summ.to_string(index=False))
    print(f"[out] {out_dir}/sweep_summary.csv")


# ─────────────────────────────────────────────────────────────────────────
# sym
# ─────────────────────────────────────────────────────────────────────────
def cmd_sym(args) -> None:
    gap = float(args.gap)
    out_dir = args.output_dir / f"sym_gap{gap:g}m"
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = list(_xd_grid_pairs())
    for run in CT26_RUNS:
        suites += list(_ct26_grid_pairs(run))

    rows = []
    for suite, region, grid, pred_path, gt_path, gt_layer, crs in suites:
        try:
            pred = _load_clean(pred_path, crs)
            gt = _load_clean(gt_path, crs, layer=gt_layer)
        except Exception as exc:
            print(f"[warn] {suite}/{grid}: {exc}")
            continue
        if len(gt) == 0:
            continue

        # baseline = 现有 installation merge profile(pred 侧 merge)@0.5
        if len(pred):
            base = dae.iou_matching(gt, pred, iou_threshold=IOU,
                                    merge_preds=True)
            base_matched_gt = set(base.get("matched_gt_indices", []))
        else:
            base = {"tp": 0, "fp": 0, "fn": len(gt)}
            base_matched_gt = set()

        dissolved, members = dissolve_gt(gt, gap)
        if len(pred):
            sym = dae.iou_matching(dissolved, pred, iou_threshold=IOU,
                                   merge_preds=True)
            sym_matched = set(sym.get("matched_gt_indices", []))
        else:
            sym = {"tp": 0, "fp": 0, "fn": len(dissolved)}
            sym_matched = set()

        fn_cluster_to_tp = 0   # 切分 artifact 回收
        tp_to_fn = 0           # 部分检出 installation 暴露
        for ci, mem in enumerate(members):
            mem_matched = [i in base_matched_gt for i in mem]
            if ci in sym_matched and not all(mem_matched):
                fn_cluster_to_tp += 1
            if ci not in sym_matched and any(mem_matched):
                tp_to_fn += 1

        bp, br, bf = _prf(base["tp"], base["fp"], base["fn"])
        sp, sr, sf = _prf(sym["tp"], sym["fp"], sym["fn"])
        rows.append({
            "suite": suite, "region": region, "grid_id": grid,
            "gap_m": gap, "iou_caliber": IOU,
            "eval_profile": "installation_sym",
            "n_gt": len(gt), "n_clusters": len(dissolved), "n_pred": len(pred),
            "base_tp": base["tp"], "base_fp": base["fp"], "base_fn": base["fn"],
            "base_F1": round(bf, 4),
            "sym_tp": sym["tp"], "sym_fp": sym["fp"], "sym_fn": sym["fn"],
            "sym_P": round(sp, 4), "sym_R": round(sr, 4), "sym_F1": round(sf, 4),
            "flip_fn_cluster_to_tp": fn_cluster_to_tp,
            "flip_tp_to_fn": tp_to_fn,
        })
        print(f"  {suite}/{grid}: base F1={bf:.3f} → sym F1={sf:.3f} "
              f"(回收 {fn_cluster_to_tp}, 暴露 {tp_to_fn})")

    _write_csv(out_dir / "sym_per_grid.csv", rows)

    df = pd.DataFrame(rows)
    summ = []
    for suite, sub in df.groupby("suite"):
        line = {"suite": suite, "gap_m": gap, "n_grids": len(sub),
                "eval_profile": "installation_sym", "iou_caliber": IOU}
        for mode in ("base", "sym"):
            tp = int(sub[f"{mode}_tp"].sum())
            fp = int(sub[f"{mode}_fp"].sum())
            fn = int(sub[f"{mode}_fn"].sum())
            p, r, f = _prf(tp, fp, fn)
            line.update({f"{mode}_tp": tp, f"{mode}_fp": fp, f"{mode}_fn": fn,
                         f"{mode}_P": round(p, 4), f"{mode}_R": round(r, 4),
                         f"{mode}_F1": round(f, 4)})
        line["delta_F1_pp"] = round((line["sym_F1"] - line["base_F1"]) * 100, 2)
        line["flip_fn_cluster_to_tp"] = int(sub["flip_fn_cluster_to_tp"].sum())
        line["flip_tp_to_fn"] = int(sub["flip_tp_to_fn"].sum())
        summ.append(line)
        print(f"[sym] {suite}: base F1={line['base_F1']} → sym F1={line['sym_F1']} "
              f"({line['delta_F1_pp']:+.2f}pp; 回收 {line['flip_fn_cluster_to_tp']}, "
              f"暴露 {line['flip_tp_to_fn']}; P {line['base_P']}→{line['sym_P']})")
    _write_csv(out_dir / "sym_summary.csv", summ)
    print(f"[out] {out_dir}/sym_summary.csv")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "results" / "analysis" / "installation_sym")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("step0", help="strict vs pred-merge @0.5 重打 xdomain60 + CT26")
    sp = sub.add_parser("sweep", help="dissolve gap sweep(GT-only 曲线)")
    sp.add_argument("--gaps", nargs="+", default=["0.5", "1", "2", "3"])
    sy = sub.add_parser("sym", help="installation_sym profile + flip counters")
    sy.add_argument("--gap", default="1.0")
    args = ap.parse_args()
    {"step0": cmd_step0, "sweep": cmd_sweep, "sym": cmd_sym}[args.cmd](args)


if __name__ == "__main__":
    main()
