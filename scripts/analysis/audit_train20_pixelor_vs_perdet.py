"""Audit: train20 pixel-or+v4_agg vs train20 per-det+SAM vs V3-C+SAM.

Run after scripts/analysis/rebuild_train20_pixelor.sh completes.
Produces a per-grid TP/FP/FN + bulk_ratio table to test whether the
postproc-only fix (pixel-or merge + v4_agg conf>=0.65) closes the gap
on small-B grids.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path("/home/gaosh/projects/ZAsolar")
GRIDS = [
    "G0772","G0773","G0774","G0775","G0776","G0814","G0815","G0816","G0817","G0818",
    "G0853","G0854","G0855","G0856","G0857","G0888","G0889","G0890","G0891","G0892",
    "G0922","G0923","G0924","G0925","G0926",
]

GT_TPL = "data/annotations_channel2_clean/{g}/{g}_clean_gt.gpkg"
MODELS = {
    "V3C+SAM_v4agg": "results/johannesburg/v3c_sam_maskbox_vexcel_2024_v4_agg/{g}/predictions_metric.gpkg",
    "train20_perdet+SAM": "results/analysis/v3c_failed_weight_compare/perdet/train20_val5_hn_perdet_sam_maskbox/{g}/predictions_metric.gpkg",
    "train20_pixor+SAM+v4agg": "results/analysis/v3c_failed_weight_compare/pixelor/train20_val5_hn_pixelor_sam_maskbox_v4agg/{g}/predictions_metric.gpkg",
}
IOU_THRESH = 0.3


def load(p: Path) -> gpd.GeoDataFrame | None:
    if not p.exists():
        return None
    g = gpd.read_file(p)
    if "EPSG:32735" not in str(g.crs):
        g = g.to_crs("EPSG:32735")
    return g[g.geometry.is_valid & ~g.geometry.is_empty].reset_index(drop=True)


def audit(gt: gpd.GeoDataFrame, pred: gpd.GeoDataFrame) -> dict:
    if pred is None or len(pred) == 0:
        return {"n_pred": 0, "TP": 0, "FP": 0, "FN": len(gt), "FP_area_m2": 0.0, "TP_area_m2": 0.0, "A_total_m2": 0.0}
    gt_sidx = gt.sindex if len(gt) else None
    matched_gt: set[int] = set()
    tp = 0; fp = 0
    fp_a = 0.0; tp_a = 0.0
    for _, prow in pred.iterrows():
        pg = prow.geometry
        is_tp = False
        if gt_sidx is not None:
            for gi in gt_sidx.intersection(pg.bounds):
                if gi in matched_gt: continue
                gg = gt.iloc[gi].geometry
                inter = pg.intersection(gg).area
                if inter <= 0: continue
                union = pg.union(gg).area
                if union > 0 and inter / union >= IOU_THRESH:
                    matched_gt.add(gi)
                    tp += 1; is_tp = True; tp_a += pg.area
                    break
        if not is_tp:
            fp += 1; fp_a += pg.area
    return {
        "n_pred": len(pred),
        "TP": tp, "FP": fp, "FN": len(gt) - len(matched_gt),
        "FP_area_m2": round(fp_a, 1),
        "TP_area_m2": round(tp_a, 1),
        "A_total_m2": round(pred.geometry.area.sum(), 1),
    }


def main():
    rows = []
    for g in GRIDS:
        gt = load(ROOT / GT_TPL.format(g=g))
        if gt is None:
            continue
        B = float(gt.geometry.area.sum())
        for mname, mtpl in MODELS.items():
            pred = load(ROOT / mtpl.format(g=g))
            stats = audit(gt, pred)
            P = stats["TP"] / stats["n_pred"] if stats["n_pred"] else 0.0
            R = stats["TP"] / (stats["TP"] + stats["FN"]) if (stats["TP"] + stats["FN"]) else 0.0
            ratio = stats["A_total_m2"] / B if B > 0 else 0.0
            rows.append({
                "grid": g, "B_m2": round(B, 1), "model": mname,
                **stats,
                "P": round(P, 3), "R": round(R, 3),
                "bulk_ratio": round(ratio, 3),
            })
    df = pd.DataFrame(rows)
    out_csv = ROOT / "results/analysis/v3c_failed_weight_compare/pixelor/per_grid_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")

    # Pivot summary: small-B grids first
    df_sorted = df.assign(B_rank=df.groupby("grid")["B_m2"].transform("first")).sort_values(["B_rank", "grid", "model"])
    print()
    cols = ["grid","B_m2","model","n_pred","TP","FP","FN","FP_area_m2","P","R","bulk_ratio"]
    fmt = "{grid:6} {B_m2:>8.0f} {model:25} {n_pred:>5d} {TP:>4d} {FP:>4d} {FN:>4d} {FP_area_m2:>9.0f} {P:>5.2f} {R:>5.2f} {bulk_ratio:>5.2f}"
    print(f"{'grid':6} {'B(m²)':>8} {'model':25} {'#pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'FP_area':>9} {'P':>5} {'R':>5} {'ratio':>5}")
    for _, r in df_sorted.iterrows():
        print(fmt.format(**{k: r[k] for k in cols}))

    # Aggregate
    print("\n=== Mean per model (across 25 grids) ===")
    agg = df.groupby("model")[["n_pred","TP","FP","FN","FP_area_m2","P","R","bulk_ratio"]].mean()
    print(agg.round(2).to_string())


if __name__ == "__main__":
    main()
