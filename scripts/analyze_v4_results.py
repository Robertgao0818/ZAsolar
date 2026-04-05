#!/usr/bin/env python3
"""Analyze V4 batch 004 inference results."""
import csv
from pathlib import Path

GRIDS = [
    "G1855", "G1856", "G1862", "G1863", "G1864",
    "G1909", "G1910", "G1911", "G1917", "G1919", "G1920", "G1921",
    "G1966", "G1970", "G1971", "G1972", "G1973", "G1974", "G1975", "G1976", "G1981",
    "G2025", "G2026", "G2027", "G2028", "G2029", "G2030", "G2031", "G2032", "G2037", "G2038",
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_metrics(grid_id, iou="0.5"):
    p = RESULTS_DIR / grid_id / "presence_metrics.csv"
    if not p.exists():
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("iou_threshold", "") in (iou, f"{float(iou):.2f}"):
            return r
    return rows[0] if rows else None


def main():
    print("=== V4 Batch 004 Results (IoU=0.5) ===\n")
    header = f"{'Grid':<8} {'GT':>5} {'Pred':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))

    total_tp = total_fp = total_fn = total_gt = total_pred = 0
    grid_results = []

    for g in sorted(GRIDS):
        r = load_metrics(g)
        if r is None:
            print(f"{g:<8}  -- no results --")
            continue

        gt = int(r.get("gt_count", r.get("n_gt", 0)))
        pred = int(r.get("pred_count", r.get("n_pred", 0)))
        tp = int(r["tp"])
        fp = int(r["fp"])
        fn = int(r["fn"])
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        print(f"{g:<8} {gt:>5} {pred:>5} {tp:>5} {fp:>5} {fn:>5} {prec:>6.1%} {rec:>6.1%} {f1:>6.1%}")

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += gt
        total_pred += pred
        grid_results.append(dict(grid=g, gt=gt, pred=pred, tp=tp, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1))

    print("-" * len(header))
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"{'TOTAL':<8} {total_gt:>5} {total_pred:>5} {total_tp:>5} {total_fp:>5} {total_fn:>5} {p:>6.1%} {r:>6.1%} {f1:>6.1%}")

    print(f"\nFP rate: {total_fp / total_pred * 100:.1f}% ({total_fp}/{total_pred})")
    print(f"FN rate: {total_fn / total_gt * 100:.1f}% ({total_fn}/{total_gt})")

    # Highlight worst FP grids
    print("\n=== Worst FP grids (by FP count) ===")
    by_fp = sorted(grid_results, key=lambda x: x["fp"], reverse=True)[:10]
    for r in by_fp:
        fp_rate = r["fp"] / (r["tp"] + r["fp"]) * 100 if (r["tp"] + r["fp"]) else 0
        print(f"  {r['grid']}: {r['fp']} FP ({fp_rate:.0f}% of predictions), P={r['prec']:.1%}")

    # Highlight low precision grids (potential water heater areas)
    print("\n=== Low precision grids (P < 40%) ===")
    low_p = [r for r in grid_results if r["prec"] < 0.4 and r["pred"] > 5]
    for r in sorted(low_p, key=lambda x: x["prec"]):
        print(f"  {r['grid']}: P={r['prec']:.1%}, {r['fp']} FP, {r['tp']} TP (GT={r['gt']})")


if __name__ == "__main__":
    main()
