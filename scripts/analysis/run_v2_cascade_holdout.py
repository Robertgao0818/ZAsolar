"""V2 cascade holdout eval with per-imagery thresholds.

Re-runs the 17-grid (CT 7 + JHB 10) holdout cascade for cls_pv_thermal_v2,
using per-imagery thresholds from configs/classifier/thresholds_v2.json.

Compared to v1 (which used a single threshold of 0.5):
  - aerial_2025 (CT): per-backbone calibrated threshold from v2 calibration
  - aerial_2023 (JHB suburb): per-backbone calibrated threshold

(GEID layer thresholds exist in thresholds_v2.json but no GEID grids are in the
17-grid holdout; the GEID benefit shows up in the JHB CBD 25-grid eval.)

Outputs mirror the v1 layout under
results/analysis/cls_cascade_holdout_v2/<backbone>_perlayer/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = Path.home() / "zasolar_data" / "tiles"

# 17-grid holdout (G1977 has no local GT — kept for classifier filtering counts
# but excluded from installation aggregate per v1 convention).
# tiles_root is the layer-specific dir because classify_predictions._find_tile
# expects a flat <root>/<grid>/ layout, so we point it at the layer dir which
# does have that layout under it.
HOLDOUT = {
    "ct": {
        "model_run": "v3c_targeted_hn_aerial_2025",
        "imagery_layer": "aerial_2025",
        "region_arg": "cape_town",
        "tiles_root": DATA_ROOT / "cape_town" / "aerial_2025",
        "grids": ["G1971", "G1973", "G1977", "G1981", "G2027", "G2029", "G2032"],
        "missing_gt_grids": {"G1977"},
    },
    "jhb": {
        "model_run": "v4_aerial_2023",
        "imagery_layer": "aerial_2023",
        "region_arg": "johannesburg",
        "tiles_root": DATA_ROOT / "johannesburg" / "aerial_2023",
        "grids": ["G0856", "G0890", "G0892", "G1110", "G1111",
                  "G1144", "G1146", "G1183", "G1250", "G1253"],
        "missing_gt_grids": set(),
    },
}

BACKBONES = {
    "effb0":    ("efficientnet_b0", "checkpoints/cls_pv_thermal_v2_efficientnet_b0/best_cls.pth"),
    "convnext": ("convnext_tiny",   "checkpoints/cls_pv_thermal_v2_convnext_tiny/best_cls.pth"),
    "dinov2":   ("dinov2_vits14",   "checkpoints/cls_pv_thermal_v2_dinov2_vits14/best_cls.pth"),
}


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def load_presence(metrics_csv: Path) -> dict | None:
    if not metrics_csv.exists():
        return None
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return {
        "gt_count":  int(row["gt_count"]),
        "pred_count": int(row["pred_count"]),
        "tp": int(row["tp"]), "fp": int(row["fp"]), "fn": int(row["fn"]),
        "precision": float(row["precision"]),
        "recall":    float(row["recall"]),
        "f1":        float(row["f1"]),
    }


def aggregate_region(rows: list[dict], region: str) -> dict:
    sub = [r for r in rows if r["region"] == region and r["variant"] == "filtered"
           and r["gt_count"] > 0]
    sub_raw = [r for r in rows if r["region"] == region and r["variant"] == "raw"
               and r["gt_count"] > 0]
    if not sub or not sub_raw:
        return {}
    tp_f = sum(r["tp"] for r in sub); fp_f = sum(r["fp"] for r in sub); fn_f = sum(r["fn"] for r in sub)
    tp_r = sum(r["tp"] for r in sub_raw); fp_r = sum(r["fp"] for r in sub_raw); fn_r = sum(r["fn"] for r in sub_raw)
    p_f = tp_f / max(tp_f + fp_f, 1); r_f = tp_f / max(tp_f + fn_f, 1); f1_f = 2*p_f*r_f / max(p_f+r_f, 1e-9)
    p_r = tp_r / max(tp_r + fp_r, 1); r_r = tp_r / max(tp_r + fn_r, 1); f1_r = 2*p_r*r_r / max(p_r+r_r, 1e-9)
    fp_removed = fp_r - fp_f
    tp_lost = tp_r - tp_f
    return {
        "region": region,
        "eval_grids": len(sub),
        "filtered_precision": p_f, "filtered_recall": r_f, "filtered_f1": f1_f,
        "raw_precision": p_r, "raw_recall": r_r, "raw_f1": f1_r,
        "precision_delta": p_f - p_r, "recall_delta": r_f - r_r, "f1_delta": f1_f - f1_r,
        "fp_removed": int(fp_removed), "tp_lost": int(tp_lost),
        "pred_removed": int((tp_r + fp_r) - (tp_f + fp_f)),
    }


def aggregate_overall(rows: list[dict]) -> dict:
    sub_f = [r for r in rows if r["variant"] == "filtered" and r["gt_count"] > 0]
    sub_r = [r for r in rows if r["variant"] == "raw" and r["gt_count"] > 0]
    tp_f = sum(r["tp"] for r in sub_f); fp_f = sum(r["fp"] for r in sub_f); fn_f = sum(r["fn"] for r in sub_f)
    tp_r = sum(r["tp"] for r in sub_r); fp_r = sum(r["fp"] for r in sub_r); fn_r = sum(r["fn"] for r in sub_r)
    p_f = tp_f / max(tp_f + fp_f, 1); r_f = tp_f / max(tp_f + fn_f, 1); f1_f = 2*p_f*r_f / max(p_f+r_f, 1e-9)
    p_r = tp_r / max(tp_r + fp_r, 1); r_r = tp_r / max(tp_r + fn_r, 1); f1_r = 2*p_r*r_r / max(p_r+r_r, 1e-9)
    return {
        "region": "overall",
        "eval_grids": len(sub_f),
        "filtered_precision": p_f, "filtered_recall": r_f, "filtered_f1": f1_f,
        "raw_precision": p_r, "raw_recall": r_r, "raw_f1": f1_r,
        "precision_delta": p_f - p_r, "recall_delta": r_f - r_r, "f1_delta": f1_f - f1_r,
        "fp_removed": int(fp_r - fp_f), "tp_lost": int(tp_r - tp_f),
        "pred_removed": int((tp_r + fp_r) - (tp_f + fp_f)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--thresholds-json",
                   default="configs/classifier/thresholds_v2.json")
    p.add_argument("--out-root",
                   default="results/analysis/cls_cascade_holdout_v2")
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES.keys()),
                   help="backbones to eval (effb0/convnext/dinov2)")
    p.add_argument("--skip-classify", action="store_true",
                   help="reuse existing predictions_metric_filtered.gpkg from a prior run "
                        "(only sane if you've just run classify_predictions for this backbone)")
    args = p.parse_args()

    thresholds = json.loads(Path(args.thresholds_json).read_text())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # Track per-backbone × per-grid metrics and aggregate across backbones for comparison.
    comparison_rows: list[dict] = []

    for backbone_key in args.backbones:
        arch, ckpt_rel = BACKBONES[backbone_key]
        ckpt = PROJECT_ROOT / ckpt_rel
        if not ckpt.exists():
            print(f"SKIP {backbone_key}: ckpt missing {ckpt}")
            continue
        per_layer_thr = thresholds["by_backbone"][arch]["thresholds"]

        bb_out = out_root / f"{backbone_key}_perlayer"
        bb_logs = bb_out / "logs"
        bb_out.mkdir(parents=True, exist_ok=True)
        bb_logs.mkdir(parents=True, exist_ok=True)

        filter_counts: list[dict] = []
        per_grid_rows: list[dict] = []
        failed: list[dict] = []

        print(f"\n========== Backbone {backbone_key} ({arch}) ==========")
        for region, info in HOLDOUT.items():
            layer = info["imagery_layer"]
            thr = per_layer_thr[layer]["threshold"]
            print(f"  region={region} layer={layer} thr={thr:.4f}")
            results_dir = PROJECT_ROOT / f"results/{info['region_arg']}/{info['model_run']}"

            for grid in info["grids"]:
                grid_dir = results_dir / grid

                # 1. Classify
                if not args.skip_classify:
                    cmd_cls = [
                        py, "scripts/classifier/classify_predictions.py",
                        "--grid-id", grid,
                        "--model-path", str(ckpt),
                        "--pv-threshold", str(thr),
                        "--area-cutoff", "30",
                        "--tiles-root", str(info["tiles_root"]),
                        "--results-dir", str(results_dir),
                    ]
                    rc = run(cmd_cls, bb_logs / f"{grid}_{region}_classify.log")
                    if rc != 0:
                        print(f"    CLASSIFY_FAIL {grid}")
                        failed.append({"backbone": backbone_key, "region": region,
                                       "grid": grid, "phase": "classify", "rc": rc})
                        continue

                # 2. Read classify summary
                cls_summary = grid_dir / "cls_summary.json"
                if cls_summary.exists():
                    s = json.loads(cls_summary.read_text())
                    filter_counts.append({
                        "region": region, "grid_id": grid,
                        "model_path": str(ckpt.relative_to(PROJECT_ROOT)),
                        "pv_threshold": thr,
                        "area_cutoff_m2": 30.0,
                        "total_detections": s.get("total_detections"),
                        "classified_count": s.get("classified_count"),
                        "extraction_failed_count": s.get("extraction_failed_count"),
                        "large_bypassed_count": s.get("large_bypassed_count"),
                        "pv_count": s.get("pv_count"),
                        "non_pv_count": s.get("non_pv_count"),
                        "filtered_count": s.get("filtered_count"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                if grid in info["missing_gt_grids"]:
                    print(f"    SKIP_EVAL {grid} (no local GT)")
                    continue

                # 3. Eval raw + filtered.
                # For both variants we pass --classifier-filtered-gpkg so
                # detect_and_evaluate runs in eval-only mode and consumes the
                # exact prediction set we want — never re-running detection.
                # raw → original predictions_metric.gpkg (V3-C/V4 baseline)
                # filtered → predictions_metric_filtered.gpkg (PV-only after cls)
                raw_gpkg = grid_dir / "predictions_metric.gpkg"
                filtered_gpkg = grid_dir / "predictions_metric_filtered.gpkg"
                eval_jobs = [
                    ("raw", raw_gpkg, f"cls_v2_{backbone_key}_raw"),
                    ("filtered", filtered_gpkg, f"cls_v2_{backbone_key}_filtered"),
                ]
                for variant, gpkg, eval_subdir in eval_jobs:
                    if not gpkg.exists():
                        print(f"    EVAL_NO_GPKG {grid}/{variant} ({gpkg.name})")
                        failed.append({"backbone": backbone_key, "region": region,
                                       "grid": grid, "phase": f"eval_{variant}",
                                       "rc": "missing_gpkg"})
                        continue
                    eval_out = grid_dir / eval_subdir
                    eval_out.mkdir(parents=True, exist_ok=True)
                    cmd_eval = [
                        py, "detect_and_evaluate.py",
                        "--grid-id", grid,
                        "--region", info["region_arg"],
                        "--imagery-layer", layer,
                        "--model-run", info["model_run"],
                        "--postproc-config", "configs/postproc/v4_canonical.json",
                        "--output-subdir", eval_subdir,
                        "--classifier-filtered-gpkg", str(gpkg),
                    ]
                    rc = run(cmd_eval, bb_logs / f"{grid}_{region}_{variant}.log")
                    if rc != 0:
                        print(f"    EVAL_FAIL {grid}/{variant} (rc={rc})")
                        failed.append({"backbone": backbone_key, "region": region,
                                       "grid": grid, "phase": f"eval_{variant}", "rc": rc})
                        continue
                    metrics = load_presence(eval_out / "presence_metrics.csv")
                    if metrics is None:
                        print(f"    EVAL_NO_METRICS {grid}/{variant}")
                        failed.append({"backbone": backbone_key, "region": region,
                                       "grid": grid, "phase": f"eval_{variant}", "rc": "no_csv"})
                        continue
                    per_grid_rows.append({
                        "region": region, "model_run": info["model_run"],
                        "grid_id": grid, "variant": variant,
                        **metrics,
                    })
                    print(f"    {grid}/{variant}: P={metrics['precision']:.3f} "
                          f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                          f"(TP {metrics['tp']}, FP {metrics['fp']}, FN {metrics['fn']})")

        # Persist per-backbone outputs
        pd.DataFrame(filter_counts).to_csv(bb_out / "classifier_filter_counts.csv", index=False)
        pd.DataFrame(per_grid_rows).to_csv(bb_out / "per_grid_metrics.csv", index=False)
        if failed:
            pd.DataFrame(failed).to_csv(bb_out / "failed_eval_grids.csv", index=False)

        # Summary metrics (raw + filtered, per region + overall)
        summary_rows: list[dict] = []
        for region in ("ct", "jhb"):
            agg = aggregate_region(per_grid_rows, region)
            if agg:
                summary_rows.append({"backbone": backbone_key, "arch": arch, **agg})
        agg_all = aggregate_overall(per_grid_rows)
        summary_rows.append({"backbone": backbone_key, "arch": arch, **agg_all})
        pd.DataFrame(summary_rows).to_csv(bb_out / "summary_metrics.csv", index=False)

        summary = {
            "eval_id": f"{backbone_key}_perlayer_v2_holdout17",
            "thresholds_per_layer": {layer: per_layer_thr[layer]["threshold"]
                                      for layer in per_layer_thr},
            "area_cutoff_m2": 30.0,
            "summary_rows": summary_rows,
        }
        (bb_out / "summary.json").write_text(json.dumps(summary, indent=2))
        comparison_rows.extend(summary_rows)
        print(f"  → {bb_out}/summary.json")

    # Cross-backbone comparison
    cmp_out = out_root / "backbone_compare_perlayer"
    cmp_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(cmp_out / "comparison_table.csv", index=False)
    (cmp_out / "summary.json").write_text(json.dumps(
        {"eval_id": "backbone_compare_perlayer_v2_holdout17",
         "thresholds_json": str(args.thresholds_json),
         "comparison_rows": comparison_rows}, indent=2))
    print(f"\n=== Comparison written to {cmp_out}/comparison_table.csv ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
