#!/usr/bin/env python3
"""V1.4 four-channel validation harness for a fresh Mask R-CNN checkpoint.

Given a new ``best_model.pth``, run the full inference + SAM + filter +
evaluation pipeline against held-out grids (default: JHB val
G0816/G0817/G0925 + CT val G2030/G1971) and produce a markdown summary
comparing the new model to the existing V3-C baseline runs.

Stages (each idempotent, controllable via --only / --skip):

  1. detect_jhb     detect_direct.py + finalize.py --postproc-config v4_canonical
  2. sam_jhb        sam_refine_maskbox.py --prompt-mode mask_box
  3. filter_jhb     filter_sam_inventory.py --config v4_agg.json
  4. detect_ct      detect_direct.py + finalize.py
  5. ch2_jhb        compute_ch2_recall.py on raw + sam_v4agg
  6. ch3_jhb        inline area-aggregate (raw + sam_v4agg)
  7. f1_ct          inline per-polygon F1 (installation profile) on CT val
  8. plaus_jhb      grid_plausibility.py on sam_v4agg
  9. plaus_ct       grid_plausibility.py on raw CT predictions
 10. summary        build comparison markdown vs V3-C baselines

Stages 1-4 need GPU + tiles (run on RunPod or local CUDA box). Stages
5-10 only need GT + prediction GPKGs.

Example:
  python scripts/analysis/validate_checkpoint.py \
    --model-path checkpoints/train20_val5_hn_20260508_v3c/best_model.pth \
    --run-name train20_val5_hn_20260508_v3c

  # Eval only, after syncing pod outputs back:
  python scripts/analysis/validate_checkpoint.py \
    --run-name train20_val5_hn_20260508_v3c \
    --only ch2_jhb,ch3_jhb,f1_ct,plaus_jhb,plaus_ct,summary

Outputs:
  results/<region>/<run-name>/<grid>/predictions_metric.gpkg     (raw)
  results/<region>/<run-name>_sam/<grid>/predictions_metric.gpkg (SAM)
  results/johannesburg/<run-name>_sam_v4agg/<grid>/...           (V1.4 inventory)
  results/validation/<run-name>_eval/                            (Ch2/Ch3/F1/plaus)
  results/validation/<run-name>_eval/summary.md                  (comparison)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detect_and_evaluate import iou_matching  # noqa: E402
from scripts.analysis.area_aggregate_eval import _sum_area_m2, summarize  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────
DEFAULT_JHB_GRIDS = ["G0816", "G0817", "G0925"]
DEFAULT_CT_GRIDS = ["G2030", "G1971"]

JHB_REGION = "jhb"
JHB_LAYER = "vexcel_2024"
JHB_TILES = "/home/gaosh/zasolar_data/tiles/johannesburg/vexcel_2024"
JHB_METRIC_CRS = "EPSG:32735"
JHB_BASELINE_CH2_DIR = (
    PROJECT_ROOT
    / "results/johannesburg/v3c_sam_maskbox_vexcel_2024_v4_agg_ch2_recall"
)
JHB_BASELINE_CH3_CSV = (
    PROJECT_ROOT
    / "results/analysis/area_aggregate_ch3_jhb_cbd25_v3c_sam_fixed/per_grid.csv"
)
JHB_BASELINE_PLAUS_DIR = (
    PROJECT_ROOT
    / "results/validation/plausibility_20260505_v3c_sam_maskbox_vexcel_2024_v4_agg"
)
JHB_CLEAN_GT_ROOT = PROJECT_ROOT / "data/annotations_channel2_clean"

CT_REGION = "ct"
CT_LAYER = "aerial_2025"
CT_METRIC_CRS = "EPSG:32734"
CT_BASELINE_RUN_DIR = PROJECT_ROOT / "results/cape_town/v3c_targeted_hn_aerial_2025"
CT_BASELINE_PLAUS_DIR = (
    PROJECT_ROOT / "results/validation/plausibility_20260505_v3c_ct_aerial_2025"
)
CT_GT_ROOT = PROJECT_ROOT / "data/annotations/Capetown"

POSTPROC_CANONICAL = PROJECT_ROOT / "configs/postproc/v4_canonical.json"
POSTPROC_V4_AGG = PROJECT_ROOT / "configs/postproc/v4_agg.json"

ALL_STAGES = [
    "detect_jhb",
    "sam_jhb",
    "filter_jhb",
    "detect_ct",
    "ch2_jhb",
    "ch3_jhb",
    "f1_ct",
    "plaus_jhb",
    "plaus_ct",
    "summary",
]


# ─────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class RunPaths:
    run_name: str
    jhb_raw: Path  # results/johannesburg/<run_name>
    jhb_sam: Path  # results/johannesburg/<run_name>_sam_maskbox
    jhb_v4agg: Path  # results/johannesburg/<run_name>_sam_maskbox_v4_agg
    ct_raw: Path  # results/cape_town/<run_name>
    eval_dir: Path  # results/validation/<run_name>_eval

    @classmethod
    def for_run(cls, run_name: str) -> "RunPaths":
        return cls(
            run_name=run_name,
            jhb_raw=PROJECT_ROOT / "results/johannesburg" / run_name,
            jhb_sam=PROJECT_ROOT / "results/johannesburg" / f"{run_name}_sam_maskbox",
            jhb_v4agg=PROJECT_ROOT
            / "results/johannesburg"
            / f"{run_name}_sam_maskbox_v4_agg",
            ct_raw=PROJECT_ROOT / "results/cape_town" / run_name,
            eval_dir=PROJECT_ROOT / "results/validation" / f"{run_name}_eval",
        )


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}", flush=True)


def run_cmd(cmd: list[str | Path], *, dry_run: bool = False) -> int:
    args = [str(c) for c in cmd]
    print(f"$ {' '.join(shlex.quote(a) for a in args)}", flush=True)
    if dry_run:
        return 0
    t0 = time.time()
    rc = subprocess.call(args, cwd=PROJECT_ROOT)
    dt = time.time() - t0
    print(f"  ↳ exit={rc} elapsed={dt:.1f}s", flush=True)
    return rc


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────
# Stage: detect (per region)
# ─────────────────────────────────────────────────────────────────────────
def stage_detect(
    *,
    region: str,
    imagery_layer: str,
    grids: list[str],
    model_path: Path,
    run_name: str,
    out_root: Path,
    args: argparse.Namespace,
) -> bool:
    banner(f"detect_{region}: {len(grids)} grid(s) → {out_root.relative_to(PROJECT_ROOT)}")
    failed: list[str] = []
    for g in grids:
        out_dir = out_root / g
        ensure_dir(out_dir)
        raw_pkl = out_dir / "raw_detections.pkl"
        gpkg = out_dir / "predictions_metric.gpkg"

        # detect_direct
        if raw_pkl.exists() and not args.force:
            print(f"[skip-detect_direct] {g}: raw_detections.pkl exists")
        else:
            rc = run_cmd(
                [
                    sys.executable,
                    "detect_direct.py",
                    "--grid-id",
                    g,
                    "--region",
                    region,
                    "--imagery-layer",
                    imagery_layer,
                    "--model-run",
                    run_name,
                    "--model-path",
                    model_path,
                    "--output-dir",
                    out_dir,
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                ],
                dry_run=args.dry_run,
            )
            if rc != 0:
                failed.append(g)
                continue

        # finalize → predictions_metric.gpkg
        if gpkg.exists() and not args.force:
            print(f"[skip-finalize] {g}: predictions_metric.gpkg exists")
            continue
        rc = run_cmd(
            [
                sys.executable,
                "finalize.py",
                "--input",
                raw_pkl,
                "--output-dir",
                out_dir,
                "--postproc-config",
                POSTPROC_CANONICAL,
                "--allow-overwrite-canonical",
            ],
            dry_run=args.dry_run,
        )
        if rc != 0:
            failed.append(g)

    if failed:
        print(f"[FAIL] detect_{region} failed grids: {failed}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────
# Stage: SAM refine (JHB)
# ─────────────────────────────────────────────────────────────────────────
def stage_sam_jhb(
    *,
    grids: list[str],
    paths: RunPaths,
    args: argparse.Namespace,
) -> bool:
    banner(f"sam_jhb: {len(grids)} grid(s) → {paths.jhb_sam.relative_to(PROJECT_ROOT)}")
    if all((paths.jhb_sam / g / "predictions_metric.gpkg").exists() for g in grids) and not args.force:
        print("[skip] all SAM outputs already present (use --force to re-run)")
        return True
    rc = run_cmd(
        [
            sys.executable,
            "scripts/analysis/sam_refine_maskbox.py",
            "--region",
            "jhb",
            "--src-results-root",
            paths.jhb_raw,
            "--tiles-root",
            JHB_TILES,
            "--output-root",
            paths.jhb_sam,
            "--grids",
            *grids,
            "--prompt-mode",
            "mask_box",
        ],
        dry_run=args.dry_run,
    )
    return rc == 0


# ─────────────────────────────────────────────────────────────────────────
# Stage: filter SAM inventory (JHB)
# ─────────────────────────────────────────────────────────────────────────
def stage_filter_jhb(
    *,
    grids: list[str],
    paths: RunPaths,
    args: argparse.Namespace,
) -> bool:
    banner(f"filter_jhb: v4_agg.json → {paths.jhb_v4agg.relative_to(PROJECT_ROOT)}")
    rc = run_cmd(
        [
            sys.executable,
            "scripts/analysis/filter_sam_inventory.py",
            "--src-root",
            paths.jhb_sam,
            "--config",
            POSTPROC_V4_AGG,
            "--output-root",
            paths.jhb_v4agg,
            "--grids",
            *grids,
            "--force",
        ],
        dry_run=args.dry_run,
    )
    return rc == 0


# ─────────────────────────────────────────────────────────────────────────
# Stage: Channel 2 recall (JHB)
# ─────────────────────────────────────────────────────────────────────────
def stage_ch2_jhb(
    *,
    grids: list[str],
    paths: RunPaths,
    args: argparse.Namespace,
) -> bool:
    banner(f"ch2_jhb: exhaustive recall on {grids}")
    eval_dir = ensure_dir(paths.eval_dir)
    ok = True
    for label, pred_root in (
        ("raw", paths.jhb_raw),
        ("sam_v4agg", paths.jhb_v4agg),
    ):
        out_dir = eval_dir / f"ch2_recall_{label}"
        rc = run_cmd(
            [
                sys.executable,
                "scripts/analysis/compute_ch2_recall.py",
                "--clean-gt-root",
                JHB_CLEAN_GT_ROOT,
                "--pred-root",
                pred_root,
                "--output-dir",
                out_dir,
                "--grids",
                *grids,
            ],
            dry_run=args.dry_run,
        )
        ok = ok and rc == 0
    return ok


# ─────────────────────────────────────────────────────────────────────────
# Stage: Channel 3 area-aggregate (JHB) — inline
# ─────────────────────────────────────────────────────────────────────────
def stage_ch3_jhb(
    *,
    grids: list[str],
    paths: RunPaths,
    args: argparse.Namespace,
) -> bool:
    banner(f"ch3_jhb: inline area-aggregate on {grids}")
    eval_dir = ensure_dir(paths.eval_dir)
    rows: list[dict] = []

    for label, pred_root in (
        ("raw", paths.jhb_raw),
        ("sam_v4agg", paths.jhb_v4agg),
    ):
        for g in grids:
            pred_path = pred_root / g / "predictions_metric.gpkg"
            gt_path = JHB_CLEAN_GT_ROOT / g / f"{g}_clean_gt.gpkg"
            if not pred_path.exists() or not gt_path.exists():
                print(f"[skip] {label}/{g}: pred or GT missing")
                continue
            try:
                n_pred, pred_m2, pred_max, _ = _sum_area_m2(
                    pred_path, JHB_METRIC_CRS, layer=None
                )
                n_gt, gt_m2, gt_max, _ = _sum_area_m2(
                    gt_path, JHB_METRIC_CRS, layer=None
                )
            except Exception as exc:
                print(f"[warn] {label}/{g}: {exc}")
                continue
            if gt_m2 <= 0:
                continue
            abs_err = pred_m2 - gt_m2
            signed_rel = abs_err / gt_m2 if gt_m2 else float("nan")
            rows.append(
                {
                    "region": "johannesburg",
                    "model_run": f"{paths.run_name}_{label}",
                    "model_version": paths.run_name,
                    "imagery_layer": JHB_LAYER,
                    "grid_id": g,
                    "gt_source": gt_path.name,
                    "n_pred": n_pred,
                    "pred_total_m2": round(pred_m2, 2),
                    "pred_max_poly_m2": round(pred_max, 2),
                    "n_gt": n_gt,
                    "gt_total_m2": round(gt_m2, 2),
                    "gt_max_poly_m2": round(gt_max, 2),
                    "abs_error_m2": round(abs_err, 2),
                    "signed_rel_error": signed_rel,
                    "abs_rel_error": abs(signed_rel) if gt_m2 else float("nan"),
                    "pred_gt_ratio": pred_m2 / gt_m2 if gt_m2 else float("nan"),
                }
            )

    if not rows:
        print("[FAIL] ch3_jhb: no rows produced")
        return False

    per_grid_df = pd.DataFrame(rows)
    per_grid_df.to_csv(eval_dir / "ch3_per_grid.csv", index=False)
    summary = summarize(rows)
    pd.DataFrame(summary).to_csv(eval_dir / "ch3_per_run.csv", index=False)
    print(f"[ok] ch3 per-grid → {eval_dir / 'ch3_per_grid.csv'}")
    print(f"[ok] ch3 per-run  → {eval_dir / 'ch3_per_run.csv'}")
    return True


# ─────────────────────────────────────────────────────────────────────────
# Stage: CT per-polygon F1 — inline
# ─────────────────────────────────────────────────────────────────────────
def _find_ct_gt(grid: str) -> Path | None:
    cands = sorted(CT_GT_ROOT.glob(f"{grid}_SAM2_*.gpkg"))
    if not cands:
        # Fall back to plain grid.gpkg if present
        plain = CT_GT_ROOT / f"{grid}.gpkg"
        return plain if plain.exists() else None
    return cands[-1]  # newest


def _per_polygon_f1(pred: gpd.GeoDataFrame, gt: gpd.GeoDataFrame) -> dict:
    if len(gt) == 0 or len(pred) == 0:
        return {
            "tp": 0,
            "fp": len(pred),
            "fn": len(gt),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou_threshold": 0.3,
            "n_pred": len(pred),
            "n_gt": len(gt),
        }
    res = iou_matching(gt, pred, iou_threshold=0.3, merge_preds=True)
    return {
        "tp": res["tp"],
        "fp": res["fp"],
        "fn": res["fn"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "iou_threshold": 0.3,
        "n_pred": len(pred),
        "n_gt": len(gt),
    }


def stage_f1_ct(
    *,
    grids: list[str],
    paths: RunPaths,
    args: argparse.Namespace,
) -> bool:
    banner(f"f1_ct: per-polygon F1 (installation profile, IoU≥0.3) on {grids}")
    eval_dir = ensure_dir(paths.eval_dir)
    rows: list[dict] = []
    for g in grids:
        pred_path = paths.ct_raw / g / "predictions_metric.gpkg"
        gt_path = _find_ct_gt(g)
        baseline_csv = CT_BASELINE_RUN_DIR / g / "presence_metrics.csv"

        if not pred_path.exists():
            print(f"[skip] {g}: pred missing ({pred_path})")
            continue
        if gt_path is None:
            print(f"[skip] {g}: GT missing under {CT_GT_ROOT}")
            continue

        pred = gpd.read_file(pred_path)
        gt = gpd.read_file(gt_path)
        if pred.crs is None:
            pred = pred.set_crs(CT_METRIC_CRS)
        if gt.crs is None:
            gt = gt.set_crs(CT_METRIC_CRS)
        if str(pred.crs) != CT_METRIC_CRS:
            pred = pred.to_crs(CT_METRIC_CRS)
        if str(gt.crs) != CT_METRIC_CRS:
            gt = gt.to_crs(CT_METRIC_CRS)

        m = _per_polygon_f1(pred, gt)
        m["grid_id"] = g
        m["run_name"] = paths.run_name
        m["pred_path"] = str(pred_path.relative_to(PROJECT_ROOT))
        m["gt_path"] = str(gt_path.relative_to(PROJECT_ROOT))
        m["baseline_presence_csv"] = (
            str(baseline_csv.relative_to(PROJECT_ROOT)) if baseline_csv.exists() else ""
        )
        rows.append(m)
        print(
            f"  {g}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
            f" (TP={m['tp']} FP={m['fp']} FN={m['fn']})"
        )

    if not rows:
        print("[FAIL] f1_ct: no rows produced")
        return False
    pd.DataFrame(rows).to_csv(eval_dir / "f1_ct_per_grid.csv", index=False)
    print(f"[ok] f1_ct → {eval_dir / 'f1_ct_per_grid.csv'}")
    return True


# ─────────────────────────────────────────────────────────────────────────
# Stage: plausibility (per region)
# ─────────────────────────────────────────────────────────────────────────
def stage_plausibility(
    *,
    region: str,
    grids: list[str],
    pred_root: Path,
    label: str,
    paths: RunPaths,
    args: argparse.Namespace,
) -> bool:
    banner(f"plaus_{region}: {pred_root.relative_to(PROJECT_ROOT)}")
    out_dir = ensure_dir(paths.eval_dir / f"plaus_{region}")
    rc = run_cmd(
        [
            sys.executable,
            "scripts/analysis/grid_plausibility.py",
            "--pred-root",
            pred_root,
            "--region",
            region,
            "--grid-list",
            ",".join(grids),
            "--output-dir",
            out_dir,
            "--label",
            label,
        ],
        dry_run=args.dry_run,
    )
    return rc == 0


# ─────────────────────────────────────────────────────────────────────────
# Stage: summary
# ─────────────────────────────────────────────────────────────────────────
def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}")
        return None


def _ch2_recall_for_grids(per_grid_csv: Path, grids: list[str]) -> dict | None:
    """compute_ch2_recall.py emits long-format CSV: one row per (grid, iou).

    Columns: grid, iou, n_gt, n_pred, matched, total, recall, ci_lo, ci_hi, ...

    Pool by summing matched/total across val grids per IoU threshold.
    """
    df = _read_csv_safe(per_grid_csv)
    if df is None or df.empty:
        return None
    if "grid" not in df.columns:
        return None
    df = df[df["grid"].isin(grids)]
    if df.empty:
        return None
    out: dict = {}
    for iou in (0.3, 0.5, 0.1):
        sub = df[df["iou"].astype(float).round(3) == iou]
        if sub.empty:
            continue
        matched = int(sub["matched"].sum())
        total = int(sub["total"].sum())
        out[f"recall@iou{iou}"] = matched / total if total else float("nan")
        out[f"n_matched_iou{iou}"] = matched
        if "n_total" not in out:
            out["n_total"] = total
    return out or None


def _ch3_for_grids(
    per_grid_csv: Path,
    grids: list[str],
    label: str = "",
    model_run: str | None = None,
) -> dict | None:
    df = _read_csv_safe(per_grid_csv)
    if df is None or df.empty:
        return None
    if "grid_id" in df.columns:
        df = df[df["grid_id"].isin(grids)]
    if model_run and "model_run" in df.columns:
        df = df[df["model_run"] == model_run]
    if df.empty:
        return None
    pred_total = df["pred_total_m2"].sum()
    gt_total = df["gt_total_m2"].sum()
    return {
        "label": label,
        "n_grids": len(df),
        "pred_total_m2": float(pred_total),
        "gt_total_m2": float(gt_total),
        "bulk_ratio": float(pred_total / gt_total) if gt_total else float("nan"),
        "bulk_signed_rel_err": float((pred_total - gt_total) / gt_total)
        if gt_total
        else float("nan"),
    }


def _ct_baseline_f1(grids: list[str]) -> dict[str, dict]:
    """Compute V3-C baseline F1 inline by re-running iou_matching on the
    baseline run's predictions_metric.gpkg + the same CT GT file.

    Keeps comparison apples-to-apples with the new-model F1 (same iou_threshold,
    same merge_preds=True, same GT file).
    """
    out: dict[str, dict] = {}
    for g in grids:
        pred_path = CT_BASELINE_RUN_DIR / g / "predictions_metric.gpkg"
        gt_path = _find_ct_gt(g)
        if not pred_path.exists() or gt_path is None:
            continue
        try:
            pred = gpd.read_file(pred_path)
            gt = gpd.read_file(gt_path)
        except Exception as exc:
            print(f"[warn] CT baseline {g}: {exc}")
            continue
        if pred.crs is None:
            pred = pred.set_crs(CT_METRIC_CRS)
        if gt.crs is None:
            gt = gt.set_crs(CT_METRIC_CRS)
        if str(pred.crs) != CT_METRIC_CRS:
            pred = pred.to_crs(CT_METRIC_CRS)
        if str(gt.crs) != CT_METRIC_CRS:
            gt = gt.to_crs(CT_METRIC_CRS)
        m = _per_polygon_f1(pred, gt)
        out[g] = m
    return out


def _diff_str(new: float, base: float, fmt: str = ".3f") -> str:
    if any(map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)), [new, base])):
        return f"{new:{fmt}}"
    delta = new - base
    sign = "+" if delta >= 0 else ""
    return f"{new:{fmt}} ({sign}{delta:{fmt}})"


def stage_summary(
    *,
    jhb_grids: list[str],
    ct_grids: list[str],
    paths: RunPaths,
    args: argparse.Namespace,
) -> bool:
    banner("summary: building comparison markdown")
    eval_dir = ensure_dir(paths.eval_dir)
    out_md = eval_dir / "summary.md"

    lines: list[str] = []
    lines.append(f"# Validation summary — `{paths.run_name}`")
    lines.append("")
    lines.append(
        f"_Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_"
    )
    lines.append("")
    lines.append(
        f"Held-out grids: JHB `{', '.join(jhb_grids)}` (Vexcel 2024) · "
        f"CT `{', '.join(ct_grids)}` (aerial 2025). Baseline = V3-C "
        "(`exp003_C_targeted_hn`)."
    )
    lines.append("")

    # ── Channel 2 recall (JHB) ──────────────────────────────────────────
    lines.append("## Channel 2 — JHB exhaustive recall (val grids only)")
    lines.append("")
    new_raw_ch2 = _ch2_recall_for_grids(
        eval_dir / "ch2_recall_raw" / f"ch2_recall_per_grid_{paths.run_name}.csv",
        jhb_grids,
    )
    new_v4agg_ch2 = _ch2_recall_for_grids(
        eval_dir
        / "ch2_recall_sam_v4agg"
        / f"ch2_recall_per_grid_{paths.run_name}_sam_maskbox_v4_agg.csv",
        jhb_grids,
    )
    base_ch2 = _ch2_recall_for_grids(
        JHB_BASELINE_CH2_DIR
        / "ch2_recall_per_grid_v3c_sam_maskbox_vexcel_2024_v4_agg.csv",
        jhb_grids,
    )

    lines.append("| Variant | recall@0.3 | recall@0.5 | n matched / total |")
    lines.append("|---|---:|---:|---|")
    for label, m in (
        ("V3-C+SAM (baseline)", base_ch2),
        ("new (raw)", new_raw_ch2),
        ("new (SAM+v4_agg)", new_v4agg_ch2),
    ):
        if m is None:
            lines.append(f"| {label} | — | — | — |")
            continue
        r03 = m.get("recall@iou0.3", float("nan"))
        r05 = m.get("recall@iou0.5", float("nan"))
        n_m = m.get("n_matched_iou0.3", "?")
        n_t = m.get("n_total", "?")
        lines.append(f"| {label} | {r03:.3f} | {r05:.3f} | {n_m} / {n_t} |")
    lines.append("")

    # ── Channel 3 area-aggregate (JHB) ──────────────────────────────────
    lines.append("## Channel 3 — JHB area-aggregate (val grids only)")
    lines.append("")
    new_ch3_csv = eval_dir / "ch3_per_grid.csv"
    new_ch3_df = _read_csv_safe(new_ch3_csv)
    base_ch3 = _ch3_for_grids(
        JHB_BASELINE_CH3_CSV,
        jhb_grids,
        label="V3-C+SAM_v4agg",
        model_run="v3c_sam_maskbox_vexcel_2024_v4_agg",
    )

    lines.append("| Variant | n_grids | Σ pred m² | Σ GT m² | bulk ratio | signed rel err |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    if base_ch3:
        lines.append(
            f"| {base_ch3['label']} | {base_ch3['n_grids']} | "
            f"{base_ch3['pred_total_m2']:.0f} | {base_ch3['gt_total_m2']:.0f} | "
            f"{base_ch3['bulk_ratio']:.4f} | {base_ch3['bulk_signed_rel_err']:+.4f} |"
        )
    else:
        lines.append("| V3-C+SAM_v4agg | — | — | — | — | — |")

    if new_ch3_df is not None and not new_ch3_df.empty:
        for label_run in new_ch3_df["model_run"].unique():
            sub = new_ch3_df[new_ch3_df["model_run"] == label_run]
            pred_total = sub["pred_total_m2"].sum()
            gt_total = sub["gt_total_m2"].sum()
            ratio = pred_total / gt_total if gt_total else float("nan")
            signed = (pred_total - gt_total) / gt_total if gt_total else float("nan")
            lines.append(
                f"| {label_run} | {len(sub)} | {pred_total:.0f} | "
                f"{gt_total:.0f} | {ratio:.4f} | {signed:+.4f} |"
            )
    else:
        lines.append("| new (raw) | — | — | — | — | — |")
        lines.append("| new (SAM+v4_agg) | — | — | — | — | — |")
    lines.append("")

    # ── CT per-polygon F1 ───────────────────────────────────────────────
    lines.append("## CT per-polygon F1 (installation profile, IoU≥0.3)")
    lines.append("")
    new_f1_df = _read_csv_safe(eval_dir / "f1_ct_per_grid.csv")
    base_f1_lookup = _ct_baseline_f1(ct_grids)
    lines.append("| Grid | Baseline F1 | New F1 | Baseline (P/R) | New (P/R) | Baseline n_pred / n_gt | New n_pred / n_gt |")
    lines.append("|---|---:|---:|---|---|---|---|")
    for g in ct_grids:
        base = base_f1_lookup.get(g)
        new_row = None
        if new_f1_df is not None and not new_f1_df.empty and "grid_id" in new_f1_df.columns:
            sel = new_f1_df[new_f1_df["grid_id"] == g]
            if len(sel):
                new_row = sel.iloc[-1].to_dict()
        if base is None and new_row is None:
            lines.append(f"| {g} | — | — | — | — | — | — |")
            continue
        b_f1_val = base["f1"] if base else float("nan")
        b_f1 = f"{b_f1_val:.3f}" if base else "—"
        if new_row:
            n_f1_val = float(new_row["f1"])
            f1_cell = _diff_str(n_f1_val, b_f1_val) if base else f"{n_f1_val:.3f}"
        else:
            f1_cell = "—"
        b_pr = (
            f"{base['precision']:.3f} / {base['recall']:.3f}" if base else "—"
        )
        n_pr = (
            f"{float(new_row['precision']):.3f} / {float(new_row['recall']):.3f}"
            if new_row
            else "—"
        )
        b_pred_gt = (
            f"{int(base['n_pred'])} / {int(base['n_gt'])}" if base else "—"
        )
        n_pred_gt = (
            f"{int(new_row['n_pred'])} / {int(new_row['n_gt'])}"
            if new_row
            else "—"
        )
        lines.append(
            f"| {g} | {b_f1} | {f1_cell} | {b_pr} | {n_pr} | {b_pred_gt} | {n_pred_gt} |"
        )
    lines.append("")

    # ── Plausibility flags ──────────────────────────────────────────────
    lines.append("## Plausibility flags")
    lines.append("")
    for region_label, new_flags_csv, base_flags_csv in (
        (
            "JHB (sam_v4agg)",
            paths.eval_dir / "plaus_jhb" / "flags.csv",
            JHB_BASELINE_PLAUS_DIR / "flags.csv",
        ),
        (
            "CT (raw)",
            paths.eval_dir / "plaus_ct" / "flags.csv",
            CT_BASELINE_PLAUS_DIR / "flags.csv",
        ),
    ):
        new_df = _read_csv_safe(new_flags_csv)
        base_df = _read_csv_safe(base_flags_csv)
        new_n = 0 if new_df is None or new_df.empty else len(new_df)
        base_n = 0 if base_df is None or base_df.empty else len(base_df)
        new_high = (
            int((new_df["severity"] == "high").sum())
            if new_df is not None and not new_df.empty and "severity" in new_df.columns
            else 0
        )
        base_high = (
            int((base_df["severity"] == "high").sum())
            if base_df is not None
            and not base_df.empty
            and "severity" in base_df.columns
            else 0
        )
        lines.append(
            f"- **{region_label}**: new flags = {new_n} (high={new_high}); "
            f"baseline flags = {base_n} (high={base_high})"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### Artifact locations")
    lines.append("")
    lines.append(f"- New raw JHB: `{paths.jhb_raw.relative_to(PROJECT_ROOT)}/<grid>/`")
    lines.append(f"- New SAM JHB: `{paths.jhb_sam.relative_to(PROJECT_ROOT)}/<grid>/`")
    lines.append(
        f"- New SAM+v4_agg JHB: `{paths.jhb_v4agg.relative_to(PROJECT_ROOT)}/<grid>/`"
    )
    lines.append(f"- New raw CT: `{paths.ct_raw.relative_to(PROJECT_ROOT)}/<grid>/`")
    lines.append(f"- Eval CSVs: `{paths.eval_dir.relative_to(PROJECT_ROOT)}/`")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] summary → {out_md}")
    return True


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to best_model.pth. Required for inference stages.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Logical run name; predictions land at results/<region>/<run-name>/<grid>/",
    )
    p.add_argument(
        "--jhb-grids",
        nargs="*",
        default=DEFAULT_JHB_GRIDS,
        help=f"JHB val grids (default: {DEFAULT_JHB_GRIDS})",
    )
    p.add_argument(
        "--ct-grids",
        nargs="*",
        default=DEFAULT_CT_GRIDS,
        help=f"CT val grids (default: {DEFAULT_CT_GRIDS})",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated list of stages to run; others skipped",
    )
    p.add_argument(
        "--skip",
        default=None,
        help="Comma-separated list of stages to skip",
    )
    p.add_argument("--batch-size", type=int, default=8, help="detect_direct batch size")
    p.add_argument(
        "--num-workers", type=int, default=4, help="detect_direct DataLoader workers"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference + finalize even if outputs exist",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every command but do not execute (still skips eval inline logic)",
    )
    p.add_argument(
        "--list-stages",
        action="store_true",
        help="Print the stage list and exit",
    )
    return p.parse_args()


def resolve_stages(args: argparse.Namespace) -> list[str]:
    if args.only:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = set(only) - set(ALL_STAGES)
        if unknown:
            sys.exit(f"unknown stage(s) in --only: {sorted(unknown)}")
        return [s for s in ALL_STAGES if s in only]
    skip = set()
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        unknown = skip - set(ALL_STAGES)
        if unknown:
            sys.exit(f"unknown stage(s) in --skip: {sorted(unknown)}")
    return [s for s in ALL_STAGES if s not in skip]


def main() -> int:
    args = parse_args()
    if args.list_stages:
        for s in ALL_STAGES:
            print(s)
        return 0
    if not args.run_name:
        sys.exit("--run-name is required (unless --list-stages)")
    stages = resolve_stages(args)
    paths = RunPaths.for_run(args.run_name)

    needs_model = bool({"detect_jhb", "detect_ct"} & set(stages))
    if needs_model and not args.model_path:
        sys.exit("--model-path is required for stages [detect_jhb, detect_ct]")
    if args.model_path and not args.model_path.exists() and not args.dry_run:
        sys.exit(f"model-path does not exist: {args.model_path}")

    print(f"Run name        : {args.run_name}")
    print(f"Model path      : {args.model_path}")
    print(f"JHB val grids   : {args.jhb_grids}")
    print(f"CT  val grids   : {args.ct_grids}")
    print(f"Stages to run   : {stages}")
    print(f"Eval dir        : {paths.eval_dir.relative_to(PROJECT_ROOT)}")

    failures: list[str] = []

    if "detect_jhb" in stages:
        ok = stage_detect(
            region=JHB_REGION,
            imagery_layer=JHB_LAYER,
            grids=args.jhb_grids,
            model_path=args.model_path,
            run_name=args.run_name,
            out_root=paths.jhb_raw,
            args=args,
        )
        if not ok:
            failures.append("detect_jhb")

    if "sam_jhb" in stages:
        ok = stage_sam_jhb(grids=args.jhb_grids, paths=paths, args=args)
        if not ok:
            failures.append("sam_jhb")

    if "filter_jhb" in stages:
        ok = stage_filter_jhb(grids=args.jhb_grids, paths=paths, args=args)
        if not ok:
            failures.append("filter_jhb")

    if "detect_ct" in stages:
        ok = stage_detect(
            region=CT_REGION,
            imagery_layer=CT_LAYER,
            grids=args.ct_grids,
            model_path=args.model_path,
            run_name=args.run_name,
            out_root=paths.ct_raw,
            args=args,
        )
        if not ok:
            failures.append("detect_ct")

    if "ch2_jhb" in stages:
        ok = stage_ch2_jhb(grids=args.jhb_grids, paths=paths, args=args)
        if not ok:
            failures.append("ch2_jhb")

    if "ch3_jhb" in stages:
        ok = stage_ch3_jhb(grids=args.jhb_grids, paths=paths, args=args)
        if not ok:
            failures.append("ch3_jhb")

    if "f1_ct" in stages:
        ok = stage_f1_ct(grids=args.ct_grids, paths=paths, args=args)
        if not ok:
            failures.append("f1_ct")

    if "plaus_jhb" in stages:
        ok = stage_plausibility(
            region=JHB_REGION,
            grids=args.jhb_grids,
            pred_root=paths.jhb_v4agg,
            label=f"{args.run_name}_sam_v4agg",
            paths=paths,
            args=args,
        )
        if not ok:
            failures.append("plaus_jhb")

    if "plaus_ct" in stages:
        ok = stage_plausibility(
            region=CT_REGION,
            grids=args.ct_grids,
            pred_root=paths.ct_raw,
            label=f"{args.run_name}_ct_raw",
            paths=paths,
            args=args,
        )
        if not ok:
            failures.append("plaus_ct")

    if "summary" in stages:
        ok = stage_summary(
            jhb_grids=args.jhb_grids,
            ct_grids=args.ct_grids,
            paths=paths,
            args=args,
        )
        if not ok:
            failures.append("summary")

    if failures:
        print(f"\n[FAIL] {len(failures)} stage(s) failed: {failures}")
        return 1
    print(f"\n[OK] all stages completed → {paths.eval_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
