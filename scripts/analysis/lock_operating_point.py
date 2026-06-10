"""Leakage-free polygon-conf operating-point locking (F1-gap Tier A2 / C11).

Fits a per-(region, imagery_layer, model-lineage) polygon-confidence
threshold on a calibration grid set that is DISJOINT from every reporting
suite, then validates transfer: the locked point's agg_area_F1 must be
within 1pp of the oracle-sweep max on every declared validation suite.

Why: operating points swept on the reporting suite itself are oracle-leaked
(wave1 rankings flipped with the sweep caliber, 2026-06-07). This script
makes threshold selection a declared, reproducible, leakage-checked step.

Config: configs/eval/operating_point_calibration.yaml
Protocol: docs/evaluation_protocol.md §2

Metric machinery is reused verbatim from area_aggregate_eval /
polygon_conf_sweep (set-theoretic union areas, Tier-1 summarize). Ranking
rule for the lock = poly_conf_sweep precedent: minimize
std_ratio_Bw + rmse_m2/1e5 subject to bulk_pred_gt_ratio ∈ [0.5, 2.0]
(feedback_tier1_metric_system: σ_Bw+RMSE 主裁判, bulk 仅 sanity gate).

Usage:
    python scripts/analysis/lock_operating_point.py --lock-id ct_aerial_2025_v3c
    python scripts/analysis/lock_operating_point.py --lock-id ct_aerial_2025_v3c --platt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.area_aggregate_eval import summarize  # noqa: E402
from scripts.analysis.polygon_conf_sweep import (  # noqa: E402
    evaluate_run_at_threshold,
)

CONFIG_PATH = REPO_ROOT / "configs" / "eval" / "operating_point_calibration.yaml"
REGISTRY_PATH = REPO_ROOT / "configs" / "eval" / "locked_operating_points.json"
OUTPUT_ROOT = REPO_ROOT / "results" / "analysis" / "operating_point_lock"

BULK_GATE = (0.5, 2.0)
ACCEPT_GAP = 0.01  # locked agg_area_F1 within 1pp of oracle max, per suite


# ─────────────────────────────────────────────────────────────────────────
# Reporting-suite registry → {region_alias: (explicit_ids, [regex patterns])}
# ─────────────────────────────────────────────────────────────────────────
def load_reporting_grids(cfg: dict) -> dict[str, tuple[set[str], list[str]]]:
    out: dict[str, tuple[set[str], list[str]]] = {}

    def _add(region: str, ids=None, pattern=None):
        ids_set, patterns = out.setdefault(region, (set(), []))
        if ids:
            ids_set.update(ids)
        if pattern:
            patterns.append(pattern)

    rs = cfg.get("reporting_suites", {})
    for preset_rel in rs.get("benchmark_presets", []):
        preset = yaml.safe_load((REPO_ROOT / preset_rel).read_text())
        for suite in preset.get("suites", []):
            # post_train.yaml convention: CT suites omit `region`
            region = suite.get("region", "cape_town")
            _add(region, ids=suite.get("grid_ids", []))
    for name, spec in rs.get("extra", {}).items():
        regions = spec.get("regions", [spec.get("region")])
        for region in regions:
            if region is None:
                raise ValueError(f"reporting_suites.extra.{name}: missing region")
            _add(region, ids=spec.get("grid_ids"),
                 pattern=spec.get("grid_id_pattern"))
    return out


def _is_reporting(grid_id: str, region: str,
                  reporting: dict[str, tuple[set[str], list[str]]]) -> bool:
    from core.grid_utils import normalize_region
    region_n = normalize_region(region)
    for r, (ids, patterns) in reporting.items():
        if normalize_region(r) != region_n:
            continue
        if grid_id in ids:
            return True
        for pat in patterns:
            if re.match(pat, grid_id):
                return True
    return False


def _suite_grid_ids(suite_id: str, cfg: dict) -> list[str]:
    for preset_rel in cfg["reporting_suites"].get("benchmark_presets", []):
        preset = yaml.safe_load((REPO_ROOT / preset_rel).read_text())
        for suite in preset.get("suites", []):
            if suite["suite_id"] == suite_id:
                return list(suite["grid_ids"])
    extra = cfg["reporting_suites"].get("extra", {})
    if suite_id in extra and "grid_ids" in extra[suite_id]:
        return list(extra[suite_id]["grid_ids"])
    raise KeyError(f"suite_id {suite_id!r} not found in any registered preset")


# ─────────────────────────────────────────────────────────────────────────
# Sweep on a grid subset
# ─────────────────────────────────────────────────────────────────────────
def sweep_subset(region: str, model_run: str, grids: list[str],
                 thresholds: list[float]) -> tuple[list[dict], list[dict]]:
    """Returns (summary_rows_per_threshold, per_grid_rows). Restricted to `grids`."""
    want = set(grids)
    summ_rows, pg_rows = [], []
    for t in thresholds:
        rows = [r for r in evaluate_run_at_threshold(region, model_run, t)
                if r["grid_id"] in want]
        missing = want - {r["grid_id"] for r in rows}
        if missing:
            print(f"  [warn] t={t}: {len(missing)} grids without preds+GT: "
                  f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        for r in rows:
            r2 = dict(r)
            r2["conf_threshold"] = t
            pg_rows.append(r2)
        summ = summarize(rows)
        if summ:
            s = dict(summ[0])
            s["conf_threshold"] = t
            summ_rows.append(s)
    return summ_rows, pg_rows


def rank_lock(summ_rows: list[dict]) -> dict | None:
    """poly_conf_sweep ranking: min σ_Bw + RMSE/1e5 s.t. bulk in gate."""
    gated = [s for s in summ_rows
             if s.get("bulk_pred_gt_ratio") is not None
             and BULK_GATE[0] <= s["bulk_pred_gt_ratio"] <= BULK_GATE[1]]
    if not gated:
        return None
    return min(gated, key=lambda s: (s["std_ratio_Bw"] or np.inf)
               + (s["rmse_m2"] or np.inf) / 1e5)


# ─────────────────────────────────────────────────────────────────────────
# Platt ablation (protocol-internal; see docs/evaluation_protocol.md §2.4)
# ─────────────────────────────────────────────────────────────────────────
def platt_ablation(region: str, model_run: str, grids: list[str],
                   n_max: int = 300, seed: int = 20260610) -> dict:
    """Fit 2-param Platt scaling on calibration detections; report fit quality.

    Label: prediction polygon is TP iff intersection-over-pred with the grid's
    GT union >= 0.5. Within one imagery layer Platt is monotonic, so threshold
    *placement* (and thus every set-level metric at the locked point) is
    unchanged — the ablation can only win once >=2 leakage-free layer locks
    exist (shared-prob-threshold transfer). Reported for the record;
    fail-closed: not retained unless it beats raw on the acceptance bar.
    """
    import geopandas as gpd
    from shapely.ops import unary_union
    from scipy.optimize import minimize

    from scripts.analysis.area_aggregate_eval import (
        _gt_spec_for, _load_run_grids, _read_polys_geom,
    )
    from core.region_registry import get_region_config

    region_cfg = get_region_config(region)
    metric_crs = region_cfg.crs_metric
    want = set(grids)
    scores, labels = [], []
    for grid_id, pred_path in _load_run_grids(region, model_run):
        if grid_id not in want:
            continue
        gt_spec = _gt_spec_for(region_cfg, grid_id)
        if gt_spec is None:
            continue
        _, _, _, _, gt_u = _read_polys_geom(gt_spec[0], metric_crs, layer=gt_spec[1])
        if gt_u is None:
            continue
        pred = gpd.read_file(pred_path)
        if pred.empty or "confidence" not in pred.columns:
            continue
        if pred.crs is None or str(pred.crs) != metric_crs:
            pred = pred.to_crs(metric_crs)
        for geom, conf in zip(pred.geometry, pred["confidence"]):
            if geom is None or geom.is_empty or not geom.is_valid:
                continue
            a = geom.area
            if a <= 0:
                continue
            iop = geom.intersection(gt_u).area / a
            scores.append(float(conf))
            labels.append(1 if iop >= 0.5 else 0)

    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)
    if len(scores_arr) > n_max:
        idx = rng.choice(len(scores_arr), n_max, replace=False)
        scores_arr, labels_arr = scores_arr[idx], labels_arr[idx]

    def nll(params):
        a, b = params
        p = 1.0 / (1.0 + np.exp(-(a * scores_arr + b)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(labels_arr * np.log(p) + (1 - labels_arr) * np.log(1 - p))

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    a, b = res.x
    p_raw = np.clip(scores_arr, 1e-9, 1 - 1e-9)
    p_platt = np.clip(1.0 / (1.0 + np.exp(-(a * scores_arr + b))), 1e-9, 1 - 1e-9)

    def brier(p):
        return float(np.mean((p - labels_arr) ** 2))

    def logloss(p):
        return float(-np.mean(labels_arr * np.log(p)
                              + (1 - labels_arr) * np.log(1 - p)))

    return {
        "n_detections": int(len(scores_arr)),
        "tp_rate": float(labels_arr.mean()),
        "platt_a": float(a),
        "platt_b": float(b),
        "brier_raw_conf_as_prob": brier(p_raw),
        "brier_platt": brier(p_platt),
        "logloss_raw_conf_as_prob": logloss(p_raw),
        "logloss_platt": logloss(p_platt),
        "monotonic": bool(a > 0),
        "verdict": (
            "within-layer threshold placement unchanged (monotonic transform); "
            "cross-layer shared-threshold transfer untestable with a single "
            "leakage-free layer lock -> NOT retained (fail-closed)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--lock-id", required=True)
    ap.add_argument("--platt", action="store_true",
                    help="run the protocol-internal Platt-scaling ablation")
    ap.add_argument("--update-registry", action="store_true",
                    help="write the lock into configs/eval/locked_operating_points.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    lock_spec = next((l for l in cfg.get("locks", [])
                      if l["lock_id"] == args.lock_id), None)
    if lock_spec is None:
        raise SystemExit(f"lock_id {args.lock_id!r} not in {args.config}")

    region = lock_spec["region"]
    fit = lock_spec["fit"]
    calib_grids = list(fit["calibration_grids"])
    thresholds = [float(t) for t in fit["thresholds"]]

    # 1) leakage check
    reporting = load_reporting_grids(cfg)
    leaked = [g for g in calib_grids if _is_reporting(g, region, reporting)]
    if leaked:
        raise SystemExit(
            f"[LEAKAGE] calibration grids intersect reporting suites: {leaked}")
    print(f"[ok] leakage check passed: {len(calib_grids)} calibration grids "
          f"disjoint from all reporting suites")

    out_dir = OUTPUT_ROOT / args.lock_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) fit on calibration set
    print(f"[fit] {region}/{fit['model_run']} on {len(calib_grids)} grids, "
          f"t in {thresholds}")
    summ_rows, pg_rows = sweep_subset(region, fit["model_run"], calib_grids,
                                      thresholds)
    locked = rank_lock(summ_rows)
    if locked is None:
        raise SystemExit("[FAIL] no threshold passes the bulk sanity gate "
                         f"{BULK_GATE} on the calibration set")
    t_star = locked["conf_threshold"]
    print(f"[lock] t* = {t_star}  (σ_Bw={locked['std_ratio_Bw']}, "
          f"RMSE={locked['rmse_m2']}, bulk={locked['bulk_pred_gt_ratio']}, "
          f"aggF1={locked['agg_area_F1']})")

    _write_csv(out_dir / "calibration_sweep.csv", summ_rows)
    _write_csv(out_dir / "calibration_per_grid.csv", pg_rows)

    # 3) validate transfer on each declared reporting suite
    val_rows = []
    all_pass = True
    for v in lock_spec.get("validate_on", []):
        suite_grids = _suite_grid_ids(v["suite_id"], cfg)
        vsumm, _ = sweep_subset(v["region"], v["model_run"], suite_grids,
                                thresholds)
        if not vsumm:
            print(f"  [warn] no rows for {v['suite_id']} × {v['model_run']}")
            continue
        by_t = {s["conf_threshold"]: s for s in vsumm}
        oracle = max(vsumm, key=lambda s: s["agg_area_F1"] or 0)
        at_lock = by_t.get(t_star)
        gap = (oracle["agg_area_F1"] or 0) - (at_lock["agg_area_F1"] or 0)
        ok = gap <= ACCEPT_GAP
        all_pass &= ok
        val_rows.append({
            "suite_id": v["suite_id"], "model_run": v["model_run"],
            "merge_mode": v.get("merge_mode", ""),
            "locked_t": t_star,
            "locked_agg_F1": at_lock["agg_area_F1"],
            "oracle_t": oracle["conf_threshold"],
            "oracle_agg_F1": oracle["agg_area_F1"],
            "gap_pp": round(gap * 100, 2),
            "pass_le_1pp": ok,
            "locked_sigma_Bw": at_lock["std_ratio_Bw"],
            "locked_bulk": at_lock["bulk_pred_gt_ratio"],
            "locked_pg_F1": at_lock["mean_per_grid_F1"],
            "oracle_pg_F1": oracle["mean_per_grid_F1"],
        })
        print(f"[val] {v['suite_id']} × {v['model_run']}: locked F1@{t_star}="
              f"{at_lock['agg_area_F1']} vs oracle F1@{oracle['conf_threshold']}="
              f"{oracle['agg_area_F1']}  gap={gap*100:.2f}pp  "
              f"{'PASS' if ok else 'FAIL'}")
    _write_csv(out_dir / "validation.csv", val_rows)

    # 4) Platt ablation (optional, protocol-internal)
    platt = None
    if args.platt:
        print("[platt] fitting 2-param Platt on calibration detections ...")
        platt = platt_ablation(region, fit["model_run"], calib_grids)
        (out_dir / "platt_ablation.json").write_text(
            json.dumps(platt, indent=2) + "\n", encoding="utf-8")
        print(f"[platt] n={platt['n_detections']} tp_rate={platt['tp_rate']:.3f} "
              f"brier {platt['brier_raw_conf_as_prob']:.4f}→{platt['brier_platt']:.4f}")
        print(f"[platt] verdict: {platt['verdict']}")

    # 5) registry entry
    entry = {
        "lock_id": args.lock_id,
        "region": region,
        "imagery_layer": lock_spec["imagery_layer"],
        "model_lineage": lock_spec["model_lineage"],
        "locked_conf_threshold": t_star,
        "fitted_on": {"model_run": fit["model_run"],
                      "calibration_grids": calib_grids,
                      "thresholds": thresholds},
        "ranking_rule": "min std_ratio_Bw + rmse_m2/1e5 s.t. bulk in [0.5,2.0]",
        "acceptance": f"agg_area_F1 within {ACCEPT_GAP*100:.0f}pp of oracle per suite",
        "validation": val_rows,
        "all_suites_pass": all_pass,
        "locked_date": date.today().isoformat(),
        "git_head": _git_head(),
    }
    (out_dir / "lock_report.json").write_text(
        json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[out] {out_dir}/lock_report.json  (all_suites_pass={all_pass})")

    if args.update_registry:
        reg = {}
        if REGISTRY_PATH.exists():
            reg = json.loads(REGISTRY_PATH.read_text())
        reg[args.lock_id] = entry
        REGISTRY_PATH.write_text(
            json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[out] registry updated: {REGISTRY_PATH}")


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=REPO_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
