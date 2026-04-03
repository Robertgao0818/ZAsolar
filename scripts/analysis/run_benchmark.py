#!/usr/bin/env python3
"""
Agent-First Model Benchmark — 模型权重对比评估工具

在固定 benchmark suites 上对比多个 checkpoint 的检测性能，
自动生成 verdict（improved/regressed/flat/mixed/failed）。

主输出:
  - summary.json  — agent 首读文件，固定 schema
  - summary.md    — 人读摘要，TL;DR 开头
  - by_suite.csv  — suite 级聚合
  - by_grid.csv   — grid 级明细

用法:
  python3 scripts/analysis/run_benchmark.py                              # 默认 preset + 默认模型
  python3 scripts/analysis/run_benchmark.py --models v3c v3_cleaned      # 指定模型
  python3 scripts/analysis/run_benchmark.py --checkpoint path --tag exp  # 临时权重
  python3 scripts/analysis/run_benchmark.py --suite cape_town_t1_smoke   # 只跑 smoke
  python3 scripts/analysis/run_benchmark.py --collect-only               # 只收集已有结果
  python3 scripts/analysis/run_benchmark.py --checksum force             # 强制重算权重 SHA256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRESET = ROOT / "configs" / "benchmarks" / "post_train.yaml"
MODEL_REGISTRY = ROOT / "configs" / "model_registry.yaml"
DETECT_SCRIPT = ROOT / "detect_and_evaluate.py"
CHECKSUM_CACHE = ROOT / ".cache" / "checksum_cache.json"


# ════════════════════════════════════════════════════════════════════════
# Config Loading
# ════════════════════════════════════════════════════════════════════════

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_preset(path: Path) -> dict:
    cfg = load_yaml(path)
    if "benchmark_id" not in cfg:
        raise ValueError(f"Invalid preset: missing 'benchmark_id' in {path}")
    return cfg


def load_model_registry(path: Path = MODEL_REGISTRY) -> dict:
    if not path.exists():
        return {}
    return load_yaml(path).get("models", {})


def resolve_models(
    registry: dict,
    model_keys: list[str] | None,
    default_keys: list[str],
    extra_checkpoint: str | None,
    extra_tag: str | None,
) -> list[dict]:
    """Resolve model keys to checkpoint paths."""
    keys = model_keys or default_keys
    models = []

    for key in keys:
        if key not in registry:
            print(f"[WARN] Model '{key}' not in registry, skipping")
            continue
        m = registry[key]
        cp = Path(m["checkpoint"])
        if not cp.is_absolute():
            cp = ROOT / cp
        if not cp.exists():
            print(f"[WARN] Checkpoint not found: {cp}, skipping '{key}'")
            continue
        models.append({
            "key": key,
            "tag": m.get("tag", key),
            "checkpoint": cp,
            "description": m.get("description", ""),
            "coco_ap50": m.get("coco_ap50"),
        })

    if extra_checkpoint:
        cp = Path(extra_checkpoint)
        if not cp.is_absolute():
            cp = ROOT / cp
        if not cp.exists():
            print(f"[ERROR] Ad-hoc checkpoint not found: {cp}")
        else:
            tag = extra_tag or cp.stem
            models.append({
                "key": f"_adhoc_{tag}",
                "tag": tag,
                "checkpoint": cp,
                "description": f"Ad-hoc: {extra_checkpoint}",
                "coco_ap50": None,
            })

    return models


def select_suites(preset: dict, requested: list[str] | None) -> list[dict]:
    suites = preset.get("suites", [])
    if not requested:
        return suites
    requested_set = set(requested)
    selected = [s for s in suites if s["suite_id"] in requested_set]
    found = {s["suite_id"] for s in selected}
    missing = requested_set - found
    if missing:
        print(f"[WARN] Unknown suite(s): {sorted(missing)}")
    return selected


# ════════════════════════════════════════════════════════════════════════
# Checksum (auto-cached)
# ════════════════════════════════════════════════════════════════════════

def _load_checksum_cache() -> dict:
    if CHECKSUM_CACHE.exists():
        try:
            return json.loads(CHECKSUM_CACHE.read_text())
        except Exception:
            pass
    return {}


def _save_checksum_cache(cache: dict) -> None:
    CHECKSUM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_CACHE.write_text(json.dumps(cache, indent=2))


def get_checksum(path: Path, mode: str = "auto") -> str | None:
    """Get SHA256 of a file with caching.

    mode: auto (cache by path+size+mtime), skip, force
    """
    if mode == "skip":
        return None

    stat = path.stat()
    cache_key = f"{path}|{stat.st_size}|{stat.st_mtime}"

    if mode == "auto":
        cache = _load_checksum_cache()
        if cache_key in cache:
            return cache[cache_key]

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    sha = digest.hexdigest()

    if mode in ("auto", "force"):
        cache = _load_checksum_cache()
        cache[cache_key] = sha
        _save_checksum_cache(cache)

    return sha


# ════════════════════════════════════════════════════════════════════════
# Inference Execution
# ════════════════════════════════════════════════════════════════════════

def build_output_subdir(run_id: str, tag: str) -> str:
    return f"benchmark_{run_id}_{tag}"


def run_one_grid(
    *,
    model: dict,
    grid_id: str,
    suite: dict,
    preset: dict,
    run_id: str,
    force: bool,
    log_dir: Path,
) -> dict | None:
    """Run detect_and_evaluate.py for one model x one grid.

    Returns failure dict on error, None on success.
    """
    tag = model["tag"]
    subdir = build_output_subdir(run_id, tag)
    args = preset.get("default_args", {})

    cmd = [
        sys.executable, str(DETECT_SCRIPT),
        "--grid-id", grid_id,
        "--model-path", str(model["checkpoint"]),
        "--output-subdir", subdir,
        "--evaluation-profile", args.get("evaluation_profile", "installation"),
        "--data-scope", suite.get("data_scope", "full_grid"),
    ]

    # Postproc config
    postproc = preset.get("postproc_config")
    if postproc:
        postproc_path = ROOT / postproc
        if postproc_path.exists():
            cmd += ["--postproc-config", str(postproc_path)]

    # Detection params
    for param, flag in [
        ("chip_size", "--chip-size"),
        ("overlap", "--overlap"),
        ("confidence_threshold", "--confidence-threshold"),
        ("mask_threshold", "--mask-threshold"),
    ]:
        val = args.get(param)
        if val is not None:
            cmd += [flag, str(val)]

    if force:
        cmd.append("--force")

    log_file = log_dir / f"{tag}_{grid_id}.log"
    print(f"  [{tag}] {grid_id} ... ", end="", flush=True)

    try:
        with open(log_file, "w") as lf:
            result = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                timeout=600, cwd=ROOT,
            )
        if result.returncode != 0:
            print(f"FAIL (exit={result.returncode})")
            return {"grid_id": grid_id, "model_tag": tag, "error": f"exit={result.returncode}", "log": str(log_file)}
        print("OK")
        return None
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"grid_id": grid_id, "model_tag": tag, "error": "timeout", "log": str(log_file)}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"grid_id": grid_id, "model_tag": tag, "error": str(e)}


# ════════════════════════════════════════════════════════════════════════
# Metrics Collection
# ════════════════════════════════════════════════════════════════════════

def collect_grid_metrics(
    model: dict, grid_id: str, suite: dict, run_id: str,
) -> dict | None:
    """Read metrics CSVs for one model x one grid. Returns None if missing."""
    subdir = build_output_subdir(run_id, model["tag"])
    result_dir = ROOT / "results" / grid_id / subdir

    # Presence metrics (required)
    presence_csv = result_dir / "presence_metrics.csv"
    if not presence_csv.exists():
        return None

    try:
        with open(presence_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        r = rows[0]
    except Exception:
        return None

    tp = int(float(r.get("tp", 0)))
    fp = int(float(r.get("fp", 0)))
    fn = int(float(r.get("fn", 0)))
    gt = int(float(r.get("gt_count", r.get("n_gt", 0))))
    pred = int(float(r.get("pred_count", r.get("n_pred", 0))))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0

    row = {
        "model_key": model["key"],
        "model_tag": model["tag"],
        "suite_id": suite["suite_id"],
        "suite_role": suite.get("role", ""),
        "leakage_risk": suite.get("leakage_risk", ""),
        "grid_id": grid_id,
        "output_subdir": subdir,
        "gt_count": gt,
        "pred_count": pred,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": rec, "f1": f1,
    }

    # Footprint metrics (optional)
    footprint_csv = result_dir / "footprint_metrics.csv"
    if footprint_csv.exists():
        try:
            with open(footprint_csv) as f:
                frows = list(csv.DictReader(f))
            if frows:
                fr = frows[0]
                row.update({
                    "n_matches": int(float(fr.get("n_matches", 0) or 0)),
                    "mean_iou": float(fr.get("mean_iou", 0) or 0),
                    "median_iou": float(fr.get("median_iou", 0) or 0),
                    "iou_ge_0.5_rate": float(fr.get("iou_ge_0.5_rate", 0) or 0),
                    "mean_dice": float(fr.get("mean_dice", 0) or 0),
                })
        except Exception:
            pass

    # Area error metrics (optional, weighted aggregate)
    area_csv = result_dir / "area_error_metrics.csv"
    if area_csv.exists():
        try:
            with open(area_csv) as f:
                arows = list(csv.DictReader(f))
            if arows:
                total_m = sum(int(float(ar.get("n_matches", 0))) for ar in arows)
                if total_m > 0:
                    w_abs = sum(
                        float(ar.get("mean_abs_error_m2", 0)) * int(float(ar.get("n_matches", 0)))
                        for ar in arows
                    ) / total_m
                    row["area_mean_abs_error_m2"] = w_abs
        except Exception:
            pass

    return row


# ════════════════════════════════════════════════════════════════════════
# Aggregation
# ════════════════════════════════════════════════════════════════════════

def _weighted_avg(rows: list[dict], val_key: str, weight_key: str) -> float:
    total_w = sum(r.get(weight_key, 0) for r in rows if val_key in r)
    if total_w <= 0:
        return 0.0
    return sum(r.get(val_key, 0) * r.get(weight_key, 0) for r in rows if val_key in r) / total_w


def build_suite_summary(grid_rows: list[dict]) -> list[dict]:
    """Aggregate grid-level metrics to suite-level."""
    from itertools import groupby

    key_fn = lambda r: (r["model_key"], r["model_tag"], r["suite_id"], r["suite_role"], r["leakage_risk"])
    sorted_rows = sorted(grid_rows, key=key_fn)
    summaries = []

    for keys, group_iter in groupby(sorted_rows, key=key_fn):
        group = list(group_iter)
        model_key, model_tag, suite_id, suite_role, leakage_risk = keys

        tp = sum(r["tp"] for r in group)
        fp = sum(r["fp"] for r in group)
        fn = sum(r["fn"] for r in group)
        p_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0

        summaries.append({
            "model_key": model_key,
            "model_tag": model_tag,
            "suite_id": suite_id,
            "suite_role": suite_role,
            "leakage_risk": leakage_risk,
            "grid_count": len(group),
            "gt_count_total": sum(r["gt_count"] for r in group),
            "tp_total": tp, "fp_total": fp, "fn_total": fn,
            "precision_micro": p_micro,
            "recall_micro": r_micro,
            "f1_micro": f1_micro,
            "precision_macro": sum(r["precision"] for r in group) / len(group),
            "recall_macro": sum(r["recall"] for r in group) / len(group),
            "f1_macro": sum(r["f1"] for r in group) / len(group),
            "mean_iou_weighted": _weighted_avg(group, "mean_iou", "n_matches"),
            "iou_ge_0.5_rate_weighted": _weighted_avg(group, "iou_ge_0.5_rate", "n_matches"),
        })

    return summaries


def add_baseline_deltas(suite_rows: list[dict], baseline_key: str) -> list[dict]:
    """Add delta columns vs baseline for each suite."""
    # Index baseline by suite_id
    baseline_by_suite = {}
    for r in suite_rows:
        if r["model_key"] == baseline_key:
            baseline_by_suite[r["suite_id"]] = r

    delta_cols = ["precision_micro", "recall_micro", "f1_micro", "fp_total", "fn_total", "mean_iou_weighted"]
    for r in suite_rows:
        base = baseline_by_suite.get(r["suite_id"])
        if base:
            for col in delta_cols:
                r[f"delta_{col}"] = r.get(col, 0) - base.get(col, 0)
        else:
            for col in delta_cols:
                r[f"delta_{col}"] = None

    return suite_rows


# ════════════════════════════════════════════════════════════════════════
# Verdict
# ════════════════════════════════════════════════════════════════════════

def compute_verdicts(
    suite_rows: list[dict],
    models: list[dict],
    baseline_key: str,
    rules: dict,
) -> list[dict]:
    """Compute per-model verdict based on primary suite metrics."""
    imp_thresh = rules.get("improved_threshold", 0.005)
    reg_thresh = rules.get("regressed_threshold", -0.005)
    mixed_drop = rules.get("mixed_drop_threshold", 0.02)

    verdicts = []
    for m in models:
        if m["key"] == baseline_key:
            continue

        # Find primary suite row for this model
        primary_rows = [
            r for r in suite_rows
            if r["model_key"] == m["key"] and r["suite_role"] == "primary"
        ]

        if not primary_rows:
            verdicts.append({
                "model_tag": m["tag"],
                "overall_status": "failed",
                "primary_metrics": {},
                "delta_vs_baseline": {},
                "alerts": ["No primary suite results found"],
            })
            continue

        pr = primary_rows[0]
        delta_f1 = pr.get("delta_f1_micro")
        delta_p = pr.get("delta_precision_micro")
        delta_r = pr.get("delta_recall_micro")

        if delta_f1 is None:
            status = "failed"
            alerts = ["No baseline delta available"]
        elif delta_f1 >= imp_thresh:
            status = "improved"
            alerts = []
            # Check for mixed: F1 up but P or R drops significantly
            if (delta_p is not None and delta_p < -mixed_drop) or \
               (delta_r is not None and delta_r < -mixed_drop):
                status = "mixed"
                if delta_p is not None and delta_p < -mixed_drop:
                    alerts.append(f"Precision dropped {delta_p:+.1%}")
                if delta_r is not None and delta_r < -mixed_drop:
                    alerts.append(f"Recall dropped {delta_r:+.1%}")
        elif delta_f1 <= reg_thresh:
            status = "regressed"
            alerts = [f"F1 dropped {delta_f1:+.1%}"]
        else:
            status = "flat"
            alerts = []

        verdicts.append({
            "model_tag": m["tag"],
            "overall_status": status,
            "primary_metrics": {
                "f1_micro": pr["f1_micro"],
                "precision_micro": pr["precision_micro"],
                "recall_micro": pr["recall_micro"],
            },
            "delta_vs_baseline": {
                k: pr.get(f"delta_{k}")
                for k in ["f1_micro", "precision_micro", "recall_micro", "fp_total", "fn_total"]
            },
            "alerts": alerts,
        })

    return verdicts


def determine_winner(verdicts: list[dict], baseline_tag: str) -> str:
    """Return tag of the best model (highest primary F1)."""
    best_tag = baseline_tag
    best_f1 = 0.0

    # Baseline F1 is implicitly 0 delta; we need its absolute value
    # but we don't have it in verdicts. Use a simpler approach:
    # winner is the model with highest absolute primary F1.
    candidates = []
    for v in verdicts:
        f1 = v.get("primary_metrics", {}).get("f1_micro", 0)
        candidates.append((v["model_tag"], f1))

    if not candidates:
        return baseline_tag

    # Sort by F1 descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# ════════════════════════════════════════════════════════════════════════
# Output Writers
# ════════════════════════════════════════════════════════════════════════

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Stable field order: union all keys
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_manifest(
    path: Path,
    *,
    preset: dict,
    run_id: str,
    models: list[dict],
    baseline_key: str,
    checksum_mode: str,
    checksums: dict,
) -> None:
    git_commit = None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        git_commit = proc.stdout.strip()
    except Exception:
        pass

    manifest = {
        "benchmark_id": preset["benchmark_id"],
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "baseline_model": baseline_key,
        "checksum_mode": checksum_mode,
        "models": [
            {
                "key": m["key"],
                "tag": m["tag"],
                "checkpoint": str(m["checkpoint"]),
                "sha256": checksums.get(m["key"]),
            }
            for m in models
        ],
        "preset_config": preset,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")


def write_summary_json(
    path: Path,
    *,
    run_id: str,
    preset: dict,
    models: list[dict],
    baseline_key: str,
    baseline_tag: str,
    verdicts: list[dict],
    suite_rows: list[dict],
    failures: list[dict],
) -> None:
    """Write the agent-first summary JSON."""
    # Find primary suite id
    primary_suites = [s for s in preset.get("suites", []) if s.get("role") == "primary"]
    primary_suite = primary_suites[0]["suite_id"] if primary_suites else None

    # Determine winner: baseline or best improved model
    winner = baseline_tag
    best_f1 = 0.0
    # Get baseline's primary F1
    for r in suite_rows:
        if r["model_key"] == baseline_key and r["suite_role"] == "primary":
            best_f1 = r["f1_micro"]
            break
    for v in verdicts:
        f1 = v.get("primary_metrics", {}).get("f1_micro", 0)
        if f1 > best_f1:
            best_f1 = f1
            winner = v["model_tag"]

    summary = {
        "run_id": run_id,
        "benchmark_id": preset["benchmark_id"],
        "baseline_model": baseline_tag,
        "winner": winner,
        "primary_suite": primary_suite,
        "models": [m["tag"] for m in models],
        "model_verdicts": verdicts,
        "failures": failures,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")


def write_summary_md(
    path: Path,
    *,
    run_id: str,
    preset: dict,
    models: list[dict],
    baseline_tag: str,
    verdicts: list[dict],
    suite_rows: list[dict],
    failures: list[dict],
) -> None:
    """Write human-readable Markdown summary with TL;DR header."""
    lines = []

    # ── TL;DR ─────────────────────────────────────────────────────
    primary_suites = [s for s in preset.get("suites", []) if s.get("role") == "primary"]
    primary_suite_id = primary_suites[0]["suite_id"] if primary_suites else "N/A"

    # Find winner
    winner = baseline_tag
    best_f1 = 0.0
    for r in suite_rows:
        if r.get("suite_role") == "primary":
            if r["f1_micro"] > best_f1:
                best_f1 = r["f1_micro"]
                winner = r["model_tag"]

    tldr_parts = [f"Winner: **{winner}** (F1={best_f1:.1%} on {primary_suite_id})"]
    for v in verdicts:
        delta = v.get("delta_vs_baseline", {}).get("f1_micro")
        if delta is not None:
            tldr_parts.append(f"{v['model_tag']}: **{v['overall_status']}** (F1 {delta:+.1%})")
    if failures:
        tldr_parts.append(f"{len(failures)} grid(s) failed")

    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("> " + " | ".join(tldr_parts))
    lines.append("")
    lines.append(f"- Run: `{run_id}`")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`")
    lines.append(f"- Baseline: `{baseline_tag}`")
    lines.append("")

    # ── Models ────────────────────────────────────────────────────
    lines.append("## Models")
    lines.append("")
    lines.append("| Tag | Description | COCO AP50 |")
    lines.append("|-----|-------------|-----------|")
    for m in models:
        ap = f"{m['coco_ap50']:.4f}" if m.get("coco_ap50") else "-"
        lines.append(f"| **{m['tag']}** | {m['description']} | {ap} |")
    lines.append("")

    # ── Verdicts ──────────────────────────────────────────────────
    if verdicts:
        lines.append("## Verdicts")
        lines.append("")
        lines.append("| Model | Status | F1 | dF1 | dPrec | dRecall | Alerts |")
        lines.append("|-------|--------|----|----|-------|---------|--------|")
        for v in verdicts:
            pm = v.get("primary_metrics", {})
            dv = v.get("delta_vs_baseline", {})
            f1 = pm.get("f1_micro")
            df1 = dv.get("f1_micro")
            dp = dv.get("precision_micro")
            dr = dv.get("recall_micro")
            alerts = "; ".join(v.get("alerts", []))
            f1_s = f"{f1:.1%}" if f1 is not None else "-"
            df1_s = f"{df1:+.1%}" if df1 is not None else "-"
            dp_s = f"{dp:+.1%}" if dp is not None else "-"
            dr_s = f"{dr:+.1%}" if dr is not None else "-"
            lines.append(
                f"| {v['model_tag']} | **{v['overall_status']}** "
                f"| {f1_s} | {df1_s} | {dp_s} | {dr_s} "
                f"| {alerts or '-'} |"
            )
        lines.append("")

    # ── Suite Summary ─────────────────────────────────────────────
    lines.append("## Suite Summary")
    lines.append("")
    lines.append("| Suite | Role | Model | P | R | F1 | dF1 | FP | FN | IoU |")
    lines.append("|-------|------|-------|---|---|----|----|----|----|-----|")
    for r in sorted(suite_rows, key=lambda x: (x["suite_id"], -x["f1_micro"])):
        df1 = r.get("delta_f1_micro")
        df1_str = f"{df1:+.4f}" if df1 is not None else "-"
        iou = r.get("mean_iou_weighted", 0)
        lines.append(
            f"| {r['suite_id']} | {r['suite_role']} | {r['model_tag']} "
            f"| {r['precision_micro']:.1%} | {r['recall_micro']:.1%} | **{r['f1_micro']:.1%}** "
            f"| {df1_str} | {r['fp_total']} | {r['fn_total']} | {iou:.3f} |"
        )
    lines.append("")

    # ── Failures ──────────────────────────────────────────────────
    if failures:
        lines.append("## Failures")
        lines.append("")
        for fail in failures:
            lines.append(f"- {fail.get('model_tag', '?')} / {fail.get('grid_id', '?')}: {fail.get('error', '?')}")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────
    lines.append("---")
    postproc = preset.get("postproc_config", "default")
    lines.append(f"*Postproc config: `{postproc}`*")

    path.write_text("\n".join(lines) + "\n")


def safe_plot(suite_rows: list[dict], plots_dir: Path) -> None:
    """Generate comparison plots. Failures are silently caught."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not suite_rows:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # F1 by suite grouped bar chart
    try:
        tags = sorted(set(r["model_tag"] for r in suite_rows))
        suites = sorted(set(r["suite_id"] for r in suite_rows))
        data = {}
        for r in suite_rows:
            data[(r["suite_id"], r["model_tag"])] = r["f1_micro"]

        import numpy as np
        x = np.arange(len(suites))
        width = 0.8 / max(len(tags), 1)

        fig, ax = plt.subplots(figsize=(max(10, len(suites) * 2), 6))
        for i, tag in enumerate(tags):
            vals = [data.get((s, tag), 0) for s in suites]
            ax.bar(x + i * width, vals, width, label=tag)

        ax.set_title("F1 (micro) by Benchmark Suite")
        ax.set_ylabel("F1")
        ax.set_xticks(x + width * (len(tags) - 1) / 2)
        ax.set_xticklabels(suites, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "f1_by_suite.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # P-R scatter
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for tag in sorted(set(r["model_tag"] for r in suite_rows)):
            rows = [r for r in suite_rows if r["model_tag"] == tag]
            ax.scatter(
                [r["recall_micro"] for r in rows],
                [r["precision_micro"] for r in rows],
                label=tag, s=80,
            )
            for r in rows:
                ax.annotate(r["suite_id"], (r["recall_micro"], r["precision_micro"]), fontsize=7)
        ax.set_title("Precision-Recall by Suite")
        ax.set_xlabel("Recall (micro)")
        ax.set_ylabel("Precision (micro)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "precision_recall.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


def print_console_summary(
    models: list[dict],
    suite_rows: list[dict],
    verdicts: list[dict],
    baseline_tag: str,
) -> None:
    """Print concise summary to terminal."""
    print("\n" + "=" * 62)
    print("BENCHMARK RESULTS")
    print("=" * 62)

    # Aggregate across primary suites
    print(f"\n{'Model':<12} {'Prec':>7} {'Recall':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}  Verdict")
    print("-" * 70)

    for m in models:
        primary = [r for r in suite_rows if r["model_key"] == m["key"] and r["suite_role"] == "primary"]
        if not primary:
            print(f"{m['tag']:<12} {'(no primary data)':>30}")
            continue
        pr = primary[0]
        verdict = next((v for v in verdicts if v["model_tag"] == m["tag"]), None)
        status = verdict["overall_status"] if verdict else "baseline"
        print(
            f"{m['tag']:<12} {pr['precision_micro']:>6.1%} {pr['recall_micro']:>6.1%} "
            f"{pr['f1_micro']:>6.1%} {pr['tp_total']:>5} {pr['fp_total']:>5} {pr['fn_total']:>5}  {status}"
        )


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Agent-First Model Benchmark — 模型权重对比评估",
    )
    parser.add_argument("--preset", default=str(DEFAULT_PRESET),
                        help="Benchmark preset YAML (default: post_train)")
    parser.add_argument("--models", nargs="+",
                        help="Model keys from registry (default: preset's default_models)")
    parser.add_argument("--checkpoint",
                        help="Ad-hoc checkpoint path to include")
    parser.add_argument("--tag",
                        help="Tag for ad-hoc checkpoint")
    parser.add_argument("--baseline", dest="baseline_key",
                        help="Baseline model key (default: preset's baseline_model)")
    parser.add_argument("--suite", action="append", dest="suites",
                        help="Only run named suite(s). Repeatable.")
    parser.add_argument("--grids", nargs="+",
                        help="Override grid list for all suites (ad-hoc testing)")
    parser.add_argument("--run-name",
                        help="Custom run name (default: auto-generated)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run inference (pass --force to detect_and_evaluate.py)")
    parser.add_argument("--collect-only", action="store_true",
                        help="Skip inference, collect existing results only")
    parser.add_argument("--checksum", choices=["auto", "skip", "force"], default="auto",
                        help="Checksum mode for model files (default: auto)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────
    preset_path = Path(args.preset)
    if not preset_path.is_absolute():
        preset_path = ROOT / preset_path
    preset = load_preset(preset_path)

    registry = load_model_registry()
    default_models = preset.get("default_models", [])
    baseline_key = args.baseline_key or preset.get("baseline_model", "")

    models = resolve_models(registry, args.models, default_models, args.checkpoint, args.tag)
    suites = select_suites(preset, args.suites)

    if not models:
        print("[ERROR] No valid models found.")
        sys.exit(1)
    if not suites:
        print("[ERROR] No suites selected.")
        sys.exit(1)

    baseline_tag = baseline_key
    for m in models:
        if m["key"] == baseline_key:
            baseline_tag = m["tag"]
            break

    # Generate run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or "_".join(m["tag"] for m in models)
    run_id = f"{run_name}_{ts}"

    # Count total grids
    total_grids = sum(len(s.get("grid_ids", [])) for s in suites)
    all_primary = [s for s in preset.get("suites", []) if s.get("role") == "primary"]
    primary_label = f"{all_primary[0]['suite_id']} ({len(all_primary[0].get('grid_ids', []))} grids)" if all_primary else "N/A"
    active_primary = [s for s in suites if s.get("role") == "primary"]
    if all_primary and not active_primary:
        primary_label += " (not in selected suites)"

    # ── Explicit startup banner ───────────────────────────────────
    print("=" * 62)
    print("Model Benchmark")
    print(f"  Preset:   {preset['benchmark_id']}")
    print(f"  Models:   {', '.join(m['tag'] for m in models)}")
    print(f"  Baseline: {baseline_tag}")
    print(f"  Primary:  {primary_label}")
    print(f"  Suites:   {len(suites)} total, {total_grids} grids")
    print(f"  Run ID:   {run_id}")
    print("=" * 62)

    # Output directory
    run_dir = ROOT / "results" / "benchmark" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # ── Checksums ─────────────────────────────────────────────────
    checksums = {}
    for m in models:
        checksums[m["key"]] = get_checksum(m["checkpoint"], mode=args.checksum)

    # ── Write manifest ────────────────────────────────────────────
    write_manifest(
        run_dir / "manifest.json",
        preset=preset, run_id=run_id, models=models,
        baseline_key=baseline_key, checksum_mode=args.checksum,
        checksums=checksums,
    )

    # ── Run inference ─────────────────────────────────────────────
    failures = []
    if not args.collect_only:
        for m in models:
            print(f"\n[{m['tag']}] {m['description']}")
            for suite in suites:
                grid_ids = args.grids or suite.get("grid_ids", [])
                for gid in grid_ids:
                    fail = run_one_grid(
                        model=m, grid_id=gid, suite=suite, preset=preset,
                        run_id=run_id, force=args.force, log_dir=log_dir,
                    )
                    if fail:
                        failures.append(fail)

    # ── Collect metrics ───────────────────────────────────────────
    print("\nCollecting metrics...")
    grid_rows = []
    for m in models:
        for suite in suites:
            grid_ids = args.grids or suite.get("grid_ids", [])
            for gid in grid_ids:
                row = collect_grid_metrics(m, gid, suite, run_id)
                if row:
                    grid_rows.append(row)
                else:
                    print(f"  [MISS] {m['tag']} x {gid}")

    if not grid_rows:
        print("[WARN] No metrics collected.")
        # Still write summary with empty results
        write_summary_json(
            run_dir / "summary.json",
            run_id=run_id, preset=preset, models=models,
            baseline_key=baseline_key, baseline_tag=baseline_tag,
            verdicts=[], suite_rows=[], failures=failures,
        )
        write_summary_md(
            run_dir / "summary.md",
            run_id=run_id, preset=preset, models=models,
            baseline_tag=baseline_tag, verdicts=[], suite_rows=[],
            failures=failures,
        )
        print(f"\nResults (empty): {run_dir}")
        return

    expected = len(models) * sum(len(args.grids or s.get("grid_ids", [])) for s in suites)
    print(f"Collected {len(grid_rows)}/{expected} results")

    # ── Aggregate ─────────────────────────────────────────────────
    suite_rows = build_suite_summary(grid_rows)
    suite_rows = add_baseline_deltas(suite_rows, baseline_key)

    # ── Verdicts ──────────────────────────────────────────────────
    rules = preset.get("verdict_rules", {})
    verdicts = compute_verdicts(suite_rows, models, baseline_key, rules)

    # ── Write outputs ─────────────────────────────────────────────
    write_csv(run_dir / "by_grid.csv", grid_rows)
    write_csv(run_dir / "by_suite.csv", suite_rows)

    write_summary_json(
        run_dir / "summary.json",
        run_id=run_id, preset=preset, models=models,
        baseline_key=baseline_key, baseline_tag=baseline_tag,
        verdicts=verdicts, suite_rows=suite_rows, failures=failures,
    )

    write_summary_md(
        run_dir / "summary.md",
        run_id=run_id, preset=preset, models=models,
        baseline_tag=baseline_tag, verdicts=verdicts, suite_rows=suite_rows,
        failures=failures,
    )

    safe_plot(suite_rows, run_dir / "plots")

    # ── Console summary ───────────────────────────────────────────
    print_console_summary(models, suite_rows, verdicts, baseline_tag)
    print(f"\nFull results: {run_dir}")
    print(f"  summary.json — agent 读取")
    print(f"  summary.md   — 人类阅读")


if __name__ == "__main__":
    main()
