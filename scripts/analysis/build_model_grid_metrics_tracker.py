#!/usr/bin/env python3
"""Aggregate per-model x per-grid P/R/F1 metrics into a single tracker CSV + markdown summary.

Scans every ``presence_metrics.csv`` under ``results/`` (and the
``results_joburg/`` symlink, if not already covered), pairs each with the
adjacent ``config.json`` to recover model/run/provenance metadata, and also
pulls footprint IoU/Dice if a sibling ``footprint_metrics.csv`` exists.

Output:
    results/analysis/model_grid_metrics_tracker.csv
    results/analysis/model_grid_metrics_tracker.md

No inference or evaluation is re-run; this only aggregates what already exists.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/gaosh/projects/ZAsolar")
OUTPUT_DIR = REPO_ROOT / "results" / "analysis"
OUTPUT_CSV = OUTPUT_DIR / "model_grid_metrics_tracker.csv"
OUTPUT_MD = OUTPUT_DIR / "model_grid_metrics_tracker.md"

COLUMNS = [
    "region",
    "model_family",
    "model_run",
    "model_checkpoint",
    "imagery_layer",
    "grid_id",
    "result_dir",
    "gt_count",
    "pred_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "footprint_mean_iou",
    "footprint_mean_dice",
    "footprint_iou_ge_0.5_rate",
    "evaluation_profile",
    "annotation_tier_mix",
    "data_scope",
    "post_conf",
    "min_area",
    "max_elong",
    "created_at_utc",
    "eval_status",
    "eval_exclusion_reason",
    "is_valid_eval",
]


def family_of(run: str, ckpt: str = "") -> str:
    """Collapse fine-grained model_run into a coarse model family.

    For ``benchmark_*`` sub-runs the convention is
    ``benchmark_<lineup>_<date>_<MODEL_TAG>`` — the trailing ``<MODEL_TAG>``
    names the actual model used for those predictions. That must win over
    tokens in the parent directory (e.g. ``v4_aerial_2023/benchmark_..._V3-A``
    is V3-A predictions, not V4).
    """
    s_full = f"{run} {ckpt}".lower()

    # Benchmark sub-runs: trust the trailing tag after the last underscore
    run_lc = run.lower()
    if "benchmark_" in run_lc:
        last = run_lc.rsplit("_", 1)[-1]
        if last in {"v3-a", "v3a"}:
            return "v3-a"
        if last in {"v3-c-hn", "v3-c", "v3c", "v3chn"}:
            return "v3-c"
        if last in {"v4.1-hn", "v4.1", "v4_1"}:
            return "v4.1"
        if last in {"v4.2", "v4_2"}:
            return "v4.2"
        if last in {"v4.3", "v4_3"}:
            return "v4.3"
        if last in {"v4", "v4-hn"}:
            return "v4"

    # Non-benchmark runs: order matters (longer/more specific first)
    s = s_full
    if "v4_3" in s or "v4.3" in s:
        return "v4.3"
    if "v4_2" in s or "v4.2" in s:
        return "v4.2"
    if "v4_1" in s or "v4.1" in s:
        return "v4.1"
    if "exp005" in s:
        return "v4.1"
    if "exp004" in s or "v4_aerial" in s or "v4_legacy" in s:
        return "v4"
    if "v3c" in s or "v3-c" in s or "v3_c" in s or "exp003" in s or "targeted_hn" in s:
        return "v3-c"
    if "v3-a" in s:
        return "v3-a"
    if "v1_ft" in s or "baseline_2025" in s or "calibrated_2025" in s:
        return "v1"
    return "other"


def read_json_safe(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text().strip()
        if not text:
            return {}
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def infer_region(rel_parts: tuple[str, ...]) -> str:
    """Infer region from path components.

    Paths we see (rel to repo root):
      results/<grid>/...                                 -> ct (legacy CT flat)
      results/cape_town/<run>/<grid>/...                 -> ct
      results/johannesburg/<run>/<grid>/...              -> jhb
      results_joburg/<grid>/...                          -> jhb (symlink)
    """
    if not rel_parts:
        return ""
    top = rel_parts[0]
    if top == "results_joburg":
        return "jhb"
    if top == "results":
        if len(rel_parts) >= 2 and rel_parts[1] == "johannesburg":
            return "jhb"
        if len(rel_parts) >= 2 and rel_parts[1] == "cape_town":
            return "ct"
        # flat results/<grid>/... is legacy CT
        return "ct"
    return ""


def infer_model_run(rel_parts: tuple[str, ...], cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (model_run, model_checkpoint)."""
    cfg_inner = cfg.get("config", {}) if isinstance(cfg, dict) else {}
    model_path = cfg_inner.get("model_path", "") or ""
    ckpt = ""
    if model_path:
        p = Path(model_path)
        # take last 2 components so "exp003_C_targeted_hn/best_model.pth"
        if len(p.parts) >= 2:
            ckpt = f"{p.parts[-2]}/{p.name}"
        else:
            ckpt = p.name

    # Path shape, ignoring the presence_metrics.csv leaf
    # parts[0] = "results" or "results_joburg"
    parts = rel_parts
    if parts[0] == "results" and len(parts) >= 3 and parts[1] in {"cape_town", "johannesburg"}:
        run_id = parts[2]
        grid = parts[3] if len(parts) >= 4 else ""
        # benchmark sub-run nested under run/grid
        if len(parts) >= 5:
            sub = parts[4]
            return f"{run_id}/{sub}", ckpt
        return run_id, ckpt

    if parts[0] == "results" and len(parts) >= 3 and parts[1].startswith("G"):
        # results/<grid>/<sub>/presence_metrics.csv
        sub = parts[2]
        return sub, ckpt

    if parts[0] == "results" and len(parts) == 2 and parts[1].startswith("G"):
        # results/<grid>/presence_metrics.csv  (flat legacy)
        label = "legacy_v3c_flat"
        if ckpt:
            if "exp004" in ckpt:
                label = "v4_legacy_flat"
            elif "exp003" in ckpt or "targeted_hn" in ckpt:
                label = "v3c_legacy_flat"
            elif "v1_ft" in ckpt:
                label = "v1_ft_legacy_flat"
        return label, ckpt

    if parts[0] == "results_joburg":
        # results_joburg/<grid>/<sub?>
        if len(parts) >= 3 and parts[1].startswith("G"):
            return parts[2], ckpt
        if len(parts) == 2:
            return "jhb_legacy_flat", ckpt

    return "unknown", ckpt


def infer_grid_id(rel_parts: tuple[str, ...], csv_row_grid: str) -> str:
    for part in rel_parts:
        if part.startswith("G") and part[1:].isdigit():
            return part
    return csv_row_grid


def infer_imagery_layer(rel_parts: tuple[str, ...], cfg: dict[str, Any]) -> str:
    cfg_inner = cfg.get("config", {}) if isinstance(cfg, dict) else {}
    tiles_dir = (cfg_inner.get("tiles_dir", "") or "").lower()
    full = "/".join(rel_parts).lower()

    if "geid" in full or "geid" in tiles_dir:
        return "geid_2024_02"
    if "aerial_2025" in full:
        return "aerial_2025"
    if "aerial_2023" in full:
        return "aerial_2023"
    # heuristics on tiles_dir
    if "tiles_joburg" in tiles_dir or "joburg" in tiles_dir:
        return "aerial_2023"
    # CT default
    if rel_parts[0] == "results" and (
        (len(rel_parts) >= 2 and rel_parts[1] == "cape_town")
        or (len(rel_parts) >= 2 and rel_parts[1].startswith("G"))
    ):
        # legacy CT used 2023 aerial (the old checkpoints predate aerial_2025)
        return "aerial_2023"
    return ""


def parse_float(s: str | None) -> float | str:
    if s is None or s == "":
        return ""
    try:
        return float(s)
    except (TypeError, ValueError):
        return ""


def parse_int(s: str | None) -> int | str:
    if s is None or s == "":
        return ""
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return ""


def classify_eval_row(
    *,
    profile: str,
    gt_count: int | str,
    precision: float | str,
    recall: float | str,
    f1: float | str,
) -> tuple[str, str, bool]:
    """Classify whether a metrics row is safe to aggregate as a configured eval.

    A real evaluation can legitimately have ``tp == 0``. Do not use TP>0 as a
    validity test; that silently hides complete model failures and inflates
    aggregate means. Instead, require explicit evaluation metadata plus numeric
    P/R/F1 fields.
    """
    has_metrics = all(isinstance(v, (int, float)) for v in (precision, recall, f1))
    if not profile:
        return "legacy_unconfigured", "missing evaluation_config/evaluation_profile", False
    if not has_metrics:
        return "invalid_metrics", "missing or non-numeric precision/recall/f1", False
    if not isinstance(gt_count, int):
        return "invalid_metrics", "missing or non-numeric gt_count", False
    if gt_count <= 0:
        return "no_gt", "gt_count <= 0", False
    return "configured_eval", "", True


def find_all_presence_csvs() -> list[Path]:
    # Walk only real dirs under results/ (results_joburg is a symlink into results/)
    out: list[Path] = []
    for base in [REPO_ROOT / "results"]:
        if not base.exists():
            continue
        for p in base.rglob("presence_metrics.csv"):
            out.append(p)
    return sorted(out)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = find_all_presence_csvs()
    print(f"[scan] found {len(csv_paths)} presence_metrics.csv files")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for csv_path in csv_paths:
        result_dir = csv_path.parent
        rel = result_dir.relative_to(REPO_ROOT)
        rel_parts = rel.parts

        cfg = read_json_safe(result_dir / "config.json")
        if not cfg:
            warnings.append(f"empty or missing config.json: {rel}")

        cfg_inner = cfg.get("config", {}) if isinstance(cfg, dict) else {}
        eval_cfg = cfg.get("evaluation_config", {}) if isinstance(cfg, dict) else {}

        model_run, ckpt = infer_model_run(rel_parts, cfg)
        region = infer_region(rel_parts)
        imagery = infer_imagery_layer(rel_parts, cfg)

        presence_rows = read_csv_rows(csv_path)
        fp_rows = read_csv_rows(result_dir / "footprint_metrics.csv")
        fp_row = fp_rows[0] if fp_rows else {}

        fam = family_of(model_run, ckpt)

        for prow in presence_rows:
            grid_id = infer_grid_id(rel_parts, prow.get("grid_id", ""))
            tp_v = parse_int(prow.get("tp"))
            gt_v = parse_int(prow.get("gt_count"))
            precision_v = parse_float(prow.get("precision"))
            recall_v = parse_float(prow.get("recall"))
            f1_v = parse_float(prow.get("f1"))
            profile = eval_cfg.get("evaluation_profile", "")
            annotation_tier_mix = eval_cfg.get("annotation_tier_mix", "")
            data_scope = eval_cfg.get("data_scope", "")
            eval_status, eval_reason, is_valid = classify_eval_row(
                profile=profile,
                gt_count=gt_v,
                precision=precision_v,
                recall=recall_v,
                f1=f1_v,
            )
            rows.append(
                {
                    "region": region,
                    "model_family": fam,
                    "model_run": model_run,
                    "model_checkpoint": ckpt,
                    "imagery_layer": imagery,
                    "grid_id": grid_id,
                    "result_dir": str(rel),
                    "gt_count": gt_v,
                    "pred_count": parse_int(prow.get("pred_count")),
                    "tp": tp_v,
                    "fp": parse_int(prow.get("fp")),
                    "fn": parse_int(prow.get("fn")),
                    "precision": precision_v,
                    "recall": recall_v,
                    "f1": f1_v,
                    "footprint_mean_iou": parse_float(fp_row.get("mean_iou")),
                    "footprint_mean_dice": parse_float(fp_row.get("mean_dice")),
                    "footprint_iou_ge_0.5_rate": parse_float(fp_row.get("iou_ge_0.5_rate")),
                    "evaluation_profile": profile,
                    "annotation_tier_mix": annotation_tier_mix,
                    "data_scope": data_scope,
                    "post_conf": parse_float(str(cfg_inner.get("post_conf_threshold", ""))),
                    "min_area": parse_float(str(cfg_inner.get("min_object_area", ""))),
                    "max_elong": parse_float(str(cfg_inner.get("max_elongation", ""))),
                    "created_at_utc": cfg.get("created_at_utc", ""),
                    "eval_status": eval_status,
                    "eval_exclusion_reason": eval_reason,
                    "is_valid_eval": is_valid,
                }
            )

    # De-duplicate on (region, model_run, grid_id): prefer most recent created_at_utc
    def sort_key(r: dict[str, Any]) -> tuple:
        return (r["region"], r["model_run"], r["grid_id"], r.get("created_at_utc") or "")

    rows.sort(key=sort_key)

    dedup: dict[tuple, dict[str, Any]] = {}
    dup_keys: list[tuple] = []
    for r in rows:
        key = (r["region"], r["model_run"], r["grid_id"])
        if key in dedup:
            dup_keys.append(key)
            # keep the later created_at_utc (rows are sorted ascending)
            dedup[key] = r
        else:
            dedup[key] = r
    deduped_rows = sorted(dedup.values(), key=lambda r: (r["region"], r["model_run"], r["grid_id"]))

    # Write CSV
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in deduped_rows:
            writer.writerow({k: r.get(k, "") for k in COLUMNS})

    # Coverage summary for stdout + md
    coverage_fam: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    coverage_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in deduped_rows:
        coverage_fam[(r["region"], r["model_family"])].append(r)
        coverage_run[(r["region"], r["model_family"], r["model_run"])].append(r)

    print(f"[write] {OUTPUT_CSV.relative_to(REPO_ROOT)}: {len(deduped_rows)} rows")
    print(f"[scan] raw row count before dedup: {len(rows)}; removed dups: {len(rows) - len(deduped_rows)}")
    print("[coverage] (region, model_family) -> grid count (valid_eval only)")
    for (region, fam), group in sorted(coverage_fam.items()):
        valid = [g for g in group if g.get("is_valid_eval")]
        print(f"  {region:4s}  {fam:10s}  n_grids={len(group)}  valid={len(valid)}")

    # Markdown summary
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _mean(vals: list) -> str:
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return "-"
        return f"{sum(vals)/len(vals):.3f}"

    def _micro_prf(valid_rows: list[dict[str, Any]]) -> tuple[str, str, str, str]:
        tp = sum(g["tp"] for g in valid_rows if isinstance(g.get("tp"), int))
        fp = sum(g["fp"] for g in valid_rows if isinstance(g.get("fp"), int))
        fn = sum(g["fn"] for g in valid_rows if isinstance(g.get("fn"), int))
        if tp + fp + fn == 0:
            return "-", "-", "-", "-"
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        return f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}", f"{tp}/{fp}/{fn}"

    md: list[str] = []
    md.append("# Model x Grid Metrics Tracker")
    md.append("")
    md.append(f"Generated: {now}")
    md.append(f"Source CSV: `{OUTPUT_CSV.relative_to(REPO_ROOT)}`")
    md.append(
        f"Rows: {len(deduped_rows)} (from {len(rows)} raw presence rows across {len(csv_paths)} CSV files)"
    )
    md.append("")
    md.append("Aggregate P/R/F1 below are computed over `is_valid_eval=True` rows only:")
    md.append("configured evaluation rows with an explicit `evaluation_profile`, numeric metrics, and gt_count > 0.")
    md.append("`mean P/R/F1` are macro means over grid rows; `micro P/R/F1` pool TP/FP/FN over all valid rows in the group.")
    md.append("Rows lacking `evaluation_config` are retained in the CSV for traceability but excluded from means/totals.")
    md.append("")
    md.append("Interpretation notes:")
    md.append("- This tracker is for model-run metrics, not post-review human-corrected inventory quality.")
    md.append("  Human review outputs should be reported in a separate review/inventory section unless they are used only as GT/provenance metadata.")
    md.append("- Current configured rows mostly use `annotation_tier_mix=T1+T2`; treat these as historical/internal diagnostics, not strict gold validation.")
    md.append("- Per-polygon P/R/F1 is diagnostic under V1.4; grid-level aggregate inventory validation remains the headline objective.")
    md.append("")

    # Primary view: region x model_family
    md.append("## Primary view: region x model_family")
    md.append("")
    md.append("| region | model_family | n_grids (valid) | mean P | mean R | mean F1 | micro P | micro R | micro F1 | TP/FP/FN | imagery_layers | tier_mix | n_runs |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|")
    for (region, fam), group in sorted(coverage_fam.items()):
        valid = [g for g in group if g.get("is_valid_eval")]
        precs = [g["precision"] for g in valid]
        recs = [g["recall"] for g in valid]
        f1s = [g["f1"] for g in valid]
        micro_p, micro_r, micro_f1, totals = _micro_prf(valid)
        imagery = sorted({g["imagery_layer"] or "?" for g in group})
        tier_mix = sorted({g["annotation_tier_mix"] or "?" for g in valid})
        runs = sorted({g["model_run"] for g in group})
        md.append(
            f"| {region} | **{fam}** | {len(valid)} / {len(group)} | "
            f"{_mean(precs)} | {_mean(recs)} | {_mean(f1s)} | "
            f"{micro_p} | {micro_r} | {micro_f1} | {totals} | "
            f"{','.join(imagery)} | {','.join(tier_mix) if tier_mix else '-'} | {len(runs)} |"
        )
    md.append("")

    # Gap matrix (unchanged shape, driven by model_family column)
    md.append("## Coverage gap matrix (valid-eval grid counts)")
    md.append("")
    families_known = ["v1", "v3-a", "v3-c", "v4", "v4.1", "v4.2", "v4.3"]
    fam_region_valid: dict[tuple[str, str], set[str]] = defaultdict(set)
    fam_region_all: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in deduped_rows:
        fam_region_all[(r["model_family"], r["region"])].add(r["grid_id"])
        if r.get("is_valid_eval"):
            fam_region_valid[(r["model_family"], r["region"])].add(r["grid_id"])

    md.append("Format: `valid_grids / total_grids`")
    md.append("")
    md.append("| family | ct | jhb |")
    md.append("|---|---|---|")
    for fam in families_known:
        ct_v = len(fam_region_valid.get((fam, "ct"), set()))
        ct_t = len(fam_region_all.get((fam, "ct"), set()))
        j_v = len(fam_region_valid.get((fam, "jhb"), set()))
        j_t = len(fam_region_all.get((fam, "jhb"), set()))
        md.append(f"| `{fam}` | {ct_v} / {ct_t} | {j_v} / {j_t} |")
    md.append("")

    # Secondary view: the fine-grained model_run detail
    md.append("## Detail view: model_run breakdown (for traceability)")
    md.append("")
    md.append("| region | model_family | model_run | n_grids (valid) | mean P | mean R | mean F1 | micro P | micro R | micro F1 | TP/FP/FN | imagery | tier_mix | row_status |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for (region, fam, run), group in sorted(coverage_run.items()):
        valid = [g for g in group if g.get("is_valid_eval")]
        precs = [g["precision"] for g in valid]
        recs = [g["recall"] for g in valid]
        f1s = [g["f1"] for g in valid]
        micro_p, micro_r, micro_f1, totals = _micro_prf(valid)
        imagery = sorted({g["imagery_layer"] or "?" for g in group})
        tier_mix = sorted({g["annotation_tier_mix"] or "?" for g in valid})
        statuses = defaultdict(int)
        for g in group:
            statuses[g["eval_status"]] += 1
        status_s = ", ".join(f"{k}:{v}" for k, v in sorted(statuses.items()))
        md.append(
            f"| {region} | {fam} | `{run}` | {len(valid)} / {len(group)} | "
            f"{_mean(precs)} | {_mean(recs)} | {_mean(f1s)} | "
            f"{micro_p} | {micro_r} | {micro_f1} | {totals} | {','.join(imagery)} | "
            f"{','.join(tier_mix) if tier_mix else '-'} | {status_s} |"
        )
    md.append("")

    # Flag runs that need interpretation before metrics are quoted.
    broken_runs: list[tuple[str, str, str]] = []
    for (region, fam, run), group in sorted(coverage_run.items()):
        tps = [g["tp"] for g in group if isinstance(g["tp"], int)]
        profiles = {g["evaluation_profile"] for g in group}
        statuses = {g["eval_status"] for g in group}
        if profiles == {""}:
            broken_runs.append((region, run, f"missing `evaluation_config`/`evaluation_profile` for all {len(group)} rows; retained in CSV but excluded from aggregate means"))
        if tps and max(tps) == 0:
            if statuses == {"legacy_unconfigured"}:
                broken_runs.append((region, run, f"all TP=0 across {len(group)} legacy/unconfigured rows; inspect CRS/profile before interpreting as model failure"))
            else:
                broken_runs.append((region, run, f"all TP=0 across {len(group)} rows; if configured, this is a legitimate zero-recall result and should not be filtered out"))

    if broken_runs:
        md.append("## Data quality notes")
        md.append("")
        for region, run, note in broken_runs:
            md.append(f"- `{region}` / `{run}`: {note}")
        md.append("")

    if warnings:
        md.append("## Scan warnings")
        md.append("")
        for w in warnings:
            md.append(f"- {w}")
        md.append("")

    if dup_keys:
        md.append("## Duplicate evaluations (kept most recent)")
        md.append("")
        for k in sorted(set(dup_keys)):
            md.append(f"- region=`{k[0]}`, model_run=`{k[1]}`, grid_id=`{k[2]}`")
        md.append("")

    OUTPUT_MD.write_text("\n".join(md))
    print(f"[write] {OUTPUT_MD.relative_to(REPO_ROOT)}")

    # Sanity spot-check: print first row's source
    if deduped_rows:
        r0 = deduped_rows[0]
        print("[spot-check] first row:")
        print(f"  source: {r0['result_dir']}/presence_metrics.csv")
        print(f"  region={r0['region']} run={r0['model_run']} grid={r0['grid_id']}")
        print(f"  P={r0['precision']} R={r0['recall']} F1={r0['f1']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
