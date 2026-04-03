#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = BASE_DIR / "configs" / "benchmarks" / "post_training_benchmark_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple model weights on fixed benchmark suites.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Benchmark config JSON path.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in the form label=path/to/best_model.pth or label=stock.",
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Label used as the delta baseline. Defaults to the first --model.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Only run selected suite(s). Can be repeated.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run inference even when per-grid benchmark outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing detect_and_evaluate.py.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "model"


def parse_model_spec(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError(f"Invalid --model spec: {spec!r}. Expected label=path.")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label:
        raise ValueError(f"Invalid --model label in spec: {spec!r}")

    if raw_path.lower() in {"stock", "builtin", "default"}:
        return {
            "label": label,
            "slug": slugify(label),
            "path": None,
            "display_path": "geoai_builtin",
        }

    model_path = Path(raw_path)
    if not model_path.is_absolute():
        model_path = (BASE_DIR / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    return {
        "label": label,
        "slug": slugify(label),
        "path": model_path,
        "display_path": str(model_path),
    }


def validate_config(config: dict[str, Any]) -> None:
    if not config.get("benchmark_name"):
        raise ValueError("benchmark_name is required in benchmark config")
    suites = config.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("benchmark config must contain a non-empty suites list")

    seen = set()
    for suite in suites:
        name = suite.get("name")
        if not name:
            raise ValueError("each suite requires a non-empty name")
        if name in seen:
            raise ValueError(f"duplicate suite name: {name}")
        seen.add(name)
        grid_ids = suite.get("grid_ids")
        if not isinstance(grid_ids, list) or not grid_ids:
            raise ValueError(f"suite {name} must define a non-empty grid_ids list")


def select_suites(config: dict[str, Any], selected_names: list[str] | None) -> list[dict[str, Any]]:
    suites = config["suites"]
    if not selected_names:
        return suites
    selected = []
    wanted = set(selected_names)
    for suite in suites:
        if suite["name"] in wanted:
            selected.append(suite)
    missing = wanted - {suite["name"] for suite in selected}
    if missing:
        raise ValueError(f"Unknown suite(s): {sorted(missing)}")
    return selected


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_single_row_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV has no rows: {path}")
    return df.iloc[0].to_dict()


def summarize_area_error(area_csv: Path) -> dict[str, float]:
    if not area_csv.exists():
        return {
            "area_n_matches": 0,
            "area_mean_abs_error_m2": 0.0,
            "area_mean_rel_error": 0.0,
        }
    df = pd.read_csv(area_csv)
    if df.empty or "n_matches" not in df.columns:
        return {
            "area_n_matches": 0,
            "area_mean_abs_error_m2": 0.0,
            "area_mean_rel_error": 0.0,
        }

    total = float(df["n_matches"].sum())
    if total <= 0:
        return {
            "area_n_matches": 0,
            "area_mean_abs_error_m2": 0.0,
            "area_mean_rel_error": 0.0,
        }

    return {
        "area_n_matches": int(total),
        "area_mean_abs_error_m2": float(
            (df["mean_abs_error_m2"] * df["n_matches"]).sum() / total
        ),
        "area_mean_rel_error": float(
            (df["mean_rel_error"] * df["n_matches"]).sum() / total
        ),
    }


def build_case_command(
    model: dict[str, Any],
    suite: dict[str, Any],
    grid_id: str,
    benchmark_name: str,
    run_id: str,
    inference_overrides: dict[str, Any],
    *,
    force: bool,
) -> tuple[list[str], str]:
    rel_output_subdir = str(
        Path("benchmarks") / benchmark_name / run_id / suite["name"] / model["slug"]
    )
    cmd = [
        sys.executable,
        "detect_and_evaluate.py",
        "--grid-id",
        grid_id,
        "--output-subdir",
        rel_output_subdir,
        "--evaluation-profile",
        suite.get("evaluation_profile", "installation"),
        "--data-scope",
        suite.get("data_scope", "full_grid"),
    ]
    if force:
        cmd.append("--force")
    if model["path"] is not None:
        cmd.extend(["--model-path", str(model["path"])])

    override_flag_map = {
        "chip_size": "--chip-size",
        "overlap": "--overlap",
        "min_object_area": "--min-object-area",
        "confidence_threshold": "--confidence-threshold",
        "mask_threshold": "--mask-threshold",
        "post_conf_threshold": "--post-conf-threshold",
        "max_elongation": "--max-elongation",
        "postproc_config": "--postproc-config",
    }
    for key, flag in override_flag_map.items():
        value = inference_overrides.get(key)
        if value is None:
            continue
        cmd.extend([flag, str(value)])

    return cmd, rel_output_subdir


def collect_case_metrics(grid_output_dir: Path) -> dict[str, Any]:
    presence = read_single_row_csv(grid_output_dir / "presence_metrics.csv")
    footprint = read_single_row_csv(grid_output_dir / "footprint_metrics.csv")
    area_summary = summarize_area_error(grid_output_dir / "area_error_metrics.csv")
    config_payload = load_json(grid_output_dir / "config.json")

    return {
        "gt_count": int(float(presence["gt_count"])),
        "pred_count": int(float(presence["pred_count"])),
        "tp": int(float(presence["tp"])),
        "fp": int(float(presence["fp"])),
        "fn": int(float(presence["fn"])),
        "precision": float(presence["precision"]),
        "recall": float(presence["recall"]),
        "f1": float(presence["f1"]),
        "n_matches": int(float(footprint.get("n_matches", 0))),
        "mean_iou": float(footprint.get("mean_iou", 0.0)),
        "median_iou": float(footprint.get("median_iou", 0.0)),
        "iou_ge_0.5_rate": float(footprint.get("iou_ge_0.5_rate", 0.0)),
        "mean_dice": float(footprint.get("mean_dice", 0.0)),
        "area_n_matches": int(area_summary["area_n_matches"]),
        "area_mean_abs_error_m2": float(area_summary["area_mean_abs_error_m2"]),
        "area_mean_rel_error": float(area_summary["area_mean_rel_error"]),
        "result_count": int(config_payload.get("result_count", 0)),
    }


def aggregate_suite_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]

    totals = {
        "gt_count": sum(row["gt_count"] for row in success_rows),
        "pred_count": sum(row["pred_count"] for row in success_rows),
        "tp": sum(row["tp"] for row in success_rows),
        "fp": sum(row["fp"] for row in success_rows),
        "fn": sum(row["fn"] for row in success_rows),
        "successful_grids": len(success_rows),
        "failed_grids": len(failed_rows),
    }
    precision = (
        totals["tp"] / (totals["tp"] + totals["fp"])
        if (totals["tp"] + totals["fp"]) > 0 else 0.0
    )
    recall = (
        totals["tp"] / (totals["tp"] + totals["fn"])
        if (totals["tp"] + totals["fn"]) > 0 else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    match_weight = sum(row["n_matches"] for row in success_rows)
    area_weight = sum(row["area_n_matches"] for row in success_rows)
    grid_count = len(success_rows)

    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_grid_f1": (
            sum(row["f1"] for row in success_rows) / grid_count if grid_count else 0.0
        ),
        "weighted_mean_iou": (
            sum(row["mean_iou"] * row["n_matches"] for row in success_rows) / match_weight
            if match_weight else 0.0
        ),
        "weighted_iou_ge_0.5_rate": (
            sum(row["iou_ge_0.5_rate"] * row["n_matches"] for row in success_rows) / match_weight
            if match_weight else 0.0
        ),
        "weighted_mean_dice": (
            sum(row["mean_dice"] * row["n_matches"] for row in success_rows) / match_weight
            if match_weight else 0.0
        ),
        "weighted_area_mean_abs_error_m2": (
            sum(row["area_mean_abs_error_m2"] * row["area_n_matches"] for row in success_rows) / area_weight
            if area_weight else 0.0
        ),
        "weighted_area_mean_rel_error": (
            sum(row["area_mean_rel_error"] * row["area_n_matches"] for row in success_rows) / area_weight
            if area_weight else 0.0
        ),
        "failed_grid_list": ",".join(row["grid_id"] for row in failed_rows),
    }


def format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    return df.to_markdown(index=False)


def build_markdown_report(
    config: dict[str, Any],
    models: list[dict[str, Any]],
    suite_summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    primary_regressions_df: pd.DataFrame,
    run_id: str,
    baseline_label: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark Report: {config['benchmark_name']}")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Primary suite: `{config.get('primary_suite', '')}`")
    lines.append(f"- Baseline label: `{baseline_label}`")
    lines.append("")

    model_df = pd.DataFrame([
        {
            "label": model["label"],
            "path": model["display_path"],
        }
        for model in models
    ])
    lines.append("## Models")
    lines.append("")
    lines.append(format_markdown_table(model_df))
    lines.append("")

    for suite_name in suite_summary_df["suite_name"].drop_duplicates().tolist():
        section_df = (
            suite_summary_df[suite_summary_df["suite_name"] == suite_name]
            .sort_values(["f1", "precision", "recall"], ascending=False)
            [[
                "model_label",
                "successful_grids",
                "failed_grids",
                "precision",
                "recall",
                "f1",
                "mean_grid_f1",
                "weighted_mean_iou",
                "weighted_iou_ge_0.5_rate",
                "weighted_area_mean_rel_error",
            ]]
            .copy()
        )
        lines.append(f"## Suite: {suite_name}")
        lines.append("")
        lines.append(format_markdown_table(section_df))
        lines.append("")

    if not delta_df.empty:
        lines.append("## Delta Vs Baseline")
        lines.append("")
        lines.append(format_markdown_table(delta_df))
        lines.append("")

    if not primary_regressions_df.empty:
        lines.append("## Primary Suite Regressions")
        lines.append("")
        lines.append(format_markdown_table(primary_regressions_df))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (BASE_DIR / config_path).resolve()
    config = load_json(config_path)
    validate_config(config)

    models = [parse_model_spec(spec) for spec in args.model]
    model_labels = [model["label"] for model in models]
    if len(set(model_labels)) != len(model_labels):
        raise ValueError("Model labels must be unique")

    baseline_label = args.baseline_label or models[0]["label"]
    if baseline_label not in model_labels:
        raise ValueError(f"Unknown baseline label: {baseline_label}")

    suites = select_suites(config, args.suite)
    inference_overrides = dict(config.get("inference_overrides", {}))

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    benchmark_name = config["benchmark_name"]

    benchmark_root = ensure_dir(BASE_DIR / "results" / "benchmarks" / benchmark_name / run_id)
    logs_root = ensure_dir(benchmark_root / "logs")

    manifest = {
        "benchmark_name": benchmark_name,
        "run_id": run_id,
        "config_path": str(config_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_label": baseline_label,
        "models": [
            {
                "label": model["label"],
                "slug": model["slug"],
                "path": model["display_path"],
            }
            for model in models
        ],
        "suites": suites,
        "inference_overrides": inference_overrides,
        "dry_run": args.dry_run,
        "force": args.force,
    }
    (benchmark_root / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    grid_rows: list[dict[str, Any]] = []
    total_cases = sum(len(suite["grid_ids"]) for suite in suites) * len(models)
    case_idx = 0

    for suite in suites:
        for model in models:
            for grid_id in suite["grid_ids"]:
                case_idx += 1
                cmd, rel_output_subdir = build_case_command(
                    model,
                    suite,
                    grid_id,
                    benchmark_name,
                    run_id,
                    inference_overrides,
                    force=args.force,
                )
                grid_output_dir = BASE_DIR / "results" / grid_id / rel_output_subdir
                log_dir = ensure_dir(logs_root / suite["name"] / model["slug"])
                log_path = log_dir / f"{grid_id}.log"

                print(
                    f"[{case_idx}/{total_cases}] suite={suite['name']} "
                    f"model={model['label']} grid={grid_id}"
                )

                if args.dry_run:
                    grid_rows.append({
                        "suite_name": suite["name"],
                        "model_label": model["label"],
                        "grid_id": grid_id,
                        "status": "dry_run",
                        "command": " ".join(cmd),
                        "output_dir": str(grid_output_dir),
                    })
                    print(f"  DRY-RUN {cmd}")
                    continue

                with log_path.open("w", encoding="utf-8") as log_file:
                    proc = subprocess.run(
                        cmd,
                        cwd=BASE_DIR,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )

                row = {
                    "suite_name": suite["name"],
                    "model_label": model["label"],
                    "grid_id": grid_id,
                    "output_dir": str(grid_output_dir),
                    "log_path": str(log_path),
                }
                if proc.returncode != 0:
                    row.update({
                        "status": "command_failed",
                        "error": f"detect_and_evaluate exited with code {proc.returncode}",
                    })
                    print(f"  FAIL exit={proc.returncode} log={log_path}")
                    grid_rows.append(row)
                    continue

                try:
                    metrics = collect_case_metrics(grid_output_dir)
                except Exception as exc:
                    row.update({
                        "status": "metrics_missing",
                        "error": str(exc),
                    })
                    print(f"  FAIL metrics log={log_path}")
                    grid_rows.append(row)
                    continue

                row.update(metrics)
                row["status"] = "ok"
                grid_rows.append(row)
                print(
                    "  OK "
                    f"P={row['precision']:.1%} R={row['recall']:.1%} "
                    f"F1={row['f1']:.1%} TP={row['tp']} FP={row['fp']} FN={row['fn']}"
                )

    grid_summary_df = pd.DataFrame(grid_rows)
    grid_summary_path = benchmark_root / "grid_summary.csv"
    grid_summary_df.to_csv(grid_summary_path, index=False)

    if args.dry_run:
        print(f"[OK] dry-run summary saved: {grid_summary_path}")
        return

    suite_summary_rows: list[dict[str, Any]] = []
    for suite in suites:
        for model in models:
            rows = [
                row for row in grid_rows
                if row["suite_name"] == suite["name"] and row["model_label"] == model["label"]
            ]
            suite_summary_rows.append({
                "suite_name": suite["name"],
                "model_label": model["label"],
                **aggregate_suite_rows(rows),
            })

    suite_summary_df = pd.DataFrame(suite_summary_rows)
    suite_summary_path = benchmark_root / "suite_summary.csv"
    suite_summary_df.to_csv(suite_summary_path, index=False)

    baseline_df = suite_summary_df[suite_summary_df["model_label"] == baseline_label].copy()
    baseline_by_suite = {
        row["suite_name"]: row for row in baseline_df.to_dict(orient="records")
    }

    delta_rows: list[dict[str, Any]] = []
    for row in suite_summary_df.to_dict(orient="records"):
        base = baseline_by_suite.get(row["suite_name"])
        if not base or row["model_label"] == baseline_label:
            continue
        delta_rows.append({
            "suite_name": row["suite_name"],
            "model_label": row["model_label"],
            "delta_precision": row["precision"] - base["precision"],
            "delta_recall": row["recall"] - base["recall"],
            "delta_f1": row["f1"] - base["f1"],
            "delta_weighted_mean_iou": row["weighted_mean_iou"] - base["weighted_mean_iou"],
            "delta_weighted_iou_ge_0.5_rate": (
                row["weighted_iou_ge_0.5_rate"] - base["weighted_iou_ge_0.5_rate"]
            ),
            "delta_weighted_area_mean_rel_error": (
                row["weighted_area_mean_rel_error"] - base["weighted_area_mean_rel_error"]
            ),
        })
    delta_df = pd.DataFrame(delta_rows)
    delta_path = benchmark_root / "delta_vs_baseline.csv"
    delta_df.to_csv(delta_path, index=False)

    primary_suite = config.get("primary_suite")
    primary_regressions_df = pd.DataFrame()
    if primary_suite:
        primary_rows = grid_summary_df[
            (grid_summary_df["suite_name"] == primary_suite)
            & (grid_summary_df["status"] == "ok")
        ].copy()
        baseline_grid = (
            primary_rows[primary_rows["model_label"] == baseline_label]
            [["grid_id", "f1", "precision", "recall"]]
            .rename(columns={
                "f1": "baseline_f1",
                "precision": "baseline_precision",
                "recall": "baseline_recall",
            })
        )
        regressions: list[pd.DataFrame] = []
        for model in models:
            if model["label"] == baseline_label:
                continue
            candidate = primary_rows[primary_rows["model_label"] == model["label"]].merge(
                baseline_grid,
                on="grid_id",
                how="inner",
            )
            if candidate.empty:
                continue
            candidate["model_label"] = model["label"]
            candidate["delta_f1"] = candidate["f1"] - candidate["baseline_f1"]
            candidate["delta_precision"] = candidate["precision"] - candidate["baseline_precision"]
            candidate["delta_recall"] = candidate["recall"] - candidate["baseline_recall"]
            regressions.append(candidate)

        if regressions:
            primary_regressions_df = (
                pd.concat(regressions, ignore_index=True)
                .sort_values("delta_f1")
                [[
                    "model_label",
                    "grid_id",
                    "baseline_f1",
                    "f1",
                    "delta_f1",
                    "delta_precision",
                    "delta_recall",
                ]]
                .head(12)
            )
            primary_regressions_df.to_csv(
                benchmark_root / "primary_suite_regressions.csv",
                index=False,
            )

    report_md = build_markdown_report(
        config,
        models,
        suite_summary_df,
        delta_df,
        primary_regressions_df,
        run_id,
        baseline_label,
    )
    report_path = benchmark_root / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"[OK] grid summary: {grid_summary_path}")
    print(f"[OK] suite summary: {suite_summary_path}")
    print(f"[OK] delta summary: {delta_path}")
    print(f"[OK] report: {report_path}")


if __name__ == "__main__":
    main()
