#!/usr/bin/env python3
"""Run Gemini matrix review on detection review chips.

Input comes from ``build_gemini_detection_review_chips.py``.  Each Gemini call
scores one chip image with T01/T02/... targets, and this script writes one
flattened JSONL record per target.  The JSONL is compatible with
``build_gemini_review_training_pool.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SOLAR_BACKDATING_ROOT = Path(
    os.environ.get("SOLAR_BACKDATING_ROOT", "/home/gaosh/projects/solar_backdating")
)
if str(SOLAR_BACKDATING_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_BACKDATING_ROOT))

from scripts.validation.gemini_solar_image_review import (  # noqa: E402
    API_FORMATS,
    DEFAULT_AGY_BIN,
    DEFAULT_MAX_MATRIX_TARGETS,
    HARD_MAX_MATRIX_CELLS,
    HARD_MAX_MATRIX_TARGETS,
    GeminiClientConfig,
    GeminiMatrixObservation,
    MatrixDatePick,
    MatrixTarget,
    RateLimiter,
    env_value,
    load_env_file,
    score_target_date_matrix,
)


DEFAULT_OUTPUT = Path("gemini_detection_review.jsonl")
DEFAULT_SUMMARY = Path("gemini_detection_review_summary.json")


@dataclass(frozen=True)
class DetectionTarget:
    candidate_id: str
    chip_id: str
    target_label: str
    target_index: int
    region_key: str
    grid_id: str
    pred_id: int
    predictions_path: str
    image_path: Path
    capture_date: str
    source_row: dict[str, str]

    @property
    def matrix_target(self) -> MatrixTarget:
        return MatrixTarget(target_id=self.candidate_id, target_label=self.target_label)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_targets(path: Path) -> dict[str, list[DetectionTarget]]:
    grouped: dict[str, list[DetectionTarget]] = {}
    for row in read_csv(path):
        chip_id = str(row.get("chip_id", "")).strip()
        candidate_id = str(row.get("candidate_id") or row.get("target_id") or "").strip()
        target_label = str(row.get("target_label", "")).strip()
        image_path = Path(str(row.get("image_path", "")).strip())
        if not chip_id or not candidate_id or not target_label or not image_path:
            continue
        target = DetectionTarget(
            candidate_id=candidate_id,
            chip_id=chip_id,
            target_label=target_label,
            target_index=int(float(row.get("target_index") or 0)),
            region_key=str(row.get("region_key", "")).strip(),
            grid_id=str(row.get("grid_id", "")).strip().upper(),
            pred_id=int(float(row.get("pred_id") or 0)),
            predictions_path=str(row.get("predictions_path", "")).strip(),
            image_path=image_path,
            capture_date=str(row.get("capture_date") or "2024-06-30").strip()[:10],
            source_row=dict(row),
        )
        grouped.setdefault(chip_id, []).append(target)
    return {
        chip_id: sorted(items, key=lambda t: (t.target_index, t.target_label, t.candidate_id))
        for chip_id, items in grouped.items()
    }


def _default_env_file() -> Path:
    local = SOLAR_BACKDATING_ROOT / ".env.gemini.local"
    if local.exists():
        return local
    return Path("/home/gaosh/projects/ZAsolar/.env.gemini.local")


def load_config(args: argparse.Namespace) -> GeminiClientConfig:
    env = load_env_file(args.env_file)
    base_url = args.base_url or env_value(env, "GOOGLE_GEMINI_BASE_URL")
    api_key = args.api_key or env_value(env, "GEMINI_API_KEY")
    model = args.model or env_value(env, "GEMINI_MODEL", "gemini-3-flash-agent")
    api_format = args.api_format or env_value(env, "GEMINI_API_FORMAT", "openai")
    native_path = args.native_path or env_value(env, "GEMINI_NATIVE_PATH", "/v1beta")
    agy_bin = args.agy_bin or env_value(env, "GEMINI_AGY_BIN", DEFAULT_AGY_BIN)
    if api_format not in API_FORMATS:
        raise SystemExit(f"Unsupported api format {api_format!r}; choose {sorted(API_FORMATS)}")
    # The agy backend drives the local Antigravity CLI, not the HTTP proxy.
    if api_format != "agy":
        if not base_url:
            raise SystemExit(f"Missing GOOGLE_GEMINI_BASE_URL in {args.env_file} or --base-url")
        if not api_key:
            raise SystemExit(f"Missing GEMINI_API_KEY in {args.env_file} or --api-key")
    return GeminiClientConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        api_format=api_format,
        native_path=native_path,
        timeout=args.timeout,
        agy_bin=agy_bin,
    )


def _chunked(items: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _audit_writer_for(audit_dir: Path | None, *, chip_id: str, target_chunk_index: int) -> Callable[[dict[str, Any]], None] | None:
    if audit_dir is None:
        return None
    path = audit_dir / chip_id / f"targets{target_chunk_index:03d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write(payload: dict[str, Any]) -> None:
        record = dict(payload)
        record["chip_id"] = chip_id
        record["target_chunk_index"] = target_chunk_index
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return _write


def flatten_observation(
    *,
    target: DetectionTarget,
    obs: GeminiMatrixObservation,
    model: str,
    raw_image_path: str,
) -> dict[str, Any]:
    return {
        "candidate_id": target.candidate_id,
        "target_id": target.candidate_id,
        "chip_id": target.chip_id,
        "target_label": target.target_label,
        "grid_id": target.grid_id,
        "pred_id": target.pred_id,
        "region_key": target.region_key,
        "region": target.region_key,
        "predictions_path": target.predictions_path,
        "image_path": str(target.image_path),
        "raw_image_path": raw_image_path,
        "capture_date": obs.capture_date,
        "pv_present": obs.pv_present,
        "confidence": obs.confidence,
        "quality_flag": obs.quality_flag,
        "evidence": obs.evidence,
        "notes": obs.notes,
        "decision_source": obs.decision_source,
        "model": model,
        "gemini_error": obs.error or "",
    }


@dataclass(frozen=True)
class _CallTask:
    """One Gemini matrix call = one (chip, target-chunk). The unit of parallelism."""

    chip_id: str
    target_chunk_index: int
    image_path: Path
    targets: tuple[DetectionTarget, ...]


def score(
    *,
    targets_by_chip: Mapping[str, Sequence[DetectionTarget]],
    config: GeminiClientConfig,
    output: Path,
    audit_dir: Path | None,
    max_targets: int,
    hard_max_targets: int,
    hard_max_cells: int,
    limit_chips: int | None,
    dry_run: bool,
    workers: int = 1,
    qps: float = 0.0,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    chip_ids = sorted(targets_by_chip)
    if limit_chips is not None:
        chip_ids = chip_ids[:limit_chips]

    # Build the full call list up front: validates chip images + chunk sizes
    # before any API call, and gives one flat work queue for the thread pool.
    tasks: list[_CallTask] = []
    total_targets = 0
    for chip_id in chip_ids:
        targets = list(targets_by_chip[chip_id])
        if not targets:
            continue
        image_path = targets[0].image_path
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        total_targets += len(targets)
        for target_chunk_index, target_chunk in enumerate(_chunked(targets, max_targets), start=1):
            if len(target_chunk) > hard_max_targets:
                raise ValueError(
                    f"{chip_id}: target chunk has {len(target_chunk)} targets, "
                    f"exceeding hard_max_targets={hard_max_targets}"
                )
            tasks.append(
                _CallTask(
                    chip_id=chip_id,
                    target_chunk_index=target_chunk_index,
                    image_path=image_path,
                    targets=tuple(target_chunk),
                )
            )

    calls = len(tasks)
    rows_written = 0
    limiter = RateLimiter(qps)

    def run_task(task: _CallTask) -> list[dict[str, Any]]:
        limiter.wait()
        date_pick = MatrixDatePick(
            date_index=1,
            chip_path=task.image_path,
            capture_date=task.targets[0].capture_date,
        )
        matrix_targets = [t.matrix_target for t in task.targets]
        observations = score_target_date_matrix(
            [date_pick],
            matrix_targets,
            config=config,
            audit_writer=_audit_writer_for(
                audit_dir,
                chip_id=task.chip_id,
                target_chunk_index=task.target_chunk_index,
            ),
            max_dates=1,
            max_targets=max_targets,
            hard_max_targets=hard_max_targets,
            hard_max_cells=hard_max_cells,
        )
        by_label = {obs.target_label: obs for obs in observations}
        raw_image_path = str(task.targets[0].source_row.get("raw_image_path", ""))
        rows: list[dict[str, Any]] = []
        for target in task.targets:
            obs = by_label.get(target.target_label)
            if obs is None:
                continue
            rows.append(
                flatten_observation(
                    target=target,
                    obs=obs,
                    model=config.model,
                    raw_image_path=raw_image_path,
                )
            )
        return rows

    with output.open("w", encoding="utf-8") as fh:
        write_lock = threading.Lock()

        def write_rows(rows: list[dict[str, Any]]) -> int:
            with write_lock:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
            return len(rows)

        if dry_run:
            pass  # chip images + chunk sizes validated above; no API calls
        elif workers <= 1:
            for task in tasks:
                rows_written += write_rows(run_task(task))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_task, task) for task in tasks]
                for future in as_completed(futures):
                    rows_written += write_rows(future.result())

    return {
        "n_chips": len(chip_ids),
        "n_targets": total_targets,
        "gemini_calls": calls,
        "rows_written": rows_written,
        "workers": workers,
        "qps": qps,
        "dry_run": dry_run,
        "output": str(output),
        "call_reduction_factor": round(total_targets / calls, 3) if calls else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip-targets-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--env-file", type=Path, default=_default_env_file())
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model", default="gemini-3-flash-agent")
    parser.add_argument("--api-format", choices=sorted(API_FORMATS))
    parser.add_argument("--native-path")
    parser.add_argument("--agy-bin", help="Antigravity CLI binary for --api-format agy (default GEMINI_AGY_BIN or 'agy').")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-targets", type=int, default=DEFAULT_MAX_MATRIX_TARGETS)
    parser.add_argument("--hard-max-targets", type=int, default=HARD_MAX_MATRIX_TARGETS)
    parser.add_argument("--hard-max-cells", type=int, default=HARD_MAX_MATRIX_CELLS)
    parser.add_argument("--limit-chips", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent Gemini calls (one chip-chunk per call). Keep <= backend "
        "capacity (accounts * per-account slots). Default 1 = serial.",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=0.0,
        help="Optional global requests/sec cap across all workers. 0 = disabled "
        "(worker count alone caps concurrency).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets_by_chip = load_targets(args.chip_targets_csv)
    config = load_config(args)
    summary = score(
        targets_by_chip=targets_by_chip,
        config=config,
        output=args.output,
        audit_dir=args.audit_dir,
        max_targets=args.max_targets,
        hard_max_targets=args.hard_max_targets,
        hard_max_cells=args.hard_max_cells,
        limit_chips=args.limit_chips,
        dry_run=args.dry_run,
        workers=args.workers,
        qps=args.qps,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
