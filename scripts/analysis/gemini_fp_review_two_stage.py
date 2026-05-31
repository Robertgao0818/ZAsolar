#!/usr/bin/env python3
"""Two-stage Gemini FP review with a skylight-specific safety pass.

Stage 1 is the existing multi-scale Gemini FP review output. This script routes
only first-pass ``not_pv`` + ``lookalike_type=skylight`` candidates through a
second, TP-protective prompt, then writes a merged JSONL with explicit
``production_action`` / ``auto_drop`` fields while preserving stage-1 fields
under ``stage1_*`` names.

The merged top-level ``pv_present`` remains compatible with
``eval_gemini_review_vs_ra.py`` for automatically decided keep/drop rows.
Rows that need human review use ``pv_present=null`` and ``auto_drop=false``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SKYLIGHT_STAGE2_PROMPT = """You are doing a SECOND-PASS safety check for rooftop-solar inventory.

A first-pass model suspected that the marked target may be a skylight or glass-roof look-alike. Your job is not just to name the object; your job is to decide whether it is safe to remove this detector prediction from a PV inventory.

You are given TWO images of the SAME target:
- IMAGE 1 is a tight crop for texture.
- IMAGE 2 is a wider crop for same-roof context.
Both images mark the SAME detector prediction with a polygon and center cross/ring.

Choose exactly one label:
- "pv": KEEP the detector prediction. Use this when the marked object itself may be PV, OR the marker/polygon overlaps part of a real PV installation, OR clear PV modules are immediately adjacent on the same roof plane / same installation even if the marker appears to land on a skylight-like element.
- "not_pv": DROP the detector prediction. Use this only when the marked target is clearly a non-PV skylight, glass canopy, roof monitor, roof window, or other roof feature, AND no same-roof PV installation is visible at or immediately adjacent to the marked target.

Decision rules:
- Bias toward "pv" under uncertainty. A false deletion is worse than a false keep.
- For "not_pv", require positive non-PV evidence such as translucent glass, thick white mullions, raised skylight curbs, roof-ridge glazing, repeating roof-monitor bays, or a glass canopy pattern.
- Do not call "not_pv" merely because the target has pale grid lines. Real PV can also show regular seams between modules.
- If real dark/blue opaque PV modules are visible on the same roof segment immediately beside the marked skylight-like object, choose "pv" unless the marked prediction is clearly unrelated to that PV installation.
- Judge the marked target and its immediate same-roof context; ignore PV on unrelated neighboring roofs.

Return ONLY a JSON object, no prose and no markdown fence:
{"label": "pv" | "not_pv", "confidence": 0.0-1.0, "lookalike_type": "water_heater|pool|skylight|hvac|tank|roof|shadow|vehicle|none", "reason": "<one short clause>"}
"""


STAGE_FIELDS = (
    "pv_present",
    "confidence",
    "quality_flag",
    "label",
    "lookalike_type",
    "reason",
    "gemini_error",
    "model",
    "decision_source",
    "latency_ms",
    "retry_count",
    "error_type",
)


def load_multiscale_helpers() -> Any:
    """Load the multi-scale scorer without importing it as ``scripts.*``.

    ``gemini_fp_review_multiscale.py`` imports ``scripts.validation`` from the
    sibling ``solar_backdating`` repo. Importing it as
    ``scripts.analysis.gemini_fp_review_multiscale`` from this repo binds the
    top-level ``scripts`` package to ZAsolar and blocks that sibling import.
    Loading by file path keeps the existing scorer reusable without changing
    its import contract.
    """

    module_path = PROJECT_ROOT / "scripts" / "analysis" / "gemini_fp_review_multiscale.py"
    spec = importlib.util.spec_from_file_location("_zasolar_gemini_fp_review_multiscale", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(obj)
    return records


def candidate_id(record: dict[str, Any]) -> str:
    cid = str(record.get("candidate_id") or record.get("target_id") or "").strip()
    if cid:
        return cid
    grid = str(record.get("grid_id") or "").strip().upper()
    pred_id = record.get("pred_id")
    if grid and pred_id not in (None, ""):
        return f"{grid}_pred{int(pred_id):06d}"
    return ""


def index_by_candidate(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        cid = candidate_id(record)
        if cid:
            out[cid] = record
    return out


def as_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text in {"true", "1", "yes", "present", "pv"}:
        return True
    if text in {"false", "0", "no", "absent", "not_pv", "non_pv"}:
        return False
    return None


def is_usable_decision(record: dict[str, Any]) -> bool:
    quality = str(record.get("quality_flag") or "").strip().lower()
    if quality and quality != "usable":
        return False
    return as_bool_or_none(record.get("pv_present")) is not None


def is_skylight_drop(record: dict[str, Any]) -> bool:
    pv = as_bool_or_none(record.get("pv_present"))
    label = str(record.get("label") or "").strip().lower()
    lookalike = str(record.get("lookalike_type") or "").strip().lower()
    return (pv is False or label == "not_pv") and lookalike == "skylight"


def needs_stage2(record: dict[str, Any]) -> bool:
    return is_usable_decision(record) and is_skylight_drop(record)


def _copy_stage_fields(out: dict[str, Any], prefix: str, record: dict[str, Any] | None) -> None:
    for key in STAGE_FIELDS:
        out[f"{prefix}_{key}"] = None if record is None else record.get(key)


def merge_decision(
    stage1: dict[str, Any],
    stage2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a production-compatible merged decision record.

    ``auto_drop`` is the authoritative deletion flag. Review/abstain rows are
    conservative keeps for automation, but retain ``pv_present=null`` so eval
    scripts count them as abstentions.
    """

    out = dict(stage1)
    _copy_stage_fields(out, "stage1", stage1)
    _copy_stage_fields(out, "stage2", stage2)

    out["decision_source"] = "gemini_fp_review_two_stage"
    out["stage2_required"] = needs_stage2(stage1)
    out["stage2_applied"] = False
    out["requires_human_review"] = False

    stage1_pv = as_bool_or_none(stage1.get("pv_present"))
    if not is_usable_decision(stage1):
        out.update(
            {
                "production_action": "review",
                "production_decision_source": "stage1_abstain",
                "auto_drop": False,
                "pv_present": None,
                "label": "",
                "quality_flag": "ambiguous",
                "confidence": None,
                "gemini_error": stage1.get("gemini_error", ""),
                "reason": "stage1 did not return a usable decision",
            }
        )
        out["requires_human_review"] = True
        return out

    if stage1_pv is True:
        out.update(
            {
                "production_action": "keep",
                "production_decision_source": "stage1_pv",
                "auto_drop": False,
                "pv_present": True,
                "label": "pv",
                "quality_flag": "usable",
            }
        )
        return out

    if not is_skylight_drop(stage1):
        out.update(
            {
                "production_action": "drop",
                "production_decision_source": "stage1_not_pv_non_skylight",
                "auto_drop": True,
                "pv_present": False,
                "label": "not_pv",
                "quality_flag": "usable",
            }
        )
        return out

    if stage2 is not None and is_usable_decision(stage2):
        out["stage2_applied"] = True
        stage2_pv = as_bool_or_none(stage2.get("pv_present"))
        if stage2_pv is True:
            out.update(
                {
                    "production_action": "keep",
                    "production_decision_source": "stage2_skylight_keep",
                    "auto_drop": False,
                    "pv_present": True,
                    "label": "pv",
                    "quality_flag": "usable",
                    "confidence": stage2.get("confidence"),
                    "lookalike_type": stage2.get("lookalike_type", "none"),
                    "reason": stage2.get("reason", ""),
                    "gemini_error": stage2.get("gemini_error", ""),
                }
            )
            return out

        out.update(
            {
                "production_action": "drop",
                "production_decision_source": "stage2_skylight_drop",
                "auto_drop": True,
                "pv_present": False,
                "label": "not_pv",
                "quality_flag": "usable",
                "confidence": stage2.get("confidence"),
                "lookalike_type": stage2.get("lookalike_type", "skylight"),
                "reason": stage2.get("reason", ""),
                "gemini_error": stage2.get("gemini_error", ""),
            }
        )
        return out

    out.update(
        {
            "production_action": "review",
            "production_decision_source": "stage2_missing_or_abstain",
            "auto_drop": False,
            "pv_present": None,
            "label": "",
            "quality_flag": "ambiguous",
            "confidence": None,
            "gemini_error": "" if stage2 is None else stage2.get("gemini_error", ""),
            "reason": "skylight candidate needs stage2 review before auto-drop",
        }
    )
    out["requires_human_review"] = True
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = Counter(str(r.get("production_action") or "") for r in records)
    sources = Counter(str(r.get("production_decision_source") or "") for r in records)
    stage2_required = sum(1 for r in records if r.get("stage2_required"))
    stage2_applied = sum(1 for r in records if r.get("stage2_applied"))
    review = sum(1 for r in records if r.get("requires_human_review"))
    auto_drop = sum(1 for r in records if r.get("auto_drop") is True)
    return {
        "n_records": len(records),
        "production_actions": dict(sorted(actions.items())),
        "production_decision_sources": dict(sorted(sources.items())),
        "stage2_required": stage2_required,
        "stage2_applied": stage2_applied,
        "requires_human_review": review,
        "auto_drop": auto_drop,
        "auto_keep_or_review": len(records) - auto_drop,
    }


def run_stage2(
    pairs: list[Any],
    *,
    helpers: Any,
    prompt: str,
    config: Any,
    max_tokens: int,
    retries: int,
    routing_salt_mode: str,
    retry_base_delay: float,
    retry_max_delay: float,
    workers: int,
    qps: float,
    worker_jitter: float,
    output_path: Path,
) -> list[dict[str, Any]]:
    limiter = helpers.RateLimiter(qps)
    jitter = worker_jitter if workers > 1 else 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    write_lock = threading.Lock()

    def process(pair: Pair) -> tuple[Pair, dict[str, Any]]:
        if jitter > 0:
            time.sleep(random.uniform(0.6 * jitter, jitter))
        limiter.wait()
        res = helpers.judge_pair(
            pair,
            prompt=prompt,
            config=config,
            max_tokens=max_tokens,
            retries=retries,
            routing_salt_mode=routing_salt_mode,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
        )
        return pair, res

    def record_result(fh: Any, pair: Pair, res: dict[str, Any]) -> None:
        row = {
            "candidate_id": pair.candidate_id,
            "target_id": pair.candidate_id,
            "chip_id": pair.chip_id,
            "grid_id": pair.grid_id,
            "pred_id": pair.pred_id,
            "region_key": pair.region_key,
            "region": pair.region_key,
            "predictions_path": pair.predictions_path,
            "image_path": str(pair.tight_image),
            "wide_image_path": str(pair.wide_image),
            "raw_image_path": pair.raw_image_path,
            "model": config.model,
            "decision_source": "gemini_fp_review_skylight_stage2",
            **res,
        }
        with rows_lock:
            rows.append(row)
        with write_lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as fh:
        if workers <= 1:
            for pair in pairs:
                pair, res = process(pair)
                record_result(fh, pair, res)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process, pair) for pair in pairs]
                for future in as_completed(futures):
                    pair, res = future.result()
                    record_result(fh, pair, res)
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-jsonl", type=Path, required=True)
    ap.add_argument("--tight-chips-csv", type=Path, required=True)
    ap.add_argument("--wide-chips-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Merged production JSONL.")
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--stage2-jsonl", type=Path, help="Raw stage2 JSONL. Defaults beside --output.")
    ap.add_argument("--reuse-stage2-jsonl", action="store_true", help="Reuse existing stage2 rows and only run missing ones.")
    ap.add_argument("--force-stage2", action="store_true", help="Overwrite --stage2-jsonl before running stage2.")
    ap.add_argument("--prompt-file", type=Path, help="Override the built-in skylight stage2 prompt.")
    ap.add_argument("--limit-stage2", type=int, help="Only run the first N missing stage2 rows; useful for smoke tests.")
    ap.add_argument("--dry-run", action="store_true", help="Do not call Gemini or write outputs; print the planned summary.")

    # Gemini client options mirror gemini_fp_review_multiscale.py.
    ap.add_argument("--env-file", type=Path, default=Path("/home/gaosh/projects/solar_backdating/.env.gemini.local"))
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--model", default="gemini-3-flash-agent")
    ap.add_argument("--api-format", choices=("agy", "native", "openai"))
    ap.add_argument("--native-path")
    ap.add_argument("--agy-bin")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--retry-base-delay", type=float, default=1.0)
    ap.add_argument("--retry-max-delay", type=float, default=30.0)
    ap.add_argument("--routing-salt-mode", choices=("auto", "none", "target"), default="auto")
    ap.add_argument("--thinking-level", choices=("minimal", "low", "medium", "high"))
    ap.add_argument("--thinking-budget", type=int)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--qps", type=float, default=0.0)
    ap.add_argument(
        "--worker-jitter",
        type=float,
        default=0.25,
        help="Max startup jitter (s) before each parallel stage-2 worker fires a "
        "request; sleeps a random 0.6x-1.0x of it so concurrent workers don't hit the "
        "gateway in the same instant (skews account routing). Default 0.25 = 150-250 ms "
        "band; 0 disables. Ignored when --workers <= 1.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stage1_records = load_jsonl(args.stage1_jsonl)
    stage2_path = args.stage2_jsonl or args.output.with_name(f"{args.output.stem}_stage2_skylight.jsonl")

    required_ids = {candidate_id(r) for r in stage1_records if needs_stage2(r)}
    required_ids.discard("")

    existing_stage2: dict[str, dict[str, Any]] = {}
    if stage2_path.exists() and args.reuse_stage2_jsonl and not args.force_stage2:
        existing_stage2 = index_by_candidate(load_jsonl(stage2_path))
    elif stage2_path.exists() and not args.force_stage2:
        raise SystemExit(f"{stage2_path} exists; pass --reuse-stage2-jsonl or --force-stage2")
    elif stage2_path.exists() and args.force_stage2:
        stage2_path.unlink()

    missing_ids = sorted(required_ids - set(existing_stage2))
    if args.limit_stage2 is not None:
        missing_ids = missing_ids[: args.limit_stage2]

    if args.dry_run:
        planned = {
            "stage1_records": len(stage1_records),
            "stage2_required": len(required_ids),
            "stage2_existing_reused": len(existing_stage2),
            "stage2_missing_planned": len(missing_ids),
            "stage2_jsonl": str(stage2_path),
            "output": str(args.output),
        }
        print(json.dumps(planned, indent=2))
        return

    stage2_records: dict[str, dict[str, Any]] = dict(existing_stage2)
    if missing_ids:
        prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else SKYLIGHT_STAGE2_PROMPT
        helpers = load_multiscale_helpers()
        config = helpers.load_config(args)
        pairs = [p for p in helpers.load_pairs(args.tight_chips_csv, args.wide_chips_csv) if p.candidate_id in set(missing_ids)]
        found_ids = {p.candidate_id for p in pairs}
        missing_pairs = sorted(set(missing_ids) - found_ids)
        if missing_pairs:
            print(f"[WARN] {len(missing_pairs)} stage2 candidate(s) missing from paired chip CSVs.")
        new_rows = run_stage2(
            pairs,
            helpers=helpers,
            prompt=prompt,
            config=config,
            max_tokens=args.max_tokens,
            retries=args.retries,
            routing_salt_mode=args.routing_salt_mode,
            retry_base_delay=args.retry_base_delay,
            retry_max_delay=args.retry_max_delay,
            workers=args.workers,
            qps=args.qps,
            worker_jitter=args.worker_jitter,
            output_path=stage2_path,
        )
        stage2_records.update(index_by_candidate(new_rows))

    merged = [merge_decision(record, stage2_records.get(candidate_id(record))) for record in stage1_records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in merged:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(merged)
    summary.update(
        {
            "stage1_jsonl": str(args.stage1_jsonl),
            "stage2_jsonl": str(stage2_path),
            "output": str(args.output),
            "stage2_existing_reused": len(existing_stage2),
            "stage2_missing_planned": len(missing_ids),
            "model": args.model,
        }
    )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
