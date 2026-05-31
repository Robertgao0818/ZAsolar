#!/usr/bin/env python3
"""Single-image FP-discrimination Gemini review (the "B" harness).

Unlike ``score_gemini_detection_review_chips.py`` (which reuses the temporal
date x target *matrix* machinery and, mis-applied to single-image review, both
abstains ~50-60% of the time and asks a generic "is PV present" question), this
scorer is purpose-built for the FP-suppression objective:

  - ONE chip -> ONE Gemini call, judging the single marked target.
  - A TP-protective prompt: keep anything plausibly PV; only call ``not_pv``
    when confidently a known look-alike (solar water heater, pool mat, skylight,
    HVAC, tank, painted roof, shadow, vehicle).  This matches the production
    goal (preserve TP recall, cut FP) rather than RA-behaviour agreement.
  - A clean JSON contract (native ``response_schema``) + lenient fallback parse,
    so parse failures are rare -> abstain rate collapses.
  - Per-chip error isolation: a failed call records an error row and continues;
    it never crashes the batch.

Output JSONL is schema-compatible with ``eval_gemini_review_vs_ra.py`` and
``build_gemini_review_training_pool.py`` (top-level ``pv_present`` /
``confidence`` / ``quality_flag``).
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
from typing import Any

SOLAR_BACKDATING_ROOT = Path(
    os.environ.get("SOLAR_BACKDATING_ROOT", "/home/gaosh/projects/solar_backdating")
)
if str(SOLAR_BACKDATING_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_BACKDATING_ROOT))

from scripts.validation.gemini_solar_image_review import (  # noqa: E402
    API_FORMATS,
    DEFAULT_AGY_BIN,
    GeminiClientConfig,
    RateLimiter,
    _call_gemini,
    env_value,
    extract_json_object,
    load_env_file,
)

FP_PROMPT = """You are auditing ONE rooftop object in a high-resolution aerial/satellite image chip to decide whether it is a real photovoltaic (PV) solar panel installation or a false-positive look-alike.

The single target is marked with a colored polygon outline and a cross/ring at its center. Judge ONLY that marked object, not other roofs in the chip.

Choose exactly one label:
- "pv": the marked object is, or is plausibly, a photovoltaic solar panel or panel array (flat rectangular dark/blue modules, often in rows on a roof).
- "not_pv": you are confident the marked object is NOT a PV panel. Common look-alikes: solar water heater / geyser collector (a flat collector plus a horizontal cylindrical tank), pool-heating mats, skylights / glass roofing, HVAC units, water tanks, painted or dark roof patches, shadows, vehicles.

Decision rules (IMPORTANT):
- Bias toward "pv" under uncertainty. We must NOT discard real installations, so only output "not_pv" when the object is clearly a non-PV look-alike.
- A solar water heater (collector + cylindrical tank) is "not_pv", even though it is solar.
- If the chip is too blurry/cropped to judge the marked object at all, still pick the more likely label but set confidence low.

Return ONLY a JSON object (no prose, no markdown fence):
{"label": "pv" | "not_pv", "confidence": 0.0-1.0, "lookalike_type": "water_heater|pool|skylight|hvac|tank|roof|shadow|vehicle|none", "reason": "<one short clause>"}
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["pv", "not_pv"]},
        "confidence": {"type": "number"},
        "lookalike_type": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["label", "confidence"],
}


@dataclass(frozen=True)
class Chip:
    candidate_id: str
    chip_id: str
    grid_id: str
    pred_id: int
    region_key: str
    predictions_path: str
    image_path: Path
    raw_image_path: str


def load_chips(path: Path) -> list[Chip]:
    out: list[Chip] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            img = Path(str(row.get("image_path", "")).strip())
            cid = str(row.get("candidate_id") or row.get("target_id") or "").strip()
            if not cid or not img:
                continue
            out.append(
                Chip(
                    candidate_id=cid,
                    chip_id=str(row.get("chip_id", "")).strip(),
                    grid_id=str(row.get("grid_id", "")).strip().upper(),
                    pred_id=int(float(row.get("pred_id") or 0)),
                    region_key=str(row.get("region_key") or row.get("region") or "").strip(),
                    predictions_path=str(row.get("predictions_path", "")).strip(),
                    image_path=img,
                    raw_image_path=str(row.get("raw_image_path", "")).strip(),
                )
            )
    # one target per chip is expected; if a chip_id repeats keep them all (each judged on its own marker)
    return out


def load_config(args: argparse.Namespace) -> GeminiClientConfig:
    env = load_env_file(args.env_file)
    api_format = args.api_format or env_value(env, "GEMINI_API_FORMAT", "native")
    if api_format not in API_FORMATS:
        raise SystemExit(f"Unsupported api format {api_format!r}; choose {sorted(API_FORMATS)}")
    base_url = args.base_url or env_value(env, "GOOGLE_GEMINI_BASE_URL")
    api_key = args.api_key or env_value(env, "GEMINI_API_KEY")
    model = args.model or env_value(env, "GEMINI_MODEL", "gemini-3-flash-agent")
    native_path = args.native_path or env_value(env, "GEMINI_NATIVE_PATH", "/v1beta")
    agy_bin = args.agy_bin or env_value(env, "GEMINI_AGY_BIN", DEFAULT_AGY_BIN)
    if api_format != "agy":
        if not base_url:
            raise SystemExit("Missing GOOGLE_GEMINI_BASE_URL")
        if not api_key:
            raise SystemExit("Missing GEMINI_API_KEY")
    return GeminiClientConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        api_format=api_format,
        native_path=native_path,
        timeout=args.timeout,
        agy_bin=agy_bin,
    )


def judge_chip(chip: Chip, *, prompt: str, config: GeminiClientConfig, max_tokens: int, retries: int) -> dict[str, Any]:
    last_err = ""
    for attempt in range(retries + 1):
        try:
            text, _raw = _call_gemini(
                image_paths=[chip.image_path],
                prompt=prompt,
                config=config,
                max_tokens=max_tokens,
                response_mime_type="application/json" if config.api_format == "native" else None,
                response_schema=RESPONSE_SCHEMA if config.api_format == "native" else None,
            )
            parsed = extract_json_object(text)
            if not isinstance(parsed, dict):
                last_err = f"non-dict json: {str(parsed)[:80]}"
                continue
            label = str(parsed.get("label", "")).strip().lower()
            if label not in ("pv", "not_pv"):
                last_err = f"bad label: {label!r}"
                continue
            conf = parsed.get("confidence")
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = None
            return {
                "pv_present": label == "pv",
                "confidence": conf,
                "quality_flag": "usable",
                "label": label,
                "lookalike_type": str(parsed.get("lookalike_type", "")).strip(),
                "reason": str(parsed.get("reason", "")).strip(),
                "gemini_error": "",
            }
        except Exception as exc:  # network / parse / transport
            last_err = f"{type(exc).__name__}: {exc}"
    return {
        "pv_present": None,
        "confidence": None,
        "quality_flag": "unusable",
        "label": "",
        "lookalike_type": "",
        "reason": "",
        "gemini_error": last_err,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chip-targets-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--env-file", type=Path, default=SOLAR_BACKDATING_ROOT / ".env.gemini.local")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--model")
    ap.add_argument("--api-format", choices=sorted(API_FORMATS))
    ap.add_argument("--native-path")
    ap.add_argument("--agy-bin")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--prompt-file", type=Path, help="Override the FP prompt.")
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent Gemini calls (one chip per call). Keep <= backend capacity "
        "(accounts * per-account slots). Default 1 = serial.",
    )
    ap.add_argument(
        "--qps",
        type=float,
        default=0.0,
        help="Optional global requests/sec cap across all workers. 0 = disabled "
        "(worker count alone caps concurrency).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else FP_PROMPT
    chips = load_chips(args.chip_targets_csv)
    if args.limit is not None:
        chips = chips[: args.limit]
    config = load_config(args)

    # Fail fast on missing chips before issuing any API call.
    for chip in chips:
        if not chip.image_path.exists():
            raise FileNotFoundError(chip.image_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_usable = n_abstain = n_pv = n_notpv = 0
    limiter = RateLimiter(args.qps)

    def process(chip: Chip) -> tuple[Chip, dict[str, Any]]:
        limiter.wait()
        res = judge_chip(
            chip, prompt=prompt, config=config, max_tokens=args.max_tokens, retries=args.retries
        )
        return chip, res

    with args.output.open("w", encoding="utf-8") as fh:
        write_lock = threading.Lock()

        def record_result(chip: Chip, res: dict[str, Any]) -> None:
            nonlocal n_usable, n_abstain, n_pv, n_notpv
            if res["quality_flag"] == "usable":
                n_usable += 1
                if res["pv_present"]:
                    n_pv += 1
                else:
                    n_notpv += 1
            else:
                n_abstain += 1
            record = {
                "candidate_id": chip.candidate_id,
                "target_id": chip.candidate_id,
                "chip_id": chip.chip_id,
                "grid_id": chip.grid_id,
                "pred_id": chip.pred_id,
                "region_key": chip.region_key,
                "region": chip.region_key,
                "predictions_path": chip.predictions_path,
                "image_path": str(chip.image_path),
                "raw_image_path": chip.raw_image_path,
                "model": config.model,
                "decision_source": "gemini_fp_review",
                **res,
            }
            with write_lock:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

        if args.dry_run:
            pass  # existence already validated; no API calls in dry-run
        elif args.workers <= 1:
            for chip in chips:
                chip, res = process(chip)
                record_result(chip, res)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process, chip) for chip in chips]
                for future in as_completed(futures):
                    chip, res = future.result()
                    record_result(chip, res)

    summary = {
        "n_chips": len(chips),
        "workers": args.workers,
        "qps": args.qps,
        "n_usable": n_usable,
        "n_abstain": n_abstain,
        "abstain_rate": round(n_abstain / max(1, len(chips)), 4),
        "gemini_pv": n_pv,
        "gemini_not_pv": n_notpv,
        "output": str(args.output),
        "model": config.model,
        "dry_run": args.dry_run,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
