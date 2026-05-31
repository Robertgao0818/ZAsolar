#!/usr/bin/env python3
"""Multi-scale FP-discrimination Gemini review (the "B-multiscale" harness).

Single-crop review forces a resolution<->context tradeoff (calibrated 2026-05-31
on JHB Vexcel conf>=0.95): a TIGHT 20 m crop makes the panel-module grid legible
so TP recall jumps (0.851 -> 0.957) but loses the surrounding roof layout, so
look-alikes that are only identifiable in context (solar water heater = collector
+ horizontal tank, pool mats, gridded skylights spanning a roof) leak through
(FP-cut 0.915 -> 0.809). A WIDE 48 m crop is the mirror image.

This scorer breaks the tradeoff by sending BOTH crops of the SAME target in one
call: image 1 = tight zoom (judge module texture), image 2 = wide context (rule
out look-alikes). Everything else mirrors ``gemini_fp_review.py`` (TP-protective
prompt, native ``response_schema``, per-chip error isolation, threaded workers
with optional global QPS cap) and the output JSONL stays schema-compatible with
``eval_gemini_review_vs_ra.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
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

MULTISCALE_PROMPT = """You are auditing ONE rooftop object to decide whether it is a real photovoltaic (PV) solar panel installation or a false-positive look-alike.

You are given TWO images of the SAME object, both marking the SAME target with a colored polygon outline and a cross/ring at its center:
- IMAGE 1 is a TIGHT zoom on the target. Use it to read fine texture: real PV is a grid of flat rectangular dark/blue modules with thin regular seams between cells.
- IMAGE 2 is a WIDER view of the same target with surrounding roof context. Use it to spot look-alikes by their layout: a solar water heater / geyser is a flat collector PLUS a horizontal cylindrical tank; pool-heating mats sit beside a pool; skylights/glazing span a roof ridge in a repeating bay pattern; HVAC units, water tanks, vehicles sit in characteristic places.

Judge ONLY the marked target (the SAME object in both images), not other roofs.

Choose exactly one label:
- "pv": the marked object is, or is plausibly, a photovoltaic solar panel or array.
- "not_pv": you are confident it is NOT PV. Common look-alikes: solar water heater / geyser (collector + cylindrical tank), pool mats, skylights / glass roofing, HVAC, water tanks, painted/dark roof patches, shadows, vehicles.

Decision rules (IMPORTANT):
- Bias toward "pv" under uncertainty. We must NOT discard real installations, so only output "not_pv" when the object is clearly a non-PV look-alike.
- Use BOTH images: confirm module texture in image 1 AND check the context in image 2. A solar water heater (collector + cylindrical tank) is "not_pv" even though it is solar — the tank in image 2 is the giveaway.
- If both images are too blurry/cropped to judge, still pick the more likely label but set confidence low.

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
class Pair:
    candidate_id: str
    chip_id: str
    grid_id: str
    pred_id: int
    region_key: str
    predictions_path: str
    tight_image: Path
    wide_image: Path
    raw_image_path: str


def _index(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = str(row.get("candidate_id") or row.get("target_id") or "").strip()
            if cid:
                out[cid] = row
    return out


def load_pairs(tight_csv: Path, wide_csv: Path) -> list[Pair]:
    tight = _index(tight_csv)
    wide = _index(wide_csv)
    common = [c for c in tight if c in wide]
    missing = sorted(set(tight) ^ set(wide))
    if missing:
        print(f"[WARN] {len(missing)} candidate(s) not in both crops; using {len(common)} paired.")
    pairs: list[Pair] = []
    for cid in common:
        t = tight[cid]
        w = wide[cid]
        ti = Path(str(t.get("image_path", "")).strip())
        wi = Path(str(w.get("image_path", "")).strip())
        if not ti or not wi:
            continue
        pairs.append(
            Pair(
                candidate_id=cid,
                chip_id=str(t.get("chip_id", "")).strip(),
                grid_id=str(t.get("grid_id", "")).strip().upper(),
                pred_id=int(float(t.get("pred_id") or 0)),
                region_key=str(t.get("region_key") or t.get("region") or "").strip(),
                predictions_path=str(t.get("predictions_path", "")).strip(),
                tight_image=ti,
                wide_image=wi,
                raw_image_path=str(t.get("raw_image_path", "")).strip(),
            )
        )
    return pairs


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
        thinking_level=args.thinking_level or "",
        thinking_budget=args.thinking_budget,
    )


def _routing_salt(mode: str, config: GeminiClientConfig, pair: Pair) -> str | None:
    if config.api_format != "native":
        return None
    if mode == "none":
        return None
    if mode == "auto" and "pro" not in config.model.lower():
        return None
    return f"{config.model}:{pair.candidate_id}:{pair.grid_id}:{pair.pred_id}"


def _sleep_before_retry(attempt: int, retries: int, base_delay: float, max_delay: float) -> None:
    if attempt >= retries or base_delay <= 0:
        return
    delay = min(max_delay, base_delay * (2**attempt))
    if delay > 0:
        time.sleep(delay * (0.75 + random.random() * 0.5))


def judge_pair(
    pair: Pair,
    *,
    prompt: str,
    config: GeminiClientConfig,
    max_tokens: int,
    retries: int,
    routing_salt_mode: str,
    retry_base_delay: float,
    retry_max_delay: float,
) -> dict[str, Any]:
    last_err = ""
    last_exc_type = ""
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            text, _raw = _call_gemini(
                image_paths=[pair.tight_image, pair.wide_image],
                prompt=prompt,
                config=config,
                max_tokens=max_tokens,
                response_mime_type="application/json" if config.api_format == "native" else None,
                response_schema=RESPONSE_SCHEMA if config.api_format == "native" else None,
                routing_salt=_routing_salt(routing_salt_mode, config, pair),
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
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "retry_count": attempt,
                "error_type": "",
            }
        except Exception as exc:  # network / parse / transport
            last_err = f"{type(exc).__name__}: {exc}"
            last_exc_type = type(exc).__name__
            _sleep_before_retry(attempt, retries, retry_base_delay, retry_max_delay)
    return {
        "pv_present": None,
        "confidence": None,
        "quality_flag": "unusable",
        "label": "",
        "lookalike_type": "",
        "reason": "",
        "gemini_error": last_err,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "retry_count": attempt,
        "error_type": last_exc_type,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tight-chips-csv", type=Path, required=True, help="chip_targets.csv for the TIGHT crop (image 1).")
    ap.add_argument("--wide-chips-csv", type=Path, required=True, help="chip_targets.csv for the WIDE/context crop (image 2).")
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
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-base-delay", type=float, default=1.0)
    ap.add_argument("--retry-max-delay", type=float, default=30.0)
    ap.add_argument(
        "--routing-salt-mode",
        choices=("auto", "none", "target"),
        default="auto",
        help="For native sub2api calls, append a per-target routing nonce. "
        "auto salts pro models only; flash defaults stay unchanged.",
    )
    ap.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        help="Gemini 3 native thinking level. Do not combine with --thinking-budget.",
    )
    ap.add_argument(
        "--thinking-budget",
        type=int,
        help="Legacy Gemini thinkingBudget token value. Do not combine with --thinking-level.",
    )
    ap.add_argument("--prompt-file", type=Path, help="Override the multi-scale prompt.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--qps", type=float, default=0.0)
    ap.add_argument(
        "--worker-jitter",
        type=float,
        default=0.25,
        help="Max startup jitter (s) before each parallel worker fires a request; "
        "sleeps a random 0.6x-1.0x of it so concurrent workers don't hit the gateway "
        "in the same instant (skews account routing). Default 0.25 = 150-250 ms band; "
        "0 disables. Ignored when --workers <= 1.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else MULTISCALE_PROMPT
    pairs = load_pairs(args.tight_chips_csv, args.wide_chips_csv)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    config = load_config(args)

    for p in pairs:
        if not p.tight_image.exists():
            raise FileNotFoundError(p.tight_image)
        if not p.wide_image.exists():
            raise FileNotFoundError(p.wide_image)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_usable = n_abstain = n_pv = n_notpv = 0
    limiter = RateLimiter(args.qps)
    jitter = args.worker_jitter if args.workers > 1 else 0.0

    def process(pair: Pair) -> tuple[Pair, dict[str, Any]]:
        if jitter > 0:
            time.sleep(random.uniform(0.6 * jitter, jitter))
        limiter.wait()
        res = judge_pair(
            pair,
            prompt=prompt,
            config=config,
            max_tokens=args.max_tokens,
            retries=args.retries,
            routing_salt_mode=args.routing_salt_mode,
            retry_base_delay=args.retry_base_delay,
            retry_max_delay=args.retry_max_delay,
        )
        return pair, res

    with args.output.open("w", encoding="utf-8") as fh:
        write_lock = threading.Lock()

        def record_result(pair: Pair, res: dict[str, Any]) -> None:
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
                "decision_source": "gemini_fp_review_multiscale",
                **res,
            }
            with write_lock:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

        if args.dry_run:
            pass
        elif args.workers <= 1:
            for pair in pairs:
                pair, res = process(pair)
                record_result(pair, res)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process, pair) for pair in pairs]
                for future in as_completed(futures):
                    pair, res = future.result()
                    record_result(pair, res)

    summary = {
        "n_pairs": len(pairs),
        "workers": args.workers,
        "qps": args.qps,
        "worker_jitter": args.worker_jitter,
        "routing_salt_mode": args.routing_salt_mode,
        "thinking_level": config.thinking_level,
        "thinking_budget": config.thinking_budget,
        "n_usable": n_usable,
        "n_abstain": n_abstain,
        "abstain_rate": round(n_abstain / max(1, len(pairs)), 4),
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
