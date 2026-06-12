#!/usr/bin/env python3
"""C-3(a) Phase 0 — gate calculator (deliverable D).

Reads the labeled ``audit.csv`` (exported by the HTML labeler / filled by a
human or Gemini) and computes the unlabeled-real-PV-as-background rate.  Prints
the PASS / KILL verdict against the 5 % threshold and the per-stratum breakdown,
and writes a ``gate_result.json``.

Decision rule (``core.training.c3a_phase0.compute_gate``):
  affected_rate >= 5 %  -> PASS   (C-3(a) RPN/box-cls ignore work proceeds)
  affected_rate <  5 %  -> KILL   (C-3(a) is killed; see plan line 207)
  < min-decided chips    -> INSUFFICIENT_DATA (audit not finished)

A chip is *affected* iff >= 1 of its background-region proposals is labeled
``confirmed_pv``.  The denominator is every sampled chip in the audit.

Usage
-----
    python scripts/analysis/compute_c3a_phase0_gate.py \
        --audit-csv results/analysis/c3a_phase0/<run_id>/audit.csv \
        --threshold 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.training.c3a_phase0 import (  # noqa: E402
    DEFAULT_GATE_THRESHOLD,
    compute_gate,
    read_audit_csv,
    validate_audit_rows,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="C-3(a) Phase 0 gate calculator")
    ap.add_argument("--audit-csv", type=Path, required=True)
    ap.add_argument("--chip-manifest", type=Path, default=None,
                    help="sampler chip_manifest.csv — used as the TRUE affected-rate "
                         "denominator (all sampled chips). Default: chip_manifest.csv "
                         "next to --audit-csv. Pass --no-manifest-denominator to use "
                         "only chips present in the audit.")
    ap.add_argument("--no-manifest-denominator", action="store_true",
                    help="denominator = distinct chips in audit.csv (NOT recommended; "
                         "undercounts chips that produced zero background proposals)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_GATE_THRESHOLD,
                    help="affected-chip fraction required to PASS (plan: 0.05)")
    ap.add_argument("--min-decided-chips", type=int, default=1,
                    help="below this many decided chips => INSUFFICIENT_DATA")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="default: <audit-csv dir>/gate_result.json")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on KILL/INSUFFICIENT_DATA (for CI gating)")
    args = ap.parse_args()

    if not args.audit_csv.exists():
        sys.exit(f"[gate] missing {args.audit_csv}")

    rows = read_audit_csv(args.audit_csv)
    errors = validate_audit_rows(rows)
    if errors:
        print("[gate][ERROR] audit.csv schema violations:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 3

    # True denominator = all sampled chips (chip_manifest.csv), so chips that
    # produced zero background proposals correctly count as 0-affected.
    sampled_chips = None
    if not args.no_manifest_denominator:
        manifest = args.chip_manifest or (args.audit_csv.parent / "chip_manifest.csv")
        if manifest.exists():
            sampled_chips = read_audit_csv(manifest)
            print(f"[gate] denominator = {len({c['chip_uid'] for c in sampled_chips})} "
                  f"sampled chips from {manifest}")
        else:
            print(f"[gate][WARN] no chip_manifest.csv at {manifest}; denominator "
                  f"falls back to chips present in audit.csv (may undercount).")

    result = compute_gate(
        rows, threshold=args.threshold,
        min_decided_chips=args.min_decided_chips,
        sampled_chips=sampled_chips,
    )

    # ── Print report ────────────────────────────────────────────────────
    print("=" * 64)
    print("C-3(a) Phase 0 GATE")
    print("=" * 64)
    print(f"threshold           : {result.threshold:.1%}")
    print(f"sampled chips        : {result.n_chips_total}")
    print(f"chips decided        : {result.n_chips_decided}")
    print(f"chips affected       : {result.n_chips_affected}")
    print(f"affected rate        : {result.affected_rate:.1%}")
    print("-" * 64)
    print("proposals:")
    print(f"  total              : {result.n_proposals_total}")
    print(f"  confirmed_pv       : {result.n_proposals_confirmed_pv}  -> positive")
    print(f"  lookalike          : {result.n_proposals_lookalike}  -> negative_pool")
    print(f"  ignore_candidate   : {result.n_proposals_ignore_candidate}  -> ignore (capped)")
    print(f"  uncertain          : {result.n_proposals_uncertain}  -> abstain")
    print("-" * 64)
    print("per-stratum (region:imagery_layer):")
    for s in result.per_stratum:
        print(f"  {s.stratum:<32} chips={s.n_chips_total:<4} "
              f"decided={s.n_chips_decided:<4} affected={s.n_chips_affected:<4} "
              f"rate={s.affected_rate:.1%}")
    print("=" * 64)
    print(f"DECISION: {result.decision}", end="")
    if result.decision == "PASS":
        print("  -> C-3(a) RPN/box-cls ignore work PROCEEDS")
    elif result.decision == "KILL":
        print("  -> C-3(a) KILLED (affected rate below threshold)")
    else:
        print("  -> audit not complete; decide more chips")
    print("=" * 64)

    out_json = args.out_json or (args.audit_csv.parent / "gate_result.json")
    payload = asdict(result)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[gate] wrote {out_json}")

    if args.strict and result.decision != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
