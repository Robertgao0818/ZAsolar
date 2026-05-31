#!/usr/bin/env python3
"""Fail-closed data-level gate for two-stage Gemini FP-review merged JSONL.

Self-contained: stdlib only (json, sys, argparse). Deliberately imports NO
repo module so it cannot race a concurrent edit of the gemini_fp_review_*
scripts. Validates the production-action / auto_drop / requires_human_review /
pv_present consistency invariants that guarantee the merge is fail-closed:

  * required fields present: production_action, auto_drop, requires_human_review
  * auto_drop is True            => production_action == 'drop' AND pv_present is False
  * requires_human_review is True => auto_drop is False AND pv_present is None
                                     AND production_action == 'review'
  * production_action == 'review' => auto_drop is False AND pv_present is None
  * production_action == 'keep'   => auto_drop is False AND pv_present is True
  * production_action == 'drop'   => auto_drop is True

Exit code is non-zero if ANY violation is found across all input files.
"""
from __future__ import annotations

import argparse
import json
import sys

REQUIRED_FIELDS = ("production_action", "auto_drop", "requires_human_review")


def validate_row(row):
    """Return a list of human-readable violation reasons for one row.

    An empty list means the row satisfies every fail-closed invariant.
    """
    reasons = []

    # --- required fields present ---------------------------------------
    missing = [f for f in REQUIRED_FIELDS if f not in row]
    if missing:
        reasons.append("missing required field(s): " + ", ".join(missing))
        # Without these we cannot meaningfully check the rest.
        return reasons

    action = row.get("production_action")
    auto_drop = row.get("auto_drop")
    requires_review = row.get("requires_human_review")
    pv_present = row.get("pv_present")  # may legitimately be absent/None

    # --- auto_drop is True => drop + pv_present False ------------------
    if auto_drop is True:
        if action != "drop":
            reasons.append(
                "auto_drop=True but production_action=%r (expected 'drop')" % action
            )
        if pv_present is not False:
            reasons.append(
                "auto_drop=True but pv_present=%r (expected False)" % pv_present
            )

    # --- requires_human_review is True => not auto_drop, pv None, review
    if requires_review is True:
        if auto_drop is not False:
            reasons.append(
                "requires_human_review=True but auto_drop=%r (expected False)"
                % auto_drop
            )
        if pv_present is not None:
            reasons.append(
                "requires_human_review=True but pv_present=%r (expected None)"
                % pv_present
            )
        if action != "review":
            reasons.append(
                "requires_human_review=True but production_action=%r (expected 'review')"
                % action
            )

    # --- production_action == 'review' => not auto_drop, pv None -------
    if action == "review":
        if auto_drop is not False:
            reasons.append(
                "production_action='review' but auto_drop=%r (expected False)"
                % auto_drop
            )
        if pv_present is not None:
            reasons.append(
                "production_action='review' but pv_present=%r (expected None)"
                % pv_present
            )

    # --- production_action == 'keep' => not auto_drop, pv True ---------
    if action == "keep":
        if auto_drop is not False:
            reasons.append(
                "production_action='keep' but auto_drop=%r (expected False)"
                % auto_drop
            )
        if pv_present is not True:
            reasons.append(
                "production_action='keep' but pv_present=%r (expected True)"
                % pv_present
            )

    # --- production_action == 'drop' => auto_drop True ----------------
    if action == "drop":
        if auto_drop is not True:
            reasons.append(
                "production_action='drop' but auto_drop=%r (expected True)"
                % auto_drop
            )

    return reasons


def validate_file(path):
    """Validate one JSONL file.

    Returns (n_rows, violations) where violations is a list of
    (lineno, candidate_id, reason_string) tuples.
    """
    n_rows = 0
    violations = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                violations.append((lineno, "<unparseable>", "invalid JSON: %s" % exc))
                continue
            cand_id = row.get("candidate_id", row.get("target_id", "<no candidate_id>"))
            for reason in validate_row(row):
                violations.append((lineno, cand_id, reason))
    return n_rows, violations


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail-closed data-level gate for two-stage merged JSONL."
    )
    parser.add_argument(
        "jsonl",
        nargs="+",
        help="One or more two_stage merged JSONL paths.",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=20,
        help="Max offending candidate_ids to print per file (default 20).",
    )
    args = parser.parse_args(argv)

    total_violations = 0
    for path in args.jsonl:
        try:
            n_rows, violations = validate_file(path)
        except FileNotFoundError:
            print("ERROR: file not found: %s" % path)
            total_violations += 1
            continue
        total_violations += len(violations)
        print(
            "%s: %d rows, %d violation(s)" % (path, n_rows, len(violations))
        )
        for lineno, cand_id, reason in violations[: args.max_show]:
            print("    line %d  %s: %s" % (lineno, cand_id, reason))
        extra = len(violations) - args.max_show
        if extra > 0:
            print("    ... and %d more violation(s)" % extra)

    if total_violations:
        print("FAIL: %d total violation(s)" % total_violations)
        return 1
    print("PASS: 0 violations across %d file(s)" % len(args.jsonl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
