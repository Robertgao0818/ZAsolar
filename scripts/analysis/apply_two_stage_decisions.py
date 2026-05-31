#!/usr/bin/env python3
"""Apply Gemini two-stage FP-review decisions to the detector predictions gpkg.

Closes the production seam between the reviewer and the inventory. Reads one or
more merged decision JSONLs (from ``gemini_fp_review_two_stage.py`` for the
conf>=0.95 path, and/or ``gemini_fp_review_multiscale.py`` for the conf<0.95
stage-1-only path), joins them to the raw predictions gpkg(s) by
``(predictions_path, pred_id)``, and writes a row-subset "filtered" gpkg per grid
with every ``auto_drop=true`` prediction removed.

STAGE-1-ONLY band (conf 0.925-0.95): ``gemini_fp_review_multiscale.py`` emits
``pv_present``/``label`` but NOT the production fields (``production_action`` /
``auto_drop`` / ``requires_human_review``) this applier drops on. Pass
``--stage1-as-drops`` to synthesize those fields in-process -- ``label=="not_pv"``
-> ``auto_drop=true`` (drop), ``label=="pv"`` -> keep, abstain/unusable -> human
review (never dropped). This mirrors the non-skylight branches of
``gemini_fp_review_two_stage.merge_decision`` (stage 2 / skylight routing is
deliberately NOT applied in the sub-0.95 band). Every synthesized row is then run
through the *same* ``check_two_stage_failclosed.validate_row`` gate before any
drop is applied, so no tiny adapter script is needed for the LO band.

The filtered gpkg keeps the SOURCE schema / CRS / layer untouched (it is a pure
row subset), so it is directly consumable by
``detect_and_evaluate.py --classifier-filtered-gpkg <grid>_filtered.gpkg``
(eval-only, skips detection -- see docs/experiments/exp_cls_detector_integration.md).

FAIL-CLOSED, by design:
- Only ``auto_drop is True`` removes a prediction. ``keep`` and ``review`` rows
  are retained. Predictions with NO decision (e.g. outside the reviewed band) are
  KEPT and counted as ``undecided`` so coverage is visible.
- ``production_action=review`` / ``requires_human_review=true`` rows are written
  to a review-queue CSV; they stay in the inventory until a human rules.
- If two decision files disagree on the same prediction, the conservative
  (non-drop) decision wins and a conflict is logged -- a drop is never applied on
  a contested row.
- Run ``check_two_stage_failclosed.py`` on the decision JSONLs first; this script
  also re-asserts ``auto_drop is True => production_action == 'drop'`` and aborts
  on violation rather than applying a malformed drop set.

``pred_id`` is the positional row index in the gpkg (``reset_index(drop=True)``),
matching ``build_gemini_review_production_manifest.py`` and the renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    return None


def _load_failclosed_validator():
    """Import the standalone fail-closed row validator (same dir, stdlib-only).

    Imported lazily so the applier only pays for it in ``--stage1-as-drops`` mode,
    and so it reuses the *exact* invariant logic the hard gate enforces rather than
    reimplementing it here.
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from check_two_stage_failclosed import validate_row  # noqa: E402

    return validate_row


def synthesize_stage1_decision(rec: dict[str, Any]) -> dict[str, Any]:
    """Add fail-closed production fields to a stage-1-only multiscale row, in place.

    Mirrors the non-skylight branches of
    ``gemini_fp_review_two_stage.merge_decision``:
      * usable ``pv_present is True``  (label 'pv')     -> keep
      * usable ``pv_present is False`` (label 'not_pv') -> drop (auto_drop=True)
      * abstain / unusable / inconsistent               -> review (never dropped)

    Stage-2 (skylight) routing is intentionally NOT applied: the conf 0.925-0.95
    band runs stage-1 only, so every ``not_pv`` becomes a drop regardless of
    ``lookalike_type``. ``pv_present`` is read as the raw JSON value (True / False /
    None) -- it must NOT go through ``_as_bool`` because ``_as_bool(None)`` returns
    False and would silently turn an abstain into a drop. The fields are set to the
    exact ``True`` / ``False`` / ``None`` literals the fail-closed checker asserts on.
    """
    quality = str(rec.get("quality_flag") or "").strip().lower()
    label = str(rec.get("label") or "").strip().lower()
    pv_present = rec.get("pv_present")  # raw JSON bool / None -- do NOT coerce
    if quality == "usable" and pv_present is True and label == "pv":
        rec["production_action"] = "keep"
        rec["auto_drop"] = False
        rec["requires_human_review"] = False
        rec["pv_present"] = True
        decision_src = "stage1_pv"
    elif quality == "usable" and pv_present is False and label == "not_pv":
        rec["production_action"] = "drop"
        rec["auto_drop"] = True
        rec["requires_human_review"] = False
        rec["pv_present"] = False
        decision_src = "stage1_not_pv"
    else:  # abstain / unusable / inconsistent -> fail-closed to human review
        rec["production_action"] = "review"
        rec["auto_drop"] = False
        rec["requires_human_review"] = True
        rec["pv_present"] = None
        decision_src = "stage1_abstain"
    rec.setdefault("production_decision_source", f"stage1_as_drops:{decision_src}")
    return rec


def load_decisions(
    paths: list[Path],
    stage1_as_drops: bool = False,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str], list[str]]:
    """Return {(predictions_path, pred_id): decision}, conflicts, violations.

    On conflict between files, keep the conservative (non-drop) record.

    When ``stage1_as_drops`` is set, rows that lack a ``production_action`` (i.e.
    stage-1-only multiscale output) get their production fields synthesized via
    ``synthesize_stage1_decision`` and EVERY row is re-checked against the
    standalone fail-closed validator, so the LO band needs no external adapter.
    """
    decisions: dict[tuple[str, int], dict[str, Any]] = {}
    conflicts: list[str] = []
    violations: list[str] = []
    validate_row = _load_failclosed_validator() if stage1_as_drops else None
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pred_path = str(rec.get("predictions_path") or "").strip()
                pred_id = rec.get("pred_id")
                if not pred_path or pred_id in (None, ""):
                    continue
                pred_path = str(Path(pred_path).resolve())
                pid = int(pred_id)
                if stage1_as_drops and not str(rec.get("production_action") or "").strip():
                    synthesize_stage1_decision(rec)
                if validate_row is not None:
                    for reason in validate_row(rec):
                        violations.append(
                            f"{path.name}:{line_no} {rec.get('candidate_id')}: {reason}"
                        )
                auto_drop = _as_bool(rec.get("auto_drop")) or False
                action = str(rec.get("production_action") or "").strip().lower()
                # Fail-closed integrity: a drop flag must agree with the action.
                if auto_drop and action and action != "drop":
                    violations.append(
                        f"{path.name}:{line_no} {rec.get('candidate_id')}: auto_drop=true but action={action!r}"
                    )
                key = (pred_path, pid)
                prev = decisions.get(key)
                if prev is None:
                    decisions[key] = rec
                    continue
                prev_drop = _as_bool(prev.get("auto_drop")) or False
                if prev_drop != auto_drop:
                    conflicts.append(f"{rec.get('candidate_id')} @ {pred_path}#{pid}")
                    # keep the non-drop record (conservative)
                    decisions[key] = prev if not prev_drop else rec
                # same verdict: keep first
    return decisions, conflicts, violations


def grid_for_path(dmap: dict[int, dict[str, Any]], pred_path: str) -> str:
    """Authoritative grid id from the decision rows; fall back to the path.

    The decision JSONL carries ``grid_id`` per row, which is robust to either
    predictions layout (``.../<grid>/predictions_metric.gpkg`` or the reviewed
    ``.../<grid>/review/<grid>_reviewed.gpkg``). Parsing ``parent.name`` breaks
    on the latter (yields ``review``), so prefer the field.
    """
    for rec in dmap.values():
        gid = str(rec.get("grid_id") or "").strip()
        if gid:
            return gid.upper()
    return Path(pred_path).parent.name.upper()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--decisions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more merged decision JSONLs (two_stage and/or stage-1-only).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the per-grid <grid>_filtered.gpkg outputs.",
    )
    ap.add_argument("--review-csv", type=Path, help="Where to write the human-review queue (default: out-dir/review_queue.csv).")
    ap.add_argument("--summary", type=Path, help="Where to write the JSON summary (default: out-dir/apply_summary.json).")
    ap.add_argument(
        "--stage1-as-drops",
        action="store_true",
        help="Treat stage-1-only multiscale rows (which carry pv_present/label but no "
        "production_action) as production decisions: label=='not_pv' -> auto_drop=true, "
        "'pv' -> keep, abstain -> human review. Synthesized rows are run through the "
        "fail-closed validator before any drop. Use for the conf 0.925-0.95 LO band.",
    )
    ap.add_argument(
        "--allow-violations",
        action="store_true",
        help="Apply even if a decision row has auto_drop=true with action!=drop (NOT recommended; "
        "run check_two_stage_failclosed.py first).",
    )
    args = ap.parse_args()

    decisions, conflicts, violations = load_decisions(args.decisions, stage1_as_drops=args.stage1_as_drops)
    if violations and not args.allow_violations:
        print("[ABORT] fail-closed integrity violations in decision input(s):", file=sys.stderr)
        for v in violations[:20]:
            print(f"  - {v}", file=sys.stderr)
        print(
            f"  ({len(violations)} total) Run check_two_stage_failclosed.py, or pass --allow-violations.",
            file=sys.stderr,
        )
        sys.exit(2)

    # group decisions by predictions gpkg
    by_path: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for (pred_path, pid), rec in decisions.items():
        by_path[pred_path][pid] = rec

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # In --stage1-as-drops mode, persist the synthesized production decisions so the
    # standalone check_two_stage_failclosed.py can be re-run on them independently.
    synthesized_jsonl: Path | None = None
    if args.stage1_as_drops:
        synthesized_jsonl = args.out_dir / "stage1_as_drops_decisions.jsonl"
        with synthesized_jsonl.open("w", encoding="utf-8") as fh:
            for rec in decisions.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[stage1-as-drops] wrote {len(decisions)} synthesized decision(s) -> {synthesized_jsonl} "
            f"(0 fail-closed violations)"
        )

    review_rows: list[dict[str, Any]] = []
    per_gpkg: list[dict[str, Any]] = []
    totals = {"n_rows": 0, "n_decided": 0, "n_drop": 0, "n_keep": 0, "n_review": 0, "n_undecided": 0}

    for pred_path in sorted(by_path):
        src = Path(pred_path)
        if not src.exists():
            print(f"[WARN] predictions gpkg not found, skipping: {pred_path}", file=sys.stderr)
            continue
        dmap = by_path[pred_path]
        grid = grid_for_path(dmap, pred_path)
        gdf = gpd.read_file(src).reset_index(drop=True)

        drop_ids: set[int] = set()
        n_review = 0
        for pid, rec in dmap.items():
            if pid < 0 or pid >= len(gdf):
                print(f"[WARN] {grid}: pred_id {pid} out of range (n={len(gdf)}), ignoring", file=sys.stderr)
                continue
            if _as_bool(rec.get("auto_drop")):
                drop_ids.add(pid)
            if _as_bool(rec.get("requires_human_review")) or str(rec.get("production_action")).strip().lower() == "review":
                n_review += 1
                review_rows.append(
                    {
                        "candidate_id": rec.get("candidate_id", f"{grid}_pred{pid:06d}"),
                        "grid_id": grid,
                        "pred_id": pid,
                        "predictions_path": pred_path,
                        "production_decision_source": rec.get("production_decision_source", ""),
                        "stage1_reason": rec.get("stage1_reason", ""),
                        "stage2_reason": rec.get("stage2_reason", ""),
                        "image_path": rec.get("image_path", ""),
                        "wide_image_path": rec.get("wide_image_path", ""),
                    }
                )

        keep_mask = ~gdf.index.isin(drop_ids)
        filtered = gdf[keep_mask].copy()
        out_path = args.out_dir / f"{grid}_filtered.gpkg"
        filtered.to_file(out_path, driver="GPKG")

        n_rows = len(gdf)
        n_decided = sum(1 for pid in dmap if 0 <= pid < n_rows)
        rec = {
            "grid_id": grid,
            "predictions_path": pred_path,
            "filtered_gpkg": str(out_path.resolve()),
            "n_rows": n_rows,
            "n_decided": n_decided,
            "n_drop": len(drop_ids),
            "n_keep": n_rows - len(drop_ids),
            "n_review": n_review,
            "n_undecided": n_rows - n_decided,
        }
        per_gpkg.append(rec)
        for k in ("n_rows", "n_decided", "n_drop", "n_review", "n_undecided"):
            totals[k] += rec[k]
        totals["n_keep"] += rec["n_keep"]
        print(
            f"[{grid}] {n_rows} rows -> dropped {len(drop_ids)}, kept {rec['n_keep']} "
            f"(decided {n_decided}, review {n_review}, undecided {rec['n_undecided']}) -> {out_path.name}"
        )

    review_csv = args.review_csv or (args.out_dir / "review_queue.csv")
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    review_fields = [
        "candidate_id", "grid_id", "pred_id", "predictions_path",
        "production_decision_source", "stage1_reason", "stage2_reason",
        "image_path", "wide_image_path",
    ]
    with review_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(review_rows)

    summary = {
        "decision_inputs": [str(p) for p in args.decisions],
        "stage1_as_drops": args.stage1_as_drops,
        "synthesized_decisions_jsonl": str(synthesized_jsonl.resolve()) if synthesized_jsonl else None,
        "n_gpkgs": len(per_gpkg),
        "totals": totals,
        "n_conflicts": len(conflicts),
        "n_integrity_violations": len(violations),
        "review_csv": str(review_csv.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "per_gpkg": per_gpkg,
    }
    summary_path = args.summary or (args.out_dir / "apply_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if conflicts:
        print(f"[WARN] {len(conflicts)} cross-file conflict(s) resolved conservatively (kept).", file=sys.stderr)
    print(
        f"\n[DONE] {len(per_gpkg)} gpkg(s): dropped {totals['n_drop']}, kept {totals['n_keep']}, "
        f"review {totals['n_review']}, undecided {totals['n_undecided']}. "
        f"Review queue -> {review_csv}; summary -> {summary_path}"
    )


if __name__ == "__main__":
    main()
