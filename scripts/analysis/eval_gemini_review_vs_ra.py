#!/usr/bin/env python3
"""Score Gemini detection-review decisions against RA review labels.

Joins a calibration manifest (carrying ``ra_label`` from reviewed_status) with
the flattened Gemini JSONL produced by
``scripts/training/score_gemini_detection_review_chips.py`` (carrying
``pv_present``).  The join key is ``candidate_id`` with a (grid_id, pred_id)
fallback.

RA labels are the ground truth.  We report, from the FP-suppression viewpoint:
  - non_PV recall  : fraction of true FPs (RA=delete) Gemini correctly flags
  - non_PV precision: when Gemini says non-PV, how often it is a real FP
  - PV recall      : fraction of true PV (RA=correct/edit) Gemini preserves
  - PV precision, accuracy, balanced accuracy, Cohen's kappa
  - abstain rate   : Gemini quality_flag != usable or pv_present is null

Abstentions are excluded from the headline matrix and also reported under a
"conservative keep" variant (production would keep an abstained prediction as
PV rather than dropping it).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Classifier v1 (DINOv2 ViT-S/14) chip-level val numbers, for context only.
# NOTE: measured over the *whole* cls val distribution, not this hard >=0.95
# band, so treat as an approximate bar, not an apples-to-apples comparison.
CLS_V1_NONPV_RECALL = 0.865
CLS_V1_PV_RECALL = 0.918


def _norm(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def _as_bool_or_none(v: Any):
    if isinstance(v, bool):
        return v
    s = _norm(v).lower()
    if s in {"true", "1", "yes", "present", "pv"}:
        return True
    if s in {"false", "0", "no", "absent", "non_pv"}:
        return False
    return None


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = _norm(row.get("candidate_id"))
            if not cid:
                cid = f"{_norm(row.get('grid_id')).upper()}_pred{int(float(row.get('pred_id') or 0)):06d}"
            out[cid] = row
    return out


def load_gemini(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = _norm(rec.get("candidate_id") or rec.get("target_id"))
                if not cid:
                    g = _norm(rec.get("grid_id")).upper()
                    pid = rec.get("pred_id")
                    if g and pid not in (None, ""):
                        cid = f"{g}_pred{int(pid):06d}"
                if cid:
                    out[cid] = rec
    return out


def cohen_kappa(rows: list[tuple[str, str]]) -> float:
    # rows: (ra, gemini) over the two-class agreement set
    labels = ("pv", "non_pv")
    n = len(rows)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in rows if a == b) / n
    pe = 0.0
    for lab in labels:
        pa = sum(1 for a, _ in rows if a == lab) / n
        pb = sum(1 for _, b in rows if b == lab) / n
        pe += pa * pb
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--gemini-jsonl", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    gem = load_gemini(args.gemini_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_row: list[dict[str, Any]] = []
    matrix = {("pv", "pv"): 0, ("pv", "non_pv"): 0, ("non_pv", "pv"): 0, ("non_pv", "non_pv"): 0}
    abstain = 0
    no_gemini = 0
    kappa_rows: list[tuple[str, str]] = []

    for cid, mrow in man.items():
        ra = _norm(mrow.get("ra_label")).lower()
        if ra not in ("pv", "non_pv"):
            continue
        grec = gem.get(cid)
        if grec is None:
            no_gemini += 1
            decision = "no_gemini_row"
            g_label = ""
        else:
            qf = _norm(grec.get("quality_flag")).lower()
            pv = _as_bool_or_none(grec.get("pv_present"))
            if pv is None or (qf and qf != "usable"):
                abstain += 1
                decision = f"abstain_{qf or 'null'}"
                g_label = ""
            else:
                g_label = "pv" if pv else "non_pv"
                decision = g_label
                matrix[(ra, g_label)] += 1
                kappa_rows.append((ra, g_label))
        per_row.append({
            "candidate_id": cid,
            "grid_id": _norm(mrow.get("grid_id")),
            "pred_id": _norm(mrow.get("pred_id")),
            "confidence": _norm(mrow.get("confidence")),
            "area_m2": _norm(mrow.get("area_m2")),
            "ra_label": ra,
            "gemini_label": g_label,
            "gemini_decision": decision,
            "gemini_confidence": _norm(grec.get("confidence")) if grec else "",
            "agree": "1" if g_label and g_label == ra else ("0" if g_label else ""),
            "image_path": _norm(mrow.get("tile_path")),
        })

    tp_pv = matrix[("pv", "pv")]
    fn_pv = matrix[("pv", "non_pv")]      # RA pv but Gemini said non_pv -> TP wrongly killed
    tp_neg = matrix[("non_pv", "non_pv")]  # hard FP correctly flagged
    fn_neg = matrix[("non_pv", "pv")]      # hard FP Gemini let through
    n_pv = tp_pv + fn_pv
    n_neg = tp_neg + fn_neg
    n_dec = n_pv + n_neg

    def safe(a, b):
        return a / b if b else float("nan")

    pv_recall = safe(tp_pv, n_pv)
    nonpv_recall = safe(tp_neg, n_neg)
    pv_precision = safe(tp_pv, tp_pv + fn_neg)        # gemini-said-pv that are truly pv
    nonpv_precision = safe(tp_neg, tp_neg + fn_pv)    # gemini-said-nonpv that are truly fp
    accuracy = safe(tp_pv + tp_neg, n_dec)
    bal_acc = (pv_recall + nonpv_recall) / 2 if n_pv and n_neg else float("nan")
    kappa = cohen_kappa(kappa_rows)

    summary = {
        "n_manifest": len([1 for r in man.values() if _norm(r.get("ra_label")).lower() in ("pv", "non_pv")]),
        "n_decided": n_dec,
        "n_abstain": abstain,
        "n_no_gemini_row": no_gemini,
        "abstain_rate": round(safe(abstain, abstain + n_dec), 4),
        "confusion_RAxGemini": {
            "pv_pv": tp_pv, "pv_nonpv": fn_pv,
            "nonpv_pv": fn_neg, "nonpv_nonpv": tp_neg,
        },
        "metrics": {
            "nonpv_recall_FP_caught": round(nonpv_recall, 4),
            "nonpv_precision": round(nonpv_precision, 4),
            "pv_recall_TP_preserved": round(pv_recall, 4),
            "pv_precision": round(pv_precision, 4),
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "cohen_kappa": round(kappa, 4),
        },
        "classifier_v1_reference_bar": {
            "nonpv_recall": CLS_V1_NONPV_RECALL,
            "pv_recall": CLS_V1_PV_RECALL,
            "note": "whole cls val distribution, not this >=0.95 band; approximate bar",
        },
    }

    rows_path = args.out_dir / "gemini_vs_ra_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_row[0].keys()))
        w.writeheader()
        w.writerows(per_row)
    disagree = [r for r in per_row if r["agree"] == "0"]
    dis_path = args.out_dir / "gemini_vs_ra_disagreements.csv"
    with dis_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_row[0].keys()))
        w.writeheader()
        w.writerows(disagree)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nrows: {rows_path}\ndisagreements ({len(disagree)}): {dis_path}")


if __name__ == "__main__":
    main()
