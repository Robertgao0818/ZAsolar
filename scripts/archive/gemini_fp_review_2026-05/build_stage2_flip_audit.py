#!/usr/bin/env python3
"""Build a Stage-2 FLIP/DROP audit queue from a two-stage Gemini FP-review run.

Lane A / T3 tooling. Self-contained: stdlib only (no repo imports, no network).

Two-stage FP review semantics (from gemini_fp_review_two_stage.py):
  - Stage 1 (multiscale) emits a per-candidate label. Candidates labelled
    skylight / not-PV are escalated to Stage 2.
  - Stage 2 re-adjudicates. The production decision source records the outcome:
      * 'stage2_skylight_keep'  -> Stage 1 said skylight/not-PV, Stage 2 RESTORED
                                   it as PV (a FLIP -- the case we audit for
                                   overfit, where stage2 may wrongly resurrect a
                                   real false positive).
      * 'stage2_skylight_drop'  -> Stage 2 agreed it is not PV and dropped it
                                   (a DROP -- audited for true-positive loss).

We join the RA (human reviewer) label per candidate and assign an auto-verdict:

  FLIP rows (stage2_skylight_keep):
    - ra_label == PV      -> 'correct_save'   (stage2 rescued a genuine PV)
    - ra_label != PV      -> 'wrong_restore'  (stage2 resurrected a real FP -- the
                                               dangerous overfit failure mode)

  DROP rows (stage2_skylight_drop), seeded-sampled up to --drop-sample-n:
    - ra_label == non_PV  -> 'correct_cut'    (legit FP suppression)
    - ra_label == PV      -> 'tp_loss'        (stage2 killed a true positive)

drop_precision is computed over the FULL drop set (not just the sample):
    drop_precision = (# drops with ra_label non_PV) / (total drops)

Outputs an audit CSV + a side-by-side tight/wide HTML gallery so a human can
eyeball whether the stage2 flips are principled.
"""

import argparse
import csv
import html
import json
import random


def _norm_ra_label(raw):
    """Normalize an RA label string to 'pv' / 'non_pv' / '' (unknown)."""
    if raw is None:
        return ""
    v = str(raw).strip().lower()
    if not v:
        return ""
    if v in ("pv", "tp", "true_positive", "positive", "solar", "correct", "edit"):
        return "pv"
    if v in ("non_pv", "nonpv", "non-pv", "fp", "false_positive", "negative",
             "not_pv", "delete"):
        return "non_pv"
    return v


def _is_pv(ra_label):
    return _norm_ra_label(ra_label) == "pv"


def load_manifest_ra_labels(manifest_path):
    """Return {candidate_id: ra_label_string} from the calibration manifest."""
    ra = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("candidate_id") or "").strip()
            if not cid:
                continue
            # Prefer explicit ra_label; fall back to review_status mapping.
            label = (row.get("ra_label") or "").strip()
            if not label:
                status = (row.get("review_status") or "").strip().lower()
                if status in ("correct", "edit"):
                    label = "pv"
                elif status == "delete":
                    label = "non_pv"
            ra[cid] = label
    return ra


def load_two_stage(jsonl_path):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_audit_record(rec, ra_label, kind):
    """kind is 'flip' or 'drop'. Returns the audit dict + auto_verdict."""
    pv = _is_pv(ra_label)
    if kind == "flip":
        verdict = "correct_save" if pv else "wrong_restore"
    else:  # drop
        verdict = "tp_loss" if pv else "correct_cut"

    def g(key):
        v = rec.get(key)
        return "" if v is None else v

    return {
        "candidate_id": g("candidate_id"),
        "grid_id": g("grid_id"),
        "ra_label": _norm_ra_label(ra_label) or "unknown",
        "stage1_label": g("stage1_label"),
        "stage1_lookalike_type": g("stage1_lookalike_type"),
        "stage1_reason": g("stage1_reason"),
        "stage2_label": g("stage2_label"),
        "stage2_confidence": g("stage2_confidence"),
        "stage2_reason": g("stage2_reason"),
        "decision_source": g("production_decision_source"),
        "auto_verdict": verdict,
        "image_path": g("image_path"),
        "wide_image_path": g("wide_image_path"),
    }


CSV_COLUMNS = [
    "candidate_id", "grid_id", "ra_label", "stage1_label",
    "stage1_lookalike_type", "stage1_reason", "stage2_label",
    "stage2_confidence", "stage2_reason", "decision_source", "auto_verdict",
    "image_path", "wide_image_path",
]


def write_csv(out_csv, records):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow({c: r.get(c, "") for c in CSV_COLUMNS})


_VERDICT_COLOR = {
    "correct_save": "#1b7a1b",   # green: good flip
    "wrong_restore": "#b30000",  # red: dangerous overfit flip
    "correct_cut": "#1b7a1b",    # green: good drop
    "tp_loss": "#b30000",        # red: lost a TP
}


def write_html(out_html, records, summary):
    parts = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>Stage-2 FLIP/DROP audit</title>")
    parts.append("<style>")
    parts.append("body{font-family:sans-serif;margin:16px;background:#fafafa;}")
    parts.append("table{border-collapse:collapse;width:100%;}")
    parts.append("td,th{border:1px solid #ccc;padding:6px;vertical-align:top;}")
    parts.append("img{max-width:380px;height:auto;border:1px solid #999;}")
    parts.append(".verdict{font-weight:bold;}")
    parts.append(".meta{font-size:12px;color:#333;}")
    parts.append(".reason{font-size:12px;color:#444;max-width:320px;}")
    parts.append("th{background:#eee;text-align:left;}")
    parts.append("</style></head><body>")

    parts.append("<h2>Stage-2 FLIP / DROP audit queue</h2>")
    parts.append("<div class='meta'><pre>")
    parts.append(html.escape(json.dumps(summary, indent=2)))
    parts.append("</pre></div>")

    parts.append("<table>")
    parts.append(
        "<tr><th>candidate / verdict</th><th>tight (module)</th>"
        "<th>wide (roof)</th><th>reasons</th></tr>"
    )
    for r in records:
        color = _VERDICT_COLOR.get(r["auto_verdict"], "#333")
        cid = html.escape(str(r["candidate_id"]))
        grid = html.escape(str(r["grid_id"]))
        ra = html.escape(str(r["ra_label"]))
        verdict = html.escape(str(r["auto_verdict"]))
        src = html.escape(str(r["decision_source"]))
        s1l = html.escape(str(r["stage1_label"]))
        s1t = html.escape(str(r["stage1_lookalike_type"]))
        s2l = html.escape(str(r["stage2_label"]))
        s2c = html.escape(str(r["stage2_confidence"]))
        s1r = html.escape(str(r["stage1_reason"]))
        s2r = html.escape(str(r["stage2_reason"]))
        tight = html.escape(str(r["image_path"]))
        wide = html.escape(str(r["wide_image_path"]))

        parts.append("<tr>")
        parts.append(
            "<td class='meta'>"
            f"<div><b>{cid}</b></div>"
            f"<div>grid: {grid}</div>"
            f"<div>RA: <b>{ra}</b></div>"
            f"<div class='verdict' style='color:{color}'>{verdict}</div>"
            f"<div>src: {src}</div>"
            f"<div>s1: {s1l} / {s1t}</div>"
            f"<div>s2: {s2l} (conf {s2c})</div>"
            "</td>"
        )
        parts.append(f"<td><img src='{tight}' alt='tight'></td>")
        parts.append(f"<td><img src='{wide}' alt='wide'></td>")
        parts.append(
            "<td class='reason'>"
            f"<div><b>stage1:</b> {s1r}</div>"
            f"<div style='margin-top:6px'><b>stage2:</b> {s2r}</div>"
            "</td>"
        )
        parts.append("</tr>")
    parts.append("</table></body></html>")

    with open(out_html, "w") as f:
        f.write("".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--two-stage-jsonl", required=True)
    ap.add_argument("--manifest", required=True,
                    help="Calibration manifest CSV carrying ra_label by candidate_id")
    ap.add_argument("--drop-sample-n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-html", required=True)
    args = ap.parse_args()

    ra_labels = load_manifest_ra_labels(args.manifest)
    rows = load_two_stage(args.two_stage_jsonl)

    flips = []
    drops = []
    for rec in rows:
        src = rec.get("production_decision_source")
        if src == "stage2_skylight_keep":
            flips.append(rec)
        elif src == "stage2_skylight_drop":
            drops.append(rec)

    # --- FLIP records (audit all) ---
    flip_records = []
    for rec in flips:
        cid = rec.get("candidate_id")
        ra = ra_labels.get(cid, "")
        flip_records.append(build_audit_record(rec, ra, "flip"))

    # --- DROP precision over the FULL drop set ---
    n_drops_total = len(drops)
    n_correct_cut_full = 0
    n_tp_loss_full = 0
    for rec in drops:
        cid = rec.get("candidate_id")
        ra = ra_labels.get(cid, "")
        if _is_pv(ra):
            n_tp_loss_full += 1
        else:
            n_correct_cut_full += 1
    drop_precision = (n_correct_cut_full / n_drops_total) if n_drops_total else 0.0

    # --- seeded sample of drops for the human audit queue ---
    rng = random.Random(args.seed)
    sample_n = min(args.drop_sample_n, n_drops_total)
    sampled_drops = rng.sample(drops, sample_n) if sample_n else []
    drop_records = []
    for rec in sampled_drops:
        cid = rec.get("candidate_id")
        ra = ra_labels.get(cid, "")
        drop_records.append(build_audit_record(rec, ra, "drop"))

    audit_records = flip_records + drop_records

    # --- flip verdict breakdown ---
    flip_breakdown = {}
    for r in flip_records:
        flip_breakdown[r["auto_verdict"]] = flip_breakdown.get(r["auto_verdict"], 0) + 1

    sample_breakdown = {}
    for r in drop_records:
        sample_breakdown[r["auto_verdict"]] = sample_breakdown.get(r["auto_verdict"], 0) + 1

    summary = {
        "two_stage_jsonl": args.two_stage_jsonl,
        "manifest": args.manifest,
        "n_total_candidates": len(rows),
        "n_flips": len(flip_records),
        "flip_auto_verdict_breakdown": flip_breakdown,
        "n_drops_total": n_drops_total,
        "drop_precision_full_set": round(drop_precision, 4),
        "n_tp_loss_full_set": n_tp_loss_full,
        "n_correct_cut_full_set": n_correct_cut_full,
        "drop_sample_n_requested": args.drop_sample_n,
        "drop_sample_n_actual": sample_n,
        "drop_sample_auto_verdict_breakdown": sample_breakdown,
        "seed": args.seed,
        "n_audited_rows": len(audit_records),
    }

    write_csv(args.out_csv, audit_records)
    write_html(args.out_html, audit_records, summary)

    # --- print summary ---
    print("=== Stage-2 FLIP/DROP audit ===")
    print(f"n_flips            : {len(flip_records)}")
    print(f"flip breakdown     : {flip_breakdown}")
    print(f"n_drops_total      : {n_drops_total}")
    print(f"drop_precision     : {drop_precision:.4f}  "
          f"(= {n_correct_cut_full} correct_cut / {n_drops_total} total drops)")
    print(f"n_tp_loss (full)   : {n_tp_loss_full}")
    print(f"drop sample        : n={sample_n} verdicts={sample_breakdown}")
    print(f"csv  -> {args.out_csv}")
    print(f"html -> {args.out_html}")


if __name__ == "__main__":
    main()
