"""TP-lost overlap and non-B subset breakdown for cls v1 holdout audit.

Inputs (manually exported from the in-browser labeler, see
project_cls_v1_holdout_audit memory):
  - tp_lost_labels_convnext.csv
  - tp_lost_labels_dinov2.csv

Outputs:
  - per-chip overlap CSV (key + cn_label + dn_label + bucket)
  - summary JSON with overlap counts, per-bucket label dists, agreement matrix

The v1 audit memory already established B=84% for ConvNeXt overall. This
script answers the follow-up: of the 16% non-B chips, how do they split by
overlap (shared vs cn-only vs dn-only) and label class (A/C/D/unlabeled),
which informs whether v2 threshold calibration should discount C (GT errors)
or A (ambiguous) cases.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

LABELS = ("A", "B", "C", "D", "UNLABELED")


def norm(label: str | None) -> str:
    s = (label or "").strip().upper()
    return s if s in {"A", "B", "C", "D"} else "UNLABELED"


def key(row: dict) -> tuple[str, str, str]:
    return (row["region"], row["grid_id"], row["pred_idx"])


def load(path: Path) -> dict[tuple[str, str, str], dict]:
    with path.open() as f:
        return {key(r): r for r in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cn-csv", default="/mnt/c/Users/gaosh/Downloads/tp_lost_labels_convnext.csv")
    p.add_argument("--dn-csv", default="/mnt/c/Users/gaosh/Downloads/tp_lost_labels_dinov2.csv")
    p.add_argument("--out-dir", default="results/analysis/cls_cascade_holdout/tp_lost_overlap")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cn = load(Path(args.cn_csv))
    dn = load(Path(args.dn_csv))

    shared = set(cn) & set(dn)
    cn_only = set(cn) - shared
    dn_only = set(dn) - shared
    union = set(cn) | set(dn)

    per_chip_rows = []
    for k in sorted(union):
        cl = norm(cn[k]["label"]) if k in cn else ""
        dl = norm(dn[k]["label"]) if k in dn else ""
        bucket = "shared" if k in shared else ("cn_only" if k in cn_only else "dn_only")
        region, grid, pidx = k
        sample = cn.get(k) or dn.get(k)
        per_chip_rows.append({
            "region": region,
            "grid_id": grid,
            "pred_idx": pidx,
            "bucket": bucket,
            "cn_label": cl,
            "dn_label": dl,
            "cn_score": cn[k]["cls_score"] if k in cn else "",
            "dn_score": dn[k]["cls_score"] if k in dn else "",
            "area_m2": sample["area_m2"],
        })

    chip_csv = out_dir / "per_chip_overlap.csv"
    with chip_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_chip_rows[0].keys()))
        w.writeheader()
        w.writerows(per_chip_rows)

    def label_dist(rows: list[dict], side: str) -> dict[str, int]:
        c = Counter(r[side] for r in rows if r[side])
        return {lab: c.get(lab, 0) for lab in LABELS}

    summary = {
        "n_convnext_lost": len(cn),
        "n_dinov2_lost": len(dn),
        "n_shared": len(shared),
        "n_convnext_only": len(cn_only),
        "n_dinov2_only": len(dn_only),
        "n_union": len(union),
        "shared_pct_of_union": round(len(shared) / len(union) * 100, 1),
        "convnext_label_dist": label_dist([r for r in per_chip_rows if r["cn_label"]], "cn_label"),
        "dinov2_label_dist": label_dist([r for r in per_chip_rows if r["dn_label"]], "dn_label"),
        "buckets": {},
        "shared_agreement_matrix": {},
    }

    for bucket in ("shared", "cn_only", "dn_only"):
        rs = [r for r in per_chip_rows if r["bucket"] == bucket]
        side = "cn_label" if bucket != "dn_only" else "dn_label"
        labelled = [r for r in rs if r[side]]
        non_b = [r for r in labelled if r[side] != "B"]
        summary["buckets"][bucket] = {
            "n_chips": len(rs),
            "label_dist": label_dist(rs, side),
            "non_b_count": len(non_b),
            "non_b_pct": round(len(non_b) / len(rs) * 100, 1) if rs else 0.0,
        }

    agree: Counter = Counter()
    for r in per_chip_rows:
        if r["bucket"] == "shared":
            agree[(r["cn_label"], r["dn_label"])] += 1
    summary["shared_agreement_matrix"] = {f"{cl}|{dl}": n for (cl, dl), n in sorted(agree.items())}

    cn_real = [r for r in per_chip_rows if r["cn_label"]]
    cn_drop_c = [r for r in cn_real if r["cn_label"] != "C"]
    cn_b_after_c = sum(1 for r in cn_drop_c if r["cn_label"] == "B")
    summary["convnext_b_pct_excluding_C"] = round(cn_b_after_c / len(cn_drop_c) * 100, 1) if cn_drop_c else 0.0

    dn_real = [r for r in per_chip_rows if r["dn_label"]]
    dn_drop_c = [r for r in dn_real if r["dn_label"] != "C"]
    dn_b_after_c = sum(1 for r in dn_drop_c if r["dn_label"] == "B")
    summary["dinov2_b_pct_excluding_C"] = round(dn_b_after_c / len(dn_drop_c) * 100, 1) if dn_drop_c else 0.0

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"per-chip CSV : {chip_csv}")
    print(f"summary JSON : {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
