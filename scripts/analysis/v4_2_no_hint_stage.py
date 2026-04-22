"""Stage V4.2 no-hint predictions for review via review_detections.py.

Copies each <grid>_no_hint.gpkg to
  results/analysis/v4_2_no_hint_staged/<grid>/predictions_metric.gpkg
so that review_detections.py --predictions-dir picks them up.

After running this script, launch:
  python scripts/annotations/review_detections.py \\
      --region jhb \\
      --predictions-dir results/analysis/v4_2_no_hint_staged \\
      --grid-id G1110 G1111 G1112 ... G1254
"""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT / "results/analysis/v4_2_conf015_sandton_no_hint"
STAGE_DIR = PROJECT / "results/analysis/v4_2_no_hint_staged"

SANDTON = [
    "G1110", "G1111", "G1112", "G1113", "G1114",
    "G1144", "G1145", "G1146", "G1147", "G1148",
    "G1179", "G1180", "G1181", "G1182", "G1183",
    "G1214", "G1215", "G1216", "G1217", "G1218",
    "G1250", "G1251", "G1252", "G1253", "G1254",
]


def main() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    skipped = []
    for g in SANDTON:
        src = SRC_DIR / f"{g}_no_hint.gpkg"
        if not src.exists():
            skipped.append(g)
            continue
        dst_dir = STAGE_DIR / g
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "predictions_metric.gpkg"
        shutil.copy2(src, dst)
        staged.append(g)

    print(f"[stage] {len(staged)} grids → {STAGE_DIR}")
    if skipped:
        print(f"[skip ] {len(skipped)} grids (no _no_hint.gpkg): {skipped}")

    cmd = (
        "python scripts/annotations/review_detections.py "
        "--region jhb "
        f"--predictions-dir {STAGE_DIR.relative_to(PROJECT)} "
        f"--grid-id {' '.join(staged)}"
    )
    print()
    print("[next] Launch review UI:")
    print(f"  {cmd}")


if __name__ == "__main__":
    main()
