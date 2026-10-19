#!/usr/bin/env bash
# CT avg2 pod runs. smoke: 100-tile block in /workspace/smoke. full: follow READY.txt until all 52,251 are done.
set -euo pipefail
PY=/workspace/venv/bin/python; S=/workspace/code/scripts/production_lock_20261020/citywide_infer/ct_city_infer_v2.py
case "${1:?smoke|full}" in
  smoke) cd /workspace/smoke && $PY $S infer --tiles-csv smoke_tiles.csv --tile-dir tiles --ckpt-root /workspace/weights \
           --out /workspace/smoke/out --writers ${WRITERS:-8}
         n=$(tail -n +2 /workspace/smoke/smoke_tiles.csv | grep -c .); d=$(wc -l < /workspace/smoke/out/done_manifest.txt)
         [ "$n" = "$d" ] || { echo "SMOKE FAILED: $d/$n tiles complete" >&2; exit 1; }; echo "SMOKE OK $d/$n" ;;
  full)  cd /workspace && $PY $S infer --tiles-csv /workspace/lists2/seg_tiles_kept_plus_qa_v1.csv --tile-dir /workspace/ct_tiles \
           --ckpt-root /workspace/weights --out /workspace/out --writers ${WRITERS:-8} --follow --ready-file /workspace/ct_tiles/READY.txt --poll-s 30 ;;
  *) echo "unknown mode $1 (smoke|full)" >&2; exit 2 ;;
esac
