#!/usr/bin/env bash
# CT avg2 pod runs. smoke: 100-tile block in /workspace/smoke. full: follow READY.txt until all 52,251 are done.
set -euo pipefail
PY=/workspace/venv/bin/python; S=/workspace/code/scripts/production_lock_20261020/citywide_infer/ct_city_infer_v2.py
case "${1:?smoke|full}" in
  smoke) cd /workspace/smoke && $PY $S infer --tiles-csv smoke_tiles.csv --tile-dir tiles --ckpt-root /workspace/weights \
           --out /workspace/smoke/out --writers ${WRITERS:-8}
         $PY -c "import sys; sys.path.insert(0, '$(dirname $S)'); import ct_city_infer_v2 as M
t = M.read_tile_list('/workspace/smoke/smoke_tiles.csv'); d = M.update_done_manifest(t, '/workspace/smoke/out')
print(f'SMOKE {d}/{len(t)} complete on disk'); sys.exit(0 if d == len(t) else 1)" ;;
  full)  cd /workspace && $PY $S infer ${MODEL:+--model $MODEL} --tiles-csv /workspace/lists2/seg_tiles_kept_plus_qa_v1.csv --tile-dir /workspace/ct_tiles \
           --ckpt-root /workspace/weights --out /workspace/out --writers ${WRITERS:-8} --follow --ready-file /workspace/ct_tiles/READY.txt --poll-s 30 ;;
  full2) # one process per model on the same GPU (main loop is single-thread launch-bound); both must finish
         MODEL=convnext_t_s0 bash "$0" full & p1=$!; MODEL=mit_b2_s2 bash "$0" full & p2=$!
         r1=0; wait $p1 || r1=$?; r2=0; wait $p2 || r2=$?; echo "convnext=$r1 b2=$r2"
         $PY -c "import sys; sys.path.insert(0, '$(dirname $S)'); import ct_city_infer_v2 as M
t = M.read_tile_list('/workspace/lists2/seg_tiles_kept_plus_qa_v1.csv'); d = M.update_done_manifest(t, '/workspace/out')
print(f'ALL MODELS {d}/{len(t)} complete on disk'); sys.exit(0 if d == len(t) and $r1 == 0 and $r2 == 0 else 1)" ;;
  *) echo "unknown mode $1 (smoke|full|full2)" >&2; exit 2 ;;
esac
