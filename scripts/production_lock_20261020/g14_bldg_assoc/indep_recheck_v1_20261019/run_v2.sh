#!/bin/bash
# resume after rog crash 23:39; skips finished outputs. tmux g14_recheck; <=2 workers; 2 GB cap/job
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
J(){ n=$1; shift; systemd-run --user --scope -q -p MemoryMax=2000M -p MemorySwapMax=0 /usr/bin/time -v python3 indep_g14.py "$@" > logs/$n.log 2>&1; echo "$(date '+%F %T %Z') $n exit $?" >> logs/jobs.log; sync; }
export -f J
echo "$(date '+%F %T %Z') resume v2 free=$(free -g|awk '/Mem/{print $7}')G" >> logs/jobs.log
{ [ -s gt/CT_regions.gpkg ] || echo "prep_CT_v2 prep CT"; for m in convnext b2 v2c avg2; do [ -s preds/JHB_$m.gpkg ] || echo "pred_JHB_$m pred JHB $m"; done; } | xargs -P 2 -L 1 bash -c 'J "$@"' _
python3 -c "
import geopandas as g
for c in ['CT','JHB']:
  for x in g.read_file(f'gt/{c}_regions.gpkg').grid: print(c,x)" | while read c x; do [ -s parts/${c}_$x.pkl ] || echo "grid_${c}_$x grid $c $x"; done | xargs -P 2 -L 1 bash -c 'J "$@"' _
J agg agg; echo "$(date '+%F %T %Z') DONE" >> logs/jobs.log
