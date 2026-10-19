#!/bin/bash
# tmux g14_recheck; CPU only; <=2 workers; 2 GB cap per job
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
J(){ n=$1; shift; systemd-run --user --scope -q -p MemoryMax=2000M -p MemorySwapMax=0 /usr/bin/time -v python3 indep_g14.py "$@" > logs/$n.log 2>&1; echo "$(date '+%F %T %Z') $n exit $?" >> logs/jobs.log; }
export -f J
echo "$(date '+%F %T %Z') start free=$(free -g|awk '/Mem/{print $7}')G" >> logs/jobs.log
printf "prep_CT prep CT\nprep_JHB prep JHB\n" | xargs -P 2 -L 1 bash -c 'J "$@"' _
{ for m in convnext b2 avg2; do echo "pred_CT_$m pred CT $m"; done; for m in convnext b2 v2c avg2; do echo "pred_JHB_$m pred JHB $m"; done; } | xargs -P 2 -L 1 bash -c 'J "$@"' _
python3 -c "
import geopandas as g
for c in ['CT','JHB']:
  for x in g.read_file(f'gt/{c}_regions.gpkg').grid: print(f'grid_{c}_{x} grid {c} {x}')" | xargs -P 2 -L 1 bash -c 'J "$@"' _
J agg agg; echo "$(date '+%F %T %Z') DONE" >> logs/jobs.log
