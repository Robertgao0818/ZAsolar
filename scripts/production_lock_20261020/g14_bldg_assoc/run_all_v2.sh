#!/bin/bash
# G14 chain, CPU only, ≤4 workers. tmux g14_bldg_assoc
cd ~/capetown_eval/g14_bldg_assoc_v1_20261019; echo "$(date '+%F %T %Z') chain start; free: $(free -g | awk '/Mem/{print $7}') GB avail" >> logs/jobs.log
{ for c in CT JHB; do [ -s gt/${c}_gt.gpkg ] || echo "gt_$c prep_gt_v1.py $c"; done
  for m in convnext b2 avg2; do [ -s preds/CT_$m.gpkg ] || echo "pred_CT_$m prep_preds_v1.py CT $m"; done
  for m in convnext b2 v2c avg2; do [ -s preds/JHB_$m.gpkg ] || echo "pred_JHB_$m prep_preds_v1.py JHB $m"; done; } | xargs -P 4 -L 1 ./run_job.sh
[ "$(cat logs/consec_fail)" -ge 2 ] && { echo "$(date '+%F %T %Z') STOP after stage1" >> logs/jobs.log; exit 2; }
ls preds/*.gpkg gt/*_gt.gpkg | wc -l | grep -qx 9 || { echo "$(date '+%F %T %Z') STOP stage1 incomplete" >> logs/jobs.log; exit 3; }
python3 -c "
import geopandas as g
for c in ['CT','JHB']:
  for x in g.read_file(f'gt/{c}_regions.gpkg').grid: print(c,x)" | while read c g; do [ -s parts/${c}_$g.pkl ] || echo "assoc2_${c}_$g assoc_grid_v2.py $c $g"; done | xargs -P 4 -L 1 ./run_job.sh
[ "$(cat logs/consec_fail)" -ge 2 ] && { echo "$(date '+%F %T %Z') STOP after stage2" >> logs/jobs.log; exit 2; }
echo "$(date '+%F %T %Z') stage2 done" >> logs/jobs.log
