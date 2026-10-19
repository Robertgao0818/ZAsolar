#!/bin/bash
# G6 GPU chain: v2c verify -> STOP on mismatch -> mit_b2_s0 -> convnext_t_s0
cd ~/capetown_eval/joburg_cbd25_rescore_v2_20261019; E=$HOME/capetown_eval/ds3m_ft
./run1_gpu.sh v2c_verify mit $E/evals/ft_best.pt || exit 1
python3 - <<'P' || { echo "$(date '+%F %T') v2c MISMATCH stop" >> logs/chain.log; exit 2; }
import json;a=json.load(open('out/v2c_verify.json'))['res'];b=json.load(open('../ds3m_ft/joburg25_vexcel/mit_v2c.json'))['res']
bad=[(c,k,a[c][k],b[c][k]) for c in b for k in b[c] if a[c][k]!=b[c][k]]
print('mismatches',len(bad)); [print(x) for x in bad[:40]]
h=a['pg_h0.9_0.5'];t=a['pg_t0.5'];ok=(h['obj_prec'],h['obj_rec'],h['obj_f1'],h['px_iou'],t['obj_f1'])==(0.856,0.947,0.899,0.86,0.892); print('headline ok',ok); exit(0 if ok else 1)
P
echo "$(date '+%F %T') v2c verified" >> logs/chain.log
./run1_gpu.sh p1_mit_b2_s0 p1 $E/phase1_runpod_20261017/mit_b2_s0/best_objf1.pt mit_b2 0
./run1_gpu.sh p1_convnext_t_s0 p1 $E/phase1_runpod_20261017/convnext_t_s0/best_objf1.pt convnext_t 0
echo "$(date '+%F %T') chain done" >> logs/chain.log
