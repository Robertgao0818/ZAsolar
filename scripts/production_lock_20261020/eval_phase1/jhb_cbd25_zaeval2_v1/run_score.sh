#!/bin/bash
# Task B CPU scoring (no GPU): 4 models x 2 cfgs, 4 parallel; skips existing outputs
cd ~/capetown_eval/jhb_cbd25_zaeval2_v1_20261019
echo "$(date '+%F %T %Z') score start" >> logs/chain.log
for m in convnext_t_s0 mit_b2_s0 v2c avg3; do for c in h0.9_0.5 t0.5; do [ -s scores/${m}__$c.json ] || echo "$m $c"; done; done | \
 xargs -P 4 -L 1 sh -c 'OMP_NUM_THREADS=2 python3 score_z2_v1.py $0 $1 > logs/score_$0__$1.log 2>&1'
echo "$(date '+%F %T %Z') score done exit $?" >> logs/chain.log
