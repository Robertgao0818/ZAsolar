#!/bin/bash
cd ~/capetown_eval/joburg_cbd25_phase1_20261018; export BS=4 HF_HUB_OFFLINE=1
for r in "convnext_t 0" "mit_b2 0" "segnext_b 0" "mit_b2 1" "mit_b2 2"; do set -- $r; [ -f preds/$1_s$2/DONE ] && continue
 echo "$(date '+%F %T') start $1 s$2" >> chain.log; python3 infer_jhb25.py $1 $2 > logs_$1_s$2.log 2>&1; echo "$(date '+%F %T') $1 s$2 exit $?" >> chain.log; done
echo "$(date '+%F %T') ALL DONE" >> chain.log
