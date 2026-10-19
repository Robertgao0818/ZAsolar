#!/bin/bash
cd ~/capetown_eval/avg2_bench_v1_20261020; export HF_HUB_OFFLINE=1
./watch.sh & 
echo "$(date '+%F %T %Z') start" >> run.log; touch started
~/bin/gpu-run --wait --owner 光伏运行 --est 0.5 -- python3 bench.py > bench.log 2>&1; echo "$(date '+%F %T %Z') exit $?" >> run.log
