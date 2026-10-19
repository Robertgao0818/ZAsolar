#!/bin/bash
# jhb_union_v1: CPU only, 3 parallel per-grid workers (~0.5 GB each), then aggregate
cd ~/capetown_eval/jhb_union_v1_20261019
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
echo "$(date '+%F %T %Z') start" >> logs/run.log
cat jobs.txt | xargs -P 3 -L 1 sh -c 'nice -n 10 python3 $0 $1 > logs/$2.log 2>&1 || echo FAIL $1 >> logs/run.log'
echo "$(date '+%F %T %Z') parts done $(ls parts | wc -l)" >> logs/run.log
nice -n 10 python3 aggregate_u1.py > logs/aggregate_u1.log 2>&1; echo "$(date '+%F %T %Z') aggregate exit $?" >> logs/run.log
