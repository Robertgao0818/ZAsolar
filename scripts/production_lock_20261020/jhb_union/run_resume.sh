#!/bin/bash
# jhb_union_v1 resume after rog reboot (2026-10-19 23:45): same as run.sh, but only jobs whose parts/<name>.pkl is missing.
# run.sh itself would clobber logs of finished jobs (and their scripts assert on existing pkl) -> not used.
cd ~/capetown_eval/jhb_union_v1_20261019
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
awk '{ if (system("test -e parts/" $3 ".pkl") != 0) print }' jobs.txt > jobs_resume.txt
echo "$(date '+%F %T %Z') RESTARTED after rog reboot: resume $(wc -l < jobs_resume.txt)/$(wc -l < jobs.txt) jobs" >> logs/run.log
cat jobs_resume.txt | xargs -P 3 -L 1 sh -c 'nice -n 10 python3 $0 $1 > logs/$2.log 2>&1 || echo FAIL $1 >> logs/run.log'
echo "$(date '+%F %T %Z') parts done $(ls parts | wc -l)" >> logs/run.log
nice -n 10 python3 aggregate_u1.py > logs/aggregate_u1.log 2>&1; echo "$(date '+%F %T %Z') aggregate exit $?" >> logs/run.log
