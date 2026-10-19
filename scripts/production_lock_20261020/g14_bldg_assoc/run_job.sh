#!/bin/bash
# one job under a hard 1.5 GB memory cap (systemd scope, no swap); 2 consecutive failures -> later jobs refuse to start
cd ~/capetown_eval/g14_bldg_assoc_v1_20261019; name=$1; shift
c=$(cat logs/consec_fail 2>/dev/null || echo 0); [ "$c" -ge 2 ] && { echo "$(date '+%F %T %Z') SKIP $name (2 consecutive failures)" >> logs/jobs.log; exit 99; }
echo "$(date '+%F %T %Z') start $name" >> logs/jobs.log
systemd-run --user --scope -q -p MemoryMax=1500M -p MemorySwapMax=0 /usr/bin/time -v env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 "$@" > logs/$name.log 2>&1; e=$?
rss=$(grep 'Maximum resident' logs/$name.log | awk '{print $NF}')
echo "$(date '+%F %T %Z') end $name exit $e maxrss_kb $rss" >> logs/jobs.log
if [ $e -eq 0 ]; then echo 0 > logs/consec_fail; else echo $((c+1)) > logs/consec_fail; fi
exit $e
