#!/bin/bash
# heat watch after rog crash 23:39; 30 s samples; if CPU pkg >=98C for >5 min (11 samples), SIGSTOP jhb_union python jobs.
L=~/capetown_eval/citywide_pregate_run_v1_20261019/gpu_watch_20261020.log; hot=0; N=${1:-40}
for i in $(seq 1 $N); do
 c=$(sensors 2>/dev/null | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}')
 echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm,clocks_event_reasons.active --format=csv,noheader) | cpu ${c}C | load $(cut -d' ' -f1 /proc/loadavg) | memavail $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)G" >> $L
 if [ "${c:-0}" -ge 98 ]; then hot=$((hot+1)); else hot=0; fi
 if [ $hot -ge 11 ]; then pk=$(pgrep -f 'score_(jhb|ct)_u1.py'); kill -STOP $pk; echo "$(date +%T) CPU>=98C >5min -> SIGSTOP jhb_union pids $pk (resume: kill -CONT $pk)" | tee -a $L >> ~/capetown_eval/jhb_union_v1_20261019/logs/run.log; hot=-1000; fi
 sleep 30
done
journalctl -k -b --no-pager | grep -iE 'xid|mce|hardware error|thermal.*(crit|trip)' | tail -5 >> $L; echo "$(date +%T) watch end" >> $L
