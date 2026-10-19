#!/bin/bash
# heat log every 5 s; kill bench python if CPU pkg >95°C for >60 s or GPU >85°C
cd ~/capetown_eval/avg2_bench_v1_20261020; hot=0
while true; do
 g=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits); c=$(sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4);print $4}')
 echo "$(date '+%F %T') gpu_temp,power,util=$g cpu_pkg=$c" >> heat.log
 gt=${g%%,*}; ci=${c%.*}; [ "$ci" -gt 95 ] && hot=$((hot+5)) || hot=0
 if [ "$gt" -gt 85 ] || [ "$hot" -gt 60 ]; then echo "$(date '+%F %T') HEAT STOP gpu=$gt cpu=$c hot_s=$hot" >> heat.log; pkill -f "python3 bench.py"; fi
 pgrep -f "python3 bench.py" >/dev/null || { [ -f started ] && { echo "$(date '+%F %T') bench gone, watcher exit" >> heat.log; exit 0; }; }
 sleep 5
done
