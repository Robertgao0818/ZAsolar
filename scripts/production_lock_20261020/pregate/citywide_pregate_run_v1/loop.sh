#!/bin/bash
# Task C: wait for Task B chain, 200-tile speed test, then rescan every 30 min. Each pass via gpu-run.
cd ~/capetown_eval/citywide_pregate_run_v1_20261019; export HF_HUB_OFFLINE=1
until grep -q 'score done' ~/capetown_eval/jhb_cbd25_zaeval2_v1_20261019/logs/chain.log 2>/dev/null; do echo "$(date '+%F %T %Z') waiting for Task B" > state.txt; sleep 120; done
if ! grep -q 'LIM=200' timing_v1.txt 2>/dev/null; then
 echo "$(date '+%F %T %Z') speedtest start" >> loop.log
 LIM=200 ~/bin/gpu-run --wait --owner 光伏运行 --est 0.1 -- python3 score_city.py >> score.log 2>&1
fi
while true; do
 echo "$(date '+%F %T %Z') pass start" >> loop.log; echo "$(date '+%F %T %Z') scoring pass" > state.txt
 ~/bin/gpu-run --wait --owner 光伏运行 --est 2 -- python3 score_city.py >> score.log 2>&1
 echo "$(date '+%F %T %Z') pass end exit $? scored=$(($(wc -l < tile_scores_v1.csv)-1))" >> loop.log
 echo "$(date '+%F %T %Z') sleeping 30 min" > state.txt; sleep 1800
done
