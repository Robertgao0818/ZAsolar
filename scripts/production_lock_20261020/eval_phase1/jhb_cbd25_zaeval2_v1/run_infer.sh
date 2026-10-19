#!/bin/bash
# Task B GPU chain: convnext_t_s0 -> mit_b2_s0 -> v2c (sequential, resumable)
cd ~/capetown_eval/jhb_cbd25_zaeval2_v1_20261019; E=$HOME/capetown_eval/ds3m_ft
export HF_HUB_OFFLINE=1 BS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
r(){ echo "$(date '+%F %T %Z') start $1" >> logs/chain.log; python3 infer_save_v1.py "$@" > logs/infer_$1.log 2>&1; e=$?; echo "$(date '+%F %T %Z') $1 exit $e" >> logs/chain.log; return $e; }
ARCH=convnext_t SEED=0 r convnext_t_s0 p1 $E/phase1_runpod_20261017/convnext_t_s0/best_objf1.pt || exit 1
ARCH=mit_b2 SEED=0 r mit_b2_s0 p1 $E/phase1_runpod_20261017/mit_b2_s0/best_objf1.pt || exit 1
r v2c mit $E/evals/ft_best.pt || exit 1
echo "$(date '+%F %T %Z') infer chain done" >> logs/chain.log
