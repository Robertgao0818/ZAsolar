#!/bin/bash
# usage: run1.sh NAME KIND CKPT [ARCH SEED] ; env TILEPAT optional. resumable via PART pickles; skips if json exists
cd ~/capetown_eval/joburg_cbd25_rescore_v2_20261019
export HF_HUB_OFFLINE=1 DEV=cuda BS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PGONLY=1 NWORK=${NWORK:-5} NTHR=${NTHR:-14} TILEDIR=../joburg25_vexcel/tiles005 GTDIR=$HOME/capetown_eval/ds3m_ft/joburg25_vexcel/clean_gt
N=$1; export ARCH=$4 SEED=$5 PART=$PWD/out/part_$N
[ -s out/$N.json ] && { echo skip $N; exit 0; }
echo "$(date '+%F %T') start $N" >> logs/chain.log
/usr/bin/time -v python3 evalobj_jhb25_p1cpu.py $2 $3 out/$N.json > logs/$N.log 2>&1; echo "$(date '+%F %T') $N exit $?" >> logs/chain.log
