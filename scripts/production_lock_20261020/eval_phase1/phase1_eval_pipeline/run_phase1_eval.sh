#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${1:?Usage: run_phase1_eval.sh ARCH [SEED=0] [extra eval arguments]}"
SEED="${2:-0}"
if [[ $# -ge 2 ]]; then shift 2; else shift; fi
case "$ARCH" in mit_b2|mit_b3|segnext_b|convnext_t) ;; *) echo 'Invalid arch' >&2; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo 'Seed must be a nonnegative integer' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export HOME="${HOME:-/home/gao}"
PYTHON="${PYTHON:-python}"
CKPT="${CKPT_ROOT:-/workspace/ct_phase1_20261017/runs}/${ARCH}_s${SEED}/best_objf1.pt"
OUT="${OUT_ROOT:-$HOME/capetown_eval/ds3m_ft/phase1_runpod_20261017/eval}/${ARCH}_s${SEED}"
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
# Do not allow trailing arguments to replace the reviewed checkpoint or output paths.
for arg in "$@"; do
    case "$arg" in --out|--out=*|--ckpt|--ckpt=*|--arch|--arch=*|--split|--split=*) echo "Reserved option: $arg" >&2; exit 2 ;; esac
done
"$PYTHON" -B "$SCRIPT_DIR/eval_phase1.py" --ckpt "$CKPT" --arch "$ARCH" --out "$OUT" --split both --device cpu "$@"
"$PYTHON" -B "$SCRIPT_DIR/render_overlays.py" --eval-dir "$OUT" --out "$OUT/overlays"
