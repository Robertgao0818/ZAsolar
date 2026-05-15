#!/usr/bin/env bash
# Three-model JHB CBD 25-grid comparison on Vexcel 2024:
#   {V3-C, train20_val5_hn, unified_reviewall_A} × {raw, SAM mask+box}
#
# Pipeline (per model):
#   Phase A  detect_direct.py + finalize.py (pixel-or merge, v4_canonical postproc)
#   Phase B  scripts/analysis/sam_refine_maskbox.py (mask+box prompt)
# Phase C (Tier-1 area_aggregate_eval against data/annotations_channel2_clean)
# is invoked once at the end for all 6 model_runs.
#
# Defaults assume RunPod /workspace layout; override with env vars.

set -euo pipefail

REPO="${REPO:-/workspace/ZAsolar}"
REGION="${REGION:-johannesburg}"
IMAGERY_LAYER="${IMAGERY_LAYER:-vexcel_2024}"
POSTPROC="${POSTPROC:-$REPO/configs/postproc/v4_canonical.json}"
PARALLEL="${PARALLEL:-2}"            # detect_direct.py is GPU-saturating
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SAM_PROMPT_MODE="${SAM_PROMPT_MODE:-mask_box}"
SAM_BATCH_SIZE="${SAM_BATCH_SIZE:-16}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs/3model_jhb_cbd25_20260514}"
RESULTS_BASE="${RESULTS_BASE:-$REPO/results/analysis/direct_maskrcnn_v1}"

GRIDS=(
  G0772 G0773 G0774 G0775 G0776
  G0814 G0815 G0816 G0817 G0818
  G0853 G0854 G0855 G0856 G0857
  G0888 G0889 G0890 G0891 G0892
  G0922 G0923 G0924 G0925 G0926
)

# model_run_id -> checkpoint path
MODEL_IDS=(
  "v3c_vexcel_2024_direct"
  "train20_val5_vexcel_2024_direct"
  "unified_reviewall_A_vexcel_2024_direct"
)
MODEL_PATHS=(
  "$REPO/checkpoints/exp003_C_targeted_hn/best_model.pth"
  "$REPO/checkpoints/train20_val5_hn_20260508_v3c/best_model.pth"
  "$REPO/checkpoints/exp_unified_reviewall_A/best_model.pth"
)

mkdir -p "$LOG_ROOT"
cd "$REPO"

run_phaseA_grid() {
  local g=$1
  local run_id=$2
  local model_path=$3
  local out_dir="$RESULTS_BASE/$REGION/$run_id/$g"
  local grid_log="$LOG_ROOT/$run_id/${g}.log"
  mkdir -p "$(dirname "$grid_log")"

  mkdir -p "$out_dir"
  echo "[$g] detect_direct.py ($run_id) -> $out_dir" >> "$grid_log"
  python3 -u "$REPO/detect_direct.py" \
    --grid-id "$g" \
    --region "$REGION" \
    --imagery-layer "$IMAGERY_LAYER" \
    --model-run "$run_id" \
    --model-path "$model_path" \
    --output-dir "$out_dir" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor 2 \
    --chip-size 400 \
    --overlap 0.25 \
    --detector-score-threshold 0.05 \
    --detections-per-img 300 \
    --mask-threshold 0.3 \
    --raw-mask-storage crop \
    --device cuda \
    --profile \
    >> "$grid_log" 2>&1

  echo "[$g] finalize.py (merge-mode=pixel-or)" >> "$grid_log"
  python3 -u "$REPO/finalize.py" \
    --input "$out_dir/raw_detections.pkl" \
    --output-dir "$out_dir" \
    --postproc-config "$POSTPROC" \
    --merge-mode pixel-or \
    >> "$grid_log" 2>&1
}

run_phaseB_model() {
  local run_id=$1
  local sam_log="$LOG_ROOT/$run_id/_sam_refine.log"
  mkdir -p "$(dirname "$sam_log")"
  local raw_root="$RESULTS_BASE/$REGION/$run_id"
  local sam_root="$RESULTS_BASE/$REGION/${run_id}_sam_maskbox"

  echo "[phase B] sam_refine for $run_id -> $sam_root"
  python3 -u "$REPO/scripts/analysis/sam_refine_maskbox.py" \
    --region "$REGION" \
    --grids "${GRIDS[@]}" \
    --src-results-root "$raw_root" \
    --imagery-layer "$IMAGERY_LAYER" \
    --output-root "$sam_root" \
    --prompt-mode "$SAM_PROMPT_MODE" \
    --sam-batch-size "$SAM_BATCH_SIZE" \
    --label "${run_id}_sam_maskbox" \
    > "$sam_log" 2>&1
}

# ── Per-model: Phase A (parallel over grids) + Phase B (one SAM pass) ────
SECONDS=0
for idx in "${!MODEL_IDS[@]}"; do
  RUN_ID="${MODEL_IDS[$idx]}"
  MODEL_PATH="${MODEL_PATHS[$idx]}"
  echo
  echo "############################################################"
  echo "## model_run = $RUN_ID"
  echo "## ckpt      = $MODEL_PATH"
  echo "## elapsed   = ${SECONDS}s"
  echo "############################################################"
  if [ ! -s "$MODEL_PATH" ]; then
    echo "[fatal] checkpoint missing: $MODEL_PATH" >&2
    exit 2
  fi

  echo "[phase A] detect+finalize × ${#GRIDS[@]} (PARALLEL=$PARALLEL)"
  running=0
  for g in "${GRIDS[@]}"; do
    run_phaseA_grid "$g" "$RUN_ID" "$MODEL_PATH" &
    running=$((running + 1))
    if [ $running -ge $PARALLEL ]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
  echo "[phase A done] $RUN_ID at ${SECONDS}s"

  run_phaseB_model "$RUN_ID"
  echo "[phase B done] $RUN_ID at ${SECONDS}s"
done

echo
echo "############################################################"
echo "## All 3 models finished Phase A+B in ${SECONDS}s"
echo "## Raw + SAM result dirs under: $RESULTS_BASE/$REGION/"
echo "############################################################"
ls -1 "$RESULTS_BASE/$REGION/" | grep -E '(_direct$|_direct_sam_maskbox$)' || true
