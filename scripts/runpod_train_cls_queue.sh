#!/usr/bin/env bash
set -euo pipefail

# Sequential classifier backbone training queue for a single-GPU RunPod pod.
# Expected usage from the pod:
#   cd /workspace/ZAsolar
#   nohup scripts/runpod_train_cls_queue.sh > logs/cls_backbone_queue.log 2>&1 &

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ZAsolar}"
DATASET_NAME="${DATASET_NAME:-cls_pv_thermal_v2}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/root/cls_data/$DATASET_NAME}"
TMPFS_DATA_DIR="${TMPFS_DATA_DIR:-/dev/shm/cls_data/$DATASET_NAME}"
RUNS="${RUNS:-convnext_tiny dinov2_vits14}"

cd "$PROJECT_ROOT"
mkdir -p logs checkpoints

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  "$PROJECT_ROOT/scripts/runpod_setup_cls_env.sh"
fi

if [[ ! -d "$TMPFS_DATA_DIR/train" || ! -d "$TMPFS_DATA_DIR/val" ]]; then
  echo "[queue] Copying dataset to tmpfs: $TMPFS_DATA_DIR"
  rm -rf "$(dirname "$TMPFS_DATA_DIR")"
  mkdir -p "$(dirname "$TMPFS_DATA_DIR")"
  cp -a "$SOURCE_DATA_DIR" "$TMPFS_DATA_DIR"
fi

echo "[queue] Dataset: $TMPFS_DATA_DIR"
find "$TMPFS_DATA_DIR/train" -maxdepth 2 -type f | wc -l | awk '{print "[queue] train_files=" $1}'
find "$TMPFS_DATA_DIR/val" -maxdepth 2 -type f | wc -l | awk '{print "[queue] val_files=" $1}'

for arch in $RUNS; do
  case "$arch" in
    convnext_tiny)
      config="configs/classifier/convnext_tiny.json"
      run_id="${DATASET_NAME}_convnext_tiny"
      ;;
    dinov2_vits14)
      config="configs/classifier/dinov2_vits14.json"
      run_id="${DATASET_NAME}_dinov2_vits14"
      ;;
    resnet18)
      config=""
      run_id="${DATASET_NAME}_resnet18"
      ;;
    *)
      echo "[queue] ERROR: unsupported arch '$arch'" >&2
      exit 2
      ;;
  esac

  out_dir="$PROJECT_ROOT/checkpoints/$run_id"
  log="$PROJECT_ROOT/logs/$run_id.log"
  mkdir -p "$out_dir"

  if [[ -f "$out_dir/config.json" && -f "$out_dir/best_cls.pth" ]]; then
    echo "[queue] SKIP $run_id: existing completed artifacts found"
    continue
  fi

  resume_args=()
  if [[ -f "$out_dir/last_cls.pth" ]]; then
    resume_args=(--resume "$out_dir/last_cls.pth")
    echo "[queue] RESUME $run_id from $out_dir/last_cls.pth"
  fi

  echo "[queue] START $run_id $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -n "$config" ]]; then
    "$PROJECT_ROOT/.venv/bin/python" scripts/classifier/train_cls.py \
      --data-dir "$TMPFS_DATA_DIR" \
      --output-dir "$out_dir" \
      --config "$config" \
      "${resume_args[@]}" \
      2>&1 | tee "$log"
  else
    "$PROJECT_ROOT/.venv/bin/python" scripts/classifier/train_cls.py \
      --data-dir "$TMPFS_DATA_DIR" \
      --output-dir "$out_dir" \
      --arch "$arch" \
      "${resume_args[@]}" \
      2>&1 | tee "$log"
  fi
  echo "[queue] DONE $run_id $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[queue] ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
