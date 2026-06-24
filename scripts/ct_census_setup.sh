#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# CT census — Phase A: pod environment setup (run ONCE after pod is up).
#
# Assumes (runpod-ops standard):
#   - ZAsolar already `git clone`d at $ZAS (default /root/ZAsolar)
#   - scripts/runpod_init.sh has run so system torch+cu128 is present. If the
#     repo venv is missing, this creates it with --system-site-packages so it
#     inherits that torch. We only ADD lockfile deps idempotently; we NEVER let
#     pip replace torch (Blackwell rule).
#
# What this does:
#   1. clone the solar_cls sibling subrepo (cls FP-suppressor lives there)
#   2. create/activate venv + pip install requirements.lock.txt (cu128-pinned)
#   3. fetch the unified_A detector checkpoint from HuggingFace
#   4. verify the cls v2 checkpoint + sibling config.json are present
#      (260 MB → ship via `runpodctl`, NOT scp; see printed instructions)
#   5. verify the rclone Dropbox remote is configured on the pod
#   6. generate the 2083-cell census G-list from the committed crosswalk
#   7. end-to-end smoke test on ONE grid (download → detect → cls)
#
# Required env (export before running):
#   HF_DETECTOR_REPO   HuggingFace repo id holding unified_A best_model.pth
# Optional env (sane defaults below).
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail

ZAS=${ZAS:-/root/ZAsolar}
CLS=${CLS:-/root/solar_cls}
CLS_GIT=${CLS_GIT:-git@github.com:Robertgao0818/solar_cls.git}  # needs github ssh key on pod
HF_DETECTOR_REPO=${HF_DETECTOR_REPO:-botao0818/zasolar-unified-reviewall-A}  # public; override to change
HF_DETECTOR_FILE=${HF_DETECTOR_FILE:-best_model.pth}
DETECTOR_DIR="$ZAS/checkpoints/exp_unified_reviewall_A"
# LOCKED CT cls = adaptive_v1 (ckpt + matched threshold table). NOT the fixed-400
# cls_pv_thermal_v2_dinov2_vits14 — classify_predictions always extracts adaptive
# chips now, so the fixed-400 ckpt is a chip-geometry mismatch. See ct_census_run.sh.
CLS_DIR="$CLS/checkpoints/cls_pv_thermal_v2_dinov2_vits14_adaptive"
CLS_CKPT="$CLS_DIR/best_cls.pth"
CLS_THRESHOLDS="$CLS/configs/classifier/thresholds_v2_adaptive.json"
STATE=${STATE:-/root/census_state}
LOGS=${LOGS:-/root/census_logs}
TILES_DISK=${TILES_DISK:-/root/tiles_disk}
GLIST="$STATE/glist.txt"

mkdir -p "$STATE" "$LOGS" "$TILES_DISK"
echo "setup" > "$STATE/PHASE"

say(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die(){ printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*"; exit 1; }

# 1 ── solar_cls sibling ------------------------------------------------------
say "1/7 solar_cls sibling subrepo"
if [ ! -d "$CLS/.git" ]; then
  git clone "$CLS_GIT" "$CLS" || die "clone solar_cls failed (check CLS_GIT / auth)"
else
  echo "  already present: $CLS"
fi

# 2 ── venv + deps (lockfile only; torch untouched) --------------------------
say "2/7 venv + pip install requirements.lock.txt (cu128-pinned, torch left alone)"
cd "$ZAS" || die "no $ZAS"
if [ ! -x "$ZAS/.venv/bin/python" ]; then
  echo "  creating $ZAS/.venv with --system-site-packages (inherits RunPod torch/cu128)"
  python3 -m venv --system-site-packages "$ZAS/.venv" \
    || die "venv creation failed (install python3-venv or create $ZAS/.venv manually)"
fi
source scripts/activate_env.sh || die "venv activation failed (run runpod_init first)"
pip install -r requirements.lock.txt 2>&1 | tail -5 || die "lockfile install failed"
python -c "import torch;assert torch.cuda.is_available();print('  torch',torch.__version__,'CUDA ok',torch.cuda.get_device_name(0))" \
  || die "CUDA not available — do NOT continue (would run on CPU / 0 detections)"

# 3 ── detector checkpoint from HF -------------------------------------------
say "3/7 detector checkpoint from HuggingFace ($HF_DETECTOR_REPO)"
mkdir -p "$DETECTOR_DIR"
if [ ! -f "$DETECTOR_DIR/best_model.pth" ]; then
  huggingface-cli download "$HF_DETECTOR_REPO" "$HF_DETECTOR_FILE" \
    --local-dir "$DETECTOR_DIR" --local-dir-use-symlinks False \
    || die "HF download failed"
  # normalise filename if HF stored it under a different name
  [ -f "$DETECTOR_DIR/$HF_DETECTOR_FILE" ] && [ "$HF_DETECTOR_FILE" != "best_model.pth" ] \
    && mv "$DETECTOR_DIR/$HF_DETECTOR_FILE" "$DETECTOR_DIR/best_model.pth"
fi
ls -lh "$DETECTOR_DIR/best_model.pth" || die "detector ckpt missing after download"

# 4 ── cls checkpoint (must arrive via runpodctl) ----------------------------
say "4/7 cls v2 checkpoint"
if [ ! -f "$CLS_CKPT" ] || [ ! -f "$(dirname "$CLS_CKPT")/config.json" ]; then
  cat <<EOF
  cls ckpt or sibling config.json missing. It is gitignored (260 MB), so it
  does NOT come with the clone. Ship it P2P (NOT scp — scp silently truncates):

    # ON LOCAL (WSL) — ship the ADAPTIVE checkpoint (the locked CT winner):
    runpodctl send /home/gaosh/projects/solar_cls/checkpoints/cls_pv_thermal_v2_dinov2_vits14_adaptive/best_cls.pth
    scp -P \$RUNPOD_SSH_PORT \\
      /home/gaosh/projects/solar_cls/checkpoints/cls_pv_thermal_v2_dinov2_vits14_adaptive/config.json \\
      \$RUNPOD_SSH_HOST:$(dirname "$CLS_CKPT")/config.json
    # ON POD (paste the code runpodctl printed):
    mkdir -p $(dirname "$CLS_CKPT")
    cd $(dirname "$CLS_CKPT") && runpodctl receive <CODE>

  Then re-run this setup script.
EOF
  die "cls ckpt not ready"
fi
ls -lh "$CLS_CKPT"
[ -f "$CLS_THRESHOLDS" ] || die "cls thresholds table missing: $CLS_THRESHOLDS"

# 5 ── rclone Dropbox remote --------------------------------------------------
say "5/7 rclone Dropbox remote"
rclone listremotes 2>/dev/null | grep -qx "dropbox:" \
  || die "no 'dropbox:' rclone remote on pod. Copy your local config first:
    scp -P \$RUNPOD_SSH_PORT ~/.config/rclone/rclone.conf \$RUNPOD_SSH_HOST:/root/.config/rclone/rclone.conf"
echo "  dropbox: remote OK"

# 6 ── census grid list: CPT ids from the authoritative census grid -----------
# task_grid_cpt.gpkg IS the 2083-cell census grid; every CPT id has geometry
# there. download_tiles/detect/cls all address by CPT (tiles are written under
# the source dir resolve_tiles_dir expects — no CPT<->G bookkeeping here).
say "6/7 census grid list (CPT ids from data/task_grid_cpt.gpkg)"
python - "$GLIST" <<'PY'
import sys, geopandas as gpd
out = sys.argv[1]
g = gpd.read_file("data/task_grid_cpt.gpkg")
ids = sorted(g["gridcell_id"].astype(str))
open(out, "w").write("\n".join(ids) + "\n")
print(f"  {len(ids)} CPT cells -> {out}  (first={ids[0]} last={ids[-1]})")
PY

# 7 ── smoke test on one grid (full chain) -----------------------------------
say "7/7 smoke test (download -> detect -> cls) on first grid"
SG=$(head -1 "$GLIST")
RUN=${RUN:-unifiedA_census_perdet}
RESULTS_DIR=$(python -c "from core.grid_utils import get_results_root;print(get_results_root('ct',model_run='$RUN'))")
echo "  results dir resolves to: $RESULTS_DIR"

SOLAR_TILES_ROOT="$TILES_DISK" python scripts/imagery/download_tiles.py --grid-id "$SG" --region ct \
  || die "smoke download failed"
# Tiles are written under the SOURCE grid dir (CPT####->G####); detect_direct now
# reads that same source dir. Assert they actually landed there — a CPT/G addressing
# mismatch would otherwise surface downstream as a silent 0-detection grid.
SRC=$(python -c "from core import region_registry as r;print(r.resolve_source_grid_id('$SG','cape_town'))")
ntiles=$(ls "$TILES_DISK/$SRC/${SG}_"*_geo.tif 2>/dev/null | wc -l)
echo "  tiles for $SG: $ntiles files under source dir $TILES_DISK/$SRC ($(du -sh "$TILES_DISK/$SRC" 2>/dev/null | cut -f1))"
[ "$ntiles" -gt 0 ] || die "no source tiles under $TILES_DISK/$SRC for $SG — download/detect tile-addressing mismatch (CPT->G)"
[ -z "$(ls -d "$TILES_DISK"/CPT* 2>/dev/null)" ] || die "stale CPT-named tile dirs under $TILES_DISK (tiles must live under G dirs) — remove them first"

# ENGINE = direct Mask R-CNN (detect_direct -> finalize per-detection), same as
# ct_census_run.sh infer_one and the LOCKED unifiedA_li_perdet baseline. NOT geoai.
# --detections-per-img 300 is detect_direct's direct-mode default, passed explicitly
# (matches run_benchmark.py's baseline; NOT 100, which is the geoai-parity value).
SOLAR_TILES_ROOT="$TILES_DISK" python detect_direct.py \
  --grid-id "$SG" --region ct --imagery-layer aerial_2025 \
  --model-run "$RUN" --model-path "$DETECTOR_DIR/best_model.pth" --output-dir "$RESULTS_DIR/$SG" \
  --detections-per-img 300 \
  --chip-size 400 --overlap 0.25 --mask-threshold 0.3 \
  --batch-size 4 --num-workers 2 --prefetch-factor 2 --device cuda \
  || die "smoke detect_direct failed"
SOLAR_TILES_ROOT="$TILES_DISK" python finalize.py \
  --input "$RESULTS_DIR/$SG/raw_detections.pkl" --output-dir "$RESULTS_DIR/$SG" \
  --postproc-config configs/postproc/v4_canonical.json --merge-mode per-detection \
  --allow-overwrite-canonical \
  || die "smoke finalize failed"
[ -f "$RESULTS_DIR/$SG/predictions_metric.gpkg" ] || die "smoke detect produced no predictions_metric.gpkg"

# R3 positive control: the head -1 smoke ($SG) ACCEPTS 0 detections (valid for a
# coastal/sparse cell), so it gives no positive evidence the detector reads real
# pixels — a silent tile-misaddress would look identical. Default to a known dense
# CT GT cell and assert >0 detections end-to-end. Set SMOKE_DENSE_GRID=skip only
# for a deliberately minimal smoke.
DG=${SMOKE_DENSE_GRID:-CPT1238}
if [ "$DG" != "skip" ]; then
  say "positive-control smoke on dense grid $DG (asserting >0 detections)"
  SOLAR_TILES_ROOT="$TILES_DISK" python scripts/imagery/download_tiles.py --grid-id "$DG" --region ct \
    || die "dense smoke download failed"
  SOLAR_TILES_ROOT="$TILES_DISK" python detect_direct.py \
    --grid-id "$DG" --region ct --imagery-layer aerial_2025 \
    --model-run "$RUN" --model-path "$DETECTOR_DIR/best_model.pth" --output-dir "$RESULTS_DIR/$DG" \
    --detections-per-img 300 --chip-size 400 --overlap 0.25 --mask-threshold 0.3 \
    --batch-size 4 --num-workers 2 --prefetch-factor 2 --device cuda \
    || die "dense smoke detect failed"
  SOLAR_TILES_ROOT="$TILES_DISK" python finalize.py \
    --input "$RESULTS_DIR/$DG/raw_detections.pkl" --output-dir "$RESULTS_DIR/$DG" \
    --postproc-config configs/postproc/v4_canonical.json --merge-mode per-detection \
    --allow-overwrite-canonical || die "dense smoke finalize failed"
  nd=$(python -c "import geopandas as gpd;print(len(gpd.read_file('$RESULTS_DIR/$DG/predictions_metric.gpkg')))" 2>/dev/null || echo 0)
  [ "$nd" -gt 0 ] || die "dense positive-control FAILED: $DG produced 0 detections — tile addressing or model load is broken"
  echo "  positive control OK: $DG -> $nd detections"
else
  echo "  NOTE: SMOKE_DENSE_GRID=skip — skipped the >0-detection positive control."
  echo "        head -1 ($SG) smoke alone cannot distinguish a silent tile-misaddress from a genuinely-empty cell."
fi

cd "$CLS" && source scripts/activate_env.sh
SOLAR_TILES_ROOT="$TILES_DISK" ZASOLAR_ROOT="$ZAS" python scripts/classifier/classify_predictions.py \
  --grid-id "$SG" --region ct --imagery-layer aerial_2025 \
  --model-path "$CLS_CKPT" --thresholds-v2 "$CLS_THRESHOLDS" --classify-all --results-dir "$RESULTS_DIR" \
  || die "smoke cls failed"
# cls produces no filtered gpkg when detect finds 0 predictions for the grid — that
# is a valid outcome (sparse/coastal grids). Accept as PASS if cls exited 0.
# If predictions_metric.gpkg has detections but filtered gpkg is missing, that IS a bug.
n_dets=$(python -c "
import geopandas as gpd, sys
try:
    g = gpd.read_file('$RESULTS_DIR/$SG/predictions_metric.gpkg'); print(len(g))
except Exception: print(0)
" 2>/dev/null || echo 0)
if [ "$n_dets" -gt 0 ]; then
  [ -f "$RESULTS_DIR/$SG/predictions_metric_cls_filtered.gpkg" ] \
    || die "smoke cls produced no filtered gpkg (had $n_dets detections — real failure)"
else
  echo "  cls skip OK: $SG has 0 detections (coastal/sparse grid — chain works)"
fi

say "SMOKE TEST PASSED for $SG — full chain works. Estimate ~$(du -sh "$TILES_DISK/$SRC"|cut -f1)/grid x 2083."
echo "Next: launch the overnight run inside tmux ->  bash scripts/ct_census_run.sh"
echo "ready" > "$STATE/PHASE"
