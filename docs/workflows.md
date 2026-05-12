# Workflows — ZAsolar

> **V1.4 (2026-04-22)**: Success metric is grid-aggregate inventory accuracy
> under the four-channel validation framework. The `installation` evaluation
> profile is the default for per-polygon diagnostics. See
> [`validation_strategy.md`](validation_strategy.md) for what each channel
> certifies.

## Environment Setup

```bash
./scripts/bootstrap_env.sh         # First-time create / refresh ./.venv
source scripts/activate_env.sh     # Activate the env
./scripts/check_env.sh             # Verify deps, runtime dirs, CUDA
```

- Virtualenv pinned at `./.venv`
- `requirements.lock.txt` is the authoritative snapshot; `requirements.txt`
  is a looser dev-friendly list
- Runtime caches stay inside the repo: `.cache/`, `.config/`, `.local/`,
  `.tmp/`

## Inference + Evaluation

```bash
# Single grid, default model + canonical post-processing
python detect_and_evaluate.py --grid-id G1688 --force

# With a fine-tuned checkpoint
python detect_and_evaluate.py \
  --model-path checkpoints/exp003_C_targeted_hn/best_model.pth \
  --postproc-config configs/postproc/v4_canonical.json --force
```

`--evaluation-profile installation` (default) does pred-side many-to-one merge
matching against installation-level GT. `--evaluation-profile legacy_instance`
is preserved for historical comparisons.

## Fine-Tuning

```bash
# 1. Regenerate annotation manifest (after annotations change)
python scripts/annotations/bootstrap_manifest.py

# 2. Export COCO with benchmark holdouts excluded
HOLDOUT="G1240 G1243 G1244 G1245 G1293 G1294 G1297 G1298 G1299 G1300 \
        G1349 G1354 G1410 G1411 G1466 G1467 G1516 G1520 G1521 G1522 \
        G1523 G1524 G1569 G1570 G1571 G1572"
python export_coco_dataset.py --output-dir data/coco \
    --exclude-grids $HOLDOUT --neg-ratio 0.15 \
    --audit-csv results/analysis/gt_heater_audit/<run_id>/audit_labels_phase1.csv

# 3. Merge targeted hard negatives (batch 003 + batch 004)
python scripts/training/export_v4_1_hn.py \
    --base-coco data/coco --output-dir data/coco_hn

# 4. Train (CUDA required)
python train.py --coco-dir data/coco_hn --output-dir checkpoints

# 5. Evaluate the new checkpoint
python detect_and_evaluate.py \
    --model-path checkpoints/best_model.pth --force
```

## Batch Inference (multi-grid)

```bash
# Canonical entry — takes a grid list, parallel param, postproc config
bash scripts/analysis/batch_inference.sh
```

## Benchmark

Standardised post-training weight comparison, agent-first design (primary
output is `summary.json`).

```bash
# Default preset + default models
python scripts/analysis/run_benchmark.py

# Compare two models
python scripts/analysis/run_benchmark.py --models v3c v3_cleaned

# Add a fresh checkpoint to the comparison
python scripts/analysis/run_benchmark.py \
    --checkpoint checkpoints/exp005_v4_1_hn/best_model.pth --tag v4_1

# Smoke suite only (fast regression check)
python scripts/analysis/run_benchmark.py --suite cape_town_t1_smoke

# Re-aggregate without re-running inference
python scripts/analysis/run_benchmark.py --collect-only

# Parallel inference on RunPod (default 6 processes)
BENCHMARK_PARALLEL=6 python scripts/analysis/run_benchmark.py
```

- Config: `configs/benchmarks/post_train.yaml` + `configs/model_registry.yaml`
- Output: `results/benchmark/<run_id>/{summary.json, summary.md, by_suite.csv, by_grid.csv}`
- Per-grid artifacts: `results/<GridID>/benchmark_<run_id>_<tag>/`
- Verdict: improved / regressed / flat / mixed / failed, based on primary
  suite F1 delta
- All benchmark runs must use `configs/postproc/v4_canonical.json`
  (`post_conf=0.85`, tiered) for cross-experiment comparability

## GT Heater Audit (filter solar-thermal contamination from training GT)

Solar-thermal water heaters in the GT contaminate the PV vs heater
discrimination boundary. The audit isolates them from training exports
without touching the curated `data/annotations/` files.

```bash
# 1. Build audit queue + 400x400 chips
python scripts/analysis/build_gt_heater_audit.py
# Output: results/analysis/gt_heater_audit/<run_id>/
#   audit_queue_full.csv, audit_queue_phase1.csv
#   chips/rgb/, chips/overlay/

# 2. Generate the labelling HTML (open in browser)
python scripts/analysis/label_gt_heater_audit.py \
    --run-dir results/analysis/gt_heater_audit/<run_id>
# Shortcuts: 1=PV  2=heater/non-PV  3=uncertain  S=skip  B=back
# Export gt_heater_audit_labeled.csv from the page

# 3. Re-export training set with contaminated entries excluded
python export_coco_dataset.py \
    --audit-csv results/analysis/gt_heater_audit/<run_id>/gt_heater_audit_labeled.csv
```

- Phase 1 audits tier A only (~855 entries); extend to tier B/C if pollution
  remains high
- `--audit-csv` filtering happens after tier-filter but before tile assignment;
  benchmark GT is not affected

## Dataset Notes

- `export_coco_dataset.py` writes 400×400 geo-referenced chips,
  `train.json` / `val.json`, and a provenance CSV
- Scan-then-write strategy: scan all chip metadata first, balance-sample,
  then only write the selected chips (no orphaned files)
- Default 1:1 pos:neg balance; pass `--no-balance` to disable; negatives are
  unannotated chips used as hard negatives
- `--manifest` + `--tier-filter` to filter by annotation quality
- Targeted hard negatives from reviewer FP exports:
  `scripts/training/export_targeted_hn.py`

## RunPod Cloud Training

### Pod bootstrap (run once per pod)

```bash
# GIS deps that the base image lacks
pip install --break-system-packages \
    pycocotools opencv-python-headless geopandas rasterio \
    huggingface_hub matplotlib seaborn geoai-py rasterstats

# Ship the small files the pipeline needs
scp data/task_grid.gpkg root@<pod-ip>:/workspace/ZAsolar/data/
```

### Data staging (IO speed)

```bash
# Copy training set to RAM disk before training (~10x faster than network volume)
cp -r /workspace/coco/<set_name> /dev/shm/
python train.py --coco-dir /dev/shm/<set_name> --output-dir /workspace/checkpoints/<exp>
```

### Training (spot-safe)

```bash
# nohup against SSH disconnect
nohup python train.py --coco-dir /dev/shm/<set_name> \
    --output-dir /workspace/checkpoints/<exp> --batch-size 32 \
    > /workspace/train_log.txt 2>&1 &

# Spot preemption recovery (resume from latest checkpoint)
ls -t /workspace/checkpoints/<exp>/stage*_epoch*.pth | head -1
python train.py --coco-dir ... --output-dir ... --resume <checkpoint.pth>
```

### Batch inference

```bash
# Verify rasterstats is installed — required for confidence backfill
python -c "import rasterstats; print('OK')"

# Single grid
python detect_and_evaluate.py --grid-id G1293 \
    --model-path /workspace/checkpoints/<exp>/best_model.pth \
    --evaluation-profile installation --force
```

### Notes

- **rasterstats is critical**: geoai mask band 2 stores confidence and needs
  rasterstats to backfill. Missing it defaults `confidence=0.5`, which the
  `post_conf_threshold=0.7` filter drops to zero detections.
- `train.py` defaults: `batch_size=32`, `num_workers=8`, AMP on
  (tuned for RTX 5090 32 GB)
- Checkpoints saved per epoch, last 2 retained, with Stage 1/2 resume support
- Network volume persists across pod restarts; the container local disk does not
