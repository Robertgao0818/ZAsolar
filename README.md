# ZAsolar — Rooftop Solar Detection for South African Cities

[简体中文](README.zh.md)

ZAsolar is a research codebase for detecting residential rooftop photovoltaic
installations from high-resolution aerial imagery, with the goal of building a
grid-aggregate panel data record of solar adoption across South Africa.
Cape Town and Johannesburg are currently covered.

The detection stack is **Mask R-CNN (ResNet-50 + FPN) + SAM 2.1 mask-prompt
refinement**. The V1.4 validation framework treats grid-aggregate installation
inventory as the primary success metric, with per-polygon F1 retained as a
diagnostic. Install-date back-dating is handled by a sibling repo,
[`solar_backdating`](https://github.com/Robertgao0818/solar_backdating).

## Headline results

Primary benchmark: **Johannesburg CBD, 25 grids, Vexcel 2024 aerial (~6.7 cm GSD)**,
detector = V3-C-HN, post-proc = SAM 2.1 mask+box refinement.

| Channel | Metric | Result | Sample |
|---|---|---|---|
| Ch1 — stratified precision | P (V3-C, hit-table) | 0.749 [0.71, 0.78] | 25 grids × stratified roof samples |
| Ch3 — inventory accuracy | area F1 | 0.821 | JHB CBD 25-grid Vexcel |
| Ch3 — inventory accuracy | aggregate \|A\|/\|B\| | 0.992 | JHB CBD 25-grid Vexcel |

See [`docs/validation_strategy.md`](docs/validation_strategy.md) for the full
four-channel framework, what each channel does and does not certify, and the
known confounders (e.g. SSEG building geocoding mismatch, vintage gaps).

## Repo layout

```
core/                     shared modules (region_registry, postproc, models)
pipeline/                 declarative dataset builder (V1.2 spec)
configs/                  region / imagery / training / model registries
data/annotations/         Cape Town + Johannesburg ground truth (gitignored)
docs/                     architecture.md, validation_strategy.md, workflows.md
scripts/
  analysis/               benchmarks, audits, calibration sweeps
  imagery/                tile download / preview / VRT
  training/               COCO export, hard-negative export
  classifier/             PV-vs-thermal binary classifier pipeline
  annotations/            review GUI, SAM FN GUI, batch finalize
detect_and_evaluate.py    primary inference + eval entry
detect_direct.py          stage 1 of direct pipeline (raw detections)
finalize.py               stage 3: raw_detections -> predictions_metric.gpkg
train.py                  Mask R-CNN fine-tune
export_coco_dataset.py    annotations -> COCO instance-segmentation dataset
```

Full structure: [`docs/architecture.md`](docs/architecture.md).
Workflow command sequences: [`docs/workflows.md`](docs/workflows.md).

## Quick start

```bash
# Environment (creates ./.venv from requirements.lock.txt)
./scripts/bootstrap_env.sh && source scripts/activate_env.sh

# Verify CUDA GPU + GIS deps
./scripts/check_env.sh

# Inference + eval on one grid (CUDA required)
python detect_and_evaluate.py \
  --grid-id G1688 \
  --model-path checkpoints/exp003_C_targeted_hn/best_model.pth \
  --postproc-config configs/postproc/v4_canonical.json \
  --force

# Primary benchmark
python scripts/analysis/run_benchmark.py --suite jhb_cbd_25_vexcel
```

Large data (tiles, COCO datasets, model weights) lives outside the repo under
`~/zasolar_data/`. `configs/datasets/regions.yaml` is the authoritative
imagery-layer and model-run registry. Annotations sync via Dropbox; checkpoints
sync via RunPod S3.

## Validation framework (V1.4)

Four orthogonal channels:

1. **Stratified precision** — random stratified roof samples on the benchmark
   grids; certifies detection precision conditioned on roof type and target
   size.
2. **Exhaustive recall** — clean GT (full panel inventory) on a small grid set;
   measures recall against an installation-merged reference.
3. **Plausibility** — hex-aggregated detections vs admin-level installation
   counts (SSEG, kW calibration); used as a sanity check, not a benchmark.
4. **Opportunistic external** — comparison with third-party datasets (e.g.
   Li GT for Cape Town) where vintage and coverage permit.

Task grid is the primary aggregation unit. Per-polygon F1 is diagnostic only.
The Tier-1 metric system uses `area_aggregate_eval.py`
(`agg_F1` + `pgF1` + `bulk` + `sigma_Bw` + log-`sigma` + RMSE + `thru0_beta`
+ R²), with `sigma_Bw` and RMSE as primary arbiters and `bulk in [0.5, 2.0]`
as a sanity gate.

## Sibling repo — `solar_backdating`

Install-date back-dating (using historical Google Earth imagery for each
detected installation footprint) lives in a sibling repository:
[Robertgao0818/solar_backdating](https://github.com/Robertgao0818/solar_backdating).
It runs as a plugin of this repo — shares the `.venv` and imports
`core.region_registry`, `core.annotation_loader`, `core.grid_utils`. Any new
temporal / GEHistoricalImagery / install-date code goes there, not here.

## Citation

Paper in preparation. Please cite as:

> Tao Yu Chen et al. (2026). *Grid-aggregate rooftop photovoltaic detection
> for South African cities.* [Manuscript in preparation].

## License

Code: MIT.
Annotations and reviewed predictions: see
[`data/annotations/ANNOTATION_SPEC.md`](data/annotations/ANNOTATION_SPEC.md).
