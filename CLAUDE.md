# CLAUDE.md

South Africa rooftop solar installation detection & evaluation pipeline. Uses geoai (Mask R-CNN ResNet50-FPN) to detect solar installations from aerial GeoTIFFs, evaluates against hand-labeled ground truth. Currently covers Cape Town and Johannesburg, targeting nationwide residential solar census. Supports fine-tuning on multi-city annotations.

**Task definition (V1.3)**: reviewed prediction footprint segmentation — model predictions reviewed and accepted by human annotators, exported as polygons. Ground-truth annotations follow installation-level rules (see `data/annotations/ANNOTATION_SPEC.md`), but the pipeline output is reviewed predictions, not installation-merged footprints.

**V1.4 pivot (2026-04-22, subrepo split landed 2026-05-05)**: success metric reframes from per-polygon F1 to an **aggregate-inventory-at-grid-level** goal suited for economic analysis. Per-polygon F1 becomes diagnostic. Validation moves to a four-channel framework (stratified precision, exhaustive recall, plausibility, opportunistic external) with the task grid as the primary aggregation unit. Install-date back-dating moved to a sibling repo **`solar_backdating`** at `/home/gaosh/projects/solar_backdating/` (plugin of this repo via shared venv + PYTHONPATH). Old `geid_bbox` prototype archived at `/home/gaosh/projects/_archive/geid_bbox_legacy_2026-05-05/`. Full spec in [`docs/validation_strategy.md`](docs/validation_strategy.md).

## Sibling subrepo: `solar_backdating`

Location: `/home/gaosh/projects/solar_backdating/` (sibling, **not** under this repo). Agents/harness primarily live here in main repo, but agents launched from here will frequently write code into `solar_backdating/` — treat that path as a first-class write target.

- Plugin runtime: subrepo shares this repo's `.venv`. From subrepo root, `source scripts/activate_env.sh` sources main repo's activator and prepends subrepo paths to `PYTHONPATH` (subrepo first, then `$ZASOLAR_ROOT`). Subrepo imports `core.region_registry`, `core.annotation_loader`, `core.grid_utils`, and reads `configs/datasets/regions.yaml` from this repo.
- Boundary: subrepo handles install-date / temporal / GEHistoricalImagery / GEID history. This repo handles aerial census, training, V1.3/V1.4 evaluation, classifier work. Do not duplicate temporal logic in this repo — `scripts/temporal/`, `scripts/validation/{probe_geid_vintages,parse_geid_probe_results,run_geid_vintage_probe}.*`, and `tests/temporal/` here are **frozen with deprecation headers**, scheduled for removal after 2026-05-31. New temporal work goes in `solar_backdating/`.
- Subrepo entry docs: `solar_backdating/{AGENTS.md, CLAUDE.md, README.md, SHARED_FROM_ZASOLAR.md}` define identity, runtime contract, and the dependency surface this repo exposes.
- GEHistoricalImagery (the GEID-replacement provider) plan + wrappers live in `solar_backdating/docs/gehi_temporal_replacement_plan.md` and `solar_backdating/scripts/temporal/gehi_*.py`. Do not add new temporal provider code to this repo.

## Key References

- Architecture and directory layout: [`docs/architecture.md`](docs/architecture.md)
- Workflows (inference, fine-tuning, analysis): [`docs/workflows.md`](docs/workflows.md)
- Validation strategy (V1.4 four-channel framework): [`docs/validation_strategy.md`](docs/validation_strategy.md)
- Repository rules (Git, directory governance): [`docs/governance/repo-rules.md`](docs/governance/repo-rules.md)
- Annotation specification (Two-Axis Model): [`data/annotations/ANNOTATION_SPEC.md`](data/annotations/ANNOTATION_SPEC.md)
- Region registry (authoritative): [`configs/datasets/regions.yaml`](configs/datasets/regions.yaml)
- Training set provenance: [`configs/datasets/training_sets.yaml`](configs/datasets/training_sets.yaml)
- Cross-review harness: [`.agents/harness/README.md`](.agents/harness/README.md)
- Subrepo (install-date back-dating): `/home/gaosh/projects/solar_backdating/` — see `solar_backdating/CLAUDE.md` and `solar_backdating/SHARED_FROM_ZASOLAR.md`

## Working Constraints

1. Preserve V1.3 reviewed-prediction-footprint semantics. Ground-truth annotations follow installation-level rules; evaluation uses the `installation` profile by default.
2. Do not silently switch evaluation profile between `installation` and `legacy_instance`; keep profile selection explicit.
3. `detect_and_evaluate.py` reuses prior outputs only when `results/<GridID>/config.json` matches current code/parameters. Use `--force` for intentional reruns.
4. Empty-target chips in exported COCO datasets are intentional hard negatives — do not drop unless explicitly requested.
5. Never commit large binary files (tiles, checkpoints, results) to git — see `docs/governance/repo-rules.md`.
6. **Temporal / install-date / GEID-history work goes in `/home/gaosh/projects/solar_backdating/`, not this repo.** Main-repo `scripts/temporal/`, `scripts/validation/{probe_geid_vintages,parse_geid_probe_results,run_geid_vintage_probe}.*`, and `tests/temporal/` are frozen with deprecation headers (removal after 2026-05-31). Bug fixes go to subrepo first. When an agent launched from this repo needs to modify temporal code, it should `cd /home/gaosh/projects/solar_backdating/` and edit there.

## Environment

- Virtualenv: `./.venv` (create via `./scripts/bootstrap_env.sh`)
- CUDA GPU required for detection and training; `./scripts/check_env.sh` verifies availability
- Training dependencies: `torch`, `torchvision`, `opencv-python-headless`, `huggingface_hub`, `pycocotools`
- **Large data in `~/zasolar_data/`** (WSL ext4, post-2026-04-26 migration from `/mnt/d/ZAsolar/`):
  - Tiles: `~/zasolar_data/tiles/<region>/<imagery_layer>/` (env: `SOLAR_TILES_ROOT=/home/gaosh/zasolar_data/tiles`)
  - COCO datasets: `~/zasolar_data/coco/coco_v4_*/`
  - Models / SAM weights: `~/zasolar_data/models/`
  - GEID raw mosaics: `~/zasolar_data/geid_raw/`
  - Inference results: `~/zasolar_data/results/`
  - Annotations inbox (QGIS handoff, NTFS-side): `/mnt/d/ZAsolar/annotations_inbox/`
  - Project `tiles/` and `results/` directories should NOT contain actual data — symlinks/migrations only

## Quick Commands

```bash
# Environment
./scripts/bootstrap_env.sh && source scripts/activate_env.sh

# Inference (needs GPU)
python detect_and_evaluate.py --model-path checkpoints/exp003_C_targeted_hn/best_model.pth --force
python detect_and_evaluate.py --postproc-config configs/postproc/v4_canonical.json --force

# Fine-tuning (needs GPU, exclude benchmark holdout)
python export_coco_dataset.py --output-dir data/coco --exclude-grids G1240 G1243 ... --neg-ratio 0.15
python scripts/training/export_v4_1_hn.py --base-coco data/coco --output-dir data/coco_hn
python train.py --coco-dir data/coco_hn --output-dir checkpoints

# Benchmark (V3-C is current best, primary suite = cape_town_independent_26)
python scripts/analysis/run_benchmark.py --models v3c v4_1

# RunPod pod management
bash scripts/runpod_pod.sh start|stop|status|ssh|init
```
