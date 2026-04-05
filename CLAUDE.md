# CLAUDE.md

Cape Town rooftop solar installation detection & evaluation pipeline. Uses geoai (Mask R-CNN ResNet50-FPN) to detect solar installations from aerial GeoTIFFs, evaluates against hand-labeled ground truth (weak supervision). Supports fine-tuning on Cape Town annotations.

**Task definition (V1.3)**: reviewed prediction footprint segmentation — model predictions reviewed and accepted by human annotators, exported as polygons. Ground-truth annotations follow installation-level rules (see `data/annotations/ANNOTATION_SPEC.md`), but the pipeline output is reviewed predictions, not installation-merged footprints.

## Key References

- Architecture and directory layout: [`docs/architecture.md`](docs/architecture.md)
- Workflows (inference, fine-tuning, analysis): [`docs/workflows.md`](docs/workflows.md)
- Repository rules (Git, directory governance): [`docs/governance/repo-rules.md`](docs/governance/repo-rules.md)
- Annotation specification: [`data/annotations/ANNOTATION_SPEC.md`](data/annotations/ANNOTATION_SPEC.md)
- Dataset registry: [`configs/datasets/regions.yaml`](configs/datasets/regions.yaml)
- Cross-review harness: [`.agents/harness/README.md`](.agents/harness/README.md)

## Working Constraints

1. Preserve V1.3 reviewed-prediction-footprint semantics. Ground-truth annotations follow installation-level rules; evaluation uses the `installation` profile by default.
2. Do not silently switch evaluation profile between `installation` and `legacy_instance`; keep profile selection explicit.
3. `detect_and_evaluate.py` reuses prior outputs only when `results/<GridID>/config.json` matches current code/parameters. Use `--force` for intentional reruns.
4. Empty-target chips in exported COCO datasets are intentional hard negatives — do not drop unless explicitly requested.
5. Never commit large binary files (tiles, checkpoints, results) to git — see `docs/governance/repo-rules.md`.

## Environment

- Virtualenv: `./.venv` (create via `./scripts/bootstrap_env.sh`)
- CUDA GPU required for detection and training; `./scripts/check_env.sh` verifies availability
- Training dependencies: `torch`, `torchvision`, `opencv-python-headless`, `huggingface_hub`, `pycocotools`
- **Large data on D drive** (not in WSL project dir):
  - Tiles: `/mnt/d/ZAsolar/tiles/` (env: `SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles`)
  - COCO datasets: `/mnt/d/ZAsolar/coco_*/`
  - Project `tiles/` directory should NOT contain actual tile data

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
