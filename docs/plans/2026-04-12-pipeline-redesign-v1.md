# ZAsolar Pipeline Redesign V1

> For Hermes: use subagent-driven-development if implementing this plan.

Goal: redesign the project pipeline so data semantics, training data construction, model training, post-processing, classifier filtering, and benchmark evaluation become explicit, reproducible, and independently swappable.

Architecture: split the current script-centric workflow into six versioned stages: data registry, curation/build, train, infer, review/refine, and benchmark/report. Keep V1.3 semantics intact: ground truth stays installation-level, reviewed predictions remain the pipeline output, and `installation` remains the default reporting frame. The redesign should decouple "training objective" from "reporting objective" and make hard-negative / classifier augmentation first-class dataset transforms instead of ad-hoc side scripts.

Tech stack: Python, YAML manifests, GeoTIFF/GPKG/COCO artifacts, Mask R-CNN, optional binary classifier, existing benchmark harness.

---

## 1. Problems with the current pipeline

1. Training data construction is spread across `export_coco_dataset.py`, `export_targeted_hn.py`, `export_v4_hn.py`, `export_v4_1_hn.py`, and audit scripts.
2. The same project has multiple evaluation layers with different semantics:
   - training metric: chip-level COCO AP50
   - inference/reporting metric: installation profile metrics
   - benchmark verdict: suite-level F1 deltas
3. Hard-negative logic exists as point solutions instead of a unified dataset transform system.
4. The emerging classifier path (`scripts/classifier/*.py`) is adjacent to the main pipeline instead of a defined stage.
5. Provenance exists, but there is no single run manifest tying together:
   - source annotations
   - tier/audit filters
   - negative policy
   - HN sources
   - classifier model
   - postproc config
   - benchmark preset
6. The repo is script-first rather than pipeline-first, which makes reproducibility and iteration harder as experiments branch.

---

## 2. Proposed top-level pipeline

### Stage A: Dataset registry and source resolution

Inputs:
- cleaned annotation GPKGs
- annotation manifest
- GT heater audit CSV
- review exports
- tile roots
- benchmark holdout definitions

Outputs:
- a single dataset-spec YAML describing the exact training/eval source universe

Proposed artifact:
- `configs/pipelines/datasets/<dataset_id>.yaml`

Example fields:
- dataset_id
- regions / grids included
- excluded grids
- annotation source pattern
- quality tier policy
- audit filter policy
- tile root resolution policy
- split seed
- split strategy

Rationale:
move all source-selection rules out of exporter code and into a versioned config.

### Stage B: Curation / build graph

This stage materializes training-ready assets from the dataset spec.

Substeps:
1. Resolve annotations and tiles
2. Apply tier filter
3. Apply audit filter
4. Build tile split manifest
5. Build chip manifest
6. Materialize selected chips
7. Apply negative policy
8. Apply hard-negative policy
9. Emit COCO + provenance + build manifest

Proposed artifact family:
- `artifacts/datasets/<dataset_build_id>/build_manifest.json`
- `artifacts/datasets/<dataset_build_id>/train.json`
- `artifacts/datasets/<dataset_build_id>/val.json`
- `artifacts/datasets/<dataset_build_id>/train_provenance.csv`
- `artifacts/datasets/<dataset_build_id>/val_provenance.csv`
- `artifacts/datasets/<dataset_build_id>/dataset_summary.json`

Key redesign change:
replace one-off exporters with a unified "dataset builder" that supports pluggable transforms.

### Stage C: Train detector

Inputs:
- detector training spec
- dataset build id
- pretrained checkpoint

Outputs:
- detector checkpoint bundle
- training history
- training summary

Proposed artifact family:
- `artifacts/models/detector/<run_id>/best_model.pth`
- `artifacts/models/detector/<run_id>/final_model.pth`
- `artifacts/models/detector/<run_id>/training_history.json`
- `artifacts/models/detector/<run_id>/train_summary.json`
- `artifacts/models/detector/<run_id>/run_manifest.json`

Key redesign change:
training config should explicitly point at dataset build ids rather than raw directories.

### Stage D: Train auxiliary classifier

Inputs:
- classifier dataset build spec
- classifier architecture / loss / augmentation config

Outputs:
- binary classifier bundle for `pv` vs `non_pv` or `pv` vs `thermal`

Proposed artifact family:
- `artifacts/models/classifier/<run_id>/best_cls.pth`
- `artifacts/models/classifier/<run_id>/training_history.json`
- `artifacts/models/classifier/<run_id>/run_manifest.json`

Rationale:
classifier is no longer a side experiment; it becomes an explicit optional stage in the inference pipeline.

### Stage E: Inference and refinement

Inputs:
- detector model bundle
- optional classifier bundle
- postproc config
- target suite / grid list

Flow:
1. raw detector inference
2. geometric/vector postproc
3. optional classifier refinement on ambiguous small targets
4. reviewed-prediction export
5. metrics generation

Outputs:
- raw predictions
- refined predictions
- reviewed predictions
- grid-level metrics
- run manifest

Proposed artifact family:
- `artifacts/inference/<run_id>/<grid_id>/raw_predictions.gpkg`
- `artifacts/inference/<run_id>/<grid_id>/refined_predictions.gpkg`
- `artifacts/inference/<run_id>/<grid_id>/review_ready.gpkg`
- `artifacts/inference/<run_id>/<grid_id>/metrics.json`
- `artifacts/inference/<run_id>/<grid_id>/run_manifest.json`

Key redesign change:
separate detector postproc from classifier filtering so their effects can be measured independently.

### Stage F: Benchmark and release decision

Inputs:
- benchmark preset
- model registry or ad-hoc run ids
- postproc config
- inference manifests

Outputs:
- machine-readable benchmark verdict
- release recommendation

Proposed artifact family:
- `results/benchmark/<run_id>/summary.json`
- `results/benchmark/<run_id>/model_cards.json`
- `results/benchmark/<run_id>/decision.md`

Key redesign change:
benchmark should operate on model bundles + inference specs, not loosely on checkpoints alone.

---

## 3. Unified dataset transform system

The most important redesign is to treat dataset construction as a pipeline with explicit transform modules.

### Required transform types

1. `tier_filter`
   - filters annotations by T1/T2 policy
2. `audit_filter`
   - removes GT rows flagged heater / uncertain
3. `empty_negative_sampler`
   - controls easy negative ratio
4. `reviewed_fp_hn`
   - generates HN from reviewed predictions with GT-overlap protection
5. `small_fp_hn`
   - generates HN from curated taxonomy shortlist
6. `classifier_crop_export`
   - generates classification chips for PV-vs-thermal training
7. `holdout_excluder`
   - excludes benchmark grids from training builds

### Why this matters

Today, V3 / V4 / V4.1 are effectively different compositions of transforms, but that composition is encoded across separate scripts. In the redesigned pipeline, each experiment becomes a declarative spec such as:

```yaml
transforms:
  - type: holdout_excluder
    grids: [cape_town_independent_26]
  - type: tier_filter
    policy: T1+T2
  - type: audit_filter
    csv: results/analysis/gt_heater_audit/.../gt_heater_audit_labeled.csv
    exclude_labels: [heater_or_non_pv, uncertain]
  - type: empty_negative_sampler
    neg_ratio: 0.15
  - type: reviewed_fp_hn
    source_batch: batch003
    max_ratio: 0.10
  - type: small_fp_hn
    shortlist: results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv
    sample_rate: 0.5
    max_ratio: 0.04
```

This makes V3, V4, V4.1, and future variants reproducible and directly comparable.

---

## 4. Separate objectives explicitly

The redesign should document three separate objectives and stop mixing them implicitly.

### Objective 1: Detector training objective
- chip-level segmentation learning
- selected by validation AP50 or another training proxy

### Objective 2: Refined prediction objective
- suppress thermal/skylight/shadow false positives after detector output
- measured via fixed-threshold precision/recall on inference runs

### Objective 3: Product/reporting objective
- reviewed prediction footprints evaluated against installation GT
- benchmark verdict based on primary suite F1 deltas

Required rule:
Every model bundle and report must state which objective it optimizes and which metric selected it.

---

## 5. Recommended new code layout

### New directories
- `pipeline/`
  - `dataset_builder.py`
  - `transforms/`
  - `model_train.py`
  - `classifier_train.py`
  - `inference_runner.py`
  - `benchmark_runner.py`
  - `manifests.py`
- `configs/pipelines/`
  - `datasets/`
  - `detectors/`
  - `classifiers/`
  - `inference/`
  - `releases/`
- `artifacts/`
  - `datasets/`
  - `models/`
  - `inference/`

### Legacy compatibility
Keep existing entrypoints as thin wrappers initially:
- `export_coco_dataset.py` -> calls dataset builder with an autogenerated spec
- `train.py` -> calls detector trainer with a resolved train spec
- `scripts/training/export_targeted_hn.py` -> wrapper over `reviewed_fp_hn` transform
- `scripts/training/export_v4_hn.py` -> wrapper over `small_fp_hn` transform
- `scripts/analysis/run_benchmark.py` -> wrapper over benchmark runner

This preserves current workflows while enabling gradual migration.

---

## 6. Concrete recommended pipeline for the next iteration

If the team wants the most practical near-term redesign, I recommend this exact chain:

1. Build `dataset_builder.py`
   - absorb current `export_coco_dataset.py`
   - add transform hooks
2. Add `reviewed_fp_hn` transform
   - absorb `export_targeted_hn.py`
3. Add `small_fp_hn` transform
   - absorb `export_v4_hn.py` and `export_v4_1_hn.py`
4. Add `audit_filter` transform
   - make heater filtering first-class
5. Add `dataset_summary.json`
   - record pos/easy-neg/targeted-HN counts and ratios
6. Modify `train.py`
   - accept `--dataset-build` or spec path
   - emit `run_manifest.json`
7. Add optional classifier refinement stage to inference
   - only applied to selected small-target candidates
8. Benchmark detector-only and detector+classifier separately
   - same benchmark preset, same postproc config, two branches

This sequence gets the project out of ad-hoc script composition without forcing a disruptive rewrite.

---

## 7. Migration plan

### Phase 1: No behavior change, only structure
- create dataset/inference/train manifests
- keep current outputs and semantics
- validate that legacy scripts and new wrappers emit identical datasets/checkpoints/results

### Phase 2: Merge HN logic into dataset transforms
- retire separate HN merge scripts from day-to-day usage
- express V3, V4, V4.1 as declarative dataset recipes

### Phase 3: Insert classifier refinement
- define classifier candidate selection policy
- benchmark detector-only vs detector+classifier
- track impact on 5-20m² bucket specifically

### Phase 4: Release pipeline
- add promotion rules:
  - must beat incumbent on primary suite F1
  - no silent postproc drift
  - classifier branch must not regress recall beyond threshold

---

## 8. Key decisions I recommend locking now

1. Keep V1.3 semantics unchanged.
2. Treat dataset construction as a versioned build product.
3. Make hard negatives a transform, not a standalone dataset family.
4. Keep classifier as a post-detector refinement stage, not a replacement for the detector.
5. Keep benchmark as the release gate; do not promote based on train AP50.
6. Cap targeted HN ratio explicitly in config and report it in every dataset summary.
7. Always isolate benchmark holdout grids at dataset-build time, not by convention.

---

## 9. Proposed first implementation tasks

### Task 1: Add pipeline config schema
Create:
- `configs/pipelines/datasets/schema_example.yaml`
- `configs/pipelines/detectors/schema_example.yaml`

### Task 2: Extract current exporter into reusable builder primitives
Create:
- `pipeline/dataset_builder.py`
Modify:
- `export_coco_dataset.py`

### Task 3: Implement transform interface
Create:
- `pipeline/transforms/base.py`
- `pipeline/transforms/tier_filter.py`
- `pipeline/transforms/audit_filter.py`
- `pipeline/transforms/empty_negative_sampler.py`

### Task 4: Port reviewed FP HN
Create:
- `pipeline/transforms/reviewed_fp_hn.py`
Modify:
- `scripts/training/export_targeted_hn.py`

### Task 5: Port small-FP HN
Create:
- `pipeline/transforms/small_fp_hn.py`
Modify:
- `scripts/training/export_v4_hn.py`
- `scripts/training/export_v4_1_hn.py`

### Task 6: Emit build manifests
Create:
- `pipeline/manifests.py`
- `artifacts/datasets/<build_id>/dataset_summary.json`

### Task 7: Make detector trainer consume build manifests
Modify:
- `train.py`

### Task 8: Add inference manifest and optional classifier stage
Modify:
- `detect_and_evaluate.py`
- classifier inference scripts

### Task 9: Upgrade benchmark runner to understand bundles
Modify:
- `scripts/analysis/run_benchmark.py`
- `configs/model_registry.yaml`

---

## 10. Success criteria for redesign

The redesign is successful if:
1. A single YAML spec can reproduce V3-C, V4, and V4.1 dataset builds.
2. Every trained model can be traced back to one dataset build manifest.
3. Detector-only and detector+classifier results are benchmarked under identical settings.
4. HN ratio, easy-neg ratio, audit filter, and holdout exclusions are visible in one machine-readable file.
5. Legacy scripts still work during migration.

---

## 11. Immediate recommendation

Do not start by rewriting `detect_and_evaluate.py`.

Start with dataset build unification.

Reason: the project's biggest conceptual drift and experimental branching currently live in data construction (tiering, audit filtering, holdout, easy negatives, reviewed FP HN, small-FP HN). Once the build layer is declarative, the rest of the pipeline becomes much easier to reason about and benchmark.
