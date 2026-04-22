# ZAsolar Pipeline Redesign V1.1

> For Hermes: use subagent-driven-development if implementing this plan.

Goal: redesign the pipeline around a declarative, reproducible dataset-build layer first, while preserving all current V1.3 semantics and minimizing migration risk to existing training, inference, review, and benchmark workflows.

Architecture: Phase 1 is intentionally narrow. Do not attempt a full pipeline rewrite yet. Build a versioned dataset-builder with validated YAML specs, explicit transform composition, and provenance manifests. Keep current training and inference entrypoints in place, and make them consume richer metadata incrementally. Treat the rest of the redesign as a roadmap gated on proving dataset-build equivalence against existing V3/V4/V4.1 outputs.

Tech stack: Python, YAML + schema validation, GeoTIFF/GPKG/COCO artifacts, Mask R-CNN, optional binary classifier, existing benchmark harness.

---

## 1. Design constraints that must not change

1. Preserve V1.3 semantics exactly.
   - GT remains installation-level.
   - pipeline output remains reviewed prediction footprints.
   - `installation` remains the default reporting frame.
2. Do not break the current `results/<GridID>/` tree in Phase 1.
3. Do not require a flag-day rewrite of `detect_and_evaluate.py`, `train.py`, or `run_benchmark.py`.
4. Benchmark holdout exclusion must happen at dataset-build time, not by convention.
5. Hard-negative ratio and easy-negative ratio must become explicit, machine-readable dataset properties.

---

## 2. Revised problem statement

The highest-leverage problem is not the entire pipeline at once; it is the dataset-construction layer.

Today the training data path is fragmented across:
- `export_coco_dataset.py`
- `scripts/training/export_targeted_hn.py`
- `scripts/training/export_v4_hn.py`
- `scripts/training/export_v4_1_hn.py`
- manifest tier filtering
- GT heater audit filtering
- holdout exclusion
- negative balancing
- reviewed-FP HN extraction
- small-FP curated HN extraction

This means that V3, V4, and V4.1 are really different implicit dataset recipes encoded in separate scripts.

V1.1 therefore narrows Phase 1 to one objective:

Create a single declarative dataset builder that can reproduce the existing dataset variants without changing downstream behavior.

---

## 3. Scope of V1.1

### In scope now

1. Dataset spec YAML
2. Dataset spec schema validation
3. Unified dataset builder
4. Function-based transform registry
5. Build manifest + dataset summary
6. Byte-equivalence or semantic-equivalence validation against existing outputs
7. Thin wrapper compatibility for existing exporter scripts

### Explicitly out of scope for Phase 1

1. Moving inference outputs into a new `artifacts/inference/` tree
2. Large-scale decomposition of `detect_and_evaluate.py`
3. Rewriting the benchmark runner around new bundle types
4. Full classifier integration into benchmark flow
5. Global repo directory reshuffle

Those remain roadmap items after the dataset-builder is proven.

---

## 4. Recommended Phase 1 architecture

### 4.1 Dataset spec

Add versioned dataset specs under:
- `configs/pipelines/datasets/`

Each dataset spec should fully describe the build recipe, including:
- dataset id / spec version
- annotation source resolution policy
- included / excluded grids
- holdout exclusions
- tier policy
- audit filter policy
- split strategy
- easy-negative policy
- targeted hard-negative policy
- output path policy
- seed values

Example shape:

```yaml
schema_version: 1
name: v4_1_base
build_family: detector_train
region: cape_town

tile_root: ${SOLAR_TILES_ROOT}
output_root: ${SOLAR_ARTIFACT_ROOT:-./artifacts/datasets}

sources:
  annotation_glob: data/annotations/Capetown/*_SAM2_*.gpkg
  manifest_csv: data/annotations/annotation_manifest.csv
  audit_csv: results/analysis/gt_heater_audit/<run_id>/gt_heater_audit_labeled.csv

selection:
  exclude_grids_file: configs/pipelines/datasets/holdouts/cape_town_independent_26.yaml
  tier_filter: T1+T2
  exclude_audit_labels: [heater_or_non_pv, uncertain]

split:
  strategy: tile_greedy_by_annotation_count
  val_fraction: 0.2
  seed: 42

transforms:
  - type: empty_negative_sampler
    neg_ratio: 0.15
  - type: reviewed_fp_hn
    enabled: true
    source_batch: batch003
    max_ratio: 0.10
  - type: small_fp_hn
    enabled: true
    shortlist_csv: results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv
    sample_rate: 0.5
    max_ratio: 0.04
```

### 4.2 Schema validation

This is now mandatory, not optional.

Implement validated config loading with either:
- `pydantic`
or
- dataclasses + strict validation layer

Required validation checks:
- unknown keys fail loudly
- enum values are validated
- ratio values are bounded
- referenced files must exist unless explicitly marked deferred
- transform names must be registered
- holdout and include rules must not conflict silently

### 4.3 Build ids

Use a deterministic, human-readable build id:
- `<spec_name>_<YYYYMMDD>_<short_hash>`

Where:
- `short_hash` = first 8 chars of SHA256 of the fully resolved dataset spec
- environment-variable substitutions are applied before hashing

This gives:
- human readability
- stable identity
- content-addressable provenance

---

## 5. Function-based transform system

V1.1 removes the idea of a transform base class.

Use a plain registry instead.

Suggested structure:
- `pipeline/transforms/__init__.py`
- `pipeline/transforms/registry.py`
- `pipeline/transforms/tier_filter.py`
- `pipeline/transforms/audit_filter.py`
- `pipeline/transforms/empty_negative_sampler.py`
- `pipeline/transforms/reviewed_fp_hn.py`
- `pipeline/transforms/small_fp_hn.py`

Suggested interface:

```python
TransformFn = Callable[[BuildContext, dict], BuildContext]
TRANSFORMS: dict[str, TransformFn]
```

Where `BuildContext` is a lightweight dataclass that carries evolving state such as:
- resolved spec
- annotations by grid
- split manifests
- scanned chip manifests
- selected train/val image records
- annotation records
- provenance rows
- dataset counters / warnings

Why this is better than inheritance right now:
- less framework code
- easier to debug
- different transform types can operate on different context fields
- no fake abstraction pressure

---

## 6. Build manifests and provenance

### 6.1 Required build outputs

For each dataset build, write:
- `train.json`
- `val.json`
- `train_provenance.csv`
- `val_provenance.csv`
- `build_manifest.json`
- `dataset_summary.json`

Recommended location for Phase 1:
- keep dataset builds under a dedicated output root
- do not force a new inference result tree yet

### 6.2 build_manifest.json must include

1. resolved absolute paths
   - resolved tile root
   - resolved output root
2. spec content and resolved hash
3. source file inventory with SHA256
   - annotation GPKGs
   - manifest CSV
   - audit CSV if used
   - HN shortlist CSV if used
4. transform list with resolved parameters
5. grid inclusion / exclusion lists
6. split seed and split strategy
7. code provenance
   - git commit if available
   - script version / entrypoint path

### 6.3 dataset_summary.json must include

At minimum:
- positive chip count
- easy negative chip count
- reviewed FP HN count
- small-FP HN count
- total train images
- total val images
- annotation counts
- effective easy-neg ratio
- effective targeted-HN ratio
- filtered annotation counts by reason
  - tier filtered
  - audit filtered
  - holdout excluded

This summary is the machine-readable answer to:
- what exactly did this model train on?

---

## 7. Input pinning and reproducibility rules

A dataset build is not reproducible unless inputs are pinned.

V1.1 adds the following rules:

1. Every annotation source file included in a build must be recorded with SHA256.
2. Every non-code tabular input must be recorded with SHA256.
3. The fully resolved spec must be stored in the manifest.
4. If path resolution differs across environments, the resolved absolute values must be recorded.
5. If a source file hash changes, the builder should treat it as a different build even if the spec name is the same.

This is more important than adding a larger artifact hierarchy.

---

## 8. Path resolution policy

The current project runs across WSL, local disk, and RunPod. V1.1 therefore requires explicit path resolution policy in the dataset build layer.

Rules:

1. Paths may be declared using environment-variable templates.
2. The builder resolves them at runtime.
3. The resolved absolute paths are written to `build_manifest.json`.
4. COCO JSONs remain portable by using relative `file_name` fields for chips.
5. Build manifests capture enough environment detail to explain path mismatches later.

Suggested variables:
- `SOLAR_TILES_ROOT`
- `SOLAR_ARTIFACT_ROOT`
- `WORKSPACE`

---

## 9. Keep current result trees for now

V1.0 proposed a future `artifacts/inference/<run_id>/<grid_id>/` tree.

V1.1 explicitly defers that.

Phase 1 decision:
- keep inference outputs in the current `results/<GridID>/` structure
- if provenance needs to improve, add `run_manifest.json` sidecars there later
- do not introduce a parallel inference tree until the dataset-builder has been proven through at least one real experiment cycle

Rationale:
- existing review tooling depends on `results/<GridID>/`
- benchmark tooling depends on current conventions
- analysis scripts already consume current paths
- introducing a second result tree would create migration work before the core redesign is validated

---

## 10. Relationship to training, inference, benchmark, and classifier

### 10.1 Training

In Phase 1, `train.py` does not need a major rewrite.

Preferred incremental path:
- continue accepting `--coco-dir`
- later add optional `--dataset-build-manifest`
- eventually emit `run_manifest.json`

But this is not required to prove the dataset-builder architecture.

### 10.2 Inference

Do not redesign inference yet.

However, the plan should acknowledge reality:
- `detect_and_evaluate.py` is a large monolith
- future decomposition will be expensive
- Stage E remains roadmap, not Phase 1 implementation scope

### 10.3 Benchmark

Do not rewrite benchmark around model bundles yet.

Instead:
- preserve `scripts/analysis/run_benchmark.py`
- let Phase 1 focus on producing equivalent datasets for training
- later, add run manifests that benchmark can read opportunistically

### 10.4 Classifier

Classifier remains a roadmap-stage refinement path.

V1.1 does not yet define the benchmark injection mechanism for detector+classifier runs. That design must wait until:
1. classifier candidate selection policy is stable
2. dataset-builder work is proven
3. one full experiment cycle confirms the new build layer is reliable

---

## 11. Acceptance criteria for Phase 1

Phase 1 is successful only if the new dataset-builder can reproduce existing dataset variants with no practical behavior drift.

### Required acceptance checks

1. Build a V4.1-equivalent dataset from a declarative spec.
2. Compare against the current legacy output.
3. Pass one of the following equivalence gates:
   - byte-identical COCO JSON outputs
   - or, if ordering differs, semantic equivalence across:
     - image count
     - annotation count
     - per-split counts
     - positive/easy-neg/HN counts
     - provenance row counts
4. Build V3-C, V4, and V4.1 from specs using the same dataset-builder.
5. Confirm old scripts can remain as wrappers without changing user-facing behavior.

### Strong recommendation

Use V4.1 as the first equivalence target because it exercises the largest amount of current complexity:
- holdout
- easy negatives
- reviewed FP HN
- curated small-FP HN

---

## 12. Phased roadmap

### Phase 1: dataset-build unification only

Deliverables:
- validated dataset spec schema
- dataset builder
- function-registry transforms
- build manifest
- dataset summary
- legacy wrapper compatibility
- V3/V4/V4.1 equivalence checks

### Phase 2: richer training provenance

Deliverables:
- `train.py` run manifest
- model output manifest linking checkpoint -> dataset build id
- model registry extension to include dataset provenance

### Phase 3: richer inference provenance

Deliverables:
- `run_manifest.json` in existing `results/<GridID>/` outputs
- optional explicit inference spec files
- no tree migration yet unless justified by real pain

### Phase 4: classifier refinement integration

Deliverables:
- stable candidate selection policy
- detector-only vs detector+classifier benchmarkable branches
- small-target bucket reporting for classifier impact

### Phase 5: inference decomposition and possible tree migration

Deliverables:
- design for decomposing `detect_and_evaluate.py`
- only after Phase 1-4 have demonstrated value

---

## 13. Recommended first implementation order

### P0
1. Create dataset spec schema and validation layer.
2. Define build id generation rule.
3. Define build manifest format.

### P1
4. Extract reusable builder primitives from `export_coco_dataset.py`.
5. Implement a dataset-builder entrypoint.
6. Implement first transforms:
   - `holdout_excluder`
   - `tier_filter`
   - `empty_negative_sampler`

### P2
7. Port `audit_filter`.
8. Port `reviewed_fp_hn` from `export_targeted_hn.py`.

### P3
9. Port `small_fp_hn` from `export_v4_hn.py` / `export_v4_1_hn.py`.
10. Write V3-C / V4 / V4.1 dataset specs.
11. Run equivalence validation.

### P4
12. Add optional training run manifest output.
13. Consider adding result-side `run_manifest.json` without changing output tree layout.

---

## 14. Concrete recommended file layout for Phase 1

Create:
- `pipeline/dataset_builder.py`
- `pipeline/build_context.py`
- `pipeline/specs.py`
- `pipeline/manifests.py`
- `pipeline/transforms/registry.py`
- `pipeline/transforms/tier_filter.py`
- `pipeline/transforms/audit_filter.py`
- `pipeline/transforms/empty_negative_sampler.py`
- `pipeline/transforms/reviewed_fp_hn.py`
- `pipeline/transforms/small_fp_hn.py`
- `configs/pipelines/datasets/`

Modify later as wrappers:
- `export_coco_dataset.py`
- `scripts/training/export_targeted_hn.py`
- `scripts/training/export_v4_hn.py`
- `scripts/training/export_v4_1_hn.py`

Do not require changes yet to:
- `detect_and_evaluate.py`
- `scripts/analysis/run_benchmark.py`

---

## 15. Success criteria for the redesign overall

The redesign is on the right track if, after Phase 1:
1. V3-C, V4, and V4.1 can each be described by a declarative dataset spec.
2. Every dataset build has machine-readable provenance and source hashes.
3. Hard-negative composition is visible in one summary file.
4. Holdout exclusion is enforced mechanically.
5. Existing training and benchmark workflows still run.

Only after that should the project proceed to deeper training/inference refactors.

---

## 16. Immediate recommendation

Do not start by rebuilding the whole project architecture.

Start by proving one thing:

A declarative dataset builder can faithfully reproduce the current best and most complex dataset recipes.

If that proof fails, the broader redesign should be reconsidered.
If that proof succeeds, the rest of the roadmap becomes much safer to execute.
