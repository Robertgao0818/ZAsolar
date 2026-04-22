# ZAsolar Pipeline Redesign V1.2

> For Hermes: use subagent-driven-development if implementing this plan.

Goal: land a narrow, executable Phase 1 that solves the immediate multi-region dataset-build bottleneck while preserving the non-negotiable V1.1 guarantees around semantics, validation, provenance, and reproducibility.

Architecture: keep the current exporter and HN scripts as the execution core, add a unified annotation discovery/loading layer on top of the existing region registry, and wrap that with a strictly validated declarative dataset-builder. Phase 1 is successful only if it reproduces current dataset recipes without practical behavior drift and without introducing a parallel source of truth.

Tech stack: Python, YAML + strict schema validation, GeoPandas/GPKG/GeoTIFF/COCO artifacts, existing `export_coco_dataset.py` and HN scripts, `configs/datasets/regions.yaml`, build manifests with source hashes.

---

## 1. Non-negotiable constraints

1. Preserve V1.3 semantics exactly.
   - GT remains installation-level.
   - Pipeline output remains reviewed prediction footprints.
   - `installation` remains the default reporting frame.
2. Do not break the current `results/<GridID>/` tree in Phase 1.
3. Do not require a flag-day rewrite of `detect_and_evaluate.py`, `train.py`, or `scripts/analysis/run_benchmark.py`.
4. Benchmark holdout exclusion must happen mechanically at dataset-build time.
5. Easy-negative ratio and hard-negative ratio must become explicit, machine-readable dataset properties.
6. Do not create a parallel authority for region/grid/annotation-path registration when `configs/datasets/regions.yaml` already owns that concept.
7. A declarative builder without strict validation and provenance does not count as a successful redesign.

---

## 2. Revised Phase 1 objective

The immediate problem is twofold:

1. The current dataset export path is fragmented across:
   - `export_coco_dataset.py`
   - `scripts/training/export_targeted_hn.py`
   - `scripts/training/export_v4_hn.py`
   - `scripts/training/export_v4_1_hn.py`
2. The current implementation is still Cape-Town-centric in several critical places, while the project now needs multi-region dataset construction.

Phase 1 therefore has one concrete proof target:

Build a strictly validated declarative dataset-builder that:
- uses the existing region registry as its source of truth,
- supports multi-region annotation discovery/loading,
- upgrades the current exporter/HN path to be region-aware,
- and reproduces the current complex dataset recipes with semantic equivalence.

This is intentionally narrower than a full pipeline rewrite.

---

## 3. What changed from V1.1

V1.2 keeps the quality bar from V1.1, but changes the execution order.

Accepted from the Claude review:
- Start with a unified `annotation_loader` grounded in `regions.yaml`.
- Upgrade `export_coco_dataset.py` to be multi-region instead of rebuilding export logic first.
- Make the HN scripts region-aware before trying to absorb all of them into a full transform framework.
- Keep the dataset builder thin in Phase 1; it should orchestrate existing working code, not replace the chip extraction core.
- Use semantic equivalence as the primary acceptance gate; byte-equivalence is optional.

Still mandatory from V1.1:
- Strict schema validation from day one.
- Deterministic build IDs.
- Input pinning and source-file hashing.
- `build_manifest.json` and `dataset_summary.json` as core deliverables, not optional add-ons.
- No silent duplication of source-of-truth registries.

---

## 4. Phase 1 architecture

### 4.1 Layered design

Use a three-layer Phase 1 architecture:

1. Layer 1: annotation discovery/loading
   - New file: `core/annotation_loader.py`
   - Responsibility: discover registered annotation sources via `core/region_registry.py`, classify schema variants, and load normalized GeoDataFrames.

2. Layer 2: region-aware dataset export
   - Modified file: `export_coco_dataset.py`
   - Responsibility: reuse current chip extraction / splitting / COCO assembly logic, but replace Cape-Town-only source discovery and tile-path assumptions with region-aware resolution.

3. Layer 3: declarative dataset-builder
   - New files under `pipeline/`
   - Responsibility: load a validated dataset spec, map it onto exporter/HN steps, write reproducible manifests, and enforce dataset-build invariants.

### 4.2 Explicit non-goal

Do not implement a broad pipeline framework first.

In particular, Phase 1 does not require:
- inference tree migration,
- `detect_and_evaluate.py` decomposition,
- benchmark runner redesign,
- classifier integration,
- or a large repo-wide artifact layout refactor.

---

## 5. Source-of-truth policy

This is a hard design rule for V1.2.

1. `configs/datasets/regions.yaml` remains the authority for:
   - region keys,
   - grid membership,
   - annotation source paths,
   - optional annotation layer hints,
   - region-scoped metadata.

2. `annotation_manifest.csv` remains the row-level executable authority for annotation tier/review filtering.

3. `configs/datasets/training_sets.yaml` may continue to describe named training-set families / historical datasets, but must not become a duplicate path registry.

4. Dataset specs under `configs/pipelines/datasets/` describe recipes, not physical source registration.

If a new config duplicates path ownership already present in `regions.yaml`, that is a design failure and should be removed.

---

## 6. Required implementation scope

### 6.1 New module: `core/annotation_loader.py`

Purpose: provide one entrypoint for discovering and loading annotations across all registered regions without hardcoded directory globs.

Required responsibilities:
- enumerate regions from `list_regions()` / `get_region_config()`;
- iterate registered grids from `regions.yaml`;
- resolve `annotation_source` using existing registry helpers;
- optionally use `annotation_layer` if provided;
- classify schema/source variants for reporting purposes only;
- load one annotation file into a normalized GeoDataFrame;
- drop empty/invalid geometries;
- normalize CRS handling;
- return at minimum geometry plus essential provenance fields.

Suggested dataclass:

```python
@dataclass
class AnnotationEntry:
    grid_id: str
    region_key: str
    path: Path
    schema_type: str
    annotation_count: int | None
    annotation_layer: str | None = None
```

Important rules:
- This loader is allowed to normalize loading behavior.
- It is not allowed to redefine project semantics.
- Extra annotation columns may be preserved when convenient, but Phase 1 export logic only requires geometry + provenance.

### 6.2 Modify `export_coco_dataset.py`

Required changes:
- Replace `_discover_cleaned_sources()` with registry-based discovery via `core.annotation_loader`.
- Replace Cape-Town-only `load_annotations()` behavior with region-aware loading.
- Add `--regions` CLI support; default should remain explicit and documented.
- Replace `TILES_ROOT / grid_id` assumptions with `resolve_tiles_dir(grid_id, region=...)`.
- Ensure provenance output captures region information.
- Keep current chip extraction, tile assignment, balancing, and COCO assembly logic unless a bug forces targeted changes.

Phase 1 preference:
- preserve as much existing exporter behavior as possible;
- change only the discovery/pathing/provenance surfaces necessary for multi-region correctness.

### 6.3 Modify HN scripts to remove hardcoded Cape Town assumptions

Target files:
- `scripts/training/export_targeted_hn.py`
- `scripts/training/export_v4_hn.py`
- `scripts/training/export_v4_1_hn.py`

Required changes:
- eliminate hardcoded region/path assumptions where they prevent multi-region use;
- use registry/path helpers for results roots, GT lookup, and tile lookup;
- replace baked-in grid batches with explicit CLI/config inputs where practical;
- keep each script operational as a standalone entrypoint in Phase 1.

Important scope rule:
- Do not force these scripts into a full transform framework before the region-aware builder path is proven.

---

## 7. Declarative dataset spec and builder

### 7.1 Required files

Create at minimum:
- `pipeline/__init__.py`
- `pipeline/dataset_builder.py`
- `pipeline/specs.py`
- `pipeline/manifests.py`
- `configs/pipelines/datasets/`

Optional in Phase 1, only if genuinely helpful:
- `pipeline/build_context.py`
- `pipeline/transforms/registry.py`

### 7.2 Builder role

The builder should be thin.

It should:
1. load and validate a dataset spec;
2. resolve paths and runtime variables;
3. call existing exporter/HN logic through Python APIs where feasible;
4. write build outputs and manifests;
5. enforce reproducibility and acceptance checks.

It should not:
- reimplement chip scanning,
- reimplement COCO assembly,
- or invent a second path-resolution system.

### 7.3 Dataset spec shape

Dataset specs belong under:
- `configs/pipelines/datasets/`

Representative shape:

```yaml
schema_version: 1
name: v4_1_hn
build_family: detector_train
regions: [cape_town]

selection:
  exclude_grids_file: configs/pipelines/datasets/holdouts/cape_town_independent_26.yaml
  tier_filter: T1+T2
  audit_csv: results/analysis/gt_heater_audit/<run_id>/audit_labels_phase1.csv
  exclude_audit_labels: [heater_or_non_pv, uncertain]

chip:
  size: 400
  overlap: 0.25

split:
  strategy: tile_greedy_by_annotation_count
  val_fraction: 0.2
  seed: 42

negatives:
  easy_neg_ratio: 0.15

hard_negatives:
  - type: reviewed_fp_hn
    region: cape_town
    grids: [G1682, G1683]
    max_ratio: 0.10
  - type: small_fp_hn
    shortlist_csv: results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv
    sample_rate: 0.5
    max_ratio: 0.04

output:
  root: ${SOLAR_ARTIFACT_ROOT:-/mnt/d/ZAsolar}
  name_template: "coco_{name}_{date}"
```

Important rule:
- Specs describe the recipe and selection logic.
- Specs do not replace `regions.yaml` as the source of path truth.

---

## 8. Schema validation is mandatory

This requirement is unchanged from V1.1.

Implementation options:
- `pydantic`, or
- dataclasses + a strict validation layer.

Minimum required validation behavior:
1. Unknown keys fail loudly.
2. Enum-like fields are validated.
3. Ratios are bounded.
4. Required files must exist unless explicitly documented as deferred.
5. Holdout/include/exclude rules must not conflict silently.
6. HN transform types or operation types must be validated against a known registry/map.
7. Region values must map to registered regions.
8. Grid IDs, when explicitly listed, must be compatible with the selected region scope.

Design rule:
- “Simple dict validation for now” is not sufficient for Phase 1 acceptance.

---

## 9. Build identity, provenance, and reproducibility

### 9.1 Deterministic build ID

Every dataset build must have a deterministic human-readable build ID:
- `<spec_name>_<YYYYMMDD>_<short_hash>`

Where:
- `short_hash` is derived from the fully resolved spec content,
- environment-variable substitutions happen before hashing.

### 9.2 Required manifest outputs

Each build must write:
- `train.json`
- `val.json`
- `train_provenance.csv`
- `val_provenance.csv`
- `build_manifest.json`
- `dataset_summary.json`

### 9.3 `build_manifest.json` must include

At minimum:
- build ID;
- original spec path;
- fully resolved spec content;
- resolved spec hash;
- resolved absolute roots/paths;
- region scope;
- explicit include/exclude/holdout lists;
- source file inventory with SHA256:
  - annotation GPKGs,
  - manifest CSV,
  - audit CSV,
  - HN shortlist CSV,
  - any other non-code tabular inputs;
- split strategy + seed;
- HN/easy-neg configuration;
- code provenance:
  - git commit if available,
  - entrypoint / script paths.

### 9.4 `dataset_summary.json` must include

At minimum:
- positive chip count;
- easy negative chip count;
- reviewed-FP HN count;
- small-FP HN count;
- total train images;
- total val images;
- annotation counts;
- effective easy-neg ratio;
- effective targeted-HN ratio;
- filtered counts by reason:
  - tier filtered,
  - audit filtered,
  - holdout excluded,
  - missing source/tiles if applicable.

These outputs are part of the definition of done.

---

## 10. Transform system policy

V1.2 narrows the requirement from V1.1.

Required now:
- the builder must have an explicit, validated way to map declared HN / filtering operations to implementation code.

Not required now:
- a fully generalized transform package with one module per operation before proving the region-aware builder works.

Practical interpretation:
- a small function registry or operation map is acceptable;
- keeping some HN logic in existing scripts is acceptable;
- a large abstraction layer is not required in Phase 1.

This is a deliberate defer, not a rejection of the broader transform idea.

---

## 11. Acceptance criteria for Phase 1

Phase 1 succeeds only if all of the following are true.

### 11.1 Core acceptance gates

1. The new builder can produce a V4.1-equivalent dataset from a declarative spec.
2. The build uses registry-based source discovery rather than Cape-Town-only globs.
3. The resulting exporter/HN path is region-aware.
4. The build writes deterministic provenance outputs.
5. No parallel source-of-truth registry has been introduced.

### 11.2 Equivalence gate

Use semantic equivalence as the required gate.

Minimum equivalence checks:
- train image count;
- val image count;
- annotation count;
- positive/easy-neg/HN counts;
- provenance row counts;
- per-grid totals where relevant.

Byte-identical COCO output is welcome if achieved, but not required.

### 11.3 Recommended first proof target

Use V4.1 first because it exercises:
- holdout exclusion,
- easy negatives,
- reviewed FP HN,
- curated small-FP HN,
- and real-world complexity already present in the repo.

### 11.4 Additional Phase 1 completion target

After V4.1 equivalence is proven, write declarative specs for:
- V3-C,
- V4,
- V4.1,
- and one explicit multi-region dataset recipe.

---

## 12. Implementation order

### P0: foundations that cannot be deferred

1. Define dataset spec schema.
2. Implement strict validation.
3. Define deterministic build ID generation.
4. Define manifest and summary schemas.

### P1: region-aware annotation/export path

5. Add `core/annotation_loader.py`.
6. Upgrade `export_coco_dataset.py` to use registry-based annotation discovery/loading.
7. Make tile-path resolution region-aware.
8. Verify Cape-Town-only behavior does not regress.
9. Verify a Johannesburg-only export works.
10. Verify a multi-region export works.

### P2: HN path hardening

11. Upgrade `export_targeted_hn.py` to remove hardcoded results/GT/tile assumptions.
12. Upgrade `export_v4_hn.py` similarly.
13. Ensure `export_v4_1_hn.py` still works through inherited/shared fixes.

### P3: declarative builder landing

14. Implement `pipeline/dataset_builder.py`.
15. Implement manifest writing.
16. Add first dataset specs.
17. Prove V4.1 equivalence.

### P4: compatibility and extension

18. Add wrapper compatibility for legacy entrypoints where useful.
19. Add V3-C and V4 specs.
20. Add one multi-region training dataset spec.

---

## 13. Verification plan

### 13.1 Region-aware exporter verification

Required checks:

```bash
python export_coco_dataset.py --regions cape_town --output-dir /tmp/test_ct --neg-ratio 0.15
python export_coco_dataset.py --regions johannesburg --output-dir /tmp/test_jhb --neg-ratio 0.15
python export_coco_dataset.py --regions cape_town johannesburg --output-dir /tmp/test_both --neg-ratio 0.15
```

Expectations:
- Cape Town run matches current behavior closely enough to pass semantic regression checks.
- Johannesburg run completes with registry-based source and tile resolution.
- Multi-region run produces coherent provenance and counts.

### 13.2 HN verification

Representative check:

```bash
python scripts/training/export_targeted_hn.py --grids G1682 G1683 --base-coco /tmp/test_ct --output-dir /tmp/test_hn
```

Expectation:
- reviewed-FP HN chips resolve against the correct results and tile roots without hardcoded Cape Town paths.

### 13.3 V4.1 builder equivalence verification

Representative check:

```bash
python -m pipeline.dataset_builder --spec configs/pipelines/datasets/v4_1_hn.yaml --output-dir /tmp/test_v4_1
```

Compare against the current V4.1 baseline using a mechanical comparison script or check routine that verifies at least:
- image counts,
- annotation counts,
- positive/easy-neg/HN counts,
- provenance row counts,
- per-grid totals.

Important rule:
- a manual eyeball check alone is not sufficient as the only acceptance gate.

---

## 14. Explicit defers

These are not Phase 1 deliverables:
- moving inference outputs into a new artifact tree;
- large-scale decomposition of `detect_and_evaluate.py`;
- redesigning `run_benchmark.py`;
- classifier integration into benchmark flow;
- a broad transform framework beyond what is needed to validate the declared builder operations.

If Phase 1 fails to prove equivalence and reproducibility, these later roadmap items should not proceed.

---

## 15. Definition of done

V1.2 Phase 1 is done only when:
1. A strict, validated dataset spec can describe V4.1.
2. The builder produces a V4.1-equivalent dataset using the new region-aware path.
3. Source inputs are pinned with hashes.
4. Build IDs are deterministic.
5. `build_manifest.json` and `dataset_summary.json` are emitted.
6. Cape Town, Johannesburg, and multi-region export paths are all mechanically verified.
7. No semantic drift has been introduced relative to V1.3 task definition.

---

## 16. Immediate recommendation

Do not start by building a full transform framework.

Start by proving this narrower claim:

A registry-grounded, strictly validated, manifest-producing dataset builder can sit on top of the existing exporter/HN code, make the export path multi-region, and faithfully reproduce the current best dataset recipes.

If that proof succeeds, deeper training/inference redesign becomes much safer.
If it fails, the broader redesign should be reconsidered before more architecture is added.
