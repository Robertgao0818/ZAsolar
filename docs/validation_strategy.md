# Validation Strategy — V1.4 (2026-04-22)

> **Scope**: replaces the implicit assumption that per-polygon F1 on a human-annotated holdout is the primary success metric. Defines a four-channel validation framework for building an **economically usable** installation inventory in a data-scarce country.

## Why this exists

The project's end product is a per-region rooftop solar installation inventory used for economic analysis (adoption rates, spatial inequity, temporal growth). For that use case:

- **Aggregate accuracy at the region level matters**; per-polygon F1 does not, because FP and FN partially cancel when counts are aggregated.
- **South Africa has no national registry** comparable to the utility-reported capacity data that DeepSolar used. JHB has a household survey; CT has partial admin data (installed-household address points, urban-only, roughly zero coverage in suburbs). Admin points are **not** on the panel and cannot be matched to detected polygons directly.
- Rigorous internal validation + sanity checks + opportunistic external agreement is the defensible package for peer review in this context. A single external correlation metric is **not feasible** and should not gate the roadmap.

## Primary aggregation unit: task grid (`G1xxx`)

All four channels aggregate at the **task grid** level (~1 km², already used for inference, RA sampling, and results storage). Rationale:

- Zero external dependency (no ward/SAL boundary shapefile needed).
- Matches existing pipeline semantics — `results/<region>/<model_run>/<GridID>/` is already grid-first.
- RA sampling already runs per grid.
- Boundary effects with external admin data are handled by joining admin points to grids, not by pulling in new polygon sets.

Ward / SAL / hex grid are **downstream re-aggregation concerns** for regression and spatial analysis; they are not needed for validation itself.

## Grid-type stratification

Each grid carries a stratum tag used to break down all four channels:

- `CBD` — downtown high-density, mixed commercial/residential
- `suburban` — formal residential (middle/high income)
- `township` — formal/informal residential (low income)
- `peri_urban` — edge developments, low density
- `rural` — agricultural / sparse settlement

Tagging workflow (to be implemented): manual tag per grid + optional StatsSA urbanization index cross-check. Stored in `configs/datasets/regions.yaml` per grid, or in a separate `configs/datasets/grid_strata.csv`.

## Four validation channels

### Channel 1 — Stratified RA precision audit (primary precision evidence)

**What**: random sample of N detected polygons per (region, grid_type), each adjudicated by an RA as true/false. Output: precision ± 95% CI by stratum.

**Why**: current per-polygon F1 is computed on grids where RAs annotated densely — biased toward urban high-density areas. A stratified audit gives honest precision per ward-type, including strata where dense annotation is infeasible.

**Deliverables**:
- `scripts/analysis/ra_precision_sample.py` — stratified sampler + adjudication queue builder
- `results/validation/ra_precision_<YYYYMMDD>.csv` — per-stratum precision with CI
- Report in `docs/experiments/exp_validation_v1_4.md`

### Channel 2 — Exhaustive small-AOI recall (primary recall evidence)

**What**: randomly sample M small AOIs (e.g. 50×50 m or 100×100 m) per (region, grid_type). RA performs **exhaustive** annotation (every visible solar install, not just easy ones) within each AOI. Compute recall = detected ∩ GT / GT inside the AOI.

**Why**: the current annotation workflow is "find panels in dense areas" — it does not produce landscape-level recall because annotators do not exhaustively sweep low-density regions. Exhaustive AOI audit gives recall that generalizes across strata.

**Deliverables**:
- `scripts/analysis/build_recall_aoi_queue.py` — AOI sampler
- RA tool for exhaustive annotation (likely extension of existing review GUI)
- `results/validation/recall_aoi_<YYYYMMDD>.csv`

### Channel 3 — Plausibility sanity checks (not GT, upper-bound guardrails)

**What**: derived ratios per grid that should fall inside plausible bounds based on known physical and behavioral constraints.

- `n_detected / total_roof_area_km²` — density ceiling (compare to Germany Mayer numbers, US DeepSolar density)
- `n_detected / solarizable_roof_area_m²` — adoption rate upper bound
- `mean(polygon_area) by grid` — expected 10–40 m² for residential; outlier grids flagged
- Outlier grids (top/bottom 5% per stratum) get RA spot-check

**Why**: catches systematic failure modes (training distribution shift, post-processing break) even when there is no GT. Provides the "I didn't hallucinate half a city" evidence reviewers expect.

**Deliverables**:
- `scripts/analysis/grid_plausibility.py` — computes ratios, flags outliers
- `results/validation/plausibility_<YYYYMMDD>.csv` with per-grid bounds check + flags

### Channel 4 — Opportunistic external agreement (supporting evidence only)

**What**: where admin/survey data exists (CT CBD address points, JHB survey), join admin points to grid and run count regression:

```
n_detected_per_grid  vs  n_admin_per_grid  (coverage grids only)
report: Pearson r, slope, intercept, R², residual spatial autocorrelation
```

**Why**: when it agrees with internal channels, it is a credible extra signal. It is **not** the primary metric because (a) admin data is urban-biased, (b) admin points are not panel coordinates (~100 m offset is normal), (c) admin data coverage is incomplete and inconsistent across cities.

**Deliverables**:
- `scripts/analysis/external_agreement_grid.py` — joins admin/survey to grid, runs regression
- `results/validation/external_agreement_<region>_<YYYYMMDD>.csv`
- One supporting figure per paper, not a headline number

### Channel 5 (bonus) — Temporal consistency (sub-repo, free validation)

**What**: once the `solar_backdating` install-date estimator is online (sibling repo at `/home/gaosh/projects/solar_backdating/`), verify that per-install dates are monotonic (no panels "disappearing" between years), and that the distribution of install dates by year matches Eskom SSEG national cumulative capacity curves.

**Why**: a cross-check on both the main-repo inventory (positions should persist across years) and the sub-repo estimator. Free — emerges naturally from the sub-repo's output.

## Calibration targets

The two postprocess thresholds (see `configs/postproc/`) are defined in terms of these channels:

- **`v4_high.json`** (high-precision seed inventory, input to sub-repo): calibrated so Channel 1 per-stratum precision ≥ 0.95 (or project-decided target) on every stratum with enough samples. Trades recall for cleanliness.
- **`v4_agg.json`** (economic aggregate inventory): calibrated so Channel 3 plausibility stays in bounds and Channel 4 regression slope is close to 1 on coverage grids. Optimizes for unbiased count per grid.

Both calibrations report Channel 2 recall as a diagnostic, not as a calibration objective.

## What V1.4 stops doing

- Stops treating per-polygon F1 as the headline number.
- Stops requiring external GT to validate new model runs.
- Stops scaling annotation in dense areas only (needed: stratified AOI sampling for recall).
- Stops implicitly calibrating thresholds to the dense-annotation F1.

## Relationship to prior work

- V1.2 / V1.3 installation-profile evaluation → retained as a **diagnostic** channel (legacy F1 reporting for continuity with historical docs). Not the primary metric.
- `scripts/analysis/area_aggregate_eval.py` (2026-04-22 WIP) → becomes the compute backbone for Channel 4 and the area-based piece of Channel 3. Demoted from "external eval harness" to "one channel of four."
- Cross-region benchmark suites (`cape_town_independent_26`, JHB primary) → retained for model-vs-model comparison; these are internal detection benchmarks, not validation.

## References

- Task definition and project constraints: `CLAUDE.md`, `docs/architecture.md`
- Annotation workflow and Two-Axis Model: `data/annotations/ANNOTATION_SPEC.md`
- Postprocess configs: `configs/postproc/`
- Sub-repo temporal estimator: sibling repo `solar_backdating` at `/home/gaosh/projects/solar_backdating/` (V1.4 pivot landed 2026-05-05). Cross-review harness: `.agents/harness/README.md`. Old `geid_bbox` prototype archived under `/home/gaosh/projects/_archive/geid_bbox_legacy_2026-05-05/`.
