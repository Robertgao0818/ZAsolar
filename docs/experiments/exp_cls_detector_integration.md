# Classifier → Detector Evaluation Integration Contract

**Status**: Active
**Plan reference**: Task 5a in
`/home/gaosh/.claude/plans/codex-efficientnetb0-convnexttiny-found-swirling-muffin.md`
**Related**: `exp_cls_backbone_ablation.md`, `exp_cls_dataset_protocol.md`

Defines how the PV vs non-PV classifier plugs into the installation-level
evaluation pipeline. The two components stay **decoupled** — the classifier
is not injected into `detect_and_evaluate.py`'s main pipeline.

## Why decoupled

Injecting the classifier into the detector pipeline would:

1. Conflate classifier config changes with detection cache keys, causing
   unnecessary re-runs every time a classifier threshold shifts.
2. Break existing `configs/postproc/*.json` semantics (`load_postproc_config`
   is for geometric post-processing only).
3. Force every caller of `detect_and_evaluate.py` to carry classifier
   knobs even when classifier filtering is not wanted.

Keeping the classifier as an **external pre-evaluation step** means each
piece can be iterated independently, and cache invalidation stays tight.

## Three-step data flow

```
┌─────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────┐
│ detect_and_eval.py  │ →  │ classify_predictions.py  │ →  │ detect_and_eval.py  │
│ (detection run)     │    │ (post-hoc classifier)    │    │ --classifier-       │
│                     │    │                          │    │ filtered-gpkg       │
│ writes              │    │ reads                    │    │                     │
│ predictions_metric  │    │ predictions_metric.gpkg  │    │ eval-only path      │
│ .gpkg               │    │                          │    │                     │
│                     │    │ writes                   │    │ reads the filtered  │
│                     │    │ predictions_metric_     │    │ gpkg and reports    │
│                     │    │ filtered.gpkg            │    │ installation-level  │
│                     │    │ (+ _cls.gpkg provenance) │    │ P/R/F1              │
└─────────────────────┘    └──────────────────────────┘    └─────────────────────┘
```

## Artifacts

All artifacts live under `results/<region-or-flat>/<model-run-or-flat>/<GRID>/`:

| File | Writer | Contents |
|---|---|---|
| `predictions_metric.gpkg` | `detect_and_evaluate.py` | Raw detections (source of truth for classifier input). |
| `predictions_metric_cls.gpkg` | `classify_predictions.py` | All detections + `cls_score`, `cls_label`, `cls_applied` columns. Full provenance record. |
| `predictions_cls.geojson` | `classify_predictions.py` | EPSG:4326 export of above, for QGIS inspection. |
| `predictions_metric_filtered.gpkg` | `classify_predictions.py` | **Canonical filtered GPKG** — only rows with `cls_label == 'pv'`. This is the file consumed by `detect_and_evaluate.py --classifier-filtered-gpkg`. |
| `predictions_metric_cls_filtered.gpkg` | `classify_predictions.py` | Identical copy for backwards compat with earlier review scripts. |
| `cls_summary.json` | `classify_predictions.py` | Counts, threshold, area_cutoff, model_path, timestamp. |

## Command sequence

**Detection (produces raw GPKG)**

```bash
python detect_and_evaluate.py --grid-id G1687 \
  --model-path checkpoints/exp003_C_targeted_hn/best_model.pth \
  --postproc-config configs/postproc/v4_canonical.json
```

**Classifier filter (produces filtered GPKG)**

```bash
python scripts/classifier/classify_predictions.py \
  --grid-id G1687 \
  --model-path checkpoints/cls_pv_thermal/best_cls.pth \
  --pv-threshold 0.5
```

**Evaluation with classifier filter applied**

```bash
python detect_and_evaluate.py --grid-id G1687 \
  --evaluation-profile installation \
  --classifier-filtered-gpkg results/G1687/predictions_metric_filtered.gpkg \
  --classifier-model-path checkpoints/cls_pv_thermal/best_cls.pth \
  --classifier-threshold 0.5
```

When `--classifier-filtered-gpkg` is provided, `detect_and_evaluate.py`:

1. **Skips detection entirely** — goes straight to `load_predictions(override_path=...)`.
2. Writes `classifier_filtered_gpkg`, `classifier_model_path`,
   `classifier_threshold` into the grid's `config.json` so the
   provenance-hashed cache key differs from the unfiltered run.
3. Uses the overridden GPKG for both presence matching and installation-
   level evaluation. `--evaluation-profile` semantics are unchanged.

## CLI contract (`detect_and_evaluate.py`)

New flags (all optional, all no-op if absent):

| Flag | Purpose |
|---|---|
| `--classifier-filtered-gpkg <path>` | Swap in the classifier-filtered GPKG. Triggers eval-only mode. |
| `--classifier-model-path <path>` | Provenance only — recorded in `config.json`. |
| `--classifier-threshold <float>` | Provenance only — recorded in `config.json`. |

`--force`, `--postproc-config`, `--evaluation-profile`, `--data-scope` all
continue to work as before. The classifier filter flags are **additive** —
they never silently change evaluation profile or postproc params.

## Cache behavior

`should_reuse_predictions` hashes `detection_config` to decide whether the
detection cache is reusable. When `--classifier-filtered-gpkg` is given,
the cache check is bypassed entirely (we are in eval-only mode). The
additional provenance fields (`classifier_*`) are recorded in
`detection_config` so that a subsequent detection-mode run against the same
grid will not reuse results that were generated under a different
classifier.

## Non-goals for this task

- **Not** fusing the classifier into `detect_solar_panels`.
- **Not** adding classifier_model as a `configs/postproc/*.json` key
  (existing `_meta` note in `detect_and_evaluate.py:226-227` about a future
  `classifier_model` postproc extension is now superseded by this
  decoupled design).
- **Not** changing evaluation profile semantics — `installation` stays
  installation-level evaluation of reviewed-prediction footprints vs
  installation-level GT.

## Verification steps

For any grid with reviewed GT available:

1. Run classifier filter to produce `predictions_metric_filtered.gpkg`.
2. Run `detect_and_evaluate.py` without the flag → record pre-filter
   installation-level P/R/F1.
3. Run `detect_and_evaluate.py --classifier-filtered-gpkg ...` → record
   post-filter P/R/F1.
4. Delta must match what `compare_results.py` reports on reviewed-grid
   filtered precision; otherwise the decoupling is leaking state.
5. `config.json` in the post-filter run must contain the three classifier
   provenance fields.

## Detector × SAM mode benchmark (2026-04-26)

Why this matters for classifier integration: the classifier is the second
stage of a (detector → SAM mask refine → classifier) cascade. Picking the
detector + SAM mode that *enters* the classifier determines (a) how many FPs
the classifier has to kill, (b) the ceiling area F1 the cascade can reach,
and (c) the recall budget. The 9-cell ablation below was run on the V1.4
benchmark blueprint (25 JHB CBD grids × Li hand-labeled GT × GEID 2024-02
chunked imagery, post-proc `configs/postproc/v4_canonical.json`,
cluster-level eval via `scripts/analysis/cluster_level_eval.py`).

### Full 3 detector × 3 SAM mode matrix

| Detector / SAM mode | matched | FP | FN | cluster R | cluster F1 | **area F1** | **balanced** |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3-C / no SAM            |  977 |  578 |  364 | 0.729 | **0.675** | 0.833 | 0.685 |
| **V3-C / box-only**      |  948 |  591 |  495 | 0.657 | 0.636 | **0.918** ⭐ | **0.798** ⭐ |
| V3-C / mask+box          |  981 |  578 |  389 | 0.716 | 0.670 | 0.895 | 0.743 |
| V4.1 / no SAM            |  851 |  508 |  332 | 0.719 | 0.670 | 0.791 | 0.620 |
| V4.1 / box-only          |  831 |  519 |  467 | 0.640 | 0.628 | 0.891 | 0.750 |
| V4.1 / mask+box          |  857 |  506 |  339 | 0.717 | 0.670 | 0.855 | 0.676 |
| V4.2 / no SAM            |  826 | 1211 |  159 | **0.839** | 0.547 | 0.730 | 0.558 |
| V4.2 / box-only          |  821 | 1211 |  244 | 0.771 | 0.530 | 0.860 | 0.696 |
| V4.2 / mask+box          |  841 | 1198 | **155** | **0.844** | 0.554 | 0.796 | 0.613 |

Source artifacts: `results/analysis/{v3c_rerun,v4_1,v4_2}_vs_li_20260426/`
(detector-only) and `results/analysis/{v3c,v4_1,v4_2}_sam_{box,mask}_vs_li_20260426/`
(+ SAM); raw masks in `results/johannesburg/{v3c,v4_1,v4_2}_sam_{box,mask}_geid_2024_02/`.

### Three findings that shape classifier-stage planning

1. **"SAM is the area F1 ceiling" — falsified.** V4.2+SAM tops at area F1
   0.860; V3-C+SAM tops at 0.918. **Detector box quality propagates
   directly into SAM output** — clean boxes let SAM pick the right
   sub-component. Detector choice is *not* recall-only.
2. **Box-only FN flip is detector-invariant.** V3-C +131 FN, V4.1 +135 FN,
   V4.2 +85 FN — every detector loses recall when SAM gets bbox-only
   prompts. The "SAM `argmax(score)` specificity bias picks
   sub-components" failure mode is general, not over-segmentation-
   specific. Classifier integration cannot fix this — it must be solved
   at the SAM-prompt layer.
3. **mask+box gives a +6pp area F1 gain across all detectors.**
   V3-C +6.2 / V4.1 +6.4 / V4.2 +6.6. The role of mask-prompt is
   sub-component-flip suppression, not over-seg repair.

### Implications for classifier integration

- **Primary cascade input = V3-C + SAM mask+box.** Detector matched=981,
  FP=578, FN=389, area F1 0.895. Classifier needs to kill ~70% of 578 FPs
  to push cluster P from 0.629 → ~0.85; estimated three-stage cluster F1
  ≈ 0.78, area F1 0.895, balanced ≈ 0.80+.
- **Aggressive variant = V3-C + SAM box-only.** Starts at area F1 0.918 /
  balanced 0.798 (already best in the no-classifier matrix). Same FP count
  (591), but recall is capped at 0.657 — choose this only if downstream KPI
  is weighted toward area precision and recall is allowed to drop.
- **Control = V4.2 + SAM mask+box.** recall 0.844 ceiling but FP 1198
  forces classifier recall ≥ 90% to break even on cluster F1; reserved for
  the case where someone trains a near-perfect water-heater discriminator.
- **V4.1 is dominated** at every SAM mode by V3-C at the same SAM mode;
  drop from candidate cascades, keep only as ablation control.

### Classifier dataset implication

The 981 matched + 578 FP from V3-C + SAM mask+box on the 25 CBD grids form
a natural labeled pool for the binary PV vs non-PV classifier:

- 981 matched (vs Li GT, IoU ≥ 0.1) → positive examples
- 578 FPs → negative examples (subtype audit feeds
  `scripts/analysis/cls_nonpv_subtype_audit.py` to ensure water-heater /
  fixture / shadow / road-marking coverage)

This dataset is built and consumed via the protocol in
`exp_cls_dataset_protocol.md`; see also `exp_cls_backbone_ablation.md` for
the backbone evaluation that will be run on it.

## Li GT audit (2026-04-27): the matrix above is biased

The 9-cell matrix above was computed against the unaltered Li GT
(`/mnt/d/ZAsolar/annotations_inbox/Joburg_CBD_Li/`, 2147 polygons across
25 grids). When auditing the **V3-C ∩ V4.2 shared FP core** (462 polygons
with `source_detector=both, label=nonpv` from the cascade pool builder)
to seed the binary classifier subtype taxonomy, we found that
**119 / 462 (25.8%) were not non-PV at all — they were real PV
installations missing from Li GT** (V3-C and V4.2 both predicted them at
≥0.98 confidence; nearest Li GT polygon often >30 m away).

Two follow-on findings from the same audit:
- JHB CBD non-PV profile differs from CT: skylight 21.4% +
  corrugated_metal_roof 14.9% + road_marking 10.4% = 47% of the FP
  core, vs the CT-batch003 finding of 77% solar thermal water heaters.
  The CT taxonomy does not transfer wholesale.
- GEID mosaics for adjacent CBD grids overlap by ~35-70 m, so the same
  physical object is detected and labeled twice across grids
  (`scripts/classifier/dedup_cls_pool.py` quantifies the redundancy:
  462 raw → 430 unique objects in the audited core, 3913 → 3728 in the
  full cascade pool).

### Re-evaluated 9-cell matrix (Li GT + 119 supplement)

GT supplement built by `scripts/classifier/build_li_supplement_gt.py`:
the 119 audit-confirmed PVs are added to their source grid's GT; a
boundary PV ends up in both grids' GT (mirroring the per-grid prediction
setup). All 9 reruns saved under
`results/analysis/<run>_vs_li_supp_20260427_supp/`.

| Detector / SAM mode | matched | FP | FN | cluster R | cluster F1 | **area F1** | **balanced** |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3-C / no SAM            | 1057 |  491 |  365 | 0.743 | 0.712 | 0.840 | 0.705 |
| **V3-C / box-only**      | 1025 |  507 |  499 | 0.673 | 0.671 | **0.920** ⭐ | **0.811** ⭐ |
| V3-C / mask+box          | 1062 |  490 |  389 | 0.732 | 0.707 | 0.906 | 0.766 |
| V4.1 / no SAM            |  921 |  432 |  344 | 0.728 | 0.704 | 0.797 | 0.639 |
| V4.1 / box-only          |  899 |  445 |  481 | 0.651 | 0.660 | 0.894 | 0.764 |
| V4.1 / mask+box          |  928 |  429 |  350 | 0.726 | 0.704 | 0.863 | 0.697 |
| V4.2 / no SAM            |  899 | 1131 |  163 | **0.847** | 0.582 | 0.736 | 0.577 |
| V4.2 / box-only          |  894 | 1132 |  251 | 0.781 | 0.564 | 0.865 | 0.716 |
| V4.2 / mask+box          |  915 | 1117 |  158 | **0.853** | 0.589 | 0.803 | 0.636 |

Direction of every shift vs the original Li-only matrix:
- matched ↑ in every cell (the 119 supp PVs joined as TPs).
- FP ↓ in every cell (those 119 left the FP bucket).
- balanced and area F1 generally up; the largest gains land on V3-C
  variants because the supplement set is dominated by polygons V3-C
  predicted at high confidence.
- FN does not shift much because the supplement was *added to TPs*,
  not removed from FNs.

### Net effect on conclusions

- **V3-C + SAM box-only stays the area-F1 / balanced leader** (0.920 /
  0.811, both nudged upward) — the cascade-input recommendation is
  unchanged.
- **V3-C + SAM mask+box stays the recall-friendly cascade input**
  (cluster R 0.732, area F1 0.906, FP only 490) — also unchanged.
- **V4.2 stays out of the cascade-input role** — even with supplement,
  cluster F1 is ≤0.59 and balanced ≤0.72; FP count (1117-1132) is still
  ~2.3× V3-C's. The high recall (0.85) doesn't compensate.
- **V4.1 stays dominated** by V3-C at every SAM mode.

### Audit-derived FP subtype distribution (V3-C, n=462 raw / 430 dedup)

Subtype labels for the 462-row V3-C ∩ V4.2 nonpv core, audited in
`data/cls_pv_nonpv_v3c_v42_cascade/labeler/v3c__both/nonpv_subtype_labeled.csv`
(de-duplicated to 430 canonical rows by
`scripts/classifier/dedup_cls_pool.py`):

| subtype | raw n | dedup n | dedup % |
|---|---:|---:|---:|
| actually_pv_mislabeled (→ supplement Li GT) | 119 | 106 | 24.7% |
| skylight_roof_window | 99 | 95 | 22.1% |
| corrugated_metal_roof | 69 | 63 | 14.7% |
| ground_road_marking | 48 | 47 | 10.9% |
| solar_thermal_water_heater | 32 | 31 | 7.2% |
| pergola_carport_shadow | 30 | 27 | 6.3% |
| roof_shadow_dark_fixture | 25 | 24 | 5.6% |
| other_unknown | 19 | 19 | 4.4% |
| hvac_rooftop_equipment | 21 | 18 | 4.2% |
| **total** | **462** | **430** | |

V4.2-side subtype labels were propagated automatically via per-grid
`source_detector=both` pairing
(`scripts/classifier/propagate_subtype_to_v42.py`); 441/441 V4.2
polygons received a propagated subtype, distribution is within ±5%
of V3-C-side (sanity confirms the cross-detector pair is the same
physical object).

Combined V3-C + V4.2 subtype-labeled set: **903 rows** in
`labeler/v3c__both/nonpv_subtype_labeled_union.csv`. After moving
`actually_pv_mislabeled` (225 rows) from the non-PV pool into the PV
pool, the cleaned non-PV training set retains **678 rows** across 9
subtypes — this is the raw input to the binary classifier ablation
(`exp_cls_backbone_ablation.md`).

## Frozen deployment baseline (CT, 2026-04-28): v1 ConvNeXt-tiny

Cape Town in-domain cascade is **frozen as a deployment baseline** at
v1 + ConvNeXt-tiny + threshold 0.5 + V3-C + SAM mask+box. All future
classifier work (v2 / v3 / ensembles / per-imagery thresholds) is
evaluated *against this baseline* on the CT side, not against raw V3-C.

### Frozen numbers

CT 6-grid holdout (no training-set leakage), 798 GT polygons,
`results/analysis/cls_cascade_holdout/backbone_compare_thr050/summary_metrics.csv`:

| Stage | P | R | F1 | FP | TP-lost vs raw |
|---|---:|---:|---:|---:|---:|
| Raw V3-C (no classifier) | 0.685 | 0.931 | 0.789 | 342 | — |
| **+ v1 ConvNeXt @ 0.5** | **0.835** | **0.890** | **0.862** | **140** | **33** |
| + v1 EffB0 @ 0.5 | 0.824 | 0.880 | 0.851 | 150 | 41 |
| + v1 DINOv2 @ 0.5 | 0.843 | 0.888 | 0.865 | 132 | 34 |

**Headline**: F1 0.789 → 0.862 (**+7.3 pp**); FP kill 202 / TP lost 33
≈ **6.1 : 1** net-positive trade. Recall drop only −4.1 pp on a CT
in-domain holdout that contains no training grids.

DINOv2 edges F1 by +0.003 but with one extra TP loss; ConvNeXt is
chosen as the default for **operational reasons**: faster inference,
fewer dependency surprises (timm vs DINOv2 weights), and the
recall-budget headroom (3 pp slack vs ConvNeXt's 4.1 pp) is too small
to outweigh the operational simplicity.

### Why this is defensible despite the OOD failure

The same v1 ConvNeXt scores **bal_acc 0.653** on the JHB CBD GEID
audit set (462 chips, see `results/analysis/cls_audit_eval_20260427/`)
and only kills **15.5%** of non-PVs at PV-recall ≥ 0.95 — far below the
40% promotion bar for v2. The two regimes are not in conflict:

- **CT in-domain non-PV ≈ {water heater, skylight, shadow}**, dominated
  77% by water heaters in the original taxonomy. v1's CT training pool
  covers this distribution well; the model learned a tight one-class PV
  boundary against this specific non-PV mix.
- **JHB CBD GEID non-PV ≈ {skylight 22%, corrugated metal 15%, road
  marking 10%, HVAC, water heater 7%}**. v1 has near-zero training
  signal for corrugated metal / HVAC / road marking and does not see
  GEID's color/sharpness statistics — the same one-class boundary
  generalises poorly.

→ v1 is a *CT-shaped* PV-vs-non-PV filter. Within CT it functions as
the intended water-heater suppressor (the dominant FP class). Outside
CT — especially on GEID CBD — it should not be deployed.

### Deployment scope

v1 ConvNeXt is approved for inference-time filtering on results runs
whose `config.json.imagery_layer_id == aerial_2025` (Cape Town
municipal WMS). Other imagery layers (`aerial_2023` JHB, `geid_2024_02`
JHB CBD, future Vexcel layers) **must not** route through v1 by
default; they wait for v2 with per-imagery thresholds (see
`exp_cls_dataset_protocol.md` v2 protocol section).

### Replacement criteria

v1 ConvNeXt is replaced as the CT default when *both* hold:

1. v2 (or later) achieves CT 6-grid holdout F1 ≥ 0.862 + 1 pp
   (i.e. ≥ 0.872) at the same recall budget (≤ 4.5 pp drop vs raw),
   *and*
2. v2 satisfies its own JHB GEID promotion rule (PV-recall ≥ 0.95,
   non-PV kill ≥ 40%) so the replacement is one model, not a per-domain
   patchwork.

If only (1) holds and (2) does not, v1 ConvNeXt **stays as the CT
deployment** while v2 ships only on the imagery layers it qualifies
for. Per-imagery model selection is acceptable; per-imagery thresholds
on a single model is preferred but not required.

### What this freeze unblocks

- Vexcel pull / new-city domain-shift work can proceed without waiting
  on classifier completion: CT is already stable with v1, and new
  cities use raw V3-C + SAM mask+box until v2 has a per-imagery
  threshold for the relevant layer.
- V1.4 Channel 1 RA precision audit on CT can use v1-filtered output as
  its input, since v1 is now the official CT inventory pipeline rather
  than a candidate.
- v2 work can be evaluated honestly against +7.3 pp baseline, not
  against raw V3-C.
