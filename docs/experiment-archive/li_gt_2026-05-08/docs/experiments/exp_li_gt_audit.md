# Experiment: Li JHB CBD GT Audit & Supplementation

**Date**: 2026-04-23 .. 2026-04-27
**Status**: Closed (v1 supplement frozen, eval reruns landed)
**V1.4 channel**: Channel 1 — Stratified RA precision audit (GT-quality precondition)

## Why this is Channel 1 evidence

Channel 1 reports per-stratum precision against an RA-adjudicated GT. The
metric is only honest if the GT itself is trustworthy: a "false positive"
that is actually a real PV install missed by the annotator inflates the
denominator and depresses precision. Before V1.4 numbers can be cited,
Channel 1's GT layer has to pass an audit that verifies the FP set isn't
secretly hiding real positives.

This experiment is that audit on the JHB CBD primary benchmark
(25 grids, Li annotations on `geid_2024_02`) — the only stratum where the
project currently has dense GT and intends to publish a precision number.

## Inputs

- **Detector predictions**: V3-C ∩ V4.2 *shared* FP set on the 25 CBD grids
  (intersection of two independently trained detectors → high-prior chips
  for "consistent disagreement with GT")
- **GT under audit**: `Joburg_CBD_Li` annotations (Li scheme, JHB CBD only,
  pre-supplement; n = 2,147 polygons across 25 grids)
- **Audit tool**: `data/cls_pv_nonpv_v3c_v42_cascade/labeler/v3c__both/nonpv_subtype_labeler.html`
  — in-browser per-chip taxonomy assignment, exports CSV

## Audit result (n = 462 chips, V3-C side)

Source: `data/cls_pv_nonpv_v3c_v42_cascade/labeler/v3c__both/nonpv_subtype_labeled.csv`

| `human_label` | n | % of audited |
|---|---:|---:|
| **`actually_pv_mislabeled`** (real PV missed by Li) | **119** | **25.8%** |
| `skylight_roof_window` | 99 | 21.4% |
| `corrugated_metal_roof` | 69 | 14.9% |
| `ground_road_marking` | 48 | 10.4% |
| `solar_thermal_water_heater` | 32 | 6.9% |
| `pergola_carport_shadow` | 30 | 6.5% |
| `roof_shadow_dark_fixture` | 25 | 5.4% |
| `hvac_rooftop_equipment` | 21 | 4.5% |
| `other_unknown` | 19 | 4.1% |
| **Total audited** | **462** | 100.0% |

V4.2-propagated copy (`nonpv_subtype_labeled_v4_2.csv`, 441 chips after
IoU pairing) and the dedup view (`nonpv_subtype_labeled_dedup.csv`, 430
chips) reproduce the same distribution to within ±1 pp per class.

**Headline finding**: 26% of what Li annotators tagged as non-PV in the
shared V3-C/V4.2 detection set is real PV. These are not detector
hallucinations; they are GT omissions.

## Why so many were missed

Cross-referencing per-grid counts and chip provenance, the 119 missed PV
break down primarily into three patterns:

1. **CBD high-rise rooftop arrays** annotated by Li on a per-roof basis
   often cover a single canonical install per building, missing a second
   smaller array on the same roof
2. **Adjacent-roof adoption clusters** (3+ neighboring buildings each
   with one install) where annotation density falls off after the first
   2-3 polygons in dense areas
3. **Thin-line / low-contrast** arrays on dark CBD roofs that read as
   non-panel without zooming — the V3-C detector finds them via texture,
   the human eye skipped them on scroll-through

These are systematic Li annotator behaviors, not random noise — meaning
re-annotation with the same workflow would reproduce most of the gap.

## Supplement GT (`Joburg_CBD_Li_supp_v1`)

Built by `scripts/classifier/build_li_supplement_gt.py` from the 119
`actually_pv_mislabeled` polygons:

- Output dir: `data/annotations/Joburg_CBD_Li_supp_v1/`
- Per-grid build summary: `_build_summary.json`
- Each supplement polygon merged into the source grid's GT file (boundary
  polygons appear in both adjacent grids, mirroring how the detector
  picks them up twice)

Aggregate impact:

| | Li orig | Supplement | Merged |
|---|---:|---:|---:|
| Polygons (25 grids) | 2,147 | 119 | 2,266 |
| Δ vs Li orig | — | +5.5% | — |

## Channel 1 precision lift after supplement

Cluster-level eval (`scripts/analysis/cluster_level_eval.py`,
match coverage 0.5 / purity 0.3) on the 25 CBD grids, before vs after
swapping `Joburg_CBD_Li` → `Joburg_CBD_Li_supp_v1`. All 9 detector ×
SAM-prompt configurations available are reported; "—" means no
pre-supplement run was archived for that combination.

Aggregate cluster metrics (full numbers in `results/analysis/v3c_*_vs_li_*`
and `results/analysis/*_vs_li_supp_20260427_supp/`):

| Detector | SAM | Cluster P | Cluster R | Cluster F1 | Balanced |
|---|---|---:|---:|---:|---:|
| V3-C | sam_box | 0.616 → **0.669** | 0.657 → 0.673 | 0.636 → **0.671** | 0.798 → **0.811** |
| V3-C | sam_mask | 0.629 → **0.684** | 0.716 → 0.732 | 0.670 → **0.707** | 0.743 → 0.766 |
| V3-C | no_sam | 0.628 → **0.683** | 0.729 → 0.743 | 0.675 → **0.712** | 0.685 → 0.705 |
| V4.2 | sam_mask | 0.413 → **0.450** | 0.844 → 0.853 | 0.554 → **0.589** | 0.613 → 0.636 |
| V4.2 | no_sam | 0.406 → **0.443** | 0.839 → 0.847 | 0.547 → **0.582** | 0.558 → 0.577 |
| V4.2 | sam_box | — / 0.441 | — / 0.781 | — / 0.564 | — / 0.716 |
| V4.1 | sam_box | — / 0.669 | — / 0.651 | — / 0.660 | — / 0.764 |
| V4.1 | sam_mask | — / 0.684 | — / 0.726 | — / 0.704 | — / 0.697 |
| V4.1 | no_sam | — / 0.681 | — / 0.728 | — / 0.704 | — / 0.639 |

Pattern across all paired runs: precision +3.7 to +5.5 pp, recall +1 to
+1.6 pp, F1 +3.4 to +3.7 pp. Recall barely moves because the 119
supplement polygons are concentrated in dense grids that already had
many TPs; precision moves because previously-counted FPs become TPs.

**Implication for Channel 1**: V3-C + SAM box-only on 25 CBD grids has
true cluster precision ≈ 0.67 (post-supplement), not 0.62 (pre-supplement
Li-only number). The pre-supplement number understated cluster precision
by ~5 pp.

## What this does and does not establish

**Establishes**:
- The Li JHB CBD GT layer used for V1.4 Channel 1 in this stratum had a
  ~5–6% omission rate at the polygon level on detector-flagged candidates
- Audit + supplement workflow shifts cluster precision uniformly upward
  across detector × SAM configurations, validating that the lift is a
  GT-quality correction, not a detector-specific artifact
- The shared V3-C ∩ V4.2 FP set is a high-yield queue for finding GT
  omissions (1 in 4 audited chips was a real positive)

**Does not establish**:
- Whether the same omission rate applies to Li annotations outside this
  25-grid CBD set — supplement is scoped to the audited grids only
- Whether non-shared FPs (V3-C only or V4.2 only) contain proportional
  missed PV — by construction the audit only looked at shared chips
- Final Channel 1 precision-with-CI: this audit cleans the GT, but the
  per-stratum CI calculation belongs in the V1.4 Channel 1 deliverable
  (`results/validation/ra_precision_<YYYYMMDD>.csv`, not yet produced)
- Anything about CT or other strata — Li scheme covers JHB only

## Carry-overs

- Same audit protocol on **non-shared** FP queues (V3-C only, V4.2 only)
  to estimate omission rate in the wider FP population
- Replicate on the matching CT primary benchmark when Cape Town reaches a
  comparable per-grid annotation density (currently the Gao scheme is
  denser; precision audit there is a separate Channel 1 task)
- Decide whether `Joburg_CBD_Li_supp_v1` becomes the canonical Li GT
  going forward, or stays parallel to `Joburg_CBD_Li` so re-audits remain
  comparable to the pre-supplement baseline

## References

- Audit CSVs: `data/cls_pv_nonpv_v3c_v42_cascade/labeler/v3c__both/`
- Supplement build script: `scripts/classifier/build_li_supplement_gt.py`
- Supplement GT: `data/annotations/Joburg_CBD_Li_supp_v1/`
- Pre-supplement eval: `results/analysis/v3c_*_vs_li_20260426/`,
  `results/analysis/v4_2_*_vs_li_20260426/`
- Post-supplement eval: `results/analysis/*_vs_li_supp_20260427_supp/`
- V1.4 framework: [`docs/validation_strategy.md`](../validation_strategy.md)
- Annotation specification (Two-Axis Model): [`data/annotations/ANNOTATION_SPEC.md`](../../data/annotations/ANNOTATION_SPEC.md)
