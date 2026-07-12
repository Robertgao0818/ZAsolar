# Building-level rooftop solar PV inventories for Cape Town and Johannesburg, South Africa, from sub-10 cm aerial imagery

<!-- ESSD Data Descriptor draft v0 (REBUILT 2026-07-12; the 2026-07-06 v0 was lost
     with /tmp handoffs during the WSL→Fedora machine migration and never committed).
     Master reference: docs/plans/2026-07-06-storyline-tech-plan-v2.md (draft v3).
     Scope: CENSUS-ONLY (owner decision 2026-07-06, reconfirmed 2026-07-12) —
     the JHB temporal panel is excluded from the v1 data release; §7 keeps the
     living-data extension slot. P0 gate chain (displacement audit → ISSUE-06 →
     Panel v2 → ISSUE-11) is deliberately NOT a dependency of anything in this draft. -->

**Status**: v0 — census sections drafted; PENDING blocks marked inline.
**Evidence discipline**: every number carries its caliber inline (analysis unit,
IoU/matching, merge-mode, GT tier) and a source path in an HTML comment. Claims are
graded `[F]` (file/number-backed) / `[H]` (must verify before submission). Cross-caliber
comparisons are forbidden, including against curated-benchmark literature numbers.

---

## Abstract (skeleton)

We present building-level inventories of rooftop solar photovoltaic (PV) installations
for South Africa's two largest metropolitan areas, derived from sub-10 cm aerial
imagery with an instance-segmentation pipeline and validated with an
aggregate-first, caliber-disciplined framework. The Cape Town inventory contains
111,801 de-duplicated installations (5.0 km² of panel area; IoU≥0.10 union-merge;
area calibration ratio 1.083 against ground truth on 99 GT-bearing census cells)
<!-- [F] docs/validation/2026-06-24-ct-census-gt-validation.md;
     results/analysis/ct_census_gt_eval/ct_census_gt_summary.csv -->
across 2,083 census cells; the Johannesburg inventory contains 41,393 installations
across 382 one-kilometre grid cells (aggregate area calibration 0.995 on the 25-grid
CBD ground-truth face)
<!-- [F] ~/zasolar_data/geid_temporal/.../summary.json n_targets=41393;
     results/analysis/full382_merge01_2026-05-15/spatial_eval_summary.json bulk=0.9948 -->.
South Africa has no national register of distributed PV and municipal small-scale
embedded-generation (SSEG) registration is severely incomplete, making these the
first building-level PV inventories for the country [H: "first" claim needs the
global-product literature check, see PENDING register]. Beyond the data, we
contribute a validation methodology for PV censuses built on imperfect,
sub-array-level ground truth: success is defined at the analysis unit of downstream
use (grid-cell area aggregates), metric calibers are reported jointly and
never mixed, and the measurable ground-truth ceiling is decomposed explicitly rather
than compared against curated benchmarks. Data are available under [PENDING: Zenodo
DOI] in GeoParquet and GeoJSON.

---

## 1. Introduction

### 1.1 Motivation

South Africa has no national registry of distributed rooftop PV. The regulatory
channel that should capture residential systems — municipal Small-Scale Embedded
Generation (SSEG) registration — is known to be severely incomplete, and the
registries that exist record billing points, not installations, with structural
mismatches to what a rooftop census observes.
<!-- [F] memory sa_admin_data_limits; sseg_structural_mismatches:
     SSEG geo = building point not panel; three confounders documented in
     results/analysis/sseg_kw_calibration_2026-05-10/ -->
Grid operators and researchers are therefore blind on both the spatial
distribution and the adoption timeline of rooftop PV — in the country where that
gap matters most: the 2022–2024 load-shedding crisis produced a quasi-experimental
adoption shock whose analysis requires exactly the space-resolved installation data
that no register provides. [H: load-shedding framing is motivation only — this
data descriptor makes **no event-study commitment**; the identification question
(right-censoring × demand withdrawal) is explicitly out of scope and flagged in §7.]

### 1.2 Why existing global products do not cover this

<!-- PENDING [H→F]: this whole subsection needs the deep-research verification pass
     (research_verification_memo — LOST with /tmp, must be re-run; see PENDING
     register). Do not cite minimum-detectable-size or date-granularity numbers
     until re-verified against the original papers. -->

Existing global solar mapping products (Kruitwagen et al. 2021; TZ-SAM; Global
Renewables Watch) target utility-scale plants from ~10 m satellite imagery;
residential rooftop systems are invisible to them by design, and none provides
building-level installation records. Residential-scale censuses exist only for
rich-data countries with national aerial survey programmes (e.g. DeepSolar for the
US). South Africa combines the double blank — no rooftop-level product, no adoption
timeline — with the highest value for filling it (no registry + a natural adoption
experiment). [H: all product characteristics to be re-verified]

### 1.3 Contributions

1. **Two metropolitan building-level PV inventories** (Cape Town, Johannesburg),
   delivered as installation polygons with per-installation area, confidence and
   provenance, and validated at the grid-cell aggregate level (§4, §5).
2. **An aggregate-first validation methodology** for PV censuses under imperfect
   ground truth: analysis-unit-aligned success metrics, joint reporting of metric
   calibers with a prohibition on cross-caliber comparison, and an explicit
   ground-truth-ceiling decomposition replacing curated-benchmark comparisons (§4).
3. **A South-Africa-specific detection contribution**: two-track suppression of
   the dominant local confusion class, solar water heaters (§3.3) — to our
   knowledge unaddressed in the PV-detection literature
   <!-- [F] memory pv_thermal_literature: literature blank documented -->.

---

## 2. Study areas and source imagery

| | Cape Town | Johannesburg |
|---|---|---|
| Census extent | 2,083 census cells (of a 4,429-cell full-metro grid; cells outside municipal WMS coverage excluded) | 382 one-km grid cells (JNB grid, metropolitan area) |
| Imagery | City of Cape Town municipal aerial orthophotos, 2025 vintage, via public WMS <!-- [F] regions.yaml imagery_layers: aerial_2025 --> | Vexcel aerial orthophotos, 2024 vintage, 6.7 cm GSD <!-- [F] results/analysis/full382_merge01_2026-05-15/README.md --> |
| Imagery access | Public municipal service | Vexcel academic partnership programme; imagery cited per programme terms <!-- [F: owner 2026-07-12] license resolved via academic partnership; PENDING: exact citation text from agreement --> |
| Native CRS of deliverable | EPSG:32734 | EPSG:32735 |

<!-- Note for §5/§8: derived-vector publication rights for JHB follow from the
     academic partnership terms (owner decision 2026-07-12: citation suffices).
     The earlier public-EULA analysis (Internal-Use-only Derivatives) is superseded
     for this dataset but keep a copy of the agreement on file with the DOI deposit. -->

Grid cells are the primary aggregation and validation unit throughout: the census
is *judged* at the cell level (§4.1), and the released data carry the cell id of
every installation.

---

## 3. Methods

### 3.1 Detection

A torchvision Mask R-CNN (ResNet-50-FPN) instance-segmentation detector is run
tile-wise over each cell (`detect_direct.py`), followed by a deterministic
finalization stage (`finalize.py`) applying the locked post-processing
configuration (`v4_canonical`) and per-detection merge mode.
<!-- [F] production caliber locked: v4_canonical + per-detection;
     CLAUDE.md production engine section; docs/validation/2026-06-24 report header -->
The detector is trained max-recall on a multi-city corpus (Cape Town SAM2-assisted
annotation batches + Johannesburg Vexcel grids, 88 grids total for the production
checkpoint) with hard-negative (HN) generations described in §3.3.
<!-- [F] results/analysis/full382_merge01_2026-05-15/README.md model card;
     configs/datasets/training_sets.yaml lineage -->
The same detector checkpoint (`unified_A`) serves both cities; post-processing is
calibrated per imagery source (§3.4).

### 3.2 Boundary refinement (Johannesburg)

Johannesburg production applies SAM 2.1 (hiera-large) mask+box-prompted boundary
refinement to accepted detections.
<!-- [F] memory sam_prompt_design (mask+box, not bbox-only);
     jhb production chain: per-det + grid-NMS@0.5 + SAM refine, README above -->
Cape Town production does **not** use SAM refinement: on the CT evaluation face the
classifier-only chain won selection (§3.4), and the asymmetry is retained as a
per-source calibration outcome rather than harmonized away (§3.4 rationale).

### 3.3 False-positive suppression — the solar-water-heater problem

The dominant confusion class in South African residential imagery is the rooftop
solar water heater (SWH) — visually a dark rectangular roof panel. Suppression is
two-track:

- **Training track** (realized in the model lineage): targeted hard-negative
  mining beginning with the V3-C generation and culminating in the V4/V4.1 HN
  generations, whose negative sampling is 77 % water heaters.
  <!-- [F] memory v4_hn_strategy; configs/datasets/training_sets.yaml -->
- **Inference track**: a DINOv2-S binary PV/non-PV classifier applied to detection
  chips, with thresholds calibrated per imagery layer (CT aerial_2025 threshold
  0.7168, classify-all mode, adaptive chip extraction).
  <!-- [F] memory cls_v2_protocol; threshold from 2026-06-24 validation report -->

For Johannesburg an additional LLM-based FP review (Gemini) was applied to the
merged inventory, removing 6,072 detections (47,465 → 41,393) with all 9
pre-registered acceptance gates passing.
<!-- [F] memory jhb_fp_cut_sweep; deliverable
     results/analysis/full382_merge01_2026-05-15/jhb_full382_unified_A_merge01_c0925_fpcut_2026-06-01.gpkg
     PENDING: link the 9-gate definition doc explicitly -->

### 3.4 Per-source post-processing calibration

The two cities' post-detection chains differ (CT: classifier-only; JHB:
SAM refinement + Gemini FP-cut). This is by design, not drift: the classifier
protocol is defined with per-imagery-layer thresholds, and each city's chain was
selected on its own Tier-1 evaluation face (§4.2's leakage rules applying).
<!-- [F] memory cls_v2_protocol (per-imagery thresholds are the protocol);
     memory ct_census_clsonly_baseline (CT selection: CLS-only won, σ_Bw 0.297 on
     its selection face); docs/validation/2026-06-24 Table A (merge-mode selection) -->
Imagery sources differ in GSD, radiometry and vintage; a single harmonized chain
would be calibrated for neither. The cost — the two inventories are not produced
by byte-identical pipelines — is disclosed here and bounded by the shared
detector, shared metric kernel, and shared acceptance framework (§4).

### 3.5 De-duplication and inventory assembly

Per-cell detections are merged city-wide with a geometric union-merge at
IoU > 0.10 (union-find transitive closure). The union is idempotent on
set-theoretic area: aggregate area metrics are mathematically unchanged by
merging; only polygon counts change.
<!-- [F] results/analysis/full382_merge01_2026-05-15/README.md ("cannot add area
     beyond what detections drew"); same merge for CT per
     docs/adr/0003-repo-evolution-log.md 2026-06-16 entry -->
Johannesburg additionally applies its production polygon-confidence operating
point (≥ 0.925) before merge.
<!-- [F] spatial_eval_summary.json params: conf_min 0.925, iou 0.1 -->

---

## 4. Validation

### 4.1 Aggregate-first framework and the metric-caliber table

Success is defined at the analysis unit of downstream use: **grid-cell area
aggregates**, not per-polygon overlap. Model and operating-point selection use the
Tier-1 metric suite — per-grid area-ratio dispersion (σ_Bw) and RMSE as primary
judges; aggregate area F1, through-origin slope and R² alongside; the bulk
(pred/GT area) ratio as a sanity gate only, never an optimization target
<!-- [F] docs/evaluation_protocol.md; memory bulk_ratio_perverse_incentive
     (bulk is non-monotone in quality — Goodhart) -->.

Because headline numbers in this paper come from different measurement calibers,
we report the caliber table up front and prohibit cross-caliber comparison —
including against curated-benchmark literature numbers:

| Metric | Analysis unit | IoU / matching | Merge-mode | GT tier | Comparable against |
|---|---|---|---|---|---|
| agg_area_F1 / bulk / σ_Bw | grid-cell aggregate **area** | none (area sums) | per-detection | A2 SAM sub-array GT (CT 99 cells / JHB clean_gt 25 cells) | same-caliber census runs only |
| polygon F1@0.5 | single installation polygon | 0.5, installation profile | per-detection | clean_gt (measurable ceiling 0.85–0.88, §4.4) | same caliber, always jointly with the ceiling |
| Human-review precision 0.749 | manually audited detection sample | human judgment (non-IoU) | — | human | not comparable to any F1 |

<!-- [F] table structure from storyline v2 §1.3 (round-3 addition);
     ceiling from docs/plans/2026-06-10-rcnn-f1-gap-review.result.json;
     precision from memory ch1_precision_results -->

### 4.2 Specification robustness: leakage-checked operating points, fail-closed

Operating points (polygon-confidence thresholds) are fitted only on calibration
cells disjoint from every reporting suite, with a mechanical leakage check and a
≤1 pp transfer-acceptance gate. The gate has teeth: the CT pixel-or lock passed
with 0.00 pp transfer gap; the production per-detection lock **failed** (5.22 pp,
structural divergence between the σ_Bw ranking rule and the agg_F1 acceptance
metric). Disposition is fail-closed: per-detection reporting uses a pre-declared
threshold from the prior trade-off analysis, never a value swept on the reporting
suite, and the failure is recorded rather than hidden.
<!-- [F] docs/validation/2026-06-21-validation-methodology.md §1;
     scripts/analysis/lock_operating_point.py -->

### 4.3 Delivered-census validation

The validation object is the **shipped deliverable**, not a dedicated evaluation
run: the released gpkg is scored against ground truth on the census cells that
carry GT, with the same metric kernels used in production
(`core.area_metrics`, `core.polygon_validation`).

**Cape Town** (99 GT-bearing census cells, A2 SAM GT, 6,787 GT polygons):

| metric | delivered census (n=99) | GT≥500 m² subset (n=63) |
|---|---|---|
| agg_area_F1 | **0.833** | 0.837 |
| bulk ratio | **1.083** | 1.083 |
| median cell ratio | 1.024 | 1.016 |
| σ_Bw | 0.516 | 0.522 |
| RMSE (m²) | 695 | 868 |
| through-origin slope | 1.019 | 1.019 |
| R² (through-origin) | 0.918 | 0.886 |
| within ±20 % | 0.626 | 0.714 |
| cov50 (count recall) | 0.856 | — |

<!-- [F] results/analysis/ct_census_gt_eval/ct_census_gt_summary.csv (verified
     2026-07-12: agg_area_F1=0.8326, bulk=1.0834, n_grids=99, n_gt_polys=6787);
     narrative from docs/validation/2026-06-24-ct-census-gt-validation.md Table C -->

**Johannesburg** (25-grid CBD clean_gt face): aggregate area calibration
bulk = 0.995, median cell ratio 0.983, σ_Bw 0.156, RMSE 878 m², through-origin
slope 0.979, R² 0.961, within ±20 % 0.800; agg_area_F1 on the same face 0.821
(from the CBD evaluation chain).
<!-- [F] spatial_eval_summary.json (bulk 0.9948) + 2026-06-24 report Table C JHB
     column + Table B (agg-F1 0.821, σ_Bw 0.180 on ch3_registered face — NOTE the
     two JHB columns are different runs of the same face; keep Table C numbers as
     the delivered-census claim and Table B as the selection-face reference) -->

**Headline discipline.** The raw installation counts (111,801 CT / 41,393 JHB)
never appear without their area-calibration ratio and its caliber: bulk is an
*area* caliber measured *on the GT-bearing subset* (CT: net area over-estimate
≈ 8 %). Area bulk must not be used to deflate counts; a count-caliber calibration
interval is [PENDING: to be derived — candidate approach: cov50 + per-cell count
regression on the 99-cell face].
<!-- [F] storyline v2 §1.2 round-3 correction -->

**Dispersion asymmetry, read correctly.** CT per-cell dispersion (σ_Bw 0.52) is
3× JHB's (0.16). This is a GT-face-composition effect, not a quality regression:
the CT GT face includes many tiny-GT cells (25th percentile 275 m²; min 7.5 m²)
where a few m² swings the ratio, while the JHB face is uniformly large commercial
roofs. The robust log-ratio dispersion (0.39 vs 0.20) narrows but does not erase
the gap; RMSE is *lower* for CT (695 vs 878 m²).
<!-- [F] 2026-06-24 report, "Reading it" + dispersion paragraphs -->

**Known tail failures (disclosed).** Four CT cells are zero-prediction misses
(GT 32–139 m² each); one dense-roof over-paint outlier (CPT1682, ratio 7.1); one
cell dropped for corrupt GT. <!-- [F] 2026-06-24 report Table C tail -->

### 4.4 The ground-truth ceiling, decomposed

The CT evaluation GT is SAM2 sub-array-level annotation (Axis-A tier A2), not
installation-level gold. Under this GT, the measurable polygon-F1@0.5 ceiling is
≈ 0.85–0.88; of the apparent ~12 pp gap to curated-benchmark literature numbers,
7–9 pp is metric/GT artifact, leaving a true model shortfall of ~4–6 pp.
<!-- [F] docs/plans/2026-06-10-rcnn-f1-gap-review.result.json;
     memory f1_gap_review_2026-06-10 -->
This decomposition **replaces** any comparison against curated benchmarks
(Dice ≈ 0.998-class numbers are a different caliber on different GT and are not
comparable); we promote it to a methodological contribution: a census on
imperfect GT should publish its ceiling, not borrow someone else's.

### 4.5 Instrument reliability and replicability

The detector and post-processing are deterministic; evaluation uses a single
metric kernel, config-matched output reuse, a frozen evaluation GT, and a locked
merge-mode + operating point, so every published number is re-derivable from its
source tiles and model run.
<!-- [F] docs/validation/2026-06-21-validation-methodology.md §2 -->
(The only stochastic instrument in the wider project — the LLM date-scorer — is
not part of this census release; see §7.)

---

## 5. Data records

<!-- PENDING: Zenodo deposit + DOI. ESSD requires the dataset deposited with DOI
     before/at submission — verify exact mechanics (the research memo covering
     this was lost; re-verify). -->

| File | Contents | CRS | Format |
|---|---|---|---|
| `ct_full_inventory_2026-06-21_merged` | 111,801 installation polygons, Cape Town | EPSG:32734 | GeoParquet + GPKG |
| `jhb_full382_..._fpcut_2026-06-01` | 41,393 installation polygons, Johannesburg | EPSG:32735 | GeoParquet + GPKG |
| per-cell aggregates | cell id, installation count, panel area (m²) | — | CSV/Parquet |
| validation pack | per-cell GT comparison tables (§4.3) | — | CSV |

Attributes per installation [PENDING: freeze exact column list against released
schema]: geometry; area_m2; detection confidence; source grid cell; imagery layer
id + model run id (provenance, recorded in every run's config.json);
merge component id.

**Install-date columns.** This release is census-only. The CT date block ships
empty with `undated_reason=ct_backdating_pending` (schema contract). JHB
install-date columns are **withheld** from v1: dates exist internally but no
independent accuracy claim is permitted before the temporal validation chain
completes (§7). <!-- ← OWNER DECISION still open per essd README: confirm
withhold-vs-ship-with-caveat for JHB dates. Draft assumes WITHHOLD. -->

---

## 6. Usage notes and limitations

1. **Intended analysis unit.** The inventories are validated for grid-cell
   aggregate use (counts, areas, spatial distribution). Per-polygon boundaries are
   diagnostic quality; individual-installation area should be used with the §4.3
   dispersion figures in mind.
2. **Area calibration.** Aggregate panel area is over-estimated ≈ 8 % in CT
   (bulk 1.083) and approximately unbiased in JHB (0.995) *on the respective GT
   faces*; users needing calibrated totals should apply these as caliber-specific
   corrections, not universal deflators, and must not deflate counts by area bulk.
3. **Coverage.** CT covers the 2,083 cells with municipal WMS imagery (of 4,429
   full-metro cells); JHB covers the 382-cell metropolitan grid. Absence of
   coverage is recorded, not conflated with absence of PV.
4. **Confusion classes.** Residual false positives concentrate in small dark roof
   furniture (water heaters, skylights) despite two-track suppression; the
   human-review precision channel measured 0.749 on a targeted audit sample
   (caliber: human judgment, not IoU — see §4.1 table).
   <!-- [F] memory ch1_precision_results; FP concentration s_xs×c_lo -->
5. **Vintage.** CT reflects 2025 imagery; JHB reflects 2024 imagery. The
   inventories are snapshots, not a time series (§7).

---

## 7. Living-data extension: the temporal panel (not in this release)

A companion workstream dates each Johannesburg installation via historical
imagery time series (first-visible-appearance semantics, interval-censored
estimates). It is excluded from this release because its validation chain
(imagery displacement audit → scorer fidelity gate → per-target panel rebuild →
human gold-set adjudication) has not closed; no independent accuracy number
exists yet, and this paper cites none. When the chain closes, the panel is
planned as an ESSD living-data update to this dataset.
<!-- [F] storyline v2 §4 claim-gate table; P0 chain §5. This paragraph is the
     ONLY place the temporal work may appear, and it makes no accuracy claim. -->

---

## 8. Code and data availability

- **Data**: [PENDING Zenodo DOI], CC-BY. Cape Town derived vectors from municipal
  public WMS imagery (unencumbered). Johannesburg derived vectors published under
  the Vexcel academic partnership programme; imagery credited per programme terms
  [PENDING: exact credit line from agreement; keep agreement copy on file].
- **Code**: detection, finalization and validation code at [PENDING repository
  DOI/URL]; the metric kernels (`core/area_metrics.py`,
  `core/polygon_validation.py`, `core/eval_matching.py`) reproduce every table in
  §4 from the released per-cell files.
- **Reproducibility**: each production run records imagery layer id, model run id
  and full parameters (config.json); a license-clean reproduction subset
  [PENDING: define — candidate: CT-only, fully open imagery] allows end-to-end
  re-derivation without third-party imagery agreements.

---

## PENDING register (pre-submission gates for this draft)

| # | Item | Blocking | Owner note |
|---|---|---|---|
| P1 | Global-product positioning literature check (§1.2 numbers) | §1.2 stays [H] until done | research memo lost with /tmp — re-run deep-research pass |
| P2 | Zenodo deposit + DOI; confirm ESSD deposit mechanics | §5, §8 | |
| P3 | Vexcel academic-programme credit line + agreement on file | §2, §8 | license question resolved 2026-07-12 (academic partnership); only citation text pending |
| P4 | Count-caliber calibration interval (headline discipline, §4.3) | abstract/§4.3 | new analysis, cheap: 99-cell count regression |
| P5 | JHB dates: confirm withhold vs ship-with-caveat | §5 | draft assumes withhold |
| P6 | Freeze released schema column list | §5 | |
| P7 | 9-gate FP-cut definition doc linked explicitly | §3.3 | locate/verify gate doc path |
| P8 | JHB installation-count area total after FP-cut | §5 table | compute from fpcut gpkg (pre-fpcut merged area was 1.415 km² at c0.925 <!-- [F] spatial_eval_summary.json merge_totals --> ) |
