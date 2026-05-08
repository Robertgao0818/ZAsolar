# V1.4 Validation Framework — Running Log

Channel results and post-processing calibration progress for the V1.4
four-channel validation framework defined in
[`docs/validation_strategy.md`](../validation_strategy.md). One section per
channel; append new entries chronologically inside each section.

---

## Channel 3 — Plausibility sanity checks

### 2026-05-05 — first pass on JHB CBD 25-grid + CT V3-C 46-grid

**Goal**: deliver `scripts/analysis/grid_plausibility.py` per the V1.4 spec
("density / solarizable-area / mean-polygon-area fall inside known physical
bounds, plus stratum-relative outlier flagging") and run it on the two model
inventories that are currently the headline runs:

- JHB CBD 25-grid Vexcel V3-C + SAM(mask+box) — current inventory winner
  (`results/johannesburg/v3c_sam_maskbox_vexcel_2024`)
- CT V3-C aerial_2025 (46 grids that have `predictions_metric.gpkg` under
  `results/cape_town/v3c_targeted_hn_aerial_2025`).

`solarizable_roof_area` is **out of scope for the RA dataset deliverable**
— downstream administrative / census building data covers that denominator.
This implementation uses grid-area as the density denominator and reports
`area_coverage_pct = total_install_m² / grid_m²` as a self-contained
sanity-check metric. The 25-grid Vexcel grids are 1 km² each, so
density/km² ≡ count/grid here.

#### Bounds (residential urban; CBD strata are looser)

Until the V1.4 stratum-tagging deliverable lands, every grid is treated as
`unstratified` against the permissive `default` band. Per-stratum bands are
already wired into the script for `residential / suburban / CBD / township /
peri_urban / rural` and will activate once `grid_strata.csv` exists.

| Stratum | mean area (m²) | median area | density/km² ≤ | coverage % ≤ | single install ≤ |
|---|---|---|---|---|---|
| default | 8 – 250 | 8 – 150 | 2000 | 10.0 | 5000 |
| CBD | 15 – 250 | 10 – 150 | 500 | 10.0 | 5000 |
| residential | 10 – 60 | 10 – 40 | 800 | 3.0 | 500 |
| township | 6 – 40 | 6 – 25 | 1500 | 2.0 | 300 |
| peri_urban | 8 – 80 | 8 – 50 | 400 | 2.0 | 800 |
| rural | 8 – 200 | 8 – 100 | 100 | 1.0 | 2000 |

Bound numbers were calibrated against the JHB CBD 25-grid clean-GT
distribution (per-grid mean area 25.5–205.4 m²; per-grid median 11.7–106.5;
coverage 0.05–1.74 %; max single install ≈ 1.7 k m²). They will tighten once
each stratum has its own GT prior.

In addition, per-stratum top/bottom 5 % of each metric is flagged as an
`info`-severity outlier (only fires when stratum has ≥ 10 grids).

#### Run 1 — JHB CBD 25-grid V3-C + SAM(mask+box)

Output: `results/validation/plausibility_20260505_v3c_sam_maskbox_vexcel_2024/`

- 25 grids analysed, 10 with at least one flag, **0 high-severity flags**.
- Per-grid summary:
  - `n_install` median = 93, range 21 – 312
  - `density/km²` median = 93, range 21 – 312
  - `mean_area_m²` median = 81.4, range 29.8 – 180.9
  - `median_area_m²` median = 26.5, range 17.2 – 96.2
  - `area_coverage_pct` median = 0.77, range 0.08 – 2.26
- All flags are stratum-relative outliers (5 % tails inside the 25-grid set).
  Notable:
  - **G0922 / G0923**: high count + density (238 / 312 per km²) — densest
    grids in the 25, plausible for inner-CBD suburbia, no other anomaly.
  - **G0816 / G0856**: high mean polygon area (180.9 / 146.9 m²) — driven
    by single very large installs (≈ 2 k m² in G0816). Consistent with the
    clean-GT large-array geometry.
  - **G0773**: low count (25) + high median area (96 m²) — sparse but
    commercial-skewed. Matches GT, not a model issue.
- **G0925** — known SAM-loss outlier from 2026-05-04 area_aggregate report
  (pred/GT = 0.52, missed three large arrays) — does **not** trigger any
  plausibility flag. Plausibility catches **distributional** failures, not
  recall holes; a low-recall grid that produces a normal-looking
  distribution slips through. Channel 2 already covers that failure mode.

The Vexcel JHB CBD 25-grid run passes Channel 3 cleanly: no out-of-bounds
metrics, no hallucination evidence, outliers all explainable from GT.

#### Run 2 — CT V3-C aerial_2025 (46 grids)

Output: `results/validation/plausibility_20260505_v3c_ct_aerial_2025/`

The formal `cape_town_independent_26` benchmark suite only overlaps this
inference run on 3 grids (G1570 / G1571 / G1572). Inference for the rest
of the 26-grid benchmark on `aerial_2025` has not yet been re-run, so this
analysis is on whatever 46 grids are currently present in
`v3c_targeted_hn_aerial_2025/`. Re-running on the formal 26 stays as a TODO
once GPU is free.

- 46 grids analysed, 15 with at least one flag, **5 high-severity flags**.
- Per-grid summary:
  - `n_install` median = 58, range 0 – 408
  - `density/km²` median = 57, range 0 – 408
  - `mean_area_m²` median = 22.4, range 6.4 – 113.5
  - `median_area_m²` median = 17.1, range 5.5 – 50.1
  - `area_coverage_pct` median = 0.20, range 0.00 – 1.36
- High-severity findings:
  - **G1977 / G2033 / G2035**: median area 5.5 – 7.7 m² (below 8 m² lower
    bound). Mean area in G2033 / G2035 also below 8 m². The lower-bound
    breach is the diagnostic Channel 3 was designed to catch — these grids
    likely have a fragmented-detection / small-FP failure mode that should
    feed Channel 1 RA review.
  - **G1457**: 0 installs (info-severity sanity flag, expected for empty
    grids; recorded as `zero_installs` not as a hallucination).
- Density outliers:
  - **G1634 / G1855 / G1971**: 313 / 385 / 408 per km² — very high. Could
    be real (CT is more residential-solar-saturated than JHB CBD) or could
    be an FP cluster from water-heater confusion. Channel 1 RA spot-check
    candidates.

Net: V3-C on CT 46-grid has a **distinct** failure mode from the JHB
inventory — small-fragment grids, not large-array misses. Channel 3 surfaces
this without needing GT.

#### Take-aways for next channels

1. Channel 3 alone does not catch G0925-style recall holes — that is
   Channel 2's job. The two channels are complementary.
2. The "median area below lower bound" flag is sharp enough to drive Channel 1
   stratified RA sampling — flagged grids should get up-weighted in the
   precision audit queue.
3. To make Channel 3 do its full job we still need `grid_strata.csv` so CBD
   vs residential bounds activate. The `solarizable_roof_area` denominator
   is downstream of the RA dataset deliverable (admin / census building
   data handles it) and is not tracked here.

#### Deliverable status

- [x] `scripts/analysis/grid_plausibility.py` — computes ratios, flags
      bounds violations + per-stratum 5 % outliers
- [x] `results/validation/plausibility_<YYYYMMDD>.csv` — produced for both
      JHB and CT runs (`per_grid.csv`, `flags.csv`, `summary.md`,
      `config.json` per run)
- [ ] grid_strata-aware bounds (blocked on stratum tagging deliverable)
- [ ] re-run on full `cape_town_independent_26` once V3-C inference is
      available for all 26 grids
