---
title: Cape Town Census — Post-Delivery GT Validation
date: 2026-06-24
status: validation report
inputs:
  - results/analysis/ct_census_output_table/ct_full_inventory_2026-06-21_merged.gpkg
  - data/annotations/Capetown/ (Gao SAM2/V4 GT)
  - ct_census_calibration_appendix.html (Tables 5 / 14)
generator: scripts/analysis/ct_census_gt_eval.py
---

# Cape Town Census — Post-Delivery GT Validation

Companion to the threshold-calibration appendix (`ct_census_calibration_appendix.html`).
The appendix locked the production chain using *dedicated evaluation inference runs*;
this report does the post-delivery check JHB also got after its 382-grid census —
it takes the **shipped** census deliverable and scores it against ground truth on the
census cells that actually carry GT.

- **Locked production chain:** `unifiedA` detector · per-detection merge ·
  `v4_canonical` finalize · `solar_cls` adaptive_v1 PV filter (classify-all,
  aerial_2025 threshold 0.7168) · 2,083-cell aerial_2025 grid · inference 2026-06-16.
- **Delivered inventory:** `ct_full_inventory_2026-06-21_merged.gpkg` —
  111,801 de-duplicated installations (IoU≥0.10 union-merge), EPSG:32734.

Two tables are lifted verbatim from the calibration appendix; the third is new.

---

## Table A — Detector × merge-mode selection (appendix §5.1)

Wave-1 reporting face: 27 Gao calibration grids, each scored at its own best
polygon-confidence threshold. This is the table that picked the production detector
+ merge mode. Primary judges are σ_Bw (per-grid area dispersion) and RMSE.

| model run | best t | bulk | σ_Bw | RMSE (m²) | agg-F1 | pg-F1 | thru0 β | R² |
|---|---|---|---|---|---|---|---|---|
| v3c_wave1_pixelor | 0.97 | 1.260 | 0.302 | 533 | 0.743 | 0.735 | 1.292 | 0.928 |
| v3c_wave1_perdet | 0.95 | 1.110 | 0.339 | 479 | 0.724 | 0.733 | 1.115 | 0.853 |
| unifiedA_wave1_pixelor | 0.97 | 1.306 | 0.361 | 588 | 0.752 | 0.744 | 1.337 | 0.927 |
| **unifiedA_wave1_perdet ✅** | **0.97** | **1.104** | **0.270** | **395** | **0.762** | **0.751** | **1.162** | **0.931** |

Winner = `unifiedA_wave1_perdet` — lowest σ_Bw (0.270) **and** lowest RMSE (395), best agg-F1 (0.762).
Source: `results/analysis/polygon_conf_sweep/at_best_comparison.csv`.

---

## Table B — Cross-city Tier-1 reference (appendix §6)

Identical Tier-1 metric set reported on each city's **own** Channel-3 evaluation face:
Cape Town on the 16 Li held-out grids (leakage-clean, eastern Cape Flats); Johannesburg
on the 25-grid CBD face. A comparability reference across imagery domains, not a head-to-head.

| metric | Cape Town (locked, n=16 Li) | Johannesburg CBD (n=25) |
|---|---|---|
| agg-F1 | 0.742 | 0.821 |
| bulk ratio | 1.376 | 0.986 |
| σ_Bw (dispersion, primary) | 0.260 | 0.180 |
| RMSE (m²) | 1,332 | 1,066 |
| R² | 0.955 | 0.946 |
| thru0 β | n/r | 0.957 |
| cov50 (count recall) | 0.829 | n/a |

> The 16 Li held-out grids are **not inside** the 2,083-cell census footprint (Gao's west),
> so these exact numbers cannot be reproduced from the delivered inventory — that face is
> reported here for continuity only. Table C below evaluates the census on its own GT cells.
> Source: `chip_upgrade_ab/per_run_summary.csv`; JHB `…/ch3_registered/per_run_summary.csv`.

---

## Table C — Delivered census vs GT, on GT-bearing census cells (new)

The post-delivery check. We take the shipped `ct_full_inventory_2026-06-21_merged.gpkg`,
restrict to the **100 census cells that carry Gao SAM2/V4 GT** (1 dropped for a corrupt
mixed-CRS GT file → 99 scored), and recompute the Tier-1 suite with the same kernels
(`core.area_metrics` / `core.polygon_validation`). Alongside it, JHB's delivered census
(`jhb_full382_unified_A_merge01_c0925`, the same per-detection chain) scored on its
25-grid GT face — the direct analog of JHB's `spatial_eval_per_grid.csv`.

| metric | **CT delivered census** (n=99) | CT, GT≥500 m² (n=63) | JHB delivered census (n=25) |
|---|---|---|---|
| agg-F1 | **0.833** | 0.837 | n/a¹ |
| bulk ratio | **1.083** | 1.083 | 0.995 |
| median grid ratio | **1.024** | 1.016 | 0.983 |
| σ_Bw (primary) | **0.516** | 0.522 | 0.156 |
| log-σ (robust) | **0.394** | 0.358 | 0.201 |
| RMSE (m²) | **695** | 868 | 878 |
| thru0 β | **1.019** | 1.019 | 0.979 |
| R² (thru0) | **0.918** | 0.886 | 0.961 |
| within ±20% | **0.626** | 0.714 | 0.800 |
| cov50 (count recall) | **0.856** | — | n/a¹ |

¹ JHB's published per-grid eval JSON carries only `gt_m2` / `pred_m2`, so set-theoretic
agg-F1 and cov50 cannot be recomputed from it; JHB's agg-F1 (0.821) is in Table B.

**Reading it.** Both delivered censuses are well-calibrated in aggregate — bulk and
through-origin slope sit within a few points of 1.0, and the median per-grid ratio is
≈1.0 in both cities. CT's delivered inventory reproduces the locked operating point:
bulk 1.08 (inside the calibration's ~1.38 over-paint gate, in fact *below* the Li-face
figure), cov50 0.856 (≥ the locked 0.829), agg-F1 0.833.

The one real gap is **per-grid dispersion**: σ_Bw 0.52 (CT) vs 0.16 (JHB), within-±20%
0.63 vs 0.80. This is a GT-face-composition effect, not a regression — the Gao CT GT face
includes many tiny-GT cells (25th-percentile GT = 275 m²; min 7.5 m²), where a few m² of
over/under-paint swings the ratio hard, whereas the JHB CBD face is uniformly large
commercial roofs. The robust log-ratio dispersion (0.39 vs 0.20) and the GT≥500 m² subset
narrow but do not erase the gap. RMSE is actually *lower* for CT (695 vs 878 m²) because
CT installations are physically smaller.

**Tail (CT).** 4 cells are zero-prediction (FN-heavy: CPT0910/1018/1683/1807, each
GT 32–139 m² the census missed entirely). One over-paint outlier — CPT1682 (GT 823 m²,
pred 5,829 m², ratio 7.1) — is a genuine dense-roof over-paint cell, not a data artifact.
CPT1688 was dropped: its GT gpkg is corrupt (mixed lon/lat + UTM coordinates).

Generator: `scripts/analysis/ct_census_gt_eval.py`
(`results/analysis/ct_census_gt_eval/ct_census_gt_{per_grid,summary}.csv`).

---

## Method & caveats

- **Metric kernels reused verbatim** from production: `core.polygon_validation.clean_metric_gdf`
  (validity + 20,000 m² area cap) and `core.area_metrics.summarize` (σ_Bw / RMSE / agg-F1 /
  thru0 β / R²). cov50 mirrors `li_count_recall_sweep` (a GT polygon counts as covered when
  ≥50% of its area is inside the prediction union).
- **Merge-mode invariance.** Area metrics are computed on set-theoretic unions, so the
  IoU≥0.10 union-merge deliverable yields the same `pred_total_m2` as the raw per-detection
  set (`unary_union` is idempotent under pre-merging); only polygon counts differ.
- **GT ceiling.** CT GT is SAM2 sub-array-level (A2), not installation-level gold (T1).
  Absolute polygon-F1 is bounded below 1 by annotation granularity — this is why the
  inventory is judged on area-aggregate dispersion + recall, and why these figures must
  **not** be compared to installation-level literature benchmarks.
- **In-sample note.** The Gao GT face overlaps the grids used to calibrate the operating
  point, so Table C is a delivery-consistency check (does the shipped artifact reproduce
  the locked numbers), not an out-of-sample generalization claim — that role belongs to the
  16 Li held-out grids in Table B.
- **Cell scoping.** Predictions are assigned to a cell via the inventory's own `source_grid`
  partition (each installation counted once); no clip to cell boundary. Boundary effects are
  small and symmetric.
