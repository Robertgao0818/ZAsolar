# JHB CBD 25-grid eval — independent analysis v1 (2026-10-17 NZDT)
Read-only, CPU only. Scripts + CSVs: `jhb25_analysis_v1/` (reagg.py, vec.py, boot.py). Source: `~/capetown_eval/ds3m_ft/joburg25_vexcel/`. Scoring rule unchanged (葉-fixed).

## 1. Independent recompute
- MiT v2c / SAT / v2d1: probabilities were in /dev/shm and deleted; only per-grid partials (`pregate/part/<model>/G*.pkl`) survive. `reagg.py` re-does union-find stitching, ≥0.25 m² / invalid<50% filters, ≥20%-in-zone TP and ≥50% coverage recall on those partials with its own code. **All reported numbers reproduce exactly** (P/R/F1/FP/ha, <4 m² 20/26, 20/26, 10/26; 4–10 m² 250/274, 255/274, 248/274). Pixel stage (rasterisation, inference) is NOT independently checked — would need re-inference (GPU).
- Mask R-CNN: fully independent **vector** implementation (`vec.py`, shapely, from predictions_metric.gpkg, gated by kept cells): P 0.677 (=), R 0.895 (vs 0.905), F1 0.771 (vs 0.775), FP/ha 0.369 (=), <4 m² 2/26 (vs 3/26), 4–10 250/275 (vs 255/274). Small R gap = raster vs vector coverage (all_touched) + 2068 vs 2065 objects; not a scorer bug.
- **Mismatch to flag: GT count is 2,068 polygons in clean_gt (2,065 scored), not 2,168.** clean_gt = union of Mask R-CNN v3c predictions reviewed "correct" + SAM-added (clean_gt.py), i.e. GT is detector-seeded.

## 2. Grid-bootstrap 95% CI (25 grids, 10k resamples)
| model | P | R | F1 | FP/ha | R 4–10 m² |
|---|---|---|---|---|---|
| MiT v2c h0.9_0.5 | 0.856 [0.808,0.890] | 0.947 [0.915,0.969] | 0.899 [0.865,0.922] | 0.100 [0.076,0.125] | 0.912 [0.817,0.962] |
| SAT interim 25k h0.9_0.5 | 0.817 [0.758,0.860] | 0.973 [0.963,0.984] | 0.888 [0.852,0.913] | 0.139 [0.106,0.173] | 0.931 [0.894,0.966] |
| MiT v2d1 t0.5 | 0.811 [0.751,0.852] | 0.947 [0.921,0.964] | 0.874 [0.835,0.900] | 0.144 [0.115,0.176] | 0.905 [0.825,0.950] |
| Mask R-CNN (vector recompute) | 0.677 [0.584,0.747] | 0.895 [0.831,0.938] | 0.771 [0.702,0.820] | 0.369 [0.296,0.453] | 0.909 |

Paired (same grid resamples) differences:
- v2c − SAT: F1 +0.011 [−0.008,+0.029] p=0.24 (**n.s.**); P +0.039 [+0.017,+0.066]; R −0.027 [−0.058,−0.006]; FP/ha −0.039. Object McNemar recall: SAT-only hits 79 vs v2c-only 24 (p<1e-4); <4 m² 2 vs 2; 4–10 7 vs 12 (p=0.36).
- v2c − v2d1: F1 **+0.025 [+0.013,+0.040] p<0.001**; P +0.045 [+0.027,+0.067]; R 0.000; FP/ha −0.044 [−0.060,−0.030]. <4 m²: v2c-only 10, v2d1-only 0 (p=0.002). 4–10: 12 vs 10 (n.s.).
- SAT − v2d1: F1 +0.014 [+0.001,+0.030] p=0.03; R +0.026; P n.s.
- Caveat: config mismatch — v2d1 reported at t0.5, others at h0.9_0.5; v2d1 at h0.9_0.5 = F1 0.881 (gap to v2c 0.018). h0.9_0.5 is SAT's best of 12 configs *on JHB* (t0.5 = 0.861); none were selected on CT-val → SAT/v2c numbers mildly optimistic.

## 3. Pre-gate impact (recall ceiling)
- 30.1% of cells kept. Vector check: 2,046/2,068 GT (98.9%) have ≥50% area in kept cells → ceiling R ≈ 0.989; 22 unreachable (bins <4:1, 4–10:6, 10–25:10, >25:5), 5 fully outside; concentrated in G0888 (5), G0922 (4), G0923 (4), G0854 (3).
- Same for all models (gate is model-independent). Mask R-CNN gated vs no-gate recall identical (0.905) → effective gate loss ≈0 for it; gate removed only FPs (P 0.665→0.677).
- Misses: v2c 110, SAT 55, v2d1 109 → ≤22 of each attributable to gate; most misses are model misses.

## 4. Per grid / FPs / error types (CSV: per_grid.csv)
- FPs fairly spread: top-5 grids hold 36–40% of FPs (v2c: G0855 23, G0923 22, G0817 19, G0814 18). Mask R-CNN: G0854 alone 106.
- Recall outliers: G0925 (v2c 0.684, v2d1 0.722, SAT 0.962, MRCNN 0.354), G0817 (v2c 0.811, SAT 1.00). Worth a visual check — likely a panel type MiT misses.
- FP type: ~99% of FPs have **zero** overlap with GT zone (isolated objects, not boundary spill), 75% with max prob ≥0.99, mostly 4–25+ m². Given GT is seeded from Mask R-CNN review, a sizeable share may be **real unlabelled panels or SWH** → precision likely underestimated for all models (bias favouring models similar to Mask R-CNN). Not checked visually.

## 5. Lenient criterion
- Recall ≥50% coverage and precision ≥20%-in-zone are permissive: split/merged arrays and spillover never count as errors, so F1 reflects detection, not delineation. Stricter IoU≥0.5 object F1 is **not cheap** from saved partials (no per-pair geometry) → not computed for MiT/SAT; skipped. Recommendation: the new unified eval module (in progress) should log per-object IoU so a supplementary strict column comes for free.

## 6. Decisions
- **SAT switch rule** (vs v2d1 on CT-val): (1) F1 ≥0.923 — not testable on JHB, CT-val full eval still pending; (2) 4–10 m² recall not lower — on JHB SAT 255 vs v2d1 248 hits (passes, n.s.); (3) cost ≤1.5× — **fails (3.1×/it; ≈3× per round)**. → Rule fails on cost regardless of JHB. JHB shows SAT = v2c within noise on F1 (higher recall, lower precision), no case to override.
- **v2d1 vs v2c**: on JHB v2c is significantly better (F1 +0.025, FP/ha −30%, <4 m² 20 vs 10 of 26), consistent with CT-val (0.926 vs 0.908). Conclusion: v2d1 regressed (precision + tiny panels); keep v2c as the reference until the v2c→v2d1 change is diagnosed.
- JHB is one city, 25 grids, detector-seeded GT: supports "v2c ≥ v2d1" and "SAT not clearly better"; does not by itself justify a model switch.

## Reviewer
Second review: see section 7 (Claude CLI) if present.

## 7. Second review (Claude CLI, read-only) and corrections — supersede §2/§6 where they differ
1. **Matched config (all three at h0.9_0.5; `jhb25_analysis_v1/matched/`)**: v2d1 P 0.827 [0.768,0.868] R 0.943 F1 0.881 [0.844,0.906] FP/ha 0.126, <4 10/26, 4–10 245/274. v2c − v2d1: **F1 +0.017 [+0.006,+0.031] p=0.005**, P +0.028, FP/ha −0.026 (−21%), R n.s.; <4 m² 10 vs 0 (p=0.002). SAT − v2d1: F1 +0.007 n.s.; R +0.030; 4–10 McNemar 15 vs 5 (p=0.04). Conclusion "v2c > v2d1" holds, smaller than §2 said.
2. h0.9_0.5 was the best of 12 configs picked on JHB; CIs do not include this selection → treat point differences ≤0.02 as within selection noise.
3. Mask R-CNN: GT seeded from v3c Mask R-CNN review (unifiedA is a different but related model); its FPs of the seed model were human-rejected while MiT/SAT FPs were never reviewed → MRCNN precision is NOT comparable as "underestimated"; its recall may be inflated by circularity. Treat MRCNN row as reference only.
4. "75% FP mp≥0.99" is uninformative under h0.9 (all kept ≥0.9). No SWH labels in JHB → SWH-as-FP hypothesis untestable here.
5. Pixel metrics already saved (supplementary, contour quality): px IoU v2c 0.860 / SAT 0.851 / v2d1 0.861 / MRCNN 0.729; boundary IoU 0.402 / 0.378 / 0.379 / 0.327 (boundary IoU known unreliable).
6. FP/ha denominator = whole valid area (2,495 ha) while only ~30% of cells are inferred; not comparable with no-gate numbers. Gate training data/threshold provenance (JHB-free?) not verified. "Gate loss ≈0" shown only for MRCNN; for MiT/SAT it is bounded by ≤22 objects.
7. vec.py FP/ha uses 2,500 ha (vs 2,495; negligible). 2068→2065 likely dedupe. SAT is an interim 25k ckpt → SAT conclusions provisional. 3.1× cost is from SAT_DECISION_RULE.md (1.19 vs 0.385 s/it), not recomputed.
