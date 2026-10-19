# VERIFY G7 cross-tile merge — independent recheck v1 (2026-10-19, NZDT, CPU only)

Scope: ConvNeXt-T s0 cached val probs, h0.9_0.5, 14 val grids, 1,703 tile entries (1,675 unique, 28 shared), 1,020 GT.
A0 = per-tile binarize, no merge; A1 = per-tile binarize + in-grid cross-tile merge. No existing file modified.

## 1. md5 + script rerun
- out_full_v3_20261019/summary.json md5 = aaed30914ee71f1214cf49b495e6055d (matches claim aaed3091…).
- Rerun `python3 eval_val_xtile_v2.py indep_recheck_v1_20261019/rerun_script 6` (tmux xtile_recheck): summary.json md5 identical (aaed3091…), agg dict equal. Log: rerun_script.log.

## 2. Independent scorer (indep_scorer.py)
Own 8-connected labelling, own union-find over global border pixels (in-grid), own valid-pixel IoU, own greedy 1:1 match
(desc IoU, ties pred idx then GT idx), min_area 0.25 m², drop preds with 0 valid px, GT in scope iff valid px > 0 —
rules read from zaeval/objects.py @ 8044d14 (mode exclude-pixels: no region drop). Reused for data prep only:
select_rows/validation_scopes, ref_for, load_grid_gt, class_labels, make_view, binarize.

| IoU≥0.5 | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| claim A0 | 885 | 303 | 135 | 0.745 | 0.868 | 0.802 |
| indep A0 | 885 | 303 | 135 | 0.7449 | 0.8676 | 0.8016 |
| claim A1 | 881 | 208 | 139 | 0.809 | 0.864 | 0.836 |
| indep A1 | 881 | 208 | 139 | 0.8090 | 0.8637 | 0.8355 |

Exact match (counts identical). IoU≥0.3 also identical (A0 F1 0.8270, A1 0.8592).
Area/IoU: pixel IoU 0.8947 and raw predicted area 42,758.9 m² are identical A0 vs A1 (same mask; merge only regroups pixels) — claim holds.
Caveats: area of *kept* objects (after the 0.25 m² filter) changes slightly, 38,820→38,908 m² (+0.23%), because small edge fragments
are no longer dropped once merged; mean IoU of matched TPs rises 0.9375→0.9573 (split objects now matched whole).

## 3. 28 shared tiles (dedup)
Each shared tile assigned to one grid: (a) owner = cell containing tile centre (4 tiles on a cell edge → first grid), (b) lexicographically first grid. Both give identical results.

| IoU≥0.5 | before (as-is) | after dedup |
|---|---|---|
| A0 unmerged | TP885 FP303 FN135, P0.7449 R0.8676 F1 0.8016 | TP885 FP301 FN135, P0.7462 R0.8676 F1 0.8024 |
| A1 merged | TP881 FP208 FN139, P0.8090 R0.8637 F1 0.8355 | TP881 FP207 FN139, P0.8097 R0.8637 F1 0.8359 |

GT count stays 1,020 (no GT polygon was double counted; shared tiles' GT belongs to one grid), only 2 (A0) / 1 (A1) duplicate FP objects removed.
Effect ≤0.001 F1; merge gain +0.034 F1 unchanged. Pixel-level metrics were not recomputed deduplicated (object-level only).

## Verdict
Claim reproduced exactly; dedup changes nothing material; conclusion (merge: F1 0.802→0.836, precision up, recall −0.004, pixel IoU/raw area unchanged) holds.
Files: indep_scorer.py, indep_out/indep_results.json, indep.log, rerun_script/, rerun_script.log.
