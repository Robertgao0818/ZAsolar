# Cross-grid summary v1 (2026-10-17, read-only check; CPU only, sealed/ not opened)
Independent recount: `cross_grid_summary_v1/recount_v1.py` -> `recount_v1.json`. Claude's raw output copied to `cross_grid_edge_v1/xgrid_claude_out_20261017.md`.

## 1. Polygon base (the "1,246 vs 2,068" question)
**2,068 is not a Cape Town number.** It is the Joburg JHB25 clean_gt count (ROADMAP_v4 D2). It has nothing to do with the Li Cape Town test set. Both numbers are correct; they describe different datasets.

Li Cape Town test set, from `li_later_all_raw.gpkg` and the z2 package:
| stage | count | area m² | use for |
|---|---|---|---|
| raw features in file | 1,493 | – | nothing (233 are labeled=False) |
| labelled Li polygons | 1,260 | 40,707.8 | the source label set |
| ... wholly outside the 17 cells | 14 | – | dropped |
| ... partly outside | 12 | (821.1 lost in total) | clipped |
| source polygons touching the cells | **1,246** | 40,470.5 (full) / 39,886.7 (clipped) | edge analysis ("how many panels") |
| per-cell scoring pieces = gt.gpkg rows | **1,260** (= 1,246 + 14 split across internal lines) | 39,886.7 | **scoring GT**. A 447 / B 403 / C 349 / D 61 |
| touching polygons dissolved | 1,154 components | – | only if 葉 chooses to merge touching labels |
The two 1,260 counts are a coincidence: 14 dropped and 14 split. They are not the same polygons.

## 2. Recomputed numbers (own script). All match Claude's.
- Within 20 m of any cell edge: 178 polygons, 9,006 m² (22.3 % of area). Within 10 m: 105 polygons, 14.2 %.
- Crossing the outer test boundary: 12 (1,923 m²). Crossing internal cell lines: 14 (467 m²).
- Clip loss: 821.1 m² (2.0 %); 26 = 12 partial + 14 wholly outside.
- 61 of 442 tiles span 2–4 cells (126 of 507 units).
- **One wording fix:** "119 (7.4 %) double counted" was mixed up. 119 polygons (9.6 %; 7.4 % of area) fall in ≥2 production grid bboxes, so they get *processed* twice. Whether they end up *counted* twice depends on dedupe (issue 4).

## 3. Mechanisms, checked against code
- (a) **True.** The Joburg gate uses the *same* DINOv2 presence pregate, applied on 3×3 cells of 667 px (33 m) per tile (`joburg25_vexcel/pregate/pregate.py:2-11`). In eval, `evalobj_jhb25_par.py:87-91` builds the keep-mask, and `:185` computes `probg=prob*km` on the stitched probability map, which zeroes kept-neighbour-window predictions inside rejected cells. This is the Joburg eval/MiT path, not a production census.
- (b) **True.** In `detect_direct.py:512-531` a rejected 400 px chip simply isn't run through the detector, and other chips are untouched. The Cape Town template doesn't pass `--pregate-head` (`detect_direct.py:18-21,28-30`; 0 matches in `scripts/runpod_detect_direct_template.sh`). Durban's driver runs with it (`scripts/gate_pilot/durban_citywide.py:2`).
- (c) **True.** `scripts/ct_census_merge.py:58` only tags `gridcell_id = g` and concatenates (`:69-73`). There is no ownership or dedupe rule. After the separate IoU-0.10 merge: 386 overlapping cross-grid pairs remain, but only 3,680 m² (0.07 % of 5.0 km²), 384 of them with IoU ≤ 0.1. 3,077 polygons (2.8 %, 209k m²) have their representative point outside the tagged cell. These are halo detections and not necessarily wrong, but per-cell counts are off by about that much.

## 4. Issue table
| # | issue | verified number | test score? | status |
|---|---|---|---|---|
| 1 | Polygon base confusion | CT test 1,246 src / 1,260 pieces; 2,068 = Joburg | no | **resolved** |
| 2 | r2 clip loss wording | 821 m², 26 = 12 + 14 | wording only | r3 text addendum proposed; on hold |
| 3 | Panels near cell edges | 178 / 22.3 % area within 20 m | no (test = whole-tile, ungated MiT) | disclosure only |
| 4 | Census merge has no single owner | 386 pairs / 3,680 m²; 3,077 mis-tagged | no, production only | fix proposed (owner = cell containing rep point); validate on CT val |
| 5 | Gate zeroes neighbour predictions (33 m cells) | 421/1,246 polys cross gate/tile lines; 8,256 m² exposed | no (not used in CT test) | matters only if a gated MiT city run happens; fix before that |
| 6 | detect_direct pregate per chip | edge panels can be dropped by both chips | no; CT census ungated, Durban gated | low; measure on Durban if needed |

## 5. Decisions for 葉 (max 3)
1. Keep the r2 scoring rules, and add the r3 text addendum (26 = 12 + 14; test score is an upper bound for gated/tiled pipelines)?
2. Fix census merge ownership (one object → one cell), checked on CT val only?
3. Fix the gate context before any gated MiT city run, or keep the gate out of the census?
