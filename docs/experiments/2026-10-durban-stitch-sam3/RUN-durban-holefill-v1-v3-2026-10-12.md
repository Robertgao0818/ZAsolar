# Durban census: interior holes, hole fill v1, slit fill v2, 葉's verdicts v3

Date: 2026-10-12 (NZDT). Machine: rog (RTX 4070). Cost $0 (Gemini 3.5 flash lite via local proxy, free Vexcel academic access, no RunPod).
Work dir: `~/zasolar_data/durban_stitch_sam3_20261009/` (`hole_diag_013743/`, `holefill_v1/`, `slitfill_v2/`, `holefill_v3/`). Nothing in the delivery folder was changed. Nothing was committed.

## Why 013743 had a 160 m2 square hole

The presence gate is not the cause. Every chip over the hole passed the gate (0.94-0.99). The hole sits in chip #1169 of DBN0262, the last chip before a tile seam (tiles do not overlap), so it is the only chip that sees the whole hole.
Re-running the detector on that one chip from the cached extract reproduced the surviving detection (0.78, 225 m2) and showed a second detection of about 625 m2 covering 99% of the hole with score 0.59-0.63. Masks of 200 m2 or more need score >= 0.70 (v4_canonical tiers), so it was dropped.
City-wide only 59 interior holes are >= 20 m2 (about 3,700 m2). Of the 43 that sit in cells with a rebuildable chip grid, 42 lie entirely in chips the detector ran on. A rule that force-includes gated-out chips would add about 3,200 chips (all 4 neighbours passed), 26,000 (3 of 4) or 560 (all 8), and would not fix this kind of hole. It was not run.
Files: `hole_diag_013743/` (hole_diag_013743.png, holes_city.gpkg/csv, chipgrid.py, neighbour_rule_count.py).

## holefill_v1: fill the 59 holes directly

Per hole: SAM3 text and text+box coverage of the hole on cached or newly fetched imagery (45 fetches, 75 MB), plus Gemini dual-scale crops (59 calls).
Fill when SAM3 coverage >= 0.5 and Gemini says pv (conf >= 0.8), only in blocks not flagged fp_candidate / fp_flag_score. Disagreement -> review.

| decision | holes | hole m2 | added m2 |
|---|---:|---:|---:|
| filled | 10 | 609 | 600 |
| review | 32 | 1,900 | 0 |
| leave | 17 | 1,172 | 0 |

26 of the 32 reviews were SAM3 high coverage vs Gemini "not panel" (mostly walkways, bare roof, rooftop equipment); in the examples Gemini was usually right.
The first inventory write left zero-area sliver rings from a 32736->3857->32736 round trip; `fix_geometry_v1.py` rebuilt it by removing the original ring exactly (draft kept as `_draft_slivers.gpkg`).
Output: `holefill_v1/durban_census_inventory_stitch025_holefill_v1.gpkg` (column `area_added_hole`).

## slitfill_v2: thin slits

All interior holes < 20 m2 in holefill_v1: a slit is a ring with no point >= 0.3 m from its edge (width < 0.6 m). Slits were closed except in fp_candidate / fp_flag_score blocks.
2,110 slits closed (711 m2, 1,007 blocks; median 0.2 m2, max 5.4 m2, incl. the 5.4 m2 slit in 013743). 594 slits in flagged blocks left (94 m2). 1,144 compact small holes left (3,842 m2; may be real HVAC / skylights).
825 thin notches open to the block edge (839 m2, 603 blocks) were counted but not changed.
Output: `slitfill_v2/durban_census_inventory_stitch025_holefill_v2.gpkg` (column `area_added_slit`).

## holefill_v3: 葉's verdicts and ground truth

葉 judged all 32 review holes on the review page (`~/zasolar_data/durban_hole_review_20261012/review_verdicts_20261012.csv`): 7 pv holes filled exactly (+331.7 m2, column `area_added_hole_review`), 25 not pv left (1,567.7 m2).
葉 also marked g0.25_026531 (XL dark roof, 2,632 m2) as entirely not solar. It is recorded with 000715 and 030281 in `fpflag_v1/anchor_corrections_20261012.csv` and appended to `fpclean_v1/fpc_common.py` ANCHORS. It had no hole or slit fill, so nothing was reverted.
v3 adds `gt_label`, `gt_fp`, `gt_source` (flag only, nothing deleted). Other blocks are geometry-identical to v2.
Output: `holefill_v3/durban_census_inventory_stitch025_holefill_v3.gpkg`, `README.md`, `totals_v3.json`.

## City totals (stitch025_v1 -> holefill_v3)

| | value |
|---|---:|
| blocks (v1 and v3) | 30,604 |
| total area v1 | 1,888,277 m2 |
| total area v3 | 1,889,920 m2 |
| added by auto hole fill | 600 m2 (9 blocks) |
| added by 葉-reviewed hole fill | 332 m2 (7 blocks) |
| added by slit fill | 711 m2 (1,007 blocks) |
| all additions | 1,643 m2 (+0.09%) |
| ground-truth FP blocks (gt_fp) | 3 blocks, 2,938 m2 |
| fp_flag_score blocks (score_max < 0.95, >= 100 m2) | 828 blocks, 211,809 m2 |
| flagged (union; the 3 GT FPs are inside the score set) | 828 blocks, 211,809 m2 |
| total area excluding flagged blocks, v3 | 1,678,112 m2 |
| same blocks in v1 | 1,676,469 m2 |

fp_flag_score is a free rule, not a verdict: on the 350 labelled blocks it also catches a few true panels. Use the fpclean_v2 identity step (score/frac50/SAM3 -> Gemini -> keep/fp_candidate/review) for a cleaner FP split before reporting a net city total.
