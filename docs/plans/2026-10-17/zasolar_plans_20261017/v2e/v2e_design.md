# v2e design (2026-10-17 NZDT) — Phase-1 training set

葉's decision: v2e = v2c labels + G1189/G1190/G1238 (as v2d2) + G1630 seg130 fix + G1972 row-113 tightening (v2d1),
WITHOUT the v2d SAM3 thin-strip replacements in G1971/G1687/G2028/G1240. Val/eval unchanged.

## Where the 4 thin-strip grids sit (v2c and v2d2 manifests)
| grid | split | v2e action |
|---|---|---|
| G1971 | train | revert to v2c labels/masks (v2d1 00047 revert becomes moot: v2c = human label) |
| G1687 | train | revert to v2c |
| G1240 | train | revert to v2c (1 tile differed) |
| **G2028** | **val** | **FLAG**: val is "unchanged" = keeps v2d2 val (SAM3 thin-strip in G2028 stays). Not changed silently. |
Also G1630 (val) carries both the seg130 fix and 6 SAM3 thin-strip adds; v2d2 val kept as is.
Proposal for 葉 (not applied): keep val = v2d2 so v2c/v2d1/v2d2/v2e val scores stay comparable; if a v2c-label val is wanted, make it a separate val_v2c_labels view, never edit v2e val.

## Construction (copy-based patch of v2d2; originals read-only)
- Start: `~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017/data` (copy).
- For train grids G1971/G1687/G1240: masks, manifest rows, chip-index rows, collectors/HN rows taken from v2c (`ct_seg_trainset_v2c_20261015`), paths remapped to v2e.
- Everything else byte-identical to v2d2: G1972 tightening + slivers (v2d/v2d1), G1189/1190/1238, G1630 val fix, G1682 zero-area drops (0 px).
- Kept as-is and flagged: v2d sliver→255 edits in train G1749/G1972/G2029 (not SAM3 thin-strip; 葉 to confirm).
- Labels copy: `~/zasolar_data/geid_temporal/ct_labels_corrected_v2e_20261017/Capetown/` = v2d1 copy with G1971/G1687/G1240 SAM2 files replaced by the originals from `~/projects/ZAsolar/data/annotations/Capetown`.
- Output: `~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2e_20261017/` + `grid_manifest_v2e.csv` (grid_id CPT####, source_grid_id, split, polygon_count, label_file, md5), v3.2 conventions.
- md5: all inputs (v2c, v2d2, labels) before/after; all outputs listed in MD5SUMS.

## Expected counts
Train 79 grids. Train polygons (non-empty label rows, v3.2 definition): 5,648 (v2d2) − 2 (G1971) − 1 (G1687) = 5,645; brief said ~5,644.
Val 14 / eval 5, val/eval masks byte-identical to v2d2.

## Not done (before launch)
- HN coord cache (`hn_clean`) must be rebuilt for v2e (masks changed in G1971/G1687/G1240); trainer copy `train2_v2e.py`. No training.
