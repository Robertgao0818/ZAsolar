# v2e build reply (2026-10-17)

`build_v2e.py` finished with exit 0, CPU only (`CUDA_VISIBLE_DEVICES=''`). No GPU and no training. v2c, v2d, v2d1, v2d2, the v2d1 labels, and `projects/ZAsolar/data/annotations/Capetown` were not written. The input md5 of all 24,487 files is identical before and after (`v2e/md5_inputs_before.txt`, `v2e/md5_inputs_after.txt`).

## Outputs

- Trainset: `/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2e_20261017/` (real copies, no links, no hardlinks).
- Labels: `/home/gao/zasolar_data/geid_temporal/ct_labels_corrected_v2e_20261017/` (same tree as the v2d1 labels; three files overwritten by the annotation originals).
- `data/grid_manifest_v2e.csv` (98 rows: train 79, val 14, eval_orig10 5).
- `MD5SUMS` in the trainset directory lists 12,491 output files (the trainset except `MD5SUMS` itself, plus the labels copy). Absolute paths, two spaces, `md5sum` format.

## What changed

Started from a full copy of v2d2. CSV paths `ct_seg_trainset_v2d2_20261017` were rewritten to `ct_seg_trainset_v2e_20261017` (manifest 12,370, chips 24,729). Train grids G1971, G1687 and G1240 then had their copied masks deleted and the v2c masks copied in (same 126 filenames each). Their rows in `manifest.csv`, `chips_512px_index.csv`, `collectors.csv`, `hard_negatives.csv` and `grids_swh_labelled.csv` were replaced with the v2c rows, paths remapped to v2e, sitting in the original v2d2 positions (matched on the row key). `v2d_tile_diff.csv` dropped 378 rows for these three grids and kept 513.

Label overwrites, copy only:

| file | non-empty geometries | was in v2d1 |
|---|---:|---:|
| `Capetown/G1971_SAM2_260403.gpkg` | 354 | 356 |
| `Capetown/G1687_SAM2_260403.gpkg` | 200 | 201 |
| `Capetown/G1240_SAM2_260322.gpkg` | 7 | 7 |

`grid_manifest_v2e.csv` repoints `label_file` from `ct_labels_corrected_v2d1_20261016` to `ct_labels_corrected_v2e_20261017`. Empty label files (hn-only grids) stay empty. CPT1688 still points at `label_audit_classB_v1_20261014/diag/fixed_G1688_SAM2_260403.gpkg`, which is outside the labels copy. `polygon_count` and `md5` were recomputed for the three reverted grids and checked against the files. `status` and `notes` are still the v3.2 text (`frozen_v2d2`, "corrected copy"), including on the three reverted rows.

## Counts

| split | grids | tiles | chips (512 px) | PV pixels |
|---|---:|---:|---:|---:|
| train | 79 | 9,970 | 19,904 | 42,372,083 |
| val | 14 | 1,768 | 3,562 | 12,645,119 |
| eval | 5 | 632 | 1,263 | 2,440,167 |

Train `polygon_count` in `grid_manifest_v2e.csv`: 5,645. That is the v3.2 train total 5,648 minus 2 (G1971) minus 1 (G1687). G1240 stays at 7 polygons. The design's "~5,644" is this 5,645.

PV pixels on the reverted grids, v2e = v2c:

| grid | v2d2 PV px | v2e PV px | masks that differ from v2d2 |
|---|---:|---:|---:|
| G1971 | 4,010,259 | 4,067,840 | 41 / 126 |
| G1687 | 960,360 | 981,634 | 17 / 126 |
| G1240 | 5,001 | 5,221 | 1 / 126 |

Train PV is the v2d2 total 42,293,008 plus 79,075. Chips and tiles per split are unchanged from v2d2, because the filename sets matched.

Val and eval masks match v2d2 on every file. Checked directly: G2028 (val) 0 / 126 differ from v2d2 and 12 / 126 differ from v2c (PV px 979,293, the v2d2 number, not v2c's 998,312). G1630 (val) 0 / 127 differ from v2d2. `summary.json` has a `v2e` key with `from`, `reverted_grids`, and `pv_px` / `chips` / `tiles` per split. The headline `grids` / `tiles` / `pv_px` / `chips` objects are the stale pre-v2d numbers inherited from the v2d2 file (train grids 76). Use the `v2e` key or the table above.

## Left undone on purpose

`hn_clean` was not rebuilt, and `train2_v2e.py` was not written. The design says both have to happen before launch, because masks changed on G1971, G1687 and G1240.
