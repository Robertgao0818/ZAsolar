# Context
Design: ~/projects/zasolar_plans_20261017/v2e/v2e_design.md (read it fully).
# Ask
Write ~/projects/zasolar_plans_20261017/v2e/build_v2e.py and run it (CPU only, CUDA_VISIBLE_DEVICES='') to build v2e exactly per the design:
1. md5 every file under v2c data, v2d2 data and ~/zasolar_data/geid_temporal/ct_labels_corrected_v2d1_20261016 -> v2e/md5_inputs_before.txt.
2. Copy v2d2 dir (README+data) to ~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2e_20261017 (real copies, no links); rewrite every path string 'ct_seg_trainset_v2d2_20261017' -> 'ct_seg_trainset_v2e_20261017' in CSVs.
3. For train grids G1971, G1687, G1240: delete the copied v2e mask files of those grids, copy v2c masks in (assert the same file-name set), and replace their rows in manifest.csv, chips_512px_index.csv, collectors.csv, hard_negatives.csv, grids_swh_labelled.csv (if present) with the v2c rows (paths remapped v2c->v2e). Keep the v2d2 row order for all other rows; put replacement rows in the positions of the removed ones where possible (same (grid,image) key order as v2d2 manifest). Drop v2d_tile_diff.csv rows of these 3 grids.
4. Labels copy: ~/zasolar_data/geid_temporal/ct_labels_corrected_v2e_20261017/ = copy of ct_labels_corrected_v2d1_20261016, then overwrite (in the copy only) Capetown/G1971_SAM2_260403.gpkg, G1687_SAM2_260403.gpkg, G1240_SAM2_260322.gpkg with the originals from ~/projects/ZAsolar/data/annotations/Capetown/.
5. Update summary.json (add 'v2e' key: from, reverted grids, pv_px per split, chips per split, tiles per split) and write a README.md for v2e.
6. grid_manifest_v2e.csv in data/: copy ~/zasolar_data/capetown_manifest_v3_2_20261017/manifest.csv rows with split train/val/eval_orig10, label_file paths repointed to the v2e labels copy, polygon_count (non-empty geometries) and md5 recomputed for the 3 reverted grids.
7. md5 inputs again -> md5_inputs_after.txt and assert identical; md5 all v2e outputs -> v2e dataset dir MD5SUMS.
# Constraints
Never write into v2c/v2d/v2d1/v2d2 or originals; refuse if output dirs exist and are non-empty. No GPU, no training. Do not change val/eval.
# Done when
Script ran successfully, prints counts; write a reply ..._reply.md next to this file with what was done and counts.
