# G7 cross-tile / cross-grid merge (val only, CPU, no inference)
Code: xtile_merge.py (MergePlan two-pass cross-tile hysteresis + merge, full-extent overlap check, valid dissolved polygons, owner by representative point in real 1 km cell), eval_val_xtile_v2.py, tests/ (11 pass).
Commits: 52662ef (v1) -> abb4709 (r1 fixes) -> 9cc7999 (r3 fixes, final). No git remote; nothing pushed.
Codex: r1 NO-GO, r2 NO-GO, r3 NO-GO, r4 GO (logs/codex_review_r*.md).
Final run: out_full_v3_20261019/ (summary.json, cross_grid.json, grids/*.json, cross_grid_objects_*.gpkg), log logs/full_v3.log, tmux ct_xtile_merge, 26m44s CPU wall.
Recompute: python3 eval_val_xtile_v2.py NEW_OUTDIR 6  (refuses to overwrite an existing dir).
Superseded/empty: out_full_v2_20261019 (stopped). Smokes: out_smoke_v2b_G1629_G1630.
Note: object matching stays per grid (GT per grid); 28 tiles are shared between adjacent val grids.
