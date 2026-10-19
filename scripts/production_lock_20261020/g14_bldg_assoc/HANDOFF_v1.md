# G14 run v1 — STOPPED by the 2-consecutive-failure rule (2026-10-19 23:26 NZDT)
- Stage 1 OK (23:12:59–23:26): gt/CT_gt.gpkg (1,033 PV polys, 14 grids, 13.76 km²), gt/JHB_gt.gpkg (2,174, 25 grids); preds/ CT convnext 1,240 / b2(s2) 1,199 / avg2 1,163 objects;
  JHB convnext 1,770 / b2(s0) 1,808 / v2c 1,693 / avg2 1,632 (A3 h0.9_0.5). Max RSS 1.16 GB (cap 1.5 GB, systemd scope MemoryMax, no swap).
- Stage 2 failed: assoc_grid_v1.py named a column 'eval', so `bld.eval` resolved to DataFrame.eval -> "ValueError: unknown type object" on
  every grid (8 CT grids failed at ~23:26, all others skipped). Code bug, not memory. No metrics produced; parts/ empty.
- Fix prepared, NOT run: assoc_grid_v2.py (column renamed is_eval, otherwise identical) + run_all_v2.sh (skips finished stage 1). Rerun needs
  `echo 0 > logs/consec_fail` and approval to restart: tmux new -d -s g14_bldg_assoc './run_all_v2.sh'. Aggregation script not yet written.

## Rerun v2 (approved 23:27 NZDT): counter reset; 1-grid tests JHB G0772 + CT G1293 OK 23:27:19–23:27:28; full stage 2 (39 grids) 23:27:4x–23:28:05, 0 failures, max RSS 0.38 GB;
aggregate_v1.py → RESULTS_v1.json/.md, results_{building,stratum,rowsplit,area}_v1.csv.
