# Phase-1 v3 run record

Status: complete. M1 split and M2 grid-cluster CIs implemented and re-scored on saved masks only.

Source of truth: TEST_REPORT_v3.md; rescore_full_attempt2_v3/metrics.json; contract_checks_v3.json; bundle_verification_v3.json.

Decisions: exact training load_M identity/index/draw order; val_sel300=300, val_holdout=1403 (main), eval=632. Legacy run_v2c/best.pt log best pixel-IoU step=7500; reconciliation=not_comparable because legacy v2c labels/pool and no obj_f1 log/config. No claimed v2f best_objf1 reproduction or historical selection independence of the legacy checkpoint's holdout score. 9 shared images cross selection/holdout, documented without altering requested partition. SWH labelled empty grids included (8 val / 4 eval).

Uncertainty: 20,000 grid bootstrap draws, seed 20261018, percentile 95%, ratios of summed integers. All 12 interval recalculations and all 2335 raster checks passed. Eval object/pixel counts unchanged; val pixel partition sums to v2 full pool.

Execution: three isolated initial check failures (selection-test expectation, saved-output path guard, live-transcript hash expectation); corrected checks passed with no consecutive pair. Distinct attempt paths/logs/status retained. No inference, no GPU, no process termination or PID 374974 access, no original Li test/cache reads, no external mutations or uploads. Original static artifacts hash-stable; host transcript original prefix preserved while its external logger appends; LOG.md appended by authorization. Clean pod_bundle_v3/ with all static files in MD5SUMS and exact per-architecture commands in README_v3.md.
