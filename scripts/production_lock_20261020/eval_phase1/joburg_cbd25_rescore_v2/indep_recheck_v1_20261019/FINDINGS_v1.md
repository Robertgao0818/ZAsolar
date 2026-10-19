# G6 Joburg independent recheck v1 (2026-10-19 NZDT)
Own re-aggregation (recheck.py) of out/part_*/G*.pkl, CPU, no GPU, ~23 s, no tmux. Results: recheck_results_v1.json, recheck_stdout_v1.txt.
- All 12 claimed numbers reproduced exactly at pg_h0.9_0.5 (n_gt 2065, 25 grids, 2500 tiles, same GT keys for all 3 models).
- Small-panel bin in RESULTS_v1 is R_lt4 = area <4 m2 (n=26), NOT 4-10 m2 (4-10 m2 bin n=274).
- Matching rule is the Joburg evalobj coverage rule (GT hit = >=50% px covered; pred TP = >=20% px in 8-px dilated GT zone), NOT zaeval-2@8044d14 (greedy 1:1 IoU>=0.5). zaeval-2 rescore impossible from pickles: prob maps were in /dev/shm and deleted; needs re-inference.
- Bootstrap is by GRID (25), not tile (pickles are per grid).
- v2c: same tiles/GT/pregate 0.45/h0.9_0.5 rule; ckpt evals/ft_best.pt (mit-b2, SegFormer HF path) vs p1 models via train2_v2e build_model. No Joburg in v2c manifest (0/25 grids, all paths cape_town 0.05 m); train2.py reads only v2c/v2b CT manifests.
