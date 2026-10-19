# G6 joburg_cbd25 official re-score — RESULTS v1 (2026-10-19, NZDT)
Scorer: evalobj_jhb25_p1cpu.py (scoring code = unmodified copy of joburg25_vexcel/evalobj_jhb25_par.py, md5 daeebe43), DEV=cuda bf16, same clean_gt, pregate 0.45, 25 grids. Headline = with pregate (pg_).

| model | cfg | P | R | F1 | pxIoU | R_lt4 |
|---|---|---|---|---|---|---|
| ConvNeXt-T s0 | h0.9_0.5 | 0.808 | 0.922 | 0.861 | 0.832 | 7/26 |
| ConvNeXt-T s0 | t0.5 | 0.771 | 0.922 | 0.840 | 0.831 | 7/26 |
| b2 (mit_b2) s0 | h0.9_0.5 | 0.819 | 0.965 | 0.886 | 0.859 | 12/26 |
| b2 (mit_b2) s0 | t0.5 | 0.773 | 0.968 | 0.860 | 0.859 | 13/26 |
| v2c rerun | h0.9_0.5 | 0.856 | 0.947 | 0.899 | 0.860 | 20/26 |
| v2c rerun | t0.5 | 0.840 | 0.952 | 0.892 | 0.860 | 21/26 |
| v2c official TABLE.md | h0.9_0.5 | 0.856 | 0.947 | 0.899 | 0.860 | 20/26 |
| v2c official TABLE.md | t0.5 | 0.840 | 0.952 | 0.892 | 0.860 | 21/26 |
| v2d1 official TABLE.md | t0.5 | 0.811 | 0.947 | 0.874 | 0.861 | 10/26 |

v2c verify: rerun matches official at both cfgs (P/R/F1/IoU/R_lt4 identical); chain_gpu.out "mismatches 0".
Runtimes (wall): v2c 29:09, b2 s0 37:23, convnext 1:02:57. Chain 12:03–14:12 NZDT.
Raw: out/{v2c_verify,p1_mit_b2_s0,p1_convnext_t_s0}.json; per-grid out/part_<name>/G*.pkl + G*.log; logs/<name>.log, logs/chain.log.
