# Phase-1 evaluation pipeline run record

Status: complete. CPU smoke evaluation, visual render verification and independent pixel recomputation passed on first execution.

Scope: only the frozen v2f val/eval splits. No Li/test access, no coordinate-cache access, no external writes during development/testing, no GPU use, process termination or push.

Decisions:

1. Safety overrides executing imported `load_bundle()` / `prep()` / `_validate_body()`: these respectively access forbidden inventories/caches or require CUDA. The source is imported unchanged; evaluation reads pinned CSVs and calls the original normalization/model constructors instead. Details: README.md.
2. Default val scope is all 1703 training-eligible rows, not the fixed 300-tile checkpoint-selection subset. Eval scope is all 632 rows of the five v2f `eval_orig10` grids. Inventory: grids_and_tiles.csv.
3. Headline objects and boundary scores are imported zaeval-2 implementations at commit `8044d1447a92cc100bbc01976bf66cb4245c89bb`; object mode is `exclude-pixels`. PV/SWH use separate raw polygon sets. Pixel scores use original semantic masks and exclude 255.
4. Protocol does not prescribe a numeric seed or run-directory names. Use seed 0 (training default and smoke config), and requested `<arch>_s0` directories.
5. Smoke uses G1689 val and G1686 eval, two full tiles per grid, alternating PV-rich and SWH-rich choices. This is a functional check, not a representative performance estimate.

Evidence: TEST_REPORT.md, test_run_smoke/, contract_checks.json, architecture_checks.json and progress/failure log LOG.md. Four full tiles / 64 windows scored in 164.303 s. Independently recomputed PV integer counts and IoU match exactly in val and eval (absolute error 0). Full three-arm scoring was not run; the existing mit_b2 smoke checkpoint is the only end-to-end checkpoint test. Three production architectures passed CPU constructor/forward checks.
