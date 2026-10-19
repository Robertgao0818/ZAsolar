# Production-lock code snapshot (G24, 2026-10-20 NZDT)

Code copied (not moved) from rog working folders so it lives in the repo. Originals are unchanged.
Only *.py / *.sh / *.md were copied; outputs, probabilities, pkl/npz/gpkg/tif, weights and raw reviewer logs are not in git.
Scripts keep their original absolute `~/capetown_eval/...` / `~/data/...` paths: they are a provenance record of what ran,
not a relocatable package.

| dir | source on rog | what |
|---|---|---|
| pregate/citywide_pregate_val_v1 | ~/capetown_eval/citywide_pregate_val_v1_20261019 | CT val pregate scoring + analysis |
| pregate/citywide_pregate_run_v1 | ~/capetown_eval/citywide_pregate_run_v1_20261019 | CT citywide pregate score-only loop (ct_pregate_city) |
| pregate/jhb_pregate_xfer_v1 | ~/capetown_eval/jhb_pregate_xfer_v1_20261019 | Joburg CBD25 pregate transfer test + τ sweep |
| eval_phase1/phase1_eval_pipeline | ~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018 | phase-1 eval pipeline (pipeline_common_v3, eval_phase1_v3) |
| eval_phase1/joburg_cbd25_phase1 | ~/capetown_eval/joburg_cbd25_phase1_20261018 | Joburg CBD25 phase-1 inference/analysis |
| eval_phase1/joburg_cbd25_rescore_v2 | ~/capetown_eval/joburg_cbd25_rescore_v2_20261019 | G6 official-scorer rescore runners |
| eval_phase1/jhb_cbd25_zaeval2_v1 | ~/capetown_eval/jhb_cbd25_zaeval2_v1_20261019 | Joburg CBD25 zaeval-2@8044d14 re-eval (zaeval itself: repo zasolar_evalfix) |
| ct_xtile_merge | ~/capetown_eval/ct_xtile_merge_v1_20261019 (local repo, b23f328) | G7 cross-tile A3 merge + tests |
| g14_bldg_assoc | ~/capetown_eval/g14_bldg_assoc_v1_20261019 | G14 building association |
| jhb_union | ~/capetown_eval/jhb_union_v1_20261019 | Joburg union scoring |
| avg2_bench | ~/capetown_eval/avg2_bench_v1_20261020 | rog 4070 avg2 timing benchmark |
| citywide_infer | new | CT citywide inference save step: per model PV **and SWH** uint8 COGs (v5) |

Tests: `python3 -m pytest tests/production_lock` and `cd scripts/production_lock_20261020/ct_xtile_merge && python3 -m pytest tests`.
