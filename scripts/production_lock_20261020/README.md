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
| citywide_infer | new | `ct_city_infer_v1.py`: first PV+SWH save step (ZSTD15, superseded, kept). `ct_city_infer_v2.py`: **production pod script** — `infer` (convnext_t_s0 + mit_b2_s2, PV+SWH uint8 COG ZSTD3 PREDICTOR2, writer processes, resumable) and `merge` (avg2 + h0.9_0.5 + A3 per 1 km block with 1-tile halo). `train2_v2e.py` = pinned model source (md5 961e2e8e…). |

Tests: `python3 -m pytest tests/production_lock` and `cd scripts/production_lock_20261020/ct_xtile_merge && python3 -m pytest tests`.

## Pod run (ct_city_infer_v2.py)
Weights and tiles are NOT in git; copy them to the pod separately (tile list: ~/capetown_eval/ct_avg2_upload_v1_20261020/files_v1.txt).
```
python3 scripts/production_lock_20261020/citywide_infer/ct_city_infer_v2.py infer --tiles-csv seg_tiles_kept_plus_qa_v1.csv \
    --tile-dir /workspace/tiles --ckpt-root /workspace/ckpt --out /workspace/out --writers 4
python3 scripts/production_lock_20261020/citywide_infer/ct_city_infer_v2.py merge --tiles-csv seg_tiles_kept_plus_qa_v1.csv --out /workspace/out --workers 4
```
`--ckpt-root` must hold `convnext_t_s0/` and `mit_b2_s2/`, each with `best_objf1.pt` (md5-checked) and `config.json`.
Needs torch, transformers, rasterio, scipy, geopandas, shapely, pandas.
