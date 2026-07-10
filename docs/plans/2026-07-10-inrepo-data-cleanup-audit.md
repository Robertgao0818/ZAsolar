## Data Audit Report — ZAsolar in-repo governance violations

**Scope**: `results/`, `tiles/`, `checkpoints/`, `cache/`, `data/coco*`, plus other multi-GB dirs found via `du` (`checkpoints_cleaned/`, `probes/`, `mosaics/`, `data/analysis/`, `data/buildings/`). `data/annotations/` excluded per instructions. Compared against `~/zasolar_data/{results,tiles,models,coco}` by name, subtree listing, and spot file-size checks.

**Correctly-externalized (not audited further)**: `data/geid_temporal` and `data/cls_pv_thermal_v2` are symlinks to `~/zasolar_data/...` — this is why `du -sh data` initially undercounted (7.3G) vs following the symlinks (77G+). These are compliant, skip them.

### Findings table

| Path | Size | Last modified | Duplication status | Recommended action |
|---|---|---|---|---|
| `results/analysis/` | 7.2G | 2026-06-24 | **unique** — `~/zasolar_data/results/analysis/` has only `fn_recut_sample20/` (17M) + `true_fn_previews/` (7.8M), no match to any of the 20+ subdirs here (biggest: `gemini_fpcut_li/` 4.0G, `gt_heater_audit/` 555M, `direct_maskrcnn_v1/` 432M incl. `raw_detections.pkl`, `ct_census_output_table/` 390M, `full382_merge01_2026-05-15/` 388M) | migrate to `zasolar_data/results_archives/` (tar) then delete |
| `results/johannesburg/` | 3.2G | 2026-05-15 | **duplicated (partial)** — only `unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox/` (914M) has a matching archive: `zasolar_data/results_archives/unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox.tar.gz` (209M compressed, base `predictions_metric.gpkg` only). The in-repo copy also holds newer `predictions_metric_nms01_c0925.gpkg` per grid (generated 23:48, archive made 23:03) that is **not** backed up. The other 29 model-run subdirs are unique | archive the delta (`*_nms01_c0925.gpkg`) then delete the archived subdir; migrate the remaining 29 subdirs then delete |
| `results/cape_town/` | 1.3G | 2026-06-11 | **unique** — `zasolar_data` has `tiles/cape_town/aerial_2025` but no `results/cape_town` | migrate then delete |
| `results/G1189/` | 479M | 2026-06-10 | **unique** — legacy per-grid layout, predates `results/<region>/<model_run>/` convention | needs human decision (superseded old-style benchmark?) |
| `results/G1238/` | 260M | 2026-03-25 | **unique**, same legacy layout | needs human decision |
| `results/vexcel/` | 158M | 2026-05-31 | **unique** (xdomain60 6-city eval outputs) | migrate then delete |
| `results/grid_previews/` (+`_startcheck`) | 129M+80K | 2026-03-29 | **unique**, likely regenerable preview thumbnails | delete (low risk, verify first) |
| `results/diag/` | 17M | 2026-05-08 | **unique** | needs human decision (small, low priority) |
| `results/{G1688,G1743,G1687,G1800,G1749,...}` (13 legacy per-grid dirs) | ~10M total | various (2026-03/04) | **unique** | needs human decision, low priority given size |
| `tiles/vexcel_eval_60_osm_stratified_seed42_2x2/` | 4.1G | 2026-04-26 | **unique** — no matching dir under `zasolar_data/tiles`; grid IDs here use xdomain Vexcel naming (`DBN0666`, `BFN0106`, `GQB0153`...) whereas `zasolar_data/tiles/durban/ethekwini_2023` uses a different naming/layer (`ETK0086` etc.) — not the same imagery | migrate to `zasolar_data/tiles` then delete |
| `checkpoints/` | 2.2G | 2026-05-13 (newest: `exp_unified_reviewall_A`) | **unique, NO backup found anywhere** — `zasolar_data/models/` holds only DINOv2/DINOv3/SAM2 backbones (different model family, unrelated to these Mask R-CNN checkpoints). Every path here (`v1_ft_cs400_tileval_20260317_r4`, `v2_sam2_260320`, `exp003_C_targeted_hn`, `exp004_v4_hn`, `exp005_v4_1_hn`, `exp006_v4_2_jhb_ft`, `exp_unified_reviewall_A`) is actively referenced by `configs/model_registry.yaml`, and `exp_unified_reviewall_A/best_model.pth` is the **current production model** cited throughout `regions.yaml` model_runs | **needs human decision — back up to `zasolar_data/models` (and verify a RunPod `/workspace` copy, not checked here) before any deletion; do not treat as disposable** |
| `checkpoints_cleaned/` | 504M | 2026-03-25 | **unique**, referenced in `model_registry.yaml`; `regions.yaml:319` notes it covers only 24 grids ("Deferred") | needs human decision, same caution as above |
| `data/coco_v1_ft_cs400_tileval_20260317_r3/` | 2.1G | 2026-03-17 | **unique** — `zasolar_data/coco/` holds only `coco_clean_hs_ct260320_jhb_sam_added_20260510` (1.2G, different name/content). Not referenced in `configs/datasets/training_sets.yaml` — possibly orphaned/superseded export | needs human decision (check if superseded by the zasolar_data coco set before deleting) |
| `data/coco_v1_ft_cs400_tileval_20260317_r2/` | 1.4G | 2026-03-17 | **unique**, same situation as r3 (likely an earlier iteration of it) | needs human decision |
| `data/coco_train20_val5_qa/` | 256M | 2026-05-08 | **unique** — QA artifacts (large-polygon review CSVs/gpkg/webapp), not a COCO export itself | needs human decision |
| `data/analysis/gemini_review_calib/` | 2.6G | 2026-06-01 | **unique** — Gemini review calibration chip corpus | needs human decision (likely disposable one-off, verify) |
| `data/buildings/overture/` | 661M (of 832M dir) | 2026-06-10 | **unique** reference dataset (Overture footprints, source download not a pipeline product) — technically outside the named governance rule (results/tiles/checkpoints/cache/coco) but still multi-GB in-repo | needs human decision (fine to keep as reference data, or relocate to `zasolar_data/buildings` for consistency) |
| `probes/durban_aerial/` | 121M | 2026-05-08 | **unique** one-off vintage-probe TIFs | delete or migrate (low priority) |
| `mosaics/G0854_compare/` | 48M | 2026-04-26 | **unique** one-off imagery-source comparison | delete or migrate (low priority) |
| `cache/` | 13M | 2026-03-25 | **unique** but trivial (OSM buildings CSV/gpkg cache) | leave as-is, not worth the migration overhead |

### Summary

- Total real in-repo data across all audited entries ≈ **27.2 GB** (results 13G + tiles 4.1G + checkpoints 2.2G + checkpoints_cleaned 0.5G + coco_r2/r3 3.5G + coco_train20_val5_qa 0.26G + data/analysis 2.6G + data/buildings 0.83G + probes/mosaics/cache ~0.18G). This is larger than the ~17.1G quoted in the task because it also picks up `checkpoints/`, `data/coco_v1_ft_*`, `data/analysis/`, `data/buildings/`, `probes/`, `mosaics/` that weren't in the original `results`+`tiles` estimate.
- **Immediately safe to delete (true duplicate confirmed)**: effectively none outright — only the base `predictions_metric.gpkg` layer inside one 914M subdir has an off-repo archive, and even that archive is missing newer derived files, so it needs a "back up the delta" step first, not a blind delete.
- **Reclaimable after migrate-then-delete**: essentially the full ~27.2 GB, but the checkpoints (2.7 GB total) are the one category that must not be treated as routine cleanup — they are unbacked-up, currently-referenced production model weights (`exp_unified_reviewall_A` is cited as the current best model in `model_registry.yaml`/`regions.yaml`). Everything else (~24.5 GB) is unique analysis/results/tiles/coco data with no known second copy.
- **Lowest-risk quick wins** (small, likely disposable, low blast radius): `results/grid_previews/` (129M, regenerable), `probes/` (121M), `mosaics/` (48M) — combined ≈300M, still worth a human confirm before deletion since none were verified against docs as explicitly deprecated.

No files were modified, moved, or deleted during this audit.
---

## Pending decisions (owner: user)

- [ ] `checkpoints/` (2.2G) + `checkpoints_cleaned/` (0.5G) — **back up to `~/zasolar_data/models/` (and verify /workspace copy) BEFORE anything else**; contains the current production model `exp_unified_reviewall_A/best_model.pth` with NO second copy found
- [ ] `results/johannesburg/` — archive the un-backed-up `*_nms01_c0925.gpkg` delta, then migrate remaining 29 model-run subdirs
- [ ] `results/analysis/` (7.2G) + `results/cape_town/` (1.3G) + `results/vexcel/` (158M) — migrate to `~/zasolar_data/results*` then delete
- [ ] `tiles/vexcel_eval_60_osm_stratified_seed42_2x2/` (4.1G) — migrate to `~/zasolar_data/tiles/` then delete
- [ ] Legacy per-grid `results/G*` dirs (~750M) + `results/diag` — human review (pre-restructure layout)
- [ ] `data/coco_v1_ft_*` (3.5G) — verify superseded by `~/zasolar_data/coco/` set, then delete
- [ ] Quick wins pending confirm: `results/grid_previews/` (129M, regenerable), `probes/` (121M), `mosaics/` (48M), `data/analysis/gemini_review_calib/` (2.6G)
- [ ] `data/buildings/overture/` (661M) — keep as reference or relocate to `~/zasolar_data/buildings/`

Companion action landed the same day: 69 concluded one-off scripts/docs archived to
`scripts/archive/` + `docs/experiment-archive/` (audit manifest: workflow
`zasolar-cleanup-audit`, 2026-07-10).

---

## Resolution

**2026-07-10 (migrate off repo tree → `~/zasolar_data/`, same physical disk):**
`checkpoints/` + `checkpoints_cleaned/` (2.7G, symlinked back; still **no off-site
copy**), `data/coco_v1_ft_*` r2/r3, `data/coco_train20_val5_qa`, `probes/`,
`mosaics/`, `data/buildings/` (symlinked), and legacy per-grid `results/` dirs.
Same-disk move → reclaims repo tree, **not** physical disk.

**2026-07-11 (off-site Dropbox archive → then delete local, real disk reclaim):**
tar → `dropbox:ZAsolar_archive/` → verify remote byte-size == local → delete local.
Manifest: `dropbox:ZAsolar_archive/ARCHIVE_INDEX.md`; repo pointer:
`ARCHIVE_LOCATIONS.md`.

- [x] `tiles/vexcel_eval_60_osm_stratified_seed42_2x2/` (4.1G) → `…seed42_2x2.tar` (4,345,364,480 B). Deleted.
- [x] `results/` **entire tree** (12G — analysis, cape_town, johannesburg, vexcel, grid_previews, validation, legacy `G*`/diag, all `*/review/*_reviewed.gpkg`) → `results.tar` (12,545,976,320 B). Deleted; replaced by tracked placeholder `results/README_ARCHIVED.md`. Config `results/...` refs now dangling **by design** (user-approved; models already trained).
- [x] `data/analysis/gemini_review_calib/` minus `prelaunch/` (2.6G) → `gemini_review_calib_no_prelaunch.tar` (2,741,985,280 B). Deleted. `prelaunch/` **kept** (live test dep `tests/analysis/test_two_stage_failclosed_data.py`).

**Still open (not disaster-recovered):** production checkpoint
`exp_unified_reviewall_A` and its siblings live only on the same physical disk
(`~/zasolar_data/models/`, symlinked into repo) — a real off-site/RunPod copy is
still owed. `~/zasolar_data/**` working big data (tiles/coco/models/geid_temporal)
is likewise same-disk only.
