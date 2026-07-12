# Archived data locations (off-repo)

Some large, historical, or regeneratable data has been moved **out of this repo
tree** to reclaim local disk. This file is the pointer so future agents/humans
can find and restore it. Two off-repo tiers exist:

1. **`~/zasolar_data/`** (same machine, same physical disk) — the working big-data
   root. Live tiles/coco/models/etc. Referenced by code via `SOLAR_TILES_ROOT`,
   `region_registry`, and in-repo symlinks. NOT a backup (same disk).
2. **Dropbox `dropbox:ZAsolar_archive/`** (rclone remote `dropbox:`) — durable
   **off-site backup** of concluded/historical data that was then **deleted
   locally** to free disk. This is the real disaster-recovery copy.

## Dropbox archive — `dropbox:ZAsolar_archive/`

Uploaded 2026-07-11. Each unit is a single `.tar`. A full manifest lives at the
archive root: `dropbox:ZAsolar_archive/ARCHIVE_INDEX.md`.

| Tar | Was at (repo-relative) | Size | Why archived |
|---|---|---|---|
| `vexcel_eval_60_osm_stratified_seed42_2x2.tar` | `tiles/vexcel_eval_60_osm_stratified_seed42_2x2/` | ~4.1G | Orphaned xdomain Vexcel eval tiles; **0 live code/config refs**; violated the "in-repo `tiles/` = placeholder only" rule. |
| `gemini_review_calib_no_prelaunch.tar` | `data/analysis/gemini_review_calib/` (all except `prelaunch/`) | ~2.5G | Regeneratable Gemini FP-review chips/eval artifacts. **`prelaunch/` KEPT in repo** — it is a live test dependency (`tests/analysis/test_two_stage_failclosed_data.py`). |
| `results.tar` | `results/` (entire tree) | ~12G | Concluded census/eval outputs (CT + JHB + analysis + xdomain vexcel). Newest file was 2026-06-24; no active writes. Human-review `*_reviewed.gpkg` were already exported to `data/annotations/` (the canonical GT, which stays). |
| `models/other_detector_checkpoints.tar` | `~/zasolar_data/models/{v1_ft_cs400_tileval_20260317_r4, v2_sam2_260320, exp003_C_targeted_hn, exp004_v4_hn, exp005_v4_1_hn, exp006_v4_2_jhb_ft, checkpoints_cleaned}/` (symlinked into `checkpoints/`) | ~2.5G | Non-production detector checkpoints (benchmark/reproducibility history). `configs/model_registry.yaml` still lists their `checkpoints/…/best_model.pth` paths — now **dangling by design**; `run_benchmark.py` on these model ids needs a restore. `validate_registry.py` does NOT assert `.pth` existence, so registry validation still passes. **Production `exp_unified_reviewall_A` and all backbones (sam2, sam_vit_l, dinov2_floor, dinov3_sat×2) were kept local** (production-critical / re-downloadable). |

### Consequence of archiving `results/` (intentional, user-approved)

`core/grid_utils.py` uses `RESULTS_ROOT = BASE_DIR / "results"` and reads it
**directly** (there is no `SOLAR_RESULTS_ROOT` redirect — only tiles have
`SOLAR_TILES_ROOT`). A dozen training/eval configs reference `results/...` paths
as inputs (e.g. `configs/datasets/training_sets.yaml`,
`configs/pipelines/datasets/*.yaml`). Those paths are now **dangling by design** —
the trained models already exist; only *re-running those exact recipes* needs the
data restored. Restore `results.tar` first if you need to reproduce them.

## How to restore

```bash
# list what's there
rclone lsl dropbox:ZAsolar_archive/
rclone cat dropbox:ZAsolar_archive/ARCHIVE_INDEX.md   # full manifest

# pull one unit back and unpack in place
rclone copy dropbox:ZAsolar_archive/results.tar /home/gao/zasolar_data/_restore/ --progress
tar xf /home/gao/zasolar_data/_restore/results.tar -C /home/gao/projects/ZAsolar   # recreates results/

# non-production detector checkpoints → unpack into the models root the symlinks point at
rclone copy dropbox:ZAsolar_archive/models/other_detector_checkpoints.tar /home/gao/zasolar_data/_restore/ --progress
tar xf /home/gao/zasolar_data/_restore/other_detector_checkpoints.tar -C /home/gao/zasolar_data/models
```

Tarballs were created with the source dir as a top-level member (`tar cf X.tar -C <parent> <dir>`),
so `tar xf` into the right parent recreates the original path exactly.

## Off-site backup of the LIVE production model (kept local, also copied out)

The production detector `exp_unified_reviewall_A/best_model.pth` (the current
census model — cited across `configs/datasets/regions.yaml` CT+JHB model_runs)
now has a durable off-site copy. It is **still live** in `~/zasolar_data/models/`
(symlinked into `checkpoints/`) — this is a backup, not an archive-and-delete.

- Off-site: `dropbox:ZAsolar_archive/models/exp_unified_reviewall_A/best_model.pth`
  (176,243,385 bytes, uploaded + byte-verified 2026-07-11).
- Restore: `rclone copy dropbox:ZAsolar_archive/models/exp_unified_reviewall_A/best_model.pth /home/gao/zasolar_data/models/exp_unified_reviewall_A/`

## What was NOT backed up (still local only / in-repo)

- `data/annotations/` — canonical GT (already mirrored to `dropbox:RA_Solar/Gao/`).
- `data/analysis/gemini_review_calib/prelaunch/` — live test dependency.
- `~/zasolar_data/**` — live working big data (tiles/coco/geid_temporal + the
  **other, non-production** checkpoints): same-disk only, no off-site copy. Only
  the production checkpoint above was pushed off-site (per user scope).
