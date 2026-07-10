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
rclone copy dropbox:ZAsolar_archive/results.tar /home/gaosh/zasolar_data/_restore/ --progress
tar xf /home/gaosh/zasolar_data/_restore/results.tar -C /home/gaosh/projects/ZAsolar   # recreates results/
```

Tarballs were created with the source dir as a top-level member (`tar cf X.tar -C <parent> <dir>`),
so `tar xf` into the right parent recreates the original path exactly.

## What was NOT archived (still local / in-repo)

- `data/annotations/` — canonical GT (also mirrored to `dropbox:RA_Solar/Gao/`).
- `data/analysis/gemini_review_calib/prelaunch/` — live test dependency.
- `~/zasolar_data/**` — live working big data (tiles/coco/models/geid_temporal…),
  managed separately; **not** yet off-site backed up (same-disk only).
- Production checkpoints (`checkpoints/` symlinks → `~/zasolar_data/models/`):
  still **no off-site copy** — a same-disk move is not disaster recovery.
