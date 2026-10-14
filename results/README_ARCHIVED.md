# `results/` was archived off-repo (2026-07-11)

The entire `results/` tree (~12G of concluded census/eval outputs: `cape_town/`,
`johannesburg/`, `analysis/`, `vexcel/`, `grid_previews/`, `validation/`, and
`*/review/*_reviewed.gpkg`) was tarred to Dropbox and deleted locally to reclaim
disk. It is **not lost** — restore it if you need to re-run a recipe that reads
`results/...`.

- Off-site backup: `dropbox:ZAsolar_archive/results.tar` (12,545,976,320 bytes)
- Full pointer + restore commands: `../ARCHIVE_LOCATIONS.md`
- Manifest at archive root: `dropbox:ZAsolar_archive/ARCHIVE_INDEX.md`

Restore in place:

```bash
rclone copy dropbox:ZAsolar_archive/results.tar /home/gaosh/zasolar_data/_restore/ --progress
tar xf /home/gaosh/zasolar_data/_restore/results.tar -C /home/gaosh/projects/ZAsolar
```

`core/grid_utils.py` still points here (`RESULTS_ROOT = BASE_DIR / "results"`); new
runs will repopulate this dir. Canonical GT is unaffected — it lives in
`data/annotations/`, not here.
