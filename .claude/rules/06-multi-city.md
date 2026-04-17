# Multi-City Architecture Rules

## Never hardcode city-specific paths

Do not hardcode `/mnt/d/ZAsolar/tiles_joburg`, `results_joburg/`, or any
city-specific directory path in new or modified code. Always use
`core/grid_utils.py` functions:

- `resolve_tiles_dir(grid_id, region=)` — tile directory
- `get_results_root(region=)` — results root
- `get_grid_paths(grid_id, region=)` — full path set
- `_resolve_gt_gpkg(grid_id, region=)` — GT annotation file

Existing hardcoded paths in legacy scripts are known tech debt. Do not
introduce new ones.

## Region must flow from config, not pattern matching

Do not infer region from grid ID naming patterns (e.g., "starts with JHB =
Johannesburg"). Region should be:
1. Explicitly passed via `--region` CLI arg, or
2. Looked up from `configs/datasets/regions.yaml` via
   `core.region_registry.lookup_region(grid_id)`

The `normalize_region()` function in `grid_utils` handles aliases
(jhb/joburg/johannesburg → jhb, ct/cape_town/capetown → ct).

## `regions.yaml` is the authoritative registry

`configs/datasets/regions.yaml` is the single source of truth for:
- Which grids exist and which region they belong to
- CRS per region
- Annotation source paths
- Region-specific infrastructure paths (tiles, results, task grid)

When adding a new city, the FIRST step is adding it to `regions.yaml`.
Code must read from this file via `core/region_registry.py`, not duplicate
its data as constants.

## CRS must be looked up, not assumed

Use `get_metric_crs(grid_id, region=)` from `core/grid_utils.py` for metric
CRS. Never hardcode `EPSG:32734` or `EPSG:32735` in scripts — these are
region-specific and will be wrong for future cities.

## Sync scripts must support all registered grid ID patterns

`scripts/sync_from_runpod.sh` and similar scripts must accept any grid ID
format registered in `regions.yaml`, not just `^G[0-9]+`. Use the
`grid_id_pattern` field from `regions.yaml` or accept an explicit grid list.
