# Multi-City Architecture Rules

## Never hardcode city-specific paths

Do not hardcode region-specific directory paths in new or modified code.
The canonical layout (post-2026-04-19 restructure, post-2026-04-26
migration to WSL ext4) is:

- Tiles: `~/zasolar_data/tiles/<region>/<imagery_layer>/`
- Results: `results/<region>/<model_run>/`

Always use `core/grid_utils.py` + `core/region_registry.py`:

- `resolve_tiles_dir(grid_id, region=, imagery_layer=)` — tile dir or mosaic file
- `get_results_root(region=, model_run=)` — results root for a specific run
- `get_grid_paths(grid_id, region=, imagery_layer=, model_run=)` — full path set
- `region_registry.get_imagery_layer_path(region, layer_id)` — raw layer dir
- `region_registry.get_model_run_path(region, run_id)` — raw run dir

Legacy transitional symlinks (`tiles_joburg`, `results_joburg`,
`tiles/joburg_geid`) were deleted on 2026-04-26. Do not re-introduce them.

## Grid IDs can overlap between regions — NEVER pick region by grid ID

CT and JHB task grids **both** contain IDs like `G1189`, `G1190`, `G1293`,
`G1513`, `G1570`, `G1630`. These cover **different physical areas** in each
region, not the same area.

Rules:
- Any API that resolves a path from `grid_id` MUST take a `region` argument.
- `core.region_registry.lookup_region(grid_id)` returns one match arbitrarily
  and is DEPRECATED for multi-region contexts. Use `lookup_regions(grid_id)`
  (plural) if you need all hits.
- Classification scripts (e.g., migrating results by grid ID) must read
  `config.json.tiles_dir` or `config.json.model_path` — never pattern-match
  on grid ID ranges.

## Region must flow from config, not pattern matching

Do not infer region from grid ID naming patterns (e.g., "starts with JHB =
Johannesburg"). Region should be:
1. Explicitly passed via `--region` CLI arg, or
2. Looked up from `configs/datasets/regions.yaml` via
   `core.region_registry.lookup_regions(grid_id)` (returns list; callers
   must disambiguate).

The `normalize_region()` function in `grid_utils` handles aliases
(jhb/joburg/johannesburg → jhb, ct/cape_town/capetown → ct).

## Imagery layers and model runs are authoritative

Each region in `regions.yaml` declares:
- `imagery_layers:` — physical tile sources (aerial_2023 / geid_2024_02 /
  aerial_2025 / ...). Each layer has `source`, `vintage`, `file_layout`
  (`chunked` or `mosaic`), `crs`, and `coverage_grids`.
- `model_runs:` — each inference batch has `model_version`, `imagery_layer`,
  `results_path`, `inference_date`, `grid_count`.

When running inference via `detect_and_evaluate.py`, prefer explicit
`--imagery-layer` and `--model-run` to get the right source tiles and the
right output directory. `config.json` in results must contain
`imagery_layer_id` and `model_run_id` to let downstream tools trace
provenance.

`chunked` vs `mosaic` layouts differ: chunked is a directory of
`{grid}_{col}_{row}_geo.tif` chunks; mosaic is a single `{grid}_mosaic.tif`.
Consumers must branch on `file_layout` (from the layer's `MANIFEST.json` or
`region_registry.get_imagery_layer(...).file_layout`).

## `regions.yaml` is the authoritative registry

`configs/datasets/regions.yaml` is the single source of truth for:
- Which grids exist and which region they belong to
- CRS per region
- Annotation source paths
- Region-specific infrastructure paths (tiles, results, task grid)
- **imagery_layers per region** (physical tile sources with vintage/format)
- **model_runs per region** (inference batches with model + layer + results_path)

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
