# Vexcel Generalization Evaluation Dataset

Date: 2026-04-26

## Scope

Build a first-pass task-grid system for six Vexcel South Africa urban aerial
coverage areas outside the existing Cape Town and Johannesburg evaluation
footprint:

- Pietermaritzburg
- Durban
- East London
- Port Elizabeth / Gqeberha
- Bloemfontein
- Pretoria

This dataset is intended to test geographic generalization of the detector.
The reporting frame remains V1.3 reviewed prediction footprints, aggregated at
task-grid level per the V1.4 validation strategy.

## Inputs

Coverage metadata is stored in:

- `configs/datasets/vexcel_urban_coverage.yaml`

Coverage footprints are fetched from Vexcel API 2.0:

- `data/vexcel_coverage/<region>_coverage.geojson`
- `data/vexcel_coverage/vexcel_coverage_footprints.geojson`

The original API-reported bboxes are retained in the config as fallback metadata,
but task grids are clipped to the fetched Vexcel collection footprints when
those GeoJSON files exist.

## Generated Artifacts

Task grids:

- `data/vexcel_task_grids/<region>_task_grid.gpkg`
- `data/vexcel_task_grids/<region>_task_grid.geojson`
- `data/vexcel_task_grids/vexcel_task_grids.geojson`
- `data/vexcel_task_grids/vexcel_task_grid_summary.csv`

Default balanced sample:

- `data/vexcel_eval_samples/vexcel_eval_grids_seed42_per_region10.csv`
- `data/vexcel_eval_samples/vexcel_eval_grids_seed42_per_region10.gpkg`
- `data/vexcel_eval_samples/vexcel_eval_grids_seed42_per_region10.geojson`

The default sample selects 10 grids per city with `coverage_fraction >= 0.75`
and seed `42`.

## Commands

Refresh coverage footprints from Vexcel:

```bash
python scripts/validation/fetch_vexcel_coverage.py --overwrite
```

Regenerate task grids:

```bash
python scripts/validation/build_vexcel_task_grids.py --overwrite
```

Regenerate the default balanced sample:

```bash
python scripts/validation/sample_vexcel_eval_grids.py --per-region 10 --seed 42 --overwrite
```

Build a proportional 120-grid sample:

```bash
python scripts/validation/sample_vexcel_eval_grids.py --total 120 --seed 42 --overwrite
```

## Next Steps

1. Drag `data/vexcel_coverage/vexcel_coverage_footprints.geojson` and
   `data/vexcel_task_grids/vexcel_task_grids.geojson` into the Vexcel web UI to
   visually confirm footprint/grid alignment.
2. Add grid-type strata (`CBD`, `suburban`, `township`, `peri_urban`, `rural`)
   before scaling beyond the first balanced city sample.
3. Download Vexcel orthos into `/mnt/d/ZAsolar/tiles/vexcel/<region>/ortho`.
4. Run detector inference into isolated result roots under
   `results/vexcel/<region>/`.
5. Send sampled grids to the reviewed-prediction annotation workflow and report
   aggregate grid-level inventory diagnostics by city and stratum.
