# Capetown_Li — Li's SAM-assisted CT annotations

Normalized from `Dropbox/RA_Solar/Li/capetown/` on 2026-04-24.

## Annotation scheme

`annotation_scheme = li_kml` — uses `cape_town_grid_Li.kml` (2215 grids
G0029–G4429) in `Dropbox/RA_Solar/grid_data/`. This is **complementary**,
not overlapping, with Gao's scheme (`cape_town_grid_Gao.kml`, which is
the basis of `data/task_grid.gpkg`): CT is split along a north–south axis
between the two RAs, so identical grid IDs refer to different physical
cells under the two schemes.

## Provenance fields

Each polygon carries:
- `grid_id` — Li-scheme grid ID (e.g. `G1842`)
- `annotation_scheme` — always `li_kml`
- `label_source` — `human_manual_sam_assisted` (SAM as a drawing tool)
- `semantic_confidence` — `A2` by default (SAM-as-tool does not guarantee
  installation-spec conformance; see `.claude/rules/07-annotation-semantics.md`)
- `quality_tier` — `T2` (must not be auto-promoted to T1 without A1 review)
- `partial` — `true` when `labeled < declared_count` in the source filename
- `original_class` — raw `class` value from source (`太阳能` on early files;
  grid-number strings like `1897`/`1896-2` on later files). `class` is
  normalized to `solar_pv`.

## Status

- 17 grids, 1,260 labeled polygons total.
- Partial files: G1843, G1844, G1846, G1895 (only 39/260 confirmed), G1896 (201/202).
- G1950 source file used class string `1903` (annotator typo); geometries
  are at G1950's true location per Li-KML.

## Pending verification (2026-04-25, with imagery)

The unlabeled NaN rows in partial files (esp. G1895: 221/260) share one
signature: `class / method / segment_id / timestamp / source_layer /
mask_file` all NaN, but geometry is present with area distribution
similar to the labeled rows (mean ~15 m² vs ~13 m²). Working hypothesis:
**pre-SAM-workflow annotations** Li imported into QGIS as reference and
that were saved into the output gpkg. Currently filtered out by
`class.notna()` — preserved in raw source files, not in the normalized
outputs.

Verify tomorrow when base imagery is available:
1. Overlay unlabeled NaN geometries on G1895 tile and eyeball — solar PV
   or not?
2. If they are valid solar polygons, re-ingest them with a separate
   provenance tag (e.g. `label_source=legacy_weak_supervision`, `A3`).
3. If they are model predictions or stale imports, leave filtered.

## Known imagery gap

As of 2026-04-24, **no aerial_2025 tiles exist** for any of these 17 grids
on D drive, RunPod S3, or in `configs/datasets/regions.yaml`
`cape_town.imagery_layers.aerial_2025.coverage_grids`. Cannot be used as a
detector holdout until imagery is sourced for Li-KML cell bounds.

## Files

- `MANIFEST.csv` — per-grid provenance + counts
- `G<id>.gpkg` — one normalized file per grid, EPSG:4326, layer name `G<id>_li`
