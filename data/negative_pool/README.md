# `data/negative_pool/` — project-level hard-negative chip catalog

Cross-version, archetype-tagged registry of detector lookalike FP chips.
This pool **accumulates monotonically** across training runs — entries
once added are never silently dropped, only re-tagged or marked
deprecated. See [`feedback_negative_pool_persistent`](../../../home/gaosh/.claude/projects/-home-gaosh-projects-ZAsolar/memory/feedback_negative_pool_persistent.md)
for the persistence rule and [`feedback_hn_breadth_dominates_size`](../../../home/gaosh/.claude/projects/-home-gaosh-projects-ZAsolar/memory/feedback_hn_breadth_dominates_size.md)
for the breadth-over-count principle.

## What this is (and isn't)

This is a **provenance manifest**, not a chip warehouse. The actual 400×400
training chips are extracted on demand from raw imagery
(`~/zasolar_data/tiles/<region>/<imagery_layer>/`) at COCO-export time.
Storing the chips inline would duplicate the tile pool and inflate
git/disk for no benefit.

| Layout                          | What lives here                                  |
| ------------------------------- | ------------------------------------------------ |
| `manifest.csv`                  | One row per cataloged FP — region/grid/imagery/archetype/source |
| `archetype_taxonomy.yaml`       | Controlled vocabulary, distinguishing features, easily-confused-with table, canonical example chip_ids |
| `previews/<archetype>/`         | 224×224 PNG symlinks for human review (gitignored). Each `canonical_examples` entry in the taxonomy YAML resolves to a file `previews/<archetype>/<chip_id>.png` — open these three first when familiarising yourself with a class. |

For the literature anchor behind the design see
[`docs/literature/2026-05-10-hn-ratio-and-negative-pool-literature.md`](../../docs/literature/2026-05-10-hn-ratio-and-negative-pool-literature.md).

## Manifest schema

`manifest.csv` columns (stable order, append-only):

| Column                  | Meaning                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `chip_id`               | Stable unique ID, format `{region}_{grid}_{detector}_{source_idx}`      |
| `archetype`             | One of `archetype_taxonomy.yaml::archetypes` keys                       |
| `archetype_confidence`  | `A1` / `A2` / `A3` (Two-Axis Model conformance)                         |
| `region`                | `cape_town` / `johannesburg`                                            |
| `imagery_layer`         | e.g. `aerial_2025`, `vexcel_2024`, `aerial_2023`, `geid_2024_02`        |
| `grid_id`               | e.g. `G0772`, `G1855` (region-scoped — see rule 06-multi-city)          |
| `detector`              | Model run that produced the FP (`v3c`, `v4_2`, `train20_v3c_warm`, …)   |
| `source_run`            | Bucket / batch identifier (e.g. `cls_pv_thermal_v2`, `train20_audit`)   |
| `source_pred_id`        | Original `pred_id` in the producing detector's `predictions.gpkg`, or empty |
| `bbox_geo_wkt`          | Bounding box in EPSG:4326 (WKT POLYGON), or empty if not yet recovered  |
| `preview_path`          | Relative path under `previews/` if a 224×224 review crop exists         |
| `added_date`            | ISO date the row entered the manifest                                   |
| `notes`                 | Free text — re-review flag, FP→TP corrections, vintage caveats          |

## Adding to the pool

Two intended ingest paths:

1. **From a classifier subtype dataset** — e.g. `cls_pv_thermal_v2`. Run
   `scripts/training/negative_pool/bootstrap_from_cls_v2.py` (see that
   script for source-bucket handling and exclusion of `actually_pv_mislabeled`).
2. **From a fresh FP review pass** — after a detector run, the standard
   audit produces a CSV with `grid_id, pred_id, status, fp_bucket, …`
   (see `results/analysis/small_fp/.../small_fp_audit.csv` schema). Run
   `scripts/training/negative_pool/ingest_fp_audit.py <audit.csv>`
   (not yet built — first user with a fresh audit writes it).

Future training runs that want the HN stream call an exporter
(`scripts/training/negative_pool/export_hn_coco.py`, also not yet built)
which reads the manifest, filters by archetype/region/imagery_layer/confidence,
and extracts 400×400 chips into a COCO bundle.

## Working rules

- **Never delete a row.** If a chip turns out to be a real PV (i.e. it was
  `actually_pv_mislabeled`), re-tag with `archetype=actually_pv_mislabeled`
  and `notes` explaining — the audit trail matters for paper-time data
  provenance.
- **Region is not derivable from `grid_id`** (CT and JHB grid IDs overlap —
  see rule `06-multi-city.md`). Always set `region` explicitly.
- **Imagery vintage matters.** A 2023 aerial Khayelitsha chip is not
  interchangeable with a 2025 aerial Khayelitsha chip — record
  `imagery_layer` exactly, never collapse to "cape_town".
- **No `actually_pv_mislabeled` in active pool** for training. Those rows
  exist as a deprecation trail; the COCO exporter must filter them out.
- **Don't commit `previews/`** — it's a developer convenience only,
  gitignored. The manifest is the canonical artifact.

## Bootstrap status

Seeded 2026-05-13 from `cls_pv_thermal_v2/subtype_labels.csv`. See
`bootstrap_log.txt` (generated alongside this README on first ingest run)
for per-source-bucket row counts and excluded entries.

> **2026-05-29 — classifier extracted to `solar_cls`.** The `cls_pv_thermal_v2`
> dataset now physically lives in `~/zasolar_data/cls/` (managed by the
> `solar_cls` subrepo). `data/cls_pv_thermal_v2` in this repo is a **gitignored
> symlink** to it, so `bootstrap_from_cls_v2.py` (a detector-side consumer that
> only *reads* the subtype CSV) keeps resolving the path unchanged. The
> bootstrap script itself stays in this repo — it belongs to the detector
> negative-pool domain, not the classifier.
