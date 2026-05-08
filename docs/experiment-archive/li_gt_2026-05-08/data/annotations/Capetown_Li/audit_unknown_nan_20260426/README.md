# Li raw NaN/unknown polygon audit (2026-04-26)

Extracted raw rows from `Dropbox/RA_Solar/Li/capetown/` where `class` is NaN. These rows have geometry but no SAM provenance fields (`segment_id`, `method`, `timestamp`, `mask_file`, etc.). They are currently excluded from normalized `data/annotations/Capetown_Li/G*.gpkg` by the `class.notna()` filter.

Files:
- `li_raw_nan_unknown_polygons.gpkg`, layer `unknown_nan_polygons`: all extracted unknown polygons.
- `summary.csv`: per-grid raw/labeled/unknown counts and area stats.

Interpretation update after visual review (2026-04-26): user judged the G1895 raw NaN/unknown polygons likely to be Li's earlier hand-drawn annotations; the later labeled rows are SAM2 plugin-cut annotations. Keep them separate from normalized GT for now because provenance/semantic tier still needs explicit handling, and G1895 also has some missed annotations to revisit later.
