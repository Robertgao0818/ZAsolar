# JHB CBD25 recall vs 葉-reviewed reference (2026-10-18, CPU)

## Reference: 2,174 objects
- Built from `G*_reviewed.gpkg` (25 grids, 2,366 rows), keeping review_status ∈ {correct, edit}:
  - **MRCNN kept** = 437 correct + 38 edit = **475**;
  - **human-added FN** (source = sam_fn_review, 葉's SAM-drawn misses/redraws) = **1,699**.
- Excluded: 190 delete, plus 1 NaN-status and 1 unreviewed object.
- Size: 292 objects in 4–10 m².
- `G0772_fn_markers.gpkg` holds 5 FN marker points with no polygon. They are not in the reference; no model polygon contains any of them.

## Matching
- Hit = IoU ≥ 0.1 OR intersection ≥ 50% of the smaller polygon (EPSG:32735), counted per reference object.
- R50 = best IoU ≥ 0.5.

Results: recall_by_model.csv; per-object table: reference_*; per-grid table: recall_by_grid.csv. No ConvNeXt-T s1 Joburg polygons exist.
