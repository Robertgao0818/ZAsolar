Warning: no stdin data received in 3s, proceeding without it. If piping from a slow command, redirect stdin explicitly: < /dev/null to skip, or wait longer.
葉 is right for the MiT gate, and only partly right for the Mask R-CNN `detect_direct` gate. The r2 headline doesn't need to change. I recommend a text-only r3 addendum, frozen before scoring. Everything was read-only and CPU-only, and no test predictions were opened.

**1) Do chips contain neighbouring-grid pixels? Yes. Are they zero-padded at grid edges? No.**
- The 0.0016° chunks cover the cell's EPSG:4326 bounding box. Because the 1 km UTM cell sits rotated about 1.35° inside that box, each grid's imagery extends past its cell as a wedge-shaped strip 0 to 23.6 m wide. That strip is 4.7% of each cell (median over 2,083 cells).
- With the 2026-10-14 `full_edge_tiles` edgefix tiles, the strip grows to 10.3% of the cell and up to 65 m wide.
- All 2,352 local chunks are 2000×2000 px, so `edge_pad` zeros never fire on Cape Town chunks.
- Chips never cross a chunk seam inside a grid. 64 of 1,246 test polygons (11.3% of PV area) cross a seam.

**2) Does the pregate use neighbouring content? No. Can a straddling panel be dropped in both grids? Yes.**
- In `detect_direct`, each chip is scored alone. A rejected chip just has no detections and does not erase overlapping kept chips. A panel is lost only if every chip that sees it is rejected.
- Both grids see an edge panel, but they decide independently, so a straddling panel can be dropped by both. The risk is highest at seams and at the narrow end of the wedge, where each side sees only a fragment.
- The Cape Town Mask R-CNN census ran without the pregate; Durban ran with it.
- The JHB MiT gate is where 葉's "覆盖" (overwrite) applies. It scores 33 m gate cells with no context and multiplies rejected cells' probabilities by 0. That erases what a kept neighbour's window predicted inside the rejected cell.
- On the test geometry, 33.8% of polygons (64.6% of PV area) cross a gate or tile line, and 8,256 m² (20.4% of PV area) lies outside each polygon's largest piece. That is the area exposed to this cut.
- In JHB, 1.1% of GT polygons became unreachable because of the gate. Area loss was never measured.

**3) Production merge vs. eval clipping**
- `ct_census_merge.py` just concatenates the per-grid outputs, with no clipping and no ownership rule. The raw inventory has 170,605 polygons, including 14,174 overlapping pairs from different grids (363k m² of overlap).
- The delivered IoU≥0.1 merge (111,801 polygons) leaves only 386 such pairs (3,680 m²).
- However, 3,077 merged polygons are labelled with a `gridcell_id` whose cell doesn't contain them, and 4.2% of the merged area sits outside its labelled cell. So per-cell totals are not clean.
- Durban's `stitch_census_v1.py` stitches across grids city-wide but doesn't assign each object to one cell either.
- The eval clips GT and predictions to the same cell, and zaeval joins objects across tiles within a cell. The test inference, though, ran MiT on whole 100 m tiles with no gate, and 61 of 442 tiles span 2–4 cells. So the 1 km edges are only scoring clips there. The test measures none of the effects in points 1–3.

**4) Test-set numbers (1,246 polygons, 40,470 m²)**

| Location | Polygons | PV area |
|---|---|---|
| Within 10 m of a cell edge | 105 (8.4%) | 5,727 m² (14.2%) |
| Within 20 m of a cell edge | 178 (14.3%) | 9,006 m² (22.3%) |
| Crossing the outer edge of the test area | 12 | 1,923 m² |
| Crossing an edge between two test cells | 14 (each split into 2 pieces) | 467 m² |
| Covered by two grids' production imagery (double-counted by a plain concat) | 119 (9.6%) | 3,007 m² (7.4%) |

- The 821.1 m² clipping loss is confirmed. One wording fix for r2 §2: the 26 polygons that lose area *include* the 14 that fall wholly outside. Only 12 are partly outside.

**5) Rules decision for 葉**
- The r2 headline needs no change: both models are scored with identical inference geometry, and scoring hasn't run.
- The proposal is a text-only r3 frozen before scoring:
  - fix the 26 = 12 + 14 wording;
  - add a scope note that test scores are an upper bound for a gated or grid-tiled pipeline;
  - optionally, a pre-registered stratification by distance to the cell edge and by tile-line crossing, reported but not used to pick models.
- Separately, two pipeline fixes, both validated on CT-val and never on the test set:
  - fix the gate (give it context, or keep whole objects that touch a kept cell), then measure what it loses;
  - stitch city-wide and assign each object to exactly one cell.

I didn't update any TRACKER or RUN docs, because you asked for new files only; the proposal and RUN.md still need those entries.

Files are in `~/projects/zasolar_plans_20261017/`:
- `cross_grid_edge_proposal_v1.md`
- `cross_grid_edge_v1/quant_edges.py`
- `cross_grid_edge_v1/results_edges.json`
- `cross_grid_edge_v1/per_cell_halo.csv`
- `cross_grid_edge_v1/census_dupes.py`
- `cross_grid_edge_v1/census_dupes.json`
