# Cross-grid / cross-cell edge effects — proposal v1 (2026-10-17 NZDT, for 葉 to decide)

Trigger: 葉: "实际检测的时候，现在pregate会覆盖相邻格子的chip？跨格子的影响不小其实".
Work was read-only and CPU-only: no training, no GPU, no model inference, and no test predictions were opened. Scripts and outputs are in `cross_grid_edge_v1/` (`quant_edges.py` → `results_edges.json`, `per_cell_halo.csv`; `census_dupes.py` → `census_dupes.json`).

## 1. Three different kinds of "格子"

| boundary | size | where it is a processing cut | what the gate does there |
|---|---|---|---|
| 1 km CPT cell | 1000 m | Mask R-CNN census: one grid per `detect_direct` run (`resolve_tile_paths`) | per-chip decisions, with no neighbour context |
| 0.0016° chunk inside a grid | ~147 × 177 m (7×6 per cell) | Mask R-CNN: `SlidingWindowDataset` slides windows per chunk, and no chip spans a chunk seam | same as above |
| 100 m tile, and the 3×3 gate cells of 33.35 m | 100 / 33 m | MiT: each tile is inferred alone. The JHB gate (`joburg25_vexcel/pregate/pregate.py`, `evalobj_jhb25_par.py`) sets prob = 0 in rejected gate cells | **zeroes predictions that a kept neighbour's window made inside the rejected cell** |

## 2. Findings

**F1. Mask R-CNN chips do contain neighbouring-grid pixels. They are never zero-padded at a grid edge.**
- Chunks tile the cell's EPSG:4326 bbox (`get_grid_spec`/`get_tile_bounds`). Seen in UTM, that bbox is larger than the 1 km cell because the cell is rotated about 1.35° by meridian convergence. The extra strip ("halo") is wedge-shaped, 0 to 23.6 m wide (median max; worst 26.4 m), and covers 47,003 m² = 4.7 % of each cell (median over 2,083 cells). In total, 93.3 km² of halo lies inside other census cells, so that area is detected by two grids.
- With the 2026-10-14 `full_edge_tiles` edgefix tiles, the halo grows to 10.3 % of the cell and up to 65 m wide (south overhang 41.8 m, east 12.5 m). This applies to any re-run that uses those tiles.
- All 2,352 local chunks are 2000×2000 px. The anchored window mode puts the last window at `width-400`, so `edge_pad` zeros never fire on CT chunks. The only partial chips are the geoai-parity padded ones, and that mode is not the default.
- Chunk seams are hard cuts: 64 of 1,246 test polygons (5.1 %, 11.3 % of PV area) cross one.

**F2. detect_direct pregate (`presence_pregate.py`).**
- Each 400 px chip is scored on its own (DINOv2 patch max ≥ 0.45 over 4 heads), and the score uses no neighbour context. A rejected chip gets an empty detection list, but it does not delete detections from overlapping kept chips. So in this pipeline the gate does not "覆盖" (overwrite) neighbours.
- A panel is lost only if every chip that sees it is rejected. Inside one chunk, chips overlap by 100 px (≈ 7–9 m).
- At a 1 km edge, both grids see the panel through the halo (fully or partly), but they decide independently. A straddling panel can be dropped by both grids. The risk is highest at the zero-width end of the wedge and at chunk seams, where each side sees only a fragment.
- The CT Mask R-CNN census (`unifiedA_census_perdet`) ran **without** the pregate. Durban ran with it.

**F3. MiT gate (JHB-style, the candidate for CT M4).**
- Gate cells are 667 px with no context and no overlap. Rejected-cell probabilities are multiplied by 0, so a panel cut by a gate-cell line loses the pieces that fall in a rejected cell.
- On the CT test geometry, 421 of 1,246 polygons (33.8 %; 64.6 % of PV area) cross a gate-cell or tile line. 8,256 m² (20.4 % of PV area) lies outside each polygon's largest gate piece. That is the area exposed to this cut.
- JHB empirical result: 22 of 2,068 GT polygons (1.1 %) became unreachable under the ≥ 50 %-coverage rule. Area loss was not measured.
- **葉's concern is right for this pipeline.**

**F4. Merging across grid edges in production.**
- `ct_census_merge.py` only concatenates the per-grid outputs. There is no cell clip and no ownership rule. The raw inventory has 170,605 polygons, with 14,174 overlapping pairs from different grids and 363 k m² of overlap (5.0 % of 7.24 km²).
- The delivered `..._merged_iou010.gpkg` (IoU ≥ 0.1 union, 111,801 polygons) removes nearly all of that. 386 cross-grid overlapping pairs remain (3,680 m²), mostly small fragments inside larger polygons, where IoU ≤ 0.1. Touching or gapped fragment duplicates are not counted here.
- 3,077 merged polygons have their representative point outside their `gridcell_id` cell, and 209 k m² (4.2 %) lies outside the labelled cell. In other words, per-cell totals are not ownership-clean.
- Durban `stitch_census_v1.py` uses a city-wide gap graph and records multi-cell blocks. It handles cross-grid fragments, but it does not assign ownership either.

**F5. Test evaluation (r2) compared with production.**
- Test inference is MiT on whole 100 m tiles, with no gate (`infer_li.py`). 61 of 442 tiles span 2–4 cells (126 of 507 units).
- So the model sees across the 1 km cell edges, and those edges are only scoring clips. They are applied to GT and predictions alike: the 254 AOI, and zaeval stitches objects across tiles within a cell.
- r2 is therefore internally consistent. But it measures neither gate loss, nor chunk/grid tiling, nor cross-grid dedupe.

**F6. Test-set edge exposure (CT-test, 1,246 polygons with area in the 17 cells, 40,470 m²).**
- Within 10 m of any cell edge: 105 polygons (8.4 %), 5,727 m² (14.2 %). Within 20 m: 178 (14.3 %), 9,006 m² (22.3 %).
- PV area inside the cells' 10 m or 20 m edge band: 2,214 m² (5.6 %) and 4,636 m² (11.6 %). Those bands cover 4.0 % and 7.8 % of the cell area, so PV is denser near the edges than on average.
- Crossing a cell edge: 12 polygons cross the external edge (1,923 m² full area) and 14 cross an internal edge (467 m²; these are split into 28 pieces, which gives 1,260 pieces from 1,246 polygons).
- In production geometry, 119 polygons (9.6 %) and 3,007 m² (7.4 %) lie where two grids' chunk sets overlap. With a plain concat they would be counted twice.

**F7. The 821 m² is confirmed** (821.1 of 40,707.8 m², 2.0 %). One wording fix for r2 §2: the 26 polygons that lose area *include* the 14 that lie wholly outside (237.3 m²). Only 12 polygons are partly outside (583.8 m²). r2 currently reads as if there were 26 + 14.

## 3. Do the r2 rules need to change?

**Recommendation: the r2 headline needs no change.** Both MiT models are scored with identical inference geometry. The cross-grid effects above belong to the pipeline, not to model ranking. Changing the scoring area now would not make the test score reflect them, because the test inference has no grid tiling and no gate. Scoring has not run (`sealed/` is empty), so any text change can still be made cleanly before scoring. Per r2 §4, that needs 葉 and a new rules file.

Options for 葉:
- **D1 (recommended): r3 = r2 plus a text-only addendum, frozen before scoring.**
  (a) Fix the 26 = 12 + 14 wording.
  (b) Add a scope disclosure: test scores come from whole-tile, ungated inference. They are an upper bound for a gated or grid-tiled pipeline and say nothing about cross-grid dedupe.
  (c) Optionally pre-register a *descriptive-only* stratification, not used for selection: GT pieces ≤ 10 m from the own-cell edge vs > 10 m (119 vs 1,141 pieces), and polygons crossing a 100 m tile line vs not (131 vs 1,115). About 119 edge pieces gives wide CIs, so report counts and CIs, not a verdict.
- **D2: r2 unchanged; record D1(a)(b) in RUN.md only.** Cheaper, but the disclosure then lives outside the frozen rules.
- **Not recommended: score in production geometry on the test set** (re-infer per grid with the gate). That needs GPU and redesigns the test after freeze, and it spends the test set on pipeline tuning.

## 4. Pipeline work, separate from the rules (validate on CT-val, never on test)

1. **Gate fix before any gated CT city run (M4/E12):**
   (i) Score each gate cell on a context window (cell + ≥ 10 m margin), or dilate the kept mask by one gate cell.
   (ii) Replace the pixel zeroing with an object rule: keep any predicted component that touches a kept cell, whole.
   (iii) Measure GT objects and area lost to the gate, with and without the fix, on CT-val at 0.05 m.
   Stop rule: if the fixed gate still loses > 1 % of PV area, run without it.
2. **City-wide stitch plus ownership:** stitch all grids together (the gap graph from `stitch_census_v1`, or the IoU merge), then assign each final object to exactly one cell by its representative point. Clip per-cell area only for per-cell totals, so that per-cell census numbers mean what r2's clipped GT means. This is needed for backdating and per-cell analysis: 4.2 % of merged area currently sits under the wrong cell id.
3. **Edgefix tiles:** if the Mask R-CNN census is re-run on full-edge tiles, the halo doubles (10.3 %). Item 2 must be in place first.
4. **Optional:** a CPU-only gate-geometry audit on CT-val, using the 33 m gate-line crossing share and the minor-piece area (computed here on test geometry only).

## 5. Decisions needed from 葉
- D1 or D2 for the test rules (pick one before `score_sealed_v3_r2.sh` runs).
- Whether the MiT city run (M4) keeps a gate at all. If it does, it needs fix 4.1 and a CT-val loss number first.
- Whether ownership/dedupe (4.2) becomes a census Definition-of-Done item.
