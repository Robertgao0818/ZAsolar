# G7 cross-tile / cross-grid merge — written plan (Grok Build: plan first, then code, then Codex review)
2026-10-19 NZDT. Source: ~/projects/zasolar_plans_20261017/cross_grid_edge_proposal_v1.md §4.2, exp_edge_handling_v1.md.

## Problem in the production (MiT/ConvNeXt, 100 m tile) pipeline
1. Each 100 m tile is inferred and **binarized alone**. With h0.9_0.5 hysteresis, a lo-component (p>=0.5) cut by a tile line
   is kept only on the side whose own max >= 0.9 -> the other piece of a large array is dropped (truncation).
2. Objects are 8-connected components per tile; nothing merges pieces across tile or 1 km grid lines in the output.
3. No ownership: a merged object must belong to exactly one 1 km cell.

## Implementation (xtile_merge.py, pure CPU, no model code)
- Global pixel key (gx,gy) = (round(c/gsd)+x, round(-f/gsd)+y) in the tile CRS (EPSG:32734); **no grid id in the key**, so
  merges cross 1 km grid lines. Tiles must not overlap (asserted).
- Pass 1: per tile, label prob>=lo (8-conn), record per-component max prob and border pixels; union-find over 8-neighbours
  across tile borders. Pass 2: keep every component whose *merged* max >= hi (cross-tile hysteresis); emit per-tile masks
  and a global object id raster per tile. For plain threshold t, pass 2 just keeps prob>=t with merged ids.
- Polygons: rasterio.features.shapes per tile on object ids, dissolved by global id (shapely union) -> one polygon per object.
- Ownership: representative_point of each merged polygon -> the 1 km cell containing it (tie/none: nearest cell, lowest id).

## Validation (val only; test, labels, v2f untouched)
- Model ConvNeXt-T s0 (locked candidate), existing cached val probs (phase1_followup_valprobs_20261018, 1,703 tiles, 14 grids).
  No new GPU inference needed (GPU approval not used).
- Arms at h0.9_0.5 (and t0.5 for reference):
  A0 per-tile binarize, no merge (production as-is);  A1 per-tile binarize + cross-tile merge (= current scorer rule);
  A2 cross-tile hysteresis + merge (proposed production).
- Metrics: object P/R/F1 at IoU 0.5 (and 0.3) with pinned zaeval ObjectAccumulator (exclude-pixels, min 0.25 m2);
  pixel IoU on valid pixels; total predicted area and n objects; area change A0->A2; # objects merged across tiles.
- Unit tests: synthetic straddling panel -> exactly 1 object, 1 owner; hysteresis seed on one side keeps both sides.
