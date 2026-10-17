# ZAsolar number verification — 2026-10-17 (NZDT)
Fresh standalone scripts in `scripts/` (dbn.py, labels.py, v2d2.py + inline checks). Read-only on sources, CPU only.
Full table: `verification.csv`. **21 items: 19 match, 2 match only under a specific definition (C9 zero-shot, C11 Mask R-CNN bIoU), 0 hard mismatches.**

## Limits
- Cape Town pixel/object metrics were checked against the raw per-run eval JSONs (counts, P/R, F1 recomputed as 2PR/(P+R)), NOT by re-running inference on the 1,768 val tiles (needs GPU; GPU busy). Probabilities exist only for a 60-tile subset (rescore_v2d1/probs_*).
- Durban areas recomputed from geometry; Gemini judgments themselves not re-audited.

## Definition ambiguities (why IoUs differ between reports)
1. Threshold: evalfull sweeps 0.3–0.7; evalobj headline uses hysteresis h0.9/0.5 (F1-selected). MiT px IoU 0.753→0.758, zero-shot 0.417→0.483 across thresholds. "0.48 zero-shot" is 0.7/hysteresis; at 0.5 it is 0.452.
2. Boundary IoU has two implementations: evalfull (torch, 10 px band, region-dilated) vs evalobj (scipy erosion). Same model/threshold: 0.459 vs 0.434 (MiT t0.3); Mask R-CNN 0.506 vs 0.485.
3. Pixel vs object level: v2d1 is better at pixel IoU (0.761 vs 0.758) but worse at object F1 (0.908 vs 0.926).
4. Ignore (255) pixels: excluded from pixel TP/FP/FN in both scripts; rescore ex255 variant removes the band entirely (F1 0.922/0.907). Object bIoU on subsets changes hugely with this (0.16→0.90).
5. Area CRS: Durban areas in EPSG:32736 (UTM36S); equal-area gives −0.015%. Cape Town labels stored in EPSG:4326, areas computed in EPSG:32734.
6. Area bias depends on threshold: −10.7% @0.3 to −12.7% @0.7 ("−12/−13%" holds at ≥0.6).
7. "Tiles" in v2d2 = 512 px chips (807); 2000 px tiles added = 379.
8. EVAL-5 grids and Mask R-CNN comparison are biased (labels derived from Mask R-CNN; val grids possibly in Mask R-CNN training).

## Data issue found
- `core_hand_sam_20260320_0324/G1630_SAM2_260324.gpkg`, segment_id 130 (Residential): GeometryCollection with a vertex at y=10,000,000 in UTM34S (≈29.6 km²). Total 51.6k m² only holds if it is excluded. Likely the known "bad G1630 label"; fix before any rasterization from these originals.
