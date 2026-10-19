# exp_edge_handling_v1 — grid-boundary handling in the census pipeline (DESIGN ONLY)
2026-10-17 (NZDT). Status: **design, not run. No GPU, no training, no test-set use.** Mode: Exploration (WORKFLOW.md Mode A).
Execution: after 葉 approves, Grok Bot hands this to 光伏普查·开普敦 as a handoff file (`handoff/YYYYMMDD_HHMM_grokbot-to-capetown_edge_handling.md`).
Sources: `cross_grid_summary_v1.md` (+ `recount_v1.json`), `cross_grid_edge_v1/` (results_edges.json, per_cell_halo.csv, census_dupes.json, Claude output), `~/projects/zasolar_verification_20261017/VERIFICATION.md`.
Tags: ✔ = independently recomputed; ⚠ = not verified / estimate.

## 1. Problem (verified context)
- `ct_census_merge.py` tags `gridcell_id` and concatenates; no ownership/dedupe (✔ code, :58, :69-73). Raw inventory 170,605 polys with 14,174 cross-grid overlapping pairs (⚠ Claude only); after the delivered IoU≥0.1 merge 386 pairs / 3,680 m² remain and 3,077 polys (2.8 %) have their rep point outside the tagged cell (✔).
- Current implicit halo: each grid's EPSG:4326 bbox covers the rotated 1 km UTM cell plus a **wedge 0–23.6 m wide (median max 23.6, max 26.4 m), 4.7 % of cell area** (✔ per_cell_halo.csv). So today's halo is <30 m, uneven, and 0 m at the narrow end of the wedge. `full_edge_tiles` (2026-10-14) give up to 65 m (⚠ Claude; check imagery coverage before running).
- Chips never cross chunk seams inside a grid; 11.3 % of test PV area crosses a seam (⚠ Claude; test geometry, used only as context).
- Gate: JHB MiT gate zeroes rejected 33 m cells (✔ code); detect_direct pregate is per chip; CT census ran ungated, Durban gated (✔).
- Panel size (longest side of min. rotated rectangle, v2e labels, polys ≤5,000 m²; script `cross_grid_summary_v1/panel_size_v2e_trainval.py`, ✔ own run, 5 label files unreadable/empty paths — ⚠ check):
  | split | n | p50 | p90 | p99 | p99.9 | max | >30 m | >50 m |
  |---|---|---|---|---|---|---|---|---|
  | train | 5,645 | 6.1 | 11.6 | 29.0 | 57.5 | 59.3 | 50 | 19 |
  | val | 1,046 | 6.5 | 12.9 | 39.3 | 88.7 | 102.2 | 15 | 8 |
  → **30 m is not larger than the largest panel** (≈1 % of objects >30 m; val max 102 m, probably merged arrays ⚠). Halo is therefore an arm parameter: 30 m (requested minimum) and 60 m (> train p99.9 / covers all but ~8 val objects).

## 2. Hypotheses
- H1: A1 (halo + city-wide stitch + dedupe + centroid ownership) raises recall in the 0–10 m edge band to within 2 pts of the >20 m band and cuts cross-cell duplicates and ownership errors to ~0, with no loss of F1 in the >20 m band.
- H2: A2 (re-infer only truncated predictions) reaches ≥90 % of A1's edge-band recall gain at lower compute, but misses objects that were never detected at all (no truncated seed).
- H3 (optional gate): hard-zero gate loses edge recall; "keep whole objects touching a kept cell" recovers most of it at zero extra compute.

## 3. Data — Cape Town val only
- v2e val = v2d2 val, 14 grids, 1,046 polygons (✔ v2e verify), ~1,768 val tiles (VERIFICATION.md). Eval-5 and the Li test set are **not used**.
- **Boundary problem:** val grids are labelled only inside themselves; neighbours are unlabelled, so real cross-grid edges cannot be scored for recall on the far side. Two views:
  - **V-sim (primary, CPU-friendly):** split each 1 km val grid into 2×2 pseudo-cells of 500 m. Internal pseudo-cell lines are labelled on both sides, so edge recall, duplicates and ownership are fully scorable. Each pseudo-cell is "inferred" as the census does it (own bbox + halo h), then stitched.
  - **V-real (secondary):** real val grids + their 8 neighbours inferred as in production; score recall only inside val grids, count duplicates/ownership on all.
- Inputs: v2e val labels (`~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2e_20261017/`, md5 from MD5SUMS); model = current census model (to be named by 葉/开普敦 bot; not chosen here); existing probs only exist for a 60-tile subset (`rescore_v2d1/probs_*`, VERIFICATION.md) → enough for a code dry run, **not** for the experiment.

## 4. Arms
| arm | what | compute |
|---|---|---|
| A0 | current: per-grid bbox (wedge halo), per-grid inference, plain concat (`ct_census_merge.py`); also report A0+delivered IoU≥0.1 merge | baseline |
| A1-30 (preferred) | each cell reads halo h=30 m (uniform buffer of the UTM cell, not the wedge) → stitch city-wide → merge cross-cell duplicates (union if IoU≥0.1 OR overlap/min-area ≥0.5) → assign to exactly one cell by representative point (tie: lowest cell id) | (1.06²−1)=12.4 % read area vs 4.7 % today → ~+7.5 % ⚠ |
| A1-60 | same, h=60 m | 25.4 % vs 4.7 % → ~+20 % ⚠ |
| A2 | no halo; for each prediction touching (≤1 px) its cell edge, re-infer a window centred on it spanning the neighbour (≥ max(2×bbox, 128 m)); replace truncated object; then same merge + ownership | ~ (#truncated × window) / cell; estimate from A0 run ⚠ |
| G (optional) | on A1-30: (g1) context-aware gate (gate window = cell + 30 m), (g2) keep whole objects touching a kept cell, (g3) no gate | g1 small, g2/g3 0 |
Fixed across arms: same model weights, threshold/hysteresis, post-processing, metric_def_version (new: `edge_v1`).

## 5. Metrics (all side by side, never one score)
- Object F1 / P / R at IoU 0.1, 0.3, 0.5 (pieces scored after ownership).
- Recall and duplicate rate by distance of GT centroid to nearest (pseudo-)cell edge: 0–10, 10–20, >20 m.
- Cross-cell duplicate pairs (count, m²) — target 0.
- Ownership errors: objects whose rep point lies outside the assigned cell, and objects assigned to 0 or ≥2 cells.
- Truncation rate: predictions cut at a cell line (A0/A2).
- Boundary F at 0.25/0.5 m (main), 0.3/1.0 m diagnostic; pixel IoU.
- Extra compute % (pixels read and wall-time vs A0) and GPU-h.
- Area total vs GT (bias, CRS EPSG:32734).

## 6. Budget
- CPU: merge/ownership/scoring, V-sim tiling, panel-size check — free.
- GPU (⚠ estimates, **needs 葉's approval**): inference only, no training. 14 val grids ≈ 14 km² ≈ 1,768 tiles. V-sim: A0 + A1-30 + A1-60 ≈ 1 + 1.124 + 1.254 ≈ 3.4× one val pass; A2 re-inference ≈ 0.1–0.2×; V-real neighbours ≈ 8× one pass (optional). Assuming one val pass ≈ 0.2–0.4 GPU-h (to be measured on the 60-tile subset first):
  - core (V-sim, A0/A1-30/A1-60/A2): **≈1.5 GPU-h, hard cap 2 GPU-h**
  - optional V-real: +2–3 GPU-h; optional gate arm: ~0 (reuses A1 probs)
- Proposed to count **outside** the Phase-1 cap (~26 GPU-h, model comparison); if 葉 prefers inside, it leaves ~24 GPU-h. Max wall time per run 1 h (kill + log crash).

## 7. Procedure
1. CPU dry run on the 60 existing prob tiles: implement stitch/merge/ownership/edge-band scoring; unit tests (synthetic straddling panel must yield exactly 1 object, 1 owner).
2. Measure GPU-s per tile on 1 grid; recompute the budget; stop if >2 GPU-h projected.
3. Run A0 → A1-30 → A1-60 → A2 on V-sim; then G on A1-30 probs.
4. Log every run (incl. crashes) to results.tsv; fresh-context recompute of headline numbers before reporting.

## 8. Stop conditions
- Two consecutive failures in one direction (an arm that crashes, or that fails to beat A0 on 0–10 m recall by ≥2 pts without hurting >20 m F1@0.5 by >0.5 pt) → stop, handoff to Claude, then 葉.
- Projected GPU >2 GPU-h, or any need to touch test/eval-5 → stop and ask.
- Any "too good" jump (e.g. >20 m band changes) → treat as bug until verified.

## 9. Decision supported
- 葉 decision 2 in cross_grid_summary_v1: adopt A1 (and which halo: 30 vs 60 m) as the census merge/ownership rule, or A2 if it matches A1 at much lower compute.
- 葉 decision 3: gate policy for any future gated city run (g1/g2/g3).
- Does NOT change r2, test scoring (still on hold), training data, or model choice.

## 10. Risks / not done
- Halo imagery beyond the wedge must exist (`full_edge_tiles`); not checked per val grid ⚠.
- V-sim pseudo-cells use the same imagery chunks; seam effects inside a grid are covered only if chunk seams fall on pseudo-cell lines — check.
- Val max object 102 m may be a merged label, not a panel ⚠.
- GPU estimate unmeasured ⚠.
