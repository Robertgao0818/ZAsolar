# Architecture optimization landing — steps 1/3/5/6/8/9 (2026-06-12)

Source review: [`docs/plans/2026-06-12-architecture-review.html`](../plans/2026-06-12-architecture-review.html)
(15 adversarially-verified candidates, 11-step plan). This session landed the six
no-prerequisite steps via 4 parallel agent chains (file-disjoint, per
`feedback_parallel_agents_shared_tree` discipline) + resolved the merge-mode config conflict.

> **Living tracker (checkboxes + decision log D1–D6):**
> [`docs/adr/0001-codebase-optimization-2026-06.md`](../adr/0001-codebase-optimization-2026-06.md)
> — this handoff is the frozen evidence record for the 2026-06-12 landing; consult the ADR for current status.

**State: all changes uncommitted in the working tree** (main repo + 2 files in `solar_cls`).
Combined-tree `pytest tests/` = **252 passed**. Net main-repo diff ≈ +373/−928 lines.

## Landed

| Step | Candidate | What | Equivalence evidence |
|---|---|---|---|
| 1 | #1 | `compute_iou`+`iou_matching` → **`core/eval_matching.py`**, one-line shim left in `detect_and_evaluate.py` (no second implementation) | synthetic snapshot byte-identical (14 scenarios); 18 new tests (`tests/eval/`); import no longer pulls matplotlib/`set_grid_context`; 5 external callers resolve |
| 3 | #6 | Tier-1 kernel (`summarize`/OLS/bootstrap) → **`core/area_metrics.py`**; `area_aggregate_eval.py` keeps I/O + re-exports for back-compat | 6-scenario snapshot byte-identical; end-to-end rerun on real run (jhb_phaseA_vexcel, 3 grids) matches committed CSVs; 18 new tests |
| 5 | #5 | `ModelRunConfig.deprecated` field; 3 raw-YAML back-channels replaced with typed accessor (`area_aggregate_eval.py` + 2 × `solar_cls/scripts/classifier/`) | before/after deprecated-flag maps identical (29 runs, only `v3c_geid_2024_02` True); 11 registry tests |
| 6 | #9 | legacy chain writes `merge_mode='per_detection_geoai'` into `config.json` (cache-safe via `_CACHE_IGNORE_KEYS`); `finalize.py` raises on CLI-vs-JSON merge_mode conflict; overnight.sh comment fixed | conflict raise tested both directions; historical caches unaffected (field stripped from comparison) |
| 8 | #4 | 9 loaders → **`core/training/positive_sources.py`** (`review_root` explicit param, lru_cache); `dataset_builder.py` monkeypatch deleted; `build_unified_reviewall.py` 720→534 (thin CLI) | full v2 dry-run `build_manifest.json` content byte-identical (fingerprint `d01e1bf1`, 68 CT grids, 5903 annotations, 1976/3927 trusted/untrusted); 7 new tests |
| 9 | #7 | crop/write/resolve → **`core/chip_extraction.py`**; 5 consumers rewired; **mosaic silent-HN-drop bug fixed** + `export_v4_hn` region= backfilled | real CT G1632 chip md5 identical; mosaic regression fixated in unit test; 11 new tests + 95 related pass |

## Decision executed: merge-mode → per-detection (user, 2026-06-12)

`merge_mode` **removed from `configs/postproc/v4_canonical.json`**. Policy now:
direct-chain callers declare `--merge-mode` explicitly on the CLI; JSON-vs-CLI
disagreement raises. This resolved **two** live conflicts the step-6 raise would
have tripped: `runpod_vexcel_jhb382_overnight.sh` (CLI per-detection) and
`validate_checkpoint.py`'s per-det leg (canonical JSON + CLI per-detection).
All pixel-or callers already pass the flag explicitly → behavior unchanged.
Legacy chain ignores the key (whitelist loader) → zero cache impact.
`v4_poly_diag.json` keeps its own `merge_mode` (diagnostic-only, no-CLI by design).

## Notable findings vs the review doc

- `hn_ops.py` lives at `pipeline/hn_ops.py`, not `scripts/training/`.
- The review's 5th eval-matching caller `evaluate_predictions.py` doesn't exist; the real 5th is `repostprocess.py`.
- The "spatial_nms left a duplicate after extraction" cautionary tale is false — it was never extracted (still inline at `detect_and_evaluate.py`).
- **`build_training_pool._source_to_label_source` fork is divergent by design** (46/112 value-matrix cells differ: train loader fail-fast raises on `google_earth`/unknown; pool builder fail-closed returns `legacy_weak_supervision`/`None`). NOT unified; locked by `test_build_training_pool_fork_is_divergent_by_design`.
- One intentional tightening in step 9: hn_ops' chunked glob `*.tif` → canonical `{grid}_*_*_geo.tif` (matches rule 06 + the other 3 consumers).

## Outstanding

1. **`sync_from_runpod.sh` L-namespace drop (candidate #11) — STILL LIVE DEFECT**, on the CT census pull path (`grep '^G[0-9]+'` silently drops L1842–L1954). Not in this batch's scope; ROADMAP flags it for early standalone fix.
2. Remaining plan steps: **2** (callers → `core.eval_matching`), **4** (route 2 Tier-1 copies → `core.area_metrics`), **7** (run_provenance, deps 5+6 now met), **10** (merge-HN unification, dep 9 met), **11** (alias narrow surgery + `resolve_gt_spec` + postproc chain — needs fall-through vs first-match semantic ruling first).
3. Side items unscheduled: #10 building_filter archive, #14 TrainRunConfig, #15 CANONICAL_DETECT_ARGS.
4. Step-8 JHB integrated dry-run needs `vexcel_2024` tiles (pod `/workspace`); CT path fully gated, JHB loader gated via geometry-hash snapshot only.
5. Commit pending user review — changes are file-disjoint per step, cleanly splittable into 6 commits (+1 for merge-mode decision).
