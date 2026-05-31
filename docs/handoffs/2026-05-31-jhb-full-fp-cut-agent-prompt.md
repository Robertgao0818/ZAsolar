# Agent prompt — full JHB Vexcel FP-cut via Gemini two-stage review

> Paste the section below as the next agent's task. It is self-contained. Everything it
> references is committed on `main` (commit `345cfac`). READ
> `docs/handoffs/2026-05-31-jhb-two-stage-fp-review-production.md` first — this prompt assumes it.

---

## MISSION

Run the validated two-stage Gemini FP-review over the **full JHB Vexcel production inventory** to
suppress detector false positives, producing one filtered gpkg per grid that becomes the cleaned JHB
rooftop-solar inventory. You are operating the pipeline, not redesigning it — the reviewer, the
fail-closed gate, the manifest builder and the applier are already built, committed and validated.

## TARGET = the deliverable inventory (do NOT use the raw file)

- Run dir: `results/johannesburg/unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox/`
- **File: `predictions_metric_merge01_c0925.gpkg`** — per-det + SAM-maskbox @ conf 0.925, the canonical
  deliverable (= the renderer's `DEFAULT_PREDICTIONS_FILENAME`). CRS EPSG:32735. Grid ids are `JNB####`.
- **362 grids, 47,465 predictions total** (already thresholded at conf ≥ 0.925):
  - **conf ≥ 0.95 → 44,213 predictions → TWO-STAGE path**
  - **conf 0.925–0.95 → 3,252 predictions → STAGE-1-ONLY path**
- ⚠️ Do NOT run on `predictions_metric.gpkg` (raw pre-merge, 77,251 rows — NOT the inventory).
- Alternative only if the owner says so: `predictions_metric_nms01_c0925.gpkg` (382 grids / 50,150).
  Default to `merge01_c0925`.

## WHAT IS ALREADY BUILT (reuse, don't rebuild) — all on `main`

| script | role |
|---|---|
| `scripts/analysis/build_gemini_review_production_manifest.py` | raw gpkg → renderable candidate manifest (no RA labels); `pred_id` = positional iloc |
| `scripts/training/build_gemini_detection_review_chips.py` | manifest → tight(20 m)+wide(48 m) chips |
| `scripts/analysis/gemini_fp_review_multiscale.py` | **stage 1** dual-crop scorer (+ latency/retry/error soak fields) |
| `scripts/analysis/gemini_fp_review_two_stage.py` | **stage 2** skylight router + fail-closed merge → `production_action`/`auto_drop`/`requires_human_review` |
| `scripts/analysis/check_two_stage_failclosed.py` | **hard gate**: data-level fail-closed invariant, exit≠0 on violation |
| `scripts/analysis/apply_two_stage_decisions.py` | merged decisions → `<grid>_filtered.gpkg` (drops `auto_drop=true`) + `review_queue.csv`. **`--stage1-as-drops`** synthesizes production fields for the LO stage-1-only band (not_pv→drop, pv→keep, abstain→review) and runs the fail-closed gate in-process — no adapter needed. |

Prelaunch (25-grid calib, 2026-05-31): two_stage **pv_recall 0.936 / nonpv_recall 0.851**, fail-closed
**0 violations**, soak **usable 1.0 / no 503**. Greenlit for JHB Vexcel conf ≥ 0.95.

## STEP 0 — PREREQUISITE: SECURE TILES (this is the real blocker)

Rendering needs `vexcel_2024` tiles for every JNB grid in your batch. **Right now only ~2 of 362 JNB
tile dirs are local** (`~/zasolar_data/tiles/johannesburg/vexcel_2024/`). Before rendering any batch:
1. Check RunPod `/workspace` for the tiles (rules `08-runpod-large-files.md`, `05-runpod-inference.md`);
   the 382-grid inference ran there, so the tiles likely persist on the network volume.
2. If absent, fetch via the `download-grids` skill / the established Vexcel pipeline
   (`project_vexcel_jhb_pipeline`).
3. **Gate:** the manifest builder prints `skipped_tile_missing` — it must be ~0 for your batch's grids
   before you spend any Gemini credit. If tiles are missing, candidates silently drop and your cut is
   incomplete.

Consider running the whole job on the pod (tiles + GPU-free rendering + Gemini from there) rather than
pulling 362 grids of tiles local.

## CONFIDENCE ROUTING (required — two separate manifests/runs)

- **conf ≥ 0.95** → two-stage (`--min-conf 0.95`).
- **conf 0.925–0.95** → stage-1 only (`--min-conf 0.925 --max-conf 0.95`); use stage1's
  `pv_present`/`label` directly. **Do NOT run stage2 here** — it was net-negative in the sub-0.95 band.

## BUDGET / CONCURRENCY / BATCHING

- ≈ 47.5k stage-1 calls + a stage-2 skylight subset (~5–15% of the ≥0.95 not_pv rows, rough ~3–6k)
  ≈ **~50–53k Gemini calls**. This is a multi-hour run.
- Gemini account pool ≈ 30 shared slots. Validated `--workers 10 --qps 4`. **Keep Σ(workers) ≤ 30**
  across concurrent runs; do NOT run `score_target_sequence` alongside (it stalls workers).
- **Batch by grid groups** (~25–50 grids/batch). Run under `tmux`/`nohup`. The reviewer is resumable:
  stage2 has `--reuse-stage2-jsonl`; renders are per-grid (re-render only missing); keep one
  output dir per batch.
- **Follow the rollout plan: run 1–2 PILOT batches first** (e.g. re-run the 25 calib grids end-to-end
  + one fresh batch), clear the gate, THEN sweep the rest. Do not fire all 362 grids blind.

## PIPELINE (per batch — `<GLOB>` restricts grids, e.g. `JNB00[0-4]*`)

```bash
RUN=results/johannesburg/unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox
PRED="$RUN/<GLOB>/predictions_metric_merge01_c0925.gpkg"
OUT=data/analysis/gemini_review_calib/prod_jhb/<batch>      # gitignored area

# 1. manifests (two bands)
.venv/bin/python scripts/analysis/build_gemini_review_production_manifest.py \
  --predictions-glob "$PRED" --region johannesburg --imagery-layer vexcel_2024 \
  --min-conf 0.95 --out-csv $OUT/manifest_hi.csv
.venv/bin/python scripts/analysis/build_gemini_review_production_manifest.py \
  --predictions-glob "$PRED" --region johannesburg --imagery-layer vexcel_2024 \
  --min-conf 0.925 --max-conf 0.95 --out-csv $OUT/manifest_lo.csv
#   -> confirm skipped_tile_missing ~0 in BOTH before continuing.

# 2. render tight z20 + wide z48 for each band (default CRS 32735 is correct for JHB)
for band in hi lo; do
  for z in "z20 20 4" "z48 48 5"; do set -- $z
    PYTHONPATH=$PWD SOLAR_TILES_ROOT=/home/gaosh/zasolar_data/tiles .venv/bin/python \
      scripts/training/build_gemini_detection_review_chips.py \
      --candidate-manifest $OUT/manifest_$band.csv --output-dir $OUT/chips_${band}_$1 \
      --chip-size-m $2 --search-radius-m $3 --output-px 768 --max-targets-per-chip 1 \
      --chip-prefix ${band}_$1
  done
done

# 3a. HI band: stage1 -> stage2  (Gemini)
.venv/bin/python scripts/analysis/gemini_fp_review_multiscale.py \
  --tight-chips-csv $OUT/chips_hi_z20/chip_targets.csv --wide-chips-csv $OUT/chips_hi_z48/chip_targets.csv \
  --output $OUT/stage1_hi.jsonl --summary $OUT/stage1_hi_summary.json \
  --model gemini-3-flash-agent --workers 10 --qps 4
#   GUARD: if stage1 summary abstain_rate > 0.30, STOP (env/gateway broken).
.venv/bin/python scripts/analysis/gemini_fp_review_two_stage.py \
  --stage1-jsonl $OUT/stage1_hi.jsonl \
  --tight-chips-csv $OUT/chips_hi_z20/chip_targets.csv --wide-chips-csv $OUT/chips_hi_z48/chip_targets.csv \
  --output $OUT/two_stage_hi.jsonl --stage2-jsonl $OUT/two_stage_hi_stage2.jsonl \
  --summary $OUT/two_stage_hi_summary.json --model gemini-3-flash-agent --workers 10 --qps 4

# 3b. LO band: stage1 only  (Gemini)
.venv/bin/python scripts/analysis/gemini_fp_review_multiscale.py \
  --tight-chips-csv $OUT/chips_lo_z20/chip_targets.csv --wide-chips-csv $OUT/chips_lo_z48/chip_targets.csv \
  --output $OUT/stage1_lo.jsonl --summary $OUT/stage1_lo_summary.json \
  --model gemini-3-flash-agent --workers 10 --qps 4

# 4. HARD GATE (must exit 0 before applying drops)
.venv/bin/python scripts/analysis/check_two_stage_failclosed.py $OUT/two_stage_hi.jsonl

# 5. apply -> filtered gpkgs (+ review queue).
#    HI band: two_stage decisions already carry auto_drop/production_action.
.venv/bin/python scripts/analysis/apply_two_stage_decisions.py \
  --decisions $OUT/two_stage_hi.jsonl --out-dir $OUT/filtered
#    LO band: stage1-only jsonl carries pv_present/label but NOT auto_drop/production_action.
#    Use --stage1-as-drops: it synthesizes the production fields (not_pv->drop, pv->keep,
#    abstain->human-review, NEVER drops an abstain) AND runs the fail-closed gate in-process,
#    writing an auditable stage1_as_drops_decisions.jsonl you can re-check with the standalone gate.
.venv/bin/python scripts/analysis/apply_two_stage_decisions.py \
  --stage1-as-drops --decisions $OUT/stage1_lo.jsonl --out-dir $OUT/filtered_lo

# 6. (optional) materialize cleaned inventory per grid for eval/QA
#    python detect_and_evaluate.py --grid-id JNB#### --region johannesburg \
#      --classifier-filtered-gpkg $OUT/filtered/JNB####_filtered.gpkg ...
```

> NOTE on the LO band: `gemini_fp_review_multiscale.py` emits `pv_present`/`label` but not the
> production fields, and `apply_two_stage_decisions.py` only drops on `auto_drop=true`. This gap is
> now closed by the built-in **`--stage1-as-drops`** flag (landed 2026-05-31) — no hand-written
> adapter. It maps `label=="not_pv"`→drop, `"pv"`→keep, abstain/unusable→human-review (an abstain is
> NEVER dropped), runs `check_two_stage_failclosed.validate_row` on every synthesized row, and aborts
> (exit 2) on any violation. So LO `not_pv` rows DO become auto-drops, but only behind the same
> fail-closed gate as the HI band — never silently.

## PER-BATCH ACCEPTANCE GATE (every batch, before drops are trusted)

- **fail-closed = 0 violations** (`check_two_stage_failclosed.py` exit 0). HARD.
- **stage1 abstain_rate ≤ 0.30** on each run (else STOP — env/gateway issue, not a result).
- **soak: usable_rate ≥ 0.97, no 503 / account-exhausted cluster** (aggregate the `latency_ms` /
  `retry_count` / `error_type` fields PATCH1 logs per row).
- **drop sanity**: spot-audit with `build_stage2_flip_audit.py` (flips rare, drop precision ~0.85).

## REPORT (per batch + final roll-up)

- Per batch: grids, n_candidates (hi/lo), n_dropped, n_kept, n_review, n_undecided, FP-cut rate,
  gate status, errors.
- Final: inventory before/after = **47,465 → 47,465 − Σ(dropped)**; per-grid `<grid>_filtered.gpkg`
  location; consolidated `review_queue.csv` size; soak metrics; gate scorecard.

## OUT OF SCOPE / KNOWN ISSUES

- **Cape Town is OUT** — cross-imagery-source fail; this task is JHB Vexcel only.
- **low-conf stage2 OFF** (routing above).
- **G0890-type over-keep**: stage2 can wrongly restore a skylight embedded in a PV array (the ROADMAP
  geometry-layer guard follow-up). Track via the review queue / flip-audit; do not block on it.
- The **20 grids without `merge01_c0925`** have no detections — nothing to cut.

## DEFINITION OF DONE

1. Every target grid (362) has a `<grid>_filtered.gpkg` with `auto_drop` applied.
2. `check_two_stage_failclosed.py` exits 0 on every batch's decisions.
3. `review_queue.csv` consolidated for human review.
4. Final report: before/after inventory counts + gate scorecard + soak summary.
