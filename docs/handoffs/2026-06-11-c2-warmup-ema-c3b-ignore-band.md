# C-2 (warmup + EMA) & C-3(b) (boundary ignore band) — pod usage

> **Status: code + CPU tests landed 2026-06-11 (no GPU run yet).** This is the
> training-recipe lever pack from the F1-gap Tier C plan
> ([`../plans/2026-06-10-rcnn-f1-gap-review.md`](../plans/2026-06-10-rcnn-f1-gap-review.md)
> lines 190-203). All three levers are **flag-gated and OFF by default** — with
> the flags unset, `train.py` is byte-for-byte equivalent to its prior behavior
> (verified: warmup-disabled scheduler is step-for-step identical to the bare
> `CosineAnnealingLR`; no EMA tensors/files; the fixed `boundary_band_iters` path
> is unchanged).

## What landed

| Lever | Flag(s) | Module |
|---|---|---|
| C-2 linear LR warmup | `--warmup-iters`, `--warmup-start-factor` | `core/training/warmup_ema.py` |
| C-2 weight EMA | `--ema`, `--ema-decay` | `core/training/warmup_ema.py` |
| C-3(b) area-adaptive ignore band | `--boundary-ignore-band`, `--boundary-ignore-band-thresholds` | `core/training/boundary_ignore_band.py` |

Run-ledger (commit `8d93473`) records all new flags in the manifest fingerprint
(`scripts/training/run_ledger.py::HYPERPARAM_KEYS`), so a warmup/EMA/band run and
its legacy-recipe sibling get **distinct deterministic `run_id`s** — attribution
is preserved (the dataset `build_id` + seed are identical; the recipe is the only
delta). EMA checkpoints are added to `output_checkpoints` in the ledger.

## C-2 details

### Warmup (Stage 2 only)
- Ramps LR from `warmup_start_factor * lr2` → `lr2` over the first `warmup_iters`
  optimizer steps, then cosine-anneals over the **remaining** budget
  (`total_steps - warmup_iters`). Warmup eats into, not adds to, the step budget
  → the schedule length is identical to a legacy run.
- **Applies to Stage 2 only — by design.** Stage 2 hot-swaps to a fresh optimizer
  and a `CosineAnnealingLR` that cold-starts from full `lr2` on its first step
  (`train.py` ~L1509-1523). The warmup smooths exactly that cold start. Stage 1
  (heads-only / `--freeze-mask-head`) keeps its **flat** SGD LR with no scheduler;
  adding a second warmup there would double-count the gentle phase. The design is
  pinned in the `warmup_ema.py` module docstring.
- Resume-safe: warmup lives in the first `warmup_iters` Stage-2 steps; the
  Stage-2 resume fast-forward (`for _ in range(stage2_start*len)... scheduler.step()`)
  advances the composed `SequentialLR` past it correctly.

### EMA
- `ema = 0.999 * ema + 0.001 * online` after **every** optimizer step (both
  stages). Integer buffers (`num_batches_tracked`) are copied verbatim, not
  blended.
- **Dual checkpoint family** (attribution requirement): a single job emits BOTH
  - raw-best: `best_model.pth` / `best_ap50_model.pth` / `final_model.pth`
  - EMA-best: `best_model_ema.pth` / `best_ap50_model_ema.pth` / `final_model_ema.pth`
  The EMA shadow is evaluated each eval epoch (online weights swapped out and
  restored) and the EMA-best files are written on EMA-metric improvement. The raw
  selection path is **untouched** by EMA being on.
- Pick the winner downstream by running BOTH `*_ema.pth` and the raw `*.pth`
  through the full `area_aggregate_eval.py` + poly-conf sweep, both merge modes.

### Explicitly NOT implemented (do not add — plan deletions)
- **SWA** — epoch averaging over ~10k noisy SAM-GT chips has a bulk-overshoot
  countersignal (train20). Revisit only after GT cleanup.
- **multi-scale `min_size` list** — torchvision eval takes `min_size[-1]`, so a
  list like `[640,800,960]` silently changes inference resolution to 960 and
  breaks all historical comparability. **Inference `min_size` must stay 800.**

## C-3(b) details

`--boundary-ignore-band` replaces the FIXED `--boundary-band-iters` width with an
**area-adaptive** ignore band on the per-source mask-BCE weight map:

| target size (mask-pixel area) | band half-width |
|---|---|
| small (< `small_max`, default 400 px) | 1 px |
| medium (< `medium_max`, default 2500 px) | 2 px |
| large (≥ `medium_max`) | 3 px |
| S-class (`sam_refined_review`, `sam_added_true_fn`) | always 3 px (force large) |
| R-class (`reviewed_prediction` etc., `boundary_w`=0) | band ignored, **core still supervised** |

- It is the **band-width policy** for `--per-source-mask-weight`; it does nothing
  on its own. Always pair it with `--per-source-mask-weight`.
- The ignore is a per-pixel **mask-BCE weight only** — it does NOT touch box or
  cls loss (those keep the full polygon). This is the orthogonal lever Phase A's
  post-mortem left un-ablated (Phase A's ignore reached the *mask* loss but its
  boundary band still contributed box+cls; `boundary_aware_mask.py` docstring
  self-documents that).
- Thresholds are in **mask-pixel area** (at the 400 px chip resolution the band
  is built on; GSD ~6.7 cm → 1 px ≈ 0.067 m). Override with
  `--boundary-ignore-band-thresholds "small_max,medium_max"`.

### ⚠️ C-3(b) RETRAIN GATE — read before scoring tomorrow's run
C-3(b) is a **single-lever single-retrain** change. Its success criteria are
**bulk_ratio / σ_Bw / area_F1** vs the unified_A baseline on the locked
JHB CBD25 `clean_gt`, scored in **BOTH** merge modes (pixel-or + per-detection)
via `scripts/analysis/area_aggregate_eval.py` + polygon-conf sweep.

**Do NOT book any polygon-F1 / polygon-recall gain as the win.** A boundary
ignore band is a boundary-quality lever, not a recall lever — if polygon recall
moves, that is noise or a confound, not the deliverable. Gate = area-side
improvement (lower σ_Bw / RMSE, bulk in [0.5, 2.0]) with **no area_F1
regression**. (The gate is also pinned in the `boundary_ignore_band.py` docstring.)

## Retrain command examples (run on pod tomorrow)

Baseline data prep + warm-start are unchanged; only the recipe flags are new.
Adjust `--coco-dir`, `--pretrained`, batch size, and `--epochs2` to the actual
unified_reviewall build. `warmup_iters` should be ~3-5% of total Stage-2 steps;
at batch 32 over the unified set that lands ~500-1000.

```bash
# ── C-2: warmup + EMA, riding the C-1 retrain (no dedicated GPU) ──────────
# Same dataset manifest + seed as the legacy recipe so recipe is the only delta.
python train.py \
  --coco-dir /workspace/coco/unified_reviewall_v2 \
  --pretrained checkpoints/exp003_C_targeted_hn/best_model.pth \
  --output-dir checkpoints/c2_warmup_ema \
  --seed 42 \
  --freeze-mask-head --per-instance-mask-trusted --per-source-mask-weight \
  --diff-lr-backbone-mult 0.1 \
  --warmup-iters 750 --warmup-start-factor 0.01 \
  --ema --ema-decay 0.999
# → emits raw-best AND EMA-best; score both with area_aggregate_eval + sweep,
#   both merge modes (per-detection via finalize.py, pixel-or).

# ── Legacy sibling for attribution (no warmup, no EMA) ───────────────────
# Identical except the recipe flags. Distinct run_id in the ledger.
python train.py \
  --coco-dir /workspace/coco/unified_reviewall_v2 \
  --pretrained checkpoints/exp003_C_targeted_hn/best_model.pth \
  --output-dir checkpoints/c2_legacy \
  --seed 42 \
  --freeze-mask-head --per-instance-mask-trusted --per-source-mask-weight \
  --diff-lr-backbone-mult 0.1

# ── C-3(b): area-adaptive boundary ignore band (separate single retrain) ──
# Pair with --per-source-mask-weight (it is that lever's band-width policy).
python train.py \
  --coco-dir /workspace/coco/unified_reviewall_v2 \
  --pretrained checkpoints/exp003_C_targeted_hn/best_model.pth \
  --output-dir checkpoints/c3b_ignore_band \
  --seed 42 \
  --freeze-mask-head --per-instance-mask-trusted --per-source-mask-weight \
  --diff-lr-backbone-mult 0.1 \
  --boundary-ignore-band \
  --boundary-ignore-band-thresholds 400,2500
# Optionally stack C-2 onto the same job, but keep C-3(b) attributable: run it
# as its OWN single-lever retrain first per the plan's "one lever at a time" rule.
```

Follow `.claude/rules/05-runpod-inference.md` / `08-runpod-large-files.md`: build
the COCO set on the pod from tiles+annotations (don't re-upload), copy hot data to
`/dev/shm`, use `--postproc-config configs/postproc/v4_canonical.json` at eval.

## Files

- `core/training/warmup_ema.py` (new) — `build_warmup_cosine_scheduler`, `ModelEMA`.
- `core/training/boundary_ignore_band.py` (new) — `BandConfig`,
  `adaptive_boundary_pixel_weights`, `parse_band_thresholds`.
- `train.py` — argparse flags, Stage-2 scheduler wiring, EMA construction +
  dual-checkpoint selection, dataset band wiring, history + ledger plumbing.
- `scripts/training/run_ledger.py` — `HYPERPARAM_KEYS` += the new recipe flags.
- `tests/training/test_warmup_ema.py`, `tests/training/test_boundary_ignore_band.py`
  (new, CPU) — 54 tests; full suite 163 passed.
