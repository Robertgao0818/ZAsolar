# Handoff — deepen the mask-supervision patch lifecycle (review candidate #2)

> **LANDED 2026-06-19.** `MaskSupervisionPatch` added to
> `core/training/boundary_aware_mask.py`; `train.py main()` now constructs it
> and `train_one_epoch` wraps each forward in `with patch.batch()`. **Step A
> only** (behaviour-preserving: `patched_maskrcnn_loss` untouched; install set
> byte-equivalent). Stale-state guard shipped as `assert_fresh_state=True`
> (never fires on reachable paths) — the preferred "assert-no-stale" choice;
> the **Step-B clear** (`clear_after_batch`) is implemented but **OFF/gated**,
> not exercised in production. 14 new CPU tests in
> `tests/training/test_mask_supervision_patch.py` (full suite 332 passed). Doc
> + ADR-0001 updated. See ADR-0001 "候选 #1/#2 后续" for the evidence record.
> The sections below are the original (pre-implementation) handoff, preserved
> as provenance.

- **Created**: 2026-06-19 · saved to OS temp dir (not the repo). Move into `docs/handoffs/` if you want it as a durable provenance artifact like the #1 handoff.
- **Repo**: `/home/gaosh/projects/ZAsolar`
- **For**: a fresh session. This thread is context-heavy (it ran the whole architecture review); start clean from this file.
- **Status**: ~~not started~~ **landed** (see banner above). Interface not locked — design first, then implement. This one touches the **training hot path**, so it is higher-risk than candidate #1.

## Where this came from

2026-06-19 architecture review (`improve-codebase-architecture`, 6-explorer + adversarial-verify workflow). Candidate #2 of 2 survivors out of 26. Adversarial verdict: `real=true, deletion_test_holds=true, overlaps_adr=none`, strength **Worth exploring**. Sibling candidate #1 (`core/polygon_validation.py`, the Top recommendation) has its own handoff at `docs/handoffs/2026-06-19-polygon-validation-extraction.md`. Ephemeral full report: `/tmp/architecture-review-20260619-002918.html`.

## Goal in one line

Give the boundary-aware-supervision monkey-patch bundle a **single lifecycle seam** (a `MaskSupervisionPatch` context manager / class) that owns install → per-batch state → clear → teardown, replacing four module-level globals coordinated by implicit ordering in `train.py main()`.

## What the code actually does (verified 2026-06-19 — supersedes the review's description)

Read `core/training/boundary_aware_mask.py` and `train.py:1060-1215`. The review under-described this; the verified picture:

The module bundles **three** monkey-patches over torchvision internals, sharing **four** module-level mutable states, installed from **scattered conditionals** in `train.py main()`:

| Patch | Installs via | Module state it reads/writes |
|---|---|---|
| `maskrcnn_loss` → `patched_maskrcnn_loss` | `install_patch()` — **two** call sites, `train.py:1070` and `:1115` (different flag branches) | `_BATCH_STATE` (ignore_masks, mask_weights, mask_pixel_weights, label_sources) |
| `transform.forward` wrapper (aux-resize + **auto-stash**) | `install_transform_aux_resize(model)` `train.py:1158` | writes `_BATCH_STATE` via `stash_batch_supervision()` at `boundary_aware_mask.py:260` |
| `fastrcnn_loss` + `select_training_samples` wrap (per-source box-loss telemetry) | `install_fastrcnn_patch()` + `wrap_select_training_samples(model)` `train.py:1170-1171` | `_BOX_LOSS_BUCKETS`, `_LAST_MATCHED_IDXS`, `_LAST_LABEL_SOURCES` |

**Three verified facts the next session must build on (the review got the first one wrong):**

1. **`train.py` never calls `stash_batch_supervision` manually.** Grep confirms zero manual stash calls. Stashing happens *only* automatically inside the transform wrapper (`patched_forward`, gated on `transform.training and "masks" in tgt`). The module's own docstring (lines 26-35) shows a manual `stash → model → clear` usage example that **does not match production** — that stale docstring is part of the friction.
2. **`clear_batch_supervision()` is dead in production.** Called only from `scripts/training/jhb_phaseA/test_boundary_loss.py` (3×), never in `train.py`. `_BATCH_STATE` is never cleared between batches; it works only because the transform wrapper overwrites it every training forward.
3. **`_clear_matched_info()` is dead everywhere** — the box-loss stash is never cleared either; relies on `wrap_select_training_samples` overwriting per call.

## The friction, precisely

- **Documented contract ≠ actual contract.** A reader following the module docstring is wrong about how training runs. Latent confirmation cost on every future change.
- **No single seam owns "this model is patched for boundary-aware supervision."** You assemble it from 4 calls in a specific order, decided by `if`-branches in a 1000-line `main()`. Install ordering and "which patches go together" are implicit.
- **Latent correctness hazard (the real prize).** Because clear is never called, if a batch ever skips the auto-stash (transform not in `.training`, or a target lacking `"masks"`), **stale supervision from the previous batch silently feeds the loss**. Today that path may be unreachable — but nothing in the interface guarantees it, and a future change could open it. A context manager that pairs stash/clear via `__enter__`/`__exit__` makes the hazard structurally impossible.

## Deletion test (why deepen, not delete)

Holds. The patches concentrate genuine complexity — per-pixel ignore band + per-instance weight + per-pixel soft weight in the mask loss, plus per-source box-loss telemetry — that any trainer reusing this supervision would otherwise replicate. The move is to give the lifecycle a seam, not to remove the patches.

## Hard constraints

1. **Not purely behaviour-preserving — this is the crux.** Candidate #1 was a clean byte-equivalent move. Here, *actually calling clear* could change gradients vs today (which never clears). So split the work:
   - **Step A (behaviour-preserving):** wrap today's exact behaviour (auto-stash via transform wrapper, **no** clear) behind the new seam. Prove identical loss/grads.
   - **Step B (intentional fix, separate + gated):** decide whether to close the stale-state leak by clearing. If yes, treat as a deliberate semantics change with its own validation and a note in the experiment ledger — never fold it silently into Step A.
2. **Validation is harder than a file diff** (GPU, training). Reuse the existing equality-test pattern in `scripts/training/jhb_phaseA/test_boundary_loss.py` (patched-vs-stock loss equality on a fixed batch). The byte-equivalence discipline from ADR-0001 (全程铁律) still applies: a fixed batch must produce identical loss tensors before vs after the refactor.
3. **Idempotency must survive.** `install_patch` is idempotent; the transform/select wrappers guard with `_aux_resize_installed` / `_select_training_samples_wrapped`. The new object must preserve that (re-entrancy, repeated install).
4. **`overlaps_adr=none`** — distinct from ADR-0001 side-item #14 (`TrainRunConfig`, which is CLI-flag plumbing). Do not merge the two; this is runtime patch lifecycle, not config. If you also touch train.py flags, keep the diffs separable.
5. **Module-move convention (ADR-0001 D3):** if you relocate anything, leave an import shim; never a second implementation. `test_boundary_loss.py` imports the module directly — keep its imports working.

## Proposed direction — settle in design, do not treat as final

A `MaskSupervisionPatch` object owning the bundle, e.g.:

```python
patch = MaskSupervisionPatch(model, enable_box_loss_telemetry=..., enable_aux_resize=...)
patch.install()                 # asserts ordering; wraps transform / fastrcnn / select_training_samples
for images, targets in loader:
    with patch.batch(targets):  # auto-stash on enter, clear on exit  (Step B behaviour)
        loss_dict = model(images, targets)
patch.teardown()                # restore_original / restore_fastrcnn for clean test isolation
```

Open question: in **Step A** the context manager must reproduce today's *no-clear* behaviour exactly (or prove clear is a no-op on every reachable path). Only **Step B** flips on the clear.

## Open design decisions (for grilling / interface design)

1. One object for all three patches, or compose smaller ones (mask-loss vs box-telemetry are independent levers)?
2. Does the seam own per-batch stash, or just install/teardown (leaving stash to the transform wrapper as today)?
3. Step B: actually clear, or assert-no-stale (raise if loss runs with state from a prior batch)? The assert is safer for validating that the leak is currently unreachable before changing anything.
4. Where does the box-loss telemetry readout (`box_loss_bucket_means`, `reset_box_loss_buckets` at `train.py:1187-1214`) attach — to the same object, or stay separate?
5. Keep monkey-patching torchvision module attributes, or move to a proper `RoIHeads` subclass? (Bigger change; the patch comment at `boundary_aware_mask.py:16-18` argues attribute-swap is sufficient. Likely out of scope — note and defer.)

## Acceptance criteria

- [ ] One lifecycle seam owns install/teardown (and, in Step B, per-batch clear); `train.py main()` expresses intent instead of ordering 4 calls.
- [ ] **Step A**: fixed-batch loss/grad identical pre/post (extend `test_boundary_loss.py`).
- [ ] Stale-state hazard either proven unreachable (assert-based test) or closed in a separately-gated Step B with ledger note.
- [ ] Idempotency + teardown covered by tests; `test_boundary_loss.py` imports still resolve (shim if moved).
- [ ] Module docstring updated to match actual production usage (kill the stale manual-stash example).
- [ ] No merge with TrainRunConfig (#14); `docs/architecture.md` + ADR-0001 updated in the same commit if files move.

## First moves for the fresh session

1. Read `core/training/boundary_aware_mask.py` (whole file) + `train.py:1060-1215` + `scripts/training/jhb_phaseA/test_boundary_loss.py`.
2. Map every reachable training path and confirm whether the auto-stash-without-clear leak is reachable today (drives Step B's shape).
3. Design the seam (skills below), then Step A behind the equality test, then decide Step B.

## Suggested skills

- **`design-an-interface`** — primary. Generate several shapes for the `MaskSupervisionPatch` seam (one object vs composed, who owns per-batch state) before committing.
- **`codebase-design`** — load the deep-module vocabulary (module/interface/depth/seam/locality/leverage) the review used, so the design conversation stays consistent.
- **`tdd`** — pin the lifecycle invariants test-first: (a) Step-A equality on a fixed batch, (b) idempotent install, (c) the stale-state assertion. The hazard is exactly what a red test should expose before any change.
- **`improve-codebase-architecture`** — optional, to re-enter the grilling loop on this specific candidate (it will write `CONTEXT.md` / offer an ADR as decisions crystallise).
- **`diagnosing-bugs`** — only if step 2 suggests the stale-state leak is actually reachable; then confirm a concrete repro before "fixing" it.

## Pointers (reference, not duplicated here)

- `core/training/boundary_aware_mask.py` — the patch bundle (413 lines).
- `train.py:1060-1215` — install ordering + box-loss readout.
- `scripts/training/jhb_phaseA/test_boundary_loss.py` — existing patched-vs-stock equality harness (reuse for validation).
- `docs/adr/0001-codebase-optimization-2026-06.md` — deepening track, 全程铁律 (byte-equivalence), D3 (move+shim), side-item #14 (do not merge).
- `docs/handoffs/2026-06-19-polygon-validation-extraction.md` — sibling candidate #1.
- `docs/evaluation_protocol.md` — what is locked.
