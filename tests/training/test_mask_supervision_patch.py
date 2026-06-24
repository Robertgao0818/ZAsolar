"""CPU tests for the MaskSupervisionPatch lifecycle seam (2026-06-19).

Locks the behaviour of the single lifecycle seam added over the boundary-aware
mask-supervision monkey-patch bundle in ``core.training.boundary_aware_mask``
(architecture-deepening candidate #2; handoff
``2026-06-19-mask-supervision-lifecycle``). The seam owns
``install → per-batch boundary → teardown`` for the three torchvision patches
that ``train.py main()`` previously assembled from four scattered calls.

What these tests pin:

- **Step-A byte-equivalence (structural):** ``patched_maskrcnn_loss`` is
  *unchanged* — the seam only changes *when/how* it is installed. So the
  active loss after ``install()`` is the same function object, and on a
  no-aux fixed batch it still equals stock ``maskrcnn_loss`` (the existing
  ``test_boundary_loss.py`` equality, re-anchored through the seam).
- **Install lever → patch set** mirrors train.py exactly (aux-only vs
  +box-telemetry), so the set of installed patches stays byte-equivalent.
- **Idempotent install** and **teardown restores** torchvision + the model
  (clean test isolation / a second in-process run).
- **Stale-state guard:** ``batch()`` raises ``StaleSupervisionError`` when a
  forward computed the mask loss without a fresh stash, and passes when the
  stash ran — the structural proof that today's silent-leak path is closed.
- **Step-B gate:** ``clear_after_batch`` actually clears (opt-in); the default
  leaves state in place (documenting today's no-clear reality).

Pure CPU. The model-level tests build a randomly-initialised
``maskrcnn_resnet50_fpn`` (no weights download) and are skipped if torchvision
is unavailable; the loss-level + lifecycle tests need no model at all.
"""
from __future__ import annotations

import pytest
import torch

from core.training import boundary_aware_mask as bam
from core.training.boundary_aware_mask import (
    MaskSupervisionPatch,
    StaleSupervisionError,
)


# ── shared fixtures ─────────────────────────────────────────────────────────
def _fixed_loss_batch(n_props: int = 4, M: int = 28, n_classes: int = 2):
    """Deterministic synthetic mask-loss inputs (mirrors test_boundary_loss)."""
    g = torch.Generator().manual_seed(7)
    mask_logits = torch.randn(n_props, n_classes, M, M, generator=g)
    per_img = [n_props // 2, n_props - n_props // 2]
    proposals = [torch.tensor([[5.0, 5.0, 25.0, 25.0]] * n) for n in per_img]
    H = 30
    gt_masks = []
    for n in per_img:
        m = torch.zeros(n, H, H, dtype=torch.uint8)
        m[:, 8:22, 8:22] = 1
        gt_masks.append(m)
    gt_labels = [torch.ones(n, dtype=torch.int64) for n in per_img]
    matched_idxs = [torch.arange(n, dtype=torch.int64) for n in per_img]
    return mask_logits, proposals, gt_masks, gt_labels, matched_idxs


@pytest.fixture
def model():
    """A fresh randomly-initialised Mask R-CNN on CPU (no weights download)."""
    tv = pytest.importorskip("torchvision")
    m = tv.models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=2)
    return m


@pytest.fixture(autouse=True)
def _restore_torchvision():
    """Guarantee module-level torchvision losses are restored after each test —
    they leak across tests (shared module attribute) if a test forgets."""
    yield
    bam.restore_original()
    bam.restore_fastrcnn()
    bam.clear_batch_supervision()


# ── 1. Step-A byte-equivalence: the loss fn is unchanged ────────────────────
def test_seam_installs_the_identical_loss_function(model):
    """install() makes the active maskrcnn_loss the *same object* as before the
    refactor — so loss/grad are identical pre/post for ANY input."""
    assert bam._rh.maskrcnn_loss is bam._ORIGINAL_LOSS  # stock to start
    MaskSupervisionPatch(model, enable_aux_resize=True).install(verbose=False)
    assert bam.is_installed()
    assert bam._rh.maskrcnn_loss is bam.patched_maskrcnn_loss


def test_step_a_no_aux_equals_stock_through_seam(model):
    """With no aux supervision fields, the patched loss the seam installs equals
    stock maskrcnn_loss on a fixed batch (the test_boundary_loss T1 anchor,
    re-run through the lifecycle seam)."""
    MaskSupervisionPatch(model, enable_aux_resize=True).install(verbose=False)
    bam.clear_batch_supervision()  # no stash → no aux fields

    args = _fixed_loss_batch()
    loss_patched = float(bam.patched_maskrcnn_loss(*args))
    loss_stock = float(bam._ORIGINAL_LOSS(*args))
    assert abs(loss_patched - loss_stock) < 1e-5, (
        f"seam changed the no-aux loss: patched={loss_patched} stock={loss_stock}"
    )


# ── 2. install lever → patch set (mirrors train.py) ─────────────────────────
def test_aux_only_installs_mask_and_transform_not_box(model):
    MaskSupervisionPatch(model, enable_aux_resize=True).install(verbose=False)
    assert bam.is_installed()                              # mask-loss patch
    assert bam.is_transform_aux_resize_installed(model)    # transform wrapper
    assert not bam.is_fastrcnn_patched()                   # NOT box telemetry
    assert not getattr(model.roi_heads, "_select_training_samples_wrapped", False)


def test_box_telemetry_installs_full_bundle(model):
    """Box telemetry installs mask patch + transform wrapper + fastrcnn +
    select wrap — exactly what train.py's pre-seam box-loss branch did, even
    with aux_resize off."""
    MaskSupervisionPatch(
        model, enable_aux_resize=False, enable_box_loss_telemetry=True,
    ).install(verbose=False)
    assert bam.is_installed()                              # mask patch (no-op on no-aux)
    assert bam.is_transform_aux_resize_installed(model)    # transform wrapper (stashes label_sources)
    assert bam.is_fastrcnn_patched()                       # box telemetry
    assert getattr(model.roi_heads, "_select_training_samples_wrapped", False)


# ── 3. idempotent install ───────────────────────────────────────────────────
def test_install_is_idempotent(model):
    base_forward = model.transform.forward
    base_sts = model.roi_heads.select_training_samples

    patch = MaskSupervisionPatch(
        model, enable_aux_resize=True, enable_box_loss_telemetry=True,
    )
    patch.install(verbose=False)
    wrapped_forward = model.transform.forward
    wrapped_sts = model.roi_heads.select_training_samples
    assert wrapped_forward is not base_forward
    assert wrapped_sts is not base_sts

    # Second install must not double-wrap (would lose the true original).
    patch.install(verbose=False)
    assert model.transform.forward is wrapped_forward
    assert model.roi_heads.select_training_samples is wrapped_sts
    # And the true original is still recoverable (bound methods compare by
    # ==: same __self__ + __func__, since each attribute access makes a fresh
    # bound-method object).
    assert model.transform._aux_resize_base_forward == base_forward
    assert model.roi_heads._select_training_samples_original == base_sts


# ── 4. teardown restores torchvision + the model ────────────────────────────
def test_teardown_restores_everything(model):
    base_forward = model.transform.forward
    base_sts = model.roi_heads.select_training_samples

    patch = MaskSupervisionPatch(
        model, enable_aux_resize=True, enable_box_loss_telemetry=True,
    )
    patch.install(verbose=False)
    wrapped_forward = model.transform.forward  # the patched closure
    patch.teardown()

    assert not bam.is_installed()
    assert bam._rh.maskrcnn_loss is bam._ORIGINAL_LOSS
    assert not bam.is_fastrcnn_patched()
    assert not bam.is_transform_aux_resize_installed(model)
    # Restored to the original (bound methods compare by ==, not is).
    assert model.transform.forward is not wrapped_forward
    assert model.transform.forward == base_forward
    assert model.roi_heads.select_training_samples == base_sts


def test_teardown_is_idempotent(model):
    patch = MaskSupervisionPatch(model, enable_aux_resize=True)
    patch.install(verbose=False)
    patch.teardown()
    patch.teardown()  # must not raise
    assert not bam.is_transform_aux_resize_installed(model)


# ── 5. stale-state guard (the real prize) ───────────────────────────────────
def test_batch_passes_when_stash_runs(model):
    """A fresh stash during the forward → batch() exits cleanly."""
    patch = MaskSupervisionPatch(model, enable_aux_resize=True).install(verbose=False)
    with patch.batch():
        # Simulate the transform wrapper's auto-stash for this forward.
        bam.stash_batch_supervision([{"masks": torch.zeros(1, 4, 4)}])
    # no exception == pass


def test_batch_raises_when_stash_skipped(model):
    """No stash during the forward → batch() raises StaleSupervisionError (the
    previous batch's supervision would otherwise leak)."""
    patch = MaskSupervisionPatch(model, enable_aux_resize=True).install(verbose=False)
    with pytest.raises(StaleSupervisionError):
        with patch.batch():
            pass  # forward computed loss but no stash happened


def test_batch_no_assert_when_disabled(model):
    """assert_fresh_state=False → no guard (raw Step-A no-clear behaviour)."""
    patch = MaskSupervisionPatch(
        model, enable_aux_resize=True, assert_fresh_state=False,
    ).install(verbose=False)
    with patch.batch():
        pass  # must NOT raise


def test_batch_does_not_assert_on_exception(model):
    """If the forward raises, the loss never ran, so batch() must surface the
    original error — not mask it with a spurious stale-state assertion."""
    patch = MaskSupervisionPatch(model, enable_aux_resize=True).install(verbose=False)
    with pytest.raises(ValueError):
        with patch.batch():
            raise ValueError("forward blew up")


# ── 6. Step-B gate: clear_after_batch ───────────────────────────────────────
def test_default_does_not_clear_state(model):
    """Default (Step A) leaves batch state in place — documents today's
    no-clear reality (state survives because the next stash overwrites it)."""
    patch = MaskSupervisionPatch(model, enable_aux_resize=True).install(verbose=False)
    with patch.batch():
        bam.stash_batch_supervision([{"masks": torch.zeros(1, 4, 4),
                                      "mask_weights": torch.ones(1)}])
    assert bam._BATCH_STATE["mask_weights"] is not None  # NOT cleared


def test_clear_after_batch_clears_state(model):
    """clear_after_batch=True (gated Step B) zeroes the batch state on exit."""
    patch = MaskSupervisionPatch(
        model, enable_aux_resize=True, clear_after_batch=True,
    ).install(verbose=False)
    with patch.batch():
        bam.stash_batch_supervision([{"masks": torch.zeros(1, 4, 4),
                                      "mask_weights": torch.ones(1)}])
    assert bam._BATCH_STATE["mask_weights"] is None  # cleared
    assert bam._BATCH_STATE["ignore_masks"] is None
    assert bam._LAST_MATCHED_IDXS is None


# ── 7. integration smoke: real forward through the seam ─────────────────────
def test_forward_through_seam_stashes_and_passes_assert(model):
    """End-to-end on CPU: a real training forward routes through the transform
    wrapper (auto-stash), the patched mask loss runs, and batch()'s assert is
    satisfied because the stash advanced the counter."""
    patch = MaskSupervisionPatch(model, enable_aux_resize=True).install(verbose=False)
    model.train()

    img = torch.rand(3, 64, 64)
    tgt = {
        "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
        "labels": torch.tensor([1]),
        "masks": torch.zeros(1, 64, 64, dtype=torch.uint8),
    }
    tgt["masks"][0, 12:38, 12:38] = 1

    before = bam.stash_counter()
    with patch.batch():
        loss_dict = model([img], [tgt])
        losses = sum(loss_dict.values())
    assert bam.stash_counter() == before + 1   # auto-stash ran this forward
    assert torch.isfinite(losses)
    losses.backward()  # gradients propagate
    patch.teardown()
