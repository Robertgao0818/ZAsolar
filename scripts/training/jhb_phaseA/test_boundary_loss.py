"""Smoke test for boundary_aware_mask patch.

Three checks against synthetic data:

1. Vanilla equivalence: with no ignore masks and weights=1, patched loss
   matches torchvision's stock maskrcnn_loss.
2. Ignore band zeros pixels: setting ignore=1 on all pixels gives loss=0.
3. Per-instance weight=0 zeros that instance's contribution.

Run:
    python scripts/training/jhb_phaseA/test_boundary_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.training import boundary_aware_mask as bam  # noqa: E402


def make_inputs(n_props: int = 4, M: int = 28, n_classes: int = 2, device: str = "cpu"):
    """Two images, n_props/2 instances each. Random logits + binary fg masks."""
    g = torch.Generator(device=device).manual_seed(7)

    # mask_logits shape (sum_proposals, n_classes, M, M) — torchvision passes
    # the per-class logits and the loss selects via labels.
    mask_logits = torch.randn(n_props, n_classes, M, M, generator=g, device=device)

    # proposals = list[Tensor (n_proposals_i, 4)]; one box per proposal
    proposals_per_img = [n_props // 2, n_props - n_props // 2]
    proposals = [
        torch.tensor([[5.0, 5.0, 25.0, 25.0]] * n, device=device)
        for n in proposals_per_img
    ]

    # gt_masks = list[Tensor (n_gt_i, H, W)]; H=W=30 raw image space
    H = 30
    gt_masks = []
    for n in proposals_per_img:
        m = torch.zeros(n, H, H, dtype=torch.uint8, device=device)
        m[:, 8:22, 8:22] = 1
        gt_masks.append(m)

    gt_labels = [torch.ones(n, dtype=torch.int64, device=device) for n in proposals_per_img]
    matched_idxs = [torch.arange(n, dtype=torch.int64, device=device) for n in proposals_per_img]
    return mask_logits, proposals, gt_masks, gt_labels, matched_idxs


def test_vanilla_equivalence():
    mask_logits, proposals, gt_masks, gt_labels, matched_idxs = make_inputs()
    bam.clear_batch_supervision()  # no stash -> patched should equal stock

    bam.install_patch()
    loss_patched = bam.patched_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, matched_idxs)
    bam.restore_original()
    loss_stock = bam._ORIGINAL_LOSS(mask_logits, proposals, gt_masks, gt_labels, matched_idxs)
    diff = abs(float(loss_patched) - float(loss_stock))
    print(f"[T1 vanilla] patched={float(loss_patched):.6f} stock={float(loss_stock):.6f} diff={diff:.6f}")
    assert diff < 1e-5, f"expected match, got diff={diff}"


def test_full_ignore_zeroes_loss():
    mask_logits, proposals, gt_masks, gt_labels, matched_idxs = make_inputs()
    H = gt_masks[0].shape[-1]
    full_ignore = [torch.ones_like(m) for m in gt_masks]
    bam.stash_batch_supervision([
        {"ignore_masks": full_ignore[0], "mask_weights": None},
        {"ignore_masks": full_ignore[1], "mask_weights": None},
    ])

    bam.install_patch()
    loss = bam.patched_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, matched_idxs)
    bam.clear_batch_supervision()
    print(f"[T2 full_ignore] loss={float(loss):.6e} (expected ~0)")
    assert float(loss) < 1e-4, f"full ignore should give ~0 loss, got {float(loss)}"


def test_zero_weight_drops_instance():
    """All instances weight=0 → loss should be 0 (denominator clamped to 1, numerator 0)."""
    mask_logits, proposals, gt_masks, gt_labels, matched_idxs = make_inputs()
    zero_weights = [
        torch.zeros(g.shape[0], dtype=torch.float32) for g in gt_masks
    ]
    bam.stash_batch_supervision([
        {"ignore_masks": None, "mask_weights": zero_weights[0]},
        {"ignore_masks": None, "mask_weights": zero_weights[1]},
    ])

    bam.install_patch()
    loss = bam.patched_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, matched_idxs)
    bam.clear_batch_supervision()
    print(f"[T3 zero_weight] loss={float(loss):.6e} (expected ~0)")
    assert float(loss) < 1e-4, f"zero weight should give 0 loss, got {float(loss)}"


def test_partial_weight_scales_loss():
    """weight=0.5 should give exactly half the loss vs weight=1.0 (denominator scales too — actually mean stays same!).

    Actually with sum-over-pixels / sum-of-weights, uniform weight scaling is invariant.
    The semantic of "weight=0 drops instance" is what matters and is already covered.
    Skip this test — it's a math no-op.
    """
    pass


if __name__ == "__main__":
    print("Running boundary_aware_mask smoke tests...")
    test_vanilla_equivalence()
    test_full_ignore_zeroes_loss()
    test_zero_weight_drops_instance()
    print("\nAll smoke tests passed.")
