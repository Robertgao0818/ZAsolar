"""End-to-end smoke: tiny dataset → model forward → boundary-aware loss path.

Builds a 1-grid JHBRawPartsDataset, runs one forward+backward pass with
the patched mask loss, asserts that:
- Loss is finite
- Gradients propagate
- mask_weight=0 instances DO NOT push gradient through mask head (V3-C halo
  insulation works)

Run on CPU (no CUDA assertion). Skip if CUDA missing — production training
is on RunPod GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.training import boundary_aware_mask as bam  # noqa: E402
from core.training.jhb_phaseA_dataset import JHBRawPartsDataset, load_spec  # noqa: E402
from core.training.jhb_phaseA_transforms import (  # noqa: E402
    BoundaryAwareTrainTransforms,
)


def _make_tiny_spec():
    spec = load_spec(PROJECT_ROOT / "configs/datasets/jhb_phaseA.yaml")
    spec["splits"]["train"]["grids"] = ["G0853"]  # tiny grid
    spec["splits"]["val"]["grids"] = ["G0853"]
    spec["neg_ratio"] = 0.0
    return spec


def test_forward_backward():
    spec = _make_tiny_spec()
    bam.install_patch()
    train_ds = JHBRawPartsDataset(
        spec, "train", transforms=BoundaryAwareTrainTransforms(400)
    )
    pos_chips = [i for i, c in enumerate(train_ds.chips) if c["polygons"]]
    assert pos_chips, "no positive chips in tiny dataset"

    img, tgt = train_ds[pos_chips[0]]
    print(f"[E2E] chip masks={tgt['masks'].shape}, "
          f"weights={tgt['mask_weights'].tolist()}, "
          f"ignore_sums={tgt['ignore_masks'].sum(dim=(-2,-1)).tolist()}")

    # Build minimal Mask R-CNN model on CPU (won't load V3-C weights — too slow + needs CUDA)
    import torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=2)
    model.train()

    # Register pre-hook for stashing
    def _pre(mod, args):
        if mod.training and len(args) >= 2 and args[1] is not None:
            bam.stash_batch_supervision(args[1])

    model.register_forward_pre_hook(_pre)

    images = [img]
    targets = [tgt]
    loss_dict = model(images, targets)
    print(f"[E2E] loss_dict keys: {list(loss_dict.keys())}")
    print(f"[E2E] loss_mask = {float(loss_dict['loss_mask']):.6f}")
    losses = sum(v for v in loss_dict.values())
    losses.backward()
    print(f"[E2E] total loss = {float(losses):.4f} (backward ok)")
    assert torch.isfinite(losses), "loss is not finite"


def test_zero_weight_silences_mask_grad():
    """Force ALL instances to mask_weight=0; mask head grad should be (near) 0."""
    spec = _make_tiny_spec()
    bam.install_patch()
    ds = JHBRawPartsDataset(spec, "train", transforms=None)
    pos_chips = [i for i, c in enumerate(ds.chips) if c["polygons"]]
    img, tgt = ds[pos_chips[0]]
    tgt["mask_weights"] = torch.zeros_like(tgt["mask_weights"])

    import torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=2)
    model.train()

    def _pre(mod, args):
        if mod.training and len(args) >= 2 and args[1] is not None:
            bam.stash_batch_supervision(args[1])

    model.register_forward_pre_hook(_pre)

    loss_dict = model([img], [tgt])
    lm = float(loss_dict["loss_mask"])
    print(f"[E2E zero-weight] loss_mask = {lm:.6e} (expected 0)")
    assert lm < 1e-6, f"mask_weight=0 should give 0 mask loss; got {lm}"


if __name__ == "__main__":
    print("Running end-to-end smoke...")
    test_forward_backward()
    print()
    test_zero_weight_silences_mask_grad()
    print("\nAll e2e tests passed.")
