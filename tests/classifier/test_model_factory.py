"""Model factory coverage + checkpoint metadata roundtrip test.

Ensures every declared classifier backbone can be constructed, state-dict
save/load works through train_cls.py ↔ classify_predictions.py, and the
metadata written with the checkpoint is what classify_predictions.py reads
back when rebuilding the model.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.classifier import train_cls, classify_predictions  # noqa: E402


@pytest.mark.parametrize("arch", list(train_cls.SUPPORTED_ARCHS))
def test_train_cls_build_model_constructs(arch):
    model = train_cls.build_model(arch, pretrained=False)
    # Forward a dummy batch to confirm structure
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, train_cls.NUM_CLASSES), (
        f"{arch} forward shape mismatch: got {y.shape}"
    )


@pytest.mark.parametrize("arch", list(train_cls.SUPPORTED_ARCHS))
def test_classify_predictions_build_model_matches(arch):
    """Inference-side build_model should produce a state-dict-compatible
    model for every training-side backbone."""
    train_model = train_cls.build_model(arch, pretrained=False)
    infer_model = classify_predictions.build_model(arch, num_classes=2)
    missing, unexpected = infer_model.load_state_dict(
        train_model.state_dict(), strict=False
    )
    assert not missing, f"{arch}: missing keys when loading train→infer: {missing}"
    assert not unexpected, f"{arch}: unexpected keys when loading train→infer: {unexpected}"


@pytest.mark.parametrize("arch", list(train_cls.SUPPORTED_ARCHS))
def test_checkpoint_roundtrip_with_metadata(arch):
    """Save a checkpoint via train_cls._save_checkpoint, load it via
    classify_predictions.load_classifier, confirm arch is reconstructed."""
    model = train_cls.build_model(arch, pretrained=False)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    meta = {
        "arch": arch,
        "training_mode": "full_ft",
        "aug_profile": "current",
        "img_size": 224,
        "num_classes": 2,
        "class_names": list(train_cls.CLASS_NAMES),
        "preprocessing": {
            "resize": 224,
            "mean": train_cls.IMAGENET_MEAN,
            "std": train_cls.IMAGENET_STD,
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "cls.pth"
        train_cls._save_checkpoint(
            model, scaler, stage=1, epoch=0, best_balanced_acc=0.0,
            path=ckpt_path, meta=meta,
        )
        device = torch.device("cpu")
        restored, config = classify_predictions.load_classifier(ckpt_path, device)
        assert config["arch"] == arch
        assert config["training_mode"] == "full_ft"
        assert config["aug_profile"] == "current"
        assert config["img_size"] == 224
        # Forward to make sure the state-dict actually matched
        restored.eval()
        with torch.no_grad():
            y = restored(torch.zeros(1, 3, 224, 224))
        assert y.shape == (1, 2)


def test_aug_profile_current_and_flip_only_both_work():
    t_curr = train_cls.get_transforms(224, is_train=True, aug_profile="current")
    t_flip = train_cls.get_transforms(224, is_train=True, aug_profile="flip_only")
    # Both must be callable on a PIL-like tensor
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    out_curr = t_curr(img)
    out_flip = t_flip(img)
    assert out_curr.shape == (3, 224, 224)
    assert out_flip.shape == (3, 224, 224)


def test_aug_profile_unknown_raises():
    with pytest.raises(ValueError):
        train_cls.get_transforms(224, is_train=True, aug_profile="nonexistent")


def test_freeze_backbone_supports_all_archs():
    for arch in train_cls.SUPPORTED_ARCHS:
        model = train_cls.build_model(arch, pretrained=False)
        train_cls.freeze_backbone(model, arch)
        # At least one param should remain trainable (the head)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert trainable > 0, f"{arch}: no trainable params after freeze_backbone"
        assert frozen > 0, f"{arch}: nothing frozen — freeze_backbone did nothing"
