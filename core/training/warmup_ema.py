"""Linear LR warmup + weight EMA for ``train.py`` (C-2 recipe lever).

Two opt-in training recipe levers, both **off by default** so that with the
flags unset ``train.py`` is byte-for-byte equivalent to its prior behavior:

1. **Linear LR warmup** (``build_warmup_cosine_scheduler``): ramp the LR
   linearly from ``warmup_start_factor * base_lr`` up to ``base_lr`` over the
   first ``warmup_iters`` optimizer steps, then hand off to the caller's
   per-step ``CosineAnnealingLR``. Implemented by composing
   ``torch.optim.lr_scheduler.LinearLR`` + the cosine scheduler under a
   ``SequentialLR`` so the *combined* schedule still steps once per iteration,
   exactly like the bare cosine scheduler ``train.py`` already drives. When
   ``warmup_iters <= 0`` this returns the cosine scheduler **unchanged** (same
   object), guaranteeing no behavior change.

2. **Weight EMA** (``ModelEMA``): maintain a shadow copy of the model weights
   updated as ``ema = decay * ema + (1 - decay) * param`` after every optimizer
   step. The EMA weights are used as an *additional* checkpoint-selection
   candidate — the training loop evaluates both the raw (online) weights and the
   EMA weights, and writes a separate ``*_ema.pth`` family of checkpoints. This
   preserves attribution: a single job emits both raw-best and EMA-best so the
   downstream selector can compare the two recipes on the same dataset + seed.

This module is intentionally torch-only (no CUDA assertion, no train.py import)
so it is unit-testable on a CPU box. ``train.py`` asserts CUDA at import time and
cannot be loaded in CI; keeping the math here keeps it testable.

Design note — Stage-2 cosine cold start (train.py Stage 2):
    train.py runs a two-stage schedule. Stage 1 (heads-only / freeze-mask-head)
    uses a *flat* SGD LR with no per-step scheduler. Stage 2 hot-swaps to a
    fresh optimizer and a ``CosineAnnealingLR`` that cold-starts from the full
    ``lr2`` on its first step (train.py ~L1413-1423). The warmup added here
    wraps **only the Stage-2 cosine scheduler**: it inserts a linear ramp at the
    very start of Stage 2, smoothing that cold start. Stage 1 keeps its flat LR
    (warmup there is unnecessary — heads-only/frozen-backbone warm-up is already
    the gentle phase, and adding a second warmup there would double-count). On a
    Stage-2 resume the warmup has already been consumed (it lives in the first
    ``warmup_iters`` steps), so the resume fast-forward over the SequentialLR
    advances past it correctly — same ``.step()``-count contract as the bare
    cosine. (Note: the bare ``scheduler.step()`` fast-forward loop in train.py
    emits PyTorch's benign "lr_scheduler.step() before optimizer.step()" warning
    — once for bare cosine, twice for the SequentialLR warmup path — because no
    ``optimizer.step()`` runs in that loop; the LR value is still correct.
    train.py suppresses only that specific message around the loop.)

Deliberately NOT implemented (per the C-2 spec, do not add):
    - SWA (stochastic weight averaging): rejected — epoch averaging over ~10k
      noisy SAM GT chips has a documented bulk-overshoot countersignal (train20).
    - multi-scale ``min_size`` list: torchvision eval takes ``min_size[-1]``, so
      a list like ``[640, 800, 960]`` would silently change inference resolution
      to 960 and break all historical comparability. Inference ``min_size`` must
      stay 800.
"""
from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_iters: int,
    warmup_start_factor: float = 0.01,
    eta_min: float = 1e-6,
    cosine_scheduler: LRScheduler | None = None,
) -> LRScheduler:
    """Return a per-step LR scheduler: linear warmup → cosine anneal.

    Args:
        optimizer: the Stage-2 optimizer.
        total_steps: total optimizer steps across Stage 2
            (``epochs2 * len(train_loader)``) — the same value train.py uses for
            the bare cosine ``T_max``.
        warmup_iters: number of warmup steps. ``<= 0`` disables warmup and this
            function returns the cosine scheduler unchanged (byte-for-byte legacy
            behavior). Typical: 500–1000.
        warmup_start_factor: LR at warmup step 0 as a fraction of ``base_lr``
            (e.g. 0.01 ⇒ start at 1% of base LR). Must be in (0, 1].
        eta_min: cosine floor LR (mirrors train.py's bare cosine).
        cosine_scheduler: optional pre-built cosine scheduler to compose with.
            If ``None``, a ``CosineAnnealingLR(T_max=total_steps - warmup_iters,
            eta_min=eta_min)`` is built. Passing one lets the caller keep full
            control of the cosine config.

    Notes:
        The cosine phase is annealed over ``total_steps - warmup_iters`` steps so
        that the *total* Stage-2 budget (``total_steps``) is unchanged: the
        warmup consumes the first ``warmup_iters`` steps, cosine the rest, and
        cosine still reaches ``eta_min`` exactly at the final step. This is the
        standard "warmup eats into, not adds to, the budget" convention and keeps
        the schedule length identical to the legacy bare-cosine run.
    """
    if warmup_iters <= 0:
        # Disabled: return the (possibly caller-supplied) plain cosine. Build
        # one over the full budget if none was supplied so the returned object
        # is identical to train.py's legacy ``CosineAnnealingLR(T_max=total_steps)``.
        if cosine_scheduler is not None:
            return cosine_scheduler
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    if not (0.0 < warmup_start_factor <= 1.0):
        raise ValueError(
            f"warmup_start_factor must be in (0, 1], got {warmup_start_factor}"
        )
    if warmup_iters >= total_steps:
        raise ValueError(
            f"warmup_iters ({warmup_iters}) must be < total_steps ({total_steps}); "
            "warmup must leave room for the cosine phase"
        )

    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    if cosine_scheduler is None:
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_iters, eta_min=eta_min
        )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine_scheduler],
        milestones=[warmup_iters],
    )


class ModelEMA:
    """Exponential moving average of model weights (shadow copy).

    Maintains a CPU- or GPU-resident shadow of the model's parameters AND
    buffers (BatchNorm running stats etc.), updated after each optimizer step:

        ema = decay * ema + (1 - decay) * online

    The shadow is a ``state_dict``-shaped dict of detached tensors so it can be
    saved directly with ``torch.save(ema.state_dict(), path)`` and loaded into a
    fresh model with ``model.load_state_dict(...)``. Integer / non-floating
    buffers (e.g. ``num_batches_tracked``) are copied verbatim rather than
    blended (averaging an int counter is meaningless).

    Usage:
        ema = ModelEMA(model, decay=0.999)
        for batch in loader:
            ... optimizer.step() ...
            ema.update(model)
        torch.save(ema.state_dict(), "best_model_ema.pth")

    The EMA does not participate in gradient computation; ``update`` runs under
    ``torch.no_grad``.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        # Deep-copy the current state_dict, detached. Keep tensors on their
        # original device (matches the model) so the elementwise update is cheap.
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Blend the model's current weights into the shadow."""
        d = self.decay
        msd = model.state_dict()
        for k, shadow_v in self.shadow.items():
            online_v = msd[k]
            if shadow_v.is_floating_point():
                # ema = d*ema + (1-d)*online, computed in-place on the shadow.
                shadow_v.mul_(d).add_(online_v.detach().to(shadow_v.device), alpha=1.0 - d)
            else:
                # Integer buffers (num_batches_tracked, etc.): track the online
                # value verbatim. Blending an int counter is meaningless.
                shadow_v.copy_(online_v.detach().to(shadow_v.device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shadow as a model-loadable state_dict (detached clones)."""
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore the EMA shadow from a saved ``state_dict`` (in place).

        Copies each tensor into the existing shadow entry so the shadow keeps its
        current device/dtype. Used on ``--resume`` so a long EMA run that is
        interrupted does not lose its accumulated momentum (the shadow is
        re-seeded from the checkpoint instead of from the resumed online weights).
        Raises ``KeyError`` if the saved keys do not match the shadow keys.
        """
        missing = set(self.shadow) - set(state_dict)
        if missing:
            raise KeyError(
                f"EMA state_dict is missing {len(missing)} key(s) present in the "
                f"shadow (e.g. {sorted(missing)[:3]})"
            )
        for k, shadow_v in self.shadow.items():
            shadow_v.copy_(state_dict[k].detach().to(shadow_v.device))

    def copy_to(self, model: torch.nn.Module) -> None:
        """Load the EMA shadow into ``model`` in place (for in-loop eval)."""
        model.load_state_dict(self.state_dict())

    def to(self, device) -> "ModelEMA":
        """Move the shadow tensors to ``device`` (in place). Returns self."""
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].to(device)
        return self


def warmup_lr_curve(
    base_lr: float,
    warmup_iters: int,
    warmup_start_factor: float,
    n_steps: int,
) -> list[float]:
    """Pure-python reference of the *warmup phase* LR curve (no optimizer).

    Returns the LR at each of the first ``n_steps`` steps for a ``LinearLR``
    warmup with ``start_factor`` → ``end_factor=1.0`` over ``warmup_iters``.
    Used only by tests to assert the schedule shape without instantiating a
    model/optimizer. Mirrors torch's ``LinearLR`` semantics: at step ``i`` the
    factor is ``start + (1 - start) * min(i, warmup_iters) / warmup_iters``.
    """
    out: list[float] = []
    start = warmup_start_factor
    for i in range(n_steps):
        frac = min(i, warmup_iters) / warmup_iters if warmup_iters > 0 else 1.0
        factor = start + (1.0 - start) * frac
        out.append(base_lr * factor)
    return out
