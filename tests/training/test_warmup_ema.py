"""CPU tests for the C-2 warmup + EMA recipe levers.

Covers (all on CPU, no train.py import — train.py asserts CUDA at import):

- warmup schedule SHAPE: linear ramp from start_factor*base up to base over
  warmup_iters, then cosine anneal to eta_min at the final step.
- warmup DISABLED == bare CosineAnnealingLR step-for-step (byte-for-byte legacy).
- warmup budget: total step count unchanged (warmup eats into, not adds to).
- EMA update MATH: shadow = decay*shadow + (1-decay)*online (floats), verbatim
  copy for integer buffers (num_batches_tracked).
- EMA RESUME persistence: load_state_dict round-trips the saved shadow (so a
  resumed --ema run recovers its momentum instead of re-seeding from online
  weights), copies in place, and rejects key mismatches.
- dual-checkpoint SELECTION plumbing: a raw-best vs EMA-best picker driven off
  two independent metric streams writes the right files independently.
- input validation guards.

These mirror the wiring in train.py Stage 2 (build_warmup_cosine_scheduler +
ModelEMA) without invoking the CUDA-only training entrypoint.
"""
from __future__ import annotations

import torch
import pytest
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.training.warmup_ema import (
    ModelEMA,
    build_warmup_cosine_scheduler,
    warmup_lr_curve,
)


def _run_schedule(scheduler_factory, base_lr, total_steps):
    """Drive a per-step scheduler for total_steps, return the LR seen each step."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=base_lr)
    sched = scheduler_factory(opt)
    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


# ── Warmup schedule shape ────────────────────────────────────────────────
def test_warmup_ramps_linearly_to_base_then_anneals():
    base_lr = 0.1
    warmup_iters = 5
    total_steps = 20
    start_factor = 0.01
    lrs = _run_schedule(
        lambda o: build_warmup_cosine_scheduler(
            o, total_steps=total_steps, warmup_iters=warmup_iters,
            warmup_start_factor=start_factor, eta_min=1e-6,
        ),
        base_lr, total_steps,
    )
    # step 0 starts at start_factor * base_lr
    assert lrs[0] == pytest.approx(base_lr * start_factor, rel=1e-5)
    # ramp is strictly increasing through the warmup window
    for i in range(warmup_iters):
        assert lrs[i] <= lrs[i + 1] + 1e-9
    # peak (== base_lr) is reached exactly at the end of warmup
    assert lrs[warmup_iters] == pytest.approx(base_lr, rel=1e-5)
    # after the peak the schedule anneals (decreasing) toward eta_min
    for i in range(warmup_iters, total_steps - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-9
    # cosine reaches the floor by the last step
    assert lrs[-1] < base_lr * 0.1


def test_warmup_curve_reference_matches_torch_linear_phase():
    """The pure-python warmup_lr_curve mirrors torch LinearLR during warmup."""
    base_lr = 0.2
    warmup_iters = 4
    start_factor = 0.05
    # torch side: only look at the warmup window (first warmup_iters+1 steps)
    torch_lrs = _run_schedule(
        lambda o: build_warmup_cosine_scheduler(
            o, total_steps=50, warmup_iters=warmup_iters,
            warmup_start_factor=start_factor, eta_min=1e-6,
        ),
        base_lr, warmup_iters + 1,
    )
    ref = warmup_lr_curve(base_lr, warmup_iters, start_factor, warmup_iters + 1)
    for a, b in zip(torch_lrs, ref):
        assert a == pytest.approx(b, rel=1e-5)


# ── Disabled path == legacy bare cosine ──────────────────────────────────
def test_warmup_disabled_equals_bare_cosine():
    base_lr = 0.5
    total_steps = 16
    bare = _run_schedule(
        lambda o: CosineAnnealingLR(o, T_max=total_steps, eta_min=1e-6),
        base_lr, total_steps,
    )
    disabled = _run_schedule(
        lambda o: build_warmup_cosine_scheduler(
            o, total_steps=total_steps, warmup_iters=0, eta_min=1e-6,
        ),
        base_lr, total_steps,
    )
    assert len(bare) == len(disabled)
    for a, b in zip(bare, disabled):
        assert a == pytest.approx(b, abs=1e-12)


def test_warmup_disabled_returns_plain_cosine_object():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=0.1)
    sched = build_warmup_cosine_scheduler(
        opt, total_steps=10, warmup_iters=0, eta_min=1e-6,
    )
    assert isinstance(sched, CosineAnnealingLR)


def test_warmup_budget_is_total_steps_not_extra():
    """With warmup on, the schedule still spans exactly total_steps and reaches
    the cosine floor at the last step (warmup eats into, not adds to, budget)."""
    base_lr = 0.1
    total_steps = 30
    warmup_iters = 8
    lrs = _run_schedule(
        lambda o: build_warmup_cosine_scheduler(
            o, total_steps=total_steps, warmup_iters=warmup_iters, eta_min=1e-6,
        ),
        base_lr, total_steps,
    )
    assert len(lrs) == total_steps
    # last step is at/near the eta_min floor (cosine completed within the budget)
    assert lrs[-1] < base_lr * 0.05


# ── Validation guards ────────────────────────────────────────────────────
def test_warmup_start_factor_must_be_in_range():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=0.1)
    with pytest.raises(ValueError):
        build_warmup_cosine_scheduler(
            opt, total_steps=100, warmup_iters=10, warmup_start_factor=0.0,
        )
    with pytest.raises(ValueError):
        build_warmup_cosine_scheduler(
            opt, total_steps=100, warmup_iters=10, warmup_start_factor=1.5,
        )


def test_warmup_iters_must_be_less_than_total():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=0.1)
    with pytest.raises(ValueError):
        build_warmup_cosine_scheduler(
            opt, total_steps=10, warmup_iters=10,
        )


# ── EMA math ─────────────────────────────────────────────────────────────
def test_ema_blends_floats():
    m = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for pr in m.parameters():
            pr.fill_(0.0)
    ema = ModelEMA(m, decay=0.9)
    # online -> 1.0, one update -> shadow == (1-0.9)*1.0 == 0.1
    with torch.no_grad():
        for pr in m.parameters():
            pr.fill_(1.0)
    ema.update(m)
    assert float(ema.shadow["weight"].flatten()[0]) == pytest.approx(0.1, abs=1e-6)
    # second update: 0.9*0.1 + 0.1*1.0 == 0.19
    ema.update(m)
    assert float(ema.shadow["weight"].flatten()[0]) == pytest.approx(0.19, abs=1e-6)


def test_ema_tracks_integer_buffers_verbatim():
    """num_batches_tracked is an int buffer: copy verbatim, never blend."""
    bn = torch.nn.BatchNorm1d(4)
    # force a known integer buffer state
    bn.num_batches_tracked.fill_(0)
    ema = ModelEMA(bn, decay=0.99)
    bn.num_batches_tracked.fill_(7)
    ema.update(bn)
    assert int(ema.shadow["num_batches_tracked"]) == 7  # verbatim, not 0.07


def test_ema_state_dict_loads_into_fresh_model():
    m = torch.nn.Linear(3, 2)
    ema = ModelEMA(m, decay=0.5)
    with torch.no_grad():
        for pr in m.parameters():
            pr.fill_(2.0)
    ema.update(m)  # shadow moves halfway toward 2.0 from init
    fresh = torch.nn.Linear(3, 2)
    fresh.load_state_dict(ema.state_dict())  # must be load-compatible
    for k, v in fresh.state_dict().items():
        assert torch.allclose(v, ema.shadow[k])


def test_ema_decay_validation():
    m = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError):
        ModelEMA(m, decay=0.0)
    with pytest.raises(ValueError):
        ModelEMA(m, decay=1.0)


def test_ema_load_state_dict_round_trips_shadow():
    """save_checkpoint persists ema.state_dict(); on --resume a fresh ModelEMA
    (seeded from the resumed online weights) must be re-seeded from the saved
    shadow via load_state_dict, recovering the accumulated momentum exactly."""
    m = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for pr in m.parameters():
            pr.fill_(0.0)
    ema = ModelEMA(m, decay=0.9)
    with torch.no_grad():
        for pr in m.parameters():
            pr.fill_(1.0)
    ema.update(m)  # shadow now partway toward 1.0
    saved = ema.state_dict()

    # Simulate resume: model loaded from a checkpoint with *online* weights at
    # 1.0, fresh EMA re-seeds from those (== 1.0 everywhere, momentum lost) ...
    resumed_model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for pr in resumed_model.parameters():
            pr.fill_(1.0)
    resumed_ema = ModelEMA(resumed_model, decay=0.9)
    assert float(resumed_ema.shadow["weight"].flatten()[0]) == pytest.approx(1.0)
    # ... then restore the saved shadow → momentum recovered (== pre-interrupt).
    resumed_ema.load_state_dict(saved)
    for k in saved:
        assert torch.allclose(resumed_ema.shadow[k], saved[k])


def test_ema_load_state_dict_preserves_device_and_is_in_place():
    """load_state_dict copies into the existing shadow tensors in place (keeps
    their device/identity), it does not rebind to the saved dict's tensors."""
    m = torch.nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.5)
    target_tensors = {k: v for k, v in ema.shadow.items()}  # identity snapshot
    saved = {k: torch.full_like(v, 3.0) for k, v in ema.shadow.items()}
    ema.load_state_dict(saved)
    for k, t in target_tensors.items():
        # same tensor object (in-place copy_), with the new values
        assert ema.shadow[k] is t
        assert torch.allclose(ema.shadow[k], torch.full_like(t, 3.0))


def test_ema_load_state_dict_rejects_missing_keys():
    m = torch.nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.5)
    partial = {"weight": ema.shadow["weight"].clone()}  # missing "bias"
    with pytest.raises(KeyError):
        ema.load_state_dict(partial)


def test_ema_copy_to_does_not_mutate_shadow():
    m = torch.nn.Linear(3, 2)
    ema = ModelEMA(m, decay=0.9)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    # mutate online, copy ema into a different model, shadow must be unchanged
    other = torch.nn.Linear(3, 2)
    ema.copy_to(other)
    for k in before:
        assert torch.allclose(ema.shadow[k], before[k])


# ── Dual-checkpoint selection plumbing ───────────────────────────────────
def test_dual_checkpoint_selection_independent_streams(tmp_path):
    """Reproduce train.py's raw-best vs EMA-best selection: two independent
    metric streams must each write/update their own checkpoint file only on
    improvement. The raw selection path is unchanged by EMA being on."""
    raw_path = tmp_path / "best_model.pth"
    ema_path = tmp_path / "best_model_ema.pth"

    raw_model = torch.nn.Linear(2, 2)
    ema = ModelEMA(raw_model, decay=0.9)

    best_raw = 0.0
    best_ema = {"f1": 0.0}
    writes = {"raw": 0, "ema": 0}

    def select(raw_f1, ema_f1):
        nonlocal best_raw
        # raw side (unchanged legacy selection)
        if raw_f1 > best_raw:
            best_raw = raw_f1
            torch.save(raw_model.state_dict(), raw_path)
            writes["raw"] += 1
        # ema side (additive)
        if ema_f1 > best_ema["f1"]:
            best_ema["f1"] = ema_f1
            torch.save(ema.state_dict(), ema_path)
            writes["ema"] += 1

    # epoch 1: raw improves, ema improves → one write each
    select(0.50, 0.40)
    assert raw_path.exists() and ema_path.exists()
    assert writes == {"raw": 1, "ema": 1}

    # epoch 2: raw REGRESSES (no raw write), ema improves (ema write) — the two
    # selection streams are independent; EMA being on never perturbs raw.
    select(0.45, 0.55)
    assert writes == {"raw": 1, "ema": 2}
    assert best_raw == pytest.approx(0.50)
    assert best_ema["f1"] == pytest.approx(0.55)

    # epoch 3: ema also regresses → no writes at all
    select(0.40, 0.50)
    assert writes == {"raw": 1, "ema": 2}

    # the EMA checkpoint must be loadable into a fresh model (attribution use)
    fresh = torch.nn.Linear(2, 2)
    fresh.load_state_dict(torch.load(ema_path, weights_only=True))
