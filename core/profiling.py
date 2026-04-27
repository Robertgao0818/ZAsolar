"""Lightweight stage profiler for training + inference loops.

Goal: quickly answer "is GPU starving on CPU work?" without pulling in
torch.profiler. Tracks wall-clock time spent inside labeled stages and
(optionally) GPU kernel time via CUDA events on the current stream.

Usage:
    prof = StageProfiler(cuda=True)
    with prof("data"): batch = next(it)
    with prof("gpu", cuda=True): loss = model(batch)
    print(prof.summary())
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


class StageProfiler:
    """Aggregate per-stage timings across many iterations.

    `cuda=True` at construction enables CUDA event timing for stages that
    opt in with `cuda=True` in their `with` call. Wall-clock totals are
    always recorded; the CUDA column is only present where requested.
    """

    def __init__(self, cuda: bool = False) -> None:
        self.cuda_enabled = bool(cuda and torch.cuda.is_available())
        self._wall: dict[str, float] = {}
        self._gpu_ms: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextmanager
    def __call__(self, name: str, cuda: bool = False) -> Iterator[None]:
        use_cuda = cuda and self.cuda_enabled
        if use_cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._wall[name] = self._wall.get(name, 0.0) + dt
            self._counts[name] = self._counts.get(name, 0) + 1
            if use_cuda:
                end_ev.record()
                torch.cuda.synchronize()
                gpu_ms = start_ev.elapsed_time(end_ev)
                self._gpu_ms[name] = self._gpu_ms.get(name, 0.0) + gpu_ms

    def reset(self) -> None:
        self._wall.clear()
        self._gpu_ms.clear()
        self._counts.clear()

    def summary(self, header: str = "") -> str:
        if not self._wall:
            return "[PROF] no stages recorded"
        total = sum(self._wall.values())
        lines = []
        if header:
            lines.append(f"[PROF] {header}")
        lines.append(
            f"[PROF] {'stage':<24} {'wall_s':>10} {'pct':>6} {'gpu_ms':>10} {'n':>6}"
        )
        for name, wall in sorted(self._wall.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * wall / total if total > 0 else 0.0
            gpu = self._gpu_ms.get(name)
            gpu_s = f"{gpu:10.1f}" if gpu is not None else f"{'-':>10}"
            n = self._counts.get(name, 0)
            lines.append(f"[PROF] {name:<24} {wall:10.2f} {pct:5.1f}% {gpu_s} {n:6d}")
        lines.append(f"[PROF] {'TOTAL':<24} {total:10.2f}")
        return "\n".join(lines)
