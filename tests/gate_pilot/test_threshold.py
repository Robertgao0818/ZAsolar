"""Tau is the highest cut that still meets both recall floors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_pilot.threshold import fold_warning, minimum_kept, select_tau


def test_minimum_kept_is_ceil_of_99_5_percent():
    assert minimum_kept(0) == 0
    assert minimum_kept(200) == 199
    assert minimum_kept(1) == 1
    assert minimum_kept(1000) == 995


def test_tau_is_capped_by_the_weakest_post_positive_chip():
    # 200 pre chips: 199 at 0.9 and one at 0.01. Post chips all sit at 0.5.
    y_pre = np.array([True] * 200 + [False] * 50)
    y_post = np.array([False] * 200 + [True] * 40 + [False] * 10)
    score = np.array([0.01] + [0.9] * 199 + [0.5] * 40 + [0.0] * 10)
    chosen = select_tau(y_pre, y_post, score)
    assert chosen["tau"] == 0.5
    assert chosen["post_recall"] == 1.0
    assert chosen["pre_recall"] == 0.995


def test_tau_drops_the_allowed_pre_filter_tail_when_post_chips_are_easy():
    y_pre = np.ones(200, dtype=bool)
    # The low score is pre-filter only. Every post-filter chip scores 0.9.
    y_post = np.array([False] + [True] * 10 + [False] * 189)
    score = np.array([0.01] + [0.9] * 199)
    # The ten post chips are the high scores, so the post cap is 0.9.
    # Pre recall of 99.5% allows dropping the single 0.01.
    chosen = select_tau(y_pre, y_post, score)
    assert chosen["tau"] == 0.9
    assert chosen["post_recall"] == 1.0
    assert abs(chosen["pre_recall"] - 0.995) < 1e-12


def test_fold_warning_fires_when_full_recall_lets_half_the_chips_through():
    y_post = np.array([True, True, False, False])
    quiet = fold_warning(y_post, np.array([0.9, 0.8, 0.1, 0.0]))
    assert quiet["pass_rate"] == 0.5
    assert not quiet["above_limit"]
    loud = fold_warning(y_post, np.array([0.2, 0.2, 0.3, 0.4]))
    assert loud["pass_rate"] == 1.0
    assert loud["above_limit"]
