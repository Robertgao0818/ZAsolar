"""Out-of-fold threshold for the presence gate.

A chip passes when its score is greater than or equal to tau. Tau is the
largest value that still keeps every post-filter positive chip and at least
99.5% of the pre-filter positive chips. Scores come only from chips the
model did not train on.
"""

from __future__ import annotations

import numpy as np


def minimum_kept(n_positive: int, recall: float = 0.995) -> int:
    """Smallest integer k with k / n >= recall. 0.995 is 995/1000."""
    if n_positive < 0:
        raise ValueError("n_positive must be non-negative")
    if n_positive == 0:
        return 0
    if recall == 0.995:
        return (995 * n_positive + 999) // 1000
    return int(np.ceil(recall * n_positive - 1e-12))


def select_tau(y_pre: np.ndarray, y_post: np.ndarray, score: np.ndarray) -> dict:
    """Return the locked tau and the recalls it actually produces.

    ``y_post`` must be contained in the rows where we also know ``y_pre``.
    A post-positive chip with no score at or above tau would make the 100%
    recall requirement impossible above that chip's score, so tau cannot
    exceed the lowest post-positive score.
    """
    y_pre = np.asarray(y_pre, dtype=bool)
    y_post = np.asarray(y_post, dtype=bool)
    score = np.asarray(score, dtype=float)
    if not (len(y_pre) == len(y_post) == len(score)):
        raise ValueError("y_pre, y_post, and score must have the same length")
    if int(y_post.sum()) == 0:
        raise ValueError("no post-filter positive chips to anchor tau")
    if np.isnan(score).any():
        raise ValueError("scores contain NaN")

    tau_post = float(score[y_post].min())
    n_pre = int(y_pre.sum())
    k_min = minimum_kept(n_pre)
    if n_pre == 0 or k_min == 0:
        tau_pre = float("inf")
    else:
        ordered = np.sort(score[y_pre])
        tau_pre = float(ordered[n_pre - k_min])
    tau = min(tau_post, tau_pre)

    pre_recall = float((score[y_pre] >= tau).mean()) if n_pre else 1.0
    post_recall = float((score[y_post] >= tau).mean())
    pass_rate = float((score >= tau).mean())
    if post_recall < 1.0 - 1e-12:
        raise RuntimeError(f"tau={tau} missed a post-positive chip")
    if n_pre and pre_recall + 1e-12 < 0.995:
        raise RuntimeError(f"tau={tau} pre-filter recall {pre_recall} is below 99.5%")
    return {
        "tau": tau,
        "tau_post_cap": tau_post,
        "tau_pre_cap": tau_pre if np.isfinite(tau_pre) else None,
        "n_chips": int(len(score)),
        "n_pre": n_pre,
        "n_post": int(y_post.sum()),
        "pre_keep_min": k_min,
        "pre_recall": pre_recall,
        "post_recall": post_recall,
        "pass_rate": pass_rate,
    }


def fold_warning(y_post: np.ndarray, score: np.ndarray, limit: float = 0.50) -> dict:
    """Pass rate at the highest tau that still recalls every post-positive chip."""
    y_post = np.asarray(y_post, dtype=bool)
    score = np.asarray(score, dtype=float)
    if int(y_post.sum()) == 0:
        raise ValueError("fold has no post-positive chip")
    tau = float(score[y_post].min())
    pass_rate = float((score >= tau).mean())
    return {
        "tau_at_full_post_recall": tau,
        "pass_rate": pass_rate,
        "above_limit": pass_rate > limit,
    }
