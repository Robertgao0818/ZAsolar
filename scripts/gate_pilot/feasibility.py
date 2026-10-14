"""Pre-registered step-0 stop rule.

The plan stops the pilot when ``0.99 * P_pre > 0.20`` on CPT1238.
``P_pre`` is the share of chips whose valid footprint has intersection
area > 0 with any pre-filter polygon. That product is the lowest pass
rate that can still recall 99% of those chips, and only if the gate
passes no other chip.

The integer reading (how many whole chips G2 actually requires) is
reported beside it. It is not a second stop rule.
"""

from __future__ import annotations

import math


def g2_minimum_pass(
    n_chips: int,
    n_positive_chips: int,
    *,
    recall: float = 0.99,
    pass_limit: float = 0.20,
) -> dict:
    if n_chips <= 0:
        raise ValueError("n_chips must be positive")
    if not 0 <= n_positive_chips <= n_chips:
        raise ValueError("n_positive_chips out of range")
    p_positive = n_positive_chips / n_chips
    continuous = recall * p_positive
    if n_positive_chips == 0:
        k_min = 0
    elif recall == 0.99:
        # Smallest integer k with k / n_positive >= 99/100.
        k_min = (99 * n_positive_chips + 99) // 100
    else:
        k_min = math.ceil(recall * n_positive_chips - 1e-12)
    integer_rate = k_min / n_chips
    return {
        "n_chips": n_chips,
        "n_positive_chips": n_positive_chips,
        "p_positive": p_positive,
        "continuous_min_pass": continuous,
        "integer_k_min": k_min,
        "integer_min_pass": integer_rate,
        "pass_limit": pass_limit,
        "stop": continuous > pass_limit,
        "integer_exceeds_limit": integer_rate > pass_limit,
    }
