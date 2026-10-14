"""Locked empty-grid draw for the presence-gate pilot.

The rule is the one in the 2026-09-23 plan, written so a re-run cannot
silently change it:

1. Keep candidates whose pre-filter polygon count is at least 1.
2. Sort by pre-filter count ascending, breaking ties by grid id.
3. The upper half is the second half of that list (index ``n // 2``
   through the end). When ``n`` is odd the median count is in the upper
   half. Ties that sit on the cut stay where the grid id places them.
4. If that upper half has fewer than 2 grids, draw from every candidate
   that survived step 1 and record the fallback.
5. Draw with ``random.Random(20260923).sample`` on the lexicographically
   sorted id list. The first id is E_train, the second is E_test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class EmptyGridDraw:
    e_train: str
    e_test: str
    upper_half: tuple[str, ...]
    draw_population: tuple[str, ...]
    used_fallback: bool
    n_candidates_pre_ge_1: int
    cutoff_pre_count: int


def select_empty_grids(
    records: list[tuple[str, int]],
    *,
    seed: int = 20260923,
) -> EmptyGridDraw:
    """Draw E_train and E_test.

    ``records`` is ``(grid_id, pre_filter_polygon_count)`` for grids that
    are inside the census footprint and have zero rows in the delivered
    inventory. Counts below 1 are dropped before the half-split.
    """
    kept = [(grid_id, int(count)) for grid_id, count in records if int(count) >= 1]
    if len(kept) < 2:
        raise ValueError(
            f"need at least 2 grids with a pre-filter polygon, found {len(kept)}"
        )

    ordered = sorted(kept, key=lambda item: (item[1], item[0]))
    upper = ordered[len(ordered) // 2 :]
    used_fallback = len(upper) < 2
    population_rows = ordered if used_fallback else upper
    population = tuple(sorted(grid_id for grid_id, _ in population_rows))
    drawn = random.Random(seed).sample(list(population), 2)
    return EmptyGridDraw(
        e_train=drawn[0],
        e_test=drawn[1],
        upper_half=tuple(grid_id for grid_id, _ in upper),
        draw_population=population,
        used_fallback=used_fallback,
        n_candidates_pre_ge_1=len(kept),
        cutoff_pre_count=min(count for _, count in upper),
    )
