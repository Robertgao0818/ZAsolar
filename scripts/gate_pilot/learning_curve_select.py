"""Locked grid draw for the 2026-09-28 presence-gate learning curve.

The 2026-09-24 brief fixed the experiment (YOLO at 10 grids, YOLO at 30
grids, one ViT control on the same 30) but did not name the seed or the
density cutpoints. This file is that operationalization. Re-running it
must reproduce the same ids. Do not edit the constants to chase a
nicer-looking sample.

Rules, in order:

1. Density is the delivered installation count in
   ``ct_full_inventory_2026-06-21_merged.csv``. Quantiles below were
   measured on all 1,301 occupied grids with pandas ``interpolation=
   "linear"`` and then frozen as integers. A later inventory that no
   longer yields these quantiles must stop, not redraw.
2. Strata, mutually exclusive:
   - dense: count >= 257 (the frozen p90)
   - mid: 16 <= count <= 63 (the frozen p40 and p60)
   - low: 1 <= count <= 6 (the frozen p25)
   - empty: delivered count 0 and pre-filter polygon count >= 1
   Counts of 7–15 and 64–256 are the unsampled gap. They are not a
   holdout and not a training stratum in this round.
3. Seed is ``random.Random(20260928)``. Every draw samples a
   lexicographically sorted id list. Holdout draws happen first.
4. The six pilot grids cannot be a new holdout. CPT1238 and CPT2172
   also stay out of training. The other four pilot grids are forced
   into both the 30-grid set and the nested 10-grid subset.
5. Quotas for the 30, counting the forced grids: dense 8, mid 8,
   low 8, empty 6. The nested 10 keeps every forced grid and a prefix
   of each extra draw (dense 2, mid 1, low 2, empty 1, in sample order).
6. After each primary draw, the same stream draws 3 reserves from what
   remains in that stratum. Reserves are not fetched unless a primary
   grid's imagery fails, and a substitution has to be written down
   before any training.
7. Empty chips are not downsampled. The ratio is 1. The pilot failed
   by passing every chip, so this round keeps the negatives.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.gate_pilot.common import INVENTORY_CSV, OUT_ROOT

SEED = 20260928
LOW_MAX = 6
MID_MIN = 16
MID_MAX = 63
DENSE_MIN = 257
HOLDOUT_PER_STRATUM = 3
RESERVE_PER_DRAW = 3
EMPTY_CHIP_KEEP_RATIO = 1.0

PILOT_GRIDS = (
    "CPT2932",
    "CPT2373",
    "CPT2182",
    "CPT2330",
    "CPT1238",
    "CPT2172",
)
SEEN_HOLDOUT = ("CPT1238", "CPT2172")
FORCED_TRAIN = {
    "dense": ("CPT2932",),
    "mid": ("CPT2373", "CPT2182"),
    "low": (),
    "empty": ("CPT2330",),
}
EXTRA_TRAIN = {"dense": 7, "mid": 6, "low": 8, "empty": 5}
NESTED_PREFIX = {"dense": 2, "mid": 1, "low": 2, "empty": 1}
STRATA = ("dense", "mid", "low", "empty")

SELECTION_DIR = OUT_ROOT / "learning_curve_2026-09-28"
SELECTION_PATH = SELECTION_DIR / "selection.json"
EMPTY_PRE_COUNTS = OUT_ROOT / "step0" / "empty_grid_pre_counts.csv"

EXPECTED_QUANTILES = {0.25: 6.0, 0.40: 16.0, 0.60: 63.0, 0.90: 257.0}


@dataclass(frozen=True)
class LearningCurveDraw:
    """One reproducible draw. Lists keep sample order, not sorted order."""

    train_30: tuple[str, ...]
    train_10: tuple[str, ...]
    holdout: tuple[str, ...]
    holdout_by_stratum: dict[str, tuple[str, ...]]
    holdout_reserve_by_stratum: dict[str, tuple[str, ...]]
    train_extra_by_stratum: dict[str, tuple[str, ...]]
    train_reserve_by_stratum: dict[str, tuple[str, ...]]
    population: dict[str, int]

    def to_dict(self) -> dict:
        def pairs(mapping: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
            return {key: list(mapping[key]) for key in STRATA if key in mapping}

        return {
            "schema": "learning_curve_selection_v1",
            "seed": SEED,
            "empty_chip_keep_ratio": EMPTY_CHIP_KEEP_RATIO,
            "edges": {
                "low_max": LOW_MAX,
                "mid_min": MID_MIN,
                "mid_max": MID_MAX,
                "dense_min": DENSE_MIN,
            },
            "unsampled_delivered_counts": "7-15 and 64-256",
            "forced_train": [grid for name in STRATA for grid in FORCED_TRAIN[name]],
            "excluded_everywhere": list(SEEN_HOLDOUT),
            "train_30": list(self.train_30),
            "train_10": list(self.train_10),
            "holdout": list(self.holdout),
            "holdout_by_stratum": pairs(self.holdout_by_stratum),
            "holdout_reserve_by_stratum": pairs(self.holdout_reserve_by_stratum),
            "train_extra_by_stratum": pairs(self.train_extra_by_stratum),
            "train_reserve_by_stratum": pairs(self.train_reserve_by_stratum),
            "population": self.population,
        }


def stratum_name(delivered_count: int | None, pre_count: int | None) -> str | None:
    """Return the stratum, ``gap`` for an unsampled occupied grid, or None."""

    if delivered_count is not None and delivered_count > 0:
        if delivered_count >= DENSE_MIN:
            return "dense"
        if MID_MIN <= delivered_count <= MID_MAX:
            return "mid"
        if 1 <= delivered_count <= LOW_MAX:
            return "low"
        return "gap"
    if pre_count is not None and pre_count >= 1:
        return "empty"
    return None


def _draw_from(rng: random.Random, members: set[str], k: int, label: str) -> list[str]:
    pool = sorted(members)
    if len(pool) < k:
        raise ValueError(f"{label} needs {k} grids, pool has {len(pool)}")
    chosen = rng.sample(pool, k)
    members.difference_update(chosen)
    return chosen


def draw_learning_curve(
    delivered_counts: dict[str, int],
    empty_pre_counts: dict[str, int],
    *,
    seed: int = SEED,
) -> LearningCurveDraw:
    """Draw the locked 9 / 30 / 10 split.

    ``delivered_counts`` maps occupied grid id to a positive installation
    count. ``empty_pre_counts`` maps inventory-empty grid id to its
    pre-filter polygon count. The two maps must be disjoint.
    """

    overlap = set(delivered_counts) & set(empty_pre_counts)
    if overlap:
        raise ValueError(f"grid is both occupied and empty: {sorted(overlap)[:5]}")

    buckets: dict[str, set[str]] = {name: set() for name in (*STRATA, "gap")}
    for grid_id, count in delivered_counts.items():
        name = stratum_name(int(count), None)
        if name is None:
            raise ValueError(f"{grid_id} delivered count {count} is not occupied")
        buckets[name].add(grid_id)
    for grid_id, pre_count in empty_pre_counts.items():
        if int(pre_count) >= 1:
            buckets["empty"].add(grid_id)

    for name, grids in FORCED_TRAIN.items():
        for grid_id in grids:
            actual = None
            if grid_id in delivered_counts:
                actual = stratum_name(int(delivered_counts[grid_id]), None)
            elif grid_id in empty_pre_counts:
                actual = stratum_name(None, int(empty_pre_counts[grid_id]))
            if actual != name:
                raise ValueError(f"forced {grid_id} landed in {actual}, expected {name}")

    population = {
        "occupied": len(delivered_counts),
        "dense": len(buckets["dense"]),
        "mid": len(buckets["mid"]),
        "low": len(buckets["low"]),
        "gap": len(buckets["gap"]),
        "empty_pre_ge_1": len(buckets["empty"]),
        "empty_pre_0": sum(1 for value in empty_pre_counts.values() if int(value) < 1),
    }

    rng = random.Random(seed)
    holdout: dict[str, tuple[str, ...]] = {}
    holdout_reserve: dict[str, tuple[str, ...]] = {}
    blocked: set[str] = set()
    for name in ("dense", "mid", "empty"):
        pool = set(buckets[name]) - set(PILOT_GRIDS)
        primary = _draw_from(rng, pool, HOLDOUT_PER_STRATUM, f"holdout {name}")
        reserve = _draw_from(rng, pool, RESERVE_PER_DRAW, f"holdout reserve {name}")
        holdout[name] = tuple(primary)
        holdout_reserve[name] = tuple(reserve)
        blocked.update(primary)
        blocked.update(reserve)
    holdout["low"] = ()
    holdout_reserve["low"] = ()

    train_extra: dict[str, tuple[str, ...]] = {}
    train_reserve: dict[str, tuple[str, ...]] = {}
    for name in STRATA:
        pool = set(buckets[name]) - set(PILOT_GRIDS) - blocked
        extra = _draw_from(rng, pool, EXTRA_TRAIN[name], f"train {name}")
        reserve = _draw_from(rng, pool, RESERVE_PER_DRAW, f"train reserve {name}")
        train_extra[name] = tuple(extra)
        train_reserve[name] = tuple(reserve)

    train_30 = (
        FORCED_TRAIN["dense"] + train_extra["dense"]
        + FORCED_TRAIN["mid"] + train_extra["mid"]
        + train_extra["low"]
        + FORCED_TRAIN["empty"] + train_extra["empty"]
    )
    train_10 = (
        FORCED_TRAIN["dense"] + train_extra["dense"][: NESTED_PREFIX["dense"]]
        + FORCED_TRAIN["mid"] + train_extra["mid"][: NESTED_PREFIX["mid"]]
        + train_extra["low"][: NESTED_PREFIX["low"]]
        + FORCED_TRAIN["empty"] + train_extra["empty"][: NESTED_PREFIX["empty"]]
    )
    holdout_ids = holdout["dense"] + holdout["mid"] + holdout["empty"]
    _assert_partition(train_30, train_10, holdout_ids, holdout_reserve, train_reserve)
    return LearningCurveDraw(
        train_30=tuple(train_30),
        train_10=tuple(train_10),
        holdout=tuple(holdout_ids),
        holdout_by_stratum=holdout,
        holdout_reserve_by_stratum=holdout_reserve,
        train_extra_by_stratum=train_extra,
        train_reserve_by_stratum=train_reserve,
        population=population,
    )


def _assert_partition(train_30, train_10, holdout, holdout_reserve, train_reserve) -> None:
    if len(train_30) != 30 or len(set(train_30)) != 30:
        raise ValueError("train_30 is not 30 distinct grids")
    if len(train_10) != 10 or len(set(train_10)) != 10:
        raise ValueError("train_10 is not 10 distinct grids")
    if not set(train_10) <= set(train_30):
        raise ValueError("train_10 is not nested in train_30")
    if len(holdout) != 9 or len(set(holdout)) != 9:
        raise ValueError("holdout is not 9 distinct grids")
    reserved = [grid for rows in holdout_reserve.values() for grid in rows]
    reserved += [grid for rows in train_reserve.values() for grid in rows]
    used = list(train_30) + list(holdout) + reserved
    if len(used) != len(set(used)):
        raise ValueError("train, holdout, and reserves overlap")
    banned = set(SEEN_HOLDOUT)
    if banned & set(used):
        raise ValueError(f"seen holdout leaked into the draw: {banned & set(used)}")
    for grid_id in PILOT_GRIDS:
        if grid_id in holdout or grid_id in reserved:
            raise ValueError(f"pilot grid {grid_id} was drawn as holdout or reserve")


def assert_citywide_quantiles(delivered_counts: dict[str, int]) -> None:
    series = pd.Series(list(delivered_counts.values()), dtype="float64")
    if len(series) != 1301:
        raise ValueError(f"expected 1,301 occupied grids, found {len(series)}")
    got = series.quantile(list(EXPECTED_QUANTILES), interpolation="linear")
    for prob, expected in EXPECTED_QUANTILES.items():
        if float(got.loc[prob]) != expected:
            raise ValueError(f"p{int(prob * 100)} is {float(got.loc[prob])}, expected {expected}")


def load_inputs() -> tuple[dict[str, int], dict[str, int]]:
    inventory = pd.read_csv(INVENTORY_CSV, usecols=["source_grid"])
    delivered = (
        inventory.groupby("source_grid").size().astype(int).to_dict()
    )
    empty = pd.read_csv(EMPTY_PRE_COUNTS, usecols=["grid_id", "pre_count"])
    pre_counts = dict(zip(empty["grid_id"].astype(str), empty["pre_count"].astype(int)))
    return {str(key): int(value) for key, value in delivered.items()}, pre_counts


def load_and_draw() -> LearningCurveDraw:
    delivered, pre_counts = load_inputs()
    assert_citywide_quantiles(delivered)
    return draw_learning_curve(delivered, pre_counts)


def main() -> int:
    draw = load_and_draw()
    payload = draw.to_dict()
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    if SELECTION_PATH.exists():
        previous = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        if previous != payload:
            raise SystemExit(f"{SELECTION_PATH} exists and does not match a fresh draw")
        print(f"unchanged {SELECTION_PATH}")
        return 0
    SELECTION_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SELECTION_PATH}")
    print(f"train_10 {draw.train_10}")
    print(f"holdout {draw.holdout}")
    print(f"population {draw.population}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
