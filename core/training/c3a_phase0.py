"""C-3(a) Phase 0 prerequisite measurement — shared CPU-only primitives.

This module backs the F1-gap Tier C, lever C-3(a) Phase 0 gate
(``docs/plans/2026-06-10-rcnn-f1-gap-review.md`` lines 205-210): quantify the
fraction of training chips where an *unlabeled real PV* installation is being
supervised as background.  If >= 5 % of chips are affected the lever proceeds;
otherwise it is killed.

Nothing here touches a GPU.  The four CLI tools (sampler / scan-runner /
audit-builder / gate-calculator) import these primitives so the stratification,
chip-window geometry, audit CSV schema, and gate arithmetic live in one place
and are unit-testable on synthetic data.

Design contract
---------------
The *audit unit* is one training **chip window**: a ``chip_size`` x
``chip_size`` pixel window on a source tile, identical to the windows
``export_coco_dataset.scan_chips_from_tile`` enumerates when the COCO dataset
is built.  A chip is reproduced from ``(tile_path, x0, y0, w, h, chip_size)``
plus the GT polygon references it intersects.  This lets the pod-side runner
re-render the exact chip pixels and overlay both the existing GT and the
low-confidence detector proposals.

A chip is **affected** (unlabeled-real-PV-as-background) iff the human/Gemini
audit confirms at least one low-confidence detector proposal in a background
region of that chip is real PV (audit label ``confirmed_pv``).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# ──────────────────────────────────────────────────────────────────────────
# Chip-window enumeration (mirrors export_coco_dataset.scan_chips_from_tile)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChipWindow:
    """One sliding-window chip on a source tile, in source-pixel coordinates.

    Mirrors the window math in ``scan_chips_from_tile``: top-left ``(x0, y0)``,
    clipped width/height ``(w, h)`` (edge chips smaller than half ``chip_size``
    are skipped upstream), and the full ``chip_size`` the chip is padded to.
    """

    tile_stem: str
    x0: int
    y0: int
    w: int
    h: int
    chip_size: int
    n_gt: int = 0  # number of GT polygons intersecting this chip


def enumerate_chip_windows(
    tile_w: int,
    tile_h: int,
    chip_size: int,
    overlap: float,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(x0, y0, w, h)`` chip windows for a ``tile_w`` x ``tile_h`` tile.

    Byte-identical iteration order and edge-skip rule to
    ``export_coco_dataset.scan_chips_from_tile`` (stride = chip_size*(1-overlap),
    skip chips whose clipped side is < chip_size // 2).
    """
    if chip_size <= 0:
        raise ValueError(f"chip_size must be positive, got {chip_size}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    stride = int(chip_size * (1.0 - overlap))
    if stride <= 0:
        raise ValueError(f"derived stride <= 0 (chip_size={chip_size}, overlap={overlap})")
    for y0 in range(0, tile_h, stride):
        for x0 in range(0, tile_w, stride):
            x1 = min(x0 + chip_size, tile_w)
            y1 = min(y0 + chip_size, tile_h)
            w = x1 - x0
            h = y1 - y0
            if w < chip_size // 2 or h < chip_size // 2:
                continue
            yield x0, y0, w, h


# ──────────────────────────────────────────────────────────────────────────
# Stratified sampling
# ──────────────────────────────────────────────────────────────────────────


def stratum_key(region: str, imagery_layer: str) -> str:
    """Canonical stratum identifier ``<region>:<imagery_layer>``."""
    return f"{region}:{imagery_layer}"


def allocate_stratified_quota(
    stratum_counts: dict[str, int],
    target_total: int,
) -> dict[str, int]:
    """Proportionally allocate ``target_total`` chips across strata.

    Uses largest-remainder (Hamilton) apportionment so the per-stratum quotas
    sum to exactly ``min(target_total, sum(stratum_counts.values()))`` and no
    stratum is allocated more chips than it actually has.

    Strata are ordered deterministically by descending available count then by
    key, so ties break reproducibly.
    """
    if target_total < 0:
        raise ValueError(f"target_total must be >= 0, got {target_total}")
    available = {k: v for k, v in stratum_counts.items() if v > 0}
    total_available = sum(available.values())
    if total_available == 0 or target_total == 0:
        return {k: 0 for k in stratum_counts}

    capped_total = min(target_total, total_available)
    # Ideal (fractional) share per stratum.
    ideal = {k: capped_total * v / total_available for k, v in available.items()}
    floors = {k: int(math.floor(val)) for k, val in ideal.items()}
    # Never exceed availability at the floor step.
    floors = {k: min(floors[k], available[k]) for k in floors}
    allocated = sum(floors.values())
    remainder = capped_total - allocated

    # Distribute the remainder by largest fractional part, skipping strata that
    # are already at their availability cap.
    order = sorted(
        available,
        key=lambda k: (-(ideal[k] - floors[k]), -available[k], k),
    )
    i = 0
    guard = 0
    n = len(order)
    while remainder > 0 and n > 0:
        k = order[i % n]
        if floors[k] < available[k]:
            floors[k] += 1
            remainder -= 1
        i += 1
        guard += 1
        if guard > capped_total + n + 1:
            break  # all strata capped; cannot place the rest

    out = {k: 0 for k in stratum_counts}
    out.update(floors)
    return out


def sample_indices(n_available: int, quota: int, seed: int) -> list[int]:
    """Return ``quota`` sorted indices in ``[0, n_available)`` chosen with a
    deterministic RNG (stdlib ``random.Random(seed)``)."""
    import random

    if quota <= 0 or n_available <= 0:
        return []
    quota = min(quota, n_available)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_available), quota))


def stratum_sub_seed(seed: int, skey: str) -> int:
    """Derive a per-stratum RNG seed that is stable across processes.

    Builtin ``hash()`` on str is salted per process (PYTHONHASHSEED), which
    would make ``--seed`` reproducibility a false promise; zlib.crc32 is
    process- and platform-stable.
    """
    import zlib

    return seed + zlib.crc32(skey.encode("utf-8")) % 10_000


# ──────────────────────────────────────────────────────────────────────────
# Background-region proposal classification (used by the scan runner)
# ──────────────────────────────────────────────────────────────────────────


def proposal_overlaps_gt(
    proposal_geom,
    gt_geoms: Sequence,
    iof_threshold: float = 0.10,
) -> bool:
    """True if ``proposal_geom`` overlaps any GT polygon enough to be a "labeled"
    detection rather than a background-region proposal.

    Uses intersection-over-foreground (IoF, intersection / proposal area) so a
    small proposal sitting inside a large GT footprint counts as labeled even
    though IoU would be tiny.  A proposal below ``iof_threshold`` against every
    GT is a *background-region* proposal — the audit candidates.
    """
    if proposal_geom is None or proposal_geom.is_empty:
        return False
    p_area = proposal_geom.area
    if p_area <= 0:
        return False
    for g in gt_geoms:
        if g is None or g.is_empty:
            continue
        if not proposal_geom.intersects(g):
            continue
        inter = proposal_geom.intersection(g).area
        if inter / p_area >= iof_threshold:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Audit CSV schema
# ──────────────────────────────────────────────────────────────────────────

# Audit decision vocabulary. The disposition column routes each decided
# background-region proposal to its downstream sink (see the runbook):
#   confirmed_pv  -> promote to positive (NEVER to ignore); marks chip affected
#   lookalike     -> data/negative_pool/ (HN); NEVER ignore (precision project)
#   ignore_candidate -> unreviewed-margin ignore region, per-chip area cap
#   not_pv_other  -> neither PV nor a catalogued lookalike (e.g. bare roof)
#   uncertain     -> abstain; does NOT count toward affected rate
AUDIT_LABELS: tuple[tuple[str, str, str], ...] = (
    ("1", "confirmed_pv", "未标注真 PV → 转正 (positive)"),
    ("2", "lookalike", "Lookalike (天热水器/天窗) → negative_pool"),
    ("3", "ignore_candidate", "无人裁决区 → ignore 候选 (带 area cap)"),
    ("4", "not_pv_other", "非 PV 其它 (裸屋顶/阴影)"),
    ("5", "uncertain", "不确定 (弃权, 不计入受影响率)"),
)

AUDIT_LABEL_VALUES: tuple[str, ...] = tuple(v for _, v, _ in AUDIT_LABELS)

# The label that flags a chip as affected (unlabeled-real-PV-as-background).
AFFECTED_LABEL = "confirmed_pv"

# Per-chip audit row schema (one row per background-region proposal).
AUDIT_CSV_COLUMNS: tuple[str, ...] = (
    "audit_id",            # stable id: <chip_uid>__p<NN>
    "chip_uid",            # <region>:<imagery_layer>:<grid_id>:<tile_stem>:<x0>_<y0>
    "region",
    "imagery_layer",
    "grid_id",
    "tile_stem",
    "x0",
    "y0",
    "chip_size",
    "proposal_index",      # background-region proposal index within the chip
    "score",               # detector confidence (low-conf scan, ~0.05 floor)
    "proposal_area_m2",    # metric area of the proposal footprint
    "max_iof_vs_gt",       # max intersection-over-foreground vs existing GT
    "n_gt_in_chip",        # existing GT polygons intersecting the chip
    "chip_png",            # relative path to chip overlay PNG (renderer output)
    "audit_label",         # one of AUDIT_LABEL_VALUES (empty = undecided)
    "ignore_area_cap_m2",  # per-chip ignore-area cap (only when ignore_candidate)
    "audit_notes",
    "reviewed_at",
)


def make_chip_uid(
    region: str, imagery_layer: str, grid_id: str, tile_stem: str, x0: int, y0: int
) -> str:
    """Stable chip identifier used to join sampler / runner / audit rows."""
    return f"{region}:{imagery_layer}:{grid_id}:{tile_stem}:{x0}_{y0}"


def make_audit_id(chip_uid: str, proposal_index: int) -> str:
    return f"{chip_uid}__p{proposal_index:02d}"


def validate_audit_rows(rows: Iterable[dict]) -> list[str]:
    """Validate audit rows against the schema.  Returns a list of error
    strings (empty = valid).  Used by the gate calculator + tests."""
    errors: list[str] = []
    cols = set(AUDIT_CSV_COLUMNS)
    valid_labels = set(AUDIT_LABEL_VALUES) | {""}
    for i, row in enumerate(rows):
        missing = cols - set(row.keys())
        if missing:
            errors.append(f"row {i}: missing columns {sorted(missing)}")
            continue
        label = (row.get("audit_label") or "").strip()
        if label not in valid_labels:
            errors.append(
                f"row {i} ({row.get('audit_id')}): invalid audit_label {label!r}"
            )
        # ignore_area_cap_m2 must be present (numeric) when ignore_candidate.
        if label == "ignore_candidate":
            cap = (row.get("ignore_area_cap_m2") or "").strip()
            if cap == "":
                errors.append(
                    f"row {i} ({row.get('audit_id')}): ignore_candidate row "
                    f"requires a non-empty ignore_area_cap_m2"
                )
            else:
                try:
                    float(cap)
                except ValueError:
                    errors.append(
                        f"row {i} ({row.get('audit_id')}): ignore_area_cap_m2 "
                        f"{cap!r} is not numeric"
                    )
    return errors


# ──────────────────────────────────────────────────────────────────────────
# Gate computation
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_GATE_THRESHOLD = 0.05  # >= 5 % affected chips => C-3(a) proceeds


@dataclass
class StratumGateResult:
    stratum: str
    n_chips_total: int
    n_chips_decided: int  # chips with >= 1 decided (non-uncertain) proposal
    n_chips_affected: int
    affected_rate: float = field(init=False)

    def __post_init__(self) -> None:
        denom = self.n_chips_total
        self.affected_rate = (
            self.n_chips_affected / denom if denom > 0 else 0.0
        )


@dataclass
class GateResult:
    threshold: float
    n_chips_total: int
    n_chips_decided: int
    n_chips_affected: int
    affected_rate: float
    decision: str  # "PASS" | "KILL" | "INSUFFICIENT_DATA"
    per_stratum: list[StratumGateResult]
    n_proposals_total: int
    n_proposals_confirmed_pv: int
    n_proposals_lookalike: int
    n_proposals_ignore_candidate: int
    n_proposals_uncertain: int


def compute_gate(
    audit_rows: Sequence[dict],
    threshold: float = DEFAULT_GATE_THRESHOLD,
    min_decided_chips: int = 1,
    sampled_chips: Sequence[dict] | None = None,
) -> GateResult:
    """Compute the C-3(a) Phase 0 gate from decided audit rows.

    A chip is *affected* iff at least one of its background-region proposals is
    labeled ``confirmed_pv``.

    The affected-rate **denominator is every SAMPLED chip**, not only the chips
    that happened to produce a background proposal.  A chip with zero
    background-region proposals correctly contributes 0 to the rate (no
    unlabeled-PV-as-background is possible there).  Pass ``sampled_chips`` —
    rows from the sampler's ``chip_manifest.csv`` (need ``chip_uid``, ``region``,
    ``imagery_layer``) — to use the true denominator.  When omitted (e.g. unit
    tests that pass an already-complete per-chip audit), the denominator falls
    back to the distinct chips present in ``audit_rows``.

    ``decision``:
      - ``PASS``  : affected_rate >= threshold  (C-3(a) proceeds)
      - ``KILL``  : affected_rate <  threshold  (C-3(a) killed)
      - ``INSUFFICIENT_DATA`` : fewer than ``min_decided_chips`` chips have any
        decided (non-empty, non-uncertain) proposal — the audit is not done.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    # Group rows per chip.
    by_chip: dict[str, list[dict]] = {}
    for row in audit_rows:
        by_chip.setdefault(row["chip_uid"], []).append(row)

    # Per-stratum and global counters.
    strat_total: dict[str, int] = {}
    strat_decided: dict[str, int] = {}
    strat_affected: dict[str, int] = {}

    # Denominator = all sampled chips (true), else distinct chips in the audit.
    if sampled_chips is not None:
        all_chip_uids: set[str] = set()
        for c in sampled_chips:
            uid = c["chip_uid"]
            if uid in all_chip_uids:
                continue
            all_chip_uids.add(uid)
            s = stratum_key(c["region"], c["imagery_layer"])
            strat_total[s] = strat_total.get(s, 0) + 1
    else:
        for chip_uid, rows in by_chip.items():
            s = stratum_key(rows[0]["region"], rows[0]["imagery_layer"])
            strat_total[s] = strat_total.get(s, 0) + 1
        all_chip_uids = set(by_chip.keys())

    n_confirmed = n_lookalike = n_ignore = n_uncertain = 0
    n_proposals = len(audit_rows)
    n_chips_decided = 0
    n_chips_affected = 0

    for chip_uid, rows in by_chip.items():
        stratum = stratum_key(rows[0]["region"], rows[0]["imagery_layer"])
        # Defensive: an audited chip not present in sampled_chips (e.g. manifest
        # drift) still gets a stratum bucket + counts toward the denominator.
        if chip_uid not in all_chip_uids:
            all_chip_uids.add(chip_uid)
            strat_total[stratum] = strat_total.get(stratum, 0) + 1
        strat_total.setdefault(stratum, 0)

        labels = [(r.get("audit_label") or "").strip() for r in rows]
        decided = [lab for lab in labels if lab and lab != "uncertain"]
        affected = any(lab == AFFECTED_LABEL for lab in labels)

        if decided:
            n_chips_decided += 1
            strat_decided[stratum] = strat_decided.get(stratum, 0) + 1
        if affected:
            n_chips_affected += 1
            strat_affected[stratum] = strat_affected.get(stratum, 0) + 1

        for lab in labels:
            if lab == "confirmed_pv":
                n_confirmed += 1
            elif lab == "lookalike":
                n_lookalike += 1
            elif lab == "ignore_candidate":
                n_ignore += 1
            elif lab == "uncertain":
                n_uncertain += 1

    n_chips_total = len(all_chip_uids)
    affected_rate = n_chips_affected / n_chips_total if n_chips_total else 0.0

    if n_chips_decided < min_decided_chips:
        decision = "INSUFFICIENT_DATA"
    elif affected_rate >= threshold:
        decision = "PASS"
    else:
        decision = "KILL"

    per_stratum = [
        StratumGateResult(
            stratum=s,
            n_chips_total=strat_total[s],
            n_chips_decided=strat_decided.get(s, 0),
            n_chips_affected=strat_affected.get(s, 0),
        )
        for s in sorted(strat_total)
    ]

    return GateResult(
        threshold=threshold,
        n_chips_total=n_chips_total,
        n_chips_decided=n_chips_decided,
        n_chips_affected=n_chips_affected,
        affected_rate=affected_rate,
        decision=decision,
        per_stratum=per_stratum,
        n_proposals_total=n_proposals,
        n_proposals_confirmed_pv=n_confirmed,
        n_proposals_lookalike=n_lookalike,
        n_proposals_ignore_candidate=n_ignore,
        n_proposals_uncertain=n_uncertain,
    )


# ──────────────────────────────────────────────────────────────────────────
# CSV helpers (thin, stdlib-only so tests don't need pandas)
# ──────────────────────────────────────────────────────────────────────────


def write_audit_csv(rows: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(AUDIT_CSV_COLUMNS))
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in AUDIT_CSV_COLUMNS})


def read_audit_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
