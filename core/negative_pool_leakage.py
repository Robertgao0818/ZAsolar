"""Eval-leakage guard for the project-level hard-negative pool.

When a grid's predictions are mined into ``data/negative_pool/manifest.csv`` as
hard negatives, that grid can no longer serve as a clean cross-domain
*evaluation* surface: a model retrained on those HN chips has effectively seen
(part of) the grid at train time, so any cross-domain improvement measured on
it is contaminated.  This module is the single machine-readable source of
"which grids have been mined", so eval surfaces (e.g. xdomain60) can exclude
them.

The mined-grid set is derived directly from the monotonic manifest — it is not
a separate hand-maintained list, so it can never drift out of sync.  Region is
read from the manifest's explicit ``region`` column (never inferred from
grid_id — CT/JHB grid IDs overlap, rule 06-multi-city).
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "negative_pool" / "manifest.csv"


def _row_training_eligible(row: dict) -> bool:
    """True if a manifest row may enter a training bundle.

    Mirrors ``pipeline.hn_ops._is_training_eligible``: an explicit ``false``
    gates the row out; ``true`` or a missing/blank value (legacy rows that
    predate the column) count as eligible.
    """
    return (row.get("training_eligible") or "").strip().lower() != "false"


def mined_grid_keys(
    manifest_csv: Path | None = None,
    *,
    include_provenance_only: bool = False,
) -> set[tuple[str, str]]:
    """Return the set of ``(region, grid_id)`` pairs that contaminate eval.

    A grid is eval-contaminated only if a retrained model could actually have
    seen its chips at train time — i.e. it carries at least one
    ``training_eligible`` row.  ``training_eligible=false`` rows are
    provenance-only (e.g. geid_2024_02 bootstrap chips and cross-domain
    empty-probe rows gated out by the imagery-layer balance rule): no model
    trains on them, so the grid is NOT leaked through this pool.  This keeps the
    leakage definition consistent with the ``training_eligible`` gate that the
    HN extractor enforces — otherwise a purely-provenance GEID row would wrongly
    evict the 25 JHB CBD benchmark grids from a JHB eval surface.

    Pass ``include_provenance_only=True`` to get the full *provenance* footprint
    (every grid ever visited by an ingest pass, regardless of eligibility) — use
    this only for auditing what the pool has touched, never for eval filtering.

    Rows tagged ``actually_pv_mislabeled`` are still counted (when otherwise
    eligible): the grid was visited by the FP-review / ingest pass even if a
    given chip was later re-tagged a GT gap, so the grid remains contaminated.
    """
    manifest_csv = manifest_csv or DEFAULT_MANIFEST
    keys: set[tuple[str, str]] = set()
    if not manifest_csv.exists():
        return keys
    with manifest_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            region = (row.get("region") or "").strip()
            grid_id = (row.get("grid_id") or "").strip()
            if not (region and grid_id):
                continue
            if not include_provenance_only and not _row_training_eligible(row):
                continue
            keys.add((region, grid_id))
    return keys


def mined_grids_for_region(
    region: str,
    manifest_csv: Path | None = None,
    *,
    include_provenance_only: bool = False,
) -> set[str]:
    """Return the set of mined ``grid_id`` for one region (eligible rows only).

    See ``mined_grid_keys`` for the ``include_provenance_only`` semantics.
    """
    return {
        g
        for (r, g) in mined_grid_keys(
            manifest_csv, include_provenance_only=include_provenance_only
        )
        if r == region
    }


def is_mined(region: str, grid_id: str,
             manifest_csv: Path | None = None,
             *, include_provenance_only: bool = False) -> bool:
    """True iff ``(region, grid_id)`` is eval-contaminated by the HN pool."""
    return (region, grid_id) in mined_grid_keys(
        manifest_csv, include_provenance_only=include_provenance_only
    )


def filter_eval_grids(
    region: str,
    grid_ids: list[str],
    manifest_csv: Path | None = None,
    *,
    include_provenance_only: bool = False,
) -> tuple[list[str], list[str]]:
    """Split ``grid_ids`` into ``(kept, excluded)`` for a clean eval surface.

    ``excluded`` are grids mined into the HN pool for ``region`` with at least
    one ``training_eligible`` row; ``kept`` is the leakage-free remainder,
    order-preserving.  Provenance-only grids are kept by default (no model
    trains on them); see ``mined_grid_keys``.
    """
    mined = mined_grids_for_region(
        region, manifest_csv, include_provenance_only=include_provenance_only
    )
    kept = [g for g in grid_ids if g not in mined]
    excluded = [g for g in grid_ids if g in mined]
    return kept, excluded
