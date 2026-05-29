"""Single source of truth for label_source → boundary-trust attributes.

Loads `data/training_pool/boundary_trust_rules.yaml`, which maps each
`label_source` enum to (mask_trusted, boundary_w). Before 2026-05-29 these
values were duplicated as hardcoded dicts in two places that could drift:
  - `export_coco_dataset.py`  →  `_MASK_TRUSTED`
  - `train.py`                →  `_LABEL_SOURCE_TO_MASK_TRUSTED`,
                                 `_LABEL_SOURCE_TO_BOUNDARY_W`

This module is the one place that owns the per-key table. Edge-case handling
(None / unknown source) is intentionally left to each consumer to preserve
existing behavior — see the YAML header for the per-consumer policy.

See .claude/rules/07-annotation-semantics.md and data/annotations/ANNOTATION_SPEC.md.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOUNDARY_TRUST_RULES_PATH = (
    PROJECT_ROOT / "data" / "training_pool" / "boundary_trust_rules.yaml"
)


@functools.lru_cache(maxsize=None)
def load_boundary_trust_rules(path: str | None = None) -> dict:
    """Load + validate the boundary-trust rules YAML.

    Returns the parsed dict with keys: ``schema_version``,
    ``fail_closed_default``, ``legacy_no_source_field``, ``map``
    (label_source → {mask_trusted, boundary_w}). Cached per path.
    """
    p = Path(path) if path else BOUNDARY_TRUST_RULES_PATH
    with open(p) as f:
        rules = yaml.safe_load(f)
    if not isinstance(rules, dict) or not isinstance(rules.get("map"), dict):
        raise ValueError(f"{p}: missing or invalid 'map' section")
    for src, attrs in rules["map"].items():
        if not isinstance(attrs, dict) or "mask_trusted" not in attrs or "boundary_w" not in attrs:
            raise ValueError(
                f"{p}: label_source {src!r} must define both 'mask_trusted' and 'boundary_w'"
            )
    return rules


def mask_trusted_map(path: str | None = None) -> dict[str, bool]:
    """label_source → mask_trusted (bool). Fresh dict each call (safe to mutate)."""
    rules = load_boundary_trust_rules(path)
    return {src: bool(a["mask_trusted"]) for src, a in rules["map"].items()}


def boundary_w_map(path: str | None = None) -> dict[str, float]:
    """label_source → boundary BCE weight (float). Fresh dict each call."""
    rules = load_boundary_trust_rules(path)
    return {src: float(a["boundary_w"]) for src, a in rules["map"].items()}
