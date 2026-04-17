"""Dataset spec schema definition and strict validation.

A dataset spec is a YAML file under ``configs/pipelines/datasets/`` that
fully describes a reproducible training-data build recipe.  Specs declare
*what* to build (regions, filters, HN policy, chip params) but never
duplicate path ownership from ``configs/datasets/regions.yaml``.

Usage::

    from pipeline.specs import load_spec

    spec = load_spec("configs/pipelines/datasets/v4_1_hn.yaml")
    # spec is a validated DatasetSpec; all invariants already checked.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Policy constants — upper bounds derived from project experience.
# V4.1 used 15.7% combined HN and suffered recall regression; V3-C used
# ~8% and was the best overall.  These caps are intentionally generous;
# the builder logs a warning when ratios exceed recommended ranges.
# ---------------------------------------------------------------------------
MAX_EASY_NEG_RATIO = 5.0       # 5:1 neg:pos is already very heavy
MAX_HN_RATIO = 0.30            # 30% HN was never tested; 15% already hurt
RECOMMENDED_MAX_HN_RATIO = 0.12  # above this, warn about recall risk

# Allowed enum values
VALID_TIER_FILTERS = {"T1", "T2", "T1+T2"}
VALID_HN_TYPES = {"reviewed_fp_hn", "small_fp_hn"}
VALID_SPLIT_STRATEGIES = {"tile_greedy_by_annotation_count"}
VALID_EVAL_REGIMES = {"historical_holdout", "parallel_ra_independent_eval"}
VALID_BUILD_FAMILIES = {"detector_train"}

# Schema version this code supports
SUPPORTED_SCHEMA_VERSIONS = {1}

# Lazy import to avoid circular deps at module level
_region_registry = None


def _get_registry():
    global _region_registry
    if _region_registry is None:
        from core import region_registry
        _region_registry = region_registry
    return _region_registry


# ---------------------------------------------------------------------------
# Sub-specs (nested dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class SelectionSpec:
    tier_filter: str = "T1+T2"
    exclude_grids: list[str] = field(default_factory=list)
    audit_csv: str | None = None
    exclude_audit_labels: list[str] = field(default_factory=lambda: ["heater_or_non_pv", "uncertain"])


@dataclass
class ChipSpec:
    size: int = 400
    overlap: float = 0.25


@dataclass
class SplitSpec:
    strategy: str = "tile_greedy_by_annotation_count"
    val_fraction: float = 0.2
    seed: int = 42


@dataclass
class NegativesSpec:
    easy_neg_ratio: float = 0.15


@dataclass
class HardNegativeEntry:
    type: str = ""
    region: str | None = None
    grids: list[str] = field(default_factory=list)
    max_ratio: float = 0.10
    shortlist_csv: str | None = None
    sample_rate: float = 0.5


@dataclass
class OutputSpec:
    root: str = "${SOLAR_ARTIFACT_ROOT:-/mnt/d/ZAsolar}"
    name_template: str = "coco_{name}_{date}"


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    schema_version: int = 1
    name: str = ""
    build_family: str = "detector_train"
    regions: list[str] = field(default_factory=lambda: ["cape_town"])
    evaluation_regime: str = "parallel_ra_independent_eval"

    selection: SelectionSpec = field(default_factory=SelectionSpec)
    chip: ChipSpec = field(default_factory=ChipSpec)
    split: SplitSpec = field(default_factory=SplitSpec)
    negatives: NegativesSpec = field(default_factory=NegativesSpec)
    hard_negatives: list[HardNegativeEntry] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)

    def validate(self, *, check_files: bool = True) -> list[str]:
        """Run all validation checks.  Returns list of warnings.

        Raises ``ValueError`` on hard failures (unknown keys are caught
        at parse time via ``_parse_raw``).
        """
        warnings: list[str] = []
        errors: list[str] = []

        # 1. schema_version
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            errors.append(
                f"schema_version {self.schema_version} not supported "
                f"(expected one of {SUPPORTED_SCHEMA_VERSIONS})"
            )

        # 2. enum fields
        if not self.name:
            errors.append("name must be non-empty")

        if self.build_family not in VALID_BUILD_FAMILIES:
            errors.append(f"build_family '{self.build_family}' not in {VALID_BUILD_FAMILIES}")

        if self.evaluation_regime not in VALID_EVAL_REGIMES:
            errors.append(
                f"evaluation_regime '{self.evaluation_regime}' "
                f"not in {VALID_EVAL_REGIMES}"
            )

        if self.selection.tier_filter not in VALID_TIER_FILTERS:
            errors.append(
                f"tier_filter '{self.selection.tier_filter}' "
                f"not in {VALID_TIER_FILTERS}"
            )

        if self.split.strategy not in VALID_SPLIT_STRATEGIES:
            errors.append(
                f"split.strategy '{self.split.strategy}' "
                f"not in {VALID_SPLIT_STRATEGIES}"
            )

        for i, hn in enumerate(self.hard_negatives):
            if hn.type not in VALID_HN_TYPES:
                errors.append(
                    f"hard_negatives[{i}].type '{hn.type}' "
                    f"not in {VALID_HN_TYPES}"
                )

        # 3. ratio bounds (policy constants)
        if not (0 <= self.negatives.easy_neg_ratio <= MAX_EASY_NEG_RATIO):
            errors.append(
                f"easy_neg_ratio {self.negatives.easy_neg_ratio} "
                f"outside [0, {MAX_EASY_NEG_RATIO}]"
            )

        for i, hn in enumerate(self.hard_negatives):
            if not (0 <= hn.max_ratio <= MAX_HN_RATIO):
                errors.append(
                    f"hard_negatives[{i}].max_ratio {hn.max_ratio} "
                    f"outside [0, {MAX_HN_RATIO}]"
                )
            if hn.max_ratio > RECOMMENDED_MAX_HN_RATIO:
                warnings.append(
                    f"hard_negatives[{i}].max_ratio {hn.max_ratio} "
                    f"exceeds recommended {RECOMMENDED_MAX_HN_RATIO} "
                    f"(recall regression risk)"
                )

        if not (0 < self.split.val_fraction < 1.0):
            errors.append(
                f"split.val_fraction {self.split.val_fraction} "
                f"must be in (0, 1)"
            )

        if self.chip.size < 64 or self.chip.size > 2048:
            errors.append(f"chip.size {self.chip.size} outside [64, 2048]")

        if not (0 <= self.chip.overlap < 1.0):
            errors.append(f"chip.overlap {self.chip.overlap} outside [0, 1)")

        # 4. referenced files exist (unless deferred)
        if check_files:
            base = Path(__file__).resolve().parent.parent
            if self.selection.audit_csv:
                p = base / self.selection.audit_csv
                if not p.exists():
                    errors.append(f"audit_csv not found: {p}")

            for i, hn in enumerate(self.hard_negatives):
                if hn.shortlist_csv:
                    p = base / hn.shortlist_csv
                    if not p.exists():
                        errors.append(
                            f"hard_negatives[{i}].shortlist_csv not found: {p}"
                        )

        # 5. include/exclude must not conflict
        exclude_set = set(self.selection.exclude_grids)
        for i, hn in enumerate(self.hard_negatives):
            overlap = exclude_set & set(hn.grids)
            if overlap:
                errors.append(
                    f"hard_negatives[{i}] grids {overlap} are also in "
                    f"exclude_grids — conflicting rules"
                )

        # 6. regions must be registered
        registry = _get_registry()
        registered = set(registry.list_regions())
        for r in self.regions:
            if r not in registered:
                errors.append(
                    f"region '{r}' not found in regions.yaml "
                    f"(available: {registered})"
                )

        # 7. grid IDs compatible with region scope
        #    Three-tier check:
        #    - grid in registry → OK (no message)
        #    - grid not in registry but discoverable via fallback scan → warning
        #    - grid not discoverable at all → error (likely typo)
        all_registered_grids: set[str] = set()
        for r in self.regions:
            if r in registered:
                all_registered_grids.update(registry.list_grids(r))

        # Build discoverable set (registry + fallback scan) for fallback check
        all_discoverable_grids: set[str] | None = None  # lazy

        def _get_discoverable() -> set[str]:
            nonlocal all_discoverable_grids
            if all_discoverable_grids is None:
                from core.annotation_loader import discover_annotations
                discovered = discover_annotations(
                    regions=self.regions,
                )
                all_discoverable_grids = set(discovered.keys())
            return all_discoverable_grids

        def _check_grid(gid: str, context: str) -> None:
            if gid in all_registered_grids:
                return  # fully registered, OK
            if gid in _get_discoverable():
                warnings.append(
                    f"{context} '{gid}' not registered in regions.yaml "
                    f"(found via fallback directory scan)"
                )
            else:
                errors.append(
                    f"{context} '{gid}' not found in any selected region "
                    f"— not in registry and not discoverable on disk"
                )

        for gid in self.selection.exclude_grids:
            _check_grid(gid, "exclude_grids")

        for i, hn in enumerate(self.hard_negatives):
            for gid in hn.grids:
                _check_grid(gid, f"hard_negatives[{i}].grids")
            if hn.region and hn.region not in registered:
                errors.append(
                    f"hard_negatives[{i}].region '{hn.region}' "
                    f"not in regions.yaml"
                )

        if errors:
            msg = "Dataset spec validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(msg)

        return warnings


# ---------------------------------------------------------------------------
# Parsing from raw YAML dict
# ---------------------------------------------------------------------------

# Known top-level keys (for unknown-key detection)
_KNOWN_TOP_KEYS = {
    "schema_version", "name", "build_family", "regions",
    "evaluation_regime", "selection", "chip", "split",
    "negatives", "hard_negatives", "output",
}
_KNOWN_SELECTION_KEYS = {
    "tier_filter", "exclude_grids", "audit_csv", "exclude_audit_labels",
}
_KNOWN_CHIP_KEYS = {"size", "overlap"}
_KNOWN_SPLIT_KEYS = {"strategy", "val_fraction", "seed"}
_KNOWN_NEGATIVES_KEYS = {"easy_neg_ratio"}
_KNOWN_HN_KEYS = {"type", "region", "grids", "max_ratio", "shortlist_csv", "sample_rate"}
_KNOWN_OUTPUT_KEYS = {"root", "name_template"}


def _check_unknown_keys(data: dict, known: set[str], context: str) -> None:
    unknown = set(data.keys()) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in {context}: {unknown}. "
            f"Allowed: {known}"
        )


def _parse_raw(raw: dict[str, Any]) -> DatasetSpec:
    """Parse a raw YAML dict into a validated DatasetSpec.

    Raises ValueError on unknown keys or type mismatches.
    """
    _check_unknown_keys(raw, _KNOWN_TOP_KEYS, "top-level spec")

    sel_raw = raw.get("selection", {})
    _check_unknown_keys(sel_raw, _KNOWN_SELECTION_KEYS, "selection")

    chip_raw = raw.get("chip", {})
    _check_unknown_keys(chip_raw, _KNOWN_CHIP_KEYS, "chip")

    split_raw = raw.get("split", {})
    _check_unknown_keys(split_raw, _KNOWN_SPLIT_KEYS, "split")

    neg_raw = raw.get("negatives", {})
    _check_unknown_keys(neg_raw, _KNOWN_NEGATIVES_KEYS, "negatives")

    out_raw = raw.get("output", {})
    _check_unknown_keys(out_raw, _KNOWN_OUTPUT_KEYS, "output")

    hn_entries = []
    for i, hn_raw in enumerate(raw.get("hard_negatives", [])):
        if not isinstance(hn_raw, dict):
            raise ValueError(f"hard_negatives[{i}] must be a mapping, got {type(hn_raw)}")
        _check_unknown_keys(hn_raw, _KNOWN_HN_KEYS, f"hard_negatives[{i}]")
        hn_entries.append(HardNegativeEntry(
            type=hn_raw.get("type", ""),
            region=hn_raw.get("region"),
            grids=hn_raw.get("grids", []),
            max_ratio=float(hn_raw.get("max_ratio", 0.10)),
            shortlist_csv=hn_raw.get("shortlist_csv"),
            sample_rate=float(hn_raw.get("sample_rate", 0.5)),
        ))

    return DatasetSpec(
        schema_version=int(raw.get("schema_version", 1)),
        name=str(raw.get("name", "")),
        build_family=str(raw.get("build_family", "detector_train")),
        regions=raw.get("regions", ["cape_town"]),
        evaluation_regime=str(raw.get("evaluation_regime", "parallel_ra_independent_eval")),
        selection=SelectionSpec(
            tier_filter=str(sel_raw.get("tier_filter", "T1+T2")),
            exclude_grids=sel_raw.get("exclude_grids", []),
            audit_csv=sel_raw.get("audit_csv"),
            exclude_audit_labels=sel_raw.get("exclude_audit_labels",
                                             ["heater_or_non_pv", "uncertain"]),
        ),
        chip=ChipSpec(
            size=int(chip_raw.get("size", 400)),
            overlap=float(chip_raw.get("overlap", 0.25)),
        ),
        split=SplitSpec(
            strategy=str(split_raw.get("strategy", "tile_greedy_by_annotation_count")),
            val_fraction=float(split_raw.get("val_fraction", 0.2)),
            seed=int(split_raw.get("seed", 42)),
        ),
        negatives=NegativesSpec(
            easy_neg_ratio=float(neg_raw.get("easy_neg_ratio", 0.15)),
        ),
        hard_negatives=hn_entries,
        output=OutputSpec(
            root=str(out_raw.get("root", "${SOLAR_ARTIFACT_ROOT:-/mnt/d/ZAsolar}")),
            name_template=str(out_raw.get("name_template", "coco_{name}_{date}")),
        ),
    )


# ---------------------------------------------------------------------------
# Environment variable resolution
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def resolve_env_vars(value: str) -> str:
    """Resolve ``${VAR:-default}`` patterns in a string."""
    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        default = m.group(2) or ""
        return os.environ.get(var_name, default)
    return _ENV_PATTERN.sub(_replace, value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_spec(path: str | Path, *, check_files: bool = True) -> DatasetSpec:
    """Load, parse, and validate a dataset spec from a YAML file.

    Args:
        path: Path to the YAML spec file.
        check_files: If True, verify that referenced files exist on disk.

    Returns:
        A fully validated ``DatasetSpec``.

    Raises:
        FileNotFoundError: If the spec file does not exist.
        ValueError: On unknown keys, type errors, or validation failures.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Spec file must be a YAML mapping, got {type(raw)}")

    spec = _parse_raw(raw)
    warnings = spec.validate(check_files=check_files)

    for w in warnings:
        print(f"[WARN] {w}")

    return spec


def spec_to_dict(spec: DatasetSpec) -> dict[str, Any]:
    """Convert a DatasetSpec back to a plain dict (for manifest serialization)."""
    from dataclasses import asdict
    return asdict(spec)
