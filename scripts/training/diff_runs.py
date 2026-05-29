#!/usr/bin/env python3
"""Diff two training runs → "what changed → how metrics moved".

Standalone, no torch. Loads two ``run_manifest.json`` files (by run_id from
``runs/<run_id>/run_manifest.json`` or by explicit path) and reports the
configuration deltas (dataset build_id, init_weights, seed, hyperparams,
boundary-aware config, git commit) alongside the metric deltas. This is the
user-requested traceability tool for Phase 3.

Usage:
    python scripts/training/diff_runs.py <runA> <runB>

where each <run> is either a run_id (resolved under runs/) or a path to a
run_manifest.json file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"


def _resolve_manifest(ref: str, runs_root: Path) -> dict[str, Any]:
    """Resolve a run reference (run_id or path) to a manifest dict."""
    p = Path(ref)
    if p.is_file():
        path = p
    elif p.is_dir() and (p / "run_manifest.json").is_file():
        path = p / "run_manifest.json"
    else:
        path = runs_root / ref / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not resolve run '{ref}' to a run_manifest.json "
            f"(tried path and {runs_root / ref / 'run_manifest.json'})"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys (lists/scalars left as-is)."""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    else:
        out[prefix] = d
    return out


def _diff_section(a: dict[str, Any], b: dict[str, Any], section: str) -> list[str]:
    """Return human-readable lines for changed keys within a flattened section."""
    fa = _flatten(a.get(section, {}))
    fb = _flatten(b.get(section, {}))
    keys = sorted(set(fa) | set(fb))
    lines: list[str] = []
    for k in keys:
        va, vb = fa.get(k), fb.get(k)
        if va != vb:
            lines.append(f"    {k}: {va!r} -> {vb!r}")
    return lines


def _scalar_diff(a: dict, b: dict, key: str, label: str | None = None) -> str | None:
    va, vb = a.get(key), b.get(key)
    if va != vb:
        return f"  {label or key}: {va!r} -> {vb!r}"
    return None


def diff_runs(man_a: dict[str, Any], man_b: dict[str, Any]) -> str:
    """Produce the "what changed → how metrics moved" report text."""
    lines: list[str] = []
    ra = man_a.get("run_id", "?")
    rb = man_b.get("run_id", "?")
    lines.append(f"Run A: {ra}")
    lines.append(f"Run B: {rb}")
    lines.append("")
    lines.append("=== Configuration changes (A -> B) ===")

    changed_any = False

    # dataset.build_id + spec
    ds_lines = _diff_section(man_a, man_b, "dataset")
    if ds_lines:
        changed_any = True
        lines.append("  [dataset]")
        lines.extend(ds_lines)

    # init weights + seed
    for key, label in (
        ("init_weights", "init_weights"),
        ("init_weights_sha256", "init_weights_sha256"),
        ("seed", "seed"),
    ):
        s = _scalar_diff(man_a, man_b, key, label)
        if s:
            changed_any = True
            lines.append(s)

    # hyperparams
    hp_lines = _diff_section(man_a, man_b, "hyperparams")
    if hp_lines:
        changed_any = True
        lines.append("  [hyperparams]")
        lines.extend(hp_lines)

    # boundary_aware
    ba_lines = _diff_section(man_a, man_b, "boundary_aware")
    if ba_lines:
        changed_any = True
        lines.append("  [boundary_aware]")
        lines.extend(ba_lines)

    # git commit
    cpa = man_a.get("code_provenance", {})
    cpb = man_b.get("code_provenance", {})
    s = _scalar_diff(cpa, cpb, "git_commit", "git_commit")
    if s:
        changed_any = True
        lines.append(s)

    if not changed_any:
        lines.append("  (no configuration differences)")

    # Metrics deltas
    lines.append("")
    lines.append("=== Metric deltas (B - A) ===")
    ma = (man_a.get("metrics") or {}).get("chip_level") or {}
    mb = (man_b.get("metrics") or {}).get("chip_level") or {}
    metric_keys = sorted(set(ma) | set(mb))
    metric_lines: list[str] = []
    for k in metric_keys:
        va, vb = ma.get(k), mb.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            arrow = "+" if delta >= 0 else ""
            metric_lines.append(f"  {k}: {va} -> {vb}  ({arrow}{delta:.4g})")
        elif va != vb:
            metric_lines.append(f"  {k}: {va!r} -> {vb!r}")
    if metric_lines:
        lines.extend(metric_lines)
    else:
        lines.append("  (no chip-level metric differences)")

    # grid-level placeholder note
    ga = (man_a.get("metrics") or {}).get("grid_level")
    gb = (man_b.get("metrics") or {}).get("grid_level")
    if ga is None and gb is None:
        lines.append("  grid_level: not yet back-filled for either run")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diff two training runs (what changed -> how metrics moved)."
    )
    parser.add_argument("run_a", help="run_id or path to run_manifest.json")
    parser.add_argument("run_b", help="run_id or path to run_manifest.json")
    parser.add_argument(
        "--runs-root", default=str(DEFAULT_RUNS_ROOT),
        help="Root dir for run_id resolution (default: <repo>/runs).",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    man_a = _resolve_manifest(args.run_a, runs_root)
    man_b = _resolve_manifest(args.run_b, runs_root)
    print(diff_runs(man_a, man_b))


if __name__ == "__main__":
    main()
