#!/usr/bin/env python3
"""Count pregate chips in a score band. Does not change the threshold.

Reads every ``*/pregate_decisions.csv`` under a root (one directory level,
the citywide layout). Prints how many extra chips the detector would see
if the keep threshold moved from the top of the band down to the bottom,
as extra chips divided by chips the gate already keeps.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_ROOT = (
    Path.home() / "zasolar_data" / "gate_pilot_2026-09" / "durban_citywide_pregate_20261008"
)


def summarize_pregate_band(root: Path, low: float = 0.25, high: float = 0.45) -> tuple[list[dict], dict]:
    """Return per-grid rows and a total summary.

    A chip is in the band when ``low <= score_max < high``. ``n_extra`` is
    the band chips that are not already kept. Those are the chips a
    threshold of ``low`` would add if today's keep threshold is ``high``.
    """
    low = float(low)
    high = float(high)
    if not (low < high):
        raise ValueError(f"band must have low < high, got [{low}, {high})")
    root = Path(root)
    rows = []
    for path in sorted(root.glob("*/pregate_decisions.csv")):
        rows.append(_one_grid(path, low, high))
    return rows, _totals(rows, low, high)


def _one_grid(path: Path, low: float, high: float) -> dict:
    n_chips = n_kept = n_band = n_extra = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "score_max" not in fields:
            raise ValueError(f"{path} has no score_max column")
        has_kept = "kept" in fields
        for record in reader:
            n_chips += 1
            try:
                score = float(record["score_max"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} has a non-numeric score_max") from exc
            kept = _is_kept(record["kept"]) if has_kept else score >= high
            if kept:
                n_kept += 1
            if low <= score < high:
                n_band += 1
                if not kept:
                    n_extra += 1
    return {
        "grid_id": path.parent.name,
        "n_chips": n_chips,
        "n_kept": n_kept,
        "n_band": n_band,
        "n_extra": n_extra,
        "extra_over_kept": _ratio(n_extra, n_kept),
    }


def _is_kept(value) -> bool:
    return str(value).strip() in {"1", "1.0", "True", "true"}


def _ratio(extra: int, kept: int):
    if kept <= 0:
        return None
    return extra / kept


def _totals(rows: list[dict], low: float, high: float) -> dict:
    n_chips = sum(row["n_chips"] for row in rows)
    n_kept = sum(row["n_kept"] for row in rows)
    n_band = sum(row["n_band"] for row in rows)
    n_extra = sum(row["n_extra"] for row in rows)
    return {
        "n_grids": len(rows),
        "n_chips": n_chips,
        "n_kept": n_kept,
        "n_band": n_band,
        "n_extra": n_extra,
        "extra_over_kept": _ratio(n_extra, n_kept),
        "low": low,
        "high": high,
    }


def write_table(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["grid_id", "n_chips", "n_kept", "n_band", "n_extra", "extra_over_kept"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            ratio = out["extra_over_kept"]
            out["extra_over_kept"] = "" if ratio is None else f"{ratio:.6f}"
            writer.writerow(out)


def format_summary(root: Path, summary: dict) -> str:
    ratio = summary["extra_over_kept"]
    ratio_text = "undefined (no kept chips)" if ratio is None else f"{ratio:.4f}"
    return (
        f"root: {root}\n"
        f"band: [{summary['low']}, {summary['high']})\n"
        f"grids: {summary['n_grids']}\n"
        f"chips: {summary['n_chips']}\n"
        f"kept: {summary['n_kept']}\n"
        f"band: {summary['n_band']}\n"
        f"extra: {summary['n_extra']}\n"
        f"lowering the threshold to {summary['low']} would add {summary['n_extra']} "
        f"chips, {summary['n_extra']}/{summary['n_kept']} = {ratio_text} "
        f"of the chips the detector already runs\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--low", type=float, default=0.25)
    parser.add_argument("--high", type=float, default=0.45)
    parser.add_argument(
        "--output", type=Path, default=Path("pregate_band_counts.csv"),
        help="Per-grid CSV. Default is pregate_band_counts.csv in the working directory.",
    )
    args = parser.parse_args(argv)
    try:
        rows, summary = summarize_pregate_band(args.root, args.low, args.high)
    except (ValueError, OSError) as exc:
        print(f"[count_pregate_band] {exc}", file=sys.stderr)
        return 2
    write_table(args.output, rows)
    print(format_summary(args.root, summary), end="")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
