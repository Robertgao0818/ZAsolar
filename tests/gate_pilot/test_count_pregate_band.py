"""Offline count of pregate scores in a band. No detector, no threshold change."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_pilot.count_pregate_band import (  # noqa: E402
    main,
    summarize_pregate_band,
)


def _write(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chip_index", "kept", "score_max"])
        for index, (kept, score) in enumerate(rows):
            writer.writerow([index, kept, score])


def test_band_counts_extra_work_against_kept_chips(tmp_path: Path, capsys):
    _write(tmp_path / "DBN0001" / "pregate_decisions.csv", [
        ("0", "0.10"),
        ("0", "0.25"),
        ("0", "0.44"),
        ("1", "0.45"),
        ("1", "0.80"),
    ])
    # Already kept, so it is in the band but would not add detector work.
    _write(tmp_path / "DBN0002" / "pregate_decisions.csv", [("1", "0.30")])
    # Nested files are not the citywide layout and must not be counted twice.
    _write(tmp_path / "DBN0001" / "nested" / "pregate_decisions.csv", [("0", "0.30")])

    rows, summary = summarize_pregate_band(tmp_path, 0.25, 0.45)
    by_id = {row["grid_id"]: row for row in rows}
    assert set(by_id) == {"DBN0001", "DBN0002"}
    assert by_id["DBN0001"]["n_chips"] == 5
    assert by_id["DBN0001"]["n_kept"] == 2
    assert by_id["DBN0001"]["n_band"] == 2
    assert by_id["DBN0001"]["n_extra"] == 2
    assert by_id["DBN0001"]["extra_over_kept"] == 1
    assert by_id["DBN0002"]["n_band"] == 1
    assert by_id["DBN0002"]["n_extra"] == 0
    assert summary["n_chips"] == 6
    assert summary["n_kept"] == 3
    assert summary["n_extra"] == 2
    assert summary["extra_over_kept"] == 2 / 3

    out = tmp_path / "counts.csv"
    code = main(["--root", str(tmp_path), "--output", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "would add 2 chips, 2/3 = 0.6667" in text
    with out.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [row["grid_id"] for row in written] == ["DBN0001", "DBN0002"]
    assert written[0]["extra_over_kept"] == "1.000000"
