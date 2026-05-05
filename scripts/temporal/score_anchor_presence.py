#!/usr/bin/env python3
# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/scripts/temporal/score_anchor_presence.py
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
"""Create anchor-level GEID historical PV presence templates.

This is the Phase-0 scorer bridge for the GEID temporal workflow.  It does not
train or run a model.  It scans GEID historical downloader outputs for each
anchor/date task, records chip/capture-date provenance, and emits a
`presence_timeseries.csv` template that can be manually filled or overridden by
a manual decisions CSV.

The important policy is unchanged: existing aerial/Vexcel masks are location
anchors only.  Presence rows should be interpreted as binary PV visibility near
the anchor in the historical chip, not as historical mask IoU labels.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.temporal.geid_temporal_common import read_csv_rows, write_csv_rows

DEFAULT_ANCHORS = PROJECT_ROOT / "data" / "geid_temporal" / "anchors.csv"
DEFAULT_TASKS = PROJECT_ROOT / "data" / "geid_temporal" / "geid_tasks.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "geid_temporal" / "presence_timeseries.csv"

JPEG_SOI = b"\xff\xd8\xff"
COMMENT_RE = re.compile(rb"\*AD\*(\d{4}):(\d{2}):(\d{2})\*")
DATE_SUFFIX_RE = re.compile(r"^(?P<anchor>.+)_(?P<date>\d{8})$")

PRESENCE_FIELDS = [
    "anchor_id",
    "region_key",
    "grid_id",
    "requested_date",
    "capture_date",
    "actual_capture_dates",
    "pv_score",
    "pv_present",
    "decision_source",
    "quality_flag",
    "chip_dir",
    "sample_chip_path",
    "n_jpg",
    "task_name",
    "save_to",
    "notes",
]

MANUAL_DECISION_FIELDS = [
    "anchor_id",
    "requested_date",
    "capture_date",
    "pv_present",
    "pv_score",
    "quality_flag",
    "decision_source",
    "notes",
]


def windows_path_to_wsl(path: str | os.PathLike[str]) -> Path | None:
    """Translate a simple Windows drive path to its WSL `/mnt/<drive>` path."""
    s = str(path).strip().strip('"')
    match = re.match(r"^([A-Za-z]):[\\/]*(.*)$", s)
    if not match:
        return None
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return Path("/mnt") / drive / rest if rest else Path("/mnt") / drive


def resolve_task_dir(save_to: str | os.PathLike[str], task_name: str) -> Path:
    """Return the WSL/POSIX directory where GEID writes a task's JPG tiles.

    The Allmapsoft CLI receives a `save_to` root and creates a child directory
    named `task_name`.  `save_to` may be a Windows drive path from the task CSV
    or a normal POSIX path used in tests/smokes.
    """
    wsl_root = windows_path_to_wsl(save_to)
    root = wsl_root if wsl_root is not None else Path(save_to)
    return root / str(task_name)


def extract_geid_capture_date(jpg_path: Path) -> str | None:
    """Extract embedded GEID capture date from a JPEG comment, if present."""
    try:
        head = jpg_path.read_bytes()[:4096]
    except OSError:
        return None
    if not head.startswith(JPEG_SOI):
        return None
    match = COMMENT_RE.search(head)
    if not match:
        return None
    return f"{match.group(1).decode()}-{match.group(2).decode()}-{match.group(3).decode()}"


def list_task_jpgs(task_dir: Path, *, min_bytes: int = 1) -> list[Path]:
    if not task_dir.exists():
        return []
    return sorted(p for p in task_dir.rglob("*.jpg") if p.is_file() and p.stat().st_size >= min_bytes)


def summarise_task_jpgs(task_dir: Path, *, min_bytes: int = 1) -> dict[str, object]:
    jpgs = list_task_jpgs(task_dir, min_bytes=min_bytes)
    dates = sorted({d for jpg in jpgs if (d := extract_geid_capture_date(jpg))})
    return {
        "n_jpg": len(jpgs),
        "sample_chip_path": str(jpgs[0]) if jpgs else "",
        "actual_capture_dates": ";".join(dates),
        "capture_date": dates[0] if dates else "",
        "n_capture_dates": len(dates),
    }


def _normalise_anchor_id(value: object) -> str:
    return str(value or "").strip()


def infer_anchor_id_from_task(task: Mapping[str, object], anchor_index: Mapping[str, Mapping[str, object]]) -> str:
    for key in ("anchor_id", "grid_id"):
        value = _normalise_anchor_id(task.get(key))
        if value in anchor_index:
            return value
    task_name = str(task.get("task_name", "")).strip()
    match = DATE_SUFFIX_RE.match(task_name)
    if match and match.group("anchor") in anchor_index:
        return match.group("anchor")
    raise ValueError(f"cannot infer anchor_id for task {task_name!r}; expected task grid_id or prefix to match anchors.csv")


def _decision_key(row: Mapping[str, object]) -> tuple[str, str] | None:
    anchor_id = _normalise_anchor_id(row.get("anchor_id") or row.get("grid_id"))
    if not anchor_id:
        return None
    for date_col in ("requested_date", "date", "capture_date", "actual_capture_date"):
        value = str(row.get(date_col, "") or "").strip()
        if value:
            return anchor_id, value[:10]
    return None


def build_manual_decision_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], dict[str, str]]:
    """Index manually reviewed presence rows by `(anchor_id, requested/capture date)`."""
    index: dict[tuple[str, str], dict[str, str]] = {}
    for raw in rows:
        key = _decision_key(raw)
        if key is None:
            continue
        index[key] = {str(k): str(v) for k, v in raw.items()}
    return index


def _manual_for_row(
    manual_decisions: Mapping[tuple[str, str], Mapping[str, str]],
    *,
    anchor_id: str,
    requested_date: str,
    capture_date: str,
) -> Mapping[str, str] | None:
    candidates = [(anchor_id, requested_date)]
    if capture_date:
        candidates.append((anchor_id, capture_date))
    for key in candidates:
        if key in manual_decisions:
            return manual_decisions[key]
    return None


def _base_quality_and_source(summary: Mapping[str, object]) -> tuple[str, str]:
    n_jpg = int(summary.get("n_jpg") or 0)
    n_capture_dates = int(summary.get("n_capture_dates") or 0)
    if n_jpg == 0:
        return "missing_chip", "missing_chip"
    if n_capture_dates == 0:
        return "manual_template", "no_date_metadata"
    if n_capture_dates > 1:
        return "manual_template", "mixed_capture_dates"
    return "manual_template", "needs_review"


def build_presence_rows(
    anchors: Sequence[Mapping[str, object]],
    tasks: Sequence[Mapping[str, object]],
    *,
    manual_decisions: Mapping[tuple[str, str], Mapping[str, str]] | None = None,
    min_bytes: int = 1,
    include_missing: bool = True,
) -> list[dict[str, object]]:
    """Build presence time-series/template rows from anchors and GEID task rows."""
    manual_decisions = manual_decisions or {}
    anchor_index = {_normalise_anchor_id(row.get("anchor_id")): row for row in anchors if _normalise_anchor_id(row.get("anchor_id"))}
    if not anchor_index:
        raise ValueError("anchors are missing anchor_id values")

    rows: list[dict[str, object]] = []
    for task in tasks:
        anchor_id = infer_anchor_id_from_task(task, anchor_index)
        anchor = anchor_index[anchor_id]
        requested_date = str(task.get("date") or task.get("requested_date") or "").strip()[:10]
        task_name = str(task.get("task_name") or "").strip()
        save_to = str(task.get("save_to") or "").strip()
        task_dir = resolve_task_dir(save_to, task_name)
        summary = summarise_task_jpgs(task_dir, min_bytes=min_bytes)
        if int(summary["n_jpg"]) == 0 and not include_missing:
            continue

        decision_source, quality_flag = _base_quality_and_source(summary)
        row: dict[str, object] = {
            "anchor_id": anchor_id,
            "region_key": anchor.get("region_key", ""),
            "grid_id": anchor.get("grid_id", ""),
            "requested_date": requested_date,
            "capture_date": summary["capture_date"],
            "actual_capture_dates": summary["actual_capture_dates"],
            "pv_score": "",
            "pv_present": "",
            "decision_source": decision_source,
            "quality_flag": quality_flag,
            "chip_dir": str(task_dir),
            "sample_chip_path": summary["sample_chip_path"],
            "n_jpg": summary["n_jpg"],
            "task_name": task_name,
            "save_to": save_to,
            "notes": "",
        }

        manual = _manual_for_row(
            manual_decisions,
            anchor_id=anchor_id,
            requested_date=requested_date,
            capture_date=str(row["capture_date"]),
        )
        if manual is not None:
            row["decision_source"] = manual.get("decision_source") or "manual"
            for field in ("pv_present", "pv_score", "quality_flag", "notes"):
                if field in manual and str(manual.get(field, "")).strip() != "":
                    row[field] = manual[field]
            # A reviewer may have inspected the chips even if GEID metadata is
            # missing.  Preserve capture-date provenance, but make the decision
            # source explicit.
            if not row["quality_flag"]:
                row["quality_flag"] = "ok"
        rows.append(row)
    return rows


def write_qa_html(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a small local HTML review sheet with sample chips and exportable decisions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for idx, row in enumerate(rows):
        sample = str(row.get("sample_chip_path", ""))
        sample_uri = Path(sample).as_uri() if sample and Path(sample).exists() else ""
        review_rows.append(
            {
                "idx": idx,
                "anchor_id": str(row.get("anchor_id", "")),
                "requested_date": str(row.get("requested_date", "")),
                "capture_date": str(row.get("capture_date", "")),
                "quality_flag": str(row.get("quality_flag", "")),
                "n_jpg": str(row.get("n_jpg", "")),
                "sample_uri": sample_uri,
                "chip_dir": str(row.get("chip_dir", "")),
            }
        )
    rows_json = json.dumps(review_rows, ensure_ascii=False)
    fields_json = json.dumps(MANUAL_DECISION_FIELDS)
    body = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>GEID temporal PV presence QA</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:16px;color:#1f2933;background:#f7f8fa}",
        "table{border-collapse:collapse;width:100%;background:white}",
        "td,th{border:1px solid #d8dde6;padding:6px;vertical-align:top;font-size:13px}",
        "th{background:#eef2f7;text-align:left}",
        "img{max-width:260px;max-height:260px}",
        ".toolbar{display:flex;gap:8px;align-items:center;margin:12px 0}",
        ".choice-btn{margin:2px;padding:6px 8px;border:1px solid #9aa5b1;background:white;border-radius:4px;cursor:pointer}",
        ".choice-btn.active{background:#1f6feb;color:white;border-color:#1f6feb}",
        ".choice-btn[data-label='unusable'].active{background:#8a4b0f;border-color:#8a4b0f}",
        ".choice-btn[data-label='unsure'].active{background:#59636e;border-color:#59636e}",
        ".notes{width:180px;min-height:42px}",
        "#csvOut{width:100%;min-height:150px;font-family:ui-monospace,monospace;font-size:12px}",
        ".muted{color:#59636e}",
        "</style>",
        "<h1>GEID temporal PV presence QA</h1>",
        "<p>Review the historical chip near the anchor. Source masks are anchors only; judge PV visibility, or mark the chip unusable when imagery is not readable.</p>",
        "<div class='toolbar'>",
        "<button id='exportBtn'>Export labeled CSV</button>",
        "<button id='clearBtn'>Clear local labels</button>",
        "<span id='status' class='muted'></span>",
        "</div>",
        "<table>",
        "<tr><th>anchor_id</th><th>requested</th><th>capture</th><th>quality</th><th>decision</th><th>notes</th><th>n_jpg</th><th>sample</th><th>chip_dir</th></tr>",
    ]
    for row in review_rows:
        sample_html = f"<img src='{html.escape(row['sample_uri'])}'>" if row["sample_uri"] else ""
        idx = html.escape(str(row["idx"]))
        body.append(
            f"<tr data-idx='{idx}'>"
            f"<td>{html.escape(row['anchor_id'])}</td>"
            f"<td>{html.escape(row['requested_date'])}</td>"
            f"<td>{html.escape(row['capture_date'])}</td>"
            f"<td>{html.escape(row['quality_flag'])}</td>"
            "<td>"
            f"<button class='choice-btn' data-idx='{idx}' data-label='present'>Present</button>"
            f"<button class='choice-btn' data-idx='{idx}' data-label='absent'>Absent</button>"
            f"<button class='choice-btn' data-idx='{idx}' data-label='unusable'>Unusable</button>"
            f"<button class='choice-btn' data-idx='{idx}' data-label='unsure'>Unsure</button>"
            "</td>"
            f"<td><textarea class='notes' data-idx='{idx}'></textarea></td>"
            f"<td>{html.escape(row['n_jpg'])}</td>"
            f"<td>{sample_html}</td>"
            f"<td>{html.escape(row['chip_dir'])}</td>"
            "</tr>"
        )
    body.extend(
        [
            "</table>",
            "<h2>Manual decisions CSV</h2>",
            "<textarea id='csvOut' readonly></textarea>",
            "<script>",
            f"const rows = {rows_json};",
            f"const fields = {fields_json};",
            "const storageKey = 'geid-temporal-presence-qa:' + location.pathname;",
            "let decisions = JSON.parse(localStorage.getItem(storageKey) || '{}');",
            "function encodeCsv(value){",
            "  const text = String(value ?? '');",
            "  return /[\",\\n]/.test(text) ? '\"' + text.replaceAll('\"', '\"\"') + '\"' : text;",
            "}",
            "function decisionToRow(row, decision){",
            "  const out = {",
            "    anchor_id: row.anchor_id,",
            "    requested_date: row.requested_date,",
            "    capture_date: row.capture_date,",
            "    pv_present: '',",
            "    pv_score: '',",
            "    quality_flag: 'ambiguous',",
            "    decision_source: 'manual',",
            "    notes: decision.notes || ''",
            "  };",
            "  if (decision.label === 'present') { out.pv_present = '1'; out.pv_score = '1.0'; out.quality_flag = 'ok'; }",
            "  if (decision.label === 'absent') { out.pv_present = '0'; out.pv_score = '0.0'; out.quality_flag = 'ok'; }",
            "  if (decision.label === 'unusable') { out.quality_flag = 'unusable'; }",
            "  if (decision.label === 'unsure') { out.quality_flag = 'ambiguous'; }",
            "  return out;",
            "}",
            "function render(){",
            "  document.querySelectorAll('.choice-btn').forEach(btn => {",
            "    const d = decisions[btn.dataset.idx];",
            "    btn.classList.toggle('active', !!d && d.label === btn.dataset.label);",
            "  });",
            "  document.querySelectorAll('.notes').forEach(area => {",
            "    const d = decisions[area.dataset.idx];",
            "    if (document.activeElement !== area) area.value = d?.notes || '';",
            "  });",
            "  const labeled = Object.values(decisions).filter(d => d && d.label).length;",
            "  document.getElementById('status').textContent = `${labeled} labeled / ${rows.length} rows`;",
            "}",
            "function exportCsv(){",
            "  const lines = [fields.join(',')];",
            "  rows.forEach(row => {",
            "    const d = decisions[row.idx];",
            "    if (!d || !d.label) return;",
            "    const out = decisionToRow(row, d);",
            "    lines.push(fields.map(f => encodeCsv(out[f])).join(','));",
            "  });",
            "  document.getElementById('csvOut').value = lines.join('\\n') + '\\n';",
            "}",
            "document.querySelectorAll('.choice-btn').forEach(btn => {",
            "  btn.addEventListener('click', () => {",
            "    const idx = btn.dataset.idx;",
            "    decisions[idx] = decisions[idx] || {};",
            "    decisions[idx].label = btn.dataset.label;",
            "    localStorage.setItem(storageKey, JSON.stringify(decisions));",
            "    render(); exportCsv();",
            "  });",
            "});",
            "document.querySelectorAll('.notes').forEach(area => {",
            "  area.addEventListener('input', () => {",
            "    const idx = area.dataset.idx;",
            "    decisions[idx] = decisions[idx] || {};",
            "    decisions[idx].notes = area.value;",
            "    localStorage.setItem(storageKey, JSON.stringify(decisions));",
            "    exportCsv();",
            "  });",
            "});",
            "document.getElementById('exportBtn').addEventListener('click', exportCsv);",
            "document.getElementById('clearBtn').addEventListener('click', () => {",
            "  if (!confirm('Clear all local labels for this QA page?')) return;",
            "  decisions = {}; localStorage.removeItem(storageKey); render(); exportCsv();",
            "});",
            "render(); exportCsv();",
            "</script>",
            "",
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--manual-decisions-csv", type=Path, help="Optional reviewed CSV with anchor_id/date/pv_present/pv_score overrides.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qa-html", type=Path, help="Optional local HTML gallery for manual review.")
    parser.add_argument("--min-bytes", type=int, default=1024, help="Minimum JPG size to count as a valid downloaded chip tile.")
    parser.add_argument("--skip-missing", action="store_true", help="Do not emit rows for tasks with no downloaded JPGs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.anchors_csv.exists():
        raise SystemExit(f"Anchor CSV not found: {args.anchors_csv}")
    if not args.tasks_csv.exists():
        raise SystemExit(f"GEID task CSV not found: {args.tasks_csv}")
    if args.min_bytes < 1:
        raise SystemExit("--min-bytes must be >= 1")

    anchors = read_csv_rows(args.anchors_csv)
    tasks = read_csv_rows(args.tasks_csv)
    manual_decisions = {}
    if args.manual_decisions_csv:
        if not args.manual_decisions_csv.exists():
            raise SystemExit(f"Manual decisions CSV not found: {args.manual_decisions_csv}")
        manual_decisions = build_manual_decision_index(read_csv_rows(args.manual_decisions_csv))

    rows = build_presence_rows(
        anchors,
        tasks,
        manual_decisions=manual_decisions,
        min_bytes=args.min_bytes,
        include_missing=not args.skip_missing,
    )
    if not rows:
        raise SystemExit("No presence rows produced.")
    write_csv_rows(args.output, rows, PRESENCE_FIELDS)
    if args.qa_html:
        write_qa_html(args.qa_html, rows)

    status_counts = Counter(str(row.get("quality_flag", "")) for row in rows)
    source_counts = Counter(str(row.get("decision_source", "")) for row in rows)
    print(f"Wrote {len(rows)} presence rows -> {args.output}")
    if args.qa_html:
        print(f"Wrote QA gallery -> {args.qa_html}")
    print("quality_flag:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print("decision_source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    print("Policy: rows are anchor-level PV presence observations, not historical mask-IoU labels.")


if __name__ == "__main__":
    main()
