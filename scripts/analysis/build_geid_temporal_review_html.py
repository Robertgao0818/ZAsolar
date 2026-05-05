#!/usr/bin/env python3
"""Build a self-contained interactive GEID temporal presence review page.

This wraps the corrected temporal QA chips into a single HTML file with embedded
images.  The page can be opened directly from disk because every JPEG is encoded
as a data URI; no local HTTP server is required.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PRESENCE = PROJECT_ROOT / "data" / "geid_temporal" / "jhb_vexcel10_smoke" / "presence_timeseries_labeled.csv"
DEFAULT_QA_DIR = (
    PROJECT_ROOT
    / "data"
    / "geid_temporal"
    / "jhb_vexcel10_smoke"
    / "temporal_stack_qa_corrected"
    / "geid_temporal_qa_20260505_150621"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "geid_temporal" / "jhb_vexcel10_smoke" / "temporal_presence_review_embedded.html"

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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def label_from_row(row: Mapping[str, str]) -> str:
    present = str(row.get("pv_present", "")).strip().lower()
    quality = str(row.get("quality_flag", "")).strip().lower()
    if present in {"1", "true", "yes", "present"}:
        return "present"
    if present in {"0", "false", "no", "absent"}:
        return "absent"
    if quality == "unusable":
        return "unusable"
    if quality == "ambiguous":
        return "unsure"
    return ""


def decision_from_label(label: str, notes: str) -> dict[str, str]:
    out = {
        "pv_present": "",
        "pv_score": "",
        "quality_flag": "ambiguous",
        "decision_source": "manual_embedded_temporal_review",
        "notes": notes,
    }
    if label == "present":
        out["pv_present"] = "1"
        out["pv_score"] = "1.0"
        out["quality_flag"] = "ok"
    elif label == "absent":
        out["pv_present"] = "0"
        out["pv_score"] = "0.0"
        out["quality_flag"] = "ok"
    elif label == "unusable":
        out["quality_flag"] = "unusable"
    return out


def build_review_rows(presence_rows: Sequence[Mapping[str, str]], qa_dir: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for idx, row in enumerate(presence_rows):
        anchor_id = str(row.get("anchor_id", "")).strip()
        task_name = str(row.get("task_name", "")).strip()
        requested_date = str(row.get("requested_date", "")).strip()
        capture_date = str(row.get("capture_date", "")).strip()
        if not anchor_id or not task_name:
            continue

        aerial_path = qa_dir / f"{anchor_id}_aerial.jpg"
        chip_path = qa_dir / f"{anchor_id}_{task_name}.jpg"
        out.append(
            {
                "idx": idx,
                "key": f"{anchor_id}|{requested_date}",
                "anchor_id": anchor_id,
                "requested_date": requested_date,
                "capture_date": capture_date,
                "quality_flag": str(row.get("quality_flag", "")),
                "n_jpg": str(row.get("n_jpg", "")),
                "task_name": task_name,
                "chip_dir": str(row.get("chip_dir", "")),
                "aerial_image": image_data_uri(aerial_path) if aerial_path.exists() else "",
                "historical_image": image_data_uri(chip_path) if chip_path.exists() else "",
                "label": label_from_row(row),
                "notes": str(row.get("notes", "")),
            }
        )
    return out


def render_html(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["anchor_id"])].append(row)

    js_rows = [
        {
            key: row.get(key, "")
            for key in (
                "idx",
                "key",
                "anchor_id",
                "requested_date",
                "capture_date",
                "quality_flag",
                "n_jpg",
                "task_name",
                "chip_dir",
                "label",
                "notes",
            )
        }
        for row in rows
    ]
    rows_json = json.dumps(js_rows, ensure_ascii=False)
    fields_json = json.dumps(MANUAL_DECISION_FIELDS)
    grouped_keys = sorted(grouped)

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>GEID Temporal Presence Review</title>",
        "<style>",
        ":root{color-scheme:dark;--bg:#111;--panel:#1a1d21;--line:#333941;--text:#f2f4f7;--muted:#a5acb8;--blue:#2f81f7;--green:#238636;--red:#da3633;--amber:#9e6a03}",
        "*{box-sizing:border-box}",
        "body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}",
        ".top{position:sticky;top:0;z-index:10;background:#0d1117;border-bottom:1px solid var(--line);padding:10px 14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}",
        "h1{font-size:17px;margin:0 12px 0 0}.muted{color:var(--muted)}",
        "button{border:1px solid #59636e;background:#20252c;color:var(--text);border-radius:5px;padding:6px 9px;cursor:pointer;font-size:12px}",
        "button:hover{border-color:#8b949e}.choice.active[data-label=present]{background:var(--green);border-color:var(--green)}",
        ".choice.active[data-label=absent]{background:var(--red);border-color:var(--red)}.choice.active[data-label=unsure]{background:#59636e;border-color:#59636e}",
        ".choice.active[data-label=unusable]{background:var(--amber);border-color:var(--amber)}",
        ".wrap{padding:12px 14px 32px}.anchor{border:1px solid var(--line);background:var(--panel);border-radius:7px;margin:0 0 14px;overflow:hidden}",
        ".anchor-head{padding:8px 10px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px;align-items:center}",
        ".anchor-title{font-weight:700;font-size:14px}.status{font-size:12px;color:var(--muted)}",
        ".panels{display:grid;grid-template-columns:220px repeat(5,minmax(190px,1fr));gap:8px;padding:10px;overflow-x:auto}",
        ".panel{min-width:190px}.panel.ref{min-width:220px}",
        ".panel-title{height:34px;font-size:12px;line-height:1.25;color:var(--muted);display:flex;align-items:flex-end}",
        ".panel img{width:100%;aspect-ratio:1/1;object-fit:contain;background:#090b0d;border:1px solid #30363d;border-radius:4px;display:block}",
        ".actions{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:6px}.actions button{padding:5px 4px}",
        "textarea{width:100%;min-height:48px;margin-top:6px;background:#0d1117;color:var(--text);border:1px solid #30363d;border-radius:4px;padding:6px;font-family:inherit;font-size:12px}",
        ".bad{color:#ff7b72}.cap{color:#f0c36d;font-weight:600}.missing{height:100%;min-height:190px;border:1px dashed #59636e;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#ff7b72}",
        "#csvOut{width:100%;min-height:190px;margin-top:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}",
        "details{margin-top:10px}summary{cursor:pointer;color:var(--muted)}",
        "</style></head><body>",
        "<div class='top'>",
        "<h1>GEID Temporal Presence Review</h1>",
        "<button id='exportBtn'>Refresh CSV</button>",
        "<button id='clearBtn'>Clear Local Edits</button>",
        "<span id='status' class='muted'></span>",
        "</div>",
        "<div class='wrap'>",
    ]

    for anchor_id in grouped_keys:
        anchor_rows = sorted(grouped[anchor_id], key=lambda r: str(r["requested_date"]))
        first = anchor_rows[0]
        parts.extend(
            [
                f"<section class='anchor' data-anchor='{html.escape(anchor_id)}'>",
                "<div class='anchor-head'>",
                f"<div class='anchor-title'>{html.escape(anchor_id)}</div>",
                f"<div class='status' data-anchor-status='{html.escape(anchor_id)}'></div>",
                "</div>",
                "<div class='panels'>",
                "<div class='panel ref'>",
                "<div class='panel-title'>Vexcel 2024-02 reference</div>",
            ]
        )
        if first.get("aerial_image"):
            parts.append(f"<img src='{first['aerial_image']}' alt='Vexcel reference'>")
        else:
            parts.append("<div class='missing'>missing reference</div>")
        parts.append("</div>")

        for row in anchor_rows:
            idx = html.escape(str(row["idx"]))
            requested = html.escape(str(row["requested_date"]))
            capture = html.escape(str(row["capture_date"]))
            drift = ""
            if requested[:4] and capture[:4] and requested[:4] != capture[:4]:
                drift = "<span class='bad'> date drift</span>"
            parts.extend(
                [
                    f"<div class='panel' data-idx='{idx}'>",
                    f"<div class='panel-title'>req {requested}<br>capture <span class='cap'>{capture}</span>{drift}</div>",
                ]
            )
            if row.get("historical_image"):
                parts.append(f"<img src='{row['historical_image']}' alt='historical chip {requested}'>")
            else:
                parts.append("<div class='missing'>missing chip</div>")
            parts.extend(
                [
                    "<div class='actions'>",
                    f"<button class='choice' data-idx='{idx}' data-label='present'>Present</button>",
                    f"<button class='choice' data-idx='{idx}' data-label='absent'>Absent</button>",
                    f"<button class='choice' data-idx='{idx}' data-label='unsure'>Unsure</button>",
                    f"<button class='choice' data-idx='{idx}' data-label='unusable'>Unusable</button>",
                    "</div>",
                    f"<textarea class='notes' data-idx='{idx}'></textarea>",
                    "</div>",
                ]
            )
        parts.extend(["</div>", "</section>"])

    parts.extend(
        [
            "<details open><summary>Manual decisions CSV</summary>",
            "<textarea id='csvOut' readonly></textarea>",
            "</details>",
            "</div>",
            "<script>",
            f"const rows = {rows_json};",
            f"const fields = {fields_json};",
            "const storageKey = 'geid-temporal-embedded-review:' + location.pathname;",
            "const initialDecisions = Object.fromEntries(rows.map(row => [row.idx, {label: row.label || '', notes: row.notes || ''}]));",
            "let decisions = {...initialDecisions, ...JSON.parse(localStorage.getItem(storageKey) || '{}')};",
            "function encodeCsv(value){const text=String(value ?? ''); return /[\",\\n]/.test(text) ? '\"' + text.replaceAll('\"','\"\"') + '\"' : text;}",
            "function decisionToRow(row, decision){",
            "  const out = {anchor_id: row.anchor_id, requested_date: row.requested_date, capture_date: row.capture_date, pv_present:'', pv_score:'', quality_flag:'ambiguous', decision_source:'manual_embedded_temporal_review', notes: decision.notes || ''};",
            "  if (decision.label === 'present') { out.pv_present='1'; out.pv_score='1.0'; out.quality_flag='ok'; }",
            "  if (decision.label === 'absent') { out.pv_present='0'; out.pv_score='0.0'; out.quality_flag='ok'; }",
            "  if (decision.label === 'unusable') { out.quality_flag='unusable'; }",
            "  return out;",
            "}",
            "function render(){",
            "  document.querySelectorAll('.choice').forEach(btn => { const d=decisions[btn.dataset.idx]; btn.classList.toggle('active', !!d && d.label === btn.dataset.label); });",
            "  document.querySelectorAll('.notes').forEach(area => { const d=decisions[area.dataset.idx]; if (document.activeElement !== area) area.value = d?.notes || ''; });",
            "  const labeled = rows.filter(row => decisions[row.idx]?.label).length;",
            "  document.getElementById('status').textContent = `${labeled} labeled / ${rows.length} observations`;",
            "  document.querySelectorAll('[data-anchor-status]').forEach(el => {",
            "    const anchor = el.dataset.anchorStatus;",
            "    const subset = rows.filter(row => row.anchor_id === anchor);",
            "    const counts = {present:0, absent:0, unsure:0, unusable:0, blank:0};",
            "    subset.forEach(row => { const label = decisions[row.idx]?.label || 'blank'; counts[label] = (counts[label] || 0) + 1; });",
            "    el.textContent = `present ${counts.present} | absent ${counts.absent} | unsure ${counts.unsure} | unusable ${counts.unusable}`;",
            "  });",
            "}",
            "function exportCsv(){",
            "  const lines = [fields.join(',')];",
            "  rows.forEach(row => { const d=decisions[row.idx]; if (!d || !d.label) return; const out=decisionToRow(row,d); lines.push(fields.map(f => encodeCsv(out[f])).join(',')); });",
            "  document.getElementById('csvOut').value = lines.join('\\n') + '\\n';",
            "}",
            "document.querySelectorAll('.choice').forEach(btn => btn.addEventListener('click', () => {",
            "  const idx=btn.dataset.idx; decisions[idx]=decisions[idx] || {}; decisions[idx].label=btn.dataset.label;",
            "  localStorage.setItem(storageKey, JSON.stringify(decisions)); render(); exportCsv();",
            "}));",
            "document.querySelectorAll('.notes').forEach(area => area.addEventListener('input', () => {",
            "  const idx=area.dataset.idx; decisions[idx]=decisions[idx] || {}; decisions[idx].notes=area.value;",
            "  localStorage.setItem(storageKey, JSON.stringify(decisions)); exportCsv();",
            "}));",
            "document.getElementById('exportBtn').addEventListener('click', exportCsv);",
            "document.getElementById('clearBtn').addEventListener('click', () => { if (!confirm('Clear local edits and return to embedded defaults?')) return; decisions={...initialDecisions}; localStorage.removeItem(storageKey); render(); exportCsv(); });",
            "render(); exportCsv();",
            "</script></body></html>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--presence-csv", type=Path, default=DEFAULT_PRESENCE)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.presence_csv.exists():
        raise SystemExit(f"presence CSV not found: {args.presence_csv}")
    if not args.qa_dir.exists():
        raise SystemExit(f"QA image dir not found: {args.qa_dir}")
    rows = build_review_rows(read_csv_rows(args.presence_csv), args.qa_dir)
    if not rows:
        raise SystemExit("No review rows generated.")
    missing_images = sum(1 for row in rows if not row["historical_image"])
    render_html(rows, args.output)
    print(f"Wrote {len(rows)} observations -> {args.output}")
    if missing_images:
        print(f"WARNING: {missing_images} historical chips were missing.")


if __name__ == "__main__":
    main()
