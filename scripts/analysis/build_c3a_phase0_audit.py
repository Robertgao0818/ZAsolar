#!/usr/bin/env python3
"""C-3(a) Phase 0 — audit package builder (deliverable C).

Turns the scan runner's ``proposals.csv`` + rendered chip PNGs into:

  - ``audit.csv`` : one row per background-region proposal, conforming to
    ``core.training.c3a_phase0.AUDIT_CSV_COLUMNS``.  Pre-filled with the
    proposal facts; ``audit_label`` / ``ignore_area_cap_m2`` / ``audit_notes``
    are left blank for the reviewer.
  - ``c3a_phase0_labeler.html`` : a self-contained HTML page (mirrors the
    ``label_gt_heater_audit.py`` / ``label_small_fp_taxonomy.py`` precedent)
    that renders each chip with the existing GT (green) + the low-conf
    proposal (red box) overlaid, lets a human or Gemini pick a disposition,
    and exports the labeled CSV.

The disposition vocabulary (``AUDIT_LABELS``) routes each decided proposal:
  confirmed_pv     -> promote to positive (NEVER ignore); chip is "affected"
  lookalike        -> data/negative_pool/ (HN); NEVER ignore
  ignore_candidate -> unreviewed-margin ignore region (per-chip area cap field)
  not_pv_other     -> bare roof / shadow / road etc.
  uncertain        -> abstain (does not count toward the affected rate)

The overlay draws the proposal box in chip-local pixel coords; the chip PNG is
the runner's render.  Proposals are sorted by ascending score so the smallest /
weakest detections (most likely lookalike) lead — same ergonomics as the heater
audit.

Usage
-----
    python scripts/analysis/build_c3a_phase0_audit.py \
        --run-dir results/analysis/c3a_phase0/<run_id>
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.training.c3a_phase0 import (  # noqa: E402
    AUDIT_CSV_COLUMNS,
    AUDIT_LABELS,
    make_audit_id,
    write_audit_csv,
)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>C-3(a) Phase 0 — Background-PV Audit</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
       display: flex; flex-direction: column; height: 100vh; }
.header { padding: 8px 16px; background: #16213e; display: flex;
          align-items: center; gap: 16px; flex-shrink: 0; }
.header h1 { font-size: 16px; }
.progress { font-size: 14px; color: #aaa; }
.progress .done { color: #4ecca3; font-weight: bold; }
.main { flex: 1; display: flex; overflow: hidden; }
.viewer { flex: 1; display: flex; align-items: center; justify-content: center;
          padding: 8px; position: relative; }
.viewer canvas { max-width: 100%; max-height: 100%; object-fit: contain;
              border: 2px solid #333; border-radius: 4px; image-rendering: pixelated; }
.sidebar { width: 300px; background: #16213e; padding: 12px; overflow-y: auto;
           display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
.info { font-size: 13px; line-height: 1.7; padding: 8px; background: #0f3460;
        border-radius: 6px; }
.info .val { color: #4ecca3; }
.label-btn { display: flex; align-items: center; gap: 8px; padding: 9px 12px;
             border: 1px solid #333; border-radius: 6px; cursor: pointer;
             font-size: 13px; transition: all 0.15s; background: transparent; color: #eee;
             width: 100%; text-align: left; }
.label-btn:hover { background: #0f3460; border-color: #4ecca3; }
.label-btn.active { background: #4ecca3; color: #1a1a2e; font-weight: bold;
                    border-color: #4ecca3; }
.label-btn .key { display: inline-block; width: 22px; height: 22px;
                  line-height: 22px; text-align: center; background: #333;
                  border-radius: 4px; font-weight: bold; font-size: 12px; flex-shrink: 0; }
.label-btn.active .key { background: #1a1a2e; color: #4ecca3; }
.capwrap { font-size: 12px; color: #ccc; display: none; gap: 6px; align-items: center;
           padding: 6px 8px; background: #0a0a1a; border-radius: 4px; }
.capwrap.show { display: flex; }
.capwrap input { width: 90px; background: #16213e; color: #eee; border: 1px solid #333;
                 border-radius: 4px; padding: 4px; }
.nav { display: flex; gap: 8px; margin-top: auto; padding-top: 8px; }
.nav button { flex: 1; padding: 8px; border: 1px solid #333; border-radius: 6px;
              background: #0f3460; color: #eee; cursor: pointer; font-size: 13px; }
.nav button:hover { background: #4ecca3; color: #1a1a2e; }
.export-btn { padding: 10px; border: none; border-radius: 6px; background: #e94560;
              color: #fff; cursor: pointer; font-size: 14px; font-weight: bold; margin-top: 4px; }
.export-btn:hover { background: #c73e54; }
.hint { font-size: 11px; color: #666; text-align: center; margin-top: 4px; }
.badge { position: absolute; top: 14px; left: 14px; padding: 4px 12px;
         border-radius: 12px; font-size: 13px; font-weight: bold; }
.badge.labeled { background: #4ecca3; color: #1a1a2e; }
.badge.unlabeled { background: #e94560; color: #fff; }
.legend { position: absolute; bottom: 14px; left: 14px; font-size: 12px;
          background: rgba(10,10,26,0.8); padding: 4px 10px; border-radius: 6px; }
.legend .gt { color: #4ecca3; } .legend .prop { color: #ff5b6e; }
.stats { font-size: 12px; color: #888; padding: 6px 8px; background: #0a0a1a;
         border-radius: 4px; text-align: center; }
</style>
</head>
<body>
<div class="header">
  <h1>C-3(a) Phase 0 背景区未标 PV 审计</h1>
  <div class="progress">
    <span class="done" id="labeledCount">0</span> / <span id="totalCount">0</span> 已裁决
    &nbsp;|&nbsp; 当前 <span id="currentIdx">1</span>
  </div>
</div>
<div class="main">
  <div class="viewer">
    <canvas id="chipCanvas" width="400" height="400"></canvas>
    <div class="badge" id="badge"></div>
    <div class="legend"><span class="gt">■ 已有 GT</span> &nbsp; <span class="prop">▭ 低 conf 提议</span></div>
  </div>
  <div class="sidebar">
    <div class="info" id="chipInfo"></div>
    <div id="labelButtons"></div>
    <div class="capwrap" id="capWrap">
      ignore-area cap (m²): <input id="capInput" type="number" step="1" min="0" />
    </div>
    <div class="stats" id="statsBar"></div>
    <div class="nav">
      <button onclick="prev()">&larr; B 回退</button>
      <button onclick="skip()">S 跳过 &rarr;</button>
    </div>
    <button class="export-btn" onclick="exportCSV()">导出 audit.csv</button>
    <div class="hint">1=真PV→转正 2=lookalike→negpool 3=ignore候选 4=非PV其它 5=不确定 S=跳过 B=回退</div>
  </div>
</div>
<script>
const LABELS = %%LABELS_JSON%%;
const ROWS = %%ROWS_JSON%%;
const COLS = %%COLS_JSON%%;
let idx = 0;
for (let i = 0; i < ROWS.length; i++) { if (!ROWS[i].audit_label) { idx = i; break; } }

function draw() {
  const c = ROWS[idx];
  const canvas = document.getElementById("chipCanvas");
  const cs = c.chip_size || 400;
  canvas.width = cs; canvas.height = cs;
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, cs, cs);
    ctx.drawImage(img, 0, 0, cs, cs);
    // GT (green) — drawn from gt_boxes_chip if present.
    ctx.lineWidth = 2; ctx.strokeStyle = "#4ecca3";
    (c.gt_boxes_chip || []).forEach(b => ctx.strokeRect(b[0], b[1], b[2]-b[0], b[3]-b[1]));
    // proposal (red)
    ctx.lineWidth = 3; ctx.strokeStyle = "#ff5b6e";
    ctx.strokeRect(c.box_chip_x0, c.box_chip_y0,
                   c.box_chip_x1 - c.box_chip_x0, c.box_chip_y1 - c.box_chip_y0);
  };
  img.src = "data:image/png;base64," + (c._img || "");
}

function render() {
  const c = ROWS[idx];
  document.getElementById("currentIdx").textContent = idx + 1;
  document.getElementById("totalCount").textContent = ROWS.length;
  document.getElementById("labeledCount").textContent =
    ROWS.filter(x => x.audit_label).length;
  document.getElementById("chipInfo").innerHTML =
    `<b>Chip:</b> <span class="val">${c.grid_id} / ${c.tile_stem}</span><br>` +
    `<b>Region:</b> <span class="val">${c.region} / ${c.imagery_layer}</span><br>` +
    `<b>Proposal:</b> <span class="val">#${c.proposal_index}</span><br>` +
    `<b>Score:</b> <span class="val">${c.score}</span><br>` +
    `<b>Area:</b> <span class="val">${c.proposal_area_m2} m²</span><br>` +
    `<b>max IoF vs GT:</b> <span class="val">${c.max_iof_vs_gt}</span><br>` +
    `<b>GT in chip:</b> <span class="val">${c.n_gt_in_chip}</span>`;
  const badge = document.getElementById("badge");
  if (c.audit_label) {
    const lbl = LABELS.find(l => l[1] === c.audit_label);
    badge.textContent = lbl ? lbl[2] : c.audit_label; badge.className = "badge labeled";
  } else { badge.textContent = "未裁决"; badge.className = "badge unlabeled"; }
  const container = document.getElementById("labelButtons");
  container.innerHTML = "";
  for (const [key, en, zh] of LABELS) {
    const btn = document.createElement("button");
    btn.className = "label-btn" + (c.audit_label === en ? " active" : "");
    btn.innerHTML = `<span class="key">${key}</span> ${zh}`;
    btn.onclick = () => applyLabel(en);
    container.appendChild(btn);
  }
  const capWrap = document.getElementById("capWrap");
  const capInput = document.getElementById("capInput");
  if (c.audit_label === "ignore_candidate") {
    capWrap.classList.add("show");
    capInput.value = c.ignore_area_cap_m2 || "";
    capInput.oninput = () => { c.ignore_area_cap_m2 = capInput.value; };
  } else { capWrap.classList.remove("show"); }
  const cnt = en => ROWS.filter(x => x.audit_label === en).length;
  document.getElementById("statsBar").textContent =
    `PV:${cnt("confirmed_pv")} | lookalike:${cnt("lookalike")} | ignore:${cnt("ignore_candidate")} | 其它:${cnt("not_pv_other")} | ?:${cnt("uncertain")}`;
  draw();
}

function applyLabel(label) {
  ROWS[idx].audit_label = label;
  ROWS[idx].reviewed_at = new Date().toISOString().slice(0, 19);
  if (label !== "ignore_candidate") { ROWS[idx].ignore_area_cap_m2 = ""; }
  if (label !== "ignore_candidate" && idx < ROWS.length - 1) idx++;
  render();
}
function skip() { if (idx < ROWS.length - 1) { idx++; render(); } }
function prev() { if (idx > 0) { idx--; render(); } }
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key;
  if (k >= "1" && k <= "5") applyLabel(LABELS[parseInt(k) - 1][1]);
  else if (k.toLowerCase() === "s") skip();
  else if (k.toLowerCase() === "b") prev();
});
function exportCSV() {
  let csv = COLS.join(",") + "\n";
  for (const c of ROWS) {
    csv += COLS.map(k => {
      let v = c[k]; if (v === undefined || v === null) v = ""; v = String(v);
      if (v.includes(",") || v.includes('"')) v = '"' + v.replace(/"/g, '""') + '"';
      return v;
    }).join(",") + "\n";
  }
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "audit.csv"; a.click();
}
render();
</script>
</body>
</html>"""


def _load_proposals(run_dir: Path) -> list[dict]:
    path = run_dir / "proposals.csv"
    if not path.exists():
        sys.exit(f"[audit] missing {path} — run the scan runner (extract phase) first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_gt_chip_boxes(run_dir: Path) -> dict[str, list[list[float]]]:
    """Return {chip_uid: [[x0,y0,x1,y1], ...]} GT boxes in chip-local pixels.

    Reconstructed from gt_refs gpkg + chip_manifest pixel windows so the HTML
    overlay can draw existing GT.  Best-effort: if geopandas/rasterio or the GT
    gpkgs are absent, returns empty boxes (overlay still shows the proposal).
    """
    try:
        import geopandas as gpd
        import rasterio
        from export_coco_dataset import polygon_to_pixel_coords
    except Exception:  # noqa: BLE001
        return {}

    manifest = run_dir / "chip_manifest.csv"
    if not manifest.exists():
        return {}
    with open(manifest, newline="", encoding="utf-8") as f:
        chip_rows = {r["chip_uid"]: r for r in csv.DictReader(f)}

    transform_cache: dict[str, object] = {}

    def _t(tile_path: str):
        if tile_path not in transform_cache:
            with rasterio.open(tile_path) as src:
                transform_cache[tile_path] = src.transform
        return transform_cache[tile_path]

    out: dict[str, list[list[float]]] = {}
    for gpkg in sorted(run_dir.glob("gt_refs__*.gpkg")):
        gdf = gpd.read_file(gpkg)
        for _, row in gdf.iterrows():
            chip_uid = row["chip_uid"]
            cr = chip_rows.get(chip_uid)
            if cr is None:
                continue
            x0, y0 = int(cr["x0"]), int(cr["y0"])
            pgeom = polygon_to_pixel_coords(row.geometry, _t(cr["tile_path"]))
            if pgeom.is_empty:
                continue
            bx0, by0, bx1, by1 = pgeom.bounds
            out.setdefault(chip_uid, []).append(
                [bx0 - x0, by0 - y0, bx1 - x0, by1 - y0]
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="C-3(a) Phase 0 audit package builder")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap proposals (sorted by ascending score)")
    ap.add_argument("--no-html", action="store_true",
                    help="write audit.csv only (skip HTML embedding)")
    args = ap.parse_args()

    run_dir = args.run_dir
    proposals = _load_proposals(run_dir)
    proposals.sort(key=lambda r: float(r.get("score") or 0.0))
    if args.limit:
        proposals = proposals[: args.limit]
    print(f"[audit] {len(proposals)} proposals")

    gt_boxes = _load_gt_chip_boxes(run_dir)

    # Build audit rows (schema-conformant) + the empty decision fields.
    audit_rows = []
    for p in proposals:
        audit_id = make_audit_id(p["chip_uid"], int(p["proposal_index"]))
        audit_rows.append({
            "audit_id": audit_id,
            "chip_uid": p["chip_uid"],
            "region": p["region"],
            "imagery_layer": p["imagery_layer"],
            "grid_id": p["grid_id"],
            "tile_stem": p["tile_stem"],
            "x0": p["x0"],
            "y0": p["y0"],
            "chip_size": p["chip_size"],
            "proposal_index": p["proposal_index"],
            "score": p["score"],
            "proposal_area_m2": p["proposal_area_m2"],
            "max_iof_vs_gt": p["max_iof_vs_gt"],
            "n_gt_in_chip": p["n_gt_in_chip"],
            "chip_png": p["chip_png"],
            "audit_label": "",
            "ignore_area_cap_m2": "",
            "audit_notes": "",
            "reviewed_at": "",
        })

    audit_csv_path = run_dir / "audit.csv"
    write_audit_csv(audit_rows, audit_csv_path)
    print(f"[audit] wrote {audit_csv_path} ({len(audit_rows)} rows, "
          f"schema={len(AUDIT_CSV_COLUMNS)} cols)")

    if args.no_html:
        return 0

    # Build the HTML rows with embedded chip PNGs + GT/proposal boxes.
    html_rows = []
    n_missing_png = 0
    for p, ar in zip(proposals, audit_rows):
        png_path = run_dir / p["chip_png"]
        img_b64 = ""
        if png_path.exists():
            img_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
        else:
            n_missing_png += 1
        row = dict(ar)
        row["_img"] = img_b64
        row["box_chip_x0"] = float(p["box_chip_x0"])
        row["box_chip_y0"] = float(p["box_chip_y0"])
        row["box_chip_x1"] = float(p["box_chip_x1"])
        row["box_chip_y1"] = float(p["box_chip_y1"])
        row["gt_boxes_chip"] = gt_boxes.get(p["chip_uid"], [])
        html_rows.append(row)

    labels_json = json.dumps([list(t) for t in AUDIT_LABELS], ensure_ascii=False)
    rows_json = json.dumps(html_rows, ensure_ascii=False)
    cols_json = json.dumps(list(AUDIT_CSV_COLUMNS), ensure_ascii=False)
    html = (HTML_TEMPLATE
            .replace("%%LABELS_JSON%%", labels_json)
            .replace("%%ROWS_JSON%%", rows_json)
            .replace("%%COLS_JSON%%", cols_json))
    html_path = run_dir / "c3a_phase0_labeler.html"
    html_path.write_text(html, encoding="utf-8")
    size_mb = html_path.stat().st_size / 1024 / 1024
    print(f"[audit] wrote {html_path} ({size_mb:.1f} MB)")
    if n_missing_png:
        print(f"[audit][WARN] {n_missing_png} proposals had no chip PNG "
              f"(run the scan runner extract phase to render chips)")
    wsl = str(html_path.resolve())
    if wsl.startswith("/home/"):
        win = wsl.replace("/home/", "\\\\wsl$\\Ubuntu\\home\\").replace("/", "\\")
        print(f"[audit]   Windows: {win}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
