#!/usr/bin/env python3
"""
GT 加热器审计标注工具 — 生成自包含 HTML 标注页面。

输入 build_gt_heater_audit.py 产出的 audit_queue_phase1.csv + chips，
生成 HTML 页面供人工标注 pv / heater_or_non_pv / uncertain。

用法：
    python scripts/analysis/label_gt_heater_audit.py \
        --run-dir results/analysis/gt_heater_audit/<run_id>

    # 使用全量队列
    python scripts/analysis/label_gt_heater_audit.py \
        --run-dir results/analysis/gt_heater_audit/<run_id> \
        --queue audit_queue_full.csv

    # 限制数量
    python scripts/analysis/label_gt_heater_audit.py \
        --run-dir results/analysis/gt_heater_audit/<run_id> --limit 200
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import pandas as pd

LABELS = [
    ("1", "pv", "光伏板 (PV)"),
    ("2", "heater_or_non_pv", "热水器/泳池加热器/非PV"),
    ("3", "uncertain", "不确定"),
]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>GT Heater Audit Labeler</title>
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
.viewer img { max-width: 100%; max-height: 100%; object-fit: contain;
              border: 2px solid #333; border-radius: 4px; }
.sidebar { width: 280px; background: #16213e; padding: 12px; overflow-y: auto;
           display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
.info { font-size: 13px; line-height: 1.8; padding: 8px; background: #0f3460;
        border-radius: 6px; }
.info .val { color: #4ecca3; }
.label-btn { display: flex; align-items: center; gap: 8px; padding: 10px 12px;
             border: 1px solid #333; border-radius: 6px; cursor: pointer;
             font-size: 14px; transition: all 0.15s; background: transparent; color: #eee;
             width: 100%; text-align: left; }
.label-btn:hover { background: #0f3460; border-color: #4ecca3; }
.label-btn.active { background: #4ecca3; color: #1a1a2e; font-weight: bold;
                    border-color: #4ecca3; }
.label-btn .key { display: inline-block; width: 24px; height: 24px;
                  line-height: 24px; text-align: center; background: #333;
                  border-radius: 4px; font-weight: bold; font-size: 13px; flex-shrink: 0; }
.label-btn.active .key { background: #1a1a2e; color: #4ecca3; }
.nav { display: flex; gap: 8px; margin-top: auto; padding-top: 8px; }
.nav button { flex: 1; padding: 8px; border: 1px solid #333; border-radius: 6px;
              background: #0f3460; color: #eee; cursor: pointer; font-size: 13px; }
.nav button:hover { background: #4ecca3; color: #1a1a2e; }
.export-btn { padding: 10px; border: none; border-radius: 6px; background: #e94560;
              color: #fff; cursor: pointer; font-size: 14px; font-weight: bold;
              margin-top: 4px; }
.export-btn:hover { background: #c73e54; }
.hint { font-size: 11px; color: #666; text-align: center; margin-top: 4px; }
.badge { position: absolute; top: 16px; left: 16px; padding: 4px 12px;
         border-radius: 12px; font-size: 13px; font-weight: bold; }
.badge.labeled { background: #4ecca3; color: #1a1a2e; }
.badge.unlabeled { background: #e94560; color: #fff; }
.stats { font-size: 12px; color: #888; padding: 6px 8px; background: #0a0a1a;
         border-radius: 4px; text-align: center; }
</style>
</head>
<body>
<div class="header">
  <h1>GT Heater Audit</h1>
  <div class="progress">
    <span class="done" id="labeledCount">0</span> / <span id="totalCount">0</span> 已标注
    &nbsp;|&nbsp; 当前 <span id="currentIdx">1</span>
  </div>
</div>
<div class="main">
  <div class="viewer">
    <img id="chipImg" src="" />
    <div class="badge" id="badge"></div>
  </div>
  <div class="sidebar">
    <div class="info" id="chipInfo"></div>
    <div id="labelButtons"></div>
    <div class="stats" id="statsBar"></div>
    <div class="nav">
      <button onclick="prev()">&larr; B 回退</button>
      <button onclick="skip()">S 跳过 &rarr;</button>
    </div>
    <button class="export-btn" onclick="exportCSV()">导出 CSV</button>
    <div class="hint">快捷键: 1=PV  2=热水器/非PV  3=不确定  S=跳过  B=回退</div>
  </div>
</div>
<script>
const LABELS = %%LABELS_JSON%%;
const CHIPS = %%CHIPS_JSON%%;
let idx = 0;

// Find first unlabeled
for (let i = 0; i < CHIPS.length; i++) {
  if (!CHIPS[i].audit_label) { idx = i; break; }
}

function render() {
  const c = CHIPS[idx];
  document.getElementById("chipImg").src = "data:image/png;base64," + c.img;
  document.getElementById("currentIdx").textContent = idx + 1;
  document.getElementById("totalCount").textContent = CHIPS.length;

  const labeled = CHIPS.filter(x => x.audit_label).length;
  document.getElementById("labeledCount").textContent = labeled;

  document.getElementById("chipInfo").innerHTML =
    `<b>Grid:</b> <span class="val">${c.grid_id}</span><br>` +
    `<b>ID:</b> <span class="val">${c.audit_id}</span><br>` +
    `<b>Area:</b> <span class="val">${c.area_m2.toFixed(1)} m²</span><br>` +
    `<b>Elongation:</b> <span class="val">${c.elongation.toFixed(2)}</span><br>` +
    `<b>Solidity:</b> <span class="val">${c.solidity.toFixed(3)}</span><br>` +
    `<b>Bucket:</b> <span class="val">${c.geometry_bucket}</span><br>` +
    `<b>Tier:</b> <span class="val">${c.risk_tier}</span><br>` +
    (c.confidence ? `<b>Conf:</b> <span class="val">${c.confidence.toFixed(3)}</span><br>` : "") +
    `<b>Source:</b> <span class="val">${c.source}</span>`;

  // Badge
  const badge = document.getElementById("badge");
  if (c.audit_label) {
    const lbl = LABELS.find(l => l[1] === c.audit_label);
    badge.textContent = lbl ? lbl[2] : c.audit_label;
    badge.className = "badge labeled";
  } else {
    badge.textContent = "未标注";
    badge.className = "badge unlabeled";
  }

  // Buttons
  const container = document.getElementById("labelButtons");
  container.innerHTML = "";
  for (const [key, en, zh] of LABELS) {
    const btn = document.createElement("button");
    btn.className = "label-btn" + (c.audit_label === en ? " active" : "");
    btn.innerHTML = `<span class="key">${key}</span> ${zh}`;
    btn.onclick = () => applyLabel(en);
    container.appendChild(btn);
  }

  // Stats bar
  const pvCount = CHIPS.filter(x => x.audit_label === "pv").length;
  const heaterCount = CHIPS.filter(x => x.audit_label === "heater_or_non_pv").length;
  const uncCount = CHIPS.filter(x => x.audit_label === "uncertain").length;
  document.getElementById("statsBar").textContent =
    `PV: ${pvCount} | 热水器: ${heaterCount} | 不确定: ${uncCount}`;
}

function applyLabel(label) {
  CHIPS[idx].audit_label = label;
  CHIPS[idx].reviewed_at = new Date().toISOString().slice(0, 19);
  if (idx < CHIPS.length - 1) idx++;
  render();
}

function skip() { if (idx < CHIPS.length - 1) { idx++; render(); } }
function prev() { if (idx > 0) { idx--; render(); } }

document.addEventListener("keydown", e => {
  const k = e.key;
  if (k >= "1" && k <= "3") { applyLabel(LABELS[parseInt(k) - 1][1]); }
  else if (k.toLowerCase() === "s") { skip(); }
  else if (k.toLowerCase() === "b") { prev(); }
});

function exportCSV() {
  const cols = ["audit_id","grid_id","row_index","source_file","area_m2","elongation",
                "solidity","geometry_bucket","risk_tier","confidence","review_status",
                "source","chip_path","overlay_path","audit_label","audit_notes","reviewed_at"];
  let csv = cols.join(",") + "\n";
  for (const c of CHIPS) {
    csv += cols.map(k => {
      let v = c[k];
      if (v === undefined || v === null) v = "";
      v = String(v);
      if (v.includes(",") || v.includes('"')) v = '"' + v.replace(/"/g, '""') + '"';
      return v;
    }).join(",") + "\n";
  }
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gt_heater_audit_labeled.csv";
  a.click();
}

render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="生成 GT heater audit HTML 标注页面")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="build_gt_heater_audit.py 的输出目录")
    parser.add_argument("--queue", default="audit_queue_phase1.csv",
                        help="队列 CSV 文件名（默认 audit_queue_phase1.csv）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只生成前 N 张（按 area_m2 升序）")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 HTML 路径（默认 run-dir/gt_heater_audit_labeler.html）")
    args = parser.parse_args()

    queue_path = args.run_dir / args.queue
    if not queue_path.exists():
        print(f"未找到 {queue_path}")
        sys.exit(1)

    df = pd.read_csv(queue_path)
    df["audit_label"] = df["audit_label"].fillna("")
    df["audit_notes"] = df["audit_notes"].fillna("")
    df["reviewed_at"] = df["reviewed_at"].fillna("")

    # Sort by area ascending (smallest = most likely heater)
    df = df.sort_values("area_m2", ascending=True).reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit)

    # Filter to only records with chip images
    df = df[df["chip_path"].notna() & (df["chip_path"] != "")].reset_index(drop=True)

    print(f"准备 {len(df)} 张 chip...")

    # Build chip data with embedded base64 images
    chips = []
    for _, row in df.iterrows():
        chip_rel = row["chip_path"]
        img_path = args.run_dir / chip_rel
        if not img_path.exists():
            # Try overlay and rgb subdirs
            overlay_path = args.run_dir / "chips" / "overlay" / f"{row['audit_id']}.png"
            rgb_path = args.run_dir / "chips" / "rgb" / f"{row['audit_id']}.png"
            if overlay_path.exists():
                img_path = overlay_path
            elif rgb_path.exists():
                img_path = rgb_path
            else:
                continue

        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        chips.append({
            "audit_id": row["audit_id"],
            "grid_id": row["grid_id"],
            "row_index": int(row["row_index"]),
            "source_file": row.get("source_file", ""),
            "area_m2": float(row["area_m2"]),
            "elongation": float(row["elongation"]),
            "solidity": float(row["solidity"]),
            "geometry_bucket": row.get("geometry_bucket", ""),
            "risk_tier": row.get("risk_tier", ""),
            "confidence": float(row["confidence"]) if row.get("confidence", "") != "" else 0,
            "review_status": row.get("review_status", ""),
            "source": row.get("source", ""),
            "chip_path": row.get("chip_path", ""),
            "overlay_path": row.get("overlay_path", ""),
            "audit_label": row.get("audit_label", ""),
            "audit_notes": row.get("audit_notes", ""),
            "reviewed_at": row.get("reviewed_at", ""),
            "img": b64,
        })

    print(f"内嵌 {len(chips)} 张图片")

    if not chips:
        print("ERROR: 无可用 chip，请先运行 build_gt_heater_audit.py 生成 chip")
        sys.exit(1)

    # Generate HTML
    labels_json = json.dumps(
        [(k, en, zh) for k, en, zh in LABELS], ensure_ascii=False,
    )
    chips_json = json.dumps(chips, ensure_ascii=False)

    html = HTML_TEMPLATE.replace("%%LABELS_JSON%%", labels_json)
    html = html.replace("%%CHIPS_JSON%%", chips_json)

    output_path = args.output or (args.run_dir / "gt_heater_audit_labeler.html")
    output_path.write_text(html, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n生成 {output_path} ({size_mb:.1f} MB)")
    # Convert WSL path to Windows path
    wsl_path = str(output_path.resolve())
    if wsl_path.startswith("/home/"):
        win_path = wsl_path.replace("/home/", "\\\\wsl$\\Ubuntu\\home\\").replace("/", "\\")
        print(f"  Windows: {win_path}")
    print(f"\n  快捷键: 1=PV  2=热水器/非PV  3=不确定  S=跳过  B=回退")
    print(f"  标完后点「导出 CSV」下载结果")


if __name__ == "__main__":
    main()
