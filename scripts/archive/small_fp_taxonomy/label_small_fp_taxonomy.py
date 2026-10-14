#!/usr/bin/env python3
"""
小目标 FP taxonomy 标注工具 — 生成 HTML 标注页面。

生成一个自包含的 HTML 文件（chip 内嵌为 base64），
在浏览器中按数字键标注，完成后导出 CSV。

用法：
    python scripts/analysis/label_small_fp_taxonomy.py \
        --run-dir results/analysis/small_fp/taxonomy_run

    # 只生成前 100 张（高置信优先）
    python scripts/analysis/label_small_fp_taxonomy.py \
        --run-dir results/analysis/small_fp/taxonomy_run --limit 100

生成后在 Windows 中打开 taxonomy_labeler.html：
    快捷键: 1-8 标注, S 跳过, B 回退
    标完后点 "导出 CSV" 按钮下载结果
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import pandas as pd

LABELS = [
    ("1", "solar_thermal_water_heater", "太阳能热水器"),
    ("2", "pergola_carport_shadow", "车棚/遮阳棚"),
    ("3", "skylight_roof_window", "天窗/屋顶窗"),
    ("4", "roof_shadow_dark_fixture", "屋顶阴影/深色物"),
    ("5", "blue_tarp_roof_cover", "蓝色防水布"),
    ("6", "vehicle_or_road_marking", "车辆/路面标线"),
    ("7", "fragment_near_true_panel", "真实板碎片"),
    ("8", "other_unknown", "其他/不确定"),
]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Small FP Taxonomy Labeler</title>
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
.sidebar { width: 260px; background: #16213e; padding: 12px; overflow-y: auto;
           display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
.info { font-size: 13px; line-height: 1.6; padding: 8px; background: #0f3460;
        border-radius: 6px; }
.info .val { color: #4ecca3; }
.label-btn { display: flex; align-items: center; gap: 8px; padding: 8px 10px;
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
</style>
</head>
<body>
<div class="header">
  <h1>Small FP Taxonomy</h1>
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
    <div class="nav">
      <button onclick="prev()">&larr; B 回退</button>
      <button onclick="skip()">S 跳过 &rarr;</button>
    </div>
    <button class="export-btn" onclick="exportCSV()">导出 CSV</button>
    <div class="hint">快捷键: 1-8 标注, S 跳过, B 回退</div>
  </div>
</div>
<script>
const LABELS = %%LABELS_JSON%%;
const CHIPS = %%CHIPS_JSON%%;
let idx = 0;

// Find first unlabeled
for (let i = 0; i < CHIPS.length; i++) {
  if (!CHIPS[i].human_label) { idx = i; break; }
}

function render() {
  const c = CHIPS[idx];
  document.getElementById("chipImg").src = "data:image/png;base64," + c.img;
  document.getElementById("currentIdx").textContent = idx + 1;
  document.getElementById("totalCount").textContent = CHIPS.length;
  document.getElementById("labeledCount").textContent =
    CHIPS.filter(x => x.human_label).length;
  document.getElementById("chipInfo").innerHTML =
    `<b>Grid:</b> <span class="val">${c.grid_id}</span><br>` +
    `<b>Pred:</b> <span class="val">${c.pred_id}</span><br>` +
    `<b>Conf:</b> <span class="val">${c.confidence.toFixed(3)}</span><br>` +
    `<b>Area:</b> <span class="val">${c.area_m2.toFixed(1)} m²</span><br>` +
    `<b>Tile:</b> <span class="val">${c.source_tile}</span>`;

  // Badge
  const badge = document.getElementById("badge");
  if (c.human_label) {
    const lbl = LABELS.find(l => l[1] === c.human_label);
    badge.textContent = lbl ? lbl[2] : c.human_label;
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
    btn.className = "label-btn" + (c.human_label === en ? " active" : "");
    btn.innerHTML = `<span class="key">${key}</span> ${zh}`;
    btn.onclick = () => applyLabel(en);
    container.appendChild(btn);
  }
}

function applyLabel(label) {
  CHIPS[idx].human_label = label;
  if (idx < CHIPS.length - 1) idx++;
  render();
}

function skip() { if (idx < CHIPS.length - 1) { idx++; render(); } }
function prev() { if (idx > 0) { idx--; render(); } }

document.addEventListener("keydown", e => {
  const k = e.key;
  if (k >= "1" && k <= "8") { applyLabel(LABELS[parseInt(k) - 1][1]); }
  else if (k.toLowerCase() === "s") { skip(); }
  else if (k.toLowerCase() === "b") { prev(); }
});

function exportCSV() {
  let csv = "pred_id,grid_id,source_tile,fp_bucket,hn_safety,confidence,area_m2,chip_path,human_label,notes\n";
  for (const c of CHIPS) {
    csv += [c.pred_id, c.grid_id, c.source_tile, c.fp_bucket, c.hn_safety,
            c.confidence, c.area_m2, c.chip_path, c.human_label || "", c.notes || ""
           ].join(",") + "\n";
  }
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "small_fp_taxonomy_labeled.csv";
  a.click();
}

render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="生成 HTML taxonomy 标注页面")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="只生成前 N 张（按置信度降序）")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 HTML 路径（默认 run-dir/taxonomy_labeler.html）")
    args = parser.parse_args()

    template_path = args.run_dir / "small_fp_taxonomy_template.csv"
    if not template_path.exists():
        print(f"未找到 {template_path}")
        sys.exit(1)

    df = pd.read_csv(template_path)
    df["human_label"] = df["human_label"].fillna("")
    df["notes"] = df["notes"].fillna("")
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit)

    print(f"准备 {len(df)} 张 chip...")

    # Build chip data with embedded base64 images
    chips = []
    overlay_dir = args.run_dir / "chips" / "safe_true_fp_overlay"
    rgb_dir = args.run_dir / "chips" / "safe_true_fp"

    for _, row in df.iterrows():
        fname = Path(row["chip_path"]).name
        img_path = overlay_dir / fname
        if not img_path.exists():
            img_path = rgb_dir / fname
        if not img_path.exists():
            continue

        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        chips.append({
            "pred_id": int(row["pred_id"]),
            "grid_id": row["grid_id"],
            "source_tile": row.get("source_tile", ""),
            "fp_bucket": row.get("fp_bucket", ""),
            "hn_safety": row.get("hn_safety", ""),
            "confidence": float(row["confidence"]),
            "area_m2": float(row["area_m2"]),
            "chip_path": row["chip_path"],
            "human_label": row.get("human_label", ""),
            "notes": row.get("notes", ""),
            "img": b64,
        })

    print(f"内嵌 {len(chips)} 张图片")

    # Generate HTML
    labels_json = json.dumps(
        [(k, en, zh) for k, en, zh in LABELS], ensure_ascii=False,
    )
    chips_json = json.dumps(chips, ensure_ascii=False)

    html = HTML_TEMPLATE.replace("%%LABELS_JSON%%", labels_json)
    html = html.replace("%%CHIPS_JSON%%", chips_json)

    output_path = args.output or (args.run_dir / "taxonomy_labeler.html")
    output_path.write_text(html, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n✓ 生成 {output_path} ({size_mb:.1f} MB)")
    print(f"  在 Windows 浏览器中打开：")
    # Convert WSL path to Windows path
    wsl_path = str(output_path.resolve())
    if wsl_path.startswith("/home/"):
        win_path = wsl_path.replace("/home/", "\\\\wsl$\\Ubuntu\\home\\").replace("/", "\\")
        print(f"  {win_path}")
    print(f"\n  快捷键: 1-8 标注, S 跳过, B 回退")
    print(f"  标完后点「导出 CSV」下载结果")


if __name__ == "__main__":
    main()
