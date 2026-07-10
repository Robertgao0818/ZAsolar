#!/usr/bin/env python3
"""Descriptive-statistics HTML report for the *delivered polygon-level* backdating dataset.

Unlike ``backdating_descriptive_stats.py`` (which reports anchor / chip-group level
from ``install_intervals.csv``), this reads the **merged, flattened polygon CSV**
produced from ``merge_three_layers.py`` — i.e. the actual deliverable where every
FP-cut detection polygon carries an install date. It is **census-aware**: polygons
narrowed by the 2023 ESRI-Wayback census (``date_provider == "gehi_census2023"``)
are highlighted, and the midpoint chart shows how the census collapses the phantom
2023 vintage-gap spike into real 2023 captures.

Each polygon = one reviewed installation footprint, so it contributes unit mass to
the adoption distributions (an area-weighted total is reported alongside).

Reuses the chart/markup helpers from ``backdating_descriptive_stats.py``.

Usage:
    python scripts/analysis/backdating_polygon_descriptive_stats.py \
        --polygon-csv <run>/..._install_dated_..._with_census.csv \
        --out <run>/..._descriptive_stats_..._with_census.html \
        --run-name jhb_full382_fpcut_with_census
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

# --- import chart/markup helpers from the anchor-level generator (same dir) ---
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "backdating_descriptive_stats", _HERE / "backdating_descriptive_stats.py"
)
bds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bds)  # type: ignore[union-attr]

esc = bds.esc
hbar = bds.hbar
card = bds.card
year_bar_chart = bds.year_bar_chart
adoption_svg = bds.adoption_svg
interval_spread_monthly = bds.interval_spread_monthly
CONF_COLORS = bds.CONF_COLORS

# Outcome categories at polygon level (date_status, else undated_reason).
OUTCOME_LABELS = {
    "done_appears": ("Point-dated", "absent->present transition bracketed in the imagery history"),
    "already_present_lower_bound": (
        "Present before history (lower bound)",
        "PV already present in the earliest available imagery — open lower bound, not point-dated",
    ),
    "done_ambiguous_no_recent_anchor": (
        "Undated — no recent anchor",
        "group-centroid marker drift / no usable PV-positive recent baseline",
    ),
    "done_ambiguous_clamp_inverted": (
        "Undated — clamp inverted",
        "last-absent imagery itself post-dates the 2024 detection flight (whole TM scan post-detection)",
    ),
    "done_ambiguous_marker_missed_pv": (
        "Undated — marker missed PV",
        "the scan marker did not land on the panel",
    ),
    "done_ambiguous_nonmonotonic": (
        "Undated — non-monotonic",
        "appears/disappears across vintages after dip-repair",
    ),
    "done_ambiguous_gemini_failed": (
        "Undated — scorer failed",
        "residual gateway failures, left un-dated",
    ),
}
OUTCOME_COLORS = {
    "done_appears": "#2a9d6f",
    "already_present_lower_bound": "#7a6fd0",
    "done_ambiguous_no_recent_anchor": "#d98a3d",
    "done_ambiguous_clamp_inverted": "#b0563d",
    "done_ambiguous_marker_missed_pv": "#b08a4f",
    "done_ambiguous_nonmonotonic": "#c2554d",
    "done_ambiguous_gemini_failed": "#8a8a8a",
}
PROVIDER_LABELS = {
    "gehi_census2023": ("2023 census (narrowed)", "#1f9d55"),
    "gehi_main": ("GEHI chip-group scan", "#2a9d6f"),
    "gehi_pertarget": ("GEHI per-detection re-scan", "#3d8ad9"),
    "gehi_already_present_bound": ("Already-present lower bound", "#7a6fd0"),
}

STYLE = """
<style>
  :root { --bg:#0f1117; --panel:#171a21; --line:#262b36; --txt:#e6e9ef; --mut:#9aa3b2; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1000px; margin:0 auto; padding:32px 22px 80px; }
  h1 { font-size:25px; margin:0 0 4px; letter-spacing:-.2px; }
  .meta { color:var(--mut); font-size:13px; margin-bottom:22px; }
  .meta code, .hint code, .note code { background:var(--panel); padding:1px 6px; border-radius:5px; color:#bcd; }
  h2 { font-size:16px; margin:34px 0 4px; }
  h2 .tag { font-size:11px; font-weight:600; color:#2a9d6f; border:1px solid #2a9d6f55;
    border-radius:20px; padding:1px 9px; margin-left:8px; vertical-align:middle; }
  h2 .tag.diag { color:#c2554d; border-color:#c2554d55; }
  .hint { color:var(--mut); font-size:13px; margin:0 0 14px; }
  .cards { display:flex; flex-wrap:wrap; gap:12px; margin:18px 0 8px; }
  .card { flex:1 1 150px; background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:16px 18px; }
  .card-val { font-size:26px; font-weight:700; letter-spacing:-.5px; }
  .card-lab { color:var(--mut); font-size:12.5px; margin-top:3px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; }
  .panel.diag { border-color:#c2554d44; }
  .row { display:grid; grid-template-columns:230px 1fr 116px; align-items:center; gap:12px; padding:5px 0; }
  .row-label { font-size:13.5px; }
  .row-label .sub { display:block; color:var(--mut); font-size:11.5px; line-height:1.35; }
  .row-track { background:#0d0f14; border-radius:6px; height:18px; overflow:hidden; }
  .row-fill { height:100%; border-radius:6px; min-width:2px; }
  .row-val { text-align:right; font-variant-numeric:tabular-nums; font-size:13.5px; }
  .row-val .pct { color:var(--mut); font-size:11.5px; margin-left:7px; }
  .ybar-chart { display:flex; align-items:flex-end; gap:6px; height:210px; padding:26px 4px 0; }
  .ybar-chart.muted { height:150px; opacity:.85; }
  .ybar-col { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
    height:100%; position:relative; }
  .ybar { width:78%; border-radius:4px 4px 0 0; min-height:2px; }
  .ybar-num { font-size:10.5px; color:var(--mut); margin-bottom:4px; font-variant-numeric:tabular-nums; }
  .ybar-lab { font-size:11px; color:var(--mut); margin-top:6px; }
  .ybar-flag { position:absolute; top:-2px; font-size:9.5px; color:#c2554d; text-align:center;
    width:150%; line-height:1.15; }
  svg.adopt { width:100%; height:auto; display:block; }
  svg.adopt .grid { stroke:var(--line); stroke-width:1; }
  svg.adopt .yl { fill:var(--mut); font-size:11px; text-anchor:end; }
  svg .xl { fill:var(--mut); font-size:11px; text-anchor:middle; }
  svg.adopt .dot { fill:#1f6feb; }
  .note { background:#14171e; border-left:3px solid #d98a3d; border-radius:0 8px 8px 0;
    padding:12px 16px; color:var(--mut); font-size:13px; margin-top:14px; }
  .note.warn { border-left-color:#c2554d; }
  .note.good { border-left-color:#2a9d6f; }
  .note b { color:var(--txt); }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  @media (max-width:720px) { .two { grid-template-columns:1fr; }
    .row { grid-template-columns:150px 1fr 92px; } }
  .stat-line { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--line); font-size:13.5px; }
  .stat-line:last-child { border-bottom:0; }
  .stat-line span:last-child { font-variant-numeric:tabular-nums; color:#bcd; }
  footer { color:var(--mut); font-size:12px; margin-top:40px; border-top:1px solid var(--line); padding-top:16px; }
</style>"""


def build(df: pd.DataFrame, run_name: str, gen_date: str) -> str:
    total = len(df)

    # outcome = date_status if present else undated_reason
    status = df["date_status"].where(df["date_status"].notna(), df.get("undated_reason"))
    status = status.fillna("undated")
    status_counts = status.value_counts()

    is_point = df["date_is_bound"].fillna(-1).astype(float) == 0
    is_bound = df["date_is_bound"].fillna(-1).astype(float) == 1
    n_point = int(is_point.sum())
    n_bound = int(is_bound.sum())
    n_undated = total - n_point - n_bound

    # point-dated subset drives the install-date distributions
    dated = df[is_point].copy()
    for c in ["install_interval_start", "install_interval_end", "install_date"]:
        dated[c] = pd.to_datetime(dated[c], errors="coerce")
    dated = dated[dated["install_date"].notna()]
    n_dated = len(dated)

    # rename to the column the shared helper expects
    spread_src = dated.rename(columns={})
    monthly = interval_spread_monthly(spread_src)
    spread_year = monthly.groupby(lambda p: p.year).sum()
    spread_year = spread_year[spread_year.index >= 2009]
    spread_peak = int(spread_year.idxmax())
    spread_peak_share = 100.0 * spread_year.max() / n_dated if n_dated else 0.0

    mid_year = dated["install_date"].dt.year.value_counts().sort_index()
    mid_peak_year = int(mid_year.idxmax())
    mid_peak_cnt = int(mid_year.max())

    # cumulative adoption (interval-spread, half-year buckets from 2016)
    hy = monthly.groupby(lambda p: f"{p.year}{'H1' if p.month <= 6 else 'H2'}").sum().sort_index()
    keys = [k for k in hy.index if int(k[:4]) >= 2016]
    cum = hy.cumsum()
    chart_labels = [k[:4] + "·" + k[4:] for k in keys]
    chart_cum = [int(cum[k]) for k in keys]

    # window width
    w = (dated["install_interval_end"] - dated["install_interval_start"]).dt.days
    wbuckets = pd.cut(
        w, [-1, 90, 180, 365, 548, 99999],
        labels=["≤ 3 mo", "3–6 mo", "6–12 mo", "12–18 mo", "> 18 mo"],
    ).value_counts().reindex(["≤ 3 mo", "3–6 mo", "6–12 mo", "12–18 mo", "> 18 mo"])
    w_med = int(w.median()) if len(w) else 0

    # confidence among point-dated
    conf_counts = dated["install_confidence"].value_counts()
    conf_total = int(conf_counts.sum())

    # provenance
    prov_counts = df["date_provider"].fillna("(undated)").value_counts()
    n_census = int((df["date_provider"] == "gehi_census2023").sum())

    # area
    total_area_ha = df["area_m2"].sum() / 1e4
    dated_area_ha = dated["area_m2"].sum() / 1e4

    # per-grid
    gpc = df["source_grid"].value_counts()
    n_grids = int(gpc.nunique() if hasattr(gpc, "nunique") else len(gpc))
    n_grids = int(df["source_grid"].nunique())

    # gap-bracket share of midpoint mass
    gap_bracket = int((
        (dated["install_interval_start"].dt.to_period("M") == "2022-10")
        & (dated["install_interval_end"].dt.to_period("M") == "2024-02")
    ).sum())

    # ---- assemble rows ----
    status_rows = "".join(
        hbar(OUTCOME_LABELS.get(st, (st, ""))[0], int(cnt), total,
             OUTCOME_COLORS.get(st, "#888"), OUTCOME_LABELS.get(st, ("", ""))[1])
        for st, cnt in status_counts.items()
    )
    conf_rows = "".join(
        hbar(c.capitalize(), int(conf_counts.get(c, 0)), conf_total, CONF_COLORS.get(c, "#888"))
        for c in ["high", "medium", "low"] if conf_total
    )
    width_rows = "".join(hbar(str(lab), int(cnt), n_dated, "#1f6feb") for lab, cnt in wbuckets.items())
    grid_rows = "".join(hbar(g, int(c), int(gpc.max()), "#7a6fd0") for g, c in gpc.head(8).items())
    prov_rows = "".join(
        hbar(PROVIDER_LABELS.get(p, (p, "#888"))[0], int(c), total, PROVIDER_LABELS.get(p, (p, "#888"))[1])
        for p, c in prov_counts.items()
    )

    cards = (
        card(f"{total:,}", "FP-cut detection polygons")
        + card(f"{n_point:,}", f"point-dated ({100*n_point/total:.1f}%)", "#2a9d6f")
        + (card(f"{n_census:,}", "narrowed by 2023 census", "#1f9d55") if n_census else "")
        + card(f"{n_grids}", "JHB task grids", "#7a6fd0")
        + card(f"{total_area_ha:,.0f} ha", "total PV footprint", "#d98a3d")
    )

    census_note = ""
    if n_census:
        census_note = f"""
  <div class="note good"><b>2023 census applied.</b>
    <b>{n_census:,}</b> polygons ({100*n_census/total:.1f}% of all; {100*n_census/n_point:.1f}% of point-dated)
    had their install bracket tightened by a real 2023 ESRI-Wayback capture falling inside the
    interval, replacing the phantom-gap midpoint with a census-narrowed date. This is what pulls the
    midpoint distribution off the single <b>2023</b> spike.
  </div>"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backdating polygon dataset — descriptive statistics</title>{STYLE}</head>
<body><div class="wrap">

  <h1>Install-date backdating &mdash; delivered dataset (polygon level)</h1>
  <div class="meta">
    Batch <code>{esc(run_name)}</code> &nbsp;&middot;&nbsp; Johannesburg, all task grids
    &nbsp;&middot;&nbsp; generated {esc(gen_date)}<br>
    One row = one reviewed FP-cut detection footprint with an install-date estimate.
    Layers: 2023&nbsp;census &rarr; GEHI chip-group &rarr; GEHI per-detection &rarr; already-present bound.
  </div>

  <div class="cards">{cards}</div>

  <div class="note warn"><b>Read this first — dates are interval-censored.</b>
    Each install is bracketed between the last imagery showing no panel and the first showing one.
    Johannesburg's GEHI history has a <b>~16-month gap (2022-10 &rarr; 2024-02)</b>, so the naive
    bracket-<i>midpoint</i> dumps gap-window installs onto a phantom <b>{mid_peak_year}</b> spike
    ({mid_peak_cnt:,} polygons; {gap_bracket:,} still share the single gap bracket). The
    <b>interval-spread</b> distribution is primary; the midpoint chart is a labelled diagnostic.
  </div>
{census_note}
  <h2>Outcome — every polygon accounted</h2>
  <p class="hint">What happened to each of the {total:,} detection polygons. Only the
     <b>point-dated</b> ones carry a usable install date; the rest are accounted but un-dated or
     open lower bounds.</p>
  <div class="panel">{status_rows}</div>

  <div class="two" style="margin-top:18px">
    <div>
      <h2>Dating confidence</h2>
      <p class="hint">Among the {n_dated:,} point-dated polygons.</p>
      <div class="panel">{conf_rows}</div>
    </div>
    <div>
      <h2>Estimate window width</h2>
      <p class="hint">Bracket span (earliest-present − latest-absent). Median {w_med} days; the
         12–18 mo mode is the imagery gap, not model uncertainty.</p>
      <div class="panel">{width_rows}</div>
    </div>
  </div>

  <h2>Installs by year <span class="tag">interval-spread · primary</span></h2>
  <p class="hint">Each point-dated polygon contributes one install spread evenly across the months its
     bracket spans. Peak year <b>{spread_peak}</b> ({spread_peak_share:.0f}% of dated). The real signal
     is a <b>late-2022 → early-2024</b> adoption wave (SA load-shedding boom), not a single year.</p>
  <div class="panel">{year_bar_chart(spread_year, color="#2a9d6f")}</div>

  <h2>Installs by year <span class="tag diag">bracket-midpoint · diagnostic only</span></h2>
  <p class="hint">Shown for comparison. The tall <b>{mid_peak_year}</b> bar is the vintage-gap artifact;
     the 2023 census ({n_census:,} polygons) is what shrinks it toward real dates &mdash; do <b>not</b>
     read it as a calendar-{mid_peak_year} install count.</p>
  <div class="panel diag">{year_bar_chart(mid_year, color="#5b6470", highlight_year=mid_peak_year, highlight_note="⚠ gap artifact", muted=True)}</div>

  <h2>Cumulative adoption <span class="tag">interval-spread</span></h2>
  <p class="hint">Running total of expected installs, half-year resolution (from 2016). The right tail is
     censored by the imagery cutoff &mdash; treat the last 2&ndash;3 points as a lower bound.</p>
  <div class="panel">{adoption_svg(chart_labels, chart_cum)}</div>

  <div class="two" style="margin-top:18px">
    <div>
      <h2>Date provenance</h2>
      <p class="hint">Which layer dated each polygon ({100*(total-n_undated)/total:.1f}% labeled).</p>
      <div class="panel">{prov_rows}</div>
    </div>
    <div>
      <h2>Per-grid coverage</h2>
      <p class="hint">{n_grids} grids &middot; median {int(gpc.median())} polys/grid &middot; max {int(gpc.max())}. Busiest:</p>
      <div class="panel">{grid_rows}</div>
    </div>
  </div>

  <div class="note">
    <b>Caveats.</b>
    &bull; Calendar-year resolution is not achievable inside the 2022-10→2024-02 gap for polygons the
    2023 census could not reach; report those by bracket, not point year.
    &bull; <b>Present before history</b> ({n_bound:,}) are open lower bounds (PV already in the earliest
    imagery), not point installs.
    &bull; Cumulative right tail is censored by the imagery cutoff and the FP-cut inventory's 2024 flight.
    &bull; Dated area = {dated_area_ha:,.0f} ha of {total_area_ha:,.0f} ha total footprint.
  </div>

  <footer>
    ZAsolar &middot; solar_backdating — install-date backdating, delivered polygon dataset.
    Input: flattened merged CSV ({total:,} polygons). Charts are inline SVG/CSS; no external assets.
  </footer>

</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polygon-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--gen-date", default="2026-06-05")
    a = ap.parse_args()

    df = pd.read_csv(a.polygon_csv)
    if "source_feature_id" not in df.columns:
        raise SystemExit(
            "--polygon-csv must be the flattened merged polygon CSV (needs source_feature_id); "
            "for anchor-level use backdating_descriptive_stats.py"
        )
    run_name = a.run_name or Path(a.polygon_csv).stem
    html_doc = build(df, run_name, a.gen_date)
    Path(a.out).write_text(html_doc, encoding="utf-8")
    print(f"wrote {a.out} ({len(html_doc):,} bytes) from {len(df):,} polygons")


if __name__ == "__main__":
    main()
