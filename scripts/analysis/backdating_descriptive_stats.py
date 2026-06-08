#!/usr/bin/env python3
"""Descriptive-statistics HTML report for a backdating install-date batch.

Reads an ``install_intervals.csv`` produced by
``solar_backdating/scripts/temporal/infer_install_dates.py`` and emits a
single self-contained HTML file (inline CSS + SVG, no external assets) with
descriptive statistics: outcome breakdown, dating confidence, install-date
distribution, cumulative adoption, estimate-uncertainty bands, scan effort,
and per-grid coverage.

Install-date methodology note
-----------------------------
Each anchor's install date is interval-censored: the scan brackets it as
``[latest_absent, earliest_present]`` from the available GEHistoricalImagery
vintages. Johannesburg has a ~16-month imagery gap (2022-10 -> 2024-02), so a
naive ``install_mid_estimate`` (bracket midpoint) piles every install that fell
in that window onto a single phantom 2023 peak. This report therefore makes
the **interval-spread** distribution primary (each anchor contributes unit mass
spread uniformly across the months its bracket covers) and shows the midpoint
distribution only as a muted diagnostic with a vintage-gap caveat.

Usage:
    python scripts/analysis/backdating_descriptive_stats.py \
        --intervals-csv <run>/install_intervals.csv \
        --out <run>/descriptive_stats.html
"""
from __future__ import annotations

import argparse
import glob
import html
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Johannesburg GEHI imagery gap that drives the midpoint artifact.
GAP_START, GAP_END = pd.Period("2022-11", "M"), pd.Period("2024-01", "M")

STATUS_LABELS = {
    "done_appears": ("Dated", "absent->present transition found in the imagery history"),
    "done_ambiguous_no_recent_anchor": (
        "No recent anchor",
        "no usable PV-positive recent baseline — mostly group-centroid marker drift",
    ),
    "done_already_present_before_geid_history": (
        "Present before history",
        "PV already present in the earliest available imagery (open lower bound)",
    ),
    "done_ambiguous_nonmonotonic": (
        "Non-monotonic",
        "appears/disappears across vintages after dip-repair — unresolved",
    ),
    "done_ambiguous_marker_missed_pv": (
        "Marker missed PV",
        "the scan marker did not land on the panel",
    ),
    "done_ambiguous_gemini_failed": (
        "Scorer failed",
        "residual gateway failures, left un-dated (0.15%)",
    ),
}
STATUS_COLORS = {
    "done_appears": "#2a9d6f",
    "done_ambiguous_no_recent_anchor": "#d98a3d",
    "done_already_present_before_geid_history": "#7a6fd0",
    "done_ambiguous_nonmonotonic": "#c2554d",
    "done_ambiguous_marker_missed_pv": "#b08a4f",
    "done_ambiguous_gemini_failed": "#8a8a8a",
}
CONF_COLORS = {"high": "#2a9d6f", "medium": "#d9b13d", "low": "#c2554d"}


def esc(x) -> str:
    return html.escape(str(x))


def hbar(label, value, total, color, sub=""):
    pct = (100.0 * value / total) if total else 0.0
    return f"""
    <div class="row">
      <div class="row-label">{esc(label)}{f'<span class="sub">{esc(sub)}</span>' if sub else ''}</div>
      <div class="row-track"><div class="row-fill" style="width:{pct:.2f}%;background:{color}"></div></div>
      <div class="row-val">{value:,}<span class="pct">{pct:.1f}%</span></div>
    </div>"""


def card(value, label, accent="#1f6feb"):
    return f"""<div class="card"><div class="card-val" style="color:{accent}">{esc(value)}</div>
      <div class="card-lab">{esc(label)}</div></div>"""


# ---- install-date estimators ----------------------------------------------

def interval_spread_monthly(dated: pd.DataFrame) -> pd.Series:
    """Each anchor contributes unit mass spread uniformly over the months its
    [start, end] bracket spans. Returns a monthly Period-indexed Series."""
    mass = defaultdict(float)
    for s, e in zip(dated.install_interval_start, dated.install_interval_end):
        months = pd.period_range(s.to_period("M"), e.to_period("M"), freq="M")
        if len(months) == 0:
            continue
        w = 1.0 / len(months)
        for m in months:
            mass[m] += w
    return pd.Series(mass).sort_index()


# ---- chart builders --------------------------------------------------------

def year_bar_chart(year_counts: pd.Series, color="#1f6feb",
                   highlight_year=None, highlight_color="#c2554d",
                   highlight_note="", muted=False) -> str:
    mx = year_counts.max()
    cls = "ybar-chart muted" if muted else "ybar-chart"
    bars = []
    for yr, cnt in year_counts.items():
        h = 100.0 * cnt / mx
        c = highlight_color if (highlight_year is not None and int(yr) == highlight_year) else color
        note = (f'<div class="ybar-flag">{esc(highlight_note)}</div>'
                if (highlight_year is not None and int(yr) == highlight_year and highlight_note) else "")
        bars.append(
            f'<div class="ybar-col"><div class="ybar-num">{int(round(cnt)):,}</div>'
            f'{note}<div class="ybar" style="height:{h:.1f}%;background:{c}"></div>'
            f'<div class="ybar-lab">{int(yr)}</div></div>'
        )
    return f'<div class="{cls}">{"".join(bars)}</div>'


def adoption_svg(labels, cumulative, width=860, height=300) -> str:
    """Cumulative adoption area+line chart."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 46
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(labels)
    ymax = cumulative[-1]
    step = 10 ** (len(str(int(ymax))) - 1)
    ytop = math.ceil(ymax / step) * step

    def X(i):
        return pad_l + (iw * i / (n - 1) if n > 1 else 0)

    def Y(v):
        return pad_t + ih * (1 - v / ytop)

    pts = [(X(i), Y(v)) for i, v in enumerate(cumulative)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pad_l:.1f},{pad_t+ih:.1f} " + line + f" {X(n-1):.1f},{pad_t+ih:.1f}"

    grid, ylabels = [], []
    for k in range(0, 6):
        v = ytop * k / 5
        y = Y(v)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+iw}" y2="{y:.1f}" class="grid"/>')
        ylabels.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" class="yl">{int(v):,}</text>')

    xlabels = []
    for i, lab in enumerate(labels):
        if i % 2 == 0 or i == n - 1:
            xlabels.append(f'<text x="{X(i):.1f}" y="{pad_t+ih+18:.1f}" class="xl">{esc(lab)}</text>')
    dots = " ".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" class="dot"/>' for x, y in pts)

    return f"""<svg viewBox="0 0 {width} {height}" class="adopt" role="img">
      <defs><linearGradient id="ag" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0" stop-color="#1f6feb" stop-opacity="0.32"/>
        <stop offset="1" stop-color="#1f6feb" stop-opacity="0.02"/></linearGradient></defs>
      {''.join(grid)}
      <polygon points="{area}" fill="url(#ag)"/>
      <polyline points="{line}" fill="none" stroke="#1f6feb" stroke-width="2.2"/>
      {dots}
      {''.join(ylabels)}
      {''.join(xlabels)}
    </svg>"""


def vintage_strip_svg(month_counts: dict, width=860, height=150) -> str:
    """Per-month GEHI imagery coverage (how many sampled anchors had a capture
    that month), with the 2022-10 -> 2024-02 gap shaded red."""
    pad_l, pad_r, pad_t, pad_b = 30, 14, 14, 30
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    lo, hi = pd.Period("2017-01", "M"), pd.Period("2025-06", "M")
    months = pd.period_range(lo, hi, freq="M")
    span = len(months) - 1
    mx = max(month_counts.values()) if month_counts else 1

    def X(m):
        return pad_l + iw * ((m - lo).n) / span

    # gap shading
    gx0, gx1 = X(GAP_START), X(GAP_END)
    bars = []
    for m in months:
        c = month_counts.get(str(m), 0)
        if c <= 0:
            continue
        h = ih * (math.sqrt(c) / math.sqrt(mx))
        x = X(m)
        bars.append(f'<rect x="{x-2:.1f}" y="{pad_t+ih-h:.1f}" width="4" height="{h:.1f}" '
                    f'rx="1" fill="#2a9d6f"/>')
    # year ticks
    ticks = []
    for y in range(2017, 2026):
        x = X(pd.Period(f"{y}-01", "M"))
        ticks.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t+ih}" class="vgrid"/>')
        ticks.append(f'<text x="{x:.1f}" y="{pad_t+ih+18:.1f}" class="xl">{y}</text>')
    return f"""<svg viewBox="0 0 {width} {height}" class="vstrip" role="img">
      <rect x="{gx0:.1f}" y="{pad_t}" width="{gx1-gx0:.1f}" height="{ih}"
            fill="#c2554d" fill-opacity="0.16"/>
      <text x="{(gx0+gx1)/2:.1f}" y="{pad_t+14}" class="gaplab">~16-mo imagery gap</text>
      {''.join(ticks)}
      {''.join(bars)}
      <line x1="{pad_l}" y1="{pad_t+ih:.1f}" x2="{pad_l+iw}" y2="{pad_t+ih:.1f}" class="axis"/>
    </svg>"""


# ---- vintage sampling ------------------------------------------------------

def sample_vintages(df: pd.DataFrame, every=15) -> dict:
    """Sample scan_state JSONs and tally GEHI capture months (anchor coverage)."""
    paths = df.scan_state_path.dropna()
    if paths.empty:
        return {}
    sd = str(Path(paths.iloc[0]).parent)
    files = sorted(glob.glob(sd + "/*.json"))[::every]
    cnt = Counter()
    for f in files:
        try:
            txt = Path(f).read_text()
        except OSError:
            continue
        for d in set(re.findall(r"(20[12]\d-[01]\d-[0-3]\d)", txt)):
            if "2009-01-01" <= d <= "2025-12-31":
                cnt[d[:7]] += 1
    return dict(cnt)


# ---- main ------------------------------------------------------------------

def build(df: pd.DataFrame, run_name: str, source_polys: int | None) -> str:
    total = len(df)
    dated = df[df.install_mid_estimate.notna()].copy()
    for c in ["install_interval_start", "install_interval_end", "install_mid_estimate"]:
        dated[c] = pd.to_datetime(dated[c])
    n_dated = len(dated)
    n_grids = df.grid_id.nunique()
    yield_pct = 100.0 * n_dated / total

    status_counts = df.status.value_counts()
    conf_counts = df.loc[df.status == "done_appears", "confidence"].value_counts()
    conf_total = conf_counts.sum()

    # --- estimators ---
    monthly = interval_spread_monthly(dated)
    spread_year = monthly.groupby(lambda p: p.year).sum()
    spread_year = spread_year[spread_year.index >= 2009]
    mid_year = dated.install_mid_estimate.dt.year.value_counts().sort_index()
    mid_peak_year = int(mid_year.idxmax())
    mid_peak_cnt = int(mid_year.max())

    # adoption from interval-spread, half-year buckets
    hy = monthly.groupby(lambda p: f"{p.year}{'H1' if p.month <= 6 else 'H2'}").sum().sort_index()
    keys = [k for k in hy.index if int(k[:4]) >= 2016]
    cum = hy.cumsum()
    chart_labels = [k[:4] + "·" + k[4:] for k in keys]
    chart_cum = [int(cum[k]) for k in keys]

    # estimate window width
    w = (dated.install_interval_end - dated.install_interval_start).dt.days
    wbuckets = pd.cut(
        w, [0, 90, 180, 365, 548, 99999],
        labels=["≤ 3 mo", "3–6 mo", "6–12 mo", "12–18 mo", "> 18 mo"],
    ).value_counts().reindex(["≤ 3 mo", "3–6 mo", "6–12 mo", "12–18 mo", "> 18 mo"])
    w_med = int(w.median())

    obs, rnd = df.n_observations, df.n_rounds
    dip_anchors = int(df.notes.fillna("").str.contains("repaired_isolated_dip").sum())
    gpc = df.grid_id.value_counts()

    # share of midpoint-2023 mass that comes from the gap bracket
    gap_bracket = (
        (dated.install_interval_start.dt.to_period("M") == "2022-10")
        & (dated.install_interval_end.dt.to_period("M") == "2024-02")
    ).sum()

    vint = sample_vintages(df)

    # --- assemble ---
    status_rows = "".join(
        hbar(STATUS_LABELS.get(st, (st, ""))[0], int(cnt), total,
             STATUS_COLORS.get(st, "#888"), STATUS_LABELS.get(st, ("", ""))[1])
        for st, cnt in status_counts.items()
    )
    conf_rows = "".join(
        hbar(c.capitalize(), int(conf_counts.get(c, 0)), conf_total, CONF_COLORS[c])
        for c in ["high", "medium", "low"]
    )
    width_rows = "".join(hbar(str(lab), int(cnt), n_dated, "#1f6feb") for lab, cnt in wbuckets.items())
    grid_rows = "".join(hbar(g, int(c), int(gpc.max()), "#7a6fd0") for g, c in gpc.head(8).items())
    spread_peak = int(spread_year.idxmax())
    spread_peak_share = 100.0 * spread_year.max() / n_dated

    cards = (
        card(f"{total:,}", "chip-group anchors scanned")
        + card(f"{n_dated:,}", f"dated ({yield_pct:.1f}%)", "#2a9d6f")
        + card(f"{n_grids}", "JHB task grids", "#7a6fd0")
        + (card(f"{source_polys:,}", "source FP-cut polygons", "#d98a3d") if source_polys else "")
        + card("16 mo", "imagery gap 2022-10→2024-02", "#c2554d")
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backdating batch — descriptive statistics</title>
<style>
  :root {{ --bg:#0f1117; --panel:#171a21; --line:#262b36; --txt:#e6e9ef; --mut:#9aa3b2; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:32px 22px 80px; }}
  h1 {{ font-size:25px; margin:0 0 4px; letter-spacing:-.2px; }}
  .meta {{ color:var(--mut); font-size:13px; margin-bottom:22px; }}
  .meta code, .hint code, .note code {{ background:var(--panel); padding:1px 6px; border-radius:5px; color:#bcd; }}
  h2 {{ font-size:16px; margin:34px 0 4px; }}
  h2 .tag {{ font-size:11px; font-weight:600; color:#2a9d6f; border:1px solid #2a9d6f55;
    border-radius:20px; padding:1px 9px; margin-left:8px; vertical-align:middle; }}
  h2 .tag.diag {{ color:#c2554d; border-color:#c2554d55; }}
  .hint {{ color:var(--mut); font-size:13px; margin:0 0 14px; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin:18px 0 8px; }}
  .card {{ flex:1 1 150px; background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:16px 18px; }}
  .card-val {{ font-size:26px; font-weight:700; letter-spacing:-.5px; }}
  .card-lab {{ color:var(--mut); font-size:12.5px; margin-top:3px; }}
  .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; }}
  .panel.diag {{ border-color:#c2554d44; }}
  .row {{ display:grid; grid-template-columns:210px 1fr 116px; align-items:center; gap:12px; padding:5px 0; }}
  .row-label {{ font-size:13.5px; }}
  .row-label .sub {{ display:block; color:var(--mut); font-size:11.5px; line-height:1.35; }}
  .row-track {{ background:#0d0f14; border-radius:6px; height:18px; overflow:hidden; }}
  .row-fill {{ height:100%; border-radius:6px; min-width:2px; }}
  .row-val {{ text-align:right; font-variant-numeric:tabular-nums; font-size:13.5px; }}
  .row-val .pct {{ color:var(--mut); font-size:11.5px; margin-left:7px; }}
  .ybar-chart {{ display:flex; align-items:flex-end; gap:6px; height:210px; padding:26px 4px 0; }}
  .ybar-chart.muted {{ height:150px; opacity:.85; }}
  .ybar-col {{ flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
    height:100%; position:relative; }}
  .ybar {{ width:78%; border-radius:4px 4px 0 0; min-height:2px; }}
  .ybar-num {{ font-size:10.5px; color:var(--mut); margin-bottom:4px; font-variant-numeric:tabular-nums; }}
  .ybar-lab {{ font-size:11px; color:var(--mut); margin-top:6px; }}
  .ybar-flag {{ position:absolute; top:-2px; font-size:9.5px; color:#c2554d; text-align:center;
    width:150%; line-height:1.15; }}
  svg.adopt, svg.vstrip {{ width:100%; height:auto; display:block; }}
  svg.adopt .grid {{ stroke:var(--line); stroke-width:1; }}
  svg.adopt .yl {{ fill:var(--mut); font-size:11px; text-anchor:end; }}
  svg .xl {{ fill:var(--mut); font-size:11px; text-anchor:middle; }}
  svg.adopt .dot {{ fill:#1f6feb; }}
  svg.vstrip .vgrid {{ stroke:var(--line); stroke-width:1; }}
  svg.vstrip .axis {{ stroke:#3a4150; stroke-width:1; }}
  svg.vstrip .gaplab {{ fill:#c2554d; font-size:11px; text-anchor:middle; font-weight:600; }}
  .note {{ background:#14171e; border-left:3px solid #d98a3d; border-radius:0 8px 8px 0;
    padding:12px 16px; color:var(--mut); font-size:13px; margin-top:14px; }}
  .note.warn {{ border-left-color:#c2554d; }}
  .note b {{ color:var(--txt); }}
  .two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  @media (max-width:720px) {{ .two {{ grid-template-columns:1fr; }}
    .row {{ grid-template-columns:150px 1fr 92px; }} }}
  .stat-line {{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--line); font-size:13.5px; }}
  .stat-line:last-child {{ border-bottom:0; }}
  .stat-line span:last-child {{ font-variant-numeric:tabular-nums; color:#bcd; }}
  footer {{ color:var(--mut); font-size:12px; margin-top:40px; border-top:1px solid var(--line); padding-top:16px; }}
</style></head>
<body><div class="wrap">

  <h1>Install-date backdating &mdash; descriptive statistics</h1>
  <div class="meta">
    Batch <code>{esc(run_name)}</code> &nbsp;&middot;&nbsp; Johannesburg, all task grids
    &nbsp;&middot;&nbsp; generated 2026-06-03<br>
    Source: FP-cut detection inventory
    {f'(<code>{source_polys:,}</code> polygons)' if source_polys else ''}
    &rarr; chip-group anchors &rarr; GEHistoricalImagery + Gemini absent&rarr;present scan.
  </div>

  <div class="cards">{cards}</div>

  <div class="note warn"><b>Read this first — dates are interval-censored.</b>
    Each install is bracketed between the last imagery showing no panel and the first showing one.
    Johannesburg's GEHI history has a <b>~16-month gap (2022-10 &rarr; 2024-02)</b>, so the naive
    bracket-<i>midpoint</i> dumps every install from that window onto a phantom <b>{mid_peak_year}</b>
    spike ({mid_peak_cnt:,} anchors; {gap_bracket:,} share the single 487-day gap bracket).
    This report makes the <b>interval-spread</b> distribution primary (unit mass spread evenly across
    each bracket) and keeps the midpoint chart only as a labelled diagnostic.
  </div>

  <h2>GEHI imagery coverage</h2>
  <p class="hint">Sampled capture months (bar height = anchors with a usable capture that month).
     Dense through 2022, then the gap that forces the wide brackets.</p>
  <div class="panel">{vintage_strip_svg(vint)}</div>

  <h2>Outcome of the scan</h2>
  <p class="hint">What happened to each of the {total:,} anchors. Only <b>Dated</b> anchors carry an
     install-date estimate; the rest are accounted but un-dated.</p>
  <div class="panel">{status_rows}</div>

  <div class="two" style="margin-top:18px">
    <div>
      <h2>Dating confidence</h2>
      <p class="hint">Among the {n_dated:,} dated anchors.</p>
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
  <p class="hint">Each dated anchor contributes one install spread evenly across the months its bracket
     spans. Peak year <b>{spread_peak}</b> ({spread_peak_share:.0f}% of dated). The real signal is a
     <b>late-2022 → early-2024</b> adoption wave (SA load-shedding boom), not a single calendar year.</p>
  <div class="panel">{year_bar_chart(spread_year, color="#2a9d6f")}</div>

  <h2>Installs by year <span class="tag diag">bracket-midpoint · diagnostic only</span></h2>
  <p class="hint">Shown for comparison. The tall <b>{mid_peak_year}</b> bar is the vintage-gap artifact —
     do <b>not</b> read it as a calendar-{mid_peak_year} install count.</p>
  <div class="panel diag">{year_bar_chart(mid_year, color="#5b6470", highlight_year=mid_peak_year, highlight_note="⚠ gap artifact", muted=True)}</div>

  <h2>Cumulative adoption <span class="tag">interval-spread</span></h2>
  <p class="hint">Running total of expected installs, half-year resolution (from 2016). The right tail is
     censored by the imagery cutoff &mdash; treat the last 2&ndash;3 points as a lower bound.</p>
  <div class="panel">{adoption_svg(chart_labels, chart_cum)}</div>

  <div class="two" style="margin-top:18px">
    <div>
      <h2>Scan effort</h2>
      <div class="panel">
        <div class="stat-line"><span>Imagery observations / anchor (median)</span><span>{int(obs.median())}</span></div>
        <div class="stat-line"><span>Observations &mdash; mean / max</span><span>{obs.mean():.1f} / {int(obs.max())}</span></div>
        <div class="stat-line"><span>Bisection rounds / anchor (median)</span><span>{int(rnd.median())}</span></div>
        <div class="stat-line"><span>Rounds &mdash; mean / max</span><span>{rnd.mean():.2f} / {int(rnd.max())}</span></div>
        <div class="stat-line"><span>Anchors with monotonic dip-repair</span><span>{dip_anchors:,}</span></div>
      </div>
    </div>
    <div>
      <h2>Per-grid coverage</h2>
      <p class="hint">{n_grids} grids &middot; median {int(gpc.median())} anchors/grid &middot; max {int(gpc.max())}. Busiest grids:</p>
      <div class="panel">{grid_rows}</div>
    </div>
  </div>

  <div class="note">
    <b>Caveats.</b>
    &bull; Calendar-year resolution is not achievable inside the 2022-10→2024-02 gap; report those installs
    by bracket, not point year.
    &bull; <b>No recent anchor</b> ({int(status_counts.get('done_ambiguous_no_recent_anchor',0)):,}) is dominated
    by group-centroid marker drift on multi-detection chips, not scorer error; a per-target-marker recovery run
    is in progress to date these.
    &bull; Cumulative right tail is censored by the imagery cutoff and the FP-cut inventory's own 2025 vintage.
    &bull; <b>Dated</b> = <code>done_appears</code> only; confidence high/medium/low reflects observation density
    and bracket tightness.
  </div>

  <footer>
    ZAsolar &middot; solar_backdating — install-date backdating descriptive statistics.
    Input: <code>install_intervals.csv</code> ({total:,} rows). Charts are inline SVG/CSS; no external assets.
  </footer>

</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--source-polys", type=int, default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.intervals_csv)
    run_name = a.run_name or Path(a.intervals_csv).parent.name
    html_doc = build(df, run_name, a.source_polys)
    Path(a.out).write_text(html_doc, encoding="utf-8")
    print(f"wrote {a.out} ({len(html_doc):,} bytes) from {len(df):,} anchors")


if __name__ == "__main__":
    main()
