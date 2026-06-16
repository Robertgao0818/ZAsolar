#!/usr/bin/env python3
"""Per-grid coverage manifest for the CT census — the grid-level deliverable.

The merged inventory (``ct_census_merge.py``) is a *polygon* list: surveyed-empty
cells contribute nothing and are therefore invisible in it. For a grid-level
census (V1.4 makes the task grid the primary aggregation unit) the denominator
matters as much as the numerator — we must be able to say, per CPT cell, whether
it was *surveyed and empty* (a valid count=0) versus *failed / never reached*
(unknown). The ``.fail`` marker historically conflated those.

This walks the full census glist and classifies every cell:

  ok              detector found >=1 panel; census_count = post-cls count
  empty           detector ran over every tile and found zero panels (count=0)
  infer_failed    detect produced no gpkg and no clean zero-detection line
                  (crash / OOM / missing tiles) — needs a retry
  download_failed tiles never downloaded (dl_<g>.fail and no dl_<g>.ok)
  not_reached     never inferred and no terminal marker

Robust to BOTH marker schemes:
  - new runs write infer_<g>.empty for surveyed-empty cells;
  - older runs wrote infer_<g>.fail for them, so when a .fail has a log
    containing the clean zero-detection line we reclassify it to ``empty`` here.

Outputs ``<out>.csv`` always, and ``<out>.gpkg`` (status joined onto the task
grid geometry) when --task-grid is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# the line detect_and_evaluate.py prints when every tile yielded zero panels
ZERO_DETECTION_MARK = "未检测到太阳能板"


def feat_count(fp: Path):
    """Fast feature count via pyogrio metadata; None if absent, -1 if unreadable."""
    if not fp.exists():
        return None
    try:
        import pyogrio

        return int(pyogrio.read_info(fp)["features"])
    except Exception:
        try:
            import geopandas as gpd

            return len(gpd.read_file(fp))
        except Exception:
            return -1


def log_says_empty(log_fp: Path) -> bool:
    if not log_fp.exists():
        return False
    try:
        return ZERO_DETECTION_MARK in log_fp.read_text(errors="ignore")
    except Exception:
        return False


def classify(g: str, state: Path, results: Path, logs: Path, layer_raw: str,
             layer_cls: str):
    n_raw = feat_count(results / g / layer_raw)
    n_cls = feat_count(results / g / layer_cls)

    dl_ok = (state / f"dl_{g}.ok").exists()
    dl_fail = (state / f"dl_{g}.fail").exists()
    i_ok = (state / f"infer_{g}.ok").exists()
    i_empty = (state / f"infer_{g}.empty").exists()
    i_fail = (state / f"infer_{g}.fail").exists()

    if i_ok:
        status = "ok"
    elif i_empty:
        status = "empty"
    elif i_fail:
        # reclassify legacy .fail using the per-grid detect log
        status = "empty" if log_says_empty(logs / f"infer_{g}.log") else "infer_failed"
    elif dl_fail and not dl_ok:
        status = "download_failed"
    else:
        status = "not_reached"

    # census count: post-cls if available, else raw; only 'ok' cells carry a count
    if status == "ok":
        count = n_cls if (n_cls is not None and n_cls >= 0) else (n_raw if (n_raw and n_raw > 0) else 0)
    else:
        count = 0

    return {
        "gridcell_id": g,
        "status": status,
        "n_raw": n_raw if n_raw is not None else 0,
        "n_cls": n_cls if n_cls is not None else "",
        "census_count": count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glist", required=True, type=Path)
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, type=Path,
                    help="output basename; writes <out>.csv and (with --task-grid) <out>.gpkg")
    ap.add_argument("--task-grid", type=Path, default=None,
                    help="task_grid_cpt.gpkg — join status onto geometry for a grid-level map")
    ap.add_argument("--layer-raw", default="predictions_metric.gpkg")
    ap.add_argument("--layer-cls", default="predictions_metric_cls_filtered.gpkg")
    args = ap.parse_args()

    grids = [g.strip() for g in args.glist.read_text().splitlines() if g.strip()]
    rows = [classify(g, args.state, args.results_dir, args.logs,
                     args.layer_raw, args.layer_cls) for g in grids]
    df = pd.DataFrame(rows)

    csv_out = args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    counts = df["status"].value_counts().to_dict()
    surveyed = int((df["status"].isin(["ok", "empty"])).sum())
    total_installs = int(df.loc[df["status"] == "ok", "census_count"].sum())
    print(f"[manifest] {len(df)} cells | surveyed={surveyed} "
          f"(ok={counts.get('ok',0)} empty={counts.get('empty',0)}) "
          f"infer_failed={counts.get('infer_failed',0)} "
          f"download_failed={counts.get('download_failed',0)} "
          f"not_reached={counts.get('not_reached',0)} | "
          f"installations={total_installs} -> {csv_out}")

    if args.task_grid and args.task_grid.exists():
        try:
            import geopandas as gpd

            grid = gpd.read_file(args.task_grid)
            key = "gridcell_id" if "gridcell_id" in grid.columns else grid.columns[0]
            grid[key] = grid[key].astype(str)
            joined = grid.merge(df, left_on=key, right_on="gridcell_id", how="left")
            joined["status"] = joined["status"].fillna("not_reached")
            joined["census_count"] = joined["census_count"].fillna(0)
            joined["model_run"] = args.run
            gpkg_out = args.out.with_suffix(".gpkg")
            joined.to_file(str(gpkg_out), driver="GPKG")
            print(f"[manifest] grid-level gpkg -> {gpkg_out} ({len(joined)} cells)")
        except Exception as e:  # noqa: BLE001
            print(f"[manifest] WARN: task-grid join skipped: {e}")


if __name__ == "__main__":
    main()
