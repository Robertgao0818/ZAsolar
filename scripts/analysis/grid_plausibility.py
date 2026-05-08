"""Channel 3 — Plausibility sanity checks for V1.4 validation framework.

Computes per-grid summary metrics on a model inventory and flags grids whose
metrics fall outside known physical bounds, plus stratum-relative outliers
(top/bottom 5%). Plausibility is *not* GT — it is the "I didn't hallucinate
half a city" guardrail described in `docs/validation_strategy.md`.

Inputs:
  --pred-root  results/<region>/<run>/   (subdirs G* with predictions_metric.gpkg)
  --region     ct | jhb | …               (used for task-grid + metric CRS)
  --grid-list  optional comma-sep ids; defaults to predictions_metric.gpkg subdirs
  --strata-csv optional CSV with columns grid_id,stratum (e.g. CBD/suburban/…)
  --min-area   drop polygons below this m² (default 2.0 — matches v4_canonical)
  --output-dir results/validation/plausibility_<YYYYMMDD>_<run>  (auto if blank)
  --label      free-text run label (e.g. v3c_sam_maskbox_vexcel_2024)

Outputs:
  per_grid.csv   one row per grid with raw metrics
  flags.csv      one row per (grid, flag) violation
  summary.md     human-readable narrative + flagged grid list
  config.json    invocation parameters for reproducibility

Bounds (residential urban; CBD strata are looser — see RESIDENTIAL_BOUNDS /
CBD_BOUNDS dicts). The defaults are informed by the JHB CBD 25-grid clean GT
distribution (median mean-area 71.8 m², max single install 1665 m², area
coverage ≤ 1.74%); they will tighten once `grid_strata.csv` lands and the
sampling protocol gives per-stratum priors.

Usage:
  python scripts/analysis/grid_plausibility.py \
    --pred-root results/johannesburg/v3c_sam_maskbox_vexcel_2024 \
    --region jhb --label v3c_sam_maskbox_vexcel_2024
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core import region_registry  # noqa: E402
from core.grid_utils import get_metric_crs  # noqa: E402


def _resolve_region(alias: str) -> str:
    canonical = region_registry.normalize_region_key(alias)
    if canonical and canonical in region_registry.list_regions():
        return canonical
    if alias in region_registry.list_regions():
        return alias
    raise SystemExit(
        f"Unknown region alias '{alias}'. Available: {region_registry.list_regions()}"
    )


# Bound table — keyed by stratum; "default" is a permissive cross-stratum
# fallback. Tightening per-stratum bounds is part of the V1.4 grid_strata
# tagging deliverable; until that lands these only catch egregious failures.
@dataclass(frozen=True)
class Bounds:
    mean_area_m2: tuple[float, float]
    median_area_m2: tuple[float, float]
    density_per_km2_max: float
    area_coverage_pct_max: float
    single_install_m2_max: float


BOUNDS_BY_STRATUM: dict[str, Bounds] = {
    "default": Bounds(
        mean_area_m2=(8.0, 250.0),
        median_area_m2=(8.0, 150.0),
        density_per_km2_max=2000.0,
        area_coverage_pct_max=10.0,
        single_install_m2_max=5000.0,
    ),
    "residential": Bounds(
        mean_area_m2=(10.0, 60.0),
        median_area_m2=(10.0, 40.0),
        density_per_km2_max=800.0,
        area_coverage_pct_max=3.0,
        single_install_m2_max=500.0,
    ),
    "suburban": Bounds(
        mean_area_m2=(10.0, 60.0),
        median_area_m2=(10.0, 40.0),
        density_per_km2_max=800.0,
        area_coverage_pct_max=3.0,
        single_install_m2_max=500.0,
    ),
    "CBD": Bounds(
        mean_area_m2=(15.0, 250.0),
        median_area_m2=(10.0, 150.0),
        density_per_km2_max=500.0,
        area_coverage_pct_max=10.0,
        single_install_m2_max=5000.0,
    ),
    "township": Bounds(
        mean_area_m2=(6.0, 40.0),
        median_area_m2=(6.0, 25.0),
        density_per_km2_max=1500.0,
        area_coverage_pct_max=2.0,
        single_install_m2_max=300.0,
    ),
    "peri_urban": Bounds(
        mean_area_m2=(8.0, 80.0),
        median_area_m2=(8.0, 50.0),
        density_per_km2_max=400.0,
        area_coverage_pct_max=2.0,
        single_install_m2_max=800.0,
    ),
    "rural": Bounds(
        mean_area_m2=(8.0, 200.0),
        median_area_m2=(8.0, 100.0),
        density_per_km2_max=100.0,
        area_coverage_pct_max=1.0,
        single_install_m2_max=2000.0,
    ),
}

OUTLIER_QUANTILE = 0.05  # top/bottom 5% per stratum


def _discover_grid_ids(pred_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in pred_root.iterdir()
        if p.is_dir() and (p / "predictions_metric.gpkg").exists()
    )


def _grid_area_km2(region_key: str, grid_ids: list[str]) -> dict[str, float]:
    tg_path = region_registry.get_task_grid_path(region_key)
    g = gpd.read_file(tg_path)
    sub = g[g["gridcell_id"].isin(grid_ids)].copy()
    metric_crs = get_metric_crs(grid_ids[0], region=region_key)
    sub_metric = sub.to_crs(metric_crs)
    sub_metric["_area_km2"] = sub_metric.geometry.area / 1e6
    return dict(zip(sub_metric["gridcell_id"], sub_metric["_area_km2"]))


def _polygon_metrics(
    pred_path: Path, metric_crs: str, min_area: float
) -> dict[str, float]:
    gdf = gpd.read_file(pred_path)
    if gdf.crs is None or str(gdf.crs).upper() != metric_crs.upper():
        gdf = gdf.to_crs(metric_crs)
    if "area_m2" in gdf.columns:
        a = gdf["area_m2"].astype(float)
    else:
        a = gdf.geometry.area
    a = a[a >= min_area]
    if len(a) == 0:
        return {
            "n_install": 0,
            "total_area_m2": 0.0,
            "mean_area_m2": float("nan"),
            "median_area_m2": float("nan"),
            "p10_area_m2": float("nan"),
            "p90_area_m2": float("nan"),
            "max_area_m2": float("nan"),
            "n_above_500": 0,
        }
    return {
        "n_install": int(len(a)),
        "total_area_m2": float(a.sum()),
        "mean_area_m2": float(a.mean()),
        "median_area_m2": float(a.median()),
        "p10_area_m2": float(a.quantile(0.10)),
        "p90_area_m2": float(a.quantile(0.90)),
        "max_area_m2": float(a.max()),
        "n_above_500": int((a >= 500.0).sum()),
    }


def _bounds_for(stratum: str | None) -> Bounds:
    if stratum and stratum in BOUNDS_BY_STRATUM:
        return BOUNDS_BY_STRATUM[stratum]
    return BOUNDS_BY_STRATUM["default"]


def _check_bounds(row: pd.Series, b: Bounds) -> list[dict]:
    flags: list[dict] = []
    if row["n_install"] == 0:
        flags.append(
            {
                "flag": "zero_installs",
                "value": 0,
                "bound": "n_install > 0",
                "severity": "info",
            }
        )
        return flags
    if not (b.mean_area_m2[0] <= row["mean_area_m2"] <= b.mean_area_m2[1]):
        flags.append(
            {
                "flag": "mean_area_out_of_bounds",
                "value": row["mean_area_m2"],
                "bound": f"{b.mean_area_m2[0]}-{b.mean_area_m2[1]} m^2",
                "severity": "high",
            }
        )
    if not (b.median_area_m2[0] <= row["median_area_m2"] <= b.median_area_m2[1]):
        flags.append(
            {
                "flag": "median_area_out_of_bounds",
                "value": row["median_area_m2"],
                "bound": f"{b.median_area_m2[0]}-{b.median_area_m2[1]} m^2",
                "severity": "high",
            }
        )
    if row["density_per_km2"] > b.density_per_km2_max:
        flags.append(
            {
                "flag": "density_too_high",
                "value": row["density_per_km2"],
                "bound": f"<= {b.density_per_km2_max}",
                "severity": "high",
            }
        )
    if row["area_coverage_pct"] > b.area_coverage_pct_max:
        flags.append(
            {
                "flag": "area_coverage_too_high",
                "value": row["area_coverage_pct"],
                "bound": f"<= {b.area_coverage_pct_max}%",
                "severity": "high",
            }
        )
    if row["max_area_m2"] > b.single_install_m2_max:
        flags.append(
            {
                "flag": "single_install_oversize",
                "value": row["max_area_m2"],
                "bound": f"<= {b.single_install_m2_max} m^2 (any one polygon)",
                "severity": "info",
            }
        )
    return flags


def _stratum_outliers(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    metrics = [
        "n_install",
        "density_per_km2",
        "area_coverage_pct",
        "mean_area_m2",
        "median_area_m2",
    ]
    for stratum, sub in df.groupby("stratum"):
        if len(sub) < 10:
            continue
        for m in metrics:
            vals = sub[m].dropna()
            if vals.empty:
                continue
            lo, hi = vals.quantile(OUTLIER_QUANTILE), vals.quantile(1 - OUTLIER_QUANTILE)
            for _, r in sub.iterrows():
                v = r[m]
                if pd.isna(v):
                    continue
                if v <= lo:
                    out.append(
                        {
                            "grid_id": r["grid_id"],
                            "stratum": stratum,
                            "flag": f"{m}_low_outlier",
                            "value": v,
                            "bound": f"<= p{int(OUTLIER_QUANTILE*100)} ({lo:.2f}) within {stratum}",
                            "severity": "info",
                        }
                    )
                elif v >= hi:
                    out.append(
                        {
                            "grid_id": r["grid_id"],
                            "stratum": stratum,
                            "flag": f"{m}_high_outlier",
                            "value": v,
                            "bound": f">= p{int((1-OUTLIER_QUANTILE)*100)} ({hi:.2f}) within {stratum}",
                            "severity": "info",
                        }
                    )
    return out


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def _write_summary(
    out_dir: Path,
    df: pd.DataFrame,
    flag_df: pd.DataFrame,
    label: str,
    pred_root: Path,
) -> None:
    lines = [f"# Plausibility report — {label}", ""]
    lines.append(f"- Pred root: `{pred_root}`")
    lines.append(
        f"- Generated: {datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds')}Z"
    )
    lines.append(f"- Grids analyzed: {len(df)}")
    n_flagged = flag_df["grid_id"].nunique() if not flag_df.empty else 0
    lines.append(f"- Grids with at least one flag: {n_flagged}")
    n_high = (flag_df["severity"] == "high").sum() if not flag_df.empty else 0
    lines.append(f"- High-severity flags: {n_high}")
    lines.append("")

    lines.append("## Per-grid metrics")
    lines.append("")
    cols = [
        "grid_id",
        "stratum",
        "n_install",
        "density_per_km2",
        "area_coverage_pct",
        "mean_area_m2",
        "median_area_m2",
        "max_area_m2",
    ]
    show = df[cols].copy()
    for c in [
        "density_per_km2",
        "area_coverage_pct",
        "mean_area_m2",
        "median_area_m2",
        "max_area_m2",
    ]:
        show[c] = show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    lines.append(show.pipe(_df_to_md))
    lines.append("")

    if not flag_df.empty:
        lines.append("## Flags")
        lines.append("")
        lines.append(
            flag_df[["grid_id", "stratum", "flag", "severity", "value", "bound"]]
            .sort_values(["severity", "grid_id", "flag"])
            .pipe(_df_to_md)
        )
    else:
        lines.append("## Flags")
        lines.append("")
        lines.append("No flags raised.")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--grid-list", default=None, help="comma-separated grid ids; default = auto-discover")
    p.add_argument("--strata-csv", type=Path, default=None)
    p.add_argument("--min-area", type=float, default=2.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    pred_root: Path = args.pred_root.resolve()
    if not pred_root.is_dir():
        sys.exit(f"pred-root not found: {pred_root}")

    label = args.label or pred_root.name
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else REPO / "results" / "validation" / f"plausibility_{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}_{label}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid_list:
        grids = [g.strip() for g in args.grid_list.split(",") if g.strip()]
    else:
        grids = _discover_grid_ids(pred_root)
    if not grids:
        sys.exit(f"no grids with predictions_metric.gpkg under {pred_root}")

    region_key = _resolve_region(args.region)
    metric_crs = get_metric_crs(grids[0], region=region_key)
    grid_areas = _grid_area_km2(region_key, grids)

    strata_lookup: dict[str, str] = {}
    if args.strata_csv:
        s = pd.read_csv(args.strata_csv)
        if "grid_id" not in s.columns or "stratum" not in s.columns:
            sys.exit("strata-csv must have columns grid_id,stratum")
        strata_lookup = dict(zip(s["grid_id"].astype(str), s["stratum"].astype(str)))

    rows: list[dict] = []
    for g in grids:
        pred_path = pred_root / g / "predictions_metric.gpkg"
        if not pred_path.exists():
            print(f"[skip] {g}: no predictions_metric.gpkg", file=sys.stderr)
            continue
        m = _polygon_metrics(pred_path, metric_crs, args.min_area)
        area_km2 = grid_areas.get(g, float("nan"))
        density = m["n_install"] / area_km2 if area_km2 and area_km2 > 0 else float("nan")
        coverage_pct = (
            (m["total_area_m2"] / (area_km2 * 1e6)) * 100 if area_km2 and area_km2 > 0 else float("nan")
        )
        rows.append(
            {
                "grid_id": g,
                "stratum": strata_lookup.get(g, "unstratified"),
                "grid_area_km2": area_km2,
                **m,
                "density_per_km2": density,
                "area_coverage_pct": coverage_pct,
            }
        )

    df = pd.DataFrame(rows).sort_values("grid_id").reset_index(drop=True)
    df.to_csv(out_dir / "per_grid.csv", index=False)

    flag_rows: list[dict] = []
    for _, r in df.iterrows():
        b = _bounds_for(r["stratum"])
        for f in _check_bounds(r, b):
            flag_rows.append({"grid_id": r["grid_id"], "stratum": r["stratum"], **f})
    flag_rows.extend(_stratum_outliers(df))
    flag_df = pd.DataFrame(flag_rows)
    if flag_df.empty:
        flag_df = pd.DataFrame(columns=["grid_id", "stratum", "flag", "value", "bound", "severity"])
    flag_df.to_csv(out_dir / "flags.csv", index=False)

    config = {
        "pred_root": str(pred_root),
        "region": args.region,
        "region_key": region_key,
        "metric_crs": metric_crs,
        "grids": grids,
        "min_area_m2": args.min_area,
        "strata_csv": str(args.strata_csv) if args.strata_csv else None,
        "outlier_quantile": OUTLIER_QUANTILE,
        "bounds_by_stratum": {
            s: {
                "mean_area_m2": list(b.mean_area_m2),
                "median_area_m2": list(b.median_area_m2),
                "density_per_km2_max": b.density_per_km2_max,
                "area_coverage_pct_max": b.area_coverage_pct_max,
                "single_install_m2_max": b.single_install_m2_max,
            }
            for s, b in BOUNDS_BY_STRATUM.items()
        },
        "label": label,
        "generated_utc": datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    _write_summary(out_dir, df, flag_df, label, pred_root)

    print(f"[ok] {len(df)} grids analysed → {out_dir}")
    n_flagged = flag_df["grid_id"].nunique() if not flag_df.empty else 0
    n_high = (flag_df["severity"] == "high").sum() if not flag_df.empty else 0
    print(f"     flagged grids: {n_flagged} (high-severity flags: {n_high})")


if __name__ == "__main__":
    main()
