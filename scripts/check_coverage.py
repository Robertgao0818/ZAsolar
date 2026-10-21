#!/usr/bin/env python3
"""Mandatory coverage gate: run BEFORE merge / delivery of any city census.

Every unit (tile, or grid for grid-level pipelines) in the city AOI must be either
  * scored   - present in at least one --scored CSV (e.g. pregate tile_scores), or
  * excluded - present in an --exclusions CSV with a non-empty reason.
Anything else is UNSCORED and the gate exits 2 with a summary. This exists because
CT 2026-10 silently lost 12,922 tiles: the downloader left already-present tiles out
of the manifest and the pregate only scores manifest tiles.

The AOI list must come from the AOI geometry (see scripts/build_aoi_tiles.py), never
from a downloader manifest or a directory listing, otherwise the gate checks nothing.

Inputs
  --aoi CSV          column --key (default "tile"); optional minx,miny,maxx,maxy (+--crs)
                     to locate reference objects on unscored units.
  --scored CSV ...   column --key; rows with an empty/NaN --score-col (if given) do not count.
  --exclusions CSV   columns --key, reason. Empty reason = invalid -> gate fails.
  --reference GPKG   optional old/reference inventory; objects whose representative point
                     falls in an unscored unit are counted (needs AOI bounds).
Outputs (--out-dir): coverage_summary.json, unscored_units.csv, [unscored_reference.csv]
Exit codes: 0 pass, 2 unscored units / bad exclusions, 1 input error. There is no tolerance knob:
a unit that is legitimately not scored must be listed in an exclusions file with its reason.

Wiring
  * scripts/ct_census_run.sh (grid-level legacy pipeline) runs this before cls+merge.
  * Tile-level pregate pipelines (CT native 0.05 m, JHB Vexcel) must run, before merge/delivery:
      python scripts/build_aoi_tiles.py ct-native --grids content_grids_1301.csv --out aoi.csv
      python scripts/check_coverage.py --city ct --aoi aoi.csv --scored tile_scores_v1.csv \
          --score-col max_score [--exclusions excl.csv] --out-dir gate/ || exit 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd


def norm(s: pd.Series, basename: bool) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.map(os.path.basename) if basename else s


def load_keys(paths, key, basename, score_col=None):
    out = set()
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in {key, score_col})
        if key not in df.columns:
            raise SystemExit(f"[gate] input error: {p} has no column '{key}'")
        if score_col:
            if score_col not in df.columns:
                raise SystemExit(f"[gate] input error: {p} has no column '{score_col}'")
            df = df[pd.to_numeric(df[score_col], errors="coerce").notna()]
        out |= set(norm(df[key].dropna(), basename))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--exclusions", nargs="*", default=[])
    ap.add_argument("--key", default="tile")
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--basename", action="store_true", help="compare file basenames only")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--crs", default=None, help="CRS of AOI bounds, e.g. EPSG:32734")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    aoi = pd.read_csv(a.aoi)
    if a.key not in aoi.columns:
        print(f"[gate] input error: AOI has no column '{a.key}'", file=sys.stderr)
        return 1
    raw_n = aoi[a.key].astype(str).str.strip().nunique()
    aoi[a.key] = norm(aoi[a.key], a.basename)
    if a.basename and aoi[a.key].nunique() != raw_n:
        print("[gate] input error: --basename merges distinct AOI units (names not unique); drop --basename", file=sys.stderr)
        return 1
    dup = int(aoi[a.key].duplicated().sum())
    aoi = aoi.drop_duplicates(a.key)
    if aoi.empty:
        print("[gate] input error: AOI is empty", file=sys.stderr)
        return 1
    A = set(aoi[a.key])
    S = load_keys(a.scored, a.key, a.basename, a.score_col)

    E, bad_ex = {}, []
    for p in a.exclusions:
        ex = pd.read_csv(p)
        if a.key not in ex.columns or "reason" not in ex.columns:
            print(f"[gate] input error: {p} needs columns {a.key},reason", file=sys.stderr)
            return 1
        for k, r in zip(norm(ex[a.key], a.basename), ex["reason"]):
            r = "" if pd.isna(r) else str(r).strip()
            (E.__setitem__(k, r) if r else bad_ex.append(k))

    scored = A & S
    excluded = (A - S) & set(E)
    unscored = sorted(A - S - set(E))
    reasons = pd.Series([E[k] for k in excluded], dtype=object).value_counts().to_dict()
    summ = {
        "city": a.city, "unit": a.key, "aoi_units": len(A), "aoi_duplicates_dropped": dup,
        "scored": len(scored), "excluded": len(excluded), "excluded_by_reason": reasons,
        "unscored": len(unscored), "unscored_share": round(len(unscored) / len(A), 6),
        "scored_outside_aoi": len(S - A), "exclusions_without_reason": len(bad_ex),
        "inputs": {"aoi": a.aoi, "scored": a.scored, "exclusions": a.exclusions, "reference": a.reference},
    }
    pd.DataFrame({a.key: unscored}).merge(aoi, on=a.key, how="left").to_csv(
        f"{a.out_dir}/unscored_units.csv", index=False)

    if a.reference:
        import geopandas as gpd
        need = {"minx", "miny", "maxx", "maxy"}
        if not need <= set(aoi.columns) or not a.crs:
            print("[gate] input error: --reference needs AOI bounds columns and --crs", file=sys.stderr)
            return 1
        from shapely.geometry import box
        cells = gpd.GeoDataFrame(aoi[[a.key]], geometry=[box(*b) for b in aoi[["minx", "miny", "maxx", "maxy"]].to_numpy()], crs=a.crs)
        ref = gpd.read_file(a.reference).to_crs(a.crs)
        ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
        area = ref.geometry.area
        pts = gpd.GeoDataFrame({"ref_idx": range(len(ref)), "area_m2": area.to_numpy()},
                               geometry=ref.geometry.representative_point().to_numpy(), crs=a.crs)
        # A point on a shared edge can hit 2 cells; count each object once (first match).
        j = gpd.sjoin(pts, cells, how="left", predicate="intersects").drop_duplicates("ref_idx")
        status = j[a.key].map(lambda k: "outside_aoi" if pd.isna(k) else
                              ("scored" if k in scored else ("excluded" if k in excluded else "unscored")))
        j["status"] = status
        g = j.groupby("status").agg(n=("ref_idx", "size"), area_m2=("area_m2", "sum")).round(2)
        summ["reference_by_status"] = {k: {"n": int(v.n), "area_m2": float(v.area_m2)} for k, v in g.iterrows()}
        j[j.status == "unscored"][["ref_idx", a.key, "area_m2"]].to_csv(f"{a.out_dir}/unscored_reference.csv", index=False)

    fail = bool(unscored) or bool(bad_ex)
    summ["result"] = "FAIL" if fail else "PASS"
    json.dump(summ, open(f"{a.out_dir}/coverage_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summ.items() if k != "inputs"}, ensure_ascii=False))
    if fail:
        print(f"[gate] FAIL {a.city}: {len(unscored)} of {len(A)} AOI {a.key}s neither scored nor excluded "
              f"(first: {unscored[:5]}); {len(bad_ex)} exclusions without reason. See {a.out_dir}", file=sys.stderr)
        return 2
    print(f"[gate] PASS {a.city}: {len(scored)} scored + {len(excluded)} excluded = {len(A)} AOI {a.key}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
