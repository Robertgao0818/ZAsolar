"""Cape Town semantic-segmentation training set v2 (resolution-agnostic).

Differences from v1:
- Imagery comes from a tile manifest (CSV with a grid column and an image-path
  column; any CRS / resolution, e.g. native 0.05 m). Labels are reprojected to
  each tile's CRS and rasterised onto that tile's own pixel grid. Ignore bands
  are metric (+-0.3 m Class B, +-0.15 m Class A) converted per tile.
- Solar water heaters (separate polygon table, confidence confirmed/likely)
  are 255 (ignore) everywhere: removed from PV (Class B labels flagged by the
  GT heater audit) and from hard negatives (reviewed FP deletes).
- Hard-negative subtypes come from a (grid, idx, subtype) CSV; pool heaters
  stay 2 and are tagged ``pool_heater`` so they can be remapped later.
- Pixels outside the grid extent or with nodata (all bands 0) are 255.

Mask: uint8, 0 bg, 1 PV, 2 hard negative, 255 ignore. Masks are written next
to nothing in the sources; images are referenced, not copied.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.build_ct_seg_trainset_v1 import (  # noqa: E402
    HN, IGNORE, PV, band_px, compose_mask, dedupe_polygons, split_grids,
)

GRID_COLS = ("grid", "grid_id", "gid", "cpt_id", "cell_id")
PATH_COLS = ("image", "path", "tif", "tile_path", "file", "filepath")


def norm_grid(x: str) -> str:
    """'CPT1631' / 'G1631' / 'cpt1631' -> 'G1631'."""
    s = str(x).strip().upper()
    if s.startswith("CPT"):
        s = "G" + s[3:]
    return s


def read_tile_manifest(path: Path, root: Path | None = None) -> pd.DataFrame:
    """Return DataFrame[grid, image] from a manifest CSV with flexible column names."""
    m = pd.read_csv(path)
    lc = {c.lower(): c for c in m.columns}
    gcol = next((lc[c] for c in GRID_COLS if c in lc), None)
    pcol = next((lc[c] for c in PATH_COLS if c in lc), None)
    if pcol is None:
        raise ValueError(f"no image-path column in {path}: {list(m.columns)}")
    out = pd.DataFrame({"image": m[pcol].astype(str)})
    if root is not None:
        out["image"] = [p if os.path.isabs(p) else str(root / p) for p in out.image]
    if gcol is not None:
        out["grid"] = m[gcol].map(norm_grid)
    else:  # infer from file name CPT####_...
        out["grid"] = [norm_grid(os.path.basename(p).split("_")[0]) for p in out.image]
    return out[["grid", "image"]]


def pixel_size_m(transform, crs, lat: float) -> tuple[float, float]:
    """Pixel width/height in metres for geographic or projected CRS."""
    if crs is not None and crs.is_geographic:
        return abs(transform.a) * 111320 * np.cos(np.radians(lat)), abs(transform.e) * 110574
    return abs(transform.a), abs(transform.e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zasolar-root", type=Path, default=Path(os.environ.get("ZASOLAR_ROOT", Path(__file__).resolve().parents[2])))
    ap.add_argument("--tile-manifest", type=Path, required=True)
    ap.add_argument("--tile-root", type=Path, default=None, help="prefix for relative manifest paths")
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--g1688-fixed", type=Path, required=True)
    ap.add_argument("--hn-csv", type=Path, required=True)
    ap.add_argument("--hn-subtypes", type=Path, required=True, help="CSV grid,idx,subtype")
    ap.add_argument("--swh-gpkg", type=Path, required=True, help="solar water heater polygons (grid, origin, ref, confidence)")
    ap.add_argument("--heldout-csv", type=Path, required=True)
    ap.add_argument("--class-a", type=Path, nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=["G1807", "G1189", "G1190", "G1238"])
    ap.add_argument("--band-b-m", type=float, default=0.3)
    ap.add_argument("--band-a-m", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--small-chip", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grids", nargs="*", default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(a.zasolar_root))
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import box
    from core.grid_utils import get_grid_spec

    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f"refusing to write into non-empty {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)

    held = {norm_grid(c) for c in pd.read_csv(a.heldout_csv).cpt_id}
    swh = gpd.read_file(a.swh_gpkg).to_crs(4326)
    swh_label_refs = set(swh.loc[swh.origin == "classB_label", "ref"])
    swh_fp_refs = set(swh.loc[swh.origin == "fp_review_delete", "ref"])

    pos = {}
    n_swh_removed_pv = 0
    for f in sorted(glob.glob(str(a.labels_dir / "G*_SAM2_*.gpkg"))):
        name = os.path.basename(f)
        g = name.split("_")[0]
        if g in a.exclude:
            continue
        src = a.g1688_fixed if g == "G1688" else Path(f)
        d = gpd.read_file(src)
        d = d.set_crs(4326) if d.crs is None else d.to_crs(4326)
        d = d.reset_index(drop=True)
        is_swh = np.array([f"{name}#{i}" in swh_label_refs for i in range(len(d))])
        n_swh_removed_pv += int(is_swh.sum())
        d = d[~is_swh]
        d = d[d.geometry.notna() & ~d.geometry.is_empty].copy()
        d["geometry"] = d.geometry.buffer(0)
        d = dedupe_polygons(d.reset_index(drop=True))
        pos[g] = {"batch": name.split("_")[-1][:6], "B": d, "A": None}
    for f in a.class_a:
        d = gpd.read_file(f)
        d = d.set_crs(4326) if d.crs is None else d.to_crs(4326)
        g = norm_grid(os.path.basename(f).split("_")[0].split(".")[0])
        pos.setdefault(g, {"batch": "classA", "B": None, "A": None})["A"] = d

    hn = pd.read_csv(a.hn_csv)
    hn = hn[hn.status == "delete"]
    sub = {(r.grid, int(r.idx)): r.subtype for r in pd.read_csv(a.hn_subtypes).itertuples()}
    hn_geoms, cache = {}, {}
    for r in hn.itertuples():
        if f"{r.src}#{r.idx}" in swh_fp_refs:
            continue
        if r.src not in cache:
            cache[r.src] = gpd.read_file(r.src).to_crs(4326)
        hn_geoms.setdefault(r.grid, []).append((cache[r.src].geometry.iloc[r.idx], sub.get((r.grid, int(r.idx)), "unknown")))

    tiles = read_tile_manifest(a.tile_manifest, a.tile_root)
    tiles = tiles[[os.path.exists(p) for p in tiles.image]]
    grids = sorted((set(pos) | set(hn_geoms)) & set(tiles.grid))
    grids = [g for g in grids if g not in a.exclude and (not a.grids or g in a.grids)]
    split = split_grids({g: pos.get(g, {}).get("batch", "hn_only") for g in grids if g not in held}, a.val_frac)
    for g in grids:
        if g in held:
            split[g] = "eval"

    rows, chips, hn_rows = [], [], []
    for g in grids:
        try:
            spec = get_grid_spec("CPT" + g[1:], region="ct")
        except Exception as e:  # noqa: BLE001
            print("SKIP no spec", g, e, flush=True)
            continue
        grid_poly = gpd.GeoSeries([box(spec.xmin, spec.ymin, spec.xmax, spec.ymax)], crs=4326)
        P = pos.get(g, {"B": None, "A": None})
        S = swh[swh.grid == g]
        H = hn_geoms.get(g, [])
        Hg = gpd.GeoSeries([x for x, _ in H], crs=4326) if H else None
        sp = split[g]
        od = a.out / sp / g
        od.mkdir(parents=True, exist_ok=True)
        for img in sorted(tiles[tiles.grid == g].image):
            with rasterio.open(img) as src:
                Hh, Ww, tr, crs = src.height, src.width, src.transform, src.crs
                lat = (src.bounds.top + src.bounds.bottom) / 2 if crs.is_geographic else -34.0
                pxw, pxh = pixel_size_m(tr, crs, lat)
                px_m = (pxw + pxh) / 2
                arr = src.read(out_shape=(src.count, Hh, Ww))
            nodata = (arr[:3] == 0).all(0) if arr.shape[0] >= 3 else (arr[0] == 0)

            def ras(geoms):
                if geoms is None or len(geoms) == 0:
                    return np.zeros((Hh, Ww), bool)
                gs = [x for x in gpd.GeoSeries(geoms, crs=4326).to_crs(crs) if x is not None and not x.is_empty]
                if not gs:
                    return np.zeros((Hh, Ww), bool)
                return rasterize([(x, 1) for x in gs], out_shape=(Hh, Ww), transform=tr, fill=0, all_touched=False).astype(bool)

            valid = ras(list(grid_poly)) & ~nodata
            pvB = ras(list(P["B"].geometry)) if P.get("B") is not None else np.zeros((Hh, Ww), bool)
            pvA = ras(list(P["A"].geometry)) if P.get("A") is not None else np.zeros((Hh, Ww), bool)
            hnm = ras(list(Hg)) if Hg is not None else np.zeros((Hh, Ww), bool)
            swm = ras(list(S.geometry)) if len(S) else np.zeros((Hh, Ww), bool)
            mask = compose_mask(pvB, pvA, hnm, valid, band_px(a.band_b_m, px_m), band_px(a.band_a_m, px_m))
            mask[swm] = IGNORE
            stem = Path(img).stem.replace("_geo", "")
            mpath = od / f"{stem}_mask.tif"
            with rasterio.open(mpath, "w", driver="GTiff", width=Ww, height=Hh, count=1, dtype="uint8",
                               crs=crs, transform=tr, compress="DEFLATE", nodata=IGNORE) as dst:
                dst.write(mask, 1)
            cnt = np.bincount(mask.ravel(), minlength=256)
            rows.append(dict(grid=g, split=sp, batch=pos.get(g, {}).get("batch", "hn_only"), image=img, mask=str(mpath),
                             width=Ww, height=Hh, px_m=round(px_m, 4), n_bg=int(cnt[0]), n_pv=int(cnt[PV]),
                             n_hn=int(cnt[HN]), n_ignore=int(cnt[IGNORE]), n_swh_px=int(swm.sum())))
            k = a.small_chip
            if k and Hh >= k and Ww >= k:
                for yy in range(0, Hh - k + 1, k):
                    for xx in range(0, Ww - k + 1, k):
                        mm = mask[yy:yy + k, xx:xx + k]
                        if (mm == IGNORE).mean() > 0.9:
                            continue
                        if not ((mm == PV).any() or (mm == HN).any()) and ((xx // k + yy // k) % 4):
                            continue
                        chips.append(dict(grid=g, split=sp, image=img, mask=str(mpath), x=xx, y=yy, size=k,
                                          n_pv=int((mm == PV).sum()), n_hn=int((mm == HN).sum())))
        for _, st in H:
            hn_rows.append(dict(grid=g, split=sp, subtype=st))
        print("done", g, sp, flush=True)

    M = pd.DataFrame(rows)
    M.to_csv(a.out / "manifest.csv", index=False)
    pd.DataFrame(chips).to_csv(a.out / f"chips_{a.small_chip}px_index.csv", index=False)
    pd.DataFrame(hn_rows).to_csv(a.out / "hard_negatives.csv", index=False)
    sw = swh[swh.grid.isin(grids)].copy()
    sw["split"] = sw.grid.map(split)
    sw.to_file(a.out / "solar_water_heaters.gpkg")
    sw.drop(columns="geometry").to_csv(a.out / "solar_water_heaters.csv", index=False)
    px = (M.width * M.height).groupby(M.split).sum() if len(M) else pd.Series(dtype=float)
    summ = {
        "grids": M.groupby("split").grid.nunique().to_dict() if len(M) else {},
        "tiles": M.groupby("split").size().to_dict() if len(M) else {},
        "px_m": sorted(M.px_m.round(3).unique().tolist()) if len(M) else [],
        "pv_px": M.groupby("split").n_pv.sum().to_dict() if len(M) else {},
        "hn_px": M.groupby("split").n_hn.sum().to_dict() if len(M) else {},
        "ignore_frac": (M.groupby("split").n_ignore.sum() / px).round(4).to_dict() if len(M) else {},
        "hn_by_subtype": pd.DataFrame(hn_rows).groupby(["split", "subtype"]).size().unstack(fill_value=0).to_dict() if hn_rows else {},
        "swh_by_split_conf": sw.groupby(["split", "confidence"]).size().unstack(fill_value=0).to_dict() if len(sw) else {},
        "swh_removed_from_pv_labels": n_swh_removed_pv,
        "heldout": sorted(held),
        "args": {k: str(v) for k, v in vars(a).items()},
    }
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "args"}, indent=1, default=str))


if __name__ == "__main__":
    main()
