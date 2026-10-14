"""Build the Cape Town semantic-segmentation training set v1 (aerial_2025).

Mask encoding (uint8): 0 background, 1 PV, 2 hard-negative object (trained as
background, tracked separately), 255 ignore (edge band around positives and
pixels outside the grid / imagery).

Sources
- Positives (Class B): Gao ``*_SAM2_*`` gpkgs. ``sam_fn_review`` and
  ``sam_fn_marker`` are treated as human-reviewed positives here (owner
  decision 2026-10-14), deliberately NOT via
  ``core.training.positive_sources._ct_source_to_label_source`` which maps
  them to untrusted. Edge ignore band: ``--band-b-m`` (default 0.3 m).
- Class A (``--band-a-m``, 0.15 m) only from explicitly passed aligned
  2025-imagery gpkgs (``--class-a``); Li KML labels are eval-only.
- Hard negatives: reviewed detector ``delete`` polygons (inventory CSV),
  optional subtype CSV keyed by (grid, idx).
- Held-out eval grids are written to ``eval/`` and never to train/val.

Imagery: interior tiles from the aerial_2025 folders; last column / row tiles
replaced by the full-square ``--edgefix-dir`` versions, masked to the grid
extent (outside -> 255).
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

BG, PV, HN, IGNORE = 0, 1, 2, 255


# ---------------------------------------------------------------- pure helpers
def band_px(band_m: float, px_m: float) -> int:
    """Ignore-band half-width in pixels (at least 1 when band_m > 0)."""
    if band_m <= 0:
        return 0
    return max(1, int(round(band_m / px_m)))


def compose_mask(pv_b, pv_a, hn, valid, r_b: int, r_a: int) -> np.ndarray:
    """Combine boolean layers into the uint8 label mask.

    Priority: outside-valid -> 255; PV edge bands -> 255; PV -> 1; HN -> 2.
    PV beats HN where they overlap; the band is applied only around PV.
    """
    from scipy import ndimage as ndi

    m = np.zeros(valid.shape, np.uint8)
    m[hn] = HN
    pv = pv_b | pv_a
    m[pv] = PV
    for layer, r in ((pv_b, r_b), (pv_a, r_a)):
        if r > 0 and layer.any():
            st = ndi.generate_binary_structure(2, 1)
            band = ndi.binary_dilation(layer, st, iterations=r) & ~ndi.binary_erosion(layer, st, iterations=r)
            m[band] = IGNORE
    m[~valid] = IGNORE
    return m


def split_grids(grid_batch: dict[str, str], val_frac: float = 0.15, seed: int = 0) -> dict[str, str]:
    """Grid-level train/val split stratified by batch (>=1 val grid per batch with >=4 grids)."""
    rng = np.random.default_rng(seed)
    out = {}
    by = {}
    for g, b in sorted(grid_batch.items()):
        by.setdefault(b, []).append(g)
    for b, gs in sorted(by.items()):
        gs = list(gs)
        rng.shuffle(gs)
        n_val = int(round(len(gs) * val_frac))
        if n_val == 0 and len(gs) >= 4:
            n_val = 1
        for i, g in enumerate(gs):
            out[g] = "val" if i < n_val else "train"
    return out


def dedupe_polygons(gdf, iou_thresh: float = 0.5):
    """Drop near-duplicate polygons (IoU > thresh), keeping the first."""
    keep = []
    geoms = list(gdf.geometry)
    sidx = gdf.sindex
    dropped = set()
    for i, g in enumerate(geoms):
        if i in dropped or g is None or g.is_empty:
            continue
        keep.append(i)
        for j in sidx.query(g):
            if j <= i or j in dropped:
                continue
            h = geoms[j]
            inter = g.intersection(h).area
            uni = g.union(h).area
            if uni > 0 and inter / uni > iou_thresh:
                dropped.add(j)
    return gdf.iloc[keep]


# ------------------------------------------------------------------- building
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zasolar-root", type=Path, default=Path(os.environ.get("ZASOLAR_ROOT", Path(__file__).resolve().parents[2])))
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--g1688-fixed", type=Path, required=True)
    ap.add_argument("--tile-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--edgefix-dir", type=Path, required=True)
    ap.add_argument("--hn-csv", type=Path, required=True)
    ap.add_argument("--hn-subtypes", type=Path, default=None)
    ap.add_argument("--heldout-csv", type=Path, required=True)
    ap.add_argument("--class-a", type=Path, nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=["G1807"])
    ap.add_argument("--band-b-m", type=float, default=0.3)
    ap.add_argument("--band-a-m", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--small-chip", type=int, default=500, help="0 disables the small-chip variant")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grids", nargs="*", default=None, help="debug subset")
    a = ap.parse_args()

    sys.path.insert(0, str(a.zasolar_root))
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely.geometry import box
    from core.grid_utils import get_grid_spec

    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f"refusing to write into non-empty {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)

    held = {"G" + c[3:] for c in pd.read_csv(a.heldout_csv).cpt_id}

    # positives
    pos = {}
    for f in sorted(glob.glob(str(a.labels_dir / "G*_SAM2_*.gpkg"))):
        name = os.path.basename(f)
        g = name.split("_")[0]
        if g in a.exclude:
            continue
        batch = name.split("_")[-1][:6]
        src = a.g1688_fixed if g == "G1688" else Path(f)
        d = gpd.read_file(src)
        d = d.set_crs(4326) if d.crs is None else d.to_crs(4326)
        d = d[d.geometry.notna() & ~d.geometry.is_empty]
        d["geometry"] = d.geometry.buffer(0)
        n0 = len(d)
        d = dedupe_polygons(d.reset_index(drop=True))
        s = d["source"].astype(str).str.lower() if "source" in d else pd.Series(["nan"] * len(d), index=d.index)
        d["label_source"] = np.where(s.str.startswith("sam_fn"), "sam_fn_human", "classB")
        pos.setdefault(g, {"batch": batch, "B": [], "A": [], "n_dup_dropped": 0})
        pos[g]["B"].append(d)
        pos[g]["n_dup_dropped"] += n0 - len(d)
    for f in a.class_a:
        d = gpd.read_file(f)
        d = d.set_crs(4326) if d.crs is None else d.to_crs(4326)
        g = os.path.basename(f).split("_")[0].split(".")[0]
        pos.setdefault(g, {"batch": "classA", "B": [], "A": [], "n_dup_dropped": 0})["A"].append(d)

    # hard negatives
    hn = pd.read_csv(a.hn_csv)
    hn = hn[hn.status == "delete"]
    sub = {}
    if a.hn_subtypes and a.hn_subtypes.exists():
        s = pd.read_csv(a.hn_subtypes)
        m = {"pool_heater_or_pool_cover": "pool_heater", "small_dark_object_thermal_heater": "thermal_heater",
             "roof_section_grey_roof_shadow": "roof", "skylight_on_roof": "skylight", "ground_car_other": "other"}
        for r in s.itertuples():
            sub[(r.grid, r.idx)] = m.get(r.visual_subtype_estimate, "unknown")
    hn_geoms = {}
    cache = {}
    for r in hn.itertuples():
        if r.src not in cache:
            cache[r.src] = gpd.read_file(r.src).to_crs(4326)
        hn_geoms.setdefault(r.grid, []).append((cache[r.src].geometry.iloc[r.idx], sub.get((r.grid, r.idx), "unknown")))

    tile_index = {}
    for td in a.tile_dirs:
        for gd in glob.glob(str(td / "G*")):
            if os.path.isdir(gd):
                tile_index.setdefault(os.path.basename(gd), gd)
    grids = sorted(set(pos) | set(hn_geoms))
    if a.grids:
        grids = [g for g in grids if g in a.grids]
    grids = [g for g in grids if g in tile_index and g not in a.exclude]

    trainval = {g: pos.get(g, {}).get("batch", "hn_only") for g in grids if g not in held}
    split = split_grids(trainval, a.val_frac)
    for g in grids:
        if g in held:
            split[g] = "eval"

    rows, hn_rows, chips = [], [], []
    for g in grids:
        try:
            spec = get_grid_spec("CPT" + g[1:], region="ct")
        except Exception as e:  # noqa: BLE001
            print("SKIP no spec", g, e, flush=True)
            continue
        grid_box = box(spec.xmin, spec.ymin, spec.xmax, spec.ymax)
        P = pos.get(g, {"B": [], "A": []})
        gB = pd.concat(P["B"]) if P["B"] else None
        gA = pd.concat(P["A"]) if P["A"] else None
        H = hn_geoms.get(g, [])
        sp = split[g]
        od = a.out / sp / g
        od.mkdir(parents=True, exist_ok=True)
        for c in range(spec.n_cols):
            for r in range(spec.n_rows):
                edge = c == spec.n_cols - 1 or r == spec.n_rows - 1
                fname = f"CPT{g[1:]}_{c}_{r}_geo.tif"
                img = (a.edgefix_dir / g / fname) if edge else Path(tile_index[g]) / fname
                if not img.exists():
                    print("MISSING", img, flush=True)
                    continue
                with rasterio.open(img) as src:
                    H_, W_ = src.height, src.width
                    tr, crs = src.transform, src.crs
                    px_m = abs(tr.a) * 111320 * np.cos(np.radians((src.bounds.top + src.bounds.bottom) / 2))
                valid = rasterize([(grid_box, 1)], out_shape=(H_, W_), transform=tr, fill=0).astype(bool)

                def ras(geoms):
                    geoms = [x for x in geoms if x is not None and not x.is_empty]
                    if not geoms:
                        return np.zeros((H_, W_), bool)
                    return rasterize([(x, 1) for x in geoms], out_shape=(H_, W_), transform=tr, fill=0).astype(bool)

                pvB = ras(list(gB.geometry)) if gB is not None else np.zeros((H_, W_), bool)
                pvA = ras(list(gA.geometry)) if gA is not None else np.zeros((H_, W_), bool)
                hnm = ras([x for x, _ in H])
                mask = compose_mask(pvB, pvA, hnm, valid, band_px(a.band_b_m, px_m), band_px(a.band_a_m, px_m))
                mpath = od / fname.replace("_geo.tif", "_mask.tif")
                with rasterio.open(mpath, "w", driver="GTiff", width=W_, height=H_, count=1, dtype="uint8",
                                   crs=crs, transform=tr, compress="DEFLATE", nodata=IGNORE) as dst:
                    dst.write(mask, 1)
                cnt = np.bincount(mask.ravel(), minlength=256)
                rows.append(dict(grid=g, split=sp, batch=pos.get(g, {}).get("batch", "hn_only"), col=c, row=r,
                                 edge_tile=edge, image=str(img), mask=str(mpath), width=W_, height=H_,
                                 px_m=round(px_m, 4), n_bg=int(cnt[BG]), n_pv=int(cnt[PV]), n_hn=int(cnt[HN]),
                                 n_ignore=int(cnt[IGNORE])))
                if a.small_chip:
                    k = a.small_chip
                    for yy in range(0, H_ - k + 1, k):
                        for xx in range(0, W_ - k + 1, k):
                            mm = mask[yy:yy + k, xx:xx + k]
                            if (mm == IGNORE).mean() > 0.9:
                                continue
                            has_obj = bool((mm == PV).any() or (mm == HN).any())
                            if not has_obj and ((xx // k + yy // k) % 4):
                                continue  # keep 1/4 of pure-background chips
                            chips.append(dict(grid=g, split=sp, image=str(img), mask=str(mpath), x=xx, y=yy, size=k,
                                              n_pv=int((mm == PV).sum()), n_hn=int((mm == HN).sum()),
                                              n_ignore=int((mm == IGNORE).sum())))
        for geom, st in H:
            hn_rows.append(dict(grid=g, split=sp, subtype=st, area_m2=None))
        print("done", g, sp, flush=True)

    M = pd.DataFrame(rows)
    M.to_csv(a.out / "manifest.csv", index=False)
    pd.DataFrame(chips).to_csv(a.out / f"chips_{a.small_chip}px_index.csv", index=False)
    pd.DataFrame(hn_rows).to_csv(a.out / "hard_negatives.csv", index=False)
    summ = {
        "grids": M.groupby("split").grid.nunique().to_dict(),
        "tiles": M.groupby("split").size().to_dict(),
        "pv_px": M.groupby("split").n_pv.sum().to_dict(),
        "hn_px": M.groupby("split").n_hn.sum().to_dict(),
        "ignore_frac": (M.groupby("split").n_ignore.sum() / (M.width * M.height).groupby(M.split).sum()).round(4).to_dict(),
        "small_chips": pd.DataFrame(chips).groupby("split").size().to_dict() if chips else {},
        "hn_by_subtype": pd.DataFrame(hn_rows).groupby(["split", "subtype"]).size().unstack(fill_value=0).to_dict() if hn_rows else {},
        "heldout": sorted(held),
        "dup_dropped": {g: v["n_dup_dropped"] for g, v in pos.items() if v.get("n_dup_dropped")},
        "args": {k: str(v) for k, v in vars(a).items()},
    }
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "args"}, indent=1, default=str))


if __name__ == "__main__":
    main()
