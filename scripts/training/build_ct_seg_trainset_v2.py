"""Cape Town semantic-segmentation training set v2 (resolution-agnostic).

Differences from v1:
- Imagery comes from a tile manifest (CSV with a grid column and an image-path
  column; any CRS / resolution, e.g. native 0.05 m). Labels are reprojected to
  each tile's CRS and rasterised onto that tile's own pixel grid. Ignore bands
  are metric (+-0.3 m Class B, +-0.15 m Class A) converted per tile.
- Solar water heaters and pool heaters are one class, solar thermal collector.
  Polygons are written to ``collectors.gpkg`` / ``collectors.csv`` (not as PV).
  Human-confirmed collectors (``confidence == confirmed``, or an HN subtype
  ``pool_heater``) are mask value 2 (hard negative). There is no ignore band
  around hard negatives. Likely or unaudited collectors are mask value 255
  (ignore), painted after the PV/HN compose.
  ``classB_label`` refs are stripped out of the PV labels, and any collector
  ref is stripped out of the generic HN delete list; confirmed collectors are
  then re-added as value 2. A pool heater that is not already in ``--swh-gpkg``
  is still value 2 and is appended to the collector table with
  confidence ``confirmed`` and kind ``collector``.
- Class B SAM2 polygons whose ``source`` starts with ``sam_fn``
  (``sam_fn_review`` / ``sam_fn_marker``, owner decision 2026-10-15) use the
  Class A band (0.15 m). Other Class B polygons stay on the 0.3 m band.
  v1 still keeps those polygons in Class B.
- Pixels outside the grid extent or with nodata (all bands 0) are 255. With
  ``--native-dir`` (EPSG:32734, 0.05 m, shared 100 m tiles, per-grid vrt =
  cell + 50 m buffer) one mask is written per (grid, tile); the 50 m buffer
  outside the cell is 255 because neighbouring cells' labels may be absent
  there (the same tile gets its own core labels under the neighbour's grid).
- Held-out eval grids (the test-grid list) are written to ``eval/`` and never
  enter train/val.

Mask: uint8, 0 bg, 1 PV, 2 hard negative, 255 ignore. Masks are written next
to nothing in the sources; images are referenced, not copied. ``--swh-gpkg``
is the collector polygon table (flag name kept).
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


def tiles_from_native_dir(native_dir: Path) -> pd.DataFrame:
    """DataFrame[grid, image] from ``vrt/CPT####.vrt`` files (only complete grids
    have a vrt). Each vrt lists the shared 100 m tiles (cell bbox + 50 m buffer)."""
    import xml.etree.ElementTree as ET

    rows = []
    for v in sorted(glob.glob(str(native_dir / "vrt" / "CPT*.vrt"))):
        g = norm_grid(Path(v).stem)
        for el in ET.parse(v).getroot().iter("SourceFilename"):
            f = el.text
            if el.get("relativeToVRT") == "1":
                f = os.path.normpath(os.path.join(os.path.dirname(v), f))
            rows.append(dict(grid=g, image=f))
    return pd.DataFrame(rows, columns=["grid", "image"]).drop_duplicates()


def pixel_size_m(transform, crs, lat: float) -> tuple[float, float]:
    """Pixel width/height in metres for geographic or projected CRS."""
    if crs is not None and crs.is_geographic:
        return abs(transform.a) * 111320 * np.cos(np.radians(lat)), abs(transform.e) * 110574
    return abs(transform.a), abs(transform.e)


def normalize_collector_confidence(confidence) -> str:
    """Map a collector confidence label to confirmed / likely / unaudited."""
    if confidence is None or (isinstance(confidence, float) and pd.isna(confidence)):
        return "unaudited"
    c = str(confidence).strip().lower()
    if c == "confirmed":
        return "confirmed"
    if c == "likely":
        return "likely"
    return "unaudited"


def is_pool_heater_subtype(subtype) -> bool:
    if subtype is None or (isinstance(subtype, float) and pd.isna(subtype)):
        return False
    return str(subtype).strip().lower() == "pool_heater"


def collector_mask_value(confidence, *, is_pool_heater: bool = False) -> int:
    """Confirmed collector → hard negative (2). Likely or unaudited → ignore (255).

    A pool-heater hard negative is human-confirmed even when the solar-water-heater
    row for the same ref says likely or is blank (owner 2026-10-15). No ignore
    band is drawn around the value-2 pixels; that is ``compose_mask``'s rule.
    """
    if is_pool_heater or normalize_collector_confidence(confidence) == "confirmed":
        return HN
    return IGNORE


def is_sam_fn_source(source) -> bool:
    """True for human sam_fn* catches (sam_fn_review, sam_fn_marker, ...)."""
    if source is None or (isinstance(source, float) and pd.isna(source)):
        return False
    return str(source).strip().lower().startswith("sam_fn")


def route_sam_fn_polygons(gdf):
    """Split a Class B label frame into ``(class_b, class_a)``.

    Rows whose ``source`` starts with ``sam_fn`` go to Class A (0.15 m band).
    A missing ``source`` column, a null source, or any other source stays in
    Class B (0.3 m). Owner decision 2026-10-15. v1 does not call this.
    """
    if gdf is None:
        return None, None
    empty = gdf.iloc[0:0]
    if len(gdf) == 0 or "source" not in gdf.columns:
        return gdf, empty
    mask = gdf["source"].map(is_sam_fn_source)
    return gdf.loc[~mask].copy(), gdf.loc[mask].copy()


def concat_layer(existing, extra):
    """Append polygon frames. ``None`` or empty on either side is a no-op."""
    if extra is None or len(extra) == 0:
        return existing
    if existing is None or len(existing) == 0:
        return extra
    import geopandas as gpd

    crs = getattr(existing, "crs", None) or getattr(extra, "crs", None)
    out = pd.concat([existing, extra], ignore_index=True)
    if "geometry" in out.columns:
        return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)
    return out


def merge_class_a(existing, explicit):
    """Keep sam_fn rows already on Class A, then apply one explicit Class A file.

    A later ``--class-a`` file replaces earlier explicit Class A rows (last
    file wins, as before) and does not drop ``sam_fn*`` rows.
    """
    sam = None
    if existing is not None and len(existing) and "source" in existing.columns:
        sam = existing.loc[existing["source"].map(is_sam_fn_source)]
    if sam is not None and len(sam) == 0:
        sam = None
    return concat_layer(sam, explicit)


def assemble_collectors(swh, pool_heaters: list[dict]):
    """Solar-water-heater rows plus pool-heater HN rows missing from that table.

    Output columns used for counting: ``confidence`` in
    {confirmed, likely, unaudited}, ``kind`` = ``collector``, ``subtype``
    (``pool_heater`` or ``solar_water_heater``), ``mask_value`` 2 or 255,
    and ``confidence_src`` (the value before a pool-heater upgrade).
    A pool heater whose ``ref`` is already in ``swh`` is not duplicated; its
    confidence is set to confirmed so the count matches the mask.
    """
    import geopandas as gpd

    sw = swh.copy()
    if "confidence" not in sw.columns:
        sw["confidence"] = "unaudited"
    if "ref" not in sw.columns:
        sw["ref"] = None
    ph_refs = {str(p["ref"]) for p in pool_heaters}
    is_ph = sw["ref"].astype(str).isin(ph_refs).to_numpy()
    sw["confidence_src"] = sw["confidence"]
    sw["confidence"] = [
        "confirmed" if ph else normalize_collector_confidence(c)
        for ph, c in zip(is_ph, sw["confidence"])
    ]
    sw["kind"] = "collector"
    if "subtype" not in sw.columns:
        sw["subtype"] = np.where(is_ph, "pool_heater", "solar_water_heater")
    else:
        subtype = sw["subtype"].astype(object).copy()
        if is_ph.any():
            subtype.iloc[np.flatnonzero(is_ph)] = "pool_heater"
        blank = subtype.isna() | (subtype.astype(str).str.strip() == "")
        fill = np.flatnonzero(blank.to_numpy() & ~is_ph)
        if len(fill):
            subtype.iloc[fill] = "solar_water_heater"
        sw["subtype"] = subtype.to_numpy()
    sw["mask_value"] = [int(collector_mask_value(c)) for c in sw["confidence"]]
    crs = getattr(sw, "crs", None) or getattr(swh, "crs", None) or "EPSG:4326"
    swh_refs = set(sw["ref"].astype(str))
    extras = [p for p in pool_heaters if str(p["ref"]) not in swh_refs]
    frames = [sw]
    if extras:
        frames.append(gpd.GeoDataFrame(
            [{
                "grid": p.get("grid"),
                "ref": p["ref"],
                "origin": p.get("origin", "hn_pool_heater"),
                "confidence": "confirmed",
                "confidence_src": "confirmed",
                "kind": "collector",
                "subtype": "pool_heater",
                "mask_value": HN,
                "geometry": p["geometry"],
            } for p in extras],
            geometry="geometry", crs=crs,
        ))
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zasolar-root", type=Path, default=Path(os.environ.get("ZASOLAR_ROOT", Path(__file__).resolve().parents[2])))
    src_grp = ap.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--tile-manifest", type=Path)
    src_grp.add_argument("--native-dir", type=Path, help="aerial_2025_native005 layout (vrt/CPT####.vrt + tiles/)")
    ap.add_argument("--tile-root", type=Path, default=None, help="prefix for relative manifest paths")
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--g1688-fixed", type=Path, required=True)
    ap.add_argument("--hn-csv", type=Path, required=True)
    ap.add_argument("--hn-subtypes", type=Path, required=True, help="CSV grid,idx,subtype")
    ap.add_argument("--swh-gpkg", type=Path, required=True,
                    help="collector polygons (solar water heaters; grid, origin, ref, confidence). Pool heaters are merged from --hn-subtypes.")
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
    swh_label_refs = set(swh.loc[swh.origin == "classB_label", "ref"].astype(str))
    swh_refs = set(swh["ref"].astype(str))

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
        b_part, a_part = route_sam_fn_polygons(d)
        pos[g] = {
            "batch": name.split("_")[-1][:6],
            "B": None if b_part is None or len(b_part) == 0 else b_part,
            "A": None if a_part is None or len(a_part) == 0 else a_part,
        }
    for f in a.class_a:
        d = gpd.read_file(f)
        d = d.set_crs(4326) if d.crs is None else d.to_crs(4326)
        g = norm_grid(os.path.basename(f).split("_")[0].split(".")[0])
        entry = pos.setdefault(g, {"batch": "classA", "B": None, "A": None})
        entry["A"] = merge_class_a(entry.get("A"), d)

    hn = pd.read_csv(a.hn_csv)
    hn = hn[hn.status == "delete"]
    sub = {(r.grid, int(r.idx)): r.subtype for r in pd.read_csv(a.hn_subtypes).itertuples()}
    hn_geoms, cache, pool_heaters = {}, {}, []
    for r in hn.itertuples():
        ref = f"{r.src}#{r.idx}"
        subtype = sub.get((r.grid, int(r.idx)), "unknown")
        # Collectors leave the generic HN list. Confirmed ones are re-added
        # from the collector table (value 2); pool heaters missing there too.
        if ref in swh_refs or is_pool_heater_subtype(subtype):
            if is_pool_heater_subtype(subtype) and ref not in swh_refs:
                if r.src not in cache:
                    cache[r.src] = gpd.read_file(r.src).to_crs(4326)
                pool_heaters.append(dict(
                    grid=r.grid, ref=ref, origin="hn_pool_heater",
                    geometry=cache[r.src].geometry.iloc[r.idx],
                ))
            elif is_pool_heater_subtype(subtype):
                pool_heaters.append(dict(grid=r.grid, ref=ref, geometry=None))
            continue
        if r.src not in cache:
            cache[r.src] = gpd.read_file(r.src).to_crs(4326)
        hn_geoms.setdefault(r.grid, []).append((cache[r.src].geometry.iloc[r.idx], subtype))
    collectors_all = assemble_collectors(swh, pool_heaters)

    tiles = tiles_from_native_dir(a.native_dir) if a.native_dir else read_tile_manifest(a.tile_manifest, a.tile_root)
    if tiles.empty:
        sys.exit("no tiles (no complete grid vrt yet?)")
    tiles = tiles[[os.path.exists(p) for p in tiles.image]]
    collector_grids = set(collectors_all.grid.dropna().astype(str)) if len(collectors_all) else set()
    grids = sorted((set(pos) | set(hn_geoms) | collector_grids) & set(tiles.grid))
    grids = [g for g in grids if g not in a.exclude and (not a.grids or g in a.grids)]
    split = split_grids({g: pos.get(g, {}).get("batch", "hn_only") for g in grids if g not in held}, a.val_frac)
    for g in grids:
        if g in held:
            split[g] = "eval"
    collectors = collectors_all[collectors_all.grid.isin(grids)].copy() if len(collectors_all) else collectors_all
    if len(collectors):
        collectors["split"] = collectors.grid.map(split)

    rows, chips, hn_rows = [], [], []
    for g in grids:
        try:
            spec = get_grid_spec("CPT" + g[1:], region="ct")
        except Exception as e:  # noqa: BLE001
            print("SKIP no spec", g, e, flush=True)
            continue
        grid_poly = gpd.GeoSeries([box(spec.xmin, spec.ymin, spec.xmax, spec.ymax)], crs=4326)
        P = pos.get(g, {"B": None, "A": None})
        C = collectors[collectors.grid == g] if len(collectors) else collectors
        conf_geoms = list(C.loc[C.mask_value == HN].geometry) if len(C) else []
        ign_geoms = list(C.loc[C.mask_value == IGNORE].geometry) if len(C) else []
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
            if not valid.any():
                continue  # tile lies entirely in the buffer / nodata
            pvB = ras(list(P["B"].geometry)) if P.get("B") is not None else np.zeros((Hh, Ww), bool)
            pvA = ras(list(P["A"].geometry)) if P.get("A") is not None else np.zeros((Hh, Ww), bool)
            hnm = ras(list(Hg)) if Hg is not None else np.zeros((Hh, Ww), bool)
            col_hn = ras(conf_geoms)
            col_ign = ras(ign_geoms)
            # Confirmed collectors join the HN layer (value 2). Likely /
            # unaudited collectors are painted 255 after compose, so they are
            # not trained as background and are not given an edge band.
            mask = compose_mask(pvB, pvA, hnm | col_hn, valid, band_px(a.band_b_m, px_m), band_px(a.band_a_m, px_m))
            mask[col_ign] = IGNORE
            stem = Path(img).stem.replace("_geo", "")
            if a.native_dir:
                stem = f"CPT{g[1:]}__{stem}"
            mpath = od / f"{stem}_mask.tif"
            with rasterio.open(mpath, "w", driver="GTiff", width=Ww, height=Hh, count=1, dtype="uint8",
                               crs=crs, transform=tr, compress="DEFLATE", nodata=IGNORE) as dst:
                dst.write(mask, 1)
            cnt = np.bincount(mask.ravel(), minlength=256)
            rows.append(dict(grid=g, split=sp, batch=pos.get(g, {}).get("batch", "hn_only"), image=img, mask=str(mpath),
                             width=Ww, height=Hh, px_m=round(px_m, 4), n_bg=int(cnt[0]), n_pv=int(cnt[PV]),
                             n_hn=int(cnt[HN]), n_ignore=int(cnt[IGNORE]),
                             n_collector_hn_px=int((col_hn & (mask == HN)).sum()),
                             n_collector_ignore_px=int((col_ign & (mask == IGNORE)).sum())))
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
    if len(collectors) == 0:
        collectors = gpd.GeoDataFrame(collectors, geometry="geometry", crs=getattr(collectors, "crs", None) or "EPSG:4326")
    collectors.to_file(a.out / "collectors.gpkg")
    collectors.drop(columns="geometry").to_csv(a.out / "collectors.csv", index=False)
    px = (M.width * M.height).groupby(M.split).sum() if len(M) else pd.Series(dtype=float)
    summ = {
        "grids": M.groupby("split").grid.nunique().to_dict() if len(M) else {},
        "tiles": M.groupby("split").size().to_dict() if len(M) else {},
        "px_m": sorted(M.px_m.round(3).unique().tolist()) if len(M) else [],
        "pv_px": M.groupby("split").n_pv.sum().to_dict() if len(M) else {},
        "hn_px": M.groupby("split").n_hn.sum().to_dict() if len(M) else {},
        "ignore_frac": (M.groupby("split").n_ignore.sum() / px).round(4).to_dict() if len(M) else {},
        "hn_by_subtype": pd.DataFrame(hn_rows).groupby(["split", "subtype"]).size().unstack(fill_value=0).to_dict() if hn_rows else {},
        "collectors_by_split_conf": collectors.groupby(["split", "confidence"]).size().unstack(fill_value=0).to_dict() if len(collectors) else {},
        "collector_neg_px": M.groupby("split").n_collector_hn_px.sum().to_dict() if len(M) else {},
        "collector_ignore_px": M.groupby("split").n_collector_ignore_px.sum().to_dict() if len(M) else {},
        "collectors_removed_from_pv_labels": n_swh_removed_pv,
        "swh_removed_from_pv_labels": n_swh_removed_pv,
        "heldout": sorted(held),
        "args": {k: str(v) for k, v in vars(a).items()},
    }
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "args"}, indent=1, default=str))


if __name__ == "__main__":
    main()
