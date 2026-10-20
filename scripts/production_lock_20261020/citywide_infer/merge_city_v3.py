"""CT citywide avg2 + h0.9_0.5 + A3 merge, v3 (citywide cross-block join). Replaces the gated per-block `merge`
stage of ct_city_infer_v2.py (kept unchanged) after Codex r3 showed block+halo ownership can double-count
long objects.

Semantics = xtile_merge.merge_plan over ALL tiles at once (unmodified rule): components of avg2 >= lo
(8-connected) inside each tile, joined across every tile edge/corner where labelled pixels touch, kept if the
merged component's max >= hi (h0.9_0.5: hi 0.9, lo 0.5). avg2 = (pv_convnext + pv_b2)/2/255 float, as bench.py.
Memory-light: pass 1 streams tiles (<= --workers processes) and keeps only per-component (max, n_px) and the
labelled pixels of each tile's 1-px border strips; union-find over those. Pass 2 streams tiles again and
polygonises; single-tile objects are final at once, multi-tile objects are dissolved at the end.
Each object is owned by exactly one 1 km block: the block containing its representative point.
Outputs (new dir, never overwritten): objects_v3.gpkg (layer objects), block_counts_v3.csv, summary_v3.json.
CPU only.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ct_xtile_merge'))
import ct_city_infer_v2 as C  # noqa: E402  (tile_xy, out_path, load_avg2, read_tile_list, publish, MODELS)

CFG = 'h0.9_0.5'; TILE_M = 100; BLOCK_M = 1000
S8 = np.ones((3, 3), bool)


def _strips(lab):
    """1-px border strips of a label raster: top row, bottom row, left col, right col (int32 copies)."""
    return dict(t=lab[0].astype(np.int32), b=lab[-1].astype(np.int32), l=lab[:, 0].astype(np.int32), r=lab[:, -1].astype(np.int32))


def _sparse(a):
    i = np.flatnonzero(a); return (i.astype(np.int32), a[i].astype(np.int32), a.shape[0])


def _dense(s):
    i, v, n = s; a = np.zeros(n, np.int32); a[i] = v; return a


def pass1_tile(args):
    """-> (tile, nl, cmax[nl], cnpx[nl], sparse strips, shape)."""
    from scipy import ndimage
    from xtile_merge import parse_cfg
    out, tile, loader = args
    hi, lo = parse_cfg(CFG)
    p = loader(out, tile)
    lab, nl = ndimage.label(p >= lo, S8)
    if nl == 0:
        return tile, 0, np.zeros(0, np.float32), np.zeros(0, np.int64), None, p.shape
    idx = np.arange(1, nl + 1)
    cmax = np.asarray(ndimage.maximum(p, lab, idx), np.float32)
    cnpx = np.bincount(lab.ravel(), minlength=nl + 1)[1:].astype(np.int64)
    return tile, nl, cmax, cnpx, {k: _sparse(v) for k, v in _strips(lab).items()}, p.shape


class UF:
    def __init__(self, n): self.p = np.arange(n, dtype=np.int64)
    def find(self, a):
        p = self.p; r = a
        while p[r] != r: r = p[r]
        while p[a] != r: p[a], a = r, p[a]
        return r
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[max(ra, rb)] = min(ra, rb)


def _pairs_1d(a, b):
    """8-connected pairs between two facing 1-px strips a (tile A side) and b (neighbour side), same length."""
    out = []
    for d in (-1, 0, 1):
        if d < 0: x, y = a[-d:], b[:d]
        elif d > 0: x, y = a[:-d], b[d:]
        else: x, y = a, b
        m = (x > 0) & (y > 0)
        if m.any(): out.append(np.stack([x[m], y[m]], 1))
    return np.unique(np.concatenate(out), axis=0) if out else np.zeros((0, 2), np.int32)


def plan_city(results):
    """results: list of pass1 tuples. Returns dict with base offsets, root, keep, rmax, cnpx, joins."""
    from xtile_merge import parse_cfg
    hi, lo = parse_cfg(CFG)
    base, nls, shapes, strips = {}, {}, {}, {}
    cmax_all, cnpx_all = [], []; off = 0
    for tile, nl, cmax, cnpx, st, shape in results:
        base[tile] = off; nls[tile] = nl; shapes[tile] = shape; off += nl
        cmax_all.append(cmax); cnpx_all.append(cnpx)
        if st is not None: strips[tile] = st
    n = off; uf = UF(n)
    cmax = np.concatenate(cmax_all) if n else np.zeros(0, np.float32)
    cnpx = np.concatenate(cnpx_all) if n else np.zeros(0, np.int64)
    by_xy = {C.tile_xy(t): t for t in strips}
    unions = 0
    for (x, y), t in by_xy.items():
        A = strips[t]; ba = base[t]
        def U(pairs, tb):
            nonlocal unions
            bb = base[tb]
            for la, lb in pairs: uf.union(ba + int(la) - 1, bb + int(lb) - 1); unions += 1
        r = by_xy.get((x + TILE_M, y))               # east neighbour: A right col vs B left col
        if r: U(_pairs_1d(_dense(A['r']), _dense(strips[r]['l'])), r)
        s = by_xy.get((x, y - TILE_M))               # south neighbour (y = lower-left northing): A bottom row vs B top row
        if s: U(_pairs_1d(_dense(A['b']), _dense(strips[s]['t'])), s)
        se = by_xy.get((x + TILE_M, y - TILE_M))     # south-east corner
        if se:
            a, b = _dense(A['b'])[-1], _dense(strips[se]['t'])[0]
            if a and b: U([(a, b)], se)
        sw = by_xy.get((x - TILE_M, y - TILE_M))     # south-west corner
        if sw:
            a, b = _dense(A['b'])[0], _dense(strips[sw]['t'])[-1]
            if a and b: U([(a, b)], sw)
    root = np.array([uf.find(i) for i in range(n)], np.int64)
    rmax = np.zeros(n, np.float32)
    if n: np.maximum.at(rmax, root, cmax)
    return dict(base=base, nls=nls, shapes=shapes, root=root, keep=rmax >= hi, rmax=rmax, cnpx=cnpx, unions=unions)


def pass2_tile(args):
    """-> (tile, GeoDataFrame pieces with oid) for one tile."""
    from scipy import ndimage
    from rasterio.transform import from_origin
    from xtile_merge import parse_cfg, polygons
    out, tile, loader, oid_map, gsd = args   # oid_map: int64[nl+1], 0 = drop
    hi, lo = parse_cfg(CFG)
    p = loader(out, tile)
    lab, nl = ndimage.label(p >= lo, S8)
    if nl != len(oid_map) - 1: raise RuntimeError(f'{tile}: non-deterministic labels')
    ids = oid_map[lab]
    x, y = C.tile_xy(tile)
    tr = from_origin(x, y + TILE_M, gsd, gsd)
    return tile, polygons({tile: ids}, {tile: tr}, 'EPSG:32734') if ids.any() else None


def run(tiles, out, dest, workers=4, loader=None, gsd=0.05, log=print):
    import concurrent.futures as cf, multiprocessing as mp
    import geopandas as gpd, pandas as pd
    loader = loader or C.load_avg2
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()): raise FileExistsError(f'{dest} not empty; use a new folder')
    dest.mkdir(parents=True, exist_ok=True)
    missing = [t for t in tiles if not all(C.out_path(out, m, 'pv', t).exists() for m in C.MODELS)] if loader is C.load_avg2 else []
    if missing: raise RuntimeError(f'{len(missing)} tiles lack PV outputs (first {missing[0]})')
    t0 = time.time(); ctx = mp.get_context('spawn')
    def pmap(fn, items, chunk):
        if workers <= 0:  # in-process (tests)
            yield from map(fn, items); return
        with cf.ProcessPoolExecutor(workers, mp_context=ctx) as ex:
            yield from ex.map(fn, items, chunksize=chunk)
    res = list(pmap(pass1_tile, [(out, t, loader) for t in tiles], 16))
    log(f'pass1 {len(res)} tiles {time.time() - t0:.0f}s')
    P = plan_city(res); del res
    root, keep = P['root'], P['keep']
    # object -> tiles it spans (kept roots only)
    obj_tiles = {}
    for t in tiles:
        b, nl = P['base'][t], P['nls'][t]
        for r in set(root[b:b + nl][keep[root[b:b + nl]]].tolist()): obj_tiles.setdefault(r, []).append(t)
    npx = np.zeros(len(root), np.int64)
    if len(root): np.add.at(npx, root, P['cnpx'])
    jobs = []
    for t in tiles:
        b, nl = P['base'][t], P['nls'][t]
        if nl == 0: continue
        r = root[b:b + nl]
        if not keep[r].any(): continue
        jobs.append((out, t, loader, np.r_[0, np.where(keep[r], r + 1, 0)].astype(np.int64), gsd))
    singles, multi = [], []
    for t, g in pmap(pass2_tile, jobs, 8):
        if g is None: continue
        nt = g.oid.map(lambda o: len(obj_tiles[o - 1]))
        singles.append(g[nt == 1]); multi.append(g[nt > 1])
    log(f'pass2 {len(jobs)} tiles {time.time() - t0:.0f}s')
    crs = 'EPSG:32734'
    S = pd.concat(singles, ignore_index=True) if singles else gpd.GeoDataFrame(dict(oid=[]), geometry=[], crs=crs)
    Mg = pd.concat(multi, ignore_index=True) if multi else gpd.GeoDataFrame(dict(oid=[]), geometry=[], crs=crs)
    if len(Mg): Mg = gpd.GeoDataFrame(Mg, geometry='geometry', crs=crs).dissolve(by='oid', as_index=False)
    G = gpd.GeoDataFrame(pd.concat([S, Mg], ignore_index=True), geometry='geometry', crs=crs).sort_values('oid').reset_index(drop=True)
    if G.oid.duplicated().any(): raise RuntimeError('duplicate object ids after dissolve')
    r0 = G.oid.values.astype(np.int64) - 1
    G['n_px'] = npx[r0]; G['max_p'] = P['rmax'][r0]
    G['n_tiles'] = [len(obj_tiles[r]) for r in r0]
    G['n_blocks'] = [len({C.block_of(t) for t in obj_tiles[r]}) for r in r0]
    rp = G.geometry.representative_point()
    G['block'] = [f'E{int(x // BLOCK_M * BLOCK_M)}_N{int(y // BLOCK_M * BLOCK_M)}' for x, y in zip(rp.x, rp.y)]
    G['area_m2'] = G.geometry.area
    px_area = gsd * gsd
    if not np.allclose(G.area_m2.values, G.n_px.values * px_area, rtol=1e-9, atol=1e-6):
        bad = int((~np.isclose(G.area_m2.values, G.n_px.values * px_area, rtol=1e-9, atol=1e-6)).sum())
        raise RuntimeError(f'area not conserved for {bad} objects')
    C.publish(dest / 'objects_v3.gpkg', lambda tmp: G.to_file(tmp, driver='GPKG', layer='objects'), suffix='.gpkg')
    bc = G.groupby('block').agg(n_objects=('oid', 'size'), area_m2=('area_m2', 'sum')).reset_index()
    C.publish(dest / 'block_counts_v3.csv', lambda tmp: bc.to_csv(tmp, index=False), suffix='.csv')
    summ = dict(cfg=CFG, n_tiles=len(tiles), n_components=int(len(root)), n_objects=int(len(G)),
                area_m2=float(G.area_m2.sum()), n_multi_tile=int((G.n_tiles > 1).sum()),
                n_cross_block=int((G.n_blocks > 1).sum()), edge_unions=int(P['unions']), n_blocks=int(len(bc)),
                wall_s=round(time.time() - t0, 1), time=time.strftime('%F %T %Z'), script_md5=_md5(__file__),
                xtile_merge_md5=_md5(HERE.parent / 'ct_xtile_merge' / 'xtile_merge.py'))
    C.publish(dest / 'summary_v3.json', lambda tmp: Path(tmp).write_text(json.dumps(summ, indent=1)), suffix='.json')
    return G, summ


def _md5(p):
    import hashlib; return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles-csv', required=True); ap.add_argument('--out', required=True, help='inference output root')
    ap.add_argument('--dest', required=True, help='NEW folder for merge outputs'); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(argv)
    _, s = run(C.read_tile_list(a.tiles_csv), a.out, a.dest, workers=a.workers, log=lambda m: print(m, flush=True))
    print(json.dumps(s), flush=True)


if __name__ == '__main__':
    main()
