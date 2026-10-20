"""CT citywide avg2 + h0.9_0.5 + A3 merge, v3 (citywide cross-block join). Replaces the gated per-block `merge`
stage of ct_city_infer_v2.py (kept unchanged) after Codex r3 showed block+halo ownership can double-count
long objects.

Semantics = xtile_merge.merge_plan over ALL tiles at once (unmodified rule): components of avg2 >= lo
(8-connected) inside each tile, joined across every tile edge/corner where labelled pixels touch, kept if the
merged component's max >= hi (h0.9_0.5: hi 0.9, lo 0.5). avg2 = (pv_convnext + pv_b2)/2/255 float, as bench.py.
Memory-light: pass 1 streams tiles (<= --workers processes) and keeps only per-component (max, n_px) and the
labelled pixels of each tile's 1-px border strips; union-find over those. Pass 2 streams tiles again (bounded in-flight tasks),
re-checks each tile against pass 1, polygonises, appends single-tile objects to the output gpkg in batches and
stages multi-tile pieces on disk; those are dissolved at the end in oid chunks.
Each object is owned by exactly one 1 km block: the block containing its representative point.
summary_v3.json is written LAST and marks a complete run; a folder without it is incomplete (do not use).
Outputs (new dir, never overwritten): objects_v3.gpkg (layer objects), multi_pieces_v3.gpkg (per-tile pieces of
multi-tile objects, for audit), block_counts_v3.csv, summary_v3.json.
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
        return tile, 0, np.zeros(0, np.float32), np.zeros(0, np.int64), None, p.shape, None
    idx = np.arange(1, nl + 1)
    cmax = np.asarray(ndimage.maximum(p, lab, idx), np.float32)
    cnpx = np.bincount(lab.ravel(), minlength=nl + 1)[1:].astype(np.int64)
    import hashlib
    return tile, nl, cmax, cnpx, {k: _sparse(v) for k, v in _strips(lab).items()}, p.shape, hashlib.md5(lab.astype(np.int32).tobytes()).hexdigest()


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
    for tile, nl, cmax, cnpx, st, shape, _h in results:
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
    """-> (tile, GeoDataFrame pieces with oid or None). Re-labels the tile and checks it matches pass 1."""
    from scipy import ndimage
    from rasterio.transform import from_origin
    from xtile_merge import parse_cfg, polygons
    out, tile, loader, oid_map, cnpx, cmax, shape, lab_md5, gsd = args   # oid_map: int64[nl+1], 0 = drop
    hi, lo = parse_cfg(CFG)
    p = loader(out, tile)
    lab, nl = ndimage.label(p >= lo, S8)
    if p.shape != tuple(shape) or nl != len(oid_map) - 1:
        raise RuntimeError(f'{tile}: input changed between passes (shape/labels)')
    idx = np.arange(1, nl + 1)
    if not (np.array_equal(np.bincount(lab.ravel(), minlength=nl + 1)[1:], cnpx)
            and np.array_equal(np.asarray(ndimage.maximum(p, lab, idx), np.float32), cmax)):
        raise RuntimeError(f'{tile}: input changed between passes (component stats)')
    import hashlib
    if hashlib.md5(lab.astype(np.int32).tobytes()).hexdigest() != lab_md5:
        raise RuntimeError(f'{tile}: input changed between passes (label raster)')
    ids = oid_map[lab]
    x, y = C.tile_xy(tile)
    tr = from_origin(x, y + TILE_M, gsd, gsd)
    return tile, polygons({tile: ids}, {tile: tr}, 'EPSG:32734') if ids.any() else None


def _bounded_map(fn, items, workers, ctx, window):
    """Like map, in order, with at most `window` tasks in flight (bounds memory of returned results)."""
    import concurrent.futures as cf, collections
    if workers <= 0:
        yield from map(fn, items); return
    it = iter(items); q = collections.deque()
    with cf.ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for x in it:
            q.append(ex.submit(fn, x))
            if len(q) >= window: yield q.popleft().result()
        while q: yield q.popleft().result()


def run(tiles, out, dest, workers=4, loader=None, gsd=0.05, log=print, batch=2000, return_gdf=False):
    import multiprocessing as mp
    import geopandas as gpd, pandas as pd, pyogrio
    loader = loader or C.load_avg2
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()): raise FileExistsError(f'{dest} not empty; use a new folder')
    dest.mkdir(parents=True, exist_ok=True)
    missing = [t for t in tiles if not all(C.out_path(out, m, 'pv', t).exists() for m in C.MODELS)] if loader is C.load_avg2 else []
    if missing: raise RuntimeError(f'{len(missing)} tiles lack PV outputs (first {missing[0]})')
    t0 = time.time(); ctx = mp.get_context('spawn'); crs = 'EPSG:32734'; px_area = gsd * gsd
    res = list(_bounded_map(pass1_tile, [(out, t, loader) for t in tiles], workers, ctx, 64))
    cn = {r[0]: (r[3], r[2], r[5], r[6]) for r in res}   # tile -> (cnpx, cmax, shape, label md5)
    log(f'pass1 {len(res)} tiles {time.time() - t0:.0f}s')
    P = plan_city(res); del res
    root, keep = P['root'], P['keep']
    obj_tiles = {}
    for t in tiles:
        b, nl = P['base'][t], P['nls'][t]
        rr = root[b:b + nl]
        for r in set(rr[keep[rr]].tolist()): obj_tiles.setdefault(r, []).append(t)
    npx = np.zeros(len(root), np.int64)
    if len(root): np.add.at(npx, root, P['cnpx'])
    def jobs():
        for t in tiles:
            b, nl = P['base'][t], P['nls'][t]
            if nl == 0: continue
            r = root[b:b + nl]
            if not keep[r].any(): continue
            c_npx, c_max, shape, h = cn[t]
            yield (out, t, loader, np.r_[0, np.where(keep[r], r + 1, 0)].astype(np.int64), c_npx, c_max, shape, h, gsd)
    part = dest / '.objects_v3.partial.gpkg'; stage = dest / '.multi_pieces_v3.partial.gpkg'
    stats = dict(n=0, area=0.0, multi=0, cross=0); blocks = {}
    def finish(G):
        """attach attributes, check area, append to the output gpkg."""
        if not len(G): return
        r0 = G.oid.values.astype(np.int64) - 1
        G['n_px'] = npx[r0]; G['max_p'] = P['rmax'][r0]
        G['n_tiles'] = [len(obj_tiles[r]) for r in r0]
        G['n_blocks'] = [len({C.block_of(t) for t in obj_tiles[r]}) for r in r0]
        rp = G.geometry.representative_point()
        G['block'] = [f'E{int(x // BLOCK_M * BLOCK_M)}_N{int(y // BLOCK_M * BLOCK_M)}' for x, y in zip(rp.x, rp.y)]
        G['area_m2'] = G.geometry.area
        ok = np.isclose(G.area_m2.values, G.n_px.values * px_area, rtol=1e-9, atol=1e-6)
        if not ok.all(): raise RuntimeError(f'area not conserved for {int((~ok).sum())} objects')
        pyogrio.write_dataframe(G, part, layer='objects', append=part.exists(), promote_to_multi=True)
        stats['n'] += len(G); stats['area'] += float(G.area_m2.sum())
        stats['multi'] += int((G.n_tiles > 1).sum()); stats['cross'] += int((G.n_blocks > 1).sum())
        for k, a in zip(G.block, G.area_m2):
            v = blocks.setdefault(k, [0, 0.0]); v[0] += 1; v[1] += a
    single_buf, multi_buf, n_jobs = [], [], 0
    def flush(force=False):
        nonlocal single_buf, multi_buf
        if single_buf and (force or sum(map(len, single_buf)) >= batch):
            finish(gpd.GeoDataFrame(pd.concat(single_buf, ignore_index=True), geometry='geometry', crs=crs)); single_buf = []
        if multi_buf and (force or sum(map(len, multi_buf)) >= batch):
            M = gpd.GeoDataFrame(pd.concat(multi_buf, ignore_index=True), geometry='geometry', crs=crs)
            pyogrio.write_dataframe(M, stage, layer='pieces', append=stage.exists(), promote_to_multi=True); multi_buf = []
    for t, g in _bounded_map(pass2_tile, jobs(), workers, ctx, 32):
        n_jobs += 1
        if g is None: continue
        nt = g.oid.map(lambda o: len(obj_tiles[o - 1]))
        single_buf.append(g[nt == 1]); multi_buf.append(g[nt > 1]); flush()
    flush(force=True)
    log(f'pass2 {n_jobs} tiles {time.time() - t0:.0f}s')
    if stage.exists():   # dissolve multi-tile objects in oid chunks (pieces on disk, bounded memory)
        oids = np.unique(pyogrio.read_dataframe(stage, layer='pieces', columns=['oid'], read_geometry=False).oid.values)
        for i in range(0, len(oids), batch):
            lo_, hi_ = int(oids[i]), int(oids[min(i + batch, len(oids)) - 1])
            Mp = pyogrio.read_dataframe(stage, layer='pieces', where=f'oid >= {lo_} AND oid <= {hi_}')
            finish(gpd.GeoDataFrame(Mp, geometry='geometry', crs=crs).dissolve(by='oid', as_index=False))
    n_kept = int(len(obj_tiles))
    if stats['n'] != n_kept: raise RuntimeError(f'wrote {stats["n"]} objects, expected {n_kept} (lost or duplicated)')
    if part.exists(): os.link(part, dest / 'objects_v3.gpkg'); os.unlink(part)
    if stage.exists(): os.link(stage, dest / 'multi_pieces_v3.gpkg'); os.unlink(stage)
    bc = pd.DataFrame([(k, v[0], v[1]) for k, v in sorted(blocks.items())], columns=['block', 'n_objects', 'area_m2'])
    C.publish(dest / 'block_counts_v3.csv', lambda tmp: bc.to_csv(tmp, index=False), suffix='.csv')
    summ = dict(cfg=CFG, n_tiles=len(tiles), n_components=int(len(root)), n_objects=stats['n'],
                area_m2=stats['area'], n_multi_tile=stats['multi'], n_cross_block=stats['cross'],
                edge_unions=int(P['unions']), n_blocks=int(len(bc)),
                wall_s=round(time.time() - t0, 1), time=time.strftime('%F %T %Z'), script_md5=_md5(__file__),
                xtile_merge_md5=_md5(HERE.parent / 'ct_xtile_merge' / 'xtile_merge.py'))
    C.publish(dest / 'summary_v3.json', lambda tmp: Path(tmp).write_text(json.dumps(summ, indent=1)), suffix='.json')
    if not return_gdf:          # production: never read the city back into memory
        return None, summ
    G = gpd.read_file(dest / 'objects_v3.gpkg', layer='objects') if (dest / 'objects_v3.gpkg').exists() else \
        gpd.GeoDataFrame(dict(oid=[], n_px=[], max_p=[], n_tiles=[], n_blocks=[], block=[], area_m2=[]), geometry=[], crs=crs)
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
