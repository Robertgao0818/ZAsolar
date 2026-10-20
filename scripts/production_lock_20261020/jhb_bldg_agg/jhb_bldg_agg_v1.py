"""JHB building aggregation (installation unit = roof). Decision 葉 2026-10-20 19:23 NZDT "OK" (DECISION_LOG, checklist v16):
rows aggregated by roof via building-mask association; one building = one installation; a row spanning 2+ buildings is split
per building; NO 1 m merge rule. Association rule reused unchanged from G14 (g14_bldg_assoc/g14_lib_v1.associate):
0.5 m building buffer; object off-roof fraction < 0.3 -> whole object area allocated to touched buildings in proportion to
overlap; else only the on-roof part is allocated and the off-roof remainder is kept as an unmatched residual (never dropped).
Buildings: union of footprint layers, make_valid, dissolved, exploded (as G14 assoc_grid_v2 JHB branch).

Streaming: the objects' extent is cut into square blocks (default 1 km). Each object is owned by the block containing its
representative point (half-open), so it is processed exactly once. Per block, buildings are bbox-read around the owned
objects and the dissolve window is grown until no dissolved building touches the window edge, so a building gets the same
geometry (and id) from every block; a block whose window does not converge fails (no part, no summary). Building id = sha1 of the
normalized 1 cm-snapped footprint (content id, same from every block). Inputs are md5-fingerprinted in config.json; a resume with
changed inputs is refused. Final outputs never overwrite existing files. One part file per block
(atomic rename) -> resumable; the final pass groups pieces by building id across blocks. summary.json written last.

usage: jhb_bldg_agg_v1.py --objects objects_v3.gpkg --buildings a.gpkg b.gpkg ... --dest NEW_DIR [--workers auto]
outputs (dest): installations.gpkg (one row per building with PV: bld_id, pv_area_m2 (allocated), clip_area_m2, n_pieces,
n_objects, installed = pv_area_m2 >= 2), pieces.gpkg (object x building clipped pieces: oid, bld_id, clip_area_m2,
alloc_area_m2, split), unmatched.gpkg (oid, kind = no_building | offroof_residual, area_m2, rule_class, geometry kept),
installations.csv, summary.json."""
import argparse, hashlib, json, os, sys, time
from multiprocessing import get_context
import numpy as np, pandas as pd, geopandas as gpd, shapely, pyogrio
from shapely import STRtree
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'citywide_infer'))
import cpu_workers as W  # noqa: E402
BUF, OFF, THR = 0.5, 0.3, 2.0   # = g14_lib_v1 constants
PER_WORKER_GB = 1.0
FINAL = ['pieces.gpkg', 'installations.gpkg', 'unmatched.gpkg', 'installations.csv', 'summary.json']
G = {}


def rule(orphan_area, dist):  # g14_lib_v1.rule with no parcel zoning (JHB has none)
    if orphan_area >= 200: return 'ground_mount_candidate'
    if dist <= 3 and orphan_area < 50: return 'misaligned_candidate'
    if dist > 3 and orphan_area >= 5: return 'missing_building_candidate'
    return 'small_or_ambiguous'


def read_bld(paths, box, crs):
    src = []
    for f in paths:
        g = gpd.read_file(f, bbox=gpd.GeoSeries([box], crs=crs))
        if len(g): src.append(g.to_crs(crs)[['geometry']])
    if not src: return gpd.GeoDataFrame(geometry=[], crs=crs)
    u = shapely.make_valid(np.concatenate([s.geometry.values for s in src]))
    u = u[~shapely.is_empty(u)]
    d = gpd.GeoDataFrame(geometry=[shapely.union_all(u)], crs=crs).explode(index_parts=False).reset_index(drop=True)
    return d[d.geom_type.isin(['Polygon', 'MultiPolygon']) & (d.area > 0)].reset_index(drop=True)


def buildings_for(paths, need, crs, pad=100.0, max_iter=50):
    """Dissolved buildings covering box `need`; window grown until no building touching `need` reaches the window edge."""
    win = shapely.box(*need.buffer(pad, join_style='mitre').bounds)
    for _ in range(max_iter):
        b = read_bld(paths, win, crs)
        b = b[b.intersects(need.buffer(pad, join_style='mitre'))].reset_index(drop=True)
        if not len(b): return b, True
        edge = b.intersects(win.exterior.buffer(1.0))
        if not edge.any(): return b, True
        win = shapely.box(*shapely.box(*b.total_bounds).union(win).buffer(pad, join_style='mitre').bounds)
    return b, False


def bid_of(geoms):
    """Content id: sha1 of the normalized 1 cm-snapped dissolved footprint -> identical from every block, distinct roofs differ."""
    g = shapely.normalize(shapely.set_precision(geoms, 0.01))
    return ['B' + hashlib.sha1(w).hexdigest()[:16] for w in shapely.to_wkb(g)]


def file_fp(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): h.update(c)
    return dict(path=os.path.abspath(p), size=os.path.getsize(p), md5=h.hexdigest())


def do_block(task):
    bx, by = task; a = G['args']; S = a.block; out = os.path.join(a.dest, 'parts', f'b_{bx}_{by}.pkl')
    if os.path.exists(out): return out, 'skip'
    x0, y0 = G['x0'] + bx * S, G['y0'] + by * S; core = shapely.box(x0, y0, x0 + S, y0 + S); crs = G['crs']
    ob = gpd.read_file(a.objects, bbox=gpd.GeoSeries([core], crs=crs))
    if len(ob):
        ob['geometry'] = shapely.make_valid(ob.geometry.values)
        rp = shapely.point_on_surface(ob.geometry.values); px, py = shapely.get_x(rp), shapely.get_y(rp)
        ob = ob[(px >= x0) & (px < x0 + S) & (py >= y0) & (py < y0 + S)].reset_index(drop=True)
    res = dict(block=(bx, by), n_obj=len(ob), buildings=None, pieces=None, unmatched=None, bld_complete=True, n_bld_window=0)
    if len(ob):
        ob['area'] = ob.area; ob = ob[ob.area > 0].reset_index(drop=True)
        need = shapely.box(*ob.total_bounds)
        bld, ok = buildings_for(a.buildings, need, crs); res['bld_complete'] = ok; res['n_bld_window'] = len(bld)
        if not ok: raise RuntimeError(f'block {task}: building window did not converge; no part written')
        bg = shapely.buffer(bld.geometry.values, BUF) if len(bld) else np.array([])
        if len(bld):
            li, ri = STRtree(bg).query(ob.geometry.values, predicate='intersects')
            pg = shapely.intersection(ob.geometry.values[li], bg[ri]) if len(li) else np.array([])
            P = pd.DataFrame({'o': li, 'b': ri, 'a': shapely.area(pg) if len(li) else []}); P['geometry'] = list(pg) if len(li) else []
            P = P[P.a > 0].reset_index(drop=True)
        else:
            P = pd.DataFrame({'o': [], 'b': [], 'a': [], 'geometry': []})
        area = ob.area.values
        on = P.groupby('o').a.sum().reindex(range(len(ob)), fill_value=0).clip(upper=area).values
        offf = 1 - on / area
        if len(P):
            sumA = P.groupby('o').a.transform('sum').values; o = P.o.values.astype(int)
            P['alloc'] = np.where(offf[o] < OFF, P.a / sumA * area[o], P.a / sumA * on[o])
            P['bld_id'] = bid_of(bld.geometry.values[P.b.values.astype(int)]); P['oid'] = ob.oid.values[o]
            P['n_bld_of_obj'] = P.groupby('o').b.transform('nunique').values
            P['split'] = P.n_bld_of_obj >= 2
            ub = np.unique(P.b.values.astype(int)); bids = bid_of(bld.geometry.values[ub])
            res['buildings'] = gpd.GeoDataFrame({'bld_id': bids, 'bld_area_m2': shapely.area(bld.geometry.values[ub])}, geometry=bld.geometry.values[ub], crs=crs)
            res['pieces'] = gpd.GeoDataFrame(P[['oid', 'bld_id', 'a', 'alloc', 'n_bld_of_obj', 'split']].rename(columns={'a': 'clip_area_m2', 'alloc': 'alloc_area_m2'}),
                                             geometry=P.geometry.values, crs=crs)
        orphan = np.where(offf < OFF, 0.0, area - on)
        k = np.nonzero(orphan > 1e-9)[0]
        if len(k):
            d = np.full(len(k), np.inf)
            if len(bld):
                qi, dist = STRtree(bld.geometry.values).query_nearest(ob.geometry.values[k], max_distance=30, return_distance=True)
                dd = pd.Series(dist).groupby(qi[0]).min(); d[dd.index.values] = dd.values
            geo = ob.geometry.values[k]
            if len(bld):  # residual = object minus buffered buildings it touches
                resid = []
                for j, kk in enumerate(k):
                    bs = P.b.values[P.o.values == kk].astype(int)
                    resid.append(shapely.difference(geo[j], shapely.union_all(bg[bs])) if len(bs) else geo[j])
                geo = np.array(resid, dtype=object)
            res['unmatched'] = gpd.GeoDataFrame({'oid': ob.oid.values[k], 'kind': np.where(on[k] > 0, 'offroof_residual', 'no_building'),
                                                 'area_m2': orphan[k], 'obj_area_m2': area[k], 'dist_bld_m': d,
                                                 'rule_class': [rule(x, y) for x, y in zip(orphan[k], d)]}, geometry=geo, crs=crs)
        res['obj_area'] = float(area.sum()); res['alloc_area'] = float(P.alloc.sum()) if len(P) else 0.0; res['orphan_area'] = float(orphan.sum())
    pd.to_pickle(res, out + '.tmp'); os.replace(out + '.tmp', out)
    return out, 'done'


def init(args, x0, y0, crs):
    G.update(args=args, x0=x0, y0=y0, crs=crs)
    os.environ.setdefault('OMP_NUM_THREADS', '1')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--objects', required=True); ap.add_argument('--buildings', nargs='+', required=True)
    ap.add_argument('--dest', required=True, help='NEW folder (resume = same dest)'); ap.add_argument('--block', type=float, default=1000.0)
    ap.add_argument('--workers', default='auto'); ap.add_argument('--mem-cap-gb', type=float, default=6.0)
    a = ap.parse_args(argv)
    if os.path.exists(os.path.join(a.dest, 'summary.json')): sys.exit(f'{a.dest}/summary.json exists: finished run, refusing to overwrite')
    os.makedirs(os.path.join(a.dest, 'parts'), exist_ok=True)
    info = pyogrio.read_info(a.objects); crs = info['crs']; x0, y0, x1, y1 = info['total_bounds']
    x0, y0 = np.floor(x0 / a.block) * a.block, np.floor(y0 / a.block) * a.block
    tasks = [(i, j) for i in range(int(np.ceil((x1 - x0) / a.block)) + 1) for j in range(int(np.ceil((y1 - y0) / a.block)) + 1)]
    for n in FINAL:
        if os.path.exists(os.path.join(a.dest, n)): sys.exit(f'{a.dest}/{n} exists: refusing to overwrite; use a new --dest')
    cfg = dict(objects=file_fp(a.objects), buildings=[file_fp(b) for b in a.buildings], block=a.block, x0=float(x0), y0=float(y0), crs=str(crs))
    cp = os.path.join(a.dest, 'config.json')
    if os.path.exists(cp):
        if json.load(open(cp)) != cfg: sys.exit('config differs from existing parts: use a new --dest')
    else: json.dump(cfg, open(cp, 'w'), indent=1)
    nw = W.resolve(a.workers, PER_WORKER_GB, a.mem_cap_gb); t0 = time.time()
    print(f'blocks {len(tasks)} workers {nw}', flush=True)
    if nw <= 1:
        init(a, x0, y0, crs); [do_block(t) for t in tasks]
    else:
        with get_context('spawn').Pool(nw, initializer=init, initargs=(a, x0, y0, crs), maxtasksperchild=20) as pool:
            for i, (p, s) in enumerate(pool.imap_unordered(do_block, tasks)):
                if i % 50 == 0: print(f'{i+1}/{len(tasks)} {s} {time.time()-t0:.0f}s', flush=True)
    finalize(a, tasks, crs, nw, time.time() - t0)


def _append(path, df, layer):
    pyogrio.write_dataframe(df, path, layer=layer, driver='GPKG', append=os.path.exists(path))


def finalize(a, tasks, crs, nw, secs):
    """Streams parts one by one: pieces/unmatched appended to *.tmp.gpkg; per-building stats kept as a small table;
    building geometry written once per id (consistency-checked by area). Final names appear only via os.replace at the end."""
    tmp = {n: os.path.join(a.dest, n + '.tmp.gpkg') for n in ['pieces', 'unmatched', 'bld']}
    for p in tmp.values():
        if os.path.exists(p): os.remove(p)  # our own unfinished temp files from an interrupted finalize
    n_obj, oa, aa, ra, seen_oid = 0, 0.0, 0.0, 0.0, set()
    st, barea, n_split_obj, n_split_pc, n_pc, objs_on = {}, {}, 0, 0, 0, set()
    um_n = {'no_building': 0, 'offroof_residual': 0}; um_a = {'no_building': 0.0, 'offroof_residual': 0.0}; rc = {}
    for t in tasks:
        r = pd.read_pickle(os.path.join(a.dest, 'parts', f'b_{t[0]}_{t[1]}.pkl')); n_obj += r['n_obj']
        if not r['bld_complete']: raise RuntimeError(f'part {t} has an incomplete building window')
        if r['n_obj']: oa += r['obj_area']; aa += r['alloc_area']; ra += r['orphan_area']
        p = r['pieces']
        if p is not None and len(p):
            _append(tmp['pieces'], p, 'pieces'); n_pc += len(p); n_split_pc += int(p.split.sum())
            objs_on.update(p.oid.tolist()); n_split_obj += int(p[p.split].oid.nunique())
            for b, g in p.groupby('bld_id'):
                s_ = st.setdefault(b, [0.0, 0.0, 0, set(), 0]); s_[0] += g.alloc_area_m2.sum(); s_[1] += g.clip_area_m2.sum()
                s_[2] += len(g); s_[3].update(g.oid.tolist()); s_[4] += int(g.split.sum())
            B = r['buildings']; new = ~B.bld_id.isin(barea.keys())
            for b, ar in zip(B.bld_id, B.bld_area_m2):
                if b in barea and abs(barea[b] - ar) > 0.01: raise RuntimeError(f'building {b} differs between blocks')
                barea[b] = ar
            if new.any(): _append(tmp['bld'], B[new], 'bld')
        u = r['unmatched']
        if u is not None and len(u):
            _append(tmp['unmatched'], u, 'unmatched')
            for k in um_n: um_n[k] += int((u.kind == k).sum()); um_a[k] += float(u.area_m2[u.kind == k].sum())
            for k, g in u.groupby('rule_class'): x = rc.setdefault(k, [0, 0.0]); x[0] += len(g); x[1] += float(g.area_m2.sum())
    src = pyogrio.read_dataframe(a.objects, columns=['oid'], read_geometry=False)
    assert n_obj == len(src) and src.oid.is_unique, f'object count {n_obj} != input {len(src)} (or oid not unique)'
    inst = pd.DataFrame([(b, v[0], v[1], v[2], len(v[3]), v[4]) for b, v in st.items()],
                        columns=['bld_id', 'pv_area_m2', 'clip_area_m2', 'n_pieces', 'n_objects', 'n_split_pieces'])
    inst['installed'] = inst.pv_area_m2 >= THR
    # installations layer = building footprint + stats, written in chunks from the bld temp layer
    ip = os.path.join(a.dest, 'installations.gpkg.tmp.gpkg')
    if os.path.exists(ip): os.remove(ip)
    if os.path.exists(tmp['bld']):
        ib = inst.set_index('bld_id'); nb = pyogrio.read_info(tmp['bld'])['features']
        for k in range(0, nb, 50000):
            g = pyogrio.read_dataframe(tmp['bld'], skip_features=k, max_features=50000)
            g = g.join(ib, on='bld_id'); _append(ip, g, 'installations')
        assert nb == len(inst), 'building table / geometry mismatch'
    S = dict(objects_in=int(len(src)), objects_area_m2=round(oa, 3), objects_on_any_building=len(objs_on),
             objects_split_2plus_buildings=n_split_obj, pieces=n_pc, split_pieces=n_split_pc,
             buildings_with_pv=int(len(inst)), installations_ge2m2=int(inst.installed.sum()),
             installations_ge2m2_area_m2=round(float(inst.pv_area_m2[inst.installed].sum()), 3), alloc_area_m2=round(aa, 3),
             unmatched_no_building_n=um_n['no_building'], unmatched_no_building_area_m2=round(um_a['no_building'], 3),
             offroof_residual_n=um_n['offroof_residual'], offroof_residual_area_m2=round(um_a['offroof_residual'], 3),
             unmatched_by_rule_class={k: dict(n=v[0], area_m2=round(v[1], 3)) for k, v in sorted(rc.items())},
             area_check_m2=round(oa - aa - ra, 6), blocks=len(tasks), workers=nw, seconds=round(secs, 1),
             rule=dict(buffer_m=BUF, offroof_frac=OFF, installed_m2=THR, merge_gap_m=0),
             config=json.load(open(os.path.join(a.dest, 'config.json'))))
    assert abs(S['area_check_m2']) < 1e-3 * max(oa, 1), 'area not conserved'
    for n in FINAL:
        if os.path.exists(os.path.join(a.dest, n)): sys.exit(f'{n} appeared during run: refusing to overwrite')
    inst.to_csv(os.path.join(a.dest, 'installations.csv.tmp'), index=False)
    for src_, n in [(tmp['pieces'], 'pieces.gpkg'), (ip, 'installations.gpkg'), (tmp['unmatched'], 'unmatched.gpkg'),
                    (os.path.join(a.dest, 'installations.csv.tmp'), 'installations.csv')]:
        if os.path.exists(src_): os.replace(src_, os.path.join(a.dest, n))
    if os.path.exists(tmp['bld']): os.remove(tmp['bld'])
    json.dump(S, open(os.path.join(a.dest, 'summary.json.tmp'), 'w'), indent=1); os.replace(os.path.join(a.dest, 'summary.json.tmp'), os.path.join(a.dest, 'summary.json'))
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
