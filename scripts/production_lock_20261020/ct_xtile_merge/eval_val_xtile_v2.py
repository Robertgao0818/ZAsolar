#!/usr/bin/env python3
"""G7 VAL-ONLY evaluation v2 (Codex r1 fixes). Cached ConvNeXt-T s0 val probs, CPU only, no inference.
Arms: A0 per-tile binarize, no merge | A1 per-tile binarize + in-grid cross-tile merge (current scorer)
      A2 in-grid cross-tile hysteresis + merge | A3 hysteresis + merge across ALL val tiles (crosses 1 km grid lines).
Object/pixel metrics use the pinned zaeval ObjectAccumulator per grid (GT is per grid, so object matching stays in-grid;
A3 differs from A2 only by pixels recovered/kept through cross-grid hysteresis). Cross-grid objects + ownership are
reported separately in cross_grid.json.
usage: eval_val_xtile_v2.py OUTDIR [NPROC]   (resumable; reuse only when the run fingerprint matches, else refuses)"""
import os, sys, json, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import multiprocessing as mp
HERE = Path(__file__).resolve().parent
PIPE = Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser()
sys.path.insert(0, str(PIPE)); sys.path.insert(0, str(HERE))
from pipeline_common_v3 import *  # noqa
from zaeval.postproc import binarize
from xtile_merge import TileIn, per_tile_masks, merge_plan, plan_tile_ids, plan_objects, global_origin, polygons, assign_owner
PROBS = Path('~/capetown_eval/ds3m_ft/phase1_followup_valprobs_20261018/convnext_t_s0').expanduser()
CELLS = Path('~/projects/ZAsolar/data/task_grid_ct_full.gpkg').expanduser()
CFGS = ('h0.9_0.5', 't0.5'); IOUS = (0.3, 0.5); ARMS = ('A0', 'A1', 'A2', 'A3')
G = {}  # filled before fork: refs_by_grid, plans3, fp_base

class NoMergeAccumulator(ObjectAccumulator):
    """Each per-tile component is its own object (production without cross-tile merge)."""
    def _merge_parts(self):
        saved, self._edges = self._edges, {}
        try:
            return super()._merge_parts()
        finally:
            self._edges = saved

def sha(b): return hashlib.sha256(b).hexdigest()
def loader(grid, tid): return lambda: np.load(PROBS / grid / f'{tid}.npz')['p'].astype(np.float32) / 255.0
def tiles_of(grid): return [TileIn(r.tile_id, *global_origin(r.transform, r.gsd_m), loader(grid, r.tile_id)) for r in G['refs'][grid]]

def fstat(f): f = Path(f); st = f.stat(); return (str(f), st.st_size, st.st_mtime_ns)

def grid_fp(grid):
    """Per-grid fingerprint: global base (code, scorer, ALL val prob files since A3 depends on them) + this grid's
    label masks, images and GT polygon source."""
    refs = G['refs'][grid]
    own = [fstat(r.mask_path) for r in refs] + [fstat(r.raster_path) for r in refs]
    label_md5 = str(manifests()[1].loc[grid].md5)  # GT polygons are md5-pinned by load_grid_gt
    return sha((G['fp_base'] + json.dumps(sorted(own)) + label_md5).encode())

def score_grid(grid):
    f = Path(G['out']) / 'grids' / f'{grid}.json'; fp = grid_fp(grid)
    if f.exists():
        old = json.loads(f.read_text())
        if old.get('fingerprint') != fp:  # covers A3's dependence on all val probs via fp_base
            raise RuntimeError(f'{f} exists with a different fingerprint; use a new OUTDIR (no overwrite)')
        load_grid_gt(grid, manifests()[1])  # re-verify actual label file md5 on reuse
        return old
    gm = manifests()[1]; gt = load_grid_gt(grid, gm)['PV']
    refs = G['refs'][grid]; tiles = tiles_of(grid); res = {}
    for cfg in CFGS:
        pt = per_tile_masks(tiles, cfg)
        r0 = refs[0]; assert (pt[r0.tile_id] == binarize(loader(grid, r0.tile_id)(), cfg)).all()  # parity with pinned binarize
        p2 = merge_plan(tiles, cfg); p3 = G['plans3'][cfg]
        get = {'A0': lambda t: pt[t.key], 'A1': lambda t: pt[t.key],
               'A2': lambda t: plan_tile_ids(p2, t) > 0, 'A3': lambda t: plan_tile_ids(p3, t) > 0}
        for arm in ARMS:
            acc = (NoMergeAccumulator if arm == 'A0' else ObjectAccumulator)(iou_thrs=IOUS)
            px = dict(tp=0, fp=0, fn=0); area = 0.0
            for ref, t in zip(refs, tiles):
                view = make_view(class_labels(ref, gt), 'exclude-pixels'); m = get[arm](t)
                acc.add_tile(ref, PredTile(ref, mask=m), view)
                g = view.gt_id > 0; v = view.valid_px
                px['tp'] += int((m & g & v).sum()); px['fp'] += int((m & ~g & v).sum()); px['fn'] += int((~m & g & v).sum())
                area += float(m.sum()) * ref.gsd_m ** 2
            fin = acc.finalize({grid: gt})
            res[f'{cfg}/{arm}'] = dict(px=px, pred_area_m2=area,
                by_iou={k: {kk: fin['by_iou'][k][kk] for kk in ('tp', 'fp', 'fn')} for k in fin['by_iou']},
                n_pred=fin['n_pred'], n_tile_parts=fin['n_tile_parts'], n_components_merged=fin['n_components_merged'],
                n_pred_dropped_area=fin['n_pred_dropped_area'], n_gt=fin['n_gt'])
        objs2 = plan_objects(p2)
        res[f'{cfg}/A2']['n_multi_tile_objects'] = sum(o['n_tiles'] > 1 for o in objs2.values())
    out = dict(grid=grid, tiles=len(refs), fingerprint=fp, res=res,
               gt_area_m2=float(gt.area.sum()) if len(gt) else 0.0)
    f.parent.mkdir(parents=True, exist_ok=True); tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(out)); os.replace(tmp, f)
    print(json.dumps(dict(grid_done=grid, tiles=len(refs))), flush=True)
    return out

def cross_grid(grid_of):
    """Objects of the all-val plan (A3) whose tiles span >1 grid, their polygons and single owner cell."""
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union
    allt = G['uniq']
    tr = {r.tile_id: r.transform for g in G['refs'] for r in G['refs'][g]}
    crs = next(iter(G['refs'].values()))[0].crs
    F = gpd.read_file(CELLS).to_crs(crs)  # real 1 km census cells (same file as cross_grid_edge_v1)
    cells = F[F.gridcell_id.isin(list(G['refs']))][['gridcell_id', 'geometry']].reset_index(drop=True)
    assert set(cells.gridcell_id) == set(G['refs']) and cells.gridcell_id.is_unique
    out = {}
    for cfg, plan in G['plans3'].items():
        objs = plan_objects(plan)
        gs = lambda o: {g for k in o['tile_keys'] for g in grid_of[k]}
        multi = {i: o for i, o in objs.items() if len(gs(o)) > 1}
        keys = sorted({k for o in multi.values() for k in o['tile_keys']})
        rec = dict(n_objects=len(objs), n_multi_grid=len(multi),
                   multi_grid_area_m2=float(sum(o['n_px'] for o in multi.values()) * G['gsd'] ** 2),
                   double_count_without_owner=int(sum(len(gs(o)) - 1 for o in multi.values())))
        if multi:
            ids = {k: np.where(np.isin(a := plan_tile_ids(plan, allt[k]), list(multi)), a, 0) for k in keys}
            poly = polygons(ids, {k: tr[k] for k in keys}, crs)
            own = assign_owner(poly, cells)
            assert len(own) == len(multi) and own.owner_cell.notna().all() and own.oid.is_unique
            rec.update(polygons_valid=bool(poly.is_valid.all()), n_with_one_owner=int(own.owner_cell.notna().sum()),
                       objects=[dict(oid=int(r.oid), owner=r.owner_cell, area_m2=round(float(r.geometry.area), 2),
                                     grids=sorted(gs(multi[int(r.oid)]))) for r in own.itertuples()])
            dst = Path(G['out']) / f'cross_grid_objects_{cfg}.gpkg'; assert not dst.exists(), dst
            own.to_file(dst, driver='GPKG')
        out[cfg] = rec
    return out

if __name__ == '__main__':
    out_dir = Path(sys.argv[1]); npr = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    prov = scorer_provenance()
    rows, gm = select_rows('val'); rows, gm, _ = validation_scopes(rows)
    assert len(rows) == 1703 and set(rows.split) == {'val'}
    import pandas as pd
    refs = {g: [ref_for(pd.Series(r)) for r in grp.to_dict('records')] for g, grp in rows.groupby('grid')}
    assert all(r.split == 'val' for v in refs.values() for r in v)
    for name in ('cross_grid.json', 'summary.json', 'cross_grid_objects_h0.9_0.5.gpkg', 'cross_grid_objects_t0.5.gpkg'):
        assert not (out_dir / name).exists(), f'{out_dir / name} exists; use a new OUTDIR (no overwrite)'
    # Adjacent val grids share some 100 m tiles (same image). One tile id -> set of grids.
    grid_of = {}
    for g, v in refs.items():
        for r in v: grid_of.setdefault(r.tile_id, set()).add(g)
    shared = sorted(k for k, v in grid_of.items() if len(v) > 1)
    gsds = {r.gsd_m for v in refs.values() for r in v}; assert len(gsds) == 1
    PIPE_META_COLLECTORS = METADATA / 'collectors.gpkg'  # SWH polygons used by the pinned GT loader
    code = {p.name: sha(p.read_bytes()) for p in [HERE / 'xtile_merge.py', HERE / 'eval_val_xtile_v2.py']
            + sorted(PIPE.glob('pipeline_common*.py'))}
    all_probs = sorted(fstat(PROBS / g / f'{r.tile_id}.npz') for g, v in refs.items() for r in v)
    G.update(out=str(out_dir), refs=refs, gsd=gsds.pop(),
             fp_base=json.dumps(dict(code=code, cfgs=CFGS, ious=IOUS, arms=ARMS, scorer=prov, probs=str(PROBS),
                                     all_probs=all_probs, cells=fstat(CELLS),
                                     collectors=sha(Path(PIPE_META_COLLECTORS).read_bytes())), sort_keys=True))
    # A3 runs on unique tiles; a shared tile must have identical probs in every grid that holds it.
    uniq = {}
    for g in sorted(refs):
        for t in tiles_of(g):
            if t.key in uniq:
                assert np.array_equal(uniq[t.key].load(), t.load()), f'shared tile {t.key} probs differ'
            else:
                uniq[t.key] = t
    G['uniq'] = uniq
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(stage='plan3_start', tiles=len(grid_of))), flush=True)
    G['plans3'] = {cfg: merge_plan(list(uniq.values()), cfg) for cfg in CFGS}
    print(json.dumps(dict(stage='plan3_done')), flush=True)
    grids = sorted(refs)
    if os.environ.get('GRIDS'): grids = [g for g in grids if g in os.environ['GRIDS'].split(',')]
    with mp.get_context('fork').Pool(npr) as p: outs = p.map(score_grid, grids)
    xg = cross_grid(grid_of)
    agg = {}
    SUMK = ('pred_area_m2', 'n_pred', 'n_tile_parts', 'n_components_merged', 'n_pred_dropped_area', 'n_gt', 'n_multi_tile_objects')
    for o in outs:
        for k, r in o['res'].items():
            a = agg.setdefault(k, dict(px=dict(tp=0, fp=0, fn=0), by_iou={}, **{s: 0 for s in SUMK}))
            for kk in a['px']: a['px'][kk] += r['px'][kk]
            for s in SUMK: a[s] += r.get(s, 0)
            for t, c in r['by_iou'].items():
                b = a['by_iou'].setdefault(t, dict(tp=0, fp=0, fn=0))
                for kk in b: b[kk] += c[kk]
    for k, a in agg.items():
        p = a['px']; a['px_iou'] = round(p['tp'] / max(p['tp'] + p['fp'] + p['fn'], 1), 4)
        for t, b in a['by_iou'].items():
            P = b['tp'] / max(b['tp'] + b['fp'], 1); R = b['tp'] / max(b['tp'] + b['fn'], 1)
            b.update(P=round(P, 4), R=round(R, 4), F1=round(2 * P * R / max(P + R, 1e-9), 4))
    for cfg in CFGS:
        a0 = agg[f'{cfg}/A0']['pred_area_m2']
        for arm in ARMS: agg[f'{cfg}/{arm}']['area_change_vs_A0_pct'] = round(100 * (agg[f'{cfg}/{arm}']['pred_area_m2'] - a0) / max(a0, 1e-9), 3)
    summ = dict(n_unique_tiles=len(grid_of), n_shared_tiles=len(shared), shared_tiles=shared, grids=len(outs), tiles=sum(o['tiles'] for o in outs), gt_area_m2=sum(o['gt_area_m2'] for o in outs),
                scorer=prov, code_sha256=code, agg=agg, cross_grid={c: {k: v for k, v in r.items() if k != 'objects'} for c, r in xg.items()},
                definitions=dict(n_components_merged='zaeval: merged objects before area/region filtering',
                                 n_multi_tile_objects='A2 kept objects whose pixels span >1 tile (same grid)',
                                 n_multi_grid='A3 kept objects whose tiles belong to >1 val 1 km grid (incl. shared tiles)',
                                 scoring_scope='object matching stays per grid (GT is per grid); grid-shared tiles are scored in each grid, as the current scorer does'))
    for name, obj in (('cross_grid.json', xg), ('summary.json', summ)):
        dst = out_dir / name; assert not dst.exists(), f'{dst} exists (no overwrite)'; dst.write_text(json.dumps(obj, indent=1))
    print('DONE')
