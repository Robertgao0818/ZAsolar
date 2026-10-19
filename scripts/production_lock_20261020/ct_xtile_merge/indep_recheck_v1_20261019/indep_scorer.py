#!/usr/bin/env python3
"""Independent G7 scorer (CPU). Own 8-conn labelling, own cross-tile union-find on global border pixels,
own pair IoU (valid px only), own greedy 1:1 match (desc IoU, tie pred idx, gt idx), min_area 0.25 m2,
drop preds with 0 valid px, GT in scope iff n_valid>0 (rules read from zaeval objects.py @8044d14).
Shared inputs reused only for data prep: row selection, ref_for, load_grid_gt, class_labels, make_view(exclude-pixels), binarize.
Variants: scope=asis (shared tiles scored in every grid) | dedup_owner (shared tile scored only in owner grid by tile centre
in real 1 km cell) | dedup_first (owner = lexicographically first grid)."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import numpy as np
from scipy import ndimage
PIPE = Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser(); sys.path.insert(0, str(PIPE))
from pipeline_common_v3 import select_rows, validation_scopes, ref_for, load_grid_gt, manifests, class_labels
from zaeval.ignore import make_view
from zaeval.postproc import binarize
import pandas as pd
PROBS = Path('~/capetown_eval/ds3m_ft/phase1_followup_valprobs_20261018/convnext_t_s0').expanduser()
CELLS = Path('~/projects/ZAsolar/data/task_grid_ct_full.gpkg').expanduser()
CFG = sys.argv[2] if len(sys.argv) > 2 else 'h0.9_0.5'
S8 = np.ones((3, 3), bool)

class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, a):
        while s.p[a] != a: s.p[a] = s.p[s.p[a]]; a = s.p[a]
        return a
    def u(s, a, b):
        a, b = s.f(a), s.f(b)
        if a != b: s.p[max(a, b)] = min(a, b)

def score_grid(grid, refs, gt):
    parts = []; edges = {}; gtv = {}
    for ref in refs:
        prob = np.load(PROBS / grid / f'{ref.tile_id}.npz')['p'].astype(np.float32) / 255.0
        m = binarize(prob, CFG).astype(bool)
        view = make_view(class_labels(ref, gt), 'exclude-pixels')
        v = view.valid_px; gid = view.gt_id
        o, c = np.unique(gid[v & (gid > 0)], return_counts=True)
        for a, b in zip(o.tolist(), c.tolist()): gtv[a] = gtv.get(a, 0) + b
        lab, n = ndimage.label(m, S8)
        if n == 0: continue
        npx = np.bincount(lab.ravel(), minlength=n + 1); nv = np.bincount(lab[v].ravel(), minlength=n + 1)
        sel = v & (lab > 0) & (gid > 0)
        inter = [dict() for _ in range(n + 1)]
        if sel.any():
            kk, cc = np.unique(lab[sel].astype(np.int64) * (1 << 32) + gid[sel].astype(np.int64), return_counts=True)
            for k, cnt in zip(kk.tolist(), cc.tolist()): inter[k >> 32][k & 0xffffffff] = cnt
        base = len(parts)
        for i in range(1, n + 1): parts.append([int(npx[i]), int(nv[i]), inter[i]])
        gx0 = int(round(ref.transform.c / ref.gsd_m)); gy0 = int(round(-ref.transform.f / ref.gsd_m))
        h, w = m.shape
        bm = np.zeros_like(m); bm[0] = bm[-1] = True; bm[:, 0] = bm[:, -1] = True
        ys, xs = np.nonzero(bm & (lab > 0))
        for y, x in zip(ys.tolist(), xs.tolist()):
            key = (gx0 + x, gy0 + y); assert key not in edges, 'overlap'
            edges[key] = base + int(lab[y, x]) - 1
    out = {}
    for merge in (False, True):
        uf = UF(len(parts))
        if merge:
            for (x, y), k in edges.items():
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        j = edges.get((x + dx, y + dy))
                        if j is not None and j != k: uf.u(j, k)
        grp = {}
        for i in range(len(parts)): grp.setdefault(uf.f(i), []).append(i)
        preds = []
        for r in sorted(grp):
            npx = sum(parts[i][0] for i in grp[r]); nv = sum(parts[i][1] for i in grp[r]); it = {}
            for i in grp[r]:
                for o, c in parts[i][2].items(): it[o] = it.get(o, 0) + c
            area = npx * refs[0].gsd_m ** 2
            if area < 0.25 or nv == 0: continue
            preds.append((npx, nv, it, area))
        gts = [o for o in gt.oid.astype(int).tolist() if gtv.get(o, 0) > 0]; gset = set(gts)
        pairs = []
        for pi, (npx, nv, it, _) in enumerate(preds):
            for o, c in it.items():
                if o in gset and c > 0: pairs.append((c / (nv + gtv[o] - c), pi, o))
        res = {}
        for thr in (0.3, 0.5):
            el = sorted([p for p in pairs if p[0] >= thr], key=lambda p: (-p[0], p[1], p[2]))
            up, ug, ious = set(), set(), []
            for iou, pi, o in el:
                if pi in up or o in ug: continue
                up.add(pi); ug.add(o); ious.append(iou)
            res[str(thr)] = dict(tp=len(up), fp=len(preds) - len(up), fn=len(gts) - len(ug), sum_iou=float(sum(ious)))
        out['A1' if merge else 'A0'] = dict(by_iou=res, n_pred=len(preds), n_gt=len(gts),
                                             pred_area_m2=float(sum(p[3] for p in preds)), n_parts=len(parts))
    return out

def main():
    outd = Path(sys.argv[1]); outd.mkdir(parents=True, exist_ok=True)
    rows, gm = select_rows('val'); rows, gm, _ = validation_scopes(rows); assert len(rows) == 1703
    refs = {g: [ref_for(pd.Series(r)) for r in grp.to_dict('records')] for g, grp in rows.groupby('grid')}
    grid_of = {}
    for g, v in refs.items():
        for r in v: grid_of.setdefault(r.tile_id, []).append(g)
    shared = {k: sorted(v) for k, v in grid_of.items() if len(v) > 1}
    import geopandas as gpd
    from shapely.geometry import Point
    F = gpd.read_file(CELLS).to_crs('EPSG:32734'); F = F[F.gridcell_id.isin(list(refs))]
    owner = {}
    for k, gs in shared.items():
        r = next(x for x in refs[gs[0]] if x.tile_id == k)
        pt = Point(r.transform.c + r.shape[1] * r.gsd_m / 2, r.transform.f - r.shape[0] * r.gsd_m / 2)
        hit = [g for g in gs if F[F.gridcell_id == g].geometry.iloc[0].contains(pt)]
        owner[k] = dict(cells_containing_centre=hit, owner=hit[0] if len(hit) == 1 else gs[0], grids=gs)
    scopes = {'asis': lambda g, t: True,
              'dedup_owner': lambda g, t: t not in shared or owner[t]['owner'] == g,
              'dedup_first': lambda g, t: t not in shared or shared[t][0] == g}
    gts = {g: load_grid_gt(g, gm)['PV'] for g in refs}
    allres = {}
    for sc, keep in scopes.items():
        agg = {}
        for g in sorted(refs):
            r = score_grid(g, [x for x in refs[g] if keep(g, x.tile_id)], gts[g])
            for arm, d in r.items():
                a = agg.setdefault(arm, dict(by_iou={}, n_pred=0, n_gt=0, pred_area_m2=0.0, n_tiles=0))
                for k in ('n_pred', 'n_gt', 'pred_area_m2'): a[k] += d[k]
                for t, b in d['by_iou'].items():
                    x = a['by_iou'].setdefault(t, dict(tp=0, fp=0, fn=0, sum_iou=0.0))
                    for kk in x: x[kk] += b[kk]
            print(sc, g, flush=True)
        for a in agg.values():
            for b in a['by_iou'].values():
                P = b['tp'] / max(b['tp'] + b['fp'], 1); R = b['tp'] / max(b['tp'] + b['fn'], 1)
                b.update(P=round(P, 4), R=round(R, 4), F1=round(2 * P * R / max(P + R, 1e-9), 4), mean_iou_tp=round(b['sum_iou'] / max(b['tp'], 1), 4))
        allres[sc] = agg
        (outd / 'indep_results.json').write_text(json.dumps(dict(cfg=CFG, owner=owner, results=allres), indent=1))
    print('DONE')
main()
