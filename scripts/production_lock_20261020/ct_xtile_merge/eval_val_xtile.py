#!/usr/bin/env python3
"""G7 VAL-ONLY evaluation of cross-tile merge arms on cached ConvNeXt-T s0 val probs (CPU only, no inference).
A0 per-tile binarize, no merge | A1 per-tile binarize + cross-tile merge (current scorer) | A2 cross-tile hysteresis + merge.
usage: eval_val_xtile.py OUTDIR [NPROC]   (resumable: per-grid JSON in OUTDIR/grids/)"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
from multiprocessing import Pool
PIPE = Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser()
sys.path.insert(0, str(PIPE)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common_v3 import *  # noqa  (select_rows, validation_scopes, manifests, load_grid_gt, ref_for, class_labels, make_view, ObjectAccumulator, PredTile)
from zaeval.postproc import binarize
from xtile_merge import TileIn, merge_tiles, per_tile_masks, global_origin
PROBS = Path('~/capetown_eval/ds3m_ft/phase1_followup_valprobs_20261018/convnext_t_s0').expanduser()
CFGS = ('h0.9_0.5', 't0.5'); IOUS = (0.3, 0.5)

class NoMergeAccumulator(ObjectAccumulator):
    """Each per-tile component is its own object (production without cross-tile merge)."""
    def _merge_parts(self):
        saved, self._edges = self._edges, {}
        try:
            return super()._merge_parts()
        finally:
            self._edges = saved

def score_grid(args):
    out_dir, grid, rowdicts = args
    f = Path(out_dir) / 'grids' / f'{grid}.json'
    if f.exists(): return json.loads(f.read_text())
    import pandas as pd
    gm = manifests()[1]; gt = load_grid_gt(grid, gm)['PV']
    refs = [ref_for(pd.Series(r)) for r in rowdicts]
    load = lambda ref: (lambda: np.load(PROBS / grid / f'{ref.tile_id}.npz')['p'].astype(np.float32) / 255.0)
    tiles = [TileIn(r.tile_id, *global_origin(r.transform, r.gsd_m), load(r)) for r in refs]
    res = {}
    for cfg in CFGS:
        pt = per_tile_masks(tiles, cfg)
        mg, _, objs = merge_tiles(tiles, cfg)
        # parity: per-tile rule == pinned zaeval binarize
        r0 = refs[0]; assert (pt[r0.tile_id] == binarize(load(r0)(), cfg)).all()
        arms = {'A0': (NoMergeAccumulator, pt), 'A1': (ObjectAccumulator, pt), 'A2': (ObjectAccumulator, mg)}
        for arm, (Acc, masks) in arms.items():
            acc = Acc(iou_thrs=IOUS); px = dict(tp=0, fp=0, fn=0); area = 0.0
            for ref in refs:
                view = make_view(class_labels(ref, gt), 'exclude-pixels'); m = masks[ref.tile_id]
                acc.add_tile(ref, PredTile(ref, mask=m), view)
                g = view.gt_id > 0; v = view.valid_px
                px['tp'] += int((m & g & v).sum()); px['fp'] += int((m & ~g & v).sum()); px['fn'] += int((~m & g & v).sum())
                area += float(m.sum()) * ref.gsd_m ** 2
            fin = acc.finalize({grid: gt})
            res[f'{cfg}/{arm}'] = dict(px=px, pred_area_m2=area,
                by_iou={k: {kk: fin['by_iou'][k][kk] for kk in ('tp', 'fp', 'fn')} for k in fin['by_iou']},
                n_pred=fin.get('n_pred', fin.get('preds', {}).get('n') if isinstance(fin.get('preds'), dict) else None),
                n_components_merged=fin.get('n_components_merged'))
        res[f'{cfg}/A2']['n_multi_tile_objects'] = sum(o['n_tiles'] > 1 for o in objs.values())
        res[f'{cfg}/A2']['gt_area_m2'] = float(gt.area.sum()) if len(gt) else 0.0
    out = dict(grid=grid, tiles=len(refs), res=res)
    f.parent.mkdir(parents=True, exist_ok=True); tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(out)); os.replace(tmp, f)
    print(json.dumps(dict(grid_done=grid, tiles=len(refs))), flush=True)
    return out

if __name__ == '__main__':
    out_dir = sys.argv[1]; npr = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rows, gm = select_rows('val'); rows, gm, _ = validation_scopes(rows)
    assert len(rows) == 1703 and set(rows.split) == {'val'}
    if os.environ.get('GRIDS'): rows = rows[rows.grid.isin(os.environ['GRIDS'].split(','))]
    jobs = [(out_dir, g, grp.to_dict('records')) for g, grp in rows.groupby('grid')]
    with Pool(npr) as p: outs = p.map(score_grid, jobs)
    agg = {}
    for o in outs:
        for k, r in o['res'].items():
            a = agg.setdefault(k, dict(px=dict(tp=0, fp=0, fn=0), pred_area_m2=0.0, by_iou={}, n_multi_tile_objects=0, gt_area_m2=0.0))
            for kk in a['px']: a['px'][kk] += r['px'][kk]
            a['pred_area_m2'] += r['pred_area_m2']; a['n_multi_tile_objects'] += r.get('n_multi_tile_objects', 0); a['gt_area_m2'] += r.get('gt_area_m2', 0.0)
            for t, c in r['by_iou'].items():
                b = a['by_iou'].setdefault(t, dict(tp=0, fp=0, fn=0))
                for kk in b: b[kk] += c[kk]
    for k, a in agg.items():
        p = a['px']; a['px_iou'] = round(p['tp'] / max(p['tp'] + p['fp'] + p['fn'], 1), 4)
        for t, b in a['by_iou'].items():
            P = b['tp'] / max(b['tp'] + b['fp'], 1); R = b['tp'] / max(b['tp'] + b['fn'], 1)
            b.update(P=round(P, 4), R=round(R, 4), F1=round(2 * P * R / max(P + R, 1e-9), 4))
    Path(out_dir, 'summary.json').write_text(json.dumps(dict(grids=len(outs), tiles=sum(o['tiles'] for o in outs), agg=agg), indent=1))
    print('DONE')
