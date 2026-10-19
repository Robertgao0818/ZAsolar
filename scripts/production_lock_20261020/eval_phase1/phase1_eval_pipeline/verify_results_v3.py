#!/usr/bin/env python3
"""Independent aggregate regression and bootstrap calculation from emitted counts."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    base=Path(a.eval_dir)
    doc=json.loads((base/'metrics.json').read_text())
    original=json.loads((Path(doc['rescore_source'])/'metrics.json').read_text())
    checked=[]
    for name in ('PV','SWH'):
        sel=doc['splits']['val_sel300'][name]
        held=doc['splits']['val_holdout'][name]
        old=original['splits']['val'][name]
        for key in ('tp','fp','fn','n_valid','union'):
            assert sel['pixel'][key]+held['pixel'][key] == old['pixel'][key]
        assert sel['n_tiles']+held['n_tiles'] == old['n_tiles']
        ev=doc['splits']['eval'][name]
        prior=original['splits']['eval'][name]
        assert ev['pixel'] == prior['pixel']
        for thr in ('0.3','0.5','0.7'):
            assert all(ev['objects']['by_iou'][thr][k] == prior['objects']['by_iou'][thr][k] for k in ('tp','fp','fn','precision','recall','f1'))
            for grid in ev['grids']:
                assert all(ev['objects']['per_grid'][grid]['by_iou'][thr][k] == prior['objects']['per_grid'][grid]['by_iou'][thr][k] for k in ('tp','fp','fn'))
    for scope,classes in doc['splits'].items():
        for name,c in classes.items():
            if c['status'] != 'scored':continue
            b=c['bootstrap_95ci'];g=sorted(c['pixel_by_grid'])
            assert b['grids']==g and len(g)==b['n_grids']
            draws=np.random.default_rng(b['seed']).choice(len(g),size=(b['reps'],len(g)),replace=True)
            for metric,factor in [('pixel_iou',1),('object_f1_0.5',2)]:
                table=c['pixel_by_grid'] if factor==1 else {grid:c['objects']['per_grid'][grid]['by_iou']['0.5'] for grid in g}
                tp=np.array([table[grid]['tp'] for grid in g],dtype=np.int64)
                fp=np.array([table[grid]['fp'] for grid in g],dtype=np.int64)
                fn=np.array([table[grid]['fn'] for grid in g],dtype=np.int64)
                den=(factor*tp+fp+fn)[draws].sum(axis=1)
                num=(factor*tp)[draws].sum(axis=1)
                values=num[den>0]/den[den>0]
                expected=np.percentile(values,[2.5,97.5]).tolist() if len(values) else [None,None]
                assert expected == [b[metric]['low'],b[metric]['high']]
                assert b[metric]['undefined_replicates']==int((den==0).sum())
                total_den=int((factor*tp+fp+fn).sum())
                point=float(factor*tp.sum()/total_den) if total_den else None
                assert b[metric]['estimate']==point
                checked.append(f'{scope}/{name}/{metric}')
    result={'passed':True,'val_pixel_partition_equals_v2_full_pool':True,'eval_pixels_objects_all_thresholds_equal_v2':True,
            'independent_bootstrap_recalculations':checked,'reconciliation_status':doc['training_log_reconciliation']['status']}
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
