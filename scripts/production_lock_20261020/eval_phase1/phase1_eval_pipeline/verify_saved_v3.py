#!/usr/bin/env python3
"""Independent raster-only pixel verification for the three v3 scopes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rasterio

VAL = {'G1293','G1354','G1410','G1523','G1629','G1630','G1689','G1691','G1743','G1798','G1863','G1920','G1979','G2028'}
EVAL = {'G1349','G1686','G1692','G1800','G2038'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    start = time.perf_counter()
    base = Path(a.eval_dir).resolve()
    out = Path(a.out).resolve()
    assert not out.exists() and out.is_relative_to(base)
    doc = json.loads((base / 'metrics.json').read_text())
    entries = json.loads((base / 'predictions.json').read_text())
    source = Path(doc.get('rescore_source', base)).resolve()
    totals = {}
    grids = {}
    seen = set()
    scopes = {}
    for e in entries:
        scope = e['scope_v3']
        assert scope in {'val_sel300','val_holdout','eval'}
        assert e['grid'] in (EVAL if scope == 'eval' else VAL)
        assert e['split'] == ('eval' if scope == 'eval' else 'val')
        key = (e['grid'], Path(e['image']).name)
        assert key not in seen
        seen.add(key)
        scopes[scope] = scopes.get(scope, 0) + 1
        gt_path = Path(e['gt']).resolve()
        pred_path = Path(e['pred']).resolve()
        assert 'test' not in str(gt_path).lower() and 'coord_cache' not in str(gt_path).lower()
        assert pred_path.is_relative_to(source / 'preds')
        assert pred_path == source / 'preds' / e['split'] / e['grid'] / Path(e['image']).name
        with rasterio.open(gt_path) as gt_src, rasterio.open(pred_path) as pred_src:
            assert gt_src.shape == pred_src.shape and gt_src.transform == pred_src.transform and gt_src.crs == pred_src.crs
            assert pred_src.count == 1 and pred_src.dtypes == ('uint8',) and pred_src.nodata is None
            gt = gt_src.read(1)
            pred = pred_src.read(1)
        assert set(np.unique(gt)) <= {0,1,2,255} and set(np.unique(pred)) <= {0,1,2}
        valid = gt != 255
        for name, value in [('PV',1),('SWH',2)]:
            if name == 'SWH' and not e['swh_labelled']:
                continue
            positive, target = pred == value, gt == value
            counts = {'tp': int(np.count_nonzero(valid & positive & target)),
                      'fp': int(np.count_nonzero(valid & positive & ~target)),
                      'fn': int(np.count_nonzero(valid & ~positive & target)),
                      'n_valid': int(np.count_nonzero(valid)), 'n_tiles': 1}
            for collection, key in [(totals,(scope,name)),(grids,(scope,name,e['grid']))]:
                dest = collection.setdefault(key, dict.fromkeys(counts, 0))
                for k,v in counts.items():
                    dest[k] += v
    for (scope,name), c in totals.items():
        expected = doc['splits'][scope][name]
        assert c['n_tiles'] == expected['n_tiles']
        for key in ('tp','fp','fn','n_valid'):
            assert c[key] == expected['pixel'][key], (scope,name,key)
        union = c['tp'] + c['fp'] + c['fn']
        c['iou'] = c['tp'] / union if union else None
        assert c['iou'] == expected['pixel']['iou']
    for (scope,name,grid), c in grids.items():
        expected = doc['splits'][scope][name]['pixel_by_grid'][grid]
        assert all(c[k] == expected[k] for k in ('tp','fp','fn','n_valid'))
    result = {'matched': True, 'independent_of_pipeline_and_scorer': True, 'checked_saved_masks': len(entries),
              'scopes': scopes, 'per_grid_counts_matched': True,
              'totals': {'/'.join(k):v for k,v in totals.items()}, 'runtime_s': time.perf_counter() - start}
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
