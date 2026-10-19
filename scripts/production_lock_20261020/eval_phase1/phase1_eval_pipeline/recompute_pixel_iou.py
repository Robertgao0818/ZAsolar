#!/usr/bin/env python3
"""Independent NumPy/rasterio check: does not import the pipeline or scorer."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import rasterio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    start = time.perf_counter()
    base = Path(a.eval_dir).resolve()
    out = Path(a.out).resolve()
    assert not out.exists() and out.is_relative_to(base), 'Use a fresh result inside evaluation directory'
    metrics = json.loads((base / 'metrics.json').read_text())
    entries = json.loads((base / 'predictions.json').read_text())
    counts = {}
    for e in entries:
        gt_path, pred_path = Path(e['gt']).resolve(), Path(e['pred']).resolve()
        assert 'test' not in str(gt_path).lower() and not re.search(r'coord_cache.*\.npz', str(gt_path))
        assert pred_path.is_relative_to(base)
        assert e['grid'] in {'G1293','G1354','G1410','G1523','G1629','G1630','G1689','G1691','G1743','G1798','G1863','G1920','G1979','G2028',
                             'G1349','G1686','G1692','G1800','G2038'}
        assert e['split'] in ('val', 'eval')
        with rasterio.open(gt_path) as src:
            gt = src.read(1)
        with rasterio.open(pred_path) as src:
            pred = src.read(1)
        assert gt.shape == pred.shape and pred.dtype == np.uint8
        valid = gt != 255
        target, positive = gt == 1, pred == 1
        c = counts.setdefault(e['split'], {'intersection': 0, 'union': 0, 'tp': 0, 'fp': 0, 'fn': 0})
        c['intersection'] += int(np.count_nonzero(valid & target & positive))
        c['union'] += int(np.count_nonzero(valid & (target | positive)))
        c['tp'] += int(np.count_nonzero(valid & target & positive))
        c['fp'] += int(np.count_nonzero(valid & ~target & positive))
        c['fn'] += int(np.count_nonzero(valid & target & ~positive))
    for split, c in counts.items():
        c['iou'] = c['intersection'] / c['union'] if c['union'] else None
        expected = metrics['splits'][split]['PV']['pixel']
        assert c['union'] == c['tp'] + c['fp'] + c['fn']
        assert all(c[k] == expected[k] for k in ('tp', 'fp', 'fn', 'union'))
        c['absolute_error'] = abs(c['iou'] - expected['iou']) if c['iou'] is not None else 0
        assert c['absolute_error'] <= 1e-12
    result = {'independent_of_pipeline_and_scorer': True, 'runtime_s': time.perf_counter() - start,
              'PV': counts, 'matched': True}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
