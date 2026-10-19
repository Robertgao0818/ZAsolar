#!/usr/bin/env python3
"""Small semantic checks of safety guards and the imported scorer contract."""
from pathlib import Path
import tempfile
import time

from pipeline_common import *
from rasterio.transform import from_origin
from shapely.geometry import box


def main():
    start = time.perf_counter()
    checks = []
    for path in ('/nonexistent/test/forbidden.tif', '/nonexistent/Li Test/tiles.csv',
                 '/nonexistent/coord_cache_fake.npz', '/nonexistent/G2905/imagery.tif'):
        try:
            safe_input(path)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'Guard accepted forbidden path: {path}')
    checks.append('Forbidden test paths, coordinate caches and nonallowlisted grids rejected before open')
    with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as tmp:
        target = Path(tmp) / 'fresh'
        new_output(target)
        try:
            new_output(target)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output directory was reused')
    checks.append('Existing output directory refused')
    # One ground-truth polygon spans two tiles. A merge failure would produce 2 predictions.
    gt = grid_gt_from_rows([{'cls': 'PV', 'geometry': box(1.2, 0.4, 2.0, 1.2)}], 'EPSG:32734')
    obj = ObjectAccumulator(iou_thrs=(0.3, 0.5, 0.7))
    for col in (0, 1):
        ref = TileRef(f'synthetic_{col}', 'G1686', from_origin(col * 1.6, 1.6, 0.1, 0.1),
                      'EPSG:32734', (16, 16), 0.1, None, 'synthetic.tif', 'eval')
        ids = rasterize([(gt.geometry.iloc[0], 1)], out_shape=ref.shape, transform=ref.transform, dtype='int32')
        tl = TileLabels(ref, np.zeros(ref.shape, dtype=np.uint8), ids)
        obj.add_tile(ref, PredTile(ref, mask=ids > 0), make_view(tl, 'exclude-pixels'))
    result = obj.finalize({'G1686': gt})
    assert result['n_gt'] == result['n_pred'] == 1, result
    for threshold, cell in result['by_iou'].items():
        assert cell['tp'] == 1 and cell['fp'] == cell['fn'] == 0 and cell['f1'] == 1
        assert sum(b['n'] for b in cell['bins'].values()) == 1
        assert next(iter(cell['bins'].values()))['hits'] == 1
    checks.append('Cross-tile object merge, one-to-one IoU matching at 0.3/0.5/0.7 and polygon-area bins passed')
    raw = np.zeros((64, 64), dtype=bool)
    raw[20:40, 20:40] = True
    bf = BoundaryAccumulator((0.25, 0.5))
    bf.add_tile(raw, raw, (slice(0, 64), slice(0, 64)), np.zeros_like(raw), True, 0.1)
    b = bf.finalize()
    assert b['n_bpx_pred'] == b['n_bpx_gt'] > 0
    assert all(row['F'] == 1 for row in b['by_tol'])
    checks.append('Perfect nonempty polygon boundary has F=1 at 0.25 m and 0.5 m')
    report = {'passed': True, 'checks': checks, 'runtime_s': time.perf_counter() - start, 'scorer': scorer_provenance()}
    write_json(ROOT / 'contract_checks.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
