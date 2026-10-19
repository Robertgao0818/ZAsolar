#!/usr/bin/env python3
"""CPU-only sliding-window evaluation of authorized phase-1 val/eval grids."""
import argparse
from contextlib import nullcontext
import importlib.metadata
from pathlib import Path
import time

from pipeline_common import *


@torch.inference_mode()
def infer(model, ref, chip=640, stride=480, batch_size=1):
    with rasterio.open(safe_input(ref.raster_path)) as src:
        x = torch.from_numpy(src.read([1, 2, 3]).astype(np.float32))
    h, w = ref.shape
    ph, pw = max(h, chip), max(w, chip)
    if (ph, pw) != (h, w):
        x = F.pad(x, (0, pw - w, 0, ph - h))
    ys = sorted(set(range(0, ph - chip + 1, stride)) | {ph - chip})
    xs = sorted(set(range(0, pw - chip + 1, stride)) | {pw - chip})
    positions = [(y, xx) for y in ys for xx in xs]
    hann = np.hanning(chip + 2)[1:-1]
    window = torch.tensor(np.outer(hann, hann) + 1e-3, dtype=torch.float32)
    prob = torch.zeros(3, ph, pw)
    weight = torch.zeros(ph, pw)
    for start in range(0, len(positions), batch_size):
        batch = positions[start:start + batch_size]
        patches = norm(torch.stack([x[:, y:y + chip, xx:xx + chip] for y, xx in batch]))
        logits = model(pixel_values=patches).logits
        p = F.interpolate(logits.float(), size=(chip, chip), mode='bilinear', align_corners=False).softmax(1)
        for i, (y, xx) in enumerate(batch):
            prob[:, y:y + chip, xx:xx + chip] += p[i] * window
            weight[y:y + chip, xx:xx + chip] += window
    assert bool((weight > 0).all()), 'Sliding window left uncovered pixels'
    return (prob / weight.clamp(min=1e-6)).argmax(0)[:h, :w].numpy().astype(np.uint8), len(positions)


def score_split(out, rows, gm):
    result = {}
    for name, value in CLASSES.items():
        use = rows if name == 'PV' else rows.loc[rows.swh_labelled]
        if use.empty:
            result[name] = {'status': 'not_labelled', 'n_tiles': 0, 'pixel': None, 'objects': None, 'boundary': None}
            continue
        obj = ObjectAccumulator(iou_thrs=(0.3, 0.5, 0.7))
        boundary = BoundaryAccumulator((0.25, 0.5))
        counts = {'tp': 0, 'fp': 0, 'fn': 0, 'n_valid': 0}
        tables = {}
        proxy = np.zeros(4, dtype=np.int64)
        for grid, group in use.groupby('grid', sort=True):
            gt = load_grid_gt(grid, gm)[name]
            tables[grid] = gt
            refs = [ref_for(row) for _, row in group.iterrows()]
            # Bound peak memory to a few tiles, while retaining real neighbour halos.
            from functools import lru_cache
            @lru_cache(maxsize=9)
            def tile_data(tile_id):
                ref = by_id[tile_id]
                tl = class_labels(ref, gt)
                pred = load_prediction(prediction_path(out, ref), ref) == value
                view = make_view(tl, 'exclude-pixels')
                return tl, pred, view
            by_id = {r.tile_id: r for r in refs}
            for ref in refs:
                tl, pred, view = tile_data(ref.tile_id)
                target = tl.mask == value
                valid = view.valid_px
                counts['tp'] += int((pred & target & valid).sum())
                counts['fp'] += int((pred & ~target & valid).sum())
                counts['fn'] += int((~pred & target & valid).sum())
                counts['n_valid'] += int(valid.sum())
                obj.add_tile(ref, PredTile(ref, mask=pred), view)
                # Supplemental imported training hit rules, explicitly not checkpoint-selection F1.
                proxy += np.asarray(object_counts(pred, tl.gt_id > 0, view.region255))
                neigh = neighbours(ref, refs)
                hw = halo_width_px(ref.gsd_m)
                gh = _halo(tl.gt_id > 0, neigh, hw, tile_data, 0)
                eh = _halo(view.region255, neigh, hw, tile_data, 2)
                pp = _halo(pred, neigh, hw, tile_data, 1)
                h, w = ref.shape
                boundary.add_tile(pp, gh, (slice(hw, hw + h), slice(hw, hw + w)), eh,
                                  all(k in neigh for k in ((-1, 0), (1, 0), (0, -1), (0, 1))), ref.gsd_m)
            tile_data.cache_clear()
        den = counts['tp'] + counts['fp'] + counts['fn']
        counts.update(iou=counts['tp'] / den if den else None, union=den,
                      gt_source='v2f semantic mask, mask!=255')
        result[name] = {'status': 'scored', 'n_tiles': len(use), 'grids': sorted(set(use.grid)),
                        'pixel': counts, 'objects': obj.finalize(tables), 'boundary': boundary.finalize(),
                        'training_hit_rule_diagnostic': dict(zip(('hits', 'n_gt', 'tps', 'n_pred'), map(int, proxy)))}
    return result


def _halo(center, neigh, hw, get, which):
    def take(ref):
        tl, pred, view = get(ref.tile_id)
        return (tl.gt_id > 0) if which == 0 else (pred if which == 1 else view.region255)
    # Private helper imported from the pinned scorer, not a reimplementation.
    from zaeval.runner import _assemble_halo
    return _assemble_halo(center, neigh, hw, take)


def markdown(doc):
    lines = ['# Phase-1 val / eval metrics', '',
             f"Scorer: zaeval-2 `{doc['scorer']['commit']}`. Device: CPU float32.", '',
             'PV = photovoltaic panels; SWH = solar water / pool heaters. P/R/F1 are precision/recall/harmonic mean.',
             'All rates below are computed from summed counts, never averaged tile rates. No test data.', '',
             f"Partial run: **{doc['partial']}**. Do not use smoke subsets to rank architectures.", '',
             'Object metrics use raw polygons rasterized at pixel centres, exclude-pixels (255 ignored),',
             '8-connected cross-tile prediction components, minimum prediction area 0.25 m² and one-to-one IoU matching.',
             'Pixel IoU uses the v2f class masks, so the eroded label band is not counted.',
             'Boundary F uses raw polygon outlines, excludes region255, includes the polygon band, and uses neighbour halos.',
             'Missing cardinal neighbours exclude boundaries within 1 m of the core tile edge.', '',
             '| Split / metric | PV | SWH |', '|---|---:|---:|']
    def fmt(v):
        return 'N/A' if v is None else f'{v:.8f}'
    for split, classes in doc['splits'].items():
        def cell(name, kind, key=None, metric=None):
            c = classes[name]
            if c['status'] != 'scored':
                return 'N/A (unlabelled)'
            if kind == 'pixel':
                return fmt(c['pixel']['iou'])
            if kind == 'object':
                b = c['objects']['by_iou'][key]
                return ' / '.join(fmt(b[k]) for k in ('precision', 'recall', 'f1'))
            if kind == 'bin':
                b = c['objects']['by_iou'][key]['bins'][metric]
                return f"{fmt(b['recall'])} ({b['hits']}/{b['n']})" if b['n'] else 'N/A (0 GT)'
            b = next(b for b in c['boundary']['by_tol'] if b['tol_m'] == key)
            return fmt(b['F']) if c['boundary']['n_bpx_pred'] + c['boundary']['n_bpx_gt'] else 'N/A (no boundary; raw scorer F=1)'
        lines.append(f"| {split}: scored tiles | {classes['PV']['n_tiles']} | {classes['SWH']['n_tiles']} |")
        for kind, key, label in [('pixel', None, 'pixel IoU'), *[('object', t, f'object P / R / F1, IoU {t}') for t in ('0.3', '0.5', '0.7')],
                                 ('boundary', 0.25, 'boundary F, 0.25 m'), ('boundary', 0.5, 'boundary F, 0.5 m')]:
            lines.append(f"| {split}: {label} | {cell('PV', kind, key)} | {cell('SWH', kind, key)} |")
        for thr in ('0.3', '0.5', '0.7'):
            scored = next((c for c in classes.values() if c['status'] == 'scored'), None)
            for bin_name in scored['objects']['by_iou'][thr]['bins'] if scored else []:
                lines.append(f"| {split}: recall {bin_name} m² @ IoU {thr} | {cell('PV', 'bin', thr, bin_name)} | {cell('SWH', 'bin', thr, bin_name)} |")
    lines += ['', 'Zero-denominator object rates retain zaeval defaults (0); use the accompanying counts to distinguish absence from failure.',
              'SWH is scored only where swh_labelled=True. Unlabelled grids are not evidence of absence.',
              'The imported training hit-rule diagnostic uses argmax and raw polygons; it is not the training best_objf1 metric.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--arch', choices=ARCHS, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=('val', 'eval', 'both'), default='both')
    p.add_argument('--max-grids', type=int)
    p.add_argument('--grids', help='Optional comma-separated allowlisted val/eval IDs')
    p.add_argument('--max-tiles-per-grid', type=int, help='Smoke only: alternate PV-rich / SWH-rich tiles, with image-path tie breaks')
    p.add_argument('--device', choices=('cpu',), default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=1)
    a = p.parse_args()
    assert a.threads > 0 and a.batch_size > 0
    started = time.perf_counter()
    checkpoint = safe_input(a.ckpt)
    assert checkpoint.is_file(), checkpoint
    scorer = scorer_provenance()
    rows, gm = select_rows(a.split, a.max_grids, a.grids, a.max_tiles_per_grid)
    for r in rows.itertuples():
        assert safe_input(r.image).is_file(), r.image
        assert safe_input(r.mask).is_file(), r.mask
    out = new_output(a.out)
    torch.set_num_threads(a.threads)
    torch.manual_seed(training.VAL_RNG_SEED)
    config_path = checkpoint.parent / 'config.json'
    config = json.loads(safe_input(config_path).read_text()) if config_path.is_file() else {}
    cfg = config.get('args', {})
    assert config.get('arch', a.arch) == a.arch, 'Checkpoint config architecture mismatch'
    assert config.get('data_md5', MANIFEST_MD5) == MANIFEST_MD5
    chip = int(cfg.get('chip', 640))
    assert chip >= 480 and chip % 32 == 0
    model = build_model(a.arch, allow_mscan_download=False, pretrained_convnext=False,
                        upernet_ch=int(cfg.get('upernet_ch', 512)), drop_path=float(cfg.get('drop_path', 0.3)))
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(training.extract_state_dict(blob), strict=True)
    del blob
    model.cpu().eval()
    entries = []
    for i, row in rows.iterrows():
        ref = ref_for(row)
        tick = time.perf_counter()
        pred, windows = infer(model, ref, chip=chip, batch_size=a.batch_size)
        dest = prediction_path(out, ref)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, 'w', driver='GTiff', height=ref.shape[0], width=ref.shape[1], count=1,
                           dtype='uint8', crs=ref.crs, transform=ref.transform, compress='deflate') as dst:
            dst.write(pred, 1)
            dst.update_tags(classes='0=background,1=PV,2=SWH', checkpoint=str(checkpoint))
        entries.append({'split': ref.split, 'grid': ref.grid, 'image': ref.raster_path, 'gt': ref.mask_path,
                        'pred': str(dest), 'swh_labelled': bool(row.swh_labelled), 'windows': windows,
                        'inference_s': time.perf_counter() - tick})
        print(json.dumps({'event': 'tile', 'done': i + 1, 'total': len(rows), **entries[-1]}), flush=True)
    del model
    write_json(out / 'predictions.json', entries)
    scores = {sp: score_split(out, group, gm) for sp, group in rows.groupby('split', sort=True)}
    doc = {'arch': a.arch, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
           'device': 'cpu', 'precision': 'float32', 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
           'scorer': scorer, 'training_source': str(TRAIN), 'training_sha256': digest(TRAIN),
           'protocol_sha256': digest(PROTOCOL), 'manifest_md5': MANIFEST_MD5,
           'grid_manifest_sha256': digest(DATA / 'grid_manifest_v2e.csv'),
           'collectors_sha256': digest(DATA / 'collectors.gpkg'),
           'source_functions_imported': ['ARCHS', 'build_model', 'load_bundle', 'prep', 'norm', '_validate_body', 'object_counts'],
           'source_functions_unused_for_safety': {'load_bundle': 'opens coordinate cache and Li test inventory',
                                                'prep': 'hardcoded CUDA; norm reused directly', '_validate_body': 'hardcoded CUDA'},
           'partial': any(v is not None for v in (a.max_grids, a.grids, a.max_tiles_per_grid)),
           'selection': vars(a), 'inference': {'chip': chip, 'stride': 480, 'window': 'Hann + 1e-3',
                                            'rng_seed': training.VAL_RNG_SEED, 'mask': 'argmax blended softmax'},
           'val_scope': '1703 eligible rows: valid>100000 and no imagery shared with eval; full pool, not training fixed 300 tiles',
           'pixel_aggregation': 'micro over grid/tile observations; shared imagery within a split retains each grid-specific mask',
           'splits': scores, 'runtime_s': time.perf_counter() - started,
           'packages': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'pandas', 'rasterio', 'scipy', 'geopandas', 'transformers', 'timm', 'Pillow')}}
    write_json(out / 'metrics.json', doc)
    (out / 'metrics.md').write_text(markdown(doc))
    print(json.dumps({'event': 'complete', 'out': str(out), 'runtime_s': doc['runtime_s']}), flush=True)


if __name__ == '__main__':
    main()
