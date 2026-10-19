#!/usr/bin/env python3
"""CPU/GPU sliding-window evaluation of authorized phase-1 val/eval grids."""
import argparse
from contextlib import nullcontext
import importlib.metadata
from pathlib import Path
import time

from pipeline_common_v3 import *
from validation_metrics_v3 import score_scopes, reconcile_training_log, markdown_v3


@torch.inference_mode()
def infer(model, ref, chip=640, stride=480, batch_size=1, device='cpu', precision='float32'):
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
        patches = patches.to(device)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' and precision == 'bfloat16' else nullcontext()
        with context:
            logits = model(pixel_values=patches).logits
        p = F.interpolate(logits.float(), size=(chip, chip), mode='bilinear', align_corners=False).softmax(1).cpu()
        del logits, patches
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
        pixel_by_grid = {}
        for grid, group in use.groupby('grid', sort=True):
            print(json.dumps({'event': 'score_grid', 'class': name, 'grid': grid, 'tiles': len(group)}), flush=True)
            before = counts.copy()
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
            pixel_by_grid[grid] = {k: counts[k] - before[k] for k in before}
            tile_data.cache_clear()
        den = counts['tp'] + counts['fp'] + counts['fn']
        counts.update(iou=counts['tp'] / den if den else None, union=den,
                      gt_source='v2f semantic mask, mask!=255')
        result[name] = {'status': 'scored', 'n_tiles': len(use), 'grids': sorted(set(use.grid)),
                        'pixel': counts, 'pixel_by_grid': pixel_by_grid, 'objects': obj.finalize(tables), 'boundary': boundary.finalize(),
                        'training_hit_rule_diagnostic': dict(zip(('hits', 'n_gt', 'tps', 'n_pred'), map(int, proxy)))}
    return result


def _halo(center, neigh, hw, get, which):
    def take(ref):
        tl, pred, view = get(ref.tile_id)
        return (tl.gt_id > 0) if which == 0 else (pred if which == 1 else view.region255)
    # Private helper imported from the pinned scorer, not a reimplementation.
    from zaeval.runner import _assemble_halo
    return _assemble_halo(center, neigh, hw, take)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--arch', choices=ARCHS, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=('val', 'eval', 'both'), default='both')
    p.add_argument('--max-grids', type=int)
    p.add_argument('--grids', help='Optional comma-separated allowlisted val/eval IDs')
    p.add_argument('--max-tiles-per-grid', type=int, help='Smoke only: alternate PV-rich / SWH-rich tiles, with image-path tie breaks')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--data-root')
    p.add_argument('--training-log', help='Optional explicit log.jsonl; default checkpoint-adjacent')
    p.add_argument('--precision', choices=('float32', 'bfloat16'), default='float32')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=1)
    a = p.parse_args()
    assert a.threads > 0 and a.batch_size > 0
    started = time.perf_counter()
    checkpoint = safe_input(a.ckpt)
    assert checkpoint.is_file(), checkpoint
    scorer = scorer_provenance()
    rows, gm = select_rows(a.split, a.max_grids, a.grids, a.max_tiles_per_grid)
    rows, gm, selection_provenance = validation_scopes(rows)
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
    if 'script_md5' in config:
        assert config['script_md5'] == digest(TRAIN, 'md5'), 'Training source MD5 differs from checkpoint config'
    cfg_arch = cfg.get('arch', a.arch)
    assert cfg_arch == a.arch
    assert not cfg.get('smoke', False) and not cfg.get('allow_shared_tiles', False)
    assert cfg.get('val_tiles', 300) == 300 and (cfg.get('max_val_tiles') is None or cfg['max_val_tiles'] >= 300)
    chip = int(cfg.get('chip', 640))
    assert chip >= 480 and chip % 32 == 0
    model = build_model(a.arch, init=str(safe_input((ROOT if BUNDLED else ROOT / 'pod_bundle') / 'sources/weights/mscan_b_20230227-3ab7d230.pth')) if a.arch == 'segnext_b' else None, allow_mscan_download=False, pretrained_convnext=False,
                        upernet_ch=int(cfg.get('upernet_ch', 512)), drop_path=float(cfg.get('drop_path', 0.3)))
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(training.extract_state_dict(blob), strict=True)
    del blob
    if a.device == 'cuda':
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0', 'Only explicitly authorized GPU 0'
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(min(0.85, 7.5e9 / torch.cuda.get_device_properties(0).total_memory), 0)
        torch.cuda.reset_peak_memory_stats()
    model.to(a.device).eval()
    inference_started = time.perf_counter()
    entries = []
    for i, row in rows.iterrows():
        ref = ref_for(row)
        tick = time.perf_counter()
        pred, windows = infer(model, ref, chip=chip, batch_size=a.batch_size, device=a.device, precision=a.precision)
        dest = prediction_path(out, ref)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, 'w', driver='GTiff', height=ref.shape[0], width=ref.shape[1], count=1,
                           dtype='uint8', crs=ref.crs, transform=ref.transform, compress='deflate') as dst:
            dst.write(pred, 1)
            dst.update_tags(classes='0=background,1=PV,2=SWH', checkpoint=str(checkpoint))
        entries.append({'split': ref.split, 'grid': ref.grid, 'image': ref.raster_path, 'gt': ref.mask_path,
                        'pred': str(dest), 'scope_v3': row.scope_v3, 'load_bundle_row': int(row.load_bundle_row),
                        'selection_rank': int(row.selection_rank) if pd.notna(row.selection_rank) else None, 'swh_labelled': bool(row.swh_labelled), 'windows': windows,
                        'inference_s': time.perf_counter() - tick})
        print(json.dumps({'event': 'tile', 'done': i + 1, 'total': len(rows), **entries[-1]}), flush=True)
    inference_runtime = time.perf_counter() - inference_started
    gpu = None
    if a.device == 'cuda':
        torch.cuda.synchronize()
        gpu = {'name': torch.cuda.get_device_name(0), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'process_limit_bytes': 7.5e9}
        assert gpu['peak_reserved_bytes'] < 8e9
    del model
    if a.device == 'cuda':
        torch.cuda.empty_cache()
    print(json.dumps({'event': 'scoring_started', 'inference_runtime_s': inference_runtime, 'gpu': gpu}), flush=True)
    write_json(out / 'predictions.json', entries)
    scores = score_scopes(out, rows, gm, score_split)
    doc = {'arch': a.arch, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
           'device': a.device, 'precision': a.precision, 'gpu': gpu, 'inference_runtime_s': inference_runtime, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
           'scorer': scorer, 'training_source': str(TRAIN), 'training_sha256': digest(TRAIN),
           'protocol_sha256': digest(PROTOCOL), 'manifest_md5': MANIFEST_MD5,
           'grid_manifest_sha256': digest(METADATA / 'grid_manifest_v2e.csv'),
           'collectors_sha256': digest(METADATA / 'collectors.gpkg'),
           'source_functions_imported': ['ARCHS', 'build_model', 'load_bundle', 'prep', 'norm', '_validate_body', 'object_counts'],
           'source_functions_unused_for_safety': {'load_bundle': 'opens coordinate cache and Li test inventory',
                                                'prep': 'hardcoded CUDA; norm reused directly', '_validate_body': 'hardcoded CUDA'},
           'partial': any(v is not None for v in (a.max_grids, a.grids, a.max_tiles_per_grid)),
           'selection': vars(a), 'inference': {'chip': chip, 'stride': 480, 'window': 'Hann + 1e-3',
                                            'rng_seed': training.VAL_RNG_SEED, 'mask': 'argmax blended softmax'},
           'val_scope': 'val_sel300=300 training selection rows; val_holdout=1403 remaining rows (main)',
           'selection_provenance_v3': selection_provenance,
           'pixel_aggregation': 'micro over grid/tile observations; shared imagery within a split retains each grid-specific mask',
           'splits': scores, 'runtime_s': time.perf_counter() - started,
           'packages': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'pandas', 'rasterio', 'scipy', 'geopandas', 'transformers', 'timm', 'Pillow')}}
    doc['training_log_reconciliation'] = reconcile_training_log(checkpoint, scores['val_sel300'],
        {'precision': a.precision, 'batch_size': a.batch_size}, a.training_log) if 'val_sel300' in scores and scores['val_sel300']['PV']['n_tiles'] == 300 else {'status': 'unavailable', 'reason': 'No complete val_sel300 in this run'}
    rows.to_csv(out / 'scope_rows_v3.csv', index=False, mode='x')
    write_json(out / 'metrics.json', doc)
    (out / 'metrics.md').write_text(markdown_v3(doc))
    print(json.dumps({'event': 'complete', 'out': str(out), 'runtime_s': doc['runtime_s']}), flush=True)


if __name__ == '__main__':
    main()
