"""Read-only phase-1 inputs and the pinned, imported metric implementation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
RUNTIME = ROOT / '.runtime'
RUNTIME.mkdir(exist_ok=True)
for key in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME', 'MPLCONFIGDIR'):
    os.environ[key] = str(RUNTIME)

# Parse only relocation here, before imports; the entry point validates all options.
def cli_data_root():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-root')
    args, _ = parser.parse_known_args()
    return Path(args.data_root).expanduser().resolve() if args.data_root else None

DATA_ROOT = cli_data_root()
BUNDLED = (ROOT / 'bundle_provenance.json').is_file()
TRAIN = ROOT / 'sources/train2_v2e.py' if BUNDLED else Path('~/capetown_eval/ds3m_ft/train2_v2e_20261017/train2_v2e.py').expanduser()
PINNED = Path('~/capetown_eval/ds3m_ft/v2f_255_r2_20261017/train2_pinned_v2f.py').expanduser()
# A same-named local audit helper is not a training model implementation.
import ast
if PINNED.is_file() and any(isinstance(n, (ast.FunctionDef, ast.ImportFrom)) and
    (getattr(n, 'name', None) == 'build_model' or any(a.name == 'build_model' for a in getattr(n, 'names', [])))
    for n in ast.parse(PINNED.read_text()).body):
    TRAIN = PINNED

SCORER = ROOT / 'vendor/zaeval' if BUNDLED else Path('~/projects/zasolar_evalfix_8044d14_wt').expanduser()
DATA = Path('~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2f_20261017/data').expanduser()
PROTOCOL = ROOT / 'sources/phase1_run_protocol_v2.md' if BUNDLED else Path('~/projects/zasolar_plans_20261017/phase1_run_protocol_v2.md').expanduser()
METADATA = ROOT / 'metadata' if BUNDLED else DATA
SCORER_COMMIT = '8044d1447a92cc100bbc01976bf66cb4245c89bb'
MANIFEST_MD5 = 'f182b9c3c770e533aa7e629ca529550e'
EVAL_GRIDS = frozenset(('G1349', 'G1686', 'G1692', 'G1800', 'G2038'))
VAL_GRIDS = frozenset(('G1293', 'G1354', 'G1410', 'G1523', 'G1629', 'G1630',
                        'G1689', 'G1691', 'G1743', 'G1798', 'G1863', 'G1920', 'G1979', 'G2028'))
ALLOWED_GRIDS = VAL_GRIDS | EVAL_GRIDS
CLASSES = {'PV': 1, 'SWH': 2}


def safe_input(path):
    """Reject forbidden inputs before stat, open, rasterio or geopandas access."""
    raw = os.fspath(path)
    def check(value):
        assert 'test' not in value.lower(), f'Test input forbidden: {value}'
        assert not re.search(r'coord_cache[^/]*\.npz', value, re.I), 'Coordinate caches forbidden'
        ids = set(re.findall(r'(?<![A-Za-z0-9])G\d{4}(?!\d)', value))
        assert ids <= ALLOWED_GRIDS, f'Non-val/eval grid forbidden: {ids - ALLOWED_GRIDS}'
    check(raw)
    resolved = Path(raw).expanduser().resolve()
    check(str(resolved))
    return resolved


def _input_audit(event, args):
    # Defense in depth: an accidental load_bundle call cannot open Li or a cache.
    # Library source modules (e.g. torch.testing) are not datasets.
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0])
    assert not re.search(r'coord_cache[^/]*\.npz', path, re.I), 'Coordinate caches forbidden'
    if Path(path).suffix.lower() in {'.tif', '.tiff', '.gpkg', '.csv', '.npz', '.pt'}:
        if not Path(path).is_relative_to(ROOT):
            assert 'test' not in path.lower(), f'Test data forbidden: {path}'
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_relative_to(ROOT):
            assert 'test' not in path.lower() and 'test' not in str(resolved).lower(), f'Test data forbidden: {path}'


sys.addaudithook(_input_audit)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import torch
import torch.nn.functional as F

_original_home = os.environ.get('HOME')
_original_cuda = os.environ['CUDA_VISIBLE_DEVICES']
if BUNDLED:
    os.environ['HOME'] = str(ROOT / 'home_mirror')
spec = importlib.util.spec_from_file_location('phase1_training_source', safe_input(TRAIN))
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
# Import, never copy or monkeypatch. The unsafe entry points are intentionally unused.
build_model = training.build_model
ARCHS = training.ARCHS
load_bundle = training.load_bundle
prep = training.prep
norm = training.norm
_validate_body = training._validate_body
object_counts = training.object_counts

sys.path.insert(0, str(safe_input(SCORER)))
import zaeval
from zaeval.boundary import BoundaryAccumulator, halo_width_px
from zaeval.ignore import make_view
from zaeval.io import neighbours
from zaeval.labelsets.common import empty_grid_gt, grid_gt_from_rows
from zaeval.labelsets.ct_sam2 import load_ct_sam2_grid
from zaeval.objects import ObjectAccumulator
from zaeval.runner import _assemble_halo, to_jsonable
from zaeval.types import PredTile, TileLabels, TileRef
if _original_home is not None:
    os.environ['HOME'] = _original_home
else:
    os.environ.pop('HOME', None)
os.environ['CUDA_VISIBLE_DEVICES'] = _original_cuda


def digest(path, kind='sha256'):
    h = hashlib.new(kind)
    with safe_input(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scorer_provenance():
    assert Path(zaeval.__file__).resolve().is_relative_to(SCORER.resolve())
    if BUNDLED:
        provenance = json.loads((ROOT / 'bundle_provenance.json').read_text())
        assert digest(ROOT / 'zaeval_8044d14.tar.gz') == provenance['scorer_archive_sha256']
        for rel, expected in provenance['scorer_files_sha256'].items():
            assert digest(SCORER / rel) == expected, f'Archived scorer differs: {rel}'
        assert zaeval.METRIC_DEF_VERSION == 'zaeval-2'
        return {'path': str(SCORER), 'commit': SCORER_COMMIT,
                'metric_def_version': zaeval.METRIC_DEF_VERSION, 'archive_verified': True}
    head = subprocess.check_output(['git', '-C', str(SCORER), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == SCORER_COMMIT, f'Scorer commit mismatch: {head}'
    changed = subprocess.check_output(['git', '-C', str(SCORER), 'diff', 'HEAD', '--', 'zaeval'], text=True)
    assert not changed, 'Scorer tracked source differs from pinned HEAD'
    assert zaeval.METRIC_DEF_VERSION == 'zaeval-2'
    return {'path': str(SCORER), 'commit': head, 'metric_def_version': zaeval.METRIC_DEF_VERSION}


def manifests():
    manifest = safe_input(METADATA / 'manifest.csv')
    assert digest(manifest, 'md5') == MANIFEST_MD5, 'v2f manifest MD5 mismatch'
    m = pd.read_csv(manifest)
    assert set(m.split) == {'train', 'val', 'eval'}, 'Unexpected data split'
    # Only inspect data-file paths from the two authorized splits.
    m = m.loc[m.split.isin(['val', 'eval'])].copy()
    gm = pd.read_csv(safe_input(METADATA / 'grid_manifest_v2e.csv'), keep_default_na=False)
    gm = gm.loc[gm.source_grid_id.isin(ALLOWED_GRIDS)].copy().set_index('source_grid_id')
    assert set(m.loc[m.split == 'val', 'grid']) == VAL_GRIDS
    assert set(m.loc[m.split == 'eval', 'grid']) == EVAL_GRIDS
    assert not m.duplicated(['grid', 'image']).any()
    assert set(gm.index) == ALLOWED_GRIDS
    assert set(gm.loc[list(VAL_GRIDS), 'split']) == {'val'}
    assert set(gm.loc[list(EVAL_GRIDS), 'split']) == {'eval_orig10'}
    if DATA_ROOT:
        candidates = [DATA_ROOT / 'manifest_relative.csv', DATA_ROOT / 'data/manifest_relative.csv']
        relative = next((p for p in candidates if p.is_file()), None)
        assert relative is not None, 'Missing manifest_relative.csv under --data-root'
        rm = pd.read_csv(safe_input(relative))
        rm = rm.loc[rm.split.isin(['val', 'eval'])].copy()
        assert set(zip(m.grid, m.image.map(lambda x: Path(x).stem))) == set(zip(rm.grid, rm.image.map(lambda x: Path(x).stem)))
        rm['_key'] = list(zip(rm.grid, rm.image.map(lambda x: Path(x).stem)))
        rm = rm.set_index('_key')
        for idx, r in m.iterrows():
            rr = rm.loc[[(r.grid, Path(r.image).stem)]].iloc[0]
            for field in ('image', 'mask'):
                value = Path(rr[field])
                assert not value.is_absolute(), f'{field} must be package-relative'
                resolved = safe_input(DATA_ROOT / value)
                assert resolved.is_relative_to(DATA_ROOT), 'Relative manifest escaped data root'
                m.at[idx, field] = str(resolved)
        assert not m.duplicated(['grid', 'image']).any()
    if BUNDLED:
        for grid in gm.index:
            if gm.at[grid, 'label_file']:
                gm.at[grid, 'label_file'] = str(ROOT / 'labels' / Path(gm.at[grid, 'label_file']).name)
    m['valid'] = m.n_bg + m.n_pv + m.n_swh
    shared = set(m.loc[m.split == 'eval', 'image'])
    m['included_default'] = (m.split == 'eval') | ((m.valid > 100000) & ~m.image.isin(shared))
    assert len(m.loc[(m.split == 'val') & m.included_default]) == 1703
    assert m.swh_labelled.dtype == bool
    for r in m.itertuples():
        assert r.grid in ALLOWED_GRIDS
        safe_input(r.image)
        safe_input(r.mask)
    return m, gm


def write_inventory(dest):
    m, gm = manifests()
    out = m[['split', 'grid', 'image', 'mask', 'swh_labelled', 'included_default']].copy()
    out['grid_manifest_split'] = out.grid.map(gm.split)
    out['image_exists'] = [safe_input(p).is_file() for p in out.image]
    out['mask_exists'] = [safe_input(p).is_file() for p in out['mask']]
    out.to_csv(dest, index=False)
    return {'rows': len(out), 'unique_images': out.image.nunique(),
            'all_images_exist': bool(out.image_exists.all()), 'all_masks_exist': bool(out.mask_exists.all()),
            'val_rows_default': int(((out.split == 'val') & out.included_default).sum()),
            'eval_rows_default': int((out.split == 'eval').sum())}


def new_output(path):
    dest = Path(path).expanduser().resolve()
    # No existing output, including a symlink, is reused.
    assert not Path(path).expanduser().is_symlink(), 'Output symlink forbidden'
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def select_rows(split, max_grids=None, grids=None, max_tiles=None):
    m, gm = manifests()
    splits = ('val', 'eval') if split == 'both' else (split,)
    m = m.loc[m.split.isin(splits) & m.included_default].copy()
    if grids:
        requested = set(grids.split(','))
        assert requested <= set(m.grid), f'Unauthorized or wrong-split grids: {requested - set(m.grid)}'
        m = m.loc[m.grid.isin(requested)]
    selected = []
    for sp in splits:
        ids = sorted(set(m.loc[m.split == sp, 'grid']))
        if max_grids is not None:
            assert max_grids > 0
            ids = ids[:max_grids]
        for grid in ids:
            rows = m.loc[(m.split == sp) & (m.grid == grid)].sort_values('image')
            if max_tiles is not None:
                assert max_tiles > 0
                # Explicit smoke-only restriction: alternate PV-rich / SWH-rich tiles.
                pv = rows.sort_values(['n_pv', 'n_swh', 'image'], ascending=[False, False, True]).index
                sw = rows.sort_values(['n_swh', 'n_pv', 'image'], ascending=[False, False, True]).index
                order = list(dict.fromkeys(i for pair in zip(pv, sw) for i in pair))
                rows = rows.loc[order[:max_tiles]]
            selected.append(rows)
    assert selected, 'No authorized rows selected'
    return pd.concat(selected).reset_index(drop=True), gm


def load_grid_gt(grid, gm):
    assert grid in ALLOWED_GRIDS
    rec = gm.loc[grid]
    path = rec.label_file
    if path:
        path = safe_input(path)
        assert digest(path, 'md5') == rec.md5, f'Label MD5 mismatch: {grid}'
        doc = {'gt_loader': 'ct_sam2_v2d1', 'gt_paths': [str(path)], 'area_crs': 'EPSG:32734',
               'swh_gpkg': str(safe_input(METADATA / 'collectors.gpkg')), 'drop_corrupt_geoms': True}
        pv = load_ct_sam2_grid(doc, grid, str(ROOT / 'readonly_adapter.json'))
        pv = pv.copy()
        pv['cls'] = 'PV'
    else:
        assert rec.label_source == 'hard_negative_no_labels'
        pv = empty_grid_gt('EPSG:32734')
    # Reuse the v2f collector table, including pool heaters and confidence upgrades.
    sw = gpd.read_file(safe_input(METADATA / 'collectors.gpkg'), where=f"grid = '{grid}' AND mask_value = 2")
    if len(sw):
        sw = sw.to_crs('EPSG:32734')
        sw = sw.loc[sw.geometry.notna() & ~sw.geometry.is_empty].copy()
        sw['geometry'] = sw.geometry.buffer(0)
    swh = grid_gt_from_rows([{'cls': 'SWH', 'geometry': g} for g in sw.geometry if not g.is_empty], 'EPSG:32734')
    return {'PV': pv, 'SWH': swh}


def ref_for(row):
    with rasterio.open(safe_input(row.image)) as src, rasterio.open(safe_input(row['mask'])) as gt:
        assert src.crs == gt.crs and src.transform == gt.transform
        assert (src.height, src.width) == (gt.height, gt.width) == (int(row.height), int(row.width))
        assert src.crs.to_epsg() == 32734 and src.count >= 3
        assert abs(src.transform.a - float(row.px_m)) < 1e-9
        assert abs(src.transform.e + float(row.px_m)) < 1e-9
        assert src.transform.b == src.transform.d == 0
        return TileRef(Path(row.image).stem, row.grid, src.transform, str(src.crs),
                       (src.height, src.width), float(row.px_m), str(row['mask']), str(row.image), row.split)


def class_labels(ref, gt):
    with rasterio.open(safe_input(ref.mask_path)) as src:
        mask = src.read(1)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 1, 2, 255}
    bounds = box(*rasterio.transform.array_bounds(*ref.shape, ref.transform))
    shapes = [(g, int(oid)) for g, oid in zip(gt.geometry, gt.oid) if g.intersects(bounds)]
    ids = rasterize(shapes, out_shape=ref.shape, transform=ref.transform, fill=0,
                    dtype='int32', all_touched=False) if shapes else np.zeros(ref.shape, np.int32)
    return TileLabels(ref, mask, ids)


def prediction_path(out, ref):
    return Path(out) / 'preds' / ref.split / ref.grid / f'{ref.tile_id}.tif'


def load_prediction(path, ref):
    # Prediction files may live in the explicitly named test_run_smoke output.
    with rasterio.open(path) as src:
        assert src.shape == ref.shape and src.transform == ref.transform and str(src.crs) == ref.crs
        arr = src.read(1)
    assert arr.dtype == np.uint8 and set(np.unique(arr)) <= {0, 1, 2}
    return arr


def write_json(path, obj):
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=2, allow_nan=False) + '\n')
