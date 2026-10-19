#!/usr/bin/env python3
"""Render saved phase-1 eval predictions; never run inference or open test data."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

from pipeline_common import *
from PIL import Image, ImageDraw
from scipy import ndimage

COLOURS = {'PV': (0, 220, 255), 'SWH': (255, 155, 0)}


def render(entry, row, ref, gt, dest, max_side):
    pred = load_prediction(entry['pred'], ref)
    with rasterio.open(safe_input(entry['image'])) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
    with rasterio.open(safe_input(entry['gt'])) as src:
        mask = src.read(1)
    # Dim ignored regions so predictions there can be distinguished visually.
    rgb[mask == 255] *= 0.45
    edges = []
    for name, value in CLASSES.items():
        colour = np.array(COLOURS[name])
        hit = pred == value
        rgb[hit] = rgb[hit] * 0.58 + colour * 0.42
        raw = class_labels(ref, gt[name]).gt_id > 0
        edge = raw & ~ndimage.binary_erosion(raw, structure=np.ones((3, 3), bool))
        edges.append((ndimage.binary_dilation(edge, iterations=2), colour))
    for edge, colour in edges:
        rgb[edge] = colour
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (max(image.width, 560), image.height + 62), (25, 28, 32))
    canvas.paste(image, (0, 62))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"{entry['grid']} / {ref.tile_id} | ignore regions dimmed", fill='white')
    draw.text((8, 23), 'PV: cyan | SWH: orange | GT: solid outline | prediction: translucent fill', fill='white')
    draw.text((8, 41), f"SWH ground truth: {'labelled' if row.swh_labelled else 'UNLABELLED (not scored)'}", fill='white')
    canvas.save(dest)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', required=True, help='Output directory from eval_phase1.py')
    p.add_argument('--out', required=True, help='New output directory; existing directories refused')
    p.add_argument('--max-side', type=int, default=1200, help='0 preserves native tile resolution')
    p.add_argument('--allow-partial', action='store_true', help='Explicitly render an incomplete smoke subset')
    a = p.parse_args()
    assert a.max_side >= 0
    started = time.perf_counter()
    base = Path(a.eval_dir).expanduser().resolve()
    assert base.is_relative_to(ROOT) or 'test' not in str(base).lower(), 'Test input forbidden'
    metrics = json.loads((base / 'metrics.json').read_text())
    assert metrics['scorer']['commit'] == scorer_provenance()['commit']
    entries = [e for e in json.loads((base / 'predictions.json').read_text()) if e['split'] == 'eval']
    assert entries, 'No authorized eval predictions'
    ids = set(e['grid'] for e in entries)
    assert ids <= EVAL_GRIDS, 'Non-eval grid forbidden'
    if not a.allow_partial:
        assert not metrics['partial'] and ids == EVAL_GRIDS, 'Full 5-grid run required; smoke needs --allow-partial'
    rows, gm = select_rows('eval')
    lookup = {(r.grid, r.image): r for _, r in rows.iterrows()}
    assert len({(e['grid'], e['image']) for e in entries}) == len(entries), 'Duplicate predictions'
    if not a.allow_partial:
        assert {(e['grid'], e['image']) for e in entries} == set(lookup), 'Missing eval tile predictions'
    out = new_output(a.out)
    thumbs = defaultdict(list)
    tables = {}
    index = []
    for e in entries:
        row = lookup[(e['grid'], e['image'])]
        ref = ref_for(row)
        assert safe_input(e['gt']) == safe_input(row['mask'])
        assert Path(e['pred']).resolve() == prediction_path(base, ref).resolve()
        assert Path(e['pred']).resolve().is_relative_to(base), 'Prediction escaped output root'
        if ref.grid not in tables:
            tables[ref.grid] = load_grid_gt(ref.grid, gm)
        dest = out / ref.grid / f'{ref.tile_id}.png'
        dest.parent.mkdir(exist_ok=True)
        canvas = render(e, row, ref, tables[ref.grid], dest, a.max_side)
        thumb = canvas.copy()
        thumb.thumbnail((280, 300))
        thumbs[ref.grid].append(thumb)
        index.append({'grid': ref.grid, 'tile': ref.tile_id, 'png': str(dest)})
    for grid, items in thumbs.items():
        cols = min(8, len(items))
        sheet = Image.new('RGB', (cols * 280, ((len(items) + cols - 1) // cols) * 300), (25, 28, 32))
        for i, thumb in enumerate(items):
            sheet.paste(thumb, ((i % cols) * 280, (i // cols) * 300))
        sheet.save(out / f'{grid}_contact_sheet.png')
    write_json(out / 'render_manifest.json', {'partial': a.allow_partial or metrics['partial'], 'source': str(base),
               'colours': COLOURS, 'runtime_s': time.perf_counter() - started, 'tiles': index})
    print(json.dumps({'event': 'render_complete', 'grids': sorted(ids), 'png_tiles': len(index), 'out': str(out)}))


if __name__ == '__main__':
    main()
