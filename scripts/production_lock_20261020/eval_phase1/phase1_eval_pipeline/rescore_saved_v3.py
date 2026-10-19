#!/usr/bin/env python3
"""Re-score complete saved v2/v3 masks, with no model construction or inference."""
import argparse
import time
from eval_phase1_v3 import score_split
from pipeline_common_v3 import *
from validation_metrics_v3 import score_scopes, reconcile_training_log, markdown_v3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-root')
    p.add_argument('--training-log')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    start = time.perf_counter()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '', 'Saved-mask scoring is CPU only'
    torch.set_num_threads(a.threads)
    source = Path(a.eval_dir).expanduser().resolve()
    original = json.loads((source / 'metrics.json').read_text())
    assert original['partial'] is False, 'Full saved validation and eval masks required'
    assert original['manifest_md5'] == MANIFEST_MD5
    assert original['training_sha256'] == digest(TRAIN)
    scorer = scorer_provenance()
    assert original['scorer']['commit'] == scorer['commit']
    entries = json.loads((source / 'predictions.json').read_text())
    rows, gm, provenance = validation_scopes()
    mapping = {(r.grid, Path(r.image).name): r for r in rows.itertuples()}
    assert len(mapping) == 2335 and len(entries) == len(mapping)
    seen = set()
    manifest = []
    for entry in entries:
        key = (entry['grid'], Path(entry['image']).name)
        assert key in mapping and key not in seen
        seen.add(key)
        row = mapping[key]
        assert entry['split'] == row.split and entry['swh_labelled'] == bool(row.swh_labelled)
        assert Path(entry['gt']).resolve() == Path(row.mask).resolve(), 'GT relocation requires fresh saved manifest'
        ref = ref_for(rows.loc[rows.load_bundle_row == row.load_bundle_row].iloc[0])
        expected = prediction_path(source, ref).resolve()
        assert Path(entry['pred']).resolve() == expected and expected.is_relative_to(source)
        assert expected.is_file(), f'Missing saved prediction: {expected}; stop before any inference'
        manifest.append({**entry, 'scope_v3': row.scope_v3, 'load_bundle_row': int(row.load_bundle_row),
                         'selection_rank': int(row.selection_rank) if pd.notna(row.selection_rank) else None})
    assert seen == set(mapping)
    out = new_output(a.out)
    # These are explicitly authorized evaluation-output JSONs, not dataset inputs.
    # The legacy input guard rejects the historical output directory's literal "test".
    saved_metadata_sha = lambda name: hashlib.sha256((source / name).read_bytes()).hexdigest()
    write_json(out / 'source_v3.json', {'eval_dir': str(source), 'metrics_sha256': saved_metadata_sha('metrics.json'),
               'predictions_sha256': saved_metadata_sha('predictions.json'), 'n_saved_masks': len(entries),
               'inference_executed': False, 'device': 'cpu'})
    write_json(out / 'predictions.json', manifest)
    rows.to_csv(out / 'scope_rows_v3.csv', index=False, mode='x')
    # No symlinks and no prediction copies: pinned scoring reads the original source tree.
    scores = score_scopes(source, rows, gm, score_split)
    doc = {k: original[k] for k in ('arch','checkpoint','checkpoint_sha256','device','precision','inference','scorer',
                                  'training_source','training_sha256','manifest_md5','partial')}
    checkpoint = safe_input(doc['checkpoint'])
    assert digest(checkpoint) == doc['checkpoint_sha256'], 'Checkpoint changed since saved inference'
    overlap = rows.loc[rows.scope_v3 == 'val_sel300', 'image'].isin(set(rows.loc[rows.scope_v3 == 'val_holdout', 'image']))
    doc.update(splits=scores, selection_provenance_v3=provenance, primary_validation='val_holdout',
               rescore_source=str(source), inference_executed=False, rescore_device='cpu',
               val_scope='300 selection rows and 1403 holdout rows; 632 eval rows unchanged',
               shared_image_selection_rows_in_holdout=int(overlap.sum()),
               training_log_reconciliation=reconcile_training_log(checkpoint, scores['val_sel300'],
                   {'precision': original['precision'], 'batch_size': original['selection']['batch_size']}, a.training_log),
               runtime_s=time.perf_counter() - start)
    write_json(out / 'metrics.json', doc)
    with (out / 'metrics.md').open('x') as f:
        f.write(markdown_v3(doc))
    print(json.dumps({'event': 'complete', 'runtime_s': doc['runtime_s'], 'out': str(out)}), flush=True)


if __name__ == '__main__':
    main()
