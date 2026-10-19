"""Grid-cluster uncertainty and honest training-log reconciliation, no inference."""
from pipeline_common_v3 import *

BOOTSTRAP_SEED = 20261018
BOOTSTRAP_REPS = 20000
IOU_TOLERANCE = 0.001


def grid_bootstrap(pixel_by_grid, objects, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Resample complete labelled grids; recompute micro ratios of summed counts."""
    grids = sorted(pixel_by_grid)
    assert grids and reps > 0
    assert set(grids) == set(objects['per_grid'])
    pixels = np.array([[pixel_by_grid[g][k] for k in ('tp', 'fp', 'fn')] for g in grids], dtype=np.int64)
    obj = np.array([[objects['per_grid'][g]['by_iou']['0.5'][k] for k in ('tp', 'fp', 'fn')]
                    for g in grids], dtype=np.int64)
    assert np.all(pixels >= 0) and np.all(obj >= 0)
    assert obj.sum(axis=0).tolist() == [objects['by_iou']['0.5'][k] for k in ('tp', 'fp', 'fn')]
    draws = np.random.default_rng(seed).integers(0, len(grids), size=(reps, len(grids)))
    result = {'unit': 'grid', 'method': 'percentile, micro ratios of resampled integer totals',
              'confidence': 0.95, 'reps': reps, 'seed': seed, 'grids': grids, 'n_grids': len(grids),
              'zero_denominator_policy': 'undefined replicates omitted; counts reported; all undefined => null',
              'conditional_on': 'fixed checkpoint, observed labelled grid population and saved predictions'}
    for name, counts, multiplier in [('pixel_iou', pixels, 1), ('object_f1_0.5', obj, 2)]:
        total = counts.sum(axis=0)
        point_den = int(multiplier * total[0] + total[1] + total[2])
        summed = counts[draws].sum(axis=1)
        den = multiplier * summed[:, 0] + summed[:, 1] + summed[:, 2]
        defined = den > 0
        values = multiplier * summed[defined, 0] / den[defined]
        limits = np.quantile(values, [0.025, 0.975], method='linear').tolist() if len(values) else [None, None]
        result[name] = {'estimate': float(multiplier * total[0] / point_den) if point_den else None,
                        'low': limits[0], 'high': limits[1], 'defined_replicates': int(defined.sum()),
                        'undefined_replicates': int((~defined).sum()), 'total_counts_tp_fp_fn': total.tolist()}
    return result


def reconcile_training_log(checkpoint, classes, inference, log_path=None):
    """Find the first strict best at a saved validation step, never choose by rescore."""
    checkpoint = Path(checkpoint)
    path = Path(log_path) if log_path else checkpoint.parent / 'log.jsonl'
    answer = {'log_path': str(path), 'absolute_iou_tolerance': IOU_TOLERANCE,
              'tolerance_note': 'Operational diagnostic tolerance, not a theorem about bf16 error.',
              'status': 'unavailable', 'best_step': None}
    if not path.is_file():
        answer['reason'] = 'No checkpoint-adjacent log.jsonl available.'
        return answer
    path = safe_input(path)
    answer['log_sha256'] = digest(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    vals = [r['val'] for r in records if 'val' in r and r['val'].get('step', 0) > 0]
    if not vals:
        answer['reason'] = 'No post-training validation/save step in log.'
        return answer
    if checkpoint.name == 'best_objf1.pt':
        key = 'obj_f1'
    elif checkpoint.name == 'best.pt' and not any('obj_f1' in v for v in vals):
        key = 'iou'
        answer['legacy_checkpoint'] = True
    else:
        answer['reason'] = 'Cannot identify this checkpoint with a best-selection rule; last/resume not guessed.'
        return answer
    if not all(key in v for v in vals):
        answer['reason'] = f'Incomplete log for selection key {key}.'
        return answer
    best = max(vals, key=lambda v: v[key])  # stable first max matches strict > save rule
    answer.update(best_step=best['step'], selection_key=key, selection_score=best[key], logged_validation=best)
    for name, field in [('PV', 'iou'), ('SWH', 'swh_iou')]:
        if field in best and classes[name]['status'] == 'scored':
            actual = classes[name]['pixel']['iou']
            delta = abs(actual - best[field]) if actual is not None else None
            answer[name] = {'saved_mask_iou': actual, 'logged_iou': best[field], 'absolute_difference': delta,
                            'within_diagnostic_tolerance': delta <= IOU_TOLERANCE if delta is not None else None}
    config_path = checkpoint.parent / 'config.json'
    config = json.loads(safe_input(config_path).read_text()) if config_path.is_file() else {}
    cfg = config.get('args', {})
    issues = []
    if answer.get('legacy_checkpoint'):
        issues.append('Legacy best.pt was selected by pixel IoU on v2c labels/pool, not v2f best_objf1.')
    if config.get('data_md5') != MANIFEST_MD5:
        issues.append('No matching v2f data_md5 in checkpoint-adjacent config.')
    if config.get('script_md5') != digest(TRAIN, 'md5'):
        issues.append('No matching training script_md5 in checkpoint-adjacent config.')
    if cfg.get('allow_shared_tiles', False) or cfg.get('smoke', False):
        issues.append('Training selection used shared tiles or smoke selection.')
    if cfg.get('val_tiles') != 300 or cfg.get('max_val_tiles') is not None and cfg['max_val_tiles'] < 300:
        issues.append('Cannot establish a full 300-tile training selection from config.')
    if inference.get('precision') != 'bfloat16':
        issues.append('Saved inference precision differs from CUDA bf16 training validation.')
    if config.get('arch', cfg.get('arch')) == 'segnext_b' and inference.get('batch_size') != 4:
        issues.append('SegNeXt stochastic NMF uses a different validation batch size; numerical equivalence unverified.')
    answer['comparability_issues'] = issues
    answer['status'] = ('not_comparable' if issues else
                        'within_tolerance' if answer.get('PV', {}).get('within_diagnostic_tolerance') else 'mismatch')
    answer['selection_verified_from_log_only'] = True
    answer['checkpoint_step_binding'] = 'inferred from best filename + logged save rule, not embedded checkpoint metadata'
    return answer


def score_scopes(pred_root, rows, gm, score_split):
    result = {}
    for scope in ('val_sel300', 'val_holdout', 'eval'):
        group = rows.loc[rows.scope_v3 == scope]
        if group.empty:
            continue
        print(json.dumps({'event': 'score_scope', 'scope': scope, 'n_rows': len(group)}), flush=True)
        result[scope] = score_split(pred_root, group, gm)
        for cls in result[scope].values():
            if cls['status'] != 'scored':
                continue
            cls['bootstrap_95ci'] = grid_bootstrap(cls['pixel_by_grid'], cls['objects'])
            assert [sum(v[k] for v in cls['pixel_by_grid'].values()) for k in ('tp','fp','fn','n_valid')] == [cls['pixel'][k] for k in ('tp','fp','fn','n_valid')]
    return result


def markdown_v3(doc):
    def fmt(value):
        return 'N/A' if value is None else f'{value:.8f}'
    def cell(scope, name, metric):
        c = doc['splits'].get(scope, {}).get(name)
        if not c or c['status'] != 'scored':
            return 'N/A (unlabelled / outside scope)'
        if metric == 'tiles':
            return str(c['n_tiles'])
        if metric in ('pixel_iou', 'object_f1_0.5'):
            b = c['bootstrap_95ci'][metric]
            return f"{fmt(b['estimate'])} [{fmt(b['low'])}, {fmt(b['high'])}]"
        return ' / '.join(fmt(c['objects']['by_iou'][metric][k]) for k in ('precision','recall','f1'))
    lines = ['# Phase-1 v3: validation selection and holdout', '',
             'Main validation result: **val_holdout**. val_sel300 is the training checkpoint-selection diagnostic.',
             'Intervals: 95% grid-cluster percentile bootstrap, 20,000 draws, seed 20261018;',
             'ratios are recomputed from summed TP/FP/FN, not averaged grid rates. Undefined draws are counted in JSON.', '',
             '| Class / metric | val_sel300 | val_holdout (main) |', '|---|---:|---:|']
    for name in CLASSES:
        for metric in ('tiles','pixel_iou','object_f1_0.5','0.3','0.5','0.7'):
            lines.append(f'| {name} {metric} | {cell("val_sel300", name, metric)} | {cell("val_holdout", name, metric)} |')
    if 'eval' in doc['splits']:
        lines += ['', '| Class / metric | eval |', '|---|---:|']
        for name in CLASSES:
            for metric in ('tiles','pixel_iou','object_f1_0.5','0.3','0.5','0.7'):
                lines.append(f'| {name} {metric} | {cell("eval", name, metric)} |')
    lines += ['', 'Object rows 0.3/0.5/0.7 show P / R / F1 under pinned zaeval one-to-one IoU matching.',
              'SWH uses labelled grids only; unlabelled grids are excluded, labelled empty grids retained.',
              'Objects are re-merged/matched within each scope; subtracting full-pool object counts would be wrong.',
              'The split is by grid/image row; shared imagery across val grids may remain in both scopes.',
              'CIs condition on this checkpoint; with five eval grids they are approximate and do not measure training-seed variability.',
              'Future architecture comparisons should use paired grid draws; CI overlap alone is not a significance test.', '',
              'Training log reconciliation:', '```json', json.dumps(doc.get('training_log_reconciliation'), indent=2), '```', '']
    return '\n'.join(lines)
