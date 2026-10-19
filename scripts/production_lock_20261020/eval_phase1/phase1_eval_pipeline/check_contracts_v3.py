#!/usr/bin/env python3
"""Research-critical v3 checks: exact sampled identities/order, cluster ratios and log selection."""
import importlib.util
from itertools import product
from pipeline_common_v3 import *
from validation_metrics_v3 import grid_bootstrap, reconcile_training_log


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    rows, gm, provenance = validation_scopes()
    source = safe_input(provenance['load_M_source'])
    assert digest(source) == provenance['load_M_sha256']
    spec = importlib.util.spec_from_file_location('independent_load_M', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    M, _ = module.load_M()  # Inspected CSV-only function; never load_bundle().
    M['valid'] = M.n_bg + M.n_pv + M.n_swh
    pool = M.loc[(M.split == 'val') & (M.valid > 100000) & ~M.image.isin(set(M.loc[M.split == 'eval','image']))]
    expected = pool.sample(300, random_state=1)
    actual = rows.loc[rows.scope_v3 == 'val_sel300'].sort_values('selection_rank')
    assert list(zip(expected.grid, expected.image.map(lambda p: Path(p).name))) == list(zip(actual.grid, actual.image.map(lambda p: Path(p).name)))
    assert expected.index.tolist() == actual.load_bundle_row.tolist()
    assert rows.scope_v3.value_counts().to_dict() == {'val_holdout':1403,'eval':632,'val_sel300':300}
    assert set(rows.loc[rows.scope_v3 == 'val_sel300','load_bundle_row']).isdisjoint(set(rows.loc[rows.scope_v3 == 'val_holdout','load_bundle_row']))
    naive, _ = manifests()
    naive = naive.loc[(naive.split == 'val') & naive.included_default].sample(300, random_state=1)
    naive_keys = set(zip(naive.grid, naive.image))
    overlap = len(naive_keys & set(zip(expected.grid, expected.image)))
    reversed_sample = pool.iloc[::-1].sample(300, random_state=1)
    reversed_overlap = len(set(zip(reversed_sample.grid,reversed_sample.image)) & set(zip(expected.grid,expected.image)))
    assert reversed_overlap < 300, 'Fixture must detect changing the training pool order'

    pix = {'A': {'tp':1,'fp':0,'fn':0}, 'B': {'tp':0,'fp':9,'fn':0}}
    obj = {'per_grid': {g: {'by_iou': {'0.5': c}} for g,c in pix.items()}, 'by_iou': {'0.5': {'tp':1,'fp':9,'fn':0}}}
    b = grid_bootstrap(pix,obj,reps=1000,seed=7)
    assert b['pixel_iou']['estimate'] == 0.1  # grid-rate mean would be 0.5
    draws = np.random.default_rng(7).integers(0,2,size=(1000,2))
    manual = []
    for draw in draws.tolist():
        c = [pix[['A','B'][i]] for i in draw]
        manual.append(sum(v['tp'] for v in c)/sum(v['tp']+v['fp']+v['fn'] for v in c))
    assert [b['pixel_iou']['low'],b['pixel_iou']['high']] == np.quantile(manual,[0.025,0.975]).tolist()
    assert b == grid_bootstrap(pix,obj,reps=1000,seed=7)
    zero = {g:dict.fromkeys(('tp','fp','fn'),0) for g in pix}
    empty_obj = {'per_grid':{g:{'by_iou':{'0.5':v}} for g,v in zero.items()},'by_iou':{'0.5':dict.fromkeys(('tp','fp','fn'),0)}}
    null = grid_bootstrap(zero,empty_obj,reps=1000,seed=7)
    assert null['object_f1_0.5']['estimate'] is None and null['pixel_iou']['low'] is None
    assert null['object_f1_0.5']['undefined_replicates'] == 1000
    mixed = {'A':zero['A'],'B':pix['B']}
    mixed_obj = {'per_grid':{g:{'by_iou':{'0.5':v}} for g,v in mixed.items()},'by_iou':{'0.5':pix['B']}}
    mx = grid_bootstrap(mixed,mixed_obj,reps=1000,seed=7)
    assert 0 < mx['pixel_iou']['undefined_replicates'] < 1000
    assert mx['pixel_iou']['low'] == mx['pixel_iou']['high'] == 0

    fixture = new_output(ROOT/'log_contract_fixture_v3')
    cfg = {'data_md5':MANIFEST_MD5,'script_md5':digest(TRAIN,'md5'),'arch':'mit_b2',
           'args':{'val_tiles':300,'max_val_tiles':None,'smoke':False,'allow_shared_tiles':False}}
    write_json(fixture/'config.json',cfg)
    vals = [{'step':0,'iou':0.99,'obj_f1':0.99}, {'step':2500,'iou':0.9,'obj_f1':0.5},
            {'step':5000,'iou':0.8,'obj_f1':0.7}, {'step':7500,'iou':0.85,'obj_f1':0.7}]
    with (fixture/'log.jsonl').open('x') as f:
        f.write(''.join(json.dumps({'val':v})+'\n' for v in vals))
    classes = {'PV':{'status':'scored','pixel':{'iou':0.8}}, 'SWH':{'status':'not_labelled'}}
    rec = reconcile_training_log(fixture/'best_objf1.pt',classes,{'precision':'bfloat16','batch_size':1})
    assert rec['best_step'] == 5000 and rec['status'] == 'within_tolerance'
    bad = reconcile_training_log(fixture/'last.pt',classes,{'precision':'bfloat16'})
    assert bad['status'] == 'unavailable'
    result = {'passed':True,'selection_matches_unmodified_load_M_including_sample_order':True,
              'counts':rows.scope_v3.value_counts().to_dict(),'raw_manifest_selection_overlap':overlap,
              'reversed_order_selection_overlap':reversed_overlap,
              'bootstrap_micro_ratio_and_manual_draws':True,'bootstrap_zero_denominators':True,
              'best_objf1_not_max_iou_first_strict_best':True,'cpu_only':True}
    with (ROOT/'contract_checks_v3.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
