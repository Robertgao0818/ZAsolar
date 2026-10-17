# CPU, read-only: for each image used by >1 split in v2d2, classify overlapping non-ignore pixels per mask pair.
import csv, collections, itertools, json, numpy as np, rasterio
M = '/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017/data/manifest.csv'
by = collections.defaultdict(list)
for r in csv.DictReader(open(M)): by[r['image']].append(r)
tot = collections.Counter(); pairs = []
for img, rs in by.items():
    if len({r['split'] for r in rs}) < 2: continue
    ms = {r['mask']: rasterio.open(r['mask']).read(1) for r in rs}
    for a, b in itertools.combinations(rs, 2):
        if a['split'] == b['split']: continue
        x, y = ms[a['mask']], ms[b['mask']]
        both = (x != 255) & (y != 255)
        pva, pvb = both & (x == 1), both & (y == 1)
        c = dict(both_valid=int(both.sum()), bg_bg=int((both & (x == 0) & (y == 0)).sum()),
                 pv_any=int((pva | pvb).sum()), pv_both=int((pva & pvb).sum()), pv_disagree=int((pva ^ pvb).sum()))
        tot['pairs'] += 1; tot['pairs_nonignore'] += c['both_valid'] > 0; tot['pairs_with_pv'] += c['pv_any'] > 0
        for k, v in c.items(): tot[k] += v
        pairs.append(dict(image=img.split('/')[-1], a=f"{a['grid']}:{a['split']}", b=f"{b['grid']}:{b['split']}", **c))
tot['shared_images'] = len({p['image'] for p in pairs})
sp = collections.Counter(tuple(sorted((p['a'].split(':')[1], p['b'].split(':')[1]))) for p in pairs)
out = dict(totals=dict(tot), split_pairs={'-'.join(k): v for k, v in sp.items()}, pairs=pairs)
json.dump(out, open('split_overlap_check_20261017.json', 'w'), indent=1)
print(json.dumps(out['totals'], indent=1)); print(out['split_pairs'])
