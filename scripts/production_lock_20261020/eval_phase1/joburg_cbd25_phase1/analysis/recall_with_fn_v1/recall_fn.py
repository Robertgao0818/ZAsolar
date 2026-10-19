#!/usr/bin/env python3
"""CPU: JHB CBD25 recall of phase-1 models vs 葉-reviewed reference (kept MRCNN correct+edit + SAM-added FN polygons; deletes excluded)."""
import glob, json, numpy as np, pandas as pd, geopandas as gpd
R = '/home/gao/projects/ZAsolar/results/johannesburg/v3c_vexcel_2024_ch1_sample'
P = '/home/gao/capetown_eval/joburg_cbd25_phase1_20261018/compare_web/polygons'
d = pd.concat([gpd.read_file(f).assign(grid=f.split('/')[-1][:5]) for f in sorted(glob.glob(f'{R}/G*/review/G*_reviewed.gpkg'))], ignore_index=True)
d = gpd.GeoDataFrame(d, crs=4326)
d['group'] = np.where(d.source == 'sam_fn_review', 'human_FN', 'MRCNN_kept')
ref = d[d.review_status.isin(['correct', 'edit'])].to_crs(32735).reset_index(drop=True)
ref['geometry'] = ref.geometry.make_valid()
ref['area'] = ref.area
ref['bin'] = pd.cut(ref.area, [0, 4, 10, np.inf], right=False, labels=['lt4', '4_10', 'ge10'])
counts = {'reviewed_total': len(d), 'by_status_source': d.groupby([d.review_status.fillna('NA'), d.group]).size().to_dict(),
          'reference': len(ref), 'ref_MRCNN_kept': int((ref.group == 'MRCNN_kept').sum()), 'ref_MRCNN_correct': int(((ref.group == 'MRCNN_kept') & (ref.review_status == 'correct')).sum()),
          'ref_MRCNN_edit': int((ref.review_status == 'edit').sum()), 'ref_human_FN': int((ref.group == 'human_FN').sum()), 'ref_4_10': int((ref.bin == '4_10').sum())}
fnpts = pd.concat([gpd.read_file(f) for f in glob.glob(f'{R}/G*/review/G*_fn_markers.gpkg')]).to_crs(32735)
res = []; per = []
for f in sorted(glob.glob(f'{P}/*.gpkg')):
    m = f.split('/')[-1][:-5]
    pr = gpd.read_file(f).to_crs(32735); pr['geometry'] = pr.geometry.make_valid(); pr = pr[pr.area > 0].reset_index(drop=True)
    j = gpd.sjoin(ref[['geometry']].reset_index(names='ri'), pr[['geometry']].reset_index(names='pi'), predicate='intersects')
    hit_lo = np.zeros(len(ref), bool); best = np.zeros(len(ref))
    for r in j.itertuples():
        a = ref.geometry.iat[r.ri]; b = pr.geometry.iat[r.pi]
        inter = a.intersection(b).area; iou = inter / a.union(b).area
        if iou >= 0.1 or inter >= 0.5 * min(a.area, b.area): hit_lo[r.ri] = True
        best[r.ri] = max(best[r.ri], iou)
    ref[f'hit_{m}'] = hit_lo; ref[f'iou_{m}'] = best
    fn_pt_hit = int(sum(pr.contains(p).any() for p in fnpts.geometry))
    row = {'model': m, 'n_pred': len(pr)}
    for name, msk in [('overall', slice(None)), ('human_FN', ref.group == 'human_FN'), ('MRCNN_kept', ref.group == 'MRCNN_kept'), ('4_10m2', ref.bin == '4_10')]:
        s = ref.loc[msk]
        row[f'R_{name}'] = round(s[f'hit_{m}'].mean(), 4); row[f'R50_{name}'] = round((s[f'iou_{m}'] >= 0.5).mean(), 4); row[f'n_{name}'] = len(s)
    row['fn_marker_points_hit_of_5'] = fn_pt_hit
    res.append(row)
T = pd.DataFrame(res); T.to_csv('recall_by_model.csv', index=False)
ref.drop(columns='geometry').to_csv('reference_objects_with_hits.csv', index=False)
ref[['grid', 'group', 'review_status', 'area', 'bin', 'geometry'] + [c for c in ref.columns if c.startswith(('hit_', 'iou_'))]].to_file('reference_with_hits.gpkg', driver='GPKG')
pg = ref.groupby('grid')[[c for c in ref.columns if c.startswith('hit_')]].mean().round(3); pg['n'] = ref.groupby('grid').size(); pg.to_csv('recall_by_grid.csv')
json.dump({k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in counts.items()}, open('reference_counts.json', 'w'), indent=1)
print(T.to_string())
