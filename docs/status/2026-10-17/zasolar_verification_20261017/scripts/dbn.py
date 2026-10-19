import geopandas as gpd, pandas as pd, glob, os
H=os.path.expanduser('~/zasolar_data/')
O=H+'durban_ruleD_city_20261012/out/'
for n in ['before','after']:
    pass
fs={'v3':glob.glob(H+'durban_stitch_sam3_20261009/holefill_v3/*.gpkg'),
 'before':[O+'durban_census_inventory_stitch025_holefill_v3_before_20261012.gpkg'],
 'after':[O+'durban_census_inventory_stitch025_holefill_v3_ruleD_after_20261012.gpkg'],
 'v4':[O+'durban_census_inventory_stitch025_holefill_v4_candidate_drop030281.gpkg']}
G={}
for k,l in fs.items():
  for f in l:
    g=gpd.read_file(f); G[k]=g
    print(k,os.path.basename(f),len(g),g.crs.to_epsg(),'utm36S',round(g.to_crs(32736).area.sum(),1),'eqarea',round(g.to_crs('ESRI:54034').area.sum(),1),[c for c in g.columns if 'area' in c.lower() or 'gem' in c.lower() or 'id' in c.lower()][:8])
b,a=G['before'],G['after']
idc='block_id' if 'block_id' in b.columns else None
if idc:
  rm=set(b[idc])-set(a[idc]); print('removed',len(rm)); print('v4 drop',set(a[idc])-set(G['v4'][idc]))
v=pd.read_csv(H+'durban_hole_review_20261012/review_verdicts_20261012.csv')
print(v.verdict.value_counts(), v.groupby('verdict').hole_area_m2.sum())
for d in ['hn_v1','hn_v1_r005']:
  p=H+'durban_overnight_20261016/'+d
  i=pd.read_csv(p+'/index.csv'); print(d,len(i),i.columns.tolist()[:10],len(os.listdir(p+'/chips')),len(os.listdir(p+'/masks')))
  for c in i.columns:
    if i[c].nunique()<6: print(' ',c,i[c].value_counts().to_dict())
