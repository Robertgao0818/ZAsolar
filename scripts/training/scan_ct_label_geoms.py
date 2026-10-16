import sys,glob,os,numpy as np,pandas as pd,geopandas as gpd
sys.path.insert(0,'/home/gao/projects/ZAsolar_segv1_wt')
from core.grid_utils import get_grid_spec
from shapely.geometry import box
from shapely.validation import explain_validity
L='/home/gao/projects/ZAsolar/data/annotations/Capetown'
G1688='/home/gao/zasolar_data/geid_temporal/label_audit_classB_v1_20261014/diag/fixed_G1688_SAM2_260403.gpkg'
EX=['G1807','G1189','G1190','G1238']
rows=[]
def chk(src,g,d,kind):
    try: spec=get_grid_spec('CPT'+g[1:],region='ct')
    except Exception as e: print('nospec',g); return
    gb=gpd.GeoSeries([box(spec.xmin,spec.ymin,spec.xmax,spec.ymax)],crs=4326).to_crs(32734).buffer(50).iloc[0]
    dm=d.to_crs(32734)
    for i,(geo,gm) in enumerate(zip(d.geometry,dm.geometry)):
        if geo is None or geo.is_empty: rows.append(dict(src=src,kind=kind,grid=g,idx=i,issue='empty'));continue
        iss=[]
        if not gb.buffer(1e-6).contains(gm): 
            out=gm.difference(gb); iss.append(f'outside_grid50:{out.area:.2f}m2,ext={gm.bounds[2]-gm.bounds[0]:.0f}x{gm.bounds[3]-gm.bounds[1]:.0f}m')
        if not geo.is_valid: iss.append('invalid:'+explain_validity(geo)[:60])
        if geo.geom_type not in('Polygon','MultiPolygon'): iss.append('type:'+geo.geom_type)
        if gm.area<0.05: iss.append(f'tiny:{gm.area:.3f}')
        if iss: rows.append(dict(src=src,kind=kind,grid=g,idx=i,issue='; '.join(iss),area=gm.area))
for f in sorted(glob.glob(L+'/G*_SAM2_*.gpkg')):
    g=os.path.basename(f).split('_')[0]
    if g in EX: continue
    src=G1688 if g=='G1688' else f
    d=gpd.read_file(src); d=d.set_crs(4326) if d.crs is None else d.to_crs(4326)
    chk(src,g,d,'pv_label')
swh=gpd.read_file('/home/gao/zasolar_data/geid_temporal/water_heater_pass_v1_20261015/solar_water_heaters.gpkg').to_crs(4326)
for g,d in swh.groupby('grid'): chk('swh',g,d.reset_index(drop=True),'swh')
hn=pd.read_csv('/home/gao/zasolar_data/geid_temporal/cls_hardneg_inventory_v1_20261014/ct_reviewed_preds.csv'); hn=hn[hn.status=='delete']
for (s,g),h in hn.groupby(['src','grid']):
    d=gpd.read_file(s).to_crs(4326).iloc[h.idx.astype(int).values].reset_index(drop=True); chk(s,g,d,'hn')
r=pd.DataFrame(rows); r.to_csv('/home/gao/capetown_eval/labelfix/scan_all_sources.csv',index=False)
print(len(r)); print(r.groupby(['kind']).size()); print(r[~r.issue.str.startswith('tiny')].to_string())
