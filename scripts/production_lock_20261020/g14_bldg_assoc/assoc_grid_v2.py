"""Stage 2 (one grid per worker): bbox-read buildings (+erf for CT) for this grid's window only, associate GT (row variants:
original, merged at 0.25 m, merged at 1 m gap) and each model's A3 polygons. Eval buildings = representative point inside the
grid's region (gt/CITY_regions.gpkg). Buildings: CT A_basemap_v1/cpt_buildings_merged_v1.gpkg + coct_parcels_v1.gpkg;
JHB google_v3+osm+microsoft union_dissolved (as trial_jhb_v1/per_building_v1.py). Old 2026-09-02 backdating layer NOT used.
usage: assoc_grid_v1.py CITY GRID -> parts/CITY_GRID.pkl  [v2: v1 used column name 'eval', which collides with DataFrame.eval -> ValueError on every grid]"""
import os,sys,glob,pickle,numpy as np,pandas as pd,geopandas as gpd,shapely
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from g14_lib_v1 import associate, merge_gap
CITY,G=sys.argv[1:3]; out=f'parts/{CITY}_{G}.pkl'; assert not os.path.exists(out),'exists'
MODELS={'CT':['convnext','b2','avg2'],'JHB':['convnext','b2','v2c','avg2']}[CITY]; GAPS=[0.0,0.25,1.0]
R=gpd.read_file(f'gt/{CITY}_regions.gpkg').set_index('grid'); CRS=R.crs; reg=R.geometry[G]
W=gpd.GeoSeries([shapely.box(*reg.bounds).buffer(100,join_style='mitre')],crs=CRS)
A='/home/gao/projects/zasolar_plans_20261017/building_mask_20261018/A_basemap_v1'; BP='/home/gao/RA_solar/census/data/buildings_public_20261018'
if CITY=='CT':
    bld=gpd.read_file(f'{A}/cpt_buildings_merged_v1.gpkg',bbox=W)[['bld_id','h3_r7','stratum','parcel_key','geometry']].to_crs(CRS)
    par=gpd.read_file(f'{A}/coct_parcels_v1.gpkg',bbox=W)[['ZONING','geometry']].to_crs(CRS)
else:
    src=[gpd.read_file(f,bbox=W).to_crs(CRS)[['geometry']] for f in [f'{BP}/google_v3/jhb_google_v3.gpkg',f'{BP}/osm/osm_buildings_jhb.gpkg']+sorted(glob.glob(f'{BP}/microsoft/jhb_*.gpkg'))]
    u=gpd.GeoDataFrame(geometry=list(pd.concat(src).geometry),crs=CRS); u['geometry']=shapely.make_valid(u.geometry.values)
    bld=u.dissolve().explode(index_parts=False).reset_index(drop=True)[['geometry']]; bld=bld[bld.geom_type.isin(['Polygon','MultiPolygon'])]
    import h3
    rp=bld.representative_point().to_crs(4326); bld['h3_r7']=[h3.latlng_to_cell(p.y,p.x,7) for p in rp]; bld['stratum']=''; bld['parcel_key']=None; bld['bld_id']=[f'{G}_{i}' for i in range(len(bld))]; par=None
bld=bld[bld.geometry.notna()&~bld.geometry.is_empty].reset_index(drop=True); bld['bid']=np.arange(len(bld))
bld['is_eval']=bld.representative_point().within(reg)
gt=gpd.read_file(f'gt/{CITY}_gt.gpkg',bbox=W); own=gt[gt.grid==G]; oth=gt[gt.grid!=G].copy()
if len(oth): oth['geometry']=shapely.difference(shapely.make_valid(oth.geometry.values),reg); oth=oth[~oth.is_empty]
gt=pd.concat([own,oth],ignore_index=True); gt=gpd.GeoDataFrame(gt,geometry='geometry',crs=CRS)
srcs={f'gt{g}':merge_gap(gt,g) for g in GAPS}
for m in MODELS: srcs[m]=gpd.read_file(f'preds/{CITY}_{m}.gpkg',bbox=W)[['geometry']]
B=bld.loc[bld.is_eval,['bld_id','h3_r7','stratum','parcel_key']].copy(); B['grid']=G; S={}
for k,pv in srcs.items():
    al,pvf=associate(pv,bld,par)
    B[k]=al.reindex(bld.bid[bld.is_eval].values,fill_value=0).values
    inR=pvf.representative_point().within(reg).values if len(pvf) else np.zeros(0,bool); f=pvf[inR]
    S[k]=dict(n_polys_in_region=int(inR.sum()),pv_area_in_region=float(f.pv_area.sum()),alloc_to_eval_bld=float(B[k].sum()),
              orphan_area=float(f.orphan_area.sum()),orphan_by_class=f[f.orphan_area>0].groupby('rule_class').orphan_area.sum().to_dict())
pickle.dump(dict(city=CITY,grid=G,n_bld_window=len(bld),n_bld_eval=int(bld.is_eval.sum()),buildings=B,summary=S),open(out+'.tmp','wb')); os.replace(out+'.tmp',out)
print('DONE',CITY,G,'bld_eval',int(bld.is_eval.sum()),{k:int((B[k]>=2).sum()) for k in srcs},flush=True)
