"""Stage 1: saved probs -> frozen h0.9_0.5 + A3 merge (xtile_merge.merge_plan over ALL tiles of the set, imported unmodified from
~/capetown_eval/ct_xtile_merge_v1_20261019, md5 d69fea35) -> one polygon per merged object -> preds/CITY_MODEL.gpkg. CPU only.
CT: 1703 val rows (pipeline_common_v3.validation_scopes), unique tiles, probs phase1_followup_valprobs_20261018 (no pregate).
JHB: 2500 tiles005, probs jhb_cbd25_zaeval2_v1_20261019/probs (official 0.45 cell pregate already applied).
avg2 = (convnext_t_s0 + b2)/2 on probs (CT b2 = mit_b2_s2, JHB b2 = mit_b2_s0), as ct_val_group_diag.  usage: prep_preds_v1.py CITY MODEL"""
import os,sys,glob,time,numpy as np,pandas as pd,geopandas as gpd,rasterio
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
X=Path('~/capetown_eval/ct_xtile_merge_v1_20261019').expanduser(); sys.path.insert(0,str(X))
import hashlib; assert hashlib.md5((X/'xtile_merge.py').read_bytes()).hexdigest()=='d69fea350e588b468b36f55bce299a0c'
from xtile_merge import TileIn, merge_plan, plan_tile_ids, plan_objects, global_origin, polygons
CITY,MODEL=sys.argv[1:3]; CFG='h0.9_0.5'; out=f'preds/{CITY}_{MODEL}.gpkg'; assert not os.path.exists(out),'exists'
t0=time.time()
if CITY=='CT':
    sys.path.insert(0,str(Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser()))
    from pipeline_common_v3 import select_rows, validation_scopes, ref_for
    rows,gm=select_rows('val'); rows,gm,_=validation_scopes(rows); assert len(rows)==1703
    PR=Path('~/capetown_eval/ds3m_ft/phase1_followup_valprobs_20261018').expanduser(); B2='mit_b2_s2'; CRS='EPSG:32734'
    tl={}
    for r in rows.to_dict('records'):
        ref=ref_for(pd.Series(r))
        if ref.tile_id not in tl: tl[ref.tile_id]=(r['grid'],ref.transform,ref.gsd_m)
else:
    J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); PR=Path('~/capetown_eval/jhb_cbd25_zaeval2_v1_20261019/probs').expanduser(); B2='mit_b2_s0'; CRS='EPSG:32735'
    tl={}
    for p in sorted(glob.glob(f'{J}/tiles005/*/*.tif')):
        with rasterio.open(p) as s: tl[Path(p).stem]=(Path(p).parent.name,s.transform,s.res[0])
    assert len(tl)==2500
def ld(m,g,k): return np.load(PR/m/g/f'{k}.npz')['p'].astype(np.float32)/255.
def loader(g,k):
    if MODEL=='avg2': return lambda: ((ld('convnext_t_s0',g,k)+ld(B2,g,k))/2).astype(np.float32)
    mm={'convnext':'convnext_t_s0','b2':B2,'v2c':'v2c'}[MODEL]; return lambda: ld(mm,g,k)
tiles=[TileIn(k,*global_origin(tr,gsd),loader(g,k)) for k,(g,tr,gsd) in tl.items()]
plan=merge_plan(tiles,CFG); objs=plan_objects(plan); print('plan',len(tiles),'objects',len(objs),f'{time.time()-t0:.0f}s',flush=True)
parts=[]
for t in tiles:
    ids=plan_tile_ids(plan,t)
    if ids.any(): parts.append(polygons({t.key:ids},{t.key:tl[t.key][1]},CRS))
P=pd.concat(parts,ignore_index=True); multi=P.oid.duplicated(keep=False)
D=pd.concat([P[~multi],P[multi].dissolve(by='oid',as_index=False)],ignore_index=True)
D=gpd.GeoDataFrame(D,geometry='geometry',crs=CRS); assert D.oid.is_unique and len(D)==len(objs),(len(D),len(objs))
D['max_p']=[objs[o]['max_p'] for o in D.oid]; D['n_tiles']=[objs[o]['n_tiles'] for o in D.oid]; D['n_px']=[objs[o]['n_px'] for o in D.oid]
D['area_m2']=D.area; D.to_file(out+'.tmp.gpkg',driver='GPKG'); os.replace(out+'.tmp.gpkg',out)
print('DONE',CITY,MODEL,len(D),'area',round(D.area_m2.sum()),f'{time.time()-t0:.0f}s',flush=True)
