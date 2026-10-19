"""Stage 1b: GT polygons + evaluation regions. CT: val labels as ct_val_group_diag score_ct_v1 (pipeline_common_v3 load_grid_gt,
v2f manifest, G1630 swapped to v2g label), PV only. Region per grid = union of its val tile extents ∩ its 1 km cell
(task_grid_ct_full.gpkg). JHB: reference_with_hits.gpkg (2,174, 葉 FN additions incl.); region = tiles005 extents ∩ joburg5 grid
polygon. Writes gt/CITY_gt.gpkg (grid, oid, geometry) and gt/CITY_regions.gpkg. usage: prep_gt_v1.py CITY"""
import os,sys,glob,hashlib,numpy as np,pandas as pd,geopandas as gpd,rasterio,shapely
from pathlib import Path
from shapely.geometry import box
CITY=sys.argv[1]; og,orr=f'gt/{CITY}_gt.gpkg',f'gt/{CITY}_regions.gpkg'; assert not os.path.exists(og) and not os.path.exists(orr)
if CITY=='CT':
    sys.path.insert(0,str(Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser()))
    from pipeline_common_v3 import select_rows, validation_scopes, load_grid_gt, ref_for
    import rasterio.transform as rt
    rows,gm=select_rows('val'); rows,gm,_=validation_scopes(rows); assert len(rows)==1703
    NEWL='/home/gao/RA_solar/census/capetown/labels/corrected/ct_labels_corrected_v2g_20261019/Capetown/G1630_SAM2_260324.gpkg'
    gm=gm.copy(); gm.loc['G1630','label_file']=NEWL; gm.loc['G1630','md5']=hashlib.md5(open(NEWL,'rb').read()).hexdigest()
    cells=gpd.read_file(os.path.expanduser('~/projects/ZAsolar/data/task_grid_ct_full.gpkg'))[['gridcell_id','geometry']].to_crs(32734).set_index('gridcell_id').geometry
    G=[];R=[]
    for g,grp in rows.groupby('grid'):
        gt=load_grid_gt(g,gm)['PV']; G.append(gpd.GeoDataFrame(dict(grid=g,oid=gt.oid.values),geometry=gt.geometry.values,crs=gt.crs))
        ex=[box(*rt.array_bounds(*ref.shape,ref.transform)) for ref in (ref_for(pd.Series(r)) for r in grp.to_dict('records'))]
        R.append(dict(grid=g,n_tiles=len(ex),geometry=shapely.intersection(shapely.union_all(ex),cells[g])))
    CRS=32734
else:
    J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel')
    ref=gpd.read_file(os.path.expanduser('~/capetown_eval/joburg_cbd25_phase1_20261018/analysis/recall_with_fn_v1/reference_with_hits.gpkg')).to_crs(32735)
    ref=ref[ref.geometry.notna()&~ref.geometry.is_empty].copy(); ref['geometry']=shapely.make_valid(ref.geometry.values)
    G=[gpd.GeoDataFrame(dict(grid=d.grid.values,oid=np.arange(1,len(d)+1)),geometry=d.geometry.values,crs=32735) for _,d in ref.groupby('grid')]
    cells=gpd.read_file(f'{J}/../joburg5/grids_32735.gpkg').set_index('gridcell_id').geometry; R=[]
    for gdir in sorted(glob.glob(f'{J}/tiles005/G*')):
        g=os.path.basename(gdir); ex=[box(*rasterio.open(p).bounds) for p in sorted(glob.glob(f'{gdir}/*.tif'))]
        R.append(dict(grid=g,n_tiles=len(ex),geometry=shapely.intersection(shapely.union_all(ex),cells[g])))
    CRS=32735
G=gpd.GeoDataFrame(pd.concat(G,ignore_index=True),crs=CRS); R=gpd.GeoDataFrame(R,crs=CRS)
G.to_file(og,driver='GPKG'); R.to_file(orr,driver='GPKG'); print('DONE',CITY,len(G),'polys',len(R),'grids',round(R.area.sum()/1e6,3),'km2')
