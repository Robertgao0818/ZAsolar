"""Object-level loss caused by the pregate on Joburg CBD25. GT = clean_gt (same as G6 / official joburg25_vexcel).
Tile score = max over 16 chips and 4 heads. Rule a: keep tile iff score>=0.45. Rule a+b: also keep if any 8-neighbour tile
(same grid, 100 m lattice) >=0.45. Object missed iff every tile it intersects is rejected. Also reports the official
G6 cell gate (pregate/scores.json, 3x3 cells/tile, native res) at object level (missed iff no kept cell intersects it) for comparison."""
import os,glob,json,math,numpy as np,pandas as pd,geopandas as gpd,rasterio
from shapely.geometry import box
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); THR=0.45
def wilson(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (round(c-h,4),round(c+h,4))
C=pd.read_csv('jhb_chip_scores_v1.csv'); assert C.image.nunique()==2500 and len(C)==40000
C['s']=C[['s0','s1','s2','s3']].max(1); T=C.groupby(['grid','image']).s.max().reset_index()
rows=[]
for r in T.itertuples():
    with rasterio.open(r.image) as s: b=s.bounds
    rows.append(dict(grid=r.grid,image=r.image,score=r.s,geometry=box(*b),mx=(b.left+b.right)/2,my=(b.bottom+b.top)/2))
T=gpd.GeoDataFrame(rows,crs=32735); T['keep_a']=T.score>=THR
T['i']=((T['mx']-T['mx'].min())/100).round().astype(int); T['j']=((T['my']-T['my'].min())/100).round().astype(int)
K={(g,i,j) for g,i,j,k in zip(T.grid,T.i,T.j,T.keep_a) if k}
T['keep_ab']=[k or any((g,i+di,j+dj) in K for di in(-1,0,1) for dj in(-1,0,1)) for g,i,j,k in zip(T.grid,T.i,T.j,T.keep_a)]
PG=json.load(open(f'{J}/pregate/scores.json'))
cells=[]
for r in T.itertuples():
    sc=PG['scores'][os.path.relpath(r.image,J)]; b=r.geometry.bounds; e=[0,667,2000-667]
    for k,v in enumerate(sc):
        i,j=divmod(k,3); y0,x0=e[i],e[j]
        cells.append(dict(grid=r.grid,keep=v>=PG['threshold'],geometry=box(b[0]+x0*.05,b[3]-(y0+667)*.05,b[0]+(x0+667)*.05,b[3]-y0*.05)))
CE=gpd.GeoDataFrame(cells,crs=32735)
G=[]
for f in sorted(glob.glob(f'{J}/clean_gt/*_clean_gt.gpkg')):
    d=gpd.read_file(f).to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0)
    d['grid']=os.path.basename(f)[:5]; d['oid']=np.arange(1,len(d)+1); G.append(d[['grid','oid','geometry']])
G=gpd.GeoDataFrame(pd.concat(G,ignore_index=True),crs=32735); G['area']=G.geometry.area
def lost(tab,col):
    j=gpd.sjoin(G[['grid','oid','geometry']],tab[['grid',col,'geometry']],predicate='intersects')
    j=j[j.grid_left==j.grid_right]; kept=j.groupby(['grid_left','oid'])[col].any()
    return np.array([not kept.get((g,o),False) for g,o in zip(G.grid,G.oid)])
_j=gpd.sjoin(G[['grid','oid','geometry']],T[['grid','geometry']],predicate='intersects'); _j=_j[_j.grid_left==_j.grid_right]
_cov=set(zip(_j.grid_left,_j.oid)); G['in_tiles']=[(g,o) in _cov for g,o in zip(G.grid,G.oid)]
n_out=int((~G.in_tiles).sum()); G=G[G.in_tiles].reset_index(drop=True)  # objects outside own grid's tiles are never scored (gt_px=0)
res=dict(n_gt_outside_own_grid_tiles_excluded=n_out,threshold=THR,n_tiles=len(T),n_gt=len(G),note='raw clean_gt polygons, no dedupe/area filter; object missed iff all intersecting units rejected')
BINS=[('all',0,1e9),('lt4',0,4),('4_10',4,10),('ge10',10,1e9)]
for nm,tab,col in (('rule_a',T,'keep_a'),('rule_ab',T,'keep_ab'),('official_g6_cellgate',CE,'keep')):
    L=lost(tab,col); r=dict(filtered_share=round(1-tab[col].mean(),4),n_units=len(tab))
    for b,lo,hi in BINS:
        s=(G.area>=lo)&(G.area<hi); k,n=int(L[s].sum()),int(s.sum()); r[f'miss_{b}']=round(k/max(n,1),4); r[f'miss_{b}_n']=f'{k}/{n}'; r[f'miss_{b}_ci']=wilson(k,n)
        a=G.area[s]; r[f'area_miss_{b}']=round(float(a[L[s]].sum()/max(a.sum(),1e-9)),4)
    res[nm]=r; G[f'lost_{nm}']=L
pv_tiles=gpd.sjoin(T,G[['grid','geometry']],predicate='intersects'); pv_tiles=pv_tiles[pv_tiles.grid_left==pv_tiles.grid_right].image.unique()
T['has_gt']=T.image.isin(pv_tiles)
for col in('keep_a','keep_ab'): res[f'tile_miss_{col}']=f'{int((T.has_gt&~T[col]).sum())}/{int(T.has_gt.sum())}'
res['per_grid_filtered_share_a']=T.groupby('grid').keep_a.apply(lambda x:round(1-x.mean(),3)).to_dict()
json.dump(res,open('results_v1.json','w'),indent=1,default=str); T.drop(columns='geometry').to_csv('tile_scores_v1.csv',index=False)
G[G.filter(like='lost_').any(axis=1)].to_file('lost_objects_v1.gpkg',driver='GPKG') if G.filter(like='lost_').any(axis=1).any() else None
print(json.dumps(res,indent=1,default=str))
