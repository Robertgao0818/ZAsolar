# independent: chip scores -> tile max; tile footprint from raster bounds; GT intersect test per grid
import os,glob,json,numpy as np,pandas as pd,geopandas as gpd,rasterio
from shapely.geometry import box
B=os.path.expanduser('~/capetown_eval/jhb_pregate_xfer_v1_20261019'); J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel')
C=pd.read_csv(f'{B}/jhb_chip_scores_v1.csv'); print('chips',len(C),'tiles',C.image.nunique(), 'chips/tile',C.groupby('image').size().unique())
S=C.set_index(['grid','image'])[['s0','s1','s2','s3']].max(axis=1).groupby(level=[0,1]).max().rename('score').reset_index()
geo=[]
for im in S.image:
    with rasterio.open(im) as r: geo.append(box(*r.bounds))
T=gpd.GeoDataFrame(S,geometry=geo,crs=32735)
old=pd.read_csv(f'{B}/tile_scores_v1.csv').set_index('image').score
print('max abs diff vs tile_scores_v1',float((T.set_index('image').score-old.reindex(T.image).values).abs().max()))
G=[]
for f in sorted(glob.glob(f'{J}/clean_gt/*_clean_gt.gpkg')):
    d=gpd.read_file(f).to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty]; d=d.assign(geometry=d.geometry.buffer(0),grid=os.path.basename(f)[:5]); G.append(d[['grid','geometry']])
G=gpd.GeoDataFrame(pd.concat(G,ignore_index=True),crs=32735); G['gid']=np.arange(len(G)); G['a']=G.area
j=gpd.sjoin(G,T[['grid','score','geometry']],predicate='intersects'); j=j[j.grid_left==j.grid_right]
print('gt total',len(G),'in tiles',j.gid.nunique())
mx=j.groupby('gid').score.max(); G=G[G.gid.isin(mx.index)]; G['best']=G.gid.map(mx)
out={}
for tau in [0.45,0.6,0.65,0.7,0.75,0.9]:
    r={'keep':round(float((T.score>=tau).mean()),4),'keep_n':int((T.score>=tau).sum())}
    for nm,lo,hi in (('all',0,1e9),('lt4',0,4),('4_10',4,10),('ge10',10,1e9)):
        m=(G.a>=lo)&(G.a<hi); r[nm]=f"{int((G.best[m]>=tau).sum())}/{int(m.sum())}"
    out[tau]=r; print(tau,r)
print('min best-score over GT objects',float(G.best.min()), '4-10 min',float(G.best[(G.a>=4)&(G.a<10)].min()))
json.dump(out,open('recheck_v1.json','w'),indent=1)
