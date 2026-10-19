"""Threshold sweep for the Joburg pregate transfer (reads tile_scores_v1.csv from analyze_jhb.py; same clean_gt object set:
raw polygons, objects outside their own grid's tiles excluded). own = keep tile iff score>=tau; nb = own OR any 8-neighbour
(same grid) >=tau. Object recall = objects with >=1 kept intersecting tile / all. Writes sweep_v1.csv / sweep_v1.md."""
import os,glob,numpy as np,pandas as pd,geopandas as gpd
from shapely.geometry import box
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel')
T=pd.read_csv('tile_scores_v1.csv'); T=gpd.GeoDataFrame(T,geometry=[box(x-50,y-50,x+50,y+50) for x,y in zip(T.mx,T.my)],crs=32735).reset_index(drop=True)
G=[]
for f in sorted(glob.glob(f'{J}/clean_gt/*_clean_gt.gpkg')):
    d=gpd.read_file(f).to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0); d['grid']=os.path.basename(f)[:5]; G.append(d[['grid','geometry']])
G=gpd.GeoDataFrame(pd.concat(G,ignore_index=True),crs=32735); G['area']=G.geometry.area; G['gid']=range(len(G))
j=gpd.sjoin(G,T[['grid','geometry']],predicate='intersects'); j=j[j.grid_left==j.grid_right][['gid','index_right']]
G=G[G.gid.isin(j.gid)]; assert len(G)==2065,len(G)
idx={(g,i,jj):k for k,(g,i,jj) in enumerate(zip(T.grid,T.i,T.j))}
nb=[[idx[(g,i+a,jj+b)] for a in(-1,0,1) for b in(-1,0,1) if (g,i+a,jj+b) in idx] for g,i,jj in zip(T.grid,T.i,T.j)]
s=T.score.values; rows=[]
for tau in [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.97,0.98,0.99]:
    own=s>=tau; nbk=np.array([own[n].any() for n in nb])
    for rule,k in (('own',own),('own+nb',nbk)):
        hit=j.assign(k=k[j.index_right.values]).groupby('gid').k.any()
        h=G.gid.map(hit).values; r=dict(tau=tau,rule=rule,tiles_kept_share=round(k.mean(),4))
        for nm,lo,hi in (('all',0,1e9),('lt4',0,4),('4_10',4,10)):
            m=(G.area>=lo)&(G.area<hi); r[f'recall_{nm}']=round(h[m].mean(),4); r[f'n_{nm}']=f'{int(h[m].sum())}/{int(m.sum())}'
        rows.append(r)
D=pd.DataFrame(rows); D.to_csv('sweep_v1.csv',index=False); open('sweep_v1.md','w').write(D.to_markdown(index=False)); print(D.to_string())
