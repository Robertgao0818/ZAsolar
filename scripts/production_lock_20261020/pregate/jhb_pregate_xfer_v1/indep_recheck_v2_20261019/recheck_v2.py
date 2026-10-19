"""Independent recheck v2 of Joburg pregate transfer: tile score = max(16 chips x 4 heads) from jhb_chip_scores_v1.csv;
tile lattice from raster bounds (own); neighbour rules: NB_same = own OR any 8-neighbour in same grid >= tau (光伏·运行 sweep def);
NB_any = same but neighbours may be in other grids (geometric adjacency). Also cell-gate (pregate/scores.json) for reference.
GT: clean_gt buffer(0), raw (no dedupe) and deduped variants; 3 outside-own-grid polygons examined."""
import os,glob,json,numpy as np,pandas as pd,geopandas as gpd,rasterio,sys
from shapely.geometry import box
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar_segv1_wt/scripts/training'))
from build_ct_seg_trainset_v1 import dedupe_polygons
B=os.path.expanduser('~/capetown_eval/jhb_pregate_xfer_v1_20261019'); J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel')
assert not os.path.exists('recheck_v2.json')
C=pd.read_csv(f'{B}/jhb_chip_scores_v1.csv'); C['s']=C[['s0','s1','s2','s3']].max(1)
S=C.groupby(['grid','image']).s.max().reset_index()
bb=[rasterio.open(p).bounds for p in S.image]
S['x0']=[b.left for b in bb]; S['y0']=[b.bottom for b in bb]
T=gpd.GeoDataFrame(S,geometry=[box(*b) for b in bb],crs=32735).reset_index(drop=True)
X0,Y0=T.x0.min(),T.y0.min(); T['ix']=np.rint((T.x0-X0)/100).astype(int); T['iy']=np.rint((T.y0-Y0)/100).astype(int)
OFF=float(max(np.abs((T.x0-X0)/100-T.ix).max(),np.abs((T.y0-Y0)/100-T.iy).max()))
pos={(x,y):k for k,(x,y) in enumerate(zip(T.ix,T.iy))}; assert len(pos)==len(T),'duplicate lattice positions across grids'
gpos={(g,x,y):k for k,(g,x,y) in enumerate(zip(T.grid,T.ix,T.iy))}
nb_same=[[gpos[(g,x+a,y+b)] for a in(-1,0,1) for b in(-1,0,1) if (g,x+a,y+b) in gpos] for g,x,y in zip(T.grid,T.ix,T.iy)]
nb_any=[[pos[(x+a,y+b)] for a in(-1,0,1) for b in(-1,0,1) if (x+a,y+b) in pos] for x,y in zip(T.ix,T.iy)]
n_cross=sum(len(a)>len(b) for a,b in zip(nb_any,nb_same))
G=[]
for f in sorted(glob.glob(f'{J}/clean_gt/*_clean_gt.gpkg')):
    d=gpd.read_file(f).to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0); g=os.path.basename(f)[:5]
    dd=dedupe_polygons(d.reset_index(drop=True))
    G.append(gpd.GeoDataFrame(dict(grid=[g]*len(d),dedup=[False]*len(d)),geometry=list(d.geometry),crs=32735)); G.append(gpd.GeoDataFrame(dict(grid=[g]*len(dd),dedup=[True]*len(dd)),geometry=list(dd.geometry),crs=32735))
G=gpd.GeoDataFrame(pd.concat(G,ignore_index=True),crs=32735); G['a']=G.area; G['gid']=range(len(G))
j=gpd.sjoin(G,T[['grid','geometry']].rename(columns={'grid':'tgrid'}),predicate='intersects')
same=j[j.grid==j.tgrid]; G['in_own']=G.gid.isin(same.gid); G['in_any']=G.gid.isin(j.gid)
s=T.s.values; res=dict(lattice_max_offset_tiles=OFF,n_tiles=len(T),n_tiles_with_crossgrid_nb=n_cross,
  gt={str(dd):dict(n=int((G.dedup==dd).sum()),in_own_grid_tiles=int((G.in_own&(G.dedup==dd)).sum()),in_any_tiles=int((G.in_any&(G.dedup==dd)).sum())) for dd in(False,True)})
out=G[(~G.in_own)&(G.dedup==False)]
res['outside_own']=[dict(grid=r.grid,area_m2=round(r.a,2),centroid=[round(r.geometry.centroid.x,1),round(r.geometry.centroid.y,1)],intersects_tiles_of=sorted(set(j[j.gid==r.gid].tgrid)),max_score_any=float(T.s[j[j.gid==r.gid].index_right].max()) if r.gid in set(j.gid) else None, dist_to_own_grid_tiles_m=round(float(T[T.grid==r.grid].geometry.union_all().distance(r.geometry)),1)) for r in out.itertuples()]
rows=[]
for tau in [0.45,0.5,0.6,0.65,0.7,0.75,0.9]:
    own=s>=tau
    for rule,k in (('own',own),('own+nb_same',np.array([own[n].any() for n in nb_same])),('own+nb_any',np.array([own[n].any() for n in nb_any]))):
        r=dict(tau=tau,rule=rule,keep=round(float(k.mean()),4),keep_n=int(k.sum()))
        jj=j.assign(k=k[j.index_right.values])
        for lab,sub,jx in (('own_grid',G.in_own,jj[jj.grid==jj.tgrid]),('incl_outside_anygrid',G.in_any,jj)):
            hit=jx.groupby('gid').k.any()
            for dd in(False,True):
                m=sub&(G.dedup==dd); h=G.gid[m].map(hit).fillna(False).astype(bool)
                a=G.a[m]; r[f'{lab}_{"dedup" if dd else "raw"}']=f'{int(h.sum())}/{int(m.sum())}'
                r[f'{lab}_{"dedup" if dd else "raw"}_4_10']=f'{int(h[(a>=4)&(a<10)].sum())}/{int(((a>=4)&(a<10)).sum())}'
        rows.append(r)
D=pd.DataFrame(rows); res['sweep']=rows
sw=pd.read_csv(f'{B}/sweep_v1.csv'); res['their_sweep_keep']={f'{t}_{r}':k for t,r,k in zip(sw.tau,sw.rule,sw.tiles_kept_share)}
json.dump(res,open('recheck_v2.json','w'),indent=1,default=str); D.to_csv('recheck_v2_sweep.csv',index=False)
print(json.dumps({k:v for k,v in res.items() if k not in('sweep',)},indent=1,default=str)); print(D.to_string())
