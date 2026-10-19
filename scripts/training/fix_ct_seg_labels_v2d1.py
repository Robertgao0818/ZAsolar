"""v2d -> v2d1 label fixes (2026-10-16 evening, 葉). New copy only.
1. G1971 sG1971_pv_00047: SAM3 replacement was a fragmented (56 parts, score 0.66) mask spilling over the
   pale roof; reverted to the original human row 334 and the SAM3 row removed.
2. G1972 row 113 (real PV per 葉): geometry replaced by SAM3 box-prompt tightened outline
   (best-score instance, clipped to orig+0.5 m, close/open, parts>=2 m2, per-part convex hull, clipped to orig).
Row order of the original source file is preserved."""
import argparse,glob,os,shutil,pandas as pd,geopandas as gpd
ap=argparse.ArgumentParser();ap.add_argument('--v2d-labels',required=True);ap.add_argument('--orig',default='/home/gao/projects/ZAsolar/data/annotations/Capetown')
ap.add_argument('--g1972-geom',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
if os.path.exists(a.out): raise SystemExit('exists '+a.out)
shutil.copytree(a.v2d_labels,a.out,copy_function=shutil.copy2); C=os.path.join(a.out,'Capetown'); ch=[]
f=glob.glob(f'{C}/G1971_SAM2_*.gpkg')[0]; d=gpd.read_file(f); o=gpd.read_file(glob.glob(f'{a.orig}/G1971_SAM2_*.gpkg')[0])
o=o.set_crs(4326) if o.crs is None else o
assert d.geometry.iloc[334] is None or d.geometry.iloc[334].is_empty
d.loc[334,'geometry']=o.to_crs(d.crs).geometry.iloc[334]; d['v2d_dropped']=d['v2d_dropped'].astype(object); d.loc[334,'v2d_dropped']=False
k=d.index[d.get('sam3_id')=='sG1971_pv_00047']; assert len(k)==1 and k[0]>=len(o); d=d.drop(index=k)
os.remove(f); d.to_file(f,driver='GPKG'); ch.append(dict(grid='G1971',action='revert_sam3_to_human',src_idx=334,sam3_id='sG1971_pv_00047'))
f=glob.glob(f'{C}/G1972_SAM2_*.gpkg')[0]; d=gpd.read_file(f); crs=d.crs or 'EPSG:4326'; d=d.set_crs(crs) if d.crs is None else d
g=gpd.read_file(a.g1972_geom).to_crs(crs).geometry.iloc[0]; old=d.to_crs(32734).geometry.iloc[113].area
d.loc[113,'geometry']=g; os.remove(f); d.to_file(f,driver='GPKG')
ch.append(dict(grid='G1972',action='tighten_sam3_box',src_idx=113,area_before_m2=round(old,1),area_after_m2=round(gpd.GeoSeries([g],crs=crs).to_crs(32734).area.iloc[0],1)))
c=pd.read_csv(os.path.join(a.out,'changes.csv')); c=c[~((c.grid=='G1971')&(c.sam3_id=='sG1971_pv_00047'))]
pd.concat([c,pd.DataFrame(ch)]).to_csv(os.path.join(a.out,'changes.csv'),index=False); print(pd.DataFrame(ch))
