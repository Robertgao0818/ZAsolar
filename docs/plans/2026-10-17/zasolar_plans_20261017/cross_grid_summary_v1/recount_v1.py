# independent recount (CPU, read-only); writes recount_v1.json here only
import json,os,re,pandas as pd,geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box
H=os.path.expanduser;R={}
G={"A":"1842 1897 1898 1900 1901 1902 1950 1951","B":"1952 1953 1954","C":"1899 1844 1896","D":"1843 1846 1895"}
grp={"CPT"+c:k for k,v in G.items() for c in v.split()}
F=gpd.read_file(H("~/projects/ZAsolar/data/task_grid_ct_full.gpkg")).to_crs(32734)
F["grid"]="CPT"+F.Name.str[1:]; C=F[F.grid.isin(grp)].set_index("grid").geometry; assert len(C)==17
U=unary_union(list(C)); ext=U.boundary
allb=unary_union([c.boundary for c in C]); intl=allb.difference(ext.buffer(0.01))
raw=gpd.read_file(H("~/zasolar_data/capetown_testset_li_v1_20261017/derived/li_later_all_raw.gpkg")).to_crs(32734)
R["raw_features"]=len(raw); L=raw[raw.labeled].copy(); L["geometry"]=L.buffer(0); R["labelled"]=len(L)
R["labelled_area"]=round(L.area.sum(),1)
ins=L.intersection(U).area; lost=L.difference(U).area
R["wholly_outside"]=int((ins<=0.01).sum()); R["partial"]=int(((ins>0.01)&(lost>0.01)).sum()); R["lost_m2"]=round(lost.sum(),1)
S=L[ins>0.01].copy(); R["src_in_cells"]=len(S); R["src_in_cells_area_full"]=round(S.area.sum(),1); R["clipped_area"]=round(ins.sum(),1)
m=unary_union(list(S.geometry)); R["merged_touching_components"]=len(getattr(m,"geoms",[m]))
pcs=gpd.overlay(S[["geometry"]],gpd.GeoDataFrame({"grid":C.index},geometry=list(C),crs=32734),how="intersection",keep_geom_type=True)
pcs=pcs[pcs.area>0.01]; R["per_cell_pieces"]=len(pcs); R["pieces_by_group"]=pcs.grid.map(grp).value_counts().to_dict()
gt=gpd.read_file(H("~/zasolar_data/capetown_testset_li_z2_20261017/gt.gpkg")); R["gt_gpkg_rows"]=len(gt)
for d in (10,20):
  for nm,ln in (("any",allb),("ext",ext),("int",intl)):
    k=S.distance(ln)<=d; R[f"src_{nm}_{d}m"]=[int(k.sum()),round(S[k].area.sum(),1),round(100*S[k].area.sum()/S.area.sum(),2)]
for nm,ln in (("ext",ext),("int",intl)):
    k=S.intersects(ln); R[f"cross_{nm}"]=[int(k.sum()),round(S[k].area.sum(),1)]
T=pd.read_csv(H("~/zasolar_data/capetown_testset_li_z2_20261017/tiles.csv")); R["units"]=len(T)
base=T.tile_id.str.split("@").str[0]; R["tiles"]=base.nunique(); n=base.value_counts(); R["tiles_2plus_units"]=int((n>1).sum())
R["units_on_multi"]=int(base.isin(n[n>1].index).sum()); R["units_by_group"]=T.grid.map(grp).value_counts().to_dict()
json.dump(R,open("recount_v1.json","w"),indent=1); print(json.dumps(R))
