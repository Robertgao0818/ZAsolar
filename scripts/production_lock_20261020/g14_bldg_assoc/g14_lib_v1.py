"""G14 shared association code (B-step rule, RESULTS_AC_v1 §B / B_offbuilding_v1/assign_pv_v1.py re-implemented per window).
0.5 m building buffer; PV off_frac<0.3 -> whole PV area allocated to touched buildings proportional to overlap; else only
on-roof part allocated and the off-roof part is off-building, classed by the assign_pv_v1 rule (dist to nearest building, zoning)."""
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely import STRtree
BUF=0.5; OFF=0.3; THR=2.0
def merge_gap(g, gap):
    """Row-split test: buffer-union at gap (m) then un-buffer; explode to parts. gap=0 -> original."""
    if gap<=0 or not len(g): return g[['geometry']].copy()
    u=shapely.union_all(shapely.buffer(g.geometry.values,gap/2)); u=shapely.buffer(u,-gap/2)
    out=gpd.GeoDataFrame(geometry=[u],crs=g.crs).explode(index_parts=False).reset_index(drop=True)
    return out[out.area>0]
def rule(orphan_area,dist,zz):
    if orphan_area>=200 or any(k in zz for k in ['agricult','utility','open space','rural','risk']) and orphan_area>=50: return 'ground_mount_candidate'
    if dist<=3 and orphan_area<50: return 'misaligned_candidate'
    if any(k in zz for k in ['general business','mixed use','industr','community','transport','general residential']) and 10<=orphan_area<200 and dist>3: return 'carport_candidate'
    if dist>3 and orphan_area>=5: return 'missing_building_candidate'
    return 'small_or_ambiguous'
def associate(pv, bld, parcels=None):
    """pv: GeoDataFrame of PV polygons; bld: buildings (window) with column bid. Returns (alloc Series by bid, pv frame with
    pv_area,on_bld_area,off_frac,orphan_area,rule_class,rep point)."""
    pv=pv[['geometry']].copy().reset_index(drop=True); pv['geometry']=shapely.make_valid(pv.geometry.values); pv['pv_area']=pv.area
    pv=pv[pv.pv_area>0].reset_index(drop=True)
    if not len(pv) or not len(bld):
        pv['on_bld_area']=0.0; pv['off_frac']=1.0; pv['orphan_area']=pv.pv_area; pv['dist']=np.inf; pv['zoning']=''
        pv['rule_class']=[rule(a,np.inf,'') for a in pv.orphan_area]; return pd.Series(dtype=float), pv
    bg=shapely.buffer(bld.geometry.values,BUF); tree=STRtree(bg); li,ri=tree.query(pv.geometry.values,predicate='intersects')
    ia=shapely.area(shapely.intersection(pv.geometry.values[li],bg[ri])) if len(li) else np.zeros(0)
    P=pd.DataFrame({'pv':li,'b':ri,'a':ia}); P=P[P.a>0]
    on=P.groupby('pv').a.sum().reindex(range(len(pv)),fill_value=0).clip(upper=pv.pv_area.values)
    pv['on_bld_area']=on.values; pv['off_frac']=1-pv.on_bld_area/pv.pv_area
    if len(P):
        sumA=P.groupby('pv').a.transform('sum').values; offf=pv.off_frac.values[P.pv]; area=pv.pv_area.values[P.pv]; onA=pv.on_bld_area.values[P.pv]
        P['alloc']=np.where(offf<OFF,P.a/sumA*area,P.a/sumA*onA); alloc=P.groupby(bld.bid.values[P.b]).alloc.sum()
    else: alloc=pd.Series(dtype=float)
    pv['orphan_area']=np.where(pv.off_frac<OFF,0.0,pv.pv_area-pv.on_bld_area)
    d=np.full(len(pv),np.inf); o=np.nonzero(pv.orphan_area.values>0)[0]
    if len(o):
        tq=STRtree(bld.geometry.values); idx,dist=tq.query_nearest(pv.geometry.values[o],max_distance=30,return_distance=True)
        dd=pd.Series(dist).groupby(idx[0]).min(); d[o[dd.index.values]]=dd.values
    pv['dist']=d; pv['zoning']=''
    if parcels is not None and len(parcels) and len(o):
        rp=gpd.GeoDataFrame(geometry=pv.geometry.iloc[o].representative_point().values,crs=pv.crs); rp['k']=o
        j=gpd.sjoin(rp,parcels[['ZONING','geometry']],how='left',predicate='within').drop_duplicates('k')
        pv.loc[j.k.values,'zoning']=j.ZONING.fillna('').astype(str).str.lower().values
    pv['rule_class']=[rule(a,dd_,z) if a>0 else '' for a,dd_,z in zip(pv.orphan_area,pv.dist,pv.zoning)]
    return alloc, pv
def wilson(k,n,z=1.96):
    import math
    if n==0: return (None,None)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (round(c-h,4),round(c+h,4))
