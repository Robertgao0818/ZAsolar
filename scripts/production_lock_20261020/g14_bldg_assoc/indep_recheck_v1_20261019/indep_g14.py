"""Independent G14 recheck (光伏·项目主管 executor, 2026-10-19). Does NOT import g14_lib_v1/assoc_grid_*.
Frozen rule: h0.9_0.5 + A3 cross-tile merge = xtile_merge.merge_plan/plan_tile_ids (G7 frozen module, md5 checked).
Own polygonisation, own association (0.5 m bld buffer; off_frac<0.3 -> whole PV area to touched buildings by overlap share,
else on-roof part only), installed = allocated >= 2 m2, eval building = representative point inside region.
modes: prep CITY | pred CITY MODEL | grid CITY GRID | agg"""
import os,sys,glob,json,math,hashlib,pickle,numpy as np,pandas as pd,geopandas as gpd,shapely,rasterio
from pathlib import Path
from shapely.geometry import box
H=Path(__file__).resolve().parent; os.chdir(H)
X=Path('~/capetown_eval/ct_xtile_merge_v1_20261019').expanduser(); sys.path.insert(0,str(X))
assert hashlib.md5((X/'xtile_merge.py').read_bytes()).hexdigest()=='d69fea350e588b468b36f55bce299a0c'
from xtile_merge import TileIn, merge_plan, plan_tile_ids, global_origin
PIPE=Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser()
J=Path('~/capetown_eval/ds3m_ft/joburg25_vexcel').expanduser(); sys.path.insert(0,str(PIPE))
A='/home/gao/projects/zasolar_plans_20261017/building_mask_20261018/A_basemap_v1'; BP='/home/gao/RA_solar/census/data/buildings_public_20261018'
CRS={'CT':32734,'JHB':32735}; MODELS={'CT':['convnext','b2','avg2'],'JHB':['convnext','b2','v2c','avg2']}
OLD='ct_citywide_install_dated_20260902'  # must never be read
def ct_rows():
    sys.path.insert(0,str(PIPE)); from pipeline_common_v3 import select_rows, validation_scopes
    rows,gm=select_rows('val'); rows,gm,_=validation_scopes(rows); assert len(rows)==1703; return rows,gm
def tiles(city):
    out={}
    if city=='CT':
        from pipeline_common_v3 import ref_for
        rows,_=ct_rows()
        for r in rows.to_dict('records'):
            ref=ref_for(pd.Series(r)); out.setdefault(ref.tile_id,(r['grid'],ref.transform,ref.shape))
    else:
        for p in sorted(glob.glob(f'{J}/tiles005/*/*.tif')):
            with rasterio.open(p) as s: out[Path(p).stem]=(Path(p).parent.name,s.transform,(s.height,s.width))
    return out
def prep(city):
    import rasterio.transform as rt
    T=tiles(city); R=[]
    if city=='CT':
        from pipeline_common_v3 import load_grid_gt
        _,gm=ct_rows(); gm=gm.copy()
        NEWL='/home/gao/RA_solar/census/capetown/labels/corrected/ct_labels_corrected_v2g_20261019/Capetown/G1630_SAM2_260324.gpkg'
        gm.loc['G1630','label_file']=NEWL; gm.loc['G1630','md5']=hashlib.md5(open(NEWL,'rb').read()).hexdigest()
        cells=gpd.read_file(os.path.expanduser('~/projects/ZAsolar/data/task_grid_ct_full.gpkg')).to_crs(32734).set_index('gridcell_id').geometry
        L=[]
        for g in sorted({v[0] for v in T.values()}):
            gt=load_grid_gt(g,gm)['PV']; L.append(gpd.GeoDataFrame(dict(grid=[g]*len(gt),lset=['v2g']*len(gt)),geometry=list(gt.geometry),crs=32734))
        gt=pd.concat(L,ignore_index=True)
    else:
        cells=gpd.read_file(J.parent/'joburg5/grids_32735.gpkg').set_index('gridcell_id').geometry
        a=gpd.read_file(os.path.expanduser('~/capetown_eval/joburg_cbd25_phase1_20261018/analysis/recall_with_fn_v1/reference_with_hits.gpkg')).to_crs(32735)
        a=gpd.GeoDataFrame(dict(grid=a.grid.values,lset='ref'),geometry=list(shapely.make_valid(a.geometry.values)),crs=32735)
        b=[gpd.read_file(f).to_crs(32735).assign(grid=Path(f).name[:5]) for f in sorted(glob.glob(f'{J}/clean_gt/G*_clean_gt.gpkg'))]
        b=pd.concat(b,ignore_index=True); b=gpd.GeoDataFrame(dict(grid=b.grid.values,lset='clean'),geometry=list(shapely.make_valid(b.geometry.values)),crs=32735)
        gt=pd.concat([a,b],ignore_index=True)
    gt=gpd.GeoDataFrame(gt,geometry='geometry',crs=CRS[city]); gt=gt[~gt.is_empty&gt.geometry.notna()]
    for g in sorted({v[0] for v in T.values()}):
        ex=[box(*rt.array_bounds(sh[0],sh[1],tr)) for gg,tr,sh in T.values() if gg==g]
        R.append(dict(grid=g,geometry=shapely.intersection(shapely.union_all(ex),cells[g])))
    gt.to_file(f'gt/{city}_gt.gpkg'); gpd.GeoDataFrame(R,crs=CRS[city]).to_file(f'gt/{city}_regions.gpkg')
    print('prep',city,gt.groupby('lset').size().to_dict(),len(R))
def pred(city,model):
    from rasterio import features
    out=f'preds/{city}_{model}.gpkg'; assert not os.path.exists(out)
    PR=Path('~/capetown_eval/ds3m_ft/phase1_followup_valprobs_20261018' if city=='CT' else '~/capetown_eval/jhb_cbd25_zaeval2_v1_20261019/probs').expanduser()
    b2='mit_b2_s2' if city=='CT' else 'mit_b2_s0'; T=tiles(city)
    def ld(m,g,k): return np.load(PR/m/g/f'{k}.npz')['p'].astype(np.float32)/255.
    def mk(g,k):
        if model=='avg2': return lambda: (ld('convnext_t_s0',g,k)+ld(b2,g,k))/2
        m={'convnext':'convnext_t_s0','b2':b2,'v2c':'v2c'}[model]; return lambda: ld(m,g,k)
    TL=[TileIn(k,*global_origin(tr,abs(tr.a)),mk(g,k)) for k,(g,tr,sh) in T.items()]
    plan=merge_plan(TL,'h0.9_0.5'); geo={}
    for t in TL:
        ids=plan_tile_ids(plan,t)
        if not ids.any(): continue
        ids32=ids.astype(np.int32); assert (ids32==ids).all()
        for gj,v in features.shapes(ids32,mask=ids32>0,transform=T[t.key][1],connectivity=8):
            geo.setdefault(int(v),[]).append(shapely.geometry.shape(gj))
    D=gpd.GeoDataFrame(dict(oid=list(geo)),geometry=[shapely.union_all(v) for v in geo.values()],crs=CRS[city])
    D.to_file(out); print('pred',city,model,len(D),round(D.area.sum()))
def merge_gap(g,gap):
    if gap<=0: return list(g)
    u=shapely.buffer(shapely.union_all(shapely.buffer(np.asarray(g),gap/2)),-gap/2)
    return [p for p in getattr(u,'geoms',[u]) if p.area>0]
def assoc(pvs,bgeom):
    """pvs list of polygons; bgeom array of building polys -> alloc array (len bgeom), offbuilding area"""
    al=np.zeros(len(bgeom)); off=0.0
    if not len(pvs) or not len(bgeom): return al,sum(p.area for p in pvs)
    bb=shapely.buffer(bgeom,0.5); tr=shapely.STRtree(bb)
    for p in pvs:
        A_=p.area
        if A_<=0: continue
        idx=tr.query(p,predicate='intersects')
        ia=shapely.area(shapely.intersection(p,bb[idx])) if len(idx) else np.zeros(0)
        on=min(ia.sum(),A_)
        if on<=0: off+=A_; continue
        if 1-on/A_<0.3: al[idx]+=ia/ia.sum()*A_
        else: al[idx]+=ia/ia.sum()*on; off+=A_-on
    return al,off
def grid(city,G):
    out=f'parts/{city}_{G}.pkl'; assert not os.path.exists(out)
    R=gpd.read_file(f'gt/{city}_regions.gpkg').set_index('grid'); reg=R.geometry[G]
    W=gpd.GeoSeries([box(*reg.buffer(100).bounds)],crs=CRS[city])
    lay={}
    if city=='CT':
        lay['basemap']=gpd.read_file(f'{A}/cpt_buildings_merged_v1.gpkg',bbox=W)[['parcel_key','geometry']].to_crs(CRS[city])
        par=gpd.read_file(f'{A}/coct_parcels_v1.gpkg',bbox=W).to_crs(CRS[city]); g3=gpd.read_file(f'{BP}/google_v3/cpt_google_v3.gpkg',bbox=W).to_crs(CRS[city])
    else:
        srcs=[gpd.read_file(f,bbox=W).to_crs(CRS[city]) for f in (f'{BP}/google_v3/jhb_google_v3.gpkg',f'{BP}/osm/osm_buildings_jhb.gpkg',f'{BP}/microsoft/jhb_microsoft_v20260724.gpkg')]
        u=shapely.union_all(shapely.make_valid(np.concatenate([s.geometry.values for s in srcs])))
        lay['union_dissolved']=gpd.GeoDataFrame(geometry=[p for p in getattr(u,'geoms',[u]) if p.geom_type=='Polygon'],crs=CRS[city])
        g3=srcs[0]; par=None
        lay['google_v3_raw']=g3[['geometry']].copy()
    g3=g3[g3.geometry.notna()]; g3pt=g3.representative_point()
    gt=gpd.read_file(f'gt/{city}_gt.gpkg',bbox=W)
    labs={}
    for ls,d in gt.groupby('lset'):
        geoms=shapely.make_valid(d.geometry.values)
        for gap in ([0,0.25,1.0] if ls in ('v2g','ref') else [0]): labs[f'{ls}_g{gap}']=merge_gap(geoms,gap)
    preds={m:list(gpd.read_file(f'preds/{city}_{m}.gpkg',bbox=W).geometry) for m in MODELS[city]}
    res={}
    for ln,L in lay.items():
        L=L[L.geometry.notna()&~L.geometry.is_empty].reset_index(drop=True); bg=shapely.make_valid(L.geometry.values)
        ev=shapely.within(shapely.point_on_surface(bg),reg)
        # multi-building blocks: polygons containing >=2 google_v3 representative points
        t=shapely.STRtree(g3pt.values); bi,_=t.query(bg[ev],predicate='contains') if len(g3pt) else (np.zeros(0,int),None)
        nsrc=np.bincount(bi,minlength=int(ev.sum())) if len(g3pt) else np.zeros(int(ev.sum()),int)
        df=pd.DataFrame(dict(nsrc=nsrc,area=shapely.area(bg[ev])))
        for k,v in {**labs,**preds}.items(): a,off=assoc(v,bg); df[k]=a[ev]; res.setdefault(ln+'|off',{})[k]=off
        if city=='CT':
            df['parcel_key']=L.parcel_key.values[ev]
        res[ln]=df
    pickle.dump(dict(city=city,grid=G,res=res),open(out,'wb')); print('grid',city,G,{k:len(v) for k,v in res.items() if '|' not in k})
def wilson(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (round(c-h,4),round(c+h,4))
def agg():
    rows=[]; blocks=[]; fpcat=[]
    for city in ('CT','JHB'):
        P=[pickle.load(open(f,'rb')) for f in sorted(glob.glob(f'parts/{city}_*.pkl'))]
        for ln in P[0]['res']:
            if '|' in ln: continue
            df=pd.concat([p['res'][ln] for p in P],ignore_index=True)
            blocks.append(dict(city=city,layer=ln,n_eval=len(df),n_multi_ge2=int((df.nsrc>=2).sum()),n_ge5=int((df.nsrc>=5).sum()),
                               frac_multi=round((df.nsrc>=2).mean(),4),gt_inst_on_multi=None))
            lcols=[c for c in df.columns if c.split('_g')[0] in ('v2g','ref','clean')]
            for lc in lcols:
                gi=df[lc]>=2
                if lc==lcols[0] or True: blocks[-1][f'multi_share_of_gtinst_{lc}']=round(float((df.nsrc[gi]>=2).mean()),4)
                for m in MODELS[city]:
                    pi=df[m]>=2; tp=int((gi&pi).sum()); fp=int((~gi&pi).sum()); fn=int((gi&~pi).sum())
                    Pp=tp/max(tp+fp,1); Rr=tp/max(tp+fn,1)
                    rows.append(dict(city=city,layer=ln,labels=lc,model=m,n_bld=len(df),gt_inst=int(gi.sum()),pred_inst=int(pi.sum()),tp=tp,fp=fp,fn=fn,
                        P=round(Pp,4),P_ci=wilson(tp,tp+fp),R=round(Rr,4),R_ci=wilson(tp,tp+fn),F1=round(2*Pp*Rr/max(Pp+Rr,1e-9),4),
                        count_err=round((pi.sum()-gi.sum())/max(gi.sum(),1),4)))
                    if lc in ('v2g_g0','ref_g0','clean_g0'):
                        f=~gi&pi; fpcat.append(dict(city=city,layer=ln,labels=lc,model=m,fp=int(f.sum()),
                            fp_gt_subthr_0to2=int((f&(df[lc]>0)).sum()),fp_gt_zero=int((f&(df[lc]==0)).sum()),
                            fp_pred_lt5m2=int((f&(df[m]<5)).sum()),fp_on_multiblock=int((f&(df.nsrc>=2)).sum()),
                            fn=int((gi&~pi).sum()),fn_pred_subthr=int((gi&~pi&(df[m]>0)).sum())))
            if city=='CT':
                for lc in ['v2g_g0']:
                    e=df.dropna(subset=['parcel_key']).groupby('parcel_key')[[lc]+MODELS[city]].sum()
                    for m in MODELS[city]:
                        gi=e[lc]>=2; pi=e[m]>=2; tp=int((gi&pi).sum()); fp=int((~gi&pi).sum()); fn=int((gi&~pi).sum())
                        Pp=tp/max(tp+fp,1); Rr=tp/max(tp+fn,1)
                        rows.append(dict(city=city,layer=ln+'_ERF',labels=lc,model=m,n_bld=len(e),gt_inst=int(gi.sum()),pred_inst=int(pi.sum()),tp=tp,fp=fp,fn=fn,P=round(Pp,4),P_ci=wilson(tp,tp+fp),R=round(Rr,4),R_ci=wilson(tp,tp+fn),F1=round(2*Pp*Rr/max(Pp+Rr,1e-9),4),count_err=round((pi.sum()-gi.sum())/max(gi.sum(),1),4)))
    pd.DataFrame(rows).to_csv('indep_building_v1.csv',index=False); pd.DataFrame(blocks).to_csv('indep_blocks_v1.csv',index=False); pd.DataFrame(fpcat).to_csv('indep_fpcat_v1.csv',index=False)
    print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(blocks).to_string()); print(pd.DataFrame(fpcat).to_string())
if __name__=='__main__':
    m=sys.argv[1]; {'prep':lambda:prep(sys.argv[2]),'pred':lambda:pred(*sys.argv[2:4]),'grid':lambda:grid(*sys.argv[2:4]),'agg':agg}[m]()
