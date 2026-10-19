"""G14 final tables from parts/*.pkl (assoc_grid_v2). Building installed iff assigned PV >= 2 m². GT reference = original labels (gt0.0).
Metrics per city×model: building P/R/F1 + Wilson 95% (P on predicted-installed, R on GT-installed); count error (pred−GT)/GT vs ±5%;
per stratum (CT basemap stratum / JHB H3-7) with n_gt>=20 vs ±10%; grid bootstrap (2000, seed 0) F1 CI and diffs vs convnext;
CT erf-level (parcel_key, sum of its eval buildings >= 2 m²); area: PV in region, assigned, off-building share by class.
Row-split: GT labels merged at 0.25 m and 1 m gap -> Δ GT installed buildings, Δ GT polygon count, model F1 vs merged GT.
Object F1 reference: ct_val_group_diag_v1_20261019/results_v1.json std_iou05 '|merge'. Writes RESULTS_v1.json/.md, results_*.csv"""
import glob,pickle,json,numpy as np,pandas as pd
from g14_lib_v1 import wilson, THR
OBJ=json.load(open('/home/gao/capetown_eval/ct_val_group_diag_v1_20261019/results_v1.json'))
OK={'CT':{'convnext':'convnext_t_s0','b2':'mit_b2_s2','avg2':'avg2'},'JHB':{'convnext':'convnext_t_s0','b2':'mit_b2_s0','v2c':'v2c','avg2':'avg2'}}
def prf(g,p):
    tp=int((g&p).sum()); fp=int((~g&p).sum()); fn=int((g&~p).sum())
    return tp,fp,fn,(tp/(tp+fp) if tp+fp else 0.0),(tp/(tp+fn) if tp+fn else 0.0),(2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.0)
rng=np.random.default_rng(0); OUT={}; rows=[]; rs=[]; strat=[]; area=[]
for city in ['CT','JHB']:
    parts=[pickle.load(open(f,'rb')) for f in sorted(glob.glob(f'parts/{city}_*.pkl'))]
    B=pd.concat([p['buildings'] for p in parts],ignore_index=True); grids=sorted(B.grid.unique()); M=list(OK[city])
    gi=B['gt0.0']>=THR; skey='stratum' if city=='CT' else 'h3_r7'
    gidx={g:np.nonzero((B.grid==g).values)[0] for g in grids}; boots=[np.concatenate([gidx[g] for g in rng.choice(grids,len(grids))]) for _ in range(2000)]
    bf1={}
    for m in M:
        pi=B[m]>=THR; tp,fp,fn,P,R,F=prf(gi,pi); ng,npd=int(gi.sum()),int(pi.sum()); ce=(npd-ng)/ng
        bf1[m]=np.array([prf(gi.values[b],pi.values[b])[5] for b in boots])
        o=OBJ[city]['models'][f'{OK[city][m]}|merge']['std_iou05']
        r=dict(city=city,model=m,n_bld_eval=len(B),gt_installed=ng,pred_installed=npd,tp=tp,fp=fp,fn=fn,P=round(P,3),P_ci=wilson(tp,tp+fp),R=round(R,3),R_ci=wilson(tp,tp+fn),
               F1=round(F,3),F1_boot_ci=tuple(np.round(np.percentile(bf1[m],[2.5,97.5]),3)),count_err=round(ce,4),within_5pct=abs(ce)<=0.05,within_10pct=abs(ce)<=0.10,
               obj_F1_ref=round(o['F1'],3),bld_F1_ge_obj_F1=F>=o['F1'])
        if city=='CT':
            E=B[B.parcel_key.notna()].groupby('parcel_key')[['gt0.0',m]].sum(); eg,ep=E['gt0.0']>=THR,E[m]>=THR; etp,efp,efn,eP,eR,eF=prf(eg,ep)
            r.update(erf_n=len(E),erf_gt=int(eg.sum()),erf_pred=int(ep.sum()),erf_P=round(eP,3),erf_P_ci=wilson(etp,etp+efp),erf_R=round(eR,3),erf_R_ci=wilson(etp,etp+efn),erf_F1=round(eF,3),erf_count_err=round((ep.sum()-eg.sum())/max(eg.sum(),1),4))
        rows.append(r)
        for s,sb in B.groupby(skey):
            n=int((sb['gt0.0']>=THR).sum())
            if n>=20: e=((sb[m]>=THR).sum()-n)/n; strat.append(dict(city=city,model=m,stratum=s,n_gt=n,pred=int((sb[m]>=THR).sum()),count_err=round(e,4),within_10pct=abs(e)<=0.10))
        for g in [0.25,1.0]:
            gm=B[f'gt{g}']>=THR; _,_,_,P2,R2,F2=prf(gm,pi); rs.append(dict(city=city,model=m,gap_m=g,F1_orig=round(F,3),F1_merged=round(F2,3),dF1=round(F2-F,4)))
    for m in M[1:]:
        d=bf1[m]-bf1['convnext']; OUT[f'{city}_boot_{m}-convnext']=dict(diff=round(float(np.median(d)),4),ci95=tuple(np.round(np.percentile(d,[2.5,97.5]),4)))
    for k in ['gt0.0','gt0.25','gt1.0']+M:
        tot=sum(p['summary'][k]['pv_area_in_region'] for p in parts); orph=sum(p['summary'][k]['orphan_area'] for p in parts)
        cls={}; [cls.__setitem__(c,cls.get(c,0)+a) for p in parts for c,a in p['summary'][k]['orphan_by_class'].items()]
        area.append(dict(city=city,source=k,n_polys=sum(p['summary'][k]['n_polys_in_region'] for p in parts),pv_m2=round(tot),assigned_to_eval_bld_m2=round(sum(p['summary'][k]['alloc_to_eval_bld'] for p in parts)),
                         off_bld_share=round(orph/tot,4) if tot else None,**{f'off_{c}':round(a/tot,4) for c,a in sorted(cls.items())}))
    g0=int(gi.sum()); OUT[f'{city}_rowsplit_gt']={str(g):dict(gt_installed=int((B[f'gt{g}']>=THR).sum()),d_pct=round(((B[f'gt{g}']>=THR).sum()-g0)/g0*100,2)) for g in [0.0,0.25,1.0]}
A=pd.DataFrame(area)
for c in ['CT','JHB']:
    OUT[f'{c}_rowsplit_gt'].update({f'n_polys_gap{g}':int(A[(A.city==c)&(A.source==f'gt{g}')].n_polys.iloc[0]) for g in [0.0,0.25,1.0]})
    OUT[f'{c}_gt_off_bld_share']=float(A[(A.city==c)&(A.source=='gt0.0')].off_bld_share.iloc[0])
D=pd.DataFrame(rows); S=pd.DataFrame(strat); RS=pd.DataFrame(rs)
for n,x in [('results_building_v1.csv',D),('results_stratum_v1.csv',S),('results_rowsplit_v1.csv',RS),('results_area_v1.csv',A)]: x.to_csv(n,index=False)
json.dump(dict(main=rows,extra=OUT),open('RESULTS_v1.json','w'),indent=1,default=str)
with open('RESULTS_v1.md','w') as f:
    for t,x in [('Building level',D),('Strata n_gt>=20',S),('Row split',RS),('Area',A)]: f.write(f'## {t}\n'+x.to_markdown(index=False)+'\n\n')
    f.write('## extra\n```\n'+json.dumps(OUT,indent=1,default=str)+'\n```\n')
pd.set_option('display.width',250); print(D.to_string()); print(S.groupby(['city','model']).within_10pct.agg(['sum','size'])); print(RS.to_string()); print(A.to_string()); print(json.dumps(OUT,indent=1,default=str))
