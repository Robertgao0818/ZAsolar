"""Aggregate parts/*.pkl -> own P/R/F1 (IoU 0.3/0.5/0.7), size-bin recall, FN/FP diagnostics, grid bootstrap of F1 diffs,
and compare with ../scores/*__h0.9_0.5.json. Writes indep_results_v1.json / .md (no overwrite)."""
import glob,json,pickle,os,numpy as np
ALL=['convnext_t_s0','mit_b2_s0','v2c','avg3']; THR=[0.3,0.5,0.7]
for f in ('indep_results_v1.json','indep_results_v1.md'): assert not os.path.exists(f)
def merge(parts,edges):
    p=list(range(len(parts)))
    def fd(a):
        while p[a]!=a: p[a]=p[p[a]]; a=p[a]
        return a
    for (x,y),k in edges.items():
        for dx in(-1,0,1):
            for dy in(-1,0,1):
                j=edges.get((x+dx,y+dy))
                if j is not None and j!=k:
                    a,b=fd(j),fd(k)
                    if a!=b: p[b]=a
    g={}
    for i in range(len(parts)): g.setdefault(fd(i),[]).append(i)
    out=[]
    for mem in g.values():
        npx=sum(parts[i][0] for i in mem); nv=sum(parts[i][1] for i in mem); it={}
        for i in mem:
            for o,c in parts[i][2].items(): it[o]=it.get(o,0)+c
        out.append((npx,nv,it,len(mem)))
    return out
def binof(a): return 'lt4' if a<4 else ('4_10' if a<10 else 'ge10')
R={m:{'grids':{}} for m in ALL}; GLOB={}
for f in sorted(glob.glob('parts/*.pkl')):
    D=pickle.load(open(f,'rb')); G=D['grid']; NG=D['NG']
    gv=D['gt_valid']; inc=[o for o in range(1,NG+1) if gv[o]>0]
    GLOB[G]=dict(n_poly=NG,n_gt_valid=len(inc),n_gt_px=int((D['gt_px'][1:]>0).sum()),n_gt_px_alltouched=int((D['gt_px_at'][1:]>0).sum()),diag={k:(sorted(v) if isinstance(v,set) else v) for k,v in D['diag'].items()})
    for m in ALL:
        mp=merge(D['parts'][m],D['edges'][m]); kept=[q for q in mp if q[0]*0.0025>=0.25 and q[1]>0]
        pairs=[]
        for pi,(npx,nv,it,_) in enumerate(kept):
            for o,c in it.items():
                if gv[o]>0 and c>0: pairs.append((c/(nv+gv[o]-c),pi,o))
        pairs.sort(key=lambda t:(-t[0],t[1],t[2]))
        res={}
        for thr in THR:
            up=set();ug={};
            for iou,pi,o in pairs:
                if iou<thr: break
                if pi in up or o in ug: continue
                up.add(pi); ug[o]=pi
            tp=len(ug); bins={b:[0,0] for b in('lt4','4_10','ge10')}
            for o in inc:
                b=binof(D['area'][o-1]); bins[b][0]+=1; bins[b][1]+=o in ug
            res[str(thr)]=dict(tp=tp,fp=len(kept)-len(up),fn=len(inc)-tp,bins=bins)
            if thr==0.5:
                # FN diagnostics
                gtpreds={}
                for iou,pi,o in pairs: gtpreds.setdefault(o,[]).append((iou,pi))
                predgts={}
                for iou,pi,o in pairs: predgts.setdefault(pi,[]).append(o)
                cat=dict(fn_no_overlap=0,fn_low_cov=0,fn_glued=0,fn_split=0,fn_boundary=0)
                for o in inc:
                    if o in ug: continue
                    cov=D['gt_cov'][m][o]/max(D['gt_px'][o],1)
                    if o not in gtpreds: cat['fn_no_overlap']+=1
                    elif cov<0.5: cat['fn_low_cov']+=1
                    else:
                        best=max(gtpreds[o])[1]
                        if len(predgts[best])>=2: cat['fn_glued']+=1
                        elif len(gtpreds[o])>=2: cat['fn_split']+=1
                        else: cat['fn_boundary']+=1
                fpc=dict(fp_no_gt_overlap=0,fp_overlaps_gt=0)
                for pi in range(len(kept)):
                    if pi in up: continue
                    fpc['fp_overlaps_gt' if pi in predgts else 'fp_no_gt_overlap']+=1
                res['diag']=dict(**cat,**fpc,n_pred_glue=sum(len(v)>=2 for v in predgts.values()))
                res['legacy_hit']=int(sum(D['gt_cov'][m][o]/D['gt_px'][o]>=0.5 for o in range(1,NG+1) if D['gt_px'][o]>0)); res['legacy_n']=int((D['gt_px'][1:]>0).sum())
        res['n_pred']=len(kept); R[m]['grids'][G]=res
def prf(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else 0; Rr=tp/(tp+fn) if tp+fn else 0; return P,Rr,(2*P*Rr/(P+Rr) if P+Rr else 0)
grids=sorted(GLOB); out=dict(grids=len(grids),gt=dict(n_poly=sum(GLOB[g]['n_poly'] for g in grids),n_gt_valid=sum(GLOB[g]['n_gt_valid'] for g in grids),n_gt_px=sum(GLOB[g]['n_gt_px'] for g in grids),n_gt_px_alltouched=sum(GLOB[g]['n_gt_px_alltouched'] for g in grids)),models={})
agg={k:sum(GLOB[g]['diag']['int_vs_float_px_diff'] for g in grids) for k in ['x']}
out['quant']=dict(int_vs_float_px_diff=sum(GLOB[g]['diag']['int_vs_float_px_diff'] for g in grids),borderline_comps_max229or230={m:sum(GLOB[g]['diag']['borderline_comps'][m] for g in grids) for m in ALL[:3]},
  gate_violation_px={m:sum(GLOB[g]['diag']['gate_violation_px'][m] for g in grids) for m in ALL[:3]},zero_tiles={m:sum(GLOB[g]['diag']['zero_tiles'][m] for g in grids) for m in ALL[:3]},gate_zero_tiles=sum(GLOB[g]['diag']['gate_zero_tiles'] for g in grids),ntiles=sum(GLOB[g]['diag']['ntiles'] for g in grids))
for m in ALL:
    M={}
    for thr in map(str,THR):
        tp,fp,fn=[sum(R[m]['grids'][g][thr][k] for g in grids) for k in('tp','fp','fn')]; P,Rr,F=prf(tp,fp,fn)
        bins={b:[sum(R[m]['grids'][g][thr]['bins'][b][i] for g in grids) for i in(1,0)] for b in('lt4','4_10','ge10')}
        M[thr]=dict(P=round(P,4),R=round(Rr,4),F1=round(F,4),tp=tp,fp=fp,fn=fn,bins={b:f'{v[0]}/{v[1]}' for b,v in bins.items()})
    M['diag']={k:sum(R[m]['grids'][g]['diag'][k] for g in grids) for k in R[m]['grids'][grids[0]]['diag']}
    M['legacy_recall']=f"{sum(R[m]['grids'][g]['legacy_hit'] for g in grids)}/{sum(R[m]['grids'][g]['legacy_n'] for g in grids)}"
    M['n_pred']=sum(R[m]['grids'][g]['n_pred'] for g in grids)
    try:
        J=json.load(open(f'../scores/{m}__h0.9_0.5.json'))['objects']; M['theirs']={t:{k:J['by_iou'][t][k] for k in('tp','fp','fn')} for t in('0.3','0.5','0.7')}; M['theirs_n_pred']=J['n_pred']; M['theirs_n_gt']=J['n_gt']
        M['theirs_per_grid_match_0.5']=sum(all(J['per_grid'][g]['by_iou']['0.5'][k]==R[m]['grids'][g]['0.5'][k] for k in('tp','fp','fn')) for g in grids)
    except Exception as e: M['theirs_err']=str(e)
    out['models'][m]=M
# grid bootstrap
rng=np.random.default_rng(20261019); B=10000
arr={m:np.array([[R[m]['grids'][g]['0.5'][k] for k in('tp','fp','fn')] for g in grids]) for m in ALL}
def f1(a): tp,fp,fn=a[...,0],a[...,1],a[...,2]; return 2*tp/(2*tp+fp+fn)
idx=rng.integers(0,len(grids),(B,len(grids)))
bs={m:f1(arr[m][idx].sum(1)) for m in ALL}
out['bootstrap']={'n_rep':B,'unit':'grid (25)','iou':0.5}
for a,b in (('mit_b2_s0','convnext_t_s0'),('v2c','mit_b2_s0'),('avg3','mit_b2_s0'),('v2c','convnext_t_s0'),('avg3','v2c')):
    d=bs[a]-bs[b]; pt=f1(arr[a].sum(0))-f1(arr[b].sum(0))
    out['bootstrap'][f'{a}-{b}']=dict(diff=round(float(pt),4),ci95=[round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)],p_le0=round(float((d<=0).mean()),4))
for m in ALL: out['bootstrap'][f'F1_{m}_ci95']=[round(float(np.percentile(bs[m],q)),4) for q in(2.5,97.5)]
out['per_grid_gt']=GLOB
json.dump(out,open('indep_results_v1.json','w'),indent=1,default=int)
print(json.dumps({k:v for k,v in out.items() if k!='per_grid_gt'},indent=1,default=int))
