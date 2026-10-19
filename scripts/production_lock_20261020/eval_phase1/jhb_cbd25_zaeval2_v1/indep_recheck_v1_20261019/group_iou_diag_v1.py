"""Diagnostic (not zaeval-2): group IoU. A pred's group = all valid GT polygons it overlaps; group IoU = sum inter/(pred valid + sum gt valid - sum inter).
GT counted hit if some pred with group IoU>=thr overlaps it with >= half its valid px covered by that pred; pred TP if group IoU>=thr. Shows how much of the
zaeval-2 gap is GT granularity (adjacent panel polygons glued by one prediction) rather than detection or boundary."""
import glob,pickle,json,os,sys
sys.argv=['x']; exec(open('aggregate_v1.py').read().split('R={m:')[0].split('for f in (')[0]+open('aggregate_v1.py').read().split("for f in ('indep_results_v1.json','indep_results_v1.md'): assert not os.path.exists(f)\n")[1].split('R={m:')[0])
assert not os.path.exists('group_iou_diag_v1.json')
out={}
for m in ALL:
    for thr in (0.5,):
        tp=fp=hit=n=0
        for f in sorted(glob.glob('parts/*.pkl')):
            D=pickle.load(open(f,'rb')); gv=D['gt_valid']; NG=D['NG']
            kept=[q for q in merge(D['parts'][m],D['edges'][m]) if q[0]*0.0025>=0.25 and q[1]>0]
            hitset=set()
            for npx,nv,it,_ in kept:
                g=[o for o,c in it.items() if gv[o]>0 and c>0]
                si=sum(it[o] for o in g); su=nv+sum(gv[o] for o in g)-si
                if g and si/su>=thr:
                    tp+=1; hitset|={o for o in g if it[o]>=0.5*gv[o]}
                else: fp+=1
            inc=[o for o in range(1,NG+1) if gv[o]>0]; n+=len(inc); hit+=sum(o in hitset for o in inc)
        P=tp/(tp+fp); R=hit/n; out[m]=dict(P=round(P,4),R=round(R,4),F1=round(2*P*R/(P+R),4),pred_tp=tp,pred_fp=fp,gt_hit=f'{hit}/{n}')
json.dump(out,open('group_iou_diag_v1.json','w'),indent=1); print(json.dumps(out,indent=1))
