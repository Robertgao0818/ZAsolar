"""CT seg trainset v2d1 -> v2d2: add new train grids (2026-10-17).

New grids (葉's earliest hand-SAM grids G1189/G1190/G1238, alignment to aerial_2025
confirmed by the owner 2026-10-17) are built with build_ct_seg_trainset_v2.py (same args,
--labels-dir = v2d1 corrected labels, --exclude G1807, --grids <new>) and
patch_ct_seg_trainset_v2c.py. Their masks, manifest rows, chip rows, collectors and
hard-negative rows are appended as split=train. Every existing row/file is copied from
v2d1 unchanged (shutil.copy2, no links); v2d1 is never written. Grids already present
in v2d1 are refused, so the val/eval/test split cannot change.
"""
import argparse,json,os,shutil
import pandas as pd
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v2d1',required=True); ap.add_argument('--sub',required=True,help='v2c-patched build of the new grids')
    ap.add_argument('--out',required=True); ap.add_argument('--heldout-csv',required=True); a=ap.parse_args()
    if os.path.exists(a.out): raise SystemExit('refusing to overwrite '+a.out)
    shutil.copytree(a.v2d1,a.out,copy_function=shutil.copy2)
    src=os.path.abspath(a.v2d1); dst=os.path.abspath(a.out)
    M=pd.read_csv(f'{a.out}/manifest.csv'); CH=pd.read_csv(f'{a.out}/chips_512px_index.csv')
    for T in (M,CH):
        T['mask']=[x.replace(src,dst,1) for x in T['mask']]; assert all(x.startswith(dst) for x in T['mask'])
    SM=pd.read_csv(f'{a.sub}/manifest.csv'); SC=pd.read_csv(f'{a.sub}/chips_512px_index.csv')
    new=sorted(SM.grid.unique()); assert not set(new)&set(M.grid), 'grid already in v2d1'
    held={('G'+str(c)[3:]) if str(c).startswith('CPT') else str(c) for c in pd.read_csv(a.heldout_csv).cpt_id}; assert not set(new)&held, 'held-out grid'
    def rp(p,g):
        d=os.path.join(a.out,'train',g); os.makedirs(d,exist_ok=True); return os.path.join(d,os.path.basename(p))
    for p,g in zip(SM['mask'],SM.grid): shutil.copy2(p,rp(p,g))
    sm=SM.copy(); sm['split']='train'; sm['mask']=[rp(p,g) for p,g in zip(sm['mask'],sm.grid)]
    sc=SC.copy(); sc['split']='train'; sc['mask']=[rp(p,g) for p,g in zip(sc['mask'],sc.grid)]
    M=pd.concat([M,sm[M.columns.intersection(sm.columns)]],ignore_index=True); CH=pd.concat([CH,sc[CH.columns.intersection(sc.columns)]],ignore_index=True)
    M.sort_values(['split','grid','mask']).to_csv(f'{a.out}/manifest.csv',index=False)
    CH.sort_values(['split','grid','mask','y','x']).to_csv(f'{a.out}/chips_512px_index.csv',index=False)
    for f in ('collectors.csv','hard_negatives.csv','grids_swh_labelled.csv'):
        if os.path.exists(f'{a.sub}/{f}') and os.path.getsize(f'{a.sub}/{f}')>1:
            o=pd.read_csv(f'{a.out}/{f}'); n=pd.read_csv(f'{a.sub}/{f}')
            if 'split' in n.columns: n['split']='train'
            pd.concat([o,n[o.columns.intersection(n.columns)]],ignore_index=True).to_csv(f'{a.out}/{f}',index=False)
    s=json.load(open(f'{a.out}/summary.json'))
    s['v2d2']=dict(from_v2d1=a.v2d1,added_train_grids=new,added_tiles=int(len(sm)),added_chips=int(len(sc)),added_pv_px=int(sm.n_pv.sum()),
        swh_labelled={g:bool(v) for g,v in sm.groupby('grid').swh_labelled.first().items()} if 'swh_labelled' in sm else None,
        pv_px=M.groupby('split').n_pv.sum().astype(int).to_dict(),chips=CH.groupby('split').size().astype(int).to_dict())
    json.dump(s,open(f'{a.out}/summary.json','w'),indent=1); print(json.dumps(s['v2d2'],indent=1))
if __name__=='__main__': main()
