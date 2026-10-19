import pandas as pd,rasterio,numpy as np,os,sys
B=os.path.expanduser('~/data/ZAsolar/geid_temporal/')
for v in ['ct_seg_trainset_v2d1_20261016','ct_seg_trainset_v2d2_20261017']:
    M=pd.read_csv(B+v+'/data/manifest.csv'); T=M[M.split=='train']
    s=0
    for p in T['mask']:
        with rasterio.open(p) as f: s+=int((f.read(1)==1).sum())
    print(v,'rows',len(M),'train_tiles',len(T),'manifest_npv',int(T.n_pv.sum()),'recount_pv',s, 'grids3',T[T.grid.isin(['G1189','G1190','G1238'])].shape[0])
