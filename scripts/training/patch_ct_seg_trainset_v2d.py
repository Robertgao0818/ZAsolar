"""CT seg trainset v2c -> v2d: label fixes only (2026-10-16).

1. Grids whose source labels changed (fix_ct_seg_labels_v2d.py) are rebuilt with
   build_ct_seg_trainset_v2.py (same args as v2/v2b, --labels-dir = corrected copy,
   --grids <changed>) and patch_ct_seg_trainset_v2c.py; their masks, manifest rows and
   chip rows replace the v2c ones (v2c split and paths kept).
2. Thin slivers in ignore_slivers.gpkg are painted 255 where the v2c mask is PV
   (geometry buffered 0.05 m); counts updated.
Everything else is copied from v2c with shutil.copy2 (no links). v2c is never written.
"""
import argparse,glob,json,os,shutil
import numpy as np,pandas as pd,geopandas as gpd,rasterio
from rasterio.features import rasterize
from scipy import ndimage
def cnt(m):
    c=np.bincount(m.ravel(),minlength=256); return dict(n_bg=int(c[0]),n_pv=int(c[1]),n_swh=int(c[2]),n_ignore=int(c[255]))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v2c',required=True);ap.add_argument('--sub',required=True,help='v2c-patched rebuild of changed grids')
    ap.add_argument('--slivers',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    if os.path.exists(a.out): raise SystemExit('refusing to overwrite '+a.out)
    shutil.copytree(a.v2c,a.out,copy_function=shutil.copy2)
    M=pd.read_csv(f'{a.out}/manifest.csv'); CH=pd.read_csv(f'{a.out}/chips_512px_index.csv')
    src=os.path.abspath(a.v2c); dst_root=os.path.abspath(a.out)
    for T in (M,CH):  # point every row at the v2d copy; never write through a v2c path
        T['mask']=[x.replace(src,dst_root,1) for x in T['mask']]
        assert all(x.startswith(dst_root) for x in T['mask'])
    SM=pd.read_csv(f'{a.sub}/manifest.csv'); SC=pd.read_csv(f'{a.sub}/chips_512px_index.csv')
    split=M.groupby('grid').split.first().to_dict(); stats=[]
    rp=lambda p,g:os.path.join(a.out,split[g],g,os.path.basename(p))
    def comps(m): return ndimage.label(m==1)[1]
    for g in sorted(SM.grid.unique()):
        for p in SM[SM.grid==g]['mask']:
            dst=rp(p,g); old=rasterio.open(dst).read(1); new=rasterio.open(p).read(1)
            stats.append(dict(grid=g,split=split[g],tile=os.path.basename(dst),changed_px=int((old!=new).sum()),pv_px_v2c=int((old==1).sum()),pv_px_v2d=int((new==1).sum()),obj_v2c=comps(old),obj_v2d=comps(new),fix='label_rebuild'))
            os.remove(dst); shutil.copy2(p,dst)
        sm=SM[SM.grid==g].copy(); sm['split']=split[g]; sm['mask']=[rp(p,g) for p in sm['mask']]
        sc=SC[SC.grid==g].copy(); sc['split']=split[g]; sc['mask']=[rp(p,g) for p in sc['mask']]
        sm=sm[M.columns.intersection(sm.columns)]; sc=sc[CH.columns.intersection(sc.columns)]
        M=pd.concat([M[M.grid!=g],sm],ignore_index=True); CH=pd.concat([CH[CH.grid!=g],sc],ignore_index=True)
    sl=gpd.read_file(a.slivers)
    for g,gg in sl.groupby('grid'):
        for i in M.index[M.grid==g]:
            p=M.at[i,'mask']; assert os.path.abspath(p).startswith(dst_root)
            with rasterio.open(p) as f: m=f.read(1); prof=f.profile; geo=gg.to_crs(f.crs).buffer(0.05)
            hit=rasterize([(x,1) for x in geo],out_shape=m.shape,transform=prof['transform']).astype(bool)&(m==1)
            if not hit.any(): continue
            new=m.copy(); new[hit]=255
            stats.append(dict(grid=g,split=split[g],tile=os.path.basename(p),changed_px=int(hit.sum()),pv_px_v2c=int((m==1).sum()),pv_px_v2d=int((new==1).sum()),obj_v2c=comps(m),obj_v2d=comps(new),fix='sliver_to_ignore'))
            os.remove(p)
            with rasterio.open(p,'w',**prof) as d: d.write(new,1)
            for k,v in cnt(new).items(): M.at[i,k]=v
            for j in CH.index[CH['mask']==p]:
                x,y,s=CH.at[j,'x'],CH.at[j,'y'],CH.at[j,'size']; w=new[y:y+s,x:x+s]
                CH.at[j,'n_pv']=int((w==1).sum()); CH.at[j,'n_swh']=int((w==2).sum())
    M.sort_values(['split','grid','mask']).to_csv(f'{a.out}/manifest.csv',index=False)
    CH.sort_values(['split','grid','mask','y','x']).to_csv(f'{a.out}/chips_512px_index.csv',index=False)
    st=pd.DataFrame(stats); st.to_csv(f'{a.out}/v2d_tile_diff.csv',index=False)
    s=json.load(open(f'{a.out}/summary.json'))
    s['v2d']=dict(from_v2c=a.v2c,label_fix_rebuild_grids=sorted(SM.grid.unique()),sliver_grids=sorted(sl.grid.unique()),
        pv_px=M.groupby('split').n_pv.sum().astype(int).to_dict(),chips=CH.groupby('split').size().astype(int).to_dict())
    json.dump(s,open(f'{a.out}/summary.json','w'),indent=1,default=str)
    print(st.groupby(['grid','split','fix'])[['changed_px','pv_px_v2c','pv_px_v2d','obj_v2c','obj_v2d']].sum())
if __name__=='__main__': main()
