"""zaeval-2 @ 8044d14 (pinned worktree ~/projects/zasolar_evalfix_8044d14_wt, clean) object/pixel scoring of saved Joburg CBD25 probs.
GT: joburg25_vexcel/clean_gt/*_clean_gt.gpkg, loaded exactly as G6 load_gt (buffer(0), dedupe_polygons), EPSG:32735, cls PV.
Tile mask built as the G6 evalobj (1 = GT polygon pixel-centre raster, 255 = 6-px inner/outer polygon band, 255 = outside grid polygon or
nodata); gt_id = pixel-centre polygon raster (all_touched=False, as CT pipeline_common_v2.class_labels). Ignore mode 'exclude-pixels',
ObjectAccumulator(iou_thrs=(0.3,0.5,0.7)) as CT eval_phase1_v3. Binarize = zaeval.postproc.binarize. Pregate is already applied in the probs.
usage: score_z2_v1.py MODEL CFG   MODEL in convnext_t_s0|mit_b2_s0|v2c|avg3 (avg3 = mean of the three uint8/255 maps). Writes scores/MODEL__CFG.json"""
import os,sys,glob,json,subprocess,numpy as np,geopandas as gpd,rasterio
from rasterio import features
from scipy import ndimage
Z=os.path.expanduser('~/projects/zasolar_evalfix_8044d14_wt'); sys.path.insert(0,Z)
import zaeval
from zaeval.objects import ObjectAccumulator
from zaeval.ignore import make_view
from zaeval.postproc import binarize
from zaeval.labelsets.common import grid_gt_from_rows
from zaeval.runner import to_jsonable
from zaeval.types import PredTile,TileLabels,TileRef
head=subprocess.check_output(['git','-C',Z,'rev-parse','HEAD'],text=True).strip(); assert head.startswith('8044d14'),head
assert subprocess.check_output(['git','-C',Z,'diff','HEAD','--','zaeval'],text=True)=='' and zaeval.METRIC_DEF_VERSION=='zaeval-2'
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar_segv1_wt/scripts/training'))
from build_ct_seg_trainset_v1 import dedupe_polygons
MODEL,CFG=sys.argv[1:3]; MODELS=['convnext_t_s0','mit_b2_s0','v2c']
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); J0=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg5')
GRIDPOLY=gpd.read_file(f'{J0}/grids_32735.gpkg').set_index('gridcell_id').geometry
out=f'scores/{MODEL}__{CFG}.json'; os.makedirs('scores',exist_ok=True); assert not os.path.exists(out),'exists; no overwrite'
def load_prob(g,t):
    if MODEL=='avg3': return np.mean([np.load(f'probs/{m}/{g}/{t}.npz')['p'].astype(np.float32)/255. for m in MODELS],0).astype(np.float32)
    return np.load(f'probs/{MODEL}/{g}/{t}.npz')['p'].astype(np.float32)/255.
obj=ObjectAccumulator(iou_thrs=(0.3,0.5,0.7)); tables={}; pix=dict(tp=0,fp=0,fn=0,n_valid=0); st=ndimage.generate_binary_structure(2,1)
P=sorted(glob.glob(f'{J}/tiles005/*/*.tif')); assert len(P)==2500
if os.environ.get('SMOKE_GRID'): P=[p for p in P if f"/{os.environ['SMOKE_GRID']}/" in p]; out=out.replace('scores/','scores_smoke/'); os.makedirs('scores_smoke',exist_ok=True)
for g in sorted({os.path.basename(os.path.dirname(p)) for p in P}):
    d=gpd.read_file(f'{J}/clean_gt/{g}_clean_gt.gpkg').to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0); d=dedupe_polygons(d.reset_index(drop=True))
    gt=grid_gt_from_rows([dict(cls='PV',geometry=q) for q in d.geometry],'EPSG:32735'); tables[g]=gt
    for p in [p for p in P if f'/{g}/' in p]:
        with rasterio.open(p) as s: img=s.read([1,2,3]); tr=s.transform; b=s.bounds
        H,W=img.shape[1:]; t=os.path.basename(p)[:-4]
        ref=TileRef(t,g,tr,'EPSG:32735',(H,W),0.05,None,p,'jhb')
        gs=gt.cx[b.left:b.right,b.bottom:b.top]
        pvm=features.rasterize([(q,1) for q in gs.geometry],(H,W),transform=tr).astype(bool) if len(gs) else np.zeros((H,W),bool)
        ids=features.rasterize([(q,int(o)) for q,o in zip(gs.geometry,gs.oid)],(H,W),transform=tr,dtype='int32',all_touched=False) if len(gs) else np.zeros((H,W),np.int32)
        valid=features.rasterize([(GRIDPOLY[g],1)],(H,W),transform=tr).astype(bool)&img.any(0)
        band=ndimage.binary_dilation(pvm,st,iterations=6)&~ndimage.binary_erosion(pvm,st,iterations=6)
        mk=np.zeros((H,W),np.uint8); mk[pvm]=1; mk[band]=255; mk[~valid]=255
        tl=TileLabels(ref,mk,ids.astype(np.int32)); view=make_view(tl,'exclude-pixels')
        pred=binarize(load_prob(g,t),CFG).astype(bool); tgt=mk==1; v=view.valid_px
        pix['tp']+=int((pred&tgt&v).sum()); pix['fp']+=int((pred&~tgt&v).sum()); pix['fn']+=int((~pred&tgt&v).sum()); pix['n_valid']+=int(v.sum())
        obj.add_tile(ref,PredTile(ref,mask=pred),view)
    print(MODEL,CFG,g,flush=True)
den=pix['tp']+pix['fp']+pix['fn']; pix['iou']=pix['tp']/den if den else None
doc=dict(metric_def_version=zaeval.METRIC_DEF_VERSION,scorer_commit=head,model=MODEL,cfg=CFG,ignore_mode='exclude-pixels',n_tiles=len(P),pixel=pix,objects=obj.finalize(tables))
json.dump(to_jsonable(doc),open(out+'.tmp','w'),indent=1); os.replace(out+'.tmp',out); print('DONE',out)
