"""[jhb_union_v1: copy of ct_val_group_diag_v1/score_ct_v1.py + union/maxp/avg2 thresholds + area px (whole tile)]
CT val (14 grids, 1703 tile rows, phase-1 saved uint8 probs) -> parts/ct_GRID.pkl. Labels: pipeline_common_v3 val rows
(v2f manifest f182b9c3) with G1630 swapped to v2g label + v2g masks (as labelfix/rescore_v2g_g1630). View exclude-pixels
(make_view used ONLY to build valid_px/gt_id, i.e. label prep). Models: convnext_t_s0, mit_b2_s2 (only b2 seed with saved val
probs), avg2=(conv+b2s2)/2. usage: score_ct_v1.py GRID"""
import os,sys,pickle,hashlib,numpy as np
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser()))
from pipeline_common_v3 import *  # noqa
import pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from core_v1 import *
G=sys.argv[1]; out=f'parts/ct_{G}.pkl'; os.makedirs('parts',exist_ok=True); assert not os.path.exists(out)
PROBS=Path('~/capetown_eval/ds3m_ft/phase1_followup_valprobs_20261018').expanduser()
NEWL='/home/gao/RA_solar/census/capetown/labels/corrected/ct_labels_corrected_v2g_20261019/Capetown/G1630_SAM2_260324.gpkg'
NEWM=Path('/home/gao/RA_solar/census/capetown/trainsets/ct_seg_trainset_v2g_g1630only_20261019/data/val/G1630')
rows,gm=select_rows('val'); rows,gm,_=validation_scopes(rows); assert len(rows)==1703
rows=rows[rows.grid==G].copy(); gm=gm.copy()
if G=='G1630':
    gm.loc['G1630','label_file']=NEWL; gm.loc['G1630','md5']=hashlib.md5(open(NEWL,'rb').read()).hexdigest()
    rows['mask']=[str(NEWM/Path(p).name) for p in rows['mask']]; assert all(Path(p).is_file() for p in rows['mask'])
gt=load_grid_gt(G,gm)['PV']; NG=int(gt.oid.max()) if len(gt) else 0
geoms={int(o):g for o,g in zip(gt.oid,gt.geometry)}; area={int(o):float(a) for o,a in zip(gt.oid,gt.area_m2)}
MODELS=['convnext_t_s0','mit_b2_s2','avg2','union','maxp','avg2_h082','avg2_h085','avg2_h088']; tiles=[]
for row in rows.to_dict('records'):
    ref=ref_for(pd.Series(row)); view=make_view(class_labels(ref,gt),'exclude-pixels')
    valid=view.valid_px; gid=view.gt_id.astype(np.int64)
    gx0=int(round(ref.transform.c/ref.gsd_m)); gy0=int(round(-ref.transform.f/ref.gsd_m))
    c=np.load(PROBS/'convnext_t_s0'/G/f'{ref.tile_id}.npz')['p']; b=np.load(PROBS/'mit_b2_s2'/G/f'{ref.tile_id}.npz')['p']
    pc=c.astype(np.float32)/255.; pb=b.astype(np.float32)/255.
    masks={'convnext_t_s0':hyst(pc,0.9,0.5),'mit_b2_s2':hyst(pb,0.9,0.5),'avg2':hyst((pc+pb)/2,0.9,0.5)}
    masks['union']=masks['convnext_t_s0']|masks['mit_b2_s2']; masks['maxp']=hyst(np.maximum(pc,pb),0.9,0.5)
    for h in (0.82,0.85,0.88): masks[f'avg2_h{int(round(h*100)):03d}']=hyst((pc+pb)/2,h,0.5)
    del c,b
    t=dict(tile=ref.tile_id,scope=row['scope_v3'],gt_valid=np.bincount(gid[valid].ravel(),minlength=NG+1),gt_px=np.bincount(view.gt_id.ravel(),minlength=NG+1),models={})
    gp=np.bincount(gid.ravel(),minlength=NG+1)
    for m in MODELS:
        t['models'][m]=tile_parts(masks[m],valid,gid,gx0,gy0); t[f'cov_{m}']=np.bincount(gid[masks[m]],minlength=NG+1)
    t['gt_area_px']=int((view.gt_id>0).sum()); t['pred_area_px']={m:int(masks[m].sum()) for m in MODELS}; del masks,pc,pb
    tiles.append(t)
pickle.dump(dict(city='CT',grid=G,NG=NG,geoms=geoms,area=area,crs=str(gt.crs),tiles=tiles,models=MODELS),open(out+'.tmp','wb')); os.replace(out+'.tmp',out); print('DONE',G,len(tiles),flush=True)
