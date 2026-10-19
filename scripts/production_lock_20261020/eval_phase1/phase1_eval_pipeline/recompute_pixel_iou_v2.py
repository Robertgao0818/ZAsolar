#!/usr/bin/env python3
"""Independent saved-mask PV/SWH IoU + georeferencing check; no pipeline/scorer imports."""
import argparse,json,re,time
from pathlib import Path
import numpy as np
import rasterio
ALLOWED={'G1293','G1354','G1410','G1523','G1629','G1630','G1689','G1691','G1743','G1798','G1863','G1920','G1979','G2028','G1349','G1686','G1692','G1800','G2038'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--eval-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    start=time.perf_counter();base=Path(a.eval_dir).expanduser().resolve();out=Path(a.out).expanduser().resolve()
    assert not out.exists() and out.is_relative_to(base)
    metrics=json.loads((base/'metrics.json').read_text());entries=json.loads((base/'predictions.json').read_text())
    totals={};grids={};unique=set();hist=[0,0,0]
    for e in entries:
        assert e['grid'] in ALLOWED and e['split'] in {'val','eval'}
        gt_path=Path(e['gt']).resolve();pred_path=Path(e['pred']).resolve()
        assert 'test' not in str(gt_path).lower() and not re.search(r'coord_cache[^/]*\.npz',str(gt_path))
        assert pred_path.is_relative_to(base)
        key=(e['split'],e['grid'],str(gt_path));assert key not in unique;unique.add(key)
        with rasterio.open(gt_path) as gs,rasterio.open(pred_path) as ps:
            assert gs.shape==ps.shape and gs.transform==ps.transform and gs.crs==ps.crs
            assert ps.count==1 and ps.dtypes[0]=='uint8' and ps.nodata is None
            gt=gs.read(1);pred=ps.read(1)
        assert set(np.unique(gt))<={0,1,2,255} and set(np.unique(pred))<={0,1,2}
        hist=[h+int(np.count_nonzero(pred==v)) for v,h in enumerate(hist)]
        valid=gt!=255
        for name,value in [('PV',1),('SWH',2)]:
            if name=='SWH' and not e['swh_labelled']:continue
            target=gt==value;positive=pred==value
            c={'tp':int(np.count_nonzero(valid & target & positive)),
               'fp':int(np.count_nonzero(valid & ~target & positive)),
               'fn':int(np.count_nonzero(valid & target & ~positive)),
               'n_valid':int(np.count_nonzero(valid)),'n_tiles':1}
            for collection,k in [(totals,(e['split'],name)),(grids,(e['split'],e['grid'],name))]:
                t=collection.setdefault(k,dict.fromkeys(c,0))
                for n,v in c.items():t[n]+=v
    errors=[]
    for (split,name),c in totals.items():
        c['union']=c['tp']+c['fp']+c['fn'];c['iou']=c['tp']/c['union'] if c['union'] else None
        expected=metrics['splits'][split][name]['pixel']
        assert all(c[k]==expected[k] for k in ('tp','fp','fn','n_valid','union'))
        assert c['n_tiles']==metrics['splits'][split][name]['n_tiles']
        c['absolute_error']=abs(c['iou']-expected['iou']) if c['iou'] is not None else 0
        assert c['absolute_error']<=1e-12
    for c in grids.values():
        c['union']=c['tp']+c['fp']+c['fn'];c['iou']=c['tp']/c['union'] if c['union'] else None
    result={'independent_of_pipeline_and_scorer':True,'matched':True,'checked_saved_masks':len(entries),
      'georeferencing_and_encoding_checked':True,'prediction_pixel_histogram_0_1_2':hist,
      'splits':{s:{n:c for (sp,n),c in totals.items() if sp==s} for s in sorted({s for s,n in totals})},
      'by_grid':{'/'.join(k):v for k,v in grids.items()},'runtime_s':time.perf_counter()-start}
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='by_grid'},indent=2))

if __name__=='__main__':main()
