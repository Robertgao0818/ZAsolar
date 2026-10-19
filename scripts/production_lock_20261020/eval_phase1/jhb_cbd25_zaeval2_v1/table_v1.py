"""Summary table from scores/*.json: zaeval-2 one-to-one IoU>=0.5 object P/R/F1, pixel IoU (valid px, exclude-pixels),
recall <4 / 4-10 m^2 (IoU 0.5), plus legacy_cov recall/precision for parity with G6. Writes RESULTS_v1.csv/.md"""
import json,glob,pandas as pd
R=[]
for f in sorted(glob.glob('scores/*.json')):
    d=json.load(open(f)); o=d['objects']; b=o['by_iou']['0.5']; L=o['legacy_cov']
    R.append(dict(model=d['model'],cfg=d['cfg'],P=round(b['precision'],3),R=round(b['recall'],3),F1=round(b['f1'],3),px_IoU=round(d['pixel']['iou'],3),
      tp=b['tp'],fp=b['fp'],fn=b['fn'],n_gt=o['n_gt'],n_pred=o['n_pred'],
      rec_lt4=f"{round(b['bins']['lt4']['recall'],3)} ({b['bins']['lt4']['hits']}/{b['bins']['lt4']['n']})",
      rec_4_10=f"{round(b['bins']['4_10']['recall'],3)} ({b['bins']['4_10']['hits']}/{b['bins']['4_10']['n']})",
      F1_iou03=round(o['by_iou']['0.3']['f1'],3),legacy_P=round(L['precision'],3),legacy_R=round(L['recall'],3),legacy_F1=round(L['f1'],3)))
D=pd.DataFrame(R); D.to_csv('RESULTS_v1.csv',index=False); open('RESULTS_v1.md','w').write(D.to_markdown(index=False)); print(D.to_string())
