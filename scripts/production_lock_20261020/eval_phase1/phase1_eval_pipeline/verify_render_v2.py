#!/usr/bin/env python3
"""Verify completed PNG artifacts and create a five-grid visual QA overview."""
import json,time
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'test_run_full_gpu_v2'
def main():
    started=time.perf_counter();out=BASE/'overlays'
    m=json.loads((out/'render_manifest.json').read_text())
    assert not m['partial'] and len(m['tiles'])==632 and len(m['full_grid_overlays'])==5
    files=[Path(x['png']) for x in m['tiles']]+[Path(x['path']) for x in m['full_grid_overlays']]+list(out.glob('G*_contact_sheet.png'))
    assert len(files)==642 and len(set(files))==642
    for p in files:
        with Image.open(p) as image:image.verify()
    canvas=Image.new('RGB',(1800,2700),(25,28,32));draw=ImageDraw.Draw(canvas)
    for i,rec in enumerate(m['full_grid_overlays']):
        with Image.open(rec['path']) as img:
            assert list(img.size)==rec['size'];img.thumbnail((900,850),Image.Resampling.LANCZOS)
            x=(i%2)*900;y=(i//2)*900
            canvas.paste(img,(x,y+30));draw.text((x+10,y+8),rec['grid']+' | complete footprint | '+str(rec['tiles'])+' native tiles',fill='white')
    dest=ROOT/'full_grid_contact_sheet_v2.png'
    with dest.open('xb') as f:canvas.save(f,format='PNG')
    result={'verified_png_files':len(files),'tile_overlays':632,'contact_sheets':5,'full_grid_overlays':5,
        'full_grid_metadata':m['full_grid_overlays'],'qa_overview':str(dest),'runtime_s':time.perf_counter()-started}
    with (ROOT/'render_integrity_v2.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
