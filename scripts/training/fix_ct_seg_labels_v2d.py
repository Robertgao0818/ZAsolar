"""CT seg label fixes for trainset v2d (2026-10-16). Writes NEW files only.

Inputs are read-only: data/annotations/Capetown/G*_SAM2_*.gpkg, the SAM3 text
pilot v1 (thin_strip.csv, sam3_<grid>.gpkg, work/compare/human_<grid>.gpkg,
work/gemini/agreement.csv) and pilot v2 long_strips.csv + work/human/human_<grid>.gpkg.

Fixes (owner decisions, 葉 2026-10-16):
 A. G1630 segment_id 130 (corrupt GeometryCollection, vertex at lat~0) is dropped
    and replaced by the SAM3 polygon sG1630_pv_00084.
 B. thin_strip cases of pilot v1: the human source polygon(s) of that array are
    replaced by the SAM3 (green) polygon. A source polygon is dropped when
    >=50% of its area lies inside the SAM3 polygon buffered 0.5 m. Cases with
    cover<0.5, no SAM3 match, or Gemini 2/2 label != pv are skipped and listed.
    Gemini full=no / nan cases are applied but flagged.
 C. Zero-area / non-polygonal geometries are dropped (G1682 idx 48-51).
 D. Thin slivers (pilot v2 long_strips, width<0.7 m, not G1630) are written to
    ignore_slivers.gpkg; the v2d patch paints them 255.
Row indices of the source files are preserved (dropped = empty geometry) because
solar_water_heaters.gpkg references label rows by '<file>#<row>'.
Output: <out>/Capetown/ (full copy, changed grids rewritten), changes.csv, skipped.csv,
ignore_slivers.gpkg.
"""
import argparse,glob,os,shutil
import numpy as np,pandas as pd,geopandas as gpd
from shapely.ops import unary_union
P1='/home/gao/zasolar_data/geid_temporal/ct_seg_sam3text_pilot_v1_20261016'
P2='/home/gao/zasolar_data/geid_temporal/ct_seg_sam3text_pilot_v2_20261016'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--labels-dir',default='/home/gao/projects/ZAsolar/data/annotations/Capetown')
    ap.add_argument('--out',required=True);a=ap.parse_args()
    out=os.path.join(a.out,'Capetown')
    if os.path.exists(out): raise SystemExit(f'refusing to overwrite {out}')
    shutil.copytree(a.labels_dir,out,copy_function=shutil.copy2)
    files={os.path.basename(f).split('_')[0]:f for f in glob.glob(f'{a.labels_dir}/G*_SAM2_*.gpkg')}
    ts=pd.read_csv(f'{P1}/thin_strip.csv'); ag=pd.read_csv(f'{P1}/work/gemini/agreement.csv')
    ag=ag[ag.stratum=='thin_strip'].set_index('human_id')
    changes,skipped=[],[]; cases={}
    for r in ts.itertuples():
        gem=ag.loc[r.human_id] if r.human_id in ag.index else None
        glab=None if gem is None else gem.label_2of2; gfull=None if gem is None else gem.full_2of2
        why=None
        if not isinstance(r.sam3_id,str): why='no_sam3_match'
        elif not (r.cover>=0.5): why=f'cover<{0.5} ({r.cover:.2f})'
        elif glab is not None and glab!='pv': why=f'gemini_label={glab}'
        if why: skipped.append(dict(grid=r.grid,human_id=r.human_id,sam3_id=r.sam3_id,reason=why,gemini_label=glab,gemini_full=gfull)); continue
        c=cases.setdefault((r.grid,r.sam3_id),dict(hids=[],gem=[]))
        c['hids'].append(r.human_id); c['gem'].append('not_judged' if gem is None else f'{glab}/full={gfull}')
    cases[('G1630','sG1630_pv_00084')]=dict(hids=['seg130'],gem=['seg130_replacement'])
    bygrid={}
    for (g,s),c in cases.items(): bygrid.setdefault(g,[]).append((s,c))
    for g,lst in bygrid.items():
        f=files[g]; d=gpd.read_file(f); crs=d.crs or 'EPSG:4326'; d=d.set_crs(crs) if d.crs is None else d
        dm=d.to_crs(32734); S=gpd.read_file(f'{P1}/sam3_{g}.gpkg').set_index('id')
        H=gpd.read_file(f'{P1}/work/compare/human_{g}.gpkg').set_index('id')
        drop=set(); add=[]
        if g=='G1630':
            i130=d.index[d.segment_id==130]; drop|=set(i130)
            for i in i130: changes.append(dict(grid=g,action='drop',src_idx=int(i),segment_id=130,sam3_id='sG1630_pv_00084',flag='corrupt GeometryCollection (lat~0)',area_m2=None))
        for s,c in lst:
            sg=S.loc[s].geometry; sb=sg.buffer(0.5)
            hz=unary_union([H.loc[h].geometry for h in c['hids'] if h in H.index]) if c['hids']!=['seg130'] else sg
            for i,pg in dm.geometry.items():
                if i in drop or pg is None or pg.is_empty or not pg.intersects(sb): continue
                if g=='G1630' and d.segment_id.iloc[i]==130: continue
                pg0=pg.buffer(0); fr=pg0.intersection(sb).area/max(pg0.area,1e-9)
                if fr>=0.5 and (pg0.intersects(hz) or fr>0.9):
                    drop.add(i); changes.append(dict(grid=g,action='drop',src_idx=int(i),segment_id=d.get('segment_id',pd.Series([None]*len(d))).iloc[i],sam3_id=s,flag='',area_m2=round(pg0.area,2),frac_in_sam3=round(fr,2)))
                elif pg0.intersects(hz):
                    skipped.append(dict(grid=g,human_id=';'.join(c['hids']),sam3_id=s,reason=f'src idx {i} only {fr:.2f} inside SAM3 -> kept',gemini_label=None,gemini_full=None))
            flag='' if all('full=yes' in x or x=='seg130_replacement' for x in c['gem']) else 'FLAG gemini '+','.join(c['gem'])
            add.append(dict(geometry=sg,sam3_id=s,human_ids=';'.join(c['hids']),flag=flag))
            changes.append(dict(grid=g,action='add_sam3',src_idx=None,segment_id=None,sam3_id=s,flag=flag,area_m2=round(sg.area,2),human_ids=';'.join(c['hids'])))
        # Row order is significant: swh_gpkg refs are '<file>#<row>'. Dropped rows keep their
        # position with an EMPTY geometry (the builder discards empties after the swh check).
        keep=d.copy(); keep['v2d_dropped']=False; keep.loc[list(drop),'v2d_dropped']=True
        keep.loc[list(drop),'geometry']=None
        new=gpd.GeoDataFrame(add,geometry='geometry',crs=32734).to_crs(crs)
        new['source']='sam3_text_pilot_v1_replace'; new['class']='Residential'; new['grid_id']=g
        new['segment_id']=-1; new['method']='sam3_text'
        merged=pd.concat([keep,new[[c for c in new.columns if c in keep.columns or c in('source','sam3_id','human_ids','flag')]]],ignore_index=True)
        merged=gpd.GeoDataFrame(merged,geometry='geometry',crs=crs)
        o=os.path.join(out,os.path.basename(f)); os.remove(o); merged.to_file(o,driver='GPKG')
    # C: degenerate
    for g,f in files.items():
        if g in ('G1688','G1807','G1189','G1190','G1238'): continue  # builder uses --g1688-fixed / excludes these
        o=os.path.join(out,os.path.basename(f)); d=gpd.read_file(o); dm=(d.set_crs(4326) if d.crs is None else d).to_crs(32734)
        def isbad(x):
            if x is None or x.is_empty: return True
            try: return x.buffer(0).area<1e-4
            except Exception: return True
        pre=set(d.index[d['v2d_dropped'].fillna(False).astype(bool)]) if 'v2d_dropped' in d.columns else set()
        bad=[i for i,x in dm.geometry.items() if i not in pre and isbad(x)]
        if bad:
            for i in bad: changes.append(dict(grid=g,action='drop_degenerate',src_idx=int(i),area_m2=float(dm.geometry[i].area) if dm.geometry[i] is not None else 0,flag=str(d.geometry[i])[:80]))
            d['geometry']=d.geometry.where(~d.index.isin(bad),None); os.remove(o); d.to_file(o,driver='GPKG')
    # D: slivers -> ignore
    ls=pd.read_csv(f'{P2}/long_strips.csv'); ls=ls[(ls.grid!='G1630')&(ls.width_m<0.7)]
    sl=[]
    for r in ls.itertuples():
        H=gpd.read_file(f'{P2}/work/human/human_{r.grid}.gpkg').set_index('id')
        sl.append(dict(grid=r.grid,human_id=r.human_id,width_m=r.width_m,length_m=r.length_m,geometry=H.loc[r.human_id].geometry))
        changes.append(dict(grid=r.grid,action='sliver_to_ignore',sam3_id=None,human_ids=r.human_id,area_m2=r.area_m2,flag=f'width {r.width_m} m'))
    gpd.GeoDataFrame(sl,geometry='geometry',crs=32734).to_file(os.path.join(a.out,'ignore_slivers.gpkg'))
    pd.DataFrame(changes).to_csv(os.path.join(a.out,'changes.csv'),index=False)
    pd.DataFrame(skipped).to_csv(os.path.join(a.out,'skipped.csv'),index=False)
    print(pd.DataFrame(changes).groupby(['grid','action']).size()); print(pd.DataFrame(skipped).to_string())
if __name__=='__main__': main()
