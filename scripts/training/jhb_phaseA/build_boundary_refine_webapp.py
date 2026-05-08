"""Build a lightweight browser editor for boundary refinement.

QGIS can become unresponsive when a project loads many large Vexcel rasters.
This builder pre-renders each candidate polygon into a small PNG chip and
creates a tiny local web app for vertex editing.  The web app saves edits as
JSON and can export edited polygons back to a GeoPackage.

Usage:
    python scripts/training/jhb_phaseA/build_boundary_refine_webapp.py
    python results/analysis/jhb_phaseA_boundary_refine_webapp/server.py
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402


DEFAULT_WORKPKG = (
    PROJECT_ROOT
    / "results/analysis/jhb_phaseA_boundary_refine_qgis/boundary_refine_workpkg.gpkg"
)
DEFAULT_OUT = PROJECT_ROOT / "results/analysis/jhb_phaseA_boundary_refine_webapp"
TARGET_CRS = "EPSG:3857"


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return None


def _grid_vrt(grid: str) -> Path:
    qgis_vrt = (
        PROJECT_ROOT
        / "results/analysis/jhb_phaseA_boundary_refine_qgis/rasters"
        / f"{grid}_vexcel_2024.vrt"
    )
    if qgis_vrt.exists():
        return qgis_vrt
    # Fallback to the first tile path.  Normal builds should use the VRT.
    tiles_dir = resolve_tiles_dir(grid, region="johannesburg", imagery_layer="vexcel_2024")
    if tiles_dir.is_file():
        return tiles_dir
    tiles = sorted(tiles_dir.glob(f"{grid}_*_*_geo.tif"))
    if not tiles:
        raise FileNotFoundError(f"No Vexcel tiles found for {grid}: {tiles_dir}")
    return tiles[0]


def _window_for_geom(src, geom: Polygon, *, pad_px: int, min_size: int):
    inv = ~src.transform
    minx, miny, maxx, maxy = geom.bounds
    c0, r0 = inv * (minx, maxy)
    c1, r1 = inv * (maxx, miny)
    x0 = math.floor(min(c0, c1)) - pad_px
    x1 = math.ceil(max(c0, c1)) + pad_px
    y0 = math.floor(min(r0, r1)) - pad_px
    y1 = math.ceil(max(r0, r1)) + pad_px

    width = max(min_size, x1 - x0)
    height = max(min_size, y1 - y0)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = cx - width // 2
    y0 = cy - height // 2
    x1 = x0 + width
    y1 = y0 + height

    x0 = max(0, min(x0, src.width - 1))
    y0 = max(0, min(y0, src.height - 1))
    x1 = max(x0 + 1, min(x1, src.width))
    y1 = max(y0 + 1, min(y1, src.height))
    return rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)


def _coords_to_pixels(coords, transform) -> list[list[float]]:
    inv = ~transform
    pts = []
    for x, y in coords:
        px, py = inv * (x, y)
        pts.append([round(float(px), 2), round(float(py), 2)])
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def build(workpkg: Path, out_dir: Path, *, pad_px: int, min_size: int) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "chips").mkdir(parents=True)
    (out_dir / "edits").mkdir()

    gdf = gpd.read_file(workpkg, layer="clean_boundary_edit")
    if str(gdf.crs) != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    gdf = gdf.sort_values(["grid", "sample_rank"]).reset_index(drop=True)

    metadata = []
    grouped = {grid: group.copy() for grid, group in gdf.groupby("grid")}
    for grid, group in grouped.items():
        vrt = _grid_vrt(str(grid))
        with rasterio.open(vrt) as src:
            for _, row in group.iterrows():
                geom = _largest_polygon(row.geometry)
                if geom is None:
                    continue
                ref_id = str(row["ref_id"])
                window = _window_for_geom(src, geom, pad_px=pad_px, min_size=min_size)
                chip_transform = src.window_transform(window)
                arr = src.read([1, 2, 3], window=window, boundless=False)
                rgb = np.transpose(arr, (1, 2, 0))
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                image_name = f"{ref_id}.png"
                Image.fromarray(rgb).save(out_dir / "chips" / image_name, optimize=True)

                polygon_px = _coords_to_pixels(list(geom.exterior.coords), chip_transform)
                metadata.append(
                    {
                        "ref_id": ref_id,
                        "sample_rank": int(row.get("sample_rank", len(metadata) + 1)),
                        "grid": str(row["grid"]),
                        "source_pool": str(row["source_pool"]),
                        "source_idx": int(row["source_idx"]),
                        "focus": str(row.get("focus", "")),
                        "area_bucket": str(row.get("area_bucket", "")),
                        "area_m2": float(row.get("area_m2", 0.0)),
                        "wobble_10m": float(row.get("wobble_10m", 0.0)),
                        "image": f"chips/{image_name}",
                        "width": int(rgb.shape[1]),
                        "height": int(rgb.shape[0]),
                        "transform": [
                            float(chip_transform.a),
                            float(chip_transform.b),
                            float(chip_transform.c),
                            float(chip_transform.d),
                            float(chip_transform.e),
                            float(chip_transform.f),
                        ],
                        "polygon": polygon_px,
                    }
                )

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (out_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    (out_dir / "style.css").write_text(STYLE_CSS, encoding="utf-8")
    (out_dir / "server.py").write_text(SERVER_PY, encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Boundary refine web app\n\n"
        "Run:\n\n"
        "```bash\n"
        "python results/analysis/jhb_phaseA_boundary_refine_webapp/server.py\n"
        "```\n\n"
        "Open http://127.0.0.1:8765/ .\n",
        encoding="utf-8",
    )
    print(f"[DONE] wrote {len(metadata)} chips -> {out_dir}")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZAsolar Boundary Editor</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div id="app">
    <aside>
      <div class="top">
        <h1>Boundary Editor</h1>
        <select id="gridFilter"></select>
        <select id="statusFilter">
          <option value="all">all status</option>
          <option value="pending">pending</option>
          <option value="done">done</option>
          <option value="skip">skip</option>
        </select>
      </div>
      <div id="list"></div>
    </aside>
    <main>
      <header>
        <div>
          <strong id="title">No item</strong>
          <span id="meta"></span>
        </div>
        <div class="buttons">
          <button id="prevBtn">Prev</button>
          <button id="nextBtn">Next</button>
          <button id="resetBtn">Reset</button>
          <button id="deleteBtn">Delete vertex</button>
          <button id="skipBtn">Skip</button>
          <button id="doneBtn">Save done</button>
          <button id="exportBtn">Export GPKG</button>
        </div>
      </header>
      <section id="stageWrap">
        <div id="stage">
          <img id="chip" alt="">
          <svg id="overlay"></svg>
        </div>
      </section>
      <footer>
        <label><input type="checkbox" id="useTraining" checked> use_for_training</label>
        <input id="notes" placeholder="notes">
        <span id="message"></span>
      </footer>
    </main>
  </div>
  <script src="app.js"></script>
</body>
</html>
"""


STYLE_CSS = """*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#111827;color:#e5e7eb}#app{display:grid;grid-template-columns:330px 1fr;height:100vh}aside{border-right:1px solid #374151;background:#0f172a;overflow:hidden;display:flex;flex-direction:column}.top{padding:12px;border-bottom:1px solid #374151}h1{font-size:17px;margin:0 0 10px}select,input,button{font:inherit}select{width:100%;margin:4px 0;padding:7px;background:#111827;color:#e5e7eb;border:1px solid #4b5563;border-radius:4px}#list{overflow:auto;padding:8px}.item{padding:8px;border:1px solid #263244;border-radius:6px;margin-bottom:6px;cursor:pointer;background:#111827}.item.active{outline:2px solid #06b6d4}.item.done{border-color:#16a34a}.item.skip{border-color:#dc2626}.item small{color:#9ca3af;display:block;margin-top:2px}main{display:grid;grid-template-rows:auto 1fr auto;min-width:0}header,footer{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border-bottom:1px solid #374151;background:#111827}footer{border-top:1px solid #374151;border-bottom:0}#meta{color:#9ca3af;margin-left:10px}.buttons{display:flex;gap:6px;flex-wrap:wrap}button{padding:7px 10px;background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;border-radius:5px;cursor:pointer}button:hover{background:#374151}#doneBtn{border-color:#16a34a}#skipBtn{border-color:#dc2626}#stageWrap{overflow:auto;padding:14px;display:grid;place-items:start center}#stage{position:relative;background:#000;line-height:0;box-shadow:0 0 0 1px #374151}#chip{display:block;max-width:min(100%,1200px);height:auto;image-rendering:auto}#overlay{position:absolute;inset:0;width:100%;height:100%;overflow:visible}.poly{fill:rgba(6,182,212,.22);stroke:#06b6d4;stroke-width:2;vector-effect:non-scaling-stroke}.handle{fill:#f8fafc;stroke:#0f172a;stroke-width:1.5;cursor:move;vector-effect:non-scaling-stroke}.handle.selected{fill:#f97316}.edge{stroke:transparent;stroke-width:14;fill:none;cursor:copy;vector-effect:non-scaling-stroke}#notes{flex:1;min-width:240px;padding:7px;background:#0f172a;color:#e5e7eb;border:1px solid #4b5563;border-radius:5px}#message{color:#9ca3af;min-width:220px;text-align:right}"""


APP_JS = r"""let items=[], edits={}, filtered=[], current=0, selectedVertex=-1;
const el=id=>document.getElementById(id);

async function init(){
  items = await (await fetch('metadata.json')).json();
  edits = await (await fetch('/api/edits')).json();
  items.forEach(x=>{ if(!edits[x.ref_id]) edits[x.ref_id]={status:'pending', polygon:x.polygon, use_for_training:true, notes:''}; });
  setupFilters();
  bind();
  applyFilters();
}

function setupFilters(){
  const grids=['all',...Array.from(new Set(items.map(x=>x.grid))).sort()];
  el('gridFilter').innerHTML=grids.map(g=>`<option value="${g}">${g}</option>`).join('');
}

function bind(){
  el('gridFilter').onchange=applyFilters;
  el('statusFilter').onchange=applyFilters;
  el('prevBtn').onclick=()=>loadIndex(Math.max(0,current-1));
  el('nextBtn').onclick=()=>loadIndex(Math.min(filtered.length-1,current+1));
  el('resetBtn').onclick=()=>{ const it=filtered[current]; edits[it.ref_id].polygon=it.polygon.map(p=>[p[0],p[1]]); render(); };
  el('deleteBtn').onclick=deleteVertex;
  el('doneBtn').onclick=()=>save('done');
  el('skipBtn').onclick=()=>save('skip');
  el('exportBtn').onclick=exportGpkg;
  document.addEventListener('keydown',e=>{
    if(e.target.tagName==='INPUT') return;
    if(e.key==='n'||e.key==='ArrowRight') el('nextBtn').click();
    if(e.key==='p'||e.key==='ArrowLeft') el('prevBtn').click();
    if(e.key==='Backspace'||e.key==='Delete') deleteVertex();
    if(e.key==='Enter') save('done');
  });
}

function applyFilters(){
  const grid=el('gridFilter').value, status=el('statusFilter').value;
  filtered=items.filter(it=>(grid==='all'||it.grid===grid) && (status==='all'||edits[it.ref_id].status===status));
  current=0;
  drawList();
  loadIndex(0);
}

function drawList(){
  el('list').innerHTML=filtered.map((it,i)=>{
    const ed=edits[it.ref_id];
    return `<div class="item ${ed.status} ${i===current?'active':''}" data-i="${i}">
      <b>${it.grid}</b> #${it.sample_rank} ${it.source_pool}
      <small>${it.focus} · ${it.area_bucket} · ${it.area_m2.toFixed(1)} m2 · ${ed.status}</small>
    </div>`;
  }).join('');
  document.querySelectorAll('.item').forEach(n=>n.onclick=()=>loadIndex(Number(n.dataset.i)));
}

function loadIndex(i){
  if(!filtered.length){ el('title').textContent='No items'; return; }
  current=i; selectedVertex=-1;
  const it=filtered[current], ed=edits[it.ref_id];
  el('chip').src=it.image;
  el('chip').onload=render;
  el('title').textContent=`${it.ref_id}`;
  el('meta').textContent=`${current+1}/${filtered.length} · ${it.focus} · ${it.area_m2.toFixed(1)} m2`;
  el('useTraining').checked=!!ed.use_for_training;
  el('notes').value=ed.notes||'';
  drawList();
}

function render(){
  const it=filtered[current]; if(!it) return;
  const svg=el('overlay'), img=el('chip'), ed=edits[it.ref_id];
  svg.setAttribute('viewBox',`0 0 ${it.width} ${it.height}`);
  const pts=ed.polygon;
  const d=pts.map(p=>p.join(',')).join(' ');
  let html=`<polygon class="poly" points="${d}"></polygon>`;
  for(let i=0;i<pts.length;i++){
    const a=pts[i], b=pts[(i+1)%pts.length];
    html+=`<line class="edge" data-edge="${i}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"></line>`;
  }
  pts.forEach((p,i)=>{html+=`<circle class="handle ${i===selectedVertex?'selected':''}" data-v="${i}" cx="${p[0]}" cy="${p[1]}" r="5"></circle>`});
  svg.innerHTML=html;
  svg.querySelectorAll('.handle').forEach(h=>{
    h.onpointerdown=e=>startDrag(e, Number(h.dataset.v));
    h.onclick=e=>{selectedVertex=Number(h.dataset.v); render(); e.stopPropagation();};
  });
  svg.querySelectorAll('.edge').forEach(edge=>{
    edge.ondblclick=e=>addVertexOnEdge(e, Number(edge.dataset.edge));
  });
}

function svgPoint(e){
  const svg=el('overlay');
  const pt=svg.createSVGPoint();
  pt.x=e.clientX; pt.y=e.clientY;
  const p=pt.matrixTransform(svg.getScreenCTM().inverse());
  return [Math.max(0,p.x), Math.max(0,p.y)];
}

function startDrag(e, idx){
  e.preventDefault(); selectedVertex=idx; e.target.setPointerCapture(e.pointerId);
  const move=ev=>{ const p=svgPoint(ev); const it=filtered[current]; edits[it.ref_id].polygon[idx]=[Math.round(p[0]*10)/10,Math.round(p[1]*10)/10]; render(); };
  const up=()=>{ window.removeEventListener('pointermove',move); window.removeEventListener('pointerup',up); };
  window.addEventListener('pointermove',move); window.addEventListener('pointerup',up);
}

function addVertexOnEdge(e, edgeIdx){
  const p=svgPoint(e), it=filtered[current], poly=edits[it.ref_id].polygon;
  poly.splice(edgeIdx+1,0,[Math.round(p[0]*10)/10,Math.round(p[1]*10)/10]);
  selectedVertex=edgeIdx+1; render();
}

function deleteVertex(){
  const it=filtered[current]; if(!it||selectedVertex<0) return;
  const poly=edits[it.ref_id].polygon;
  if(poly.length<=3) return;
  poly.splice(selectedVertex,1); selectedVertex=-1; render();
}

async function save(status){
  const it=filtered[current]; if(!it) return;
  const ed=edits[it.ref_id];
  ed.status=status;
  ed.use_for_training=el('useTraining').checked;
  ed.notes=el('notes').value;
  const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ref_id:it.ref_id,...ed})});
  el('message').textContent=await res.text();
  drawList();
}

async function exportGpkg(){
  const res=await fetch('/api/export',{method:'POST'});
  el('message').textContent=await res.text();
}

init();
"""


SERVER_PY = r"""from __future__ import annotations
import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
EDITS = ROOT / 'edits'
METADATA = json.loads((ROOT / 'metadata.json').read_text(encoding='utf-8'))

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == '/api/edits':
            return self._json(load_edits())
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/save':
            n = int(self.headers.get('Content-Length', '0'))
            data = json.loads(self.rfile.read(n).decode('utf-8'))
            ref_id = data['ref_id']
            safe = ''.join(ch for ch in ref_id if ch.isalnum() or ch in '_-')
            (EDITS / f'{safe}.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
            return self._text(f"saved {ref_id} as {data.get('status')}")
        if self.path == '/api/export':
            path = export_gpkg()
            return self._text(f'exported {path}')
        self.send_error(404)

    def _json(self, obj):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _text(self, text):
        data = text.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

def load_edits():
    out = {}
    for p in EDITS.glob('*.json'):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
            out[d['ref_id']] = d
        except Exception:
            pass
    return out

def pixel_to_world(transform, pt):
    a,b,c,d,e,f = transform
    x,y = pt
    return (a*x + b*y + c, d*x + e*y + f)

def export_gpkg():
    import geopandas as gpd
    from shapely.geometry import Polygon
    edits = load_edits()
    rows = []
    for item in METADATA:
        ed = edits.get(item['ref_id'])
        if not ed:
            continue
        poly_px = ed.get('polygon') or []
        if len(poly_px) < 3:
            continue
        coords = [pixel_to_world(item['transform'], p) for p in poly_px]
        coords.append(coords[0])
        geom = Polygon(coords)
        rows.append({
            'ref_id': item['ref_id'],
            'grid': item['grid'],
            'source_pool': item['source_pool'],
            'focus': item['focus'],
            'area_m2_orig': item['area_m2'],
            'edit_status': ed.get('status', 'pending'),
            'use_for_training': int(bool(ed.get('use_for_training', True))),
            'edit_notes': ed.get('notes', ''),
            'geometry': geom,
        })
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:3857')
    out = ROOT / 'clean_boundary_web_edits.gpkg'
    if out.exists():
        out.unlink()
    if len(gdf):
        gdf.to_file(out, layer='clean_boundary_edit', driver='GPKG')
    else:
        out.write_text('')
    return out

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    print(f'Open http://{args.host}:{args.port}/')
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workpkg", type=Path, default=DEFAULT_WORKPKG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pad-px", type=int, default=128)
    ap.add_argument("--min-size", type=int, default=640)
    args = ap.parse_args()
    build(args.workpkg, args.out, pad_px=args.pad_px, min_size=args.min_size)


if __name__ == "__main__":
    main()
