"""Cross-grid edge quantification on the Li Cape Town test set (CPU, read-only).
Writes only into this script's directory (results_edges.json, per_cell_halo.csv).

Inputs (all read-only):
  cells   ~/projects/ZAsolar/data/task_grid_ct_full.gpkg  (r2 scoring cells, same as build_li_testset_z2.py)
  prod    ~/projects/ZAsolar/data/task_grid_cpt.gpkg      (production census cells; chunk lattice = 4326 bbox, 0.0016 deg tiles)
  li      ~/zasolar_data/capetown_testset_li_v1_20261017/derived/li_later_all_raw.gpkg (labelled==True, 1260 polys)
  pieces  ~/zasolar_data/capetown_testset_li_z2_20261017/gt.gpkg
  tiles   ~/zasolar_data/capetown_testset_li_z2_20261017/tiles.csv
"""
import json, math, os, sys
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box, LineString, MultiLineString
from shapely.ops import unary_union
from shapely import segmentize

H = os.path.expanduser
OUT = os.path.dirname(os.path.abspath(__file__))
GROUPS = {"A": ["CPT1842", "CPT1897", "CPT1898", "CPT1900", "CPT1901", "CPT1902", "CPT1950", "CPT1951"],
          "B": ["CPT1952", "CPT1953", "CPT1954"], "C": ["CPT1899", "CPT1844", "CPT1896"],
          "D": ["CPT1843", "CPT1846", "CPT1895"]}
TEST = [g for v in GROUPS.values() for g in v]
TILE_DEG = 0.0016
R = {}

# ---- cells (r2 definition)
F = gpd.read_file(H("~/projects/ZAsolar/data/task_grid_ct_full.gpkg")).to_crs(32734)
F["grid"] = "CPT" + F.Name.str[1:]
cells = F[F.grid.isin(TEST)].set_index("grid").geometry
assert len(cells) == 17
U = unary_union(list(cells.values))

# ---- production cells (4326 native) and their chunk bbox
P4 = gpd.read_file(H("~/projects/ZAsolar/data/task_grid_cpt.gpkg")).set_index("gridcell_id")
P = P4.to_crs(32734).geometry
# same geometry as r2 cells?
R["r2_vs_prod_cell_max_symdiff_m2"] = round(max(cells[g].symmetric_difference(P[g]).area for g in TEST), 4)

def bbox_utm(g):
    x0, y0, x1, y1 = P4.loc[g].geometry.bounds
    b = segmentize(box(x0, y0, x1, y1), 1e-5)
    return gpd.GeoSeries([b], crs=4326).to_crs(32734).iloc[0]

BB = {g: bbox_utm(g) for g in P.index}
prod_cells = gpd.GeoDataFrame(geometry=list(P.values), index=P.index, crs=32734)
sidx = prod_cells.sindex

rows = []
for g in P.index:
    bb, c = BB[g], P[g]
    halo = bb.difference(c)
    # max halo width = Hausdorff distance bbox -> cell
    hw = bb.hausdorff_distance(c)
    nb = [P.index[i] for i in sidx.query(bb) if P.index[i] != g]
    dbl = halo.intersection(unary_union([P[n] for n in nb])).area if nb else 0.0
    rows.append(dict(cell=g, bbox_area=bb.area, halo_area=halo.area, halo_max_w=hw,
                     halo_in_other_cells=dbl, n_neigh=len(nb), test=g in TEST))
H_ = pd.DataFrame(rows)
H_.to_csv(f"{OUT}/per_cell_halo.csv", index=False)
R["prod_halo"] = dict(
    n_cells=len(H_),
    halo_area_m2_median=round(H_.halo_area.median()), halo_area_m2_min=round(H_.halo_area.min()), halo_area_m2_max=round(H_.halo_area.max()),
    halo_frac_of_cell_median=round(H_.halo_area.median() / 1e6, 4),
    halo_max_width_m_median=round(H_.halo_max_w.median(), 1), halo_max_width_m_max=round(H_.halo_max_w.max(), 1),
    halo_area_inside_other_census_cells_m2_total=round(H_.halo_in_other_cells.sum()),
    test_cells_halo_area_m2_mean=round(H_[H_.test].halo_area.mean()),
)

# ---- Li labelled polygons
A = gpd.read_file(H("~/zasolar_data/capetown_testset_li_v1_20261017/derived/li_later_all_raw.gpkg")).to_crs(32734).reset_index(names="src_row")
lab = A[A.labeled].copy(); lab["geometry"] = lab.geometry.buffer(0)
assert len(lab) == 1260
lost = lab.geometry.difference(U).area
inside = lab.geometry.intersection(U).area
R["clip_loss_recheck"] = dict(li_total_m2=round(lab.area.sum(), 1), lost_m2=round(lost.sum(), 1),
                              n_partly_outside=int(((lost > 0.01) & (inside > 0.01)).sum()),
                              n_wholly_outside=int((inside <= 0.01).sum()),
                              n_lost_gt0p01=int((lost > 0.01).sum()))

# boundary of test-area union = external edges; internal edges = all cell edges minus external
ext = U.boundary
all_edges = unary_union([c.boundary for c in cells.values])
internal = all_edges.difference(ext.buffer(0.01))
R["edge_length_m"] = dict(external=round(ext.length), internal=round(internal.length))

# ---- source polygons vs cell boundaries
L = lab[inside > 0.01].copy()
L["area"] = L.area
L["d_ext"] = L.geometry.distance(ext)
L["d_int"] = L.geometry.distance(internal) if not internal.is_empty else np.inf
L["d_any"] = np.minimum(L.d_ext, L.d_int)
L["cross_ext"] = L.geometry.intersects(ext)
L["cross_int"] = L.geometry.intersects(internal)
tot_n, tot_a = len(L), L.area.sum()
def frac(m): return dict(n=int(m.sum()), n_pct=round(100 * m.sum() / tot_n, 2), area_m2=round(L.area[m].sum(), 1), area_pct=round(100 * L.area[m].sum() / tot_a, 2))
R["src_polys_in_test_cells"] = dict(n=tot_n, area_m2=round(tot_a, 1))
R["src_vs_any_cell_edge"] = {f"within_{d}m": frac(L.d_any <= d) for d in (0, 5, 10, 20)}
R["src_vs_external_edge"] = {f"within_{d}m": frac(L.d_ext <= d) for d in (0, 10, 20)}
R["src_vs_internal_edge"] = {f"within_{d}m": frac(L.d_int <= d) for d in (0, 10, 20)}
R["src_crossing"] = dict(external=frac(L.cross_ext), internal=frac(L.cross_int))
# area of PV lying in edge bands (on the inside of the cells)
for d in (10, 20):
    band_ext = U.difference(U.buffer(-d))
    band_any = unary_union([c.difference(c.buffer(-d)) for c in cells.values])
    gi = L.geometry.intersection(U)
    R[f"pv_area_in_band_{d}m"] = dict(any_edge_m2=round(gi.intersection(band_any).area.sum(), 1),
                                      external_edge_m2=round(gi.intersection(band_ext).area.sum(), 1))

# ---- scored pieces
G = gpd.read_file(H("~/zasolar_data/capetown_testset_li_z2_20261017/gt.gpkg"))
R["pieces"] = dict(n=len(G), n_src=int(G.src_row.nunique()), split_src=int((G.groupby("src_row").size() > 1).sum()),
                   pv_pieces=int((G.cls == "pv").sum()) if "cls" in G else None)
G["d_edge"] = [geom.distance(cells[g].exterior) for geom, g in zip(G.geometry, G.grid)]
R["pieces_vs_own_cell_edge"] = {f"within_{d}m": dict(n=int((G.d_edge <= d).sum()), area_m2=round(G.area[G.d_edge <= d].sum(), 1)) for d in (0.01, 10, 20)}
R["pieces_vs_own_cell_edge_by_group"] = {k: {f"within_{d}m": int(((G.d_edge <= d) & (G.group == k)).sum()) for d in (10, 20)} | {"n": int((G.group == k).sum())} for k in "ABCD"}

# ---- tiles / units
T = pd.read_csv(H("~/zasolar_data/capetown_testset_li_z2_20261017/tiles.csv"))
T["tile"] = T.tile_id.str.split("@").str[0]
k = T.groupby("tile").size()
R["tiles"] = dict(units=len(T), tiles=int(k.size), tiles_in_2plus_cells=int((k > 1).sum()), units_on_multi_cell_tiles=int(k[k > 1].sum()),
                  max_cells_per_tile=int(k.max()))

# ---- production double coverage of GT (plain-concat census: each grid's chunk bbox processed, outputs not clipped)
mult = []
for geom in L.geometry:
    cand = [P.index[i] for i in sidx.query(geom.buffer(30))]
    # area-weighted number of grid bboxes covering the polygon, and area covered by >=2 bboxes
    covers = [BB[c].intersection(geom).area for c in cand]
    pieces2 = [BB[a].intersection(BB[b]).intersection(geom).area for i, a in enumerate(cand) for b in cand[i + 1:]]
    mult.append((sum(covers) / geom.area, sum(pieces2)))
L["bbox_mult"] = [m[0] for m in mult]; L["dbl_area"] = [m[1] for m in mult]
R["prod_double_processing_of_gt"] = dict(
    n_src_in_ge2_grid_bboxes=int((L.dbl_area > 0.01).sum()), pct=round(100 * (L.dbl_area > 0.01).mean(), 2),
    pv_area_in_ge2_bboxes_m2=round(L.dbl_area.sum(), 1), pv_area_pct=round(100 * L.dbl_area.sum() / tot_a, 2),
    area_weighted_mean_bbox_multiplicity=round(float((L.bbox_mult * L.area).sum() / tot_a), 4))

# ---- production 0.0016 deg chunk seams (detect_direct windows never span a chunk seam)
def seams(g):
    x0, y0, x1, y1 = P4.loc[g].geometry.bounds
    nc, nr = math.ceil((x1 - x0) / TILE_DEG), math.ceil((y1 - y0) / TILE_DEG)
    ls = [LineString([(x0 + i * TILE_DEG, y0), (x0 + i * TILE_DEG, y1)]) for i in range(1, nc)]
    ls += [LineString([(x0, y1 - j * TILE_DEG), (x1, y1 - j * TILE_DEG)]) for j in range(1, nr)]
    return gpd.GeoSeries(ls, crs=4326).to_crs(32734).union_all(), nc, nr
SE = {}
for g in TEST:
    s, nc, nr = seams(g); SE[g] = s.intersection(cells[g])
SEU = unary_union(list(SE.values()))
cs = L.geometry.intersects(SEU)
R["prod_chunk_seams"] = dict(chunks_per_cell=f"{nc}x{nr}", seam_len_per_cell_m=round(SEU.length / 17), cross=frac(cs))

# ---- MiT gate cells (3x3 per 100 m tile, 667 px @0.05 m) and 100 m tile lines
def lattice(offsets):
    x0, y0, x1, y1 = U.bounds
    xs = [b + o for b in range(int(x0 // 100 * 100), int(x1) + 100, 100) for o in offsets]
    ys = [b + o for b in range(int(y0 // 100 * 100), int(y1) + 100, 100) for o in offsets]
    return MultiLineString([[(x, y0 - 1), (x, y1 + 1)] for x in xs] + [[(x0 - 1, y), (x1 + 1, y)] for y in ys])
tile_l = lattice([0.0]); gate_l = lattice([0.0, 33.35, 66.65])
ct, cg = L.geometry.intersects(tile_l), L.geometry.intersects(gate_l)
# area on the smaller side when cut by gate lines (= worst-case area lost if the other gate cell is rejected... reported as the minor share)
def minor_share(geom, lines):
    from shapely.ops import split
    parts = split(geom, lines) if geom.geom_type == "Polygon" else None
    if parts is None or len(parts.geoms) < 2:
        return 0.0
    a = sorted(p.area for p in parts.geoms)
    return geom.area - a[-1]
R["mit_tile_lines_100m"] = frac(ct)
R["mit_gate_cell_lines_33m"] = frac(cg)
R["mit_gate_cut_area_outside_largest_piece_m2"] = round(sum(minor_share(g, gate_l) for g in L.geometry[cg]), 1)

json.dump(R, open(f"{OUT}/results_edges.json", "w"), indent=1)
print(json.dumps(R, indent=1))
