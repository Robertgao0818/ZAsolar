"""Cross-tile / cross-grid merge for the 100 m-tile production pipeline (G7, PLAN.md).

Tiles are given as TileIn(key, gx0, gy0, load) where (gx0, gy0) is the global pixel origin
(round(transform.c/gsd), round(-transform.f/gsd)) in one projected CRS, and load() returns the
2-D float PV probability in [0, 1]. Pure CPU; no model code.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy import ndimage

S8 = np.ones((3, 3), bool)


@dataclass(frozen=True)
class TileIn:
    key: str
    gx0: int
    gy0: int
    load: Callable[[], np.ndarray]


def global_origin(transform, gsd: float) -> tuple[int, int]:
    return int(round(transform.c / gsd)), int(round(-transform.f / gsd))


class UnionFind:
    def __init__(self) -> None:
        self.p: list[int] = []

    def add(self, n: int) -> int:
        b = len(self.p)
        self.p.extend(range(b, b + n))
        return b

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def _border(lab: np.ndarray):
    h, w = lab.shape
    ys = np.r_[np.zeros(w, int), np.full(w, h - 1), np.arange(h), np.arange(h)]
    xs = np.r_[np.arange(w), np.arange(w), np.zeros(h, int), np.full(h, w - 1)]
    v = lab[ys, xs]
    fg = v > 0
    return ys[fg], xs[fg], v[fg]


def check_no_overlap(extents) -> None:
    """Raise if any two tile rectangles (x0, y0, x1, y1, key; half-open, global px) share a pixel."""
    if len(extents) < 2:
        return
    e = np.array([x[:4] for x in extents], np.int64)
    keys = [x[4] for x in extents]
    order = np.argsort(e[:, 0], kind='stable')
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if e[j, 0] >= e[i, 2]:
                break
            if e[j, 1] < e[i, 3] and e[i, 1] < e[j, 3]:
                raise ValueError(f'tiles overlap: {keys[i]} and {keys[j]}')


def parse_cfg(cfg: str) -> tuple[float, float]:
    """'t0.5' -> (0.5, 0.5); 'h0.9_0.5' -> (0.9, 0.5). Returns (hi, lo)."""
    if cfg.startswith('t'):
        t = float(cfg[1:])
        return t, t
    if cfg.startswith('h'):
        hi, lo = map(float, cfg[1:].split('_'))
        if not lo <= hi:
            raise ValueError(cfg)
        return hi, lo
    raise ValueError(cfg)


@dataclass
class MergePlan:
    """Result of pass 1: per-component merged root and keep flag; masks are rebuilt per tile on demand."""
    cfg: str
    lo: float
    info: dict  # key -> (base, nl, shape)
    root: np.ndarray
    keep_root: np.ndarray
    rmax: np.ndarray
    cnpx: np.ndarray


def merge_plan(tiles: list[TileIn], cfg: str) -> MergePlan:
    """Pass 1 of cross-tile hysteresis + merge (memory: O(#components + #border pixels))."""
    hi, lo = parse_cfg(cfg)
    if len({t.key for t in tiles}) != len(tiles):
        raise ValueError('duplicate tile keys')
    uf = UnionFind()
    edges: dict[tuple[int, int], int] = {}
    owner: dict[tuple[int, int], str] = {}
    info = {}
    cmax: list[float] = []
    cnpx: list[int] = []
    extents: list[tuple[int, int, int, int, str]] = []
    for t in tiles:
        p = np.asarray(t.load(), dtype=np.float32)
        if p.ndim != 2:
            raise ValueError(f'{t.key}: prob must be 2-D')
        h, w = p.shape
        extents.append((t.gx0, t.gy0, t.gx0 + w, t.gy0 + h, t.key))
        lab, nl = ndimage.label(p >= lo, S8)
        base = uf.add(nl)
        info[t.key] = (base, nl, (h, w))
        if nl:
            idx = np.arange(1, nl + 1)
            cmax.extend(ndimage.maximum(p, lab, idx).tolist())
            cnpx.extend(np.bincount(lab.ravel(), minlength=nl + 1)[1:].tolist())
        ys, xs, v = _border(lab)
        for y, x, c in zip(ys.tolist(), xs.tolist(), v.tolist()):
            k = (t.gx0 + x, t.gy0 + y)
            edges[k] = base + c - 1
            owner[k] = t.key
    check_no_overlap(extents)
    for (x, y), g in edges.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    j = edges.get((x + dx, y + dy))
                    if j is not None and owner[(x + dx, y + dy)] != owner[(x, y)]:
                        uf.union(g, j)
    n = len(uf.p)
    root = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    rmax = np.zeros(n, np.float32)
    if n:
        np.maximum.at(rmax, root, np.asarray(cmax, np.float32))
    return MergePlan(cfg, lo, info, root, rmax >= hi, rmax, np.asarray(cnpx, np.int64))


def plan_tile_ids(plan: MergePlan, t: TileIn) -> np.ndarray:
    """Pass 2 for one tile: int64 raster of global object id (0 = background)."""
    base, nl, shape = plan.info[t.key]
    if nl == 0:
        return np.zeros(shape, np.int64)
    p = np.asarray(t.load(), dtype=np.float32)
    lab, nl2 = ndimage.label(p >= plan.lo, S8)
    if nl2 != nl or p.shape != shape:
        raise RuntimeError(f'{t.key}: non-deterministic load')
    r = plan.root[base:base + nl]
    oid = np.r_[0, np.where(plan.keep_root[r], r + 1, 0)]
    return oid[lab]


def plan_objects(plan: MergePlan) -> dict[int, dict]:
    """id -> dict(n_px, max_p, n_tiles, tile_keys) for kept merged objects."""
    objects: dict[int, dict] = {}
    for key, (base, nl, _) in plan.info.items():
        for c in range(nl):
            r = int(plan.root[base + c])
            if plan.keep_root[r]:
                o = objects.setdefault(r + 1, dict(n_px=0, max_p=float(plan.rmax[r]), tile_keys=set()))
                o['n_px'] += int(plan.cnpx[base + c])
                o['tile_keys'].add(key)
    for o in objects.values():
        o['tile_keys'] = tuple(sorted(o['tile_keys']))
        o['n_tiles'] = len(o['tile_keys'])
    return objects


def merge_tiles(tiles: list[TileIn], cfg: str):
    """Cross-tile hysteresis + merge.

    Returns (masks, ids, objects): masks[key] bool mask; ids[key] int64 raster of global object id
    (0 = background); objects dict id -> dict(n_px, max_p, n_tiles, tile_keys). Hysteresis is applied
    to the merged component, so a seed (p>=hi) on one tile keeps the connected lo-region on all tiles.
    """
    plan = merge_plan(tiles, cfg)
    ids = {t.key: plan_tile_ids(plan, t) for t in tiles}
    masks = {k: v > 0 for k, v in ids.items()}
    return masks, ids, plan_objects(plan)


def per_tile_masks(tiles: list[TileIn], cfg: str) -> dict[str, np.ndarray]:
    """Production as-is: binarize each tile alone (same rule as zaeval.postproc.binarize)."""
    hi, lo = parse_cfg(cfg)
    out = {}
    for t in tiles:
        p = np.asarray(t.load(), dtype=np.float32)
        lab, nl = ndimage.label(p >= lo, S8)
        out[t.key] = (np.r_[False, ndimage.maximum(p, lab, np.arange(1, nl + 1)) >= hi][lab]) if nl else np.zeros(p.shape, bool)
    return out


def polygons(ids: dict[str, np.ndarray], transforms: dict, crs):
    """One polygon (possibly multi) per global object id, dissolved across tiles."""
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape
    rows = []
    for k, a in ids.items():
        if not a.any():
            continue
        a32 = a.astype(np.int32) if a.max() < 2**31 else None
        if a32 is None:
            raise OverflowError('object id exceeds int32')
        for geom, v in features.shapes(a32, mask=a32 > 0, transform=transforms[k], connectivity=8):
            rows.append(dict(oid=int(v), geometry=shape(geom)))
    g = gpd.GeoDataFrame(rows, geometry='geometry', crs=crs) if rows else gpd.GeoDataFrame(dict(oid=[]), geometry=[], crs=crs)
    if not len(g):
        return g
    from shapely import make_valid
    from shapely.ops import unary_union
    d = g.dissolve(by='oid', as_index=False)
    area0 = d.area.values.copy()
    fixed = []
    for geom in d.geometry:
        if not geom.is_valid:
            geom = make_valid(geom)
            if geom.geom_type == 'GeometryCollection':
                geom = unary_union([x for x in geom.geoms if x.geom_type in ('Polygon', 'MultiPolygon')])
        fixed.append(geom)
    d = d.set_geometry(gpd.GeoSeries(fixed, crs=crs, index=d.index))
    if not d.is_valid.all() or not np.allclose(d.area.values, area0, rtol=1e-9, atol=1e-9):
        raise RuntimeError('polygon repair changed validity or area')
    return d


def assign_owner(objs, cells, cell_col: str = 'gridcell_id'):
    """Each object -> exactly one cell, by representative point; outside all cells -> nearest cell.
    Ties (point on a shared edge) -> lowest cell id."""
    import geopandas as gpd
    cells = cells.to_crs(objs.crs).sort_values(cell_col).reset_index(drop=True)
    rp = gpd.GeoDataFrame(dict(oid=objs.oid.values), geometry=objs.geometry.representative_point().values, crs=objs.crs)
    j = gpd.sjoin(rp, cells[[cell_col, 'geometry']], how='left', predicate='intersects')
    j = j.sort_values(['oid', cell_col]).drop_duplicates('oid')
    miss = j[cell_col].isna()
    if miss.any():
        nn = gpd.sjoin_nearest(rp[rp.oid.isin(j.oid[miss])], cells[[cell_col, 'geometry']], how='left')
        nn = nn.sort_values(['oid', cell_col]).drop_duplicates('oid').set_index('oid')[cell_col]
        j.loc[miss, cell_col] = j.loc[miss, 'oid'].map(nn).values
    out = objs.copy()
    out['owner_cell'] = out.oid.map(j.set_index('oid')[cell_col])
    return out
