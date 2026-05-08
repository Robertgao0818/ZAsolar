"""Boundary-aware dataset for JHB Phase A retrain.

Reads ``data/annotations_channel2_clean/<G>/<G>_clean_gt.gpkg`` (the dissolved
Channel 2 clean GT, same file as evaluation), then differentiates mask
supervision by the ``source`` provenance field on each polygon:

  source                  | mask_weight | boundary_ignore_px (m²-bucketed)
  ---                     | ---         | ---
  ``SAM_supp+V3C_TP``     | 1.0         | 2 (<600 m²) / 3 (≥600 m²)
  ``SAM_supp`` only       | 1.0         | 2 / 3
  ``V3C_TP`` only         | 0.0         | — (no mask BCE; halo not learned)
  ``Li_marked`` (any)     | dropped     | — (Li is eval cross-check, not train)

Why clean_gt and not raw parts: V3C_correct often overlaps sam_added because
RA used SAM as a re-annotation tool to fill in panels V3C had cut too small
(not just FN). Treating them as separate instances would split a single
install into two boxes; treating them via dedup drop would lose the V3C
portion. The dissolve-cluster union in clean_gt is the correct semantics:
one install = one polygon, with SAM-corrected boundary where SAM contributed.

Used together with ``core.training.boundary_aware_mask.install_patch()`` and
``stash_batch_supervision`` in the training loop.

Companion spec: ``configs/datasets/jhb_phaseA.yaml``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from shapely.geometry import box as shapely_box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402

CLEAN_GT_ROOT = PROJECT_ROOT / "data/annotations_channel2_clean"


def _rasterize_with_ignore(
    pts_pixel: np.ndarray, h: int, w: int, band_px: int
) -> tuple[np.ndarray, np.ndarray]:
    fg = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(pts_pixel).astype(np.int32).reshape(-1, 2)
    cv2.fillPoly(fg, [pts], 1)
    if band_px <= 0:
        return fg, np.zeros_like(fg)
    k = 2 * band_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(fg, kernel)
    ero = cv2.erode(fg, kernel)
    return fg, (dil & ~ero).astype(np.uint8)


def _classify_source(source: str) -> tuple[str, float]:
    """Returns (kind, mask_weight). kind in {'sam', 'v3c_only', 'drop'}."""
    if not source:
        return "drop", 0.0
    if "Li_marked" in source:
        return "drop", 0.0
    if "SAM_supp" in source:
        return "sam", 1.0
    if "V3C_TP" in source:
        return "v3c_only", 0.0
    return "drop", 0.0


def _resolve_tiles(grid_id: str, region: str, layer: str) -> list[Path]:
    d = resolve_tiles_dir(grid_id, region=region, imagery_layer=layer)
    if d.is_file():
        return [d]
    tiles = sorted(d.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted(p for p in d.glob(f"{grid_id}_*.tif") if "mosaic" not in p.stem)
    return tiles


def _load_clean_gt(grid_id: str, metric_crs: str) -> gpd.GeoDataFrame:
    p = CLEAN_GT_ROOT / grid_id / f"{grid_id}_clean_gt.gpkg"
    if not p.exists():
        return gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    gdf = gpd.read_file(p)
    if str(gdf.crs) != metric_crs:
        gdf = gdf.to_crs(metric_crs)
    if "area_m2" not in gdf.columns:
        gdf = gdf.copy()
        gdf["area_m2"] = gdf.geometry.area
    return gdf.reset_index(drop=True)


class JHBPhaseADataset(torch.utils.data.Dataset):
    """Chip-scanning dataset over Channel 2 clean GT, with source-aware mask
    supervision (SAM-corrected geometry → mask BCE + ignore band; V3C-only
    geometry → box+cls only)."""

    def __init__(
        self,
        spec: dict,
        split: str,
        transforms=None,
    ):
        assert split in ("train", "val")
        self.spec = spec
        self.split = split
        self.transforms = transforms
        self.chip_size = int(spec["imagery"]["chip_size"])
        self.overlap = float(spec["imagery"]["overlap"])
        self.region = spec["imagery"]["region"]
        self.layer = spec["imagery"]["imagery_layer"]
        self.sup = spec["supervision"]
        self.band_schedule = self.sup["sam_added"]["boundary_ignore_px"]
        self.metric_crs = spec["sources"]["metric_crs"]

        # chip dict:
        #   {tile_path, x0, y0, grid_id,
        #    polygons: list of {geom_world, kind, mask_weight, band_px, area_m2}}
        self.chips: list[dict] = []

        grids = spec["splits"][split]["grids"]
        for g in grids:
            self._index_grid(g)

        if split == "train":
            self._balance(float(spec.get("neg_ratio", 0.15)),
                          seed=int(spec.get("seed", 42)))

        n_pos = sum(1 for c in self.chips if c["polygons"])
        n_neg = len(self.chips) - n_pos
        print(f"[JHBPhaseADataset/{split}] {len(self.chips)} chips ({n_pos} pos, {n_neg} neg)")

    def _index_grid(self, grid_id: str):
        gt = _load_clean_gt(grid_id, self.metric_crs)
        if gt.empty:
            print(f"[WARN] clean_gt missing for {grid_id}", file=sys.stderr)
            return
        tiles = _resolve_tiles(grid_id, self.region, self.layer)
        if not tiles:
            print(f"[WARN] no tiles for {grid_id}", file=sys.stderr)
            return

        sched_lo = int(self.band_schedule.get("<600", 2))
        sched_hi = int(self.band_schedule.get(">=600", 3))

        # Pre-classify polygons
        rows = []
        for j, src in enumerate(gt["source"]):
            kind, w = _classify_source(str(src) if src is not None else "")
            if kind == "drop":
                continue
            area_m2 = float(gt["area_m2"].iloc[j])
            band_px = (sched_hi if area_m2 >= 600 else sched_lo) if kind == "sam" else 0
            rows.append({
                "j": j,
                "kind": kind,
                "mask_weight": w,
                "band_px": band_px,
                "area_m2": area_m2,
            })

        if not rows:
            return

        chip = self.chip_size
        step = max(1, int(chip * (1 - self.overlap)))

        for tile_path in tiles:
            with rasterio.open(tile_path) as src:
                tw, th = src.width, src.height
                tile_crs = src.crs
                gt_tile = gt.to_crs(tile_crs) if str(gt.crs) != str(tile_crs) else gt

                tb = src.bounds
                tile_box = shapely_box(tb.left, tb.bottom, tb.right, tb.top)

                # Filter polygons to this tile
                local = []
                for r in rows:
                    geom = gt_tile.geometry.iloc[r["j"]]
                    if geom is None or geom.is_empty:
                        continue
                    if not geom.intersects(tile_box):
                        continue
                    local.append({**r, "geom_tile": geom})
                if not local:
                    continue

                local_gdf = gpd.GeoDataFrame(
                    {"j": [r["j"] for r in local]},
                    geometry=[r["geom_tile"] for r in local],
                    crs=tile_crs,
                )
                local_sidx = local_gdf.sindex

                for y0 in range(0, th - chip + 1, step):
                    for x0 in range(0, tw - chip + 1, step):
                        win = Window(x0, y0, chip, chip)
                        wb = rasterio.windows.bounds(win, src.transform)
                        chip_box = shapely_box(*wb)

                        chip_polys = []
                        for k_local in local_sidx.intersection(chip_box.bounds):
                            r = local[int(k_local)]
                            geom = r["geom_tile"]
                            if not geom.intersects(chip_box):
                                continue
                            chip_polys.append({
                                "geom_world": geom,
                                "kind": r["kind"],
                                "mask_weight": r["mask_weight"],
                                "band_px": r["band_px"],
                                "area_m2": r["area_m2"],
                            })

                        self.chips.append({
                            "tile_path": str(tile_path),
                            "x0": int(x0),
                            "y0": int(y0),
                            "polygons": chip_polys,
                            "grid_id": grid_id,
                        })

    def _balance(self, neg_ratio: float, seed: int):
        pos = [c for c in self.chips if c["polygons"]]
        neg = [c for c in self.chips if not c["polygons"]]
        rng = np.random.default_rng(seed)
        n_neg_keep = int(round(len(pos) * neg_ratio))
        if n_neg_keep < len(neg):
            picked = rng.choice(len(neg), size=n_neg_keep, replace=False)
            neg = [neg[int(i)] for i in picked]
        rng.shuffle(pos)
        self.chips = pos + neg

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        c = self.chips[idx]
        with rasterio.open(c["tile_path"]) as src:
            win = Window(c["x0"], c["y0"], self.chip_size, self.chip_size)
            data = src.read(window=win)
            inv = ~src.transform
        if data.shape[0] >= 3:
            data = data[:3]
        image = torch.as_tensor(data, dtype=torch.float32) / 255.0

        H = W = self.chip_size
        boxes, labels, fg_masks, ig_masks, weights, areas = [], [], [], [], [], []

        for p in c["polygons"]:
            geom = p["geom_world"]
            if geom.geom_type == "Polygon":
                rings = [list(geom.exterior.coords)]
            elif geom.geom_type == "MultiPolygon":
                rings = [list(g.exterior.coords) for g in geom.geoms]
            else:
                continue

            poly_fg = np.zeros((H, W), dtype=np.uint8)
            poly_ig = np.zeros((H, W), dtype=np.uint8)
            for ring in rings:
                ring = np.asarray(ring)
                cols, rows = inv * (ring[:, 0], ring[:, 1])
                pts_pix = np.stack(
                    [np.asarray(cols) - c["x0"], np.asarray(rows) - c["y0"]],
                    axis=1,
                )
                fg, ig = _rasterize_with_ignore(pts_pix, H, W, p["band_px"])
                poly_fg = np.maximum(poly_fg, fg)
                poly_ig = np.maximum(poly_ig, ig)

            ys, xs = np.where(poly_fg > 0)
            if len(xs) == 0:
                continue
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max + 1, y_max + 1])
            labels.append(1)
            fg_masks.append(poly_fg)
            ig_masks.append(poly_ig)
            weights.append(p["mask_weight"])
            areas.append(float((x_max - x_min) * (y_max - y_min)))

        if not boxes:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "ignore_masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "mask_weights": torch.zeros(0, dtype=torch.float32),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(fg_masks), dtype=torch.uint8),
                "ignore_masks": torch.as_tensor(np.stack(ig_masks), dtype=torch.uint8),
                "mask_weights": torch.as_tensor(weights, dtype=torch.float32),
                "image_id": torch.tensor([idx]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# Backwards-compatible alias for the prior, raw-parts-based class name.
JHBRawPartsDataset = JHBPhaseADataset


def load_spec(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
