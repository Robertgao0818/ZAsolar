"""Sample FP detection images with polygon overlay for visual inspection."""
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from shapely.ops import unary_union

outdir = Path("/workspace/fp_samples")
outdir.mkdir(exist_ok=True)

sample_count = 0
for grid_dir in sorted(Path("results").iterdir()):
    if sample_count >= 12:
        break
    pred_file = grid_dir / "predictions_metric.gpkg"
    if not pred_file.exists():
        continue

    grid_id = grid_dir.name
    p = gpd.read_file(str(pred_file))
    gt_files = list(Path("data/annotations/cleaned").glob(f"{grid_id}*"))
    if not gt_files:
        continue
    g = gpd.read_file(str(gt_files[0])).to_crs(p.crs)
    gu = unary_union(g.geometry)
    p["is_tp"] = p.geometry.apply(
        lambda x: x.intersection(gu).area / x.area > 0.1 if x.area > 0 else False
    )

    fps = p[~p["is_tp"]]
    print(f"  {grid_id}: {len(p)} pred, {len(fps)} FP, {len(p)-len(fps)} TP")
    if len(fps) == 0:
        continue

    samples = fps.sample(min(3, len(fps)), random_state=42)

    for idx, row in samples.iterrows():
        tile_name = row.get("source_tile", "")
        tile_path = Path(f"tiles/{grid_id}/{tile_name}.tif")
        if not tile_path.exists():
            tile_path = Path(f"/workspace/tiles/{grid_id}/{tile_name}.tif")
        if not tile_path.exists():
            print(f"    tile not found: {tile_name}")
            continue

        try:
            with rasterio.open(str(tile_path)) as src:
                # Reproject polygon to tile CRS if needed
                from pyproj import Transformer
                tile_crs = src.crs
                if p.crs != tile_crs:
                    row_geom = gpd.GeoSeries([row.geometry], crs=p.crs).to_crs(tile_crs).iloc[0]
                else:
                    row_geom = row.geometry
                cx, cy = row_geom.centroid.x, row_geom.centroid.y

                # Convert polygon centroid to pixel coords in the tile
                px, py = ~src.transform * (cx, cy)
                px, py = int(px), int(py)

                # Read 200x200 pixel window around centroid
                half = 100
                row_start = max(0, py - half)
                col_start = max(0, px - half)
                row_stop = min(src.height, py + half)
                col_stop = min(src.width, px + half)

                window = rasterio.windows.Window.from_slices(
                    (row_start, row_stop), (col_start, col_stop)
                )
                data = src.read(window=window)

                if data.shape[1] < 20 or data.shape[2] < 20:
                    print(f"    too small: {data.shape}")
                    continue

                # Convert to PIL and draw polygon
                img = Image.fromarray(data[:3].transpose(1, 2, 0))
                draw = ImageDraw.Draw(img)

                # Transform polygon coords to pixel coords relative to window
                win_transform = rasterio.windows.transform(window, src.transform)
                coords = []
                if row_geom.geom_type == "Polygon":
                    for x, y in row_geom.exterior.coords:
                        px_rel, py_rel = ~win_transform * (x, y)
                        coords.append((int(px_rel), int(py_rel)))
                    if len(coords) >= 3:
                        draw.polygon(coords, outline="red", width=2)
                elif row_geom.geom_type == "MultiPolygon":
                    for poly in row_geom.geoms:
                        c = []
                        for x, y in poly.exterior.coords:
                            px_rel, py_rel = ~win_transform * (x, y)
                            c.append((int(px_rel), int(py_rel)))
                        if len(c) >= 3:
                            draw.polygon(c, outline="red", width=2)

                conf = row.get("confidence", 0)
                area = row.get("area_m2", 0)
                fname = f"FP_{sample_count:02d}_{grid_id}_conf{conf:.2f}_area{area:.0f}m2.png"
                img.save(str(outdir / fname))
                print(f"    Saved: {fname}")
                sample_count += 1
        except Exception as e:
            import traceback
            print(f"    Error {grid_id}: {e}")
            traceback.print_exc()

print(f"\nTotal: {sample_count} FP samples in {outdir}")
