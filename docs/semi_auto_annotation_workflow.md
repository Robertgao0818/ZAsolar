# Semi-Auto Annotation Workflow

模型预标注 → Web GUI 审查 → SAM 补标 → 导出 GPKG → 加入训练集

## Prerequisites

- Tiles 存储在 `D:\ZAsolar\tiles\` (WSL: `/mnt/d/ZAsolar/tiles/`)
- 环境变量: `SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles`
- Best model: `checkpoints/best_model.pth`
- Python 环境: `source scripts/activate_env.sh`

## Step 1: Grid 预览筛选 + 下载

```bash
# 生成 grid 预览（从上次结束的 grid 开始，每批 100 个）
python scripts/imagery/grid_preview_batch.py --start-grid-id G1847 --batch-index 1 --batch-size 100 \
  --output-dir results/grid_previews/batch_004

# 生成 OSM 建筑掩膜（buffer=2 圈邻居）
python scripts/imagery/filter_grids_osm.py --buffer-rings 2

# 启动筛选 GUI (http://localhost:8765)
python scripts/imagery/review_grid_previews.py --batch-dir results/grid_previews/batch_004

# 下载选中的 grid（输出到 D 盘，OSM 掩膜过滤）
SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python scripts/imagery/download_reviewed_grids.py \
  --batch-dir results/grid_previews/batch_004 --use-tile-mask --workers 4
```

**注意**: 启动筛选 GUI 前，可以先自动排除 0% valid imagery 的 grid：
```python
# 在 Python 中运行
import csv, pandas as pd
from datetime import datetime, timezone
metrics = pd.read_csv("results/grid_previews/batch_004/grid_preview_metrics.csv")
decisions_path = "results/grid_previews/batch_004/grid_review_decisions.csv"
now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
blank = metrics[metrics["valid_imagery_ratio"] == 0.0]
with open(decisions_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["grid_id", "decision", "notes", "updated_at"])
    writer.writeheader()
    for _, row in blank.iterrows():
        writer.writerow({"grid_id": row["grid_id"], "decision": "exclude",
                        "notes": "auto: 0% valid imagery", "updated_at": now})
```

## Step 2: 模型推理 (Detection-Only)

```bash
# 单 grid 推理
SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python detect_and_evaluate.py \
  --grid-id G1686 --model-path checkpoints/best_model.pth --force

# 批量推理（所有新下载的 grid）
for gid in G1682 G1683 G1685 G1686 G1687 G1688 G1689 G1690 G1691 G1692 G1693 \
           G1743 G1744 G1747 G1749 G1750 G1798 G1800 G1801 G1806 G1807; do
  echo "=== $gid ==="
  SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python detect_and_evaluate.py \
    --grid-id $gid --model-path checkpoints/best_model.pth --force
done
```

输出: `results/<GridID>/predictions_metric.gpkg` (无 GT 时自动跳过评估)

## Step 3: Review GUI 审查

```bash
SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python scripts/annotations/review_detections.py \
  --grid-id G1686
```

打开 **http://localhost:8766**

### 操作方式

| 操作 | 说明 |
|------|------|
| 点击多边形 | 选中（蓝色边框高亮，无填充遮挡） |
| A | 选中多边形标为 correct |
| D | 选中多边形标为 delete (误检) |
| E | 选中多边形标为 edit (需修正) |
| C | 清除选中多边形状态 |
| Q | 整个 tile 全标 correct (快速通过) |
| M | 切换 FN 标记模式 |
| 左键 (FN 模式) | 放置 FN 标记点（漏标位置） |
| 右键 (FN 模式) | 删除最近的 FN 标记 |
| Esc | 取消选中 / 退出 FN 模式 / 重置缩放 |
| 滚轮 | 缩放（以鼠标为中心） |
| 拖拽 | 平移 |
| 双击 | 重置缩放 |
| F | 聚焦选中多边形 |
| ← → | 上/下一个 tile |

### 审查后点 Export GPKG

输出:
- `results/<GridID>/review/G1686_reviewed.gpkg` — 带 review_status 的预测
- `results/<GridID>/review/G1686_reviewed.qml` — QGIS 配色样式
- `results/<GridID>/review/G1686_fn_markers.gpkg` — FN 标记点
- `results/<GridID>/review/detection_review_decisions.csv` — 审查决策记录

## Step 4: SAM 自动补标 FN

对 FN 标记点用 SAM-ViT-Large 点提示分割，自动生成多边形并合并到 reviewed GPKG：

```python
# 在项目根目录运行
source scripts/activate_env.sh

python3 << 'PYEOF'
import torch, numpy as np, geopandas as gpd, csv, pandas as pd
from transformers import SamModel, SamProcessor
from PIL import Image
from shapely.geometry import shape
import rasterio
from rasterio.features import shapes as rio_shapes
from pathlib import Path

GRID_ID = "G1686"  # ← 改成你的 grid
TILES_DIR = Path(f"/mnt/d/ZAsolar/tiles/{GRID_ID}")
REVIEW_DIR = Path(f"results/{GRID_ID}/review")
REVIEWED_PATH = REVIEW_DIR / f"{GRID_ID}_reviewed.gpkg"

# Load existing, remove old SAM polygons
gdf = gpd.read_file(str(REVIEWED_PATH))
crs = gdf.crs
original = gdf[gdf.get("source", "") != "sam_fn_marker"].copy()

# Load FN markers
markers = []
with open(REVIEW_DIR / "fn_markers.csv") as f:
    for row in csv.DictReader(f): markers.append(row)
print(f"{len(markers)} FN markers")

# Load SAM (首次运行会下载 ~1.2GB 模型权重到 ~/.cache/huggingface)
processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
model = SamModel.from_pretrained("facebook/sam-vit-large")
if torch.cuda.is_available(): model = model.to("cuda")

new_polys = []
for m in markers:
    tile_key, px, py = m["tile_key"], float(m["px"]), float(m["py"])
    with rasterio.open(TILES_DIR / f"{tile_key}_geo.tif") as ds:
        data = ds.read(); bounds = ds.bounds; transform = ds.transform
    rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
    img = Image.fromarray(rgb)

    inputs = processor(img, input_points=[[[int(px), int(py)]]], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") if hasattr(v,"to") else v for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    masks_list = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores.cpu().numpy()[0][0]

    # 选 mask index 1（中间粒度，最适合光伏）
    mask = masks_list[0][0][1].numpy().astype(np.uint8)
    print(f"[{tile_key}] score={scores[1]:.3f} pixels={mask.sum()}")

    for geom, val in rio_shapes(mask, transform=transform):
        if val == 1:
            poly = shape(geom)
            if poly.is_valid and poly.area > 0:
                new_polys.append({"geometry": poly, "source": "sam_fn_marker",
                    "tile_key": tile_key, "confidence": float(scores[1]), "review_status": "correct"})
                break

# Merge and save
fn_gdf = gpd.GeoDataFrame(new_polys, crs=crs)
for col in original.columns:
    if col not in fn_gdf.columns and col != "geometry": fn_gdf[col] = None
for col in fn_gdf.columns:
    if col not in original.columns and col != "geometry": original[col] = None
merged = gpd.GeoDataFrame(pd.concat([original, fn_gdf], ignore_index=True), crs=crs)
merged.to_file(str(REVIEWED_PATH), driver="GPKG")
print(f"Saved {len(merged)} polygons ({len(original)} det + {len(new_polys)} SAM)")
PYEOF
```

**SAM mask 选择**: index 0=最紧凑, 1=中间(推荐), 2=最宽松。光伏场景固定用 1。

**注意**: SAM 模型约 1.2GB，首次运行会下载到 `~/.cache/huggingface/`。用完后可删除释放 C 盘空间：
```bash
rm -rf ~/.cache/huggingface/hub/models--facebook--sam-vit-large
```

## Step 5: 导出为训练标注

审查完成后，将 reviewed GPKG 中 `correct` 的多边形转为标注文件：

```bash
# 将 correct 多边形复制到 annotations/cleaned/ 作为 GT
python3 -c "
import geopandas as gpd
gdf = gpd.read_file('results/G1686/review/G1686_reviewed.gpkg')
gt = gdf[gdf['review_status'] == 'correct'].copy()
gt.to_file('data/annotations/cleaned/G1686_SAM2_260325.gpkg', driver='GPKG')
print(f'Exported {len(gt)} GT polygons')
"

# 重新导出 COCO 训练集（包含新标注）
python export_coco_dataset.py --output-dir data/coco
```

## Step 6: 增量训练

```bash
# 本地
python train.py --coco-dir data/coco --output-dir checkpoints \
  --resume checkpoints/best_model.pth

# RunPod（见 cloud_setup.sh）
bash cloud_setup.sh --resume checkpoints/best_model.pth
```

## 文件路径汇总

| 文件 | 路径 |
|------|------|
| Tiles (D盘) | `/mnt/d/ZAsolar/tiles/<GridID>/` |
| 预测结果 | `results/<GridID>/predictions_metric.gpkg` |
| 审查决策 | `results/<GridID>/review/detection_review_decisions.csv` |
| FN 标记 | `results/<GridID>/review/fn_markers.csv` |
| 审查后 GPKG | `results/<GridID>/review/<GridID>_reviewed.gpkg` |
| QGIS 样式 | `results/<GridID>/review/<GridID>_reviewed.qml` |
| 训练标注 | `data/annotations/cleaned/<GridID>_SAM2_YYMMDD.gpkg` |

## 环境变量

```bash
export SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles
```

可加入 `scripts/activate_env.sh` 或 `.bashrc` 中。
