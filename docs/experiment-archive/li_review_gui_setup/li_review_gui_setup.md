# Review GUI Setup — Handoff for Li

**目的**: 让你在自己机器上启动浏览器版 GUI，对任意 grid 的瓦片做**穷尽标注**（exhaustive annotation：SAM 点击直接出 polygon，逐 tile 过完整个 grid），完全绕开 QGIS。

GUI 来源：`scripts/annotations/review_detections.py`，加 `--exhaustive` flag 切到穷尽标注模式。

适用范围：你这边在跑的 **6 个 Vexcel 城市，60 grid**（pietermaritzburg / durban / east_london / gqeberha / bloemfontein / pretoria，grid ID 形如 `PMB0042`/`DBN0007`/`ELS0015`/`GQB0033`/`BFN0021`/`PTA0005`）。这些 grid 都已在 `configs/datasets/regions.yaml` 注册，GUI 会从 grid_id 前缀自动 infer region，不用手动传 `--region`。

---

## 1. 一次性环境 setup

需要 Linux / macOS / WSL2 + Python 3.10+。Windows 原生不建议（rasterio + sam2 麻烦）。

```bash
# Clone 仓库（只要代码 + 配置，不带大数据）
git clone <ZAsolar repo URL> ZAsolar
cd ZAsolar

# 建 venv 装依赖
./scripts/bootstrap_env.sh          # 创建 .venv 并 pip install
source scripts/activate_env.sh
```

`bootstrap_env.sh` 走 `requirements.lock.txt`，关键包：`geopandas`, `rasterio`, `shapely`, `pandas`, `numpy`, `Pillow`, `pyproj`, `pyyaml`。GPU / 训练相关的不是 GUI 必需。

GUI 不强制 GPU。**只有 SAM 在线分割时才用 GPU**（见 §6）；不开 SAM 就纯 CPU 标点。

---

## 2. 每个 grid 的数据包结构

穷尽标注模式**不需要 predictions GPKG**。每个 grid 你会收到的只有瓦片：

```
<grid_id>/                     # 例如 PMB0042/
  PMB0042_0_0_geo.tif
  PMB0042_0_1_geo.tif
  ...
```

> 少数 grid 会是单文件 `PMB0042_mosaic.tif`，GUI 一样支持。

把所有 grid 的瓦片目录放到本地一个固定根目录下：

```
$TILES_ROOT/PMB0042/PMB0042_0_0_geo.tif
$TILES_ROOT/PMB0042/PMB0042_0_1_geo.tif
$TILES_ROOT/DBN0007/DBN0007_0_0_geo.tif
...
```

只要 `$TILES_ROOT/<grid_id>/` 存在，GUI 会优先读它。`configs/datasets/regions.yaml` 仓库里已经把 60 个 grid 全注册了，你不用改任何配置。

---

## 3. 启动 GUI（穷尽标注模式）

```bash
source scripts/activate_env.sh
export SOLAR_TILES_ROOT=/absolute/path/to/your/tiles_root

# 单 grid
python scripts/annotations/review_detections.py --grid-id PMB0042 --exhaustive

# 批量（一次开多个 grid，连续审）
python scripts/annotations/review_detections.py --exhaustive \
  --grid-id PMB0042 PMB0043 DBN0007 GQB0033
```

启动后控制台打印：

```
[INFO] Auto-inferred region=pietermaritzburg for PMB0042 from grid_id pattern
[INIT] Exhaustive-annotation mode (no predictions required)
Open in browser: http://127.0.0.1:8766
```

Chrome / Firefox 打开即可。

**穷尽模式下的页面默认状态**：
- 标题显示 `Exhaustive Annotation`
- 默认 `SAM FN` 模式开启（不用按 M）
- Filter 默认 `All + empty tiles`（列出 grid 内所有 tile，不只是有 prediction 的）
- 进度条显示 `tiles done / total tiles`（不再追 prediction-level 数字）
- 多一个 **Mark Empty** 按钮（快捷键 `X`）

> 端口被占就 `--port 8767`。要从同局域网另一台机器访问就 `--host 0.0.0.0`，注意防火墙。

---

## 4. 操作 cheatsheet（穷尽模式）

每个 tile 三种结局之一：(1) 全部用 SAM 点上 polygon → 自动算"做完"；(2) 看了发现没板子，按 `X` 标 Empty；(3) 之后再看，先翻过。

| 操作 | 说明 |
|------|------|
| 左键点击屋顶 | SAM 出候选 polygon（黄色） |
| 右键点击 | SAM 负点（refine 候选） |
| `R` | 重置 SAM 点，只保留最后一个正点 |
| **`A` / Enter** | **接受当前 SAM polygon**（绿色，进 sam_added） |
| `Esc` | 拒绝当前 SAM 候选（保留 FN 模式不退） |
| `Z` | 撤销最近一次 accept |
| **`X`** | 把当前 tile 标记为 Empty（"这块屋顶就是没板子"） |
| **← →** | 上 / 下一个 tile |
| 滚轮 / 拖拽 / 双击 | 缩放 / 平移 / 重置视图 |
| Space + 拖拽 | 平移（不出现误点击） |

底部状态栏实时显示 `Tile pos/total | done/total tiles done`。每次按键都自动写盘，关浏览器不会丢。

接受 polygon 后 GUI **不会退出 SAM 模式**，直接在原地继续点下一个屋顶，连续穷尽标注一个 tile 内几十个 polygon 不用切模式。

---

## 5. 输出文件 — 回传清单

每个 grid 审完后，在 `results/<grid_id>/review/` 下会有：

```
<grid_id>_sam_added.gpkg                 # ← 主产物：所有 SAM 接受的 polygon
<grid_id>_reviewed.gpkg                  # 合并视图（穷尽模式下内容 == sam_added）
<grid_id>_reviewed.qml                   # QGIS 配色（可不传）
empty_tiles.csv                          # 你标记为 Empty 的 tile 列表
detection_review_decisions.csv           # 决策记录（穷尽模式下都是 correct）
fn_markers.csv                           # （穷尽模式下基本为空，SAM 直接出 polygon）
```

**回传给 Gao**：整个 `results/<grid_id>/review/` 目录打包即可。不要回传 tiles。

> 用 `--review-root /some/path` 可以把输出 park 到别的位置；默认是 `./results/<grid_id>/review/`。

---

## 6. SAM 在线补标（强烈推荐，需要 GPU）

穷尽标注**几乎一定要开 SAM**——徒手画 polygon 不现实。需要 CUDA GPU。

1. 下载 SAM2 checkpoint：`sam2.1_hiera_large.pt`，放到：
   ```
   ~/zasolar_data/models/sam2/checkpoints/sam2.1_hiera_large.pt
   ```
   （路径硬编码在 `review_detections.py` 顶部 `_SAM_CHECKPOINT_CANDIDATES`，要改自己改。）

2. 装 sam2：
   ```bash
   pip install "sam2 @ git+https://github.com/facebookresearch/sam2.git"
   ```

3. GUI 启动时控制台打印 `[SAM] Ready`。之后页面右上角 Mark FN 按钮自动变成 `SAM FN`。

如果没 GPU，跳过这一段——FN 还可以打点（左键放标记），后续 Gao 这边用 GPU 跑批量 SAM 出 polygon。

---

## 7. 常见问题

| 症状 | 排查 |
|------|------|
| `Tiles not found` | `SOLAR_TILES_ROOT` 没 export，或 `$TILES_ROOT/<grid_id>/` 路径不对 |
| `Auto-inferred region` 没出现 | grid_id 拼错（必须 `PMB0042` 这种全大写，4 位数字） |
| `matches multiple regions` | 同一 grid_id 命中多 region（不应该出现，碰到就 `--region pietermaritzburg` 显式传） |
| GUI 打开看不到 tile 列表 | 检查 filter，应该是 `All + empty tiles`；若被改成 `All tiles` 则只看 prediction-有效 tile，穷尽模式下会显示空 |
| 端口冲突 | `--port 8767` |
| SAM 没加载 | 检查 CUDA、checkpoint 路径、`sam2` pip 装没 |

---

## 8. 跑通最小验证

收到第一个 grid（比如 PMB0042）后建议先跑：

```bash
ls $TILES_ROOT/PMB0042/                  # 应该看到 *_geo.tif
python scripts/annotations/review_detections.py --grid-id PMB0042 --exhaustive
# 打开 http://127.0.0.1:8766
# 应该看到 "Exhaustive Annotation" 标题 + Mark Empty 按钮
# 在屋顶上左键点一下 → SAM 出黄色候选 → 按 A 接受 → 变绿
# 找一个明显没板子的 tile → 按 X → 按钮变 "Empty ✓"
# Ctrl+C 关掉，看 results/PMB0042/review/ 下有 sam_added.gpkg + empty_tiles.csv
```

跑通后告诉 Gao，就可以正式发批次了。

---

## 9. 多 grid 工作流建议

60 个 grid 分批，每次起 4–8 个一组（同一区域优先），一次性把瓦片放好：

```bash
export SOLAR_TILES_ROOT=/path/to/vexcel_tiles
python scripts/annotations/review_detections.py --exhaustive \
  --grid-id PMB0042 PMB0043 PMB0044 PMB0045
# 一组做完关浏览器，启动下一组
python scripts/annotations/review_detections.py --exhaustive \
  --grid-id DBN0007 DBN0008 DBN0009 DBN0010
```

每个 grid 的输出独立 park 在 `results/<grid_id>/review/`，做完一批就回传一批，不用等全部跑完。

---

参考：项目内传统 review 工作流（带 predictions）见 [`docs/semi_auto_annotation_workflow.md`](../semi_auto_annotation_workflow.md)；本文是它的"无预测穷尽版"。
