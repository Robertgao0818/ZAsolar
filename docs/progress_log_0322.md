# 工作记录 2026-03-22

**项目**: Cape Town 屋顶太阳能安装检测 (ZAsolar)

---

## 一、OSM 建筑过滤与选择性下载

**目标**: 利用 OSM 建筑数据过滤纯荒地 grid/tile，减少无效下载量。

**实现**:
- `filter_grids_osm.py`: 一次性从 Overpass API 批量下载整个研究区 127,203 个建筑中心点，本地 spatial join 到 task_grid
- Grid 级过滤: 2,214 个 grid 中仅 714 个有建筑 (32.2%)，1,500 个纯荒地跳过
- Tile 级过滤: 对有建筑的 grid，逐 tile 判定建筑存在性，仅下载含建筑的 tile + 上下左右 1 圈 buffer 邻居
- 结果: 92,988 tiles 中仅需下载 16,782 tiles (18.0%)，节省 82% 下载量
- 缓存: `cache/osm_buildings_centroids.gpkg` (建筑中心点), `cache/tile_download_mask.csv` (tile 级掩膜)

**改造下载管线**:
- `download_tiles.py`: 新增 `tile_mask` 参数，支持选择性下载
- `download_reviewed_grids.py`: 新增 `--use-tile-mask` 标志，自动读取 tile 掩膜
- `build_grid_vrts.py`: 添加缺失目录 skip 逻辑

**已知问题**:
- 开普敦郊区扩张区域 OSM 覆盖滞后（如 G1572），新建房屋未被 OSM 收录导致 tile 被误跳
- 对策: 考虑将 buffer 从 1 圈增加到 2 圈（覆盖率 55%→71%，全局仍节省 29%），或对需要的 grid 手动全量下载

---

## 二、Batch 002 下载与标注

**Grid 审查**: Batch 002 (102 grids reviewed)，34 grids 标记为 keep

**下载**:
- 31 grids 通过 OSM 过滤后下载 (3 个无建筑跳过: G1517, G1573, G1632)
- 742 tiles 选择性下载，workers=6 并行，全程 0 errors
- G1572 后续补充全量下载 (42 tiles) 以覆盖新建区域
- 全部 VRT mosaic 生成并同步至 `D:\ZAsolar\tiles\`

**标注** (SAM2.1 / GeoSAM+QGIS):
- 今日完成 26 个 grid 标注 (batch 001: 14 grids + batch 002: 12 grids)

**Batch 001 (14 grids)**:

| Grid ID | Panels | Installations | Multi-part |
|---------|--------|---------------|------------|
| G1240   |      7 |             1 |          1 |
| G1243   |     71 |            38 |         21 |
| G1244   |     43 |            26 |          9 |
| G1245   |    112 |            57 |         33 |
| G1293   |      6 |             3 |          2 |
| G1294   |     10 |             6 |          4 |
| G1297   |      1 |             1 |          0 |
| G1298   |     26 |            13 |          7 |
| G1299   |     97 |            58 |         21 |
| G1300   |     34 |            25 |          6 |
| G1349   |     20 |             5 |          4 |
| G1354   |     26 |            15 |          6 |
| G1410   |      4 |             2 |          2 |
| G1411   |     35 |            14 |          9 |
| **小计** | **492** | **264** | **125** |

**Batch 002 (12 grids)**:

| Grid ID | Panels | Installations | Multi-part |
|---------|--------|---------------|------------|
| G1466   |      6 |             5 |          1 |
| G1467   |     93 |            39 |         23 |
| G1516   |      5 |             5 |          0 |
| G1520   |     13 |            10 |          2 |
| G1521   |     70 |            43 |         17 |
| G1522   |     46 |            35 |         10 |
| G1523   |     75 |            54 |         18 |
| G1524   |      9 |             5 |          3 |
| G1569   |    141 |            80 |         35 |
| G1570   |    138 |            74 |         39 |
| G1571   |      7 |             1 |          1 |
| G1572   |     53 |            29 |         16 |
| **小计** | **656** | **380** | **165** |

**今日合计: 26 grids, 1,148 panels, 644 installations**

---

## 三、累计标注汇总

| 日期 | Grids | Panels | Installations | Multi-part |
|------|-------|--------|---------------|------------|
| 2026-03-20 | 3 | 475 | 232 | 118 |
| 2026-03-22 | 26 | 1,148 | 644 | 290 |
| **合计** | **29** | **1,623** | **876** | **408** |

---

## 四、Git 提交

- `f91c42c` V2 annotation expansion: SAM2 batch labeling pipeline, manifest update, and weekly report
- `0c75640` Add OSM building filter for tile-level selective download

---

## 五、待办 (Next)

- 完成 batch 002 剩余 19 grids 标注
- 注意区分泳池太阳能加热系统 (solar thermal) 与光伏板 (PV)，前者不标注
- 标注完成后重新导出 COCO 数据集，从 V2 checkpoint 继续 fine-tune (LR=5e-5, ~15 epochs)
- 评估 buffer=2 作为默认 tile 过滤策略
