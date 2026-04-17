# 标注数据说明

所有人工标注均为弱监督标注（weak supervision）。

V1.3 起，流水线任务定义为 **reviewed prediction footprint segmentation**；GT 标注仍遵循 installation-level 规则。详见 [ANNOTATION_SPEC.md](ANNOTATION_SPEC.md)。

## 目录结构 (2026-04-08 重组)

标注按区域分两个子目录，平行存放：

```
data/annotations/
├── Capetown/          # Cape Town: SAM2 cleaned + 早期 legacy
│   ├── G*_SAM2_*.gpkg               # SAM2 review pipeline 输出（COCO export 主源）
│   ├── G{0854,0855,0910,1018,...}.gpkg  # 早期 legacy 弱监督标注
│   ├── G1189.gpkg, G1190.gpkg, G1238.gpkg  # 早期 Google Earth 标注
│   ├── solarpanel_g0001_g1190.gpkg  # Google Earth 全局批量
│   └── all_annotations_cleaned.gpkg # 汇总快照
├── Joburg/            # Johannesburg
│   ├── JHB0[1-6].gpkg               # Li 6-grid 手标 pilot (legacy)
│   └── G{07,08,09}xx_V4_260407.gpkg # CBD batch1 25 grids (V4 推理 → review → SAM 重切)
├── ANNOTATION_SPEC.md
├── PROGRESS.md
├── README.md          # 本文件
└── annotation_manifest.csv
```

## 数据集

| 文件 / 模式 | 来源 | Grid 范围 | 状态 |
|------|------|-----------|------|
| `Capetown/G1238_SAM2_260320.gpkg` | SAM2.1 (GeoSAM/QGIS) 精细切割 | G1238 | 已完成（242 polygons） |
| `Capetown/G*_SAM2_*.gpkg` | review GUI + SAM2 fill (batch 001-004) | 100+ grids | 见 PROGRESS.md |
| `Capetown/solarpanel_g0001_g1190.gpkg` | Google Earth 网页端标注 → QGIS 转换 | G0001-G1190 | 已完成（已人工校准位置偏移） |
| `Joburg/JHB0[1-6].gpkg` | Li 手标 (legacy) | 6 grids | 已完成（191 installations） |
| `Joburg/G*_V4_260407.gpkg` | V4 推理 → review GUI → SAM 2.1 重切 | 25 CBD grids | 已完成（808 installations，含 146 个 SAM 重切） |

## 标注规范

- 坐标系：EPSG:4326 (WGS84)
- 标注对象：屋顶太阳能安装轮廓（polygon），一个 polygon = 一个 installation footprint
- 质量级别：弱监督（人工标注，未经交叉验证）
- 项目约定：标注文件保持 `EPSG:4326`，进入检测/评估脚本后按区域重投影做米制计算（Cape Town: `EPSG:32734` UTM 34S；JHB: `EPSG:32735` UTM 35S；代码通过 `core/grid_utils.py` 动态确定 CRS）

## 质量分层 (V1.3)

| Tier | 含义 | 用途 |
|------|------|------|
| T1 | 已按 ANNOTATION_SPEC.md 审查，几何精度满足 IoU>=0.3 | 验证集；所有评估结论 |
| T2 | 原始弱监督标注，未审查 | 训练集（与 T1 合用） |

标注 manifest: `annotation_manifest.csv` — 每个标注一行，记录 grid_id、quality_tier、review_status 等。
使用 `scripts/bootstrap_manifest.py` 初始化。
