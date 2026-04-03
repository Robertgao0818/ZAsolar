# 标注数据说明

所有人工标注均为弱监督标注（weak supervision）。

V1.3 起，流水线任务定义为 **reviewed prediction footprint segmentation**；GT 标注仍遵循 installation-level 规则。详见 [ANNOTATION_SPEC.md](ANNOTATION_SPEC.md)。

## 数据集

| 文件 | 来源 | Grid 范围 | 状态 |
|------|------|-----------|------|
| G1238_detailed.gpkg | SAM2.1 (GeoSAM/QGIS) 精细切割 | G1238 | 已完成（248 polygons，替代原 G1238.gpkg） |
| solarpanel_g0001_g1190.gpkg | Google Earth 网页端标注 → QGIS 转换 | G0001-G1190 | 已完成（已人工校准位置偏移） |

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
