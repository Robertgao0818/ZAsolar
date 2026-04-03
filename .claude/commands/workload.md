统计标注工作量。

扫描 `data/annotations/cleaned/` 目录下所有 `*_SAM2_*.gpkg` 文件，对每个文件：

1. 读取 polygon 数量（= 标注工作量）
2. 用 3m buffer 合并计算 installation 数量（不修改原始文件，只统计）
3. 统计 multi-part installation 数量

输出格式为表格，包含：
- Grid ID
- Polygons（多边形数 = 标注工作量）
- Installations（合并后的安装数）
- Multi-part（含多面板的安装数）
- 标注日期（从文件名提取）

表格末尾附合计行。同时输出按日期分组的工作量汇总。

注意：
- 使用区域对应的 UTM CRS 计算空间距离（Cape Town: EPSG:32734, JHB: EPSG:32735）
- 合并距离阈值 = 3m
- 不修改任何文件，只读取统计
- 也扫描原始的 3 个 grid（G1238/G1189/G1190 的 `*_SAM2_260320.gpkg`），这些在 `data/annotations/` 目录下
