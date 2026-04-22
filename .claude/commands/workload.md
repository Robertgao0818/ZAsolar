统计标注工作量。

扫描 `data/annotations/Capetown/` 和 `data/annotations/Joburg/` 目录下所有标注文件，对每个文件读取 polygon 数量（= 标注工作量）。

输出格式为表格，包含：
- Grid ID
- Polygons（多边形数 = 标注工作量）
- 标注日期（从文件名提取）

表格末尾附合计行。同时输出按日期分组的工作量汇总。

注意：
- 不修改任何文件，只读取统计
- 不做 buffer 合并或聚类，直接统计 polygon 个数
- Cape Town 标注全部位于 `data/annotations/Capetown/`，包含 `*_SAM2_*.gpkg`（SAM2 review 流） 和早期 legacy `G*.gpkg`
- Joburg 标注位于 `data/annotations/Joburg/`，包含 `JHB0[1-6].gpkg` (Li 手标 legacy) 和 `G*_V4_*.gpkg` (CBD batch1 V4 review + SAM 重切)
