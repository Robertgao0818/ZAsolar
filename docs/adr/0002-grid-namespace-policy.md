# ADR-0002: Grid namespace 政策 — 机场码 canonical + retired namespace 两级解析

- **Status**: **Accepted** — 政策完全实施;JHB 退役 + CT CPT regrid 均已于 2026-06-12 落地(见下表)
- **Created**: 2026-06-12
- **Driver**: `lookup_region` 弃用 finding 复核 → 根因定位为 namespace 设计而非 lookup 函数

## Context

CT 与 JHB 的 `G\d{4}` namespace 历史性撞名:G1189/G1190/G1293/G1513/G1570/G1630
在两区域都注册但覆盖不同物理区域(rule 06-multi-city)。`lookup_region(grid_id)`
对它们静默返回首个命中(cape_town),JHB 意图的调用会安静拿到 CT 路径。
docstring 级软弃用实测挡不住 copy-paste(`export_v4_hn.py` docstring 引用
rule 06 后下一行即调单数 lookup)。

同时:JHB 已于 2026-06-03 裁决 JNB Vexcel-382 为 canonical scheme(老
Gxxxx/JHBnn 退役);后加 7 城已全部采用 disjoint 机场码前缀
(JNB/PMB/DBN/ELS/GQB/BFN/PTA);CT 在全量 census 推理前必然重切 grid。
约定已经存在,缺的是把它变成 registry 强制的不变量。

**硬约束**:退役 ≠ 可遗忘。JHB CBD 25-grid 评估套件(G0774/G0816/G0817/
G0922/G0925…)是锁定的 eval GT 面(`feedback_eval_gt_lock_clean`),
benchmark / Ch1-Ch3 历史结果 / training_sets.yaml provenance / negative_pool
全部以老 G-ID 为 key,必须永久可解析。

## Decision

1. **机场码 canonical namespace**:每区域的 active grid namespace 用 disjoint
   前缀(JNB/PMB/DBN/ELS/GQB/BFN/PTA/CPT);active namespace 之间两两不相交是
   registry 不变量。
2. **retired namespace 字段**:`regions.yaml` 区域级
   `retired_grid_id_patterns`(regex 列表,fullmatch)。退役 ID 原地保留
   (task_grid / coverage_grids / results 路径不动),仅改变 lookup 裁决层级。
3. **两级解析**:`lookup_regions(grid_id)` 先返回 active 命中;仅当无任何
   active 区域认领时回落 retired 命中。`include_retired=True` 返回两层
   (active 在前,即旧全量行为)。多 active 命中 = 不变量违例,发
   UserWarning。
4. **不变量执法点**:ID 级全量 disjointness 校验放测试层
   (`test_active_namespaces_pairwise_disjoint`,枚举 coverage/grids/task_grid
   全部 ID),不放加载期 —— 加载期枚举需读全部 task_grid gpkg,IO 代价不可
   接受;运行时由多 active 告警兜底。*(偏离最初"加载期断言"提法,刻意取舍。)*
5. **CT regrid 纳入同一约定**:CT census grid 重切时采用 `CPT\d{4}`
   (复用 JNB-382 切法 + vexcel_task_grids 命名约定),**同时产出 G→CPT
   crosswalk**(老 G-ID 挂着标注 provenance 与 Li 校准集对应,regrid 时生成
   成本最低),随后 CT 的 `G\d{4}` 同样进 `retired_grid_id_patterns`。
   Li 的 `L\d{4}` 是标注 namespace,保持 active。

终态(已达成) active namespace = `CPT/JNB/PMB/DBN/ELS/GQB/BFN/PTA + L`,
按前缀天然 disjoint;`region` 参数从"防错必需"降级为"性能优化"。

## 语义变更(已全部生效)

- 裸重叠 G-ID(G1189 等 6 个)在 JHB 退役后曾暂由 cape_town(active 所有者)
  解析;CT CPT regrid 落地后,这些 ID 在两区域均进入 retired 层,无任何
  active 所有者。`lookup_regions('G1189')` 通过 retired fallback 返回
  `['cape_town']`(首个命中),`lookup_region` 单数形式同样返回 `'cape_town'`
  以保持向后兼容;JHB 历史流程必须显式传 `--region`。
- 此前 plural 调用方(`review_detections` / `build_gemini_review_training_pool`)
  对 `len(hits)>1` 报错/跳过,改为 fallback to `hits[0]`(cape_town);JHB
  意图的调用从未依赖这条路径(必须 `--region jhb`)。
- `lookup_region` 单数形式在 active namespace 上 safe by construction
  (active namespaces 是 pairwise disjoint 不变量),原弃用 finding 以此
  关闭,不另加 DeprecationWarning(Python 默认隐藏)。

## Implementation status

| 项 | 状态 | 证据 |
|---|---|---|
| `retired_grid_id_patterns` 字段 + 加载(`RegionConfig`) | ✅ 2026-06-12 | `core/region_registry.py` |
| JHB `G\d{4}` / `JHB\d{2}` 退役 | ✅ 2026-06-12 | `configs/datasets/regions.yaml` johannesburg 节 |
| 两级解析 + 多 active UserWarning | ✅ 2026-06-12 | `lookup_regions(include_retired=)` |
| 不变量测试 + 行为测试(16 项) | ✅ 2026-06-12 | `tests/test_region_registry_namespace.py`;全仓 268 passed |
| CT `CPT\d{4}` regrid + task_grid | ✅ 2026-06-12 | `data/task_grid_cpt.gpkg` — 1103 covered cells kept / 1111 ocean cells dropped; digit-preserving G→CPT (G1240→CPT1240); WMS blank-probe coverage method (`threshold=0.05`), probe CSV at `results/analysis/ct_wms_coverage_probe/probe.csv`; 119 G-prefixed `aerial_2025` anchor grids all kept |
| G→CPT crosswalk | ✅ 2026-06-12 | `data/ct_grid_crosswalk_g_to_cpt.csv` — 1103 rows, digit-preserving, column: `g_id`, `cpt_id` |
| CT `G\d{4}` 退役 | ✅ 2026-06-12 | `configs/datasets/regions.yaml` cape_town 节 `retired_grid_id_patterns: ['^G\\d{4}$']` |

## 语义决策记录 (2026-06-12 实施期)

### TRAP A — `get_grid_spec` 在 region=None 时 KeyError

将 `cape_town.paths.task_grid` 指向 `data/task_grid_cpt.gpkg` 后,裸调用
`get_grid_record('G1240')` / `get_grid_spec('G1240')` 在 `region=None` 时
KeyError 崩溃:原因是 `rkey` 留 None,跳过了 scheme-fallback 块和跨区域聚合
(JHB unified grid 不含 CT G-ID)。

**修复**(`core/grid_utils.py` `get_grid_record`):当 `region=None` 时,在
task-grid 解析前先用 `region_registry.lookup_region(grid_id)` 推断 `rkey`。
这同时解决了重叠陷阱:`lookup_region('G1189') == 'cape_town'`(regions.yaml
注册顺序),裸 G1189 解析到 CT cell(lon 18.45, < 18.7),不会拿到 JHB cell
(lon 28.1);显式 `region='jhb'` 仍能正确拿到 JHB cell(验证 lon 28.1,CRS
32735)。传统 `data/task_grid.gpkg` (gao annotation-scheme grid,仍持有退役
G-cell,是 CPT 的 digit-preserving 来源)在 scheme-fallback 路径下仍可查询。

### TRAP B — 退役后双区域声明

CT 退役 `G\d{4}` 后,重叠 ID(G1189 等 6 个)在两区域均不再是 active namespace
的成员。采用 **HONEST-AMBIGUITY** 解释:
- `lookup_regions('G1189', include_retired=True)` 返回 `['cape_town', 'johannesburg']`(均通过 retired 层命中)
- `lookup_regions('G1189')` (无 include_retired)返回 `['cape_town']`(retired tier,首个命中,与之前行为兼容)
- `lookup_region('G1189')` 单数形式继续返回首个注册命中 `'cape_town'`,保持向后兼容
- CT 独有历史 ID(如 G1240)通过 retired 层仍返回 `['cape_town']`(仅 CT gao scheme 持有)
- 之前对 `len(hits) > 1` 报错/跳过的复数调用方(`review_detections`、`build_gemini_review_training_pool`)改为 fallback to `hits[0]`(`cape_town`),不再 block 合法 CT G-ID 流程
- `region_registry.py` 无需代码改动(retired 机制已于 2026-06-12 一起落地;CPT 流程通过 active CPT task grid 进行)

## Consequences

- 新城接入若 namespace 撞名,在测试层 fail fast + 运行时告警,不再静默
  mis-resolve。
- `grid_id_pattern`(区域可解析 ID 全集)语义不变,sync 脚本等消费者不受
  影响;retired 只影响 lookup 裁决层级。
- 历史 JHB G-ID 工具链(benchmark、25-grid eval、provenance 回溯)行为
  完全不变(retired 兜底命中同一结果)。
- CT regrid 已于 2026-06-12 落地:CT active namespace 为 `CPT\d{4}`,
  `G\d{4}` 进入 retired 层。历史 CT G-ID 工具链(标注、评估、negative pool
  provenance)通过 retired 兜底命中继续可用,行为完全不变。
