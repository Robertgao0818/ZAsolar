# Tiles & Results Restructure — 目录分层 + 影像源/模型显式化

> Status: DRAFT,待用户 review 后进入执行
> Created: 2026-04-19
> Related: `docs/plans/2026-04-12-pipeline-redesign-v1_2.md`(v1.2 显式 defer 了 results 重构,本 plan 解除这一 defer,因为错放问题已触发实际讨论混乱)

---

## 0. 背景与动机

当前 Joburg 相关产物分布在多个目录,**目录名只编码了"region",没编码"影像源/vintage/模型版本"**,导致三类问题:

1. **GEID 与 aerial 混淆**: `tiles_joburg/` 曾被 GEID 覆盖、后又被 aerial 覆盖回来;本地若无时间戳保护,无法区分当前内容是哪一版。事实上 GEID stitched mosaic 目前藏在 `/mnt/d/ZAsolar/tiles/joburg_geid/` 下,不在 Joburg 根目录内。
2. **Results 错位**: `results/G07xx-G09xx` 是 V3C on GEID 的 JHB 推理结果,按 `regions.yaml` 约定 `results/` 是 CT 根——JHB 结果错放在 CT 根下,且跟 `results_joburg/` 里的 V4 结果同名 grid,**内容不同但无任何 provenance 能区分**。
3. **讨论基线漂移**: 讨论 "Joburg V4 的 recall" 或 "CBD 上的 V3C 数字" 时,必须口头澄清 "是哪个目录下的哪次跑",无法让多方对齐。

**本 plan 的目标**: 让目录路径本身携带完整 provenance (region × imagery_source × vintage × model),让 `regions.yaml` 成为影像层和模型跑批的权威注册表,停止依赖口述知识。

---

## 🚨 Grid ID 跨区重叠 (Critical 设计约束)

PR1 扫描 tiles 目录发现:**CT 和 JHB task_grid 的数字 ID 严重重叠**。同一个 Grid ID (如 `G1189`, `G1293`, `G1513`, `G1570`, `G1630`) 在两个 region 都有真实的覆盖 tile,不是同一块地。

**影响:**
- `lookup_region(grid_id)` 不可靠 — 同一 ID 可属两个 region
- 任何 API 不能只接受 `grid_id`,必须显式 `(region, imagery_layer)` 或 `(region, model_run)`
- PR5 分类迁移 results **必须** 读 `config.json.tiles_dir` 路径判断归属,不能靠 grid_id 范围猜
- `region_registry.lookup_region()` 现在的实现(返回单一 region)要标记为 `deprecated,仅用于单区场景`;新代码用 `lookup_regions() -> list[str]`

**具体跨区 grid (PR3 迁移注意):**
- G1189, G1190, G1293, G1294, G1297 (CT + JHB aerial + JHB geid)
- G1513, G1515, G1516 (CT + JHB)
- G1570, G1571, G1572 (CT + JHB)
- G1630, G1631, G1632, G1633, G1634 (CT + JHB)

---

## 1. Inventory Freeze (事实盘点)

本节是**当前物理状态的冻结快照**,迁移前必须先与用户对齐。

### 1.1 Tiles

| 当前路径 | 内容 | 格式 | 来源 | Vintage | 备注 |
|---|---|---|---|---|---|
| `/mnt/d/ZAsolar/tiles/G{1189..1847}/` | 127 grids | chunked (`G_0_0_geo.tif` x N) | CT aerial | 2023? | 正确位置 |
| `/mnt/d/ZAsolar/tiles/JHB{01..06}/` | 6 grids | chunked | JHB aerial legacy | — | Legacy pilot,早期 Li 手工 GT 对应 |
| `/mnt/d/ZAsolar/tiles/joburg_geid/G*_mosaic.tif` | 100 files | **mosaic (single file)** | JHB GEID | 2024-02 | 藏在 CT 根下,**格式不同于 aerial** |
| `/mnt/d/ZAsolar/tiles/__pycache__/` | — | — | — | — | 垃圾,删 |
| `/mnt/d/ZAsolar/tiles_joburg/G*/` | 100 grids | chunked | JHB aerial | 2023 | 刚重下,已恢复 |
| `/mnt/d/ZAsolar/tiles_jhb_test/G0772/` | 1 grid | chunked | — | — | staging 残留,删 |
| `/mnt/d/ZAsolar/geid_raw/{joburg_cbd_geid,joburg_geid,joburg_geid_python}/` | 原始 capture | GEID task 文件 | JHB GEID raw | 2024-02 | 保留,作为 stitching 上游 |

### 1.2 Results

| 当前路径 | Grid 范围 | 模型 | 底图 | 记录时间 | 备注 |
|---|---|---|---|---|---|
| `<repo>/results/G{1189..1847}` (约 103 grids) | CT | V3-C / V4 / ... | CT aerial | 多批次 | CT 正确 |
| `<repo>/results/G{0772..0926}` (25 grids) | JHB CBD | **V3-C (exp003_C)** | **JHB GEID 2024-02** | 2026-04-08 | ❌ JHB 错放在 CT 根下 |
| `<repo>/results/G{1110..1XXX}` (75 grids) | JHB 外围 | 空? 需核 | — | — | **需核:本地 results/ 里有没有这 75 个** |
| `<repo>/results_joburg/G{0772..1XXX}` (100 grids) | JHB all | **V4 (exp004_v4_hn)** | **JHB aerial 2023** | 2026-04-05 | 正确位置,但未标注底图 |
| `/mnt/d/ZAsolar/results/G{18xx..19xx}` | CT | (核) | — | — | 另一批 CT,本地 results/ 无 |

**权威来源**: 每个 `results/<grid>/config.json` 里的 `model_path` 和 `tiles_dir` 字段是真相来源。迁移脚本必须读 config.json 分类,不能靠路径猜。

### 1.3 开放问题 (迁移前必须确认)

- **Q1.** `results/` 里 G1110-G1XXX 的 75 个 JHB 外围 grid 实际存在吗? (第 1 轮盘点说 local `results/` 203 条, 100 条跟 `results_joburg/` 重合 → 103 条 CT 独有。但你说 "V3C CBD 25 + V4 剩下 75",总共 100 个 JHB,都在 `results_joburg/`。所以 `results/` 可能只含 25 个错放的 V3C CBD,而不是全部 100)
- **Q2.** `/mnt/d/ZAsolar/results/` 里 G18xx/G19xx 是哪个批次、什么模型? 本 plan 默认按 "CT V?" 处理,但具体需你标注。
- **Q3.** GEID mosaic 格式 (`G0772_mosaic.tif` 单文件) vs aerial chunked 格式——迁移后**保留双格式**还是**统一**成一种?
  - 保留双格式: MANIFEST 声明 `file_layout`,`detect_and_evaluate` 读取分支 → 迁移最轻
  - 统一成 chunked: 需重新切分 100 个 GEID mosaic → 工作量大但后续更统一
  - **建议先保留双格式**,后续评估是否统一。

---

## 2. Target 结构

### 2.1 物理目录

```
/mnt/d/ZAsolar/
├── tiles/
│   ├── cape_town/
│   │   └── aerial_2025/                    # 原 /tiles/G*/ 整体 (CT aerial 2025)
│   │       ├── MANIFEST.json
│   │       └── G1189/G1189_0_0_geo.tif ...
│   └── johannesburg/
│       ├── aerial_2023/                    # 原 /tiles_joburg/
│       │   ├── MANIFEST.json
│       │   └── G0772/G0772_0_0_geo.tif ...
│       ├── aerial_legacy_2020/             # JHB01-06 (需确认 vintage)
│       │   ├── MANIFEST.json
│       │   └── JHB01/ ...
│       └── geid_2024_02/                   # 原 /tiles/joburg_geid/
│           ├── MANIFEST.json
│           └── G0772_mosaic.tif ...        # 保留 mosaic 格式
├── results/
│   ├── cape_town/
│   │   └── <model_version>/                # e.g. v3c_targeted_hn / v4_hn / v4_1_hn
│   │       ├── RUN_MANIFEST.json
│   │       └── G1238/...
│   └── johannesburg/
│       ├── v3c_geid_2024_02/               # 原 results/G07xx-G09xx (25 grids)
│       │   ├── RUN_MANIFEST.json
│       │   └── G0772/...
│       └── v4_aerial_2023/                 # 原 results_joburg/ (100 grids)
│           ├── RUN_MANIFEST.json
│           └── G0772/...
└── geid_raw/                               # 不动,raw capture 保留原样
```

### 2.2 MANIFEST.json 结构

**Tile layer MANIFEST** (`tiles/<region>/<layer>/MANIFEST.json`):
```json
{
  "region": "johannesburg",
  "imagery_layer_id": "geid_2024_02",
  "source": "geid",
  "vintage": "2024-02",
  "crs": "EPSG:4326",
  "file_layout": "mosaic",
  "file_pattern": "{grid_id}_mosaic.tif",
  "provenance": "GEID 6.48 RE captures, stitched via scripts/imagery/chip_mosaic.py at 2026-04-XX",
  "coverage_grids": ["G0772", "G0773", "..."],
  "created_at_utc": "2026-04-19T..."
}
```

**Results run MANIFEST** (`results/<region>/<model_run>/RUN_MANIFEST.json`):
```json
{
  "region": "johannesburg",
  "model_run_id": "v3c_geid_2024_02",
  "model_version": "exp003_C_targeted_hn",
  "model_path": "checkpoints/exp003_C_targeted_hn/best_model.pth",
  "imagery_layer_id": "geid_2024_02",
  "postproc_config": "configs/postproc/v4_canonical.json",
  "inference_date_utc": "2026-04-08",
  "coverage_grids": ["G0772", "..."],
  "notes": "CBD 25-grid V3-C inference on GEID for batch1 review"
}
```

### 2.3 `regions.yaml` 扩展

新增 `imagery_layers:` 和 `model_runs:` 两段:

```yaml
regions:
  johannesburg:
    # ... 保留原字段 ...
    paths:
      tiles_root: "tiles/johannesburg"       # 改:指向新结构
      results_root: "results/johannesburg"   # 改
      annotations_dir: "data/annotations/Joburg"
      task_grid: "data/jhb_task_grid.gpkg"

    imagery_layers:
      aerial_2023:
        path: "tiles/johannesburg/aerial_2023"
        source: "aerial"
        vintage: "2023"
        file_layout: "chunked"
        crs: "EPSG:4326"
        coverage_grids: [G0772, G0773, ..., G1XXX]
      geid_2024_02:
        path: "tiles/johannesburg/geid_2024_02"
        source: "geid"
        vintage: "2024-02"
        file_layout: "mosaic"
        crs: "EPSG:4326"
        coverage_grids: [G0772, ..., G0926, G1110, ...]
      aerial_legacy_2020:
        path: "tiles/johannesburg/aerial_legacy_2020"
        source: "aerial"
        vintage: "2020?"
        file_layout: "chunked"
        coverage_grids: [JHB01, JHB02, ..., JHB06]
    default_imagery_layer: "aerial_2023"     # 不指定 layer 时的回退

    model_runs:
      v3c_geid_2024_02:
        model_version: "exp003_C_targeted_hn"
        imagery_layer: "geid_2024_02"
        results_path: "results/johannesburg/v3c_geid_2024_02"
        inference_date: "2026-04-08"
      v4_aerial_2023:
        model_version: "exp004_v4_hn"
        imagery_layer: "aerial_2023"
        results_path: "results/johannesburg/v4_aerial_2023"
        inference_date: "2026-04-05"
```

---

## 3. 代码迁移 Surface

### 3.1 `core/region_registry.py` 扩展

新增 API:
- `list_imagery_layers(region_key) -> list[str]`
- `get_imagery_layer(region_key, layer_id) -> ImageryLayerConfig`
- `get_default_imagery_layer(region_key) -> str`
- `resolve_imagery_layer_for_grid(grid_id, region_key) -> str` (按 `coverage_grids` 反查,返回 default_imagery_layer 或 raise)
- `list_model_runs(region_key) -> list[str]`
- `get_model_run(region_key, run_id) -> ModelRunConfig`

### 3.2 `core/grid_utils.py` 扩展

新增签名 (保留旧签名作为 shim,内部 route 到新 API):
- `resolve_tiles_dir(grid_id, *, region=None, imagery_layer=None) -> Path`
  - `imagery_layer` 未指定 → 走 registry default
  - 返回带 `file_layout` 提示的 dataclass? 或者约定 `tiles_dir.parent / MANIFEST.json` 可读
- `get_results_root(*, region=None, model_run=None) -> Path`
  - `model_run` 未指定 → 返回 region results_root (后续必须指定 run 才能定位具体 grid)
- `get_grid_paths(grid_id, *, region=None, imagery_layer=None, model_run=None) -> GridPaths`

### 3.3 `detect_and_evaluate.py` 入参

新增 CLI:
- `--imagery-layer <layer_id>` (可选,默认走 registry default)
- `--model-run <run_id>` (可选,用于把输出写到对应 results/<region>/<run_id>/<grid>/)

向后兼容: 旧命令 `python detect_and_evaluate.py --grid-id G0772` 仍能跑,但会在 config.json 里写入解析出的 `imagery_layer_id` 和 `model_run_id` 字段。

### 3.4 Hardcoded paths 待扫清单

已 grep 到的文件 (10 个,不完整,迁移前需重新 grep):
```
scripts/imagery/chip_mosaic.py
scripts/annotations/sam_fn_review.py
scripts/analysis/sam_recut_joburg.py
scripts/imagery/download_jhb_tiles.py
docs/joburg_batch1_plan.md        # 文档,扫过但迁移时一起更新
docs/progress_log/week_2026-04-14/2026-04-14.md  # 历史日志,不改
docs/progress_log/week_2026-04-07/2026-04-13.md  # 历史日志,不改
configs/datasets/regions.yaml     # 本 plan 要改
core/grid_utils.py                # 本 plan 要改
.claude/rules/06-multi-city.md    # 规则文件,扫过确认
```

迁移前需再跑一遍 grep 保证覆盖所有引用。

---

## 4. 物理迁移策略

### 4.1 分阶段 + 过渡期

**Phase A (单次大迁移,一次性搞定)** — 不建议:风险大,容易留脏数据。

**Phase B (分 3 步,每步有回滚点)** — 推荐:

- **B1. Tiles 迁移 (原子步骤)**
  1. 创建目标结构: `tiles/cape_town/aerial_2025/`, `tiles/johannesburg/{aerial_2023,aerial_legacy_{year},geid_2024_02}/`
  2. 用 `mv` 逐目录移动 (D 盘内移动是 rename,秒级)
  3. 写入每层 MANIFEST.json
  4. 旧路径留 **symlink** 指向新路径 (`tiles_joburg` → `tiles/johannesburg/aerial_2023` 等),过渡 2 周
  5. 验证一个 grid 可读 (抽 G0772 GEID + G1110 aerial 各跑一次 `rasterio.open`)

- **B2. Results 迁移 (按模型 run 分组)**
  1. 扫 `results/G*/config.json` 和 `results_joburg/G*/config.json`,按 `model_path` 分类
  2. 创建 `results/johannesburg/{v3c_geid_2024_02, v4_aerial_2023}/`
  3. 批量 `mv` 按分类结果
  4. 写 RUN_MANIFEST.json
  5. CT 那边同样: `results/` CT 部分 → `results/cape_town/<run>/`
  6. `/mnt/d/ZAsolar/results/` 的 CT 那批也迁入统一结构
  7. 保留旧 symlink

- **B3. 代码 + 配置迁移**
  1. 扩 `regions.yaml`
  2. 扩 `region_registry.py` + `grid_utils.py`
  3. 改 10 个已知脚本的硬编码路径
  4. 更新 `docs/architecture.md` (同 commit,按 rule 03-doc-sync)
  5. 更新 `.claude/rules/06-multi-city.md` 加入 imagery_layer 规则
  6. 跑 end-to-end 验证 (§5)
  7. 2 周后删除旧 symlink

### 4.2 RunPod 侧

RunPod 上的 `/workspace/tiles_joburg/` 和 `/workspace/ZAsolar/results*/` 也需要同步迁移,但可以**在本地迁移验证后再处理**,通过 `scripts/sync_from_runpod.sh` 反向更新,或下次拉取时直接按新结构拉。

### 4.3 不动的东西

- `geid_raw/` — 上游原始 capture,不挪
- `data/annotations/` — 标注 GPKG 不动 (region 已按 `Capetown/` `Joburg/` 分,够用)
- `checkpoints/` — 模型权重不动
- `mosaics/` (repo 内) — 只有 G0854 两个文件,评估后清掉或挪入 `tiles/johannesburg/geid_2024_02/`

---

## 5. 验证方案

### 5.1 端到端 smoke test

迁移后必须跑通 3 组 smoke test,确认 3 个典型推理路径没坏:

```bash
# 1. CT aerial (regression)
python detect_and_evaluate.py \
  --grid-id G1238 \
  --model-path checkpoints/exp003_C_targeted_hn/best_model.pth \
  --postproc-config configs/postproc/v4_canonical.json --force

# 2. JHB aerial V4 (regression)
python detect_and_evaluate.py \
  --grid-id G0772 --region jhb \
  --imagery-layer aerial_2023 \
  --model-path checkpoints/exp004_v4_hn/best_model.pth \
  --postproc-config configs/postproc/v4_canonical.json --force

# 3. JHB GEID V3-C (regression,mosaic 格式)
python detect_and_evaluate.py \
  --grid-id G0772 --region jhb \
  --imagery-layer geid_2024_02 \
  --model-path checkpoints/exp003_C_targeted_hn/best_model.pth \
  --postproc-config configs/postproc/v4_canonical.json --force
```

三组都必须:
- 输出写入正确的 `results/<region>/<run_id>/<grid>/`
- `config.json` 里 `imagery_layer_id` 和 `model_run_id` 字段已填
- 结果数量跟迁移前 snapshot 一致 (抽 grid 检查)

### 5.2 Snapshot compare

迁移**前**: 记录每个 grid 的 `result_count` 和 `predictions_metric.gpkg` 的 row count 到 `docs/plans/joburg_pre_migration_snapshot.csv`。
迁移**后**: 重新扫一遍,diff 应为空 (除了路径变化)。

### 5.3 Config.json 向前兼容

历史 `config.json` 里的 `tiles_dir: /dev/shm/tiles_joburg_geid/G0772` 之类路径,迁移后**不改**(它是历史事实记录)。只在新跑的 run 里用新字段。

---

## 6. 执行分解 (建议 PR 切分)

| PR# | 标题 | 范围 | 前置依赖 |
|---|---|---|---|
| 1 | Inventory snapshot + open questions 澄清 | `docs/plans/joburg_pre_migration_snapshot.csv`, 更新本 plan 的 §1.3 | — |
| 2 | `regions.yaml` schema 扩展 + registry API | `configs/datasets/regions.yaml`, `core/region_registry.py`, 新增测试 | PR1 |
| 3 | Tiles 物理迁移 + MANIFEST | 物理 `mv`, 写 MANIFEST, 旧路径 symlink | PR2 |
| 4 | `grid_utils.py` 扩展 + `detect_and_evaluate.py` CLI | 新 `imagery_layer` 入参, 旧 API shim | PR2, PR3 |
| 5 | Results 物理迁移 + RUN_MANIFEST | 按 config.json 分类 mv, 写 RUN_MANIFEST | PR3, PR4 |
| 6 | 扫清 10 个 hardcoded path 脚本 | `scripts/imagery/*`, `scripts/annotations/*`, `scripts/analysis/*` | PR4, PR5 |
| 7 | 文档同步 + 规则更新 | `docs/architecture.md`, `.claude/rules/06-multi-city.md`, `CLAUDE.md` Key References | PR6 |
| 8 | 端到端验证 + snapshot diff + 删除旧 symlink | smoke test 3 组 + snapshot 比对 | PR1-7 |

PR 切分的原因: 每个 PR 独立可回滚,PR3/PR5 是物理迁移最危险的两步,必须在 PR2/PR4 的代码到位后才做。

---

## 7. 风险与回滚

| 风险 | 影响 | 回滚 |
|---|---|---|
| 物理 `mv` 中断 | tile 丢失 | `mv` 单向,中断后部分数据在新位置部分在旧位置。**对策**: D 盘内部 `mv` 是原子 rename,一个目录一次成功或失败,整体脚本用 `set -e`,任何一步失败立即停 |
| 旧 symlink 被过早删除 | 仍指向旧路径的脚本/文档断裂 | 保留 2 周,在 PR8 之后才删。删除前用 `grep -r` 最后扫一次 |
| RunPod 侧仍按旧路径拉 | 远端跑任务时找不到 tile | `scripts/sync_from_runpod.sh` 和 `scripts/runpod_pod.sh` 需在 PR6 里同步更新 |
| 未发现的 hardcoded 路径 | 跑到运行时才报错 | PR8 smoke test 跑 3 组推理时能覆盖大多数场景 |
| `results/G1110-G1XXX` 那 75 个 grid 实际位置跟盘点不符 | 迁移脚本分类错 | PR1 的 snapshot 脚本必须读每个 `config.json` 的 `model_path`,不靠 grid id 范围猜 |

---

## 8. Definition of Done

1. `regions.yaml` 包含 `imagery_layers` + `model_runs` 两段,通过 schema 校验。
2. 物理目录按 §2.1 结构重组,每个 layer/run 有 MANIFEST。
3. `core/region_registry.py` 和 `core/grid_utils.py` 新 API 可用,旧 API shim 仍工作。
4. `detect_and_evaluate.py` 接受 `--imagery-layer` 和 `--model-run`,不指定时走 registry default。
5. 10 个已知 hardcoded 脚本已改用 registry API。
6. `docs/architecture.md` 与新结构同步。
7. 3 组 smoke test 全部通过,snapshot diff 为空。
8. 新讨论中任何 "Joburg V4 的结果" 都能一句话 locate 到 `results/johannesburg/v4_aerial_2023/<grid>/`,不再需要口述澄清。

---

## 9. 不在本 plan 范围

- 重新切分 GEID mosaic 为 chunked 格式 (§1.3 Q3 保留双格式)
- COCO 数据集产物 (`coco_*/`) 的目录重构 —— v1.2 pipeline-redesign 接管
- checkpoints 目录重构
- 新增其他城市 (Durban 等) 的 registry 扩展 —— 另起 plan

---

## 10. 待你 (用户) 回答

1. §1.3 Q1: local `results/` 里 G11xx-G18xx 的 JHB 外围 grid 实际有吗? (迁移前用脚本扫 config.json 能最终确认)
2. §1.3 Q2: `/mnt/d/ZAsolar/results/G18xx` 那批 CT 结果是哪个模型?
3. §1.3 Q3: GEID mosaic 格式保留还是统一到 chunked?(建议保留)
4. §2.3 `aerial_legacy_2020` 的 vintage 实际是哪一年? JHB01-06 的底图是什么时候的?
5. §4.1 物理迁移是 1 个 session 内全部做完,还是分几天?
6. §6 PR 切分是否 OK? 还是希望合并成更少的 PR?

回答完这些我就按 PR1 开始执行。
