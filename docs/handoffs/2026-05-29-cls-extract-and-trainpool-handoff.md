# Handoff — 分类器抽出 (solar_cls 子仓) + 训练集资源规范化 (Phase 0–4)

**创建**: 2026-05-29 · **状态**: 已完成全部调研与设计、决策已拍板，**尚未动任何代码**(工作树干净，仅新增了 `docs/plans/2026-05-29-codebase-optimization.md`)。
**给新窗口的指令**: 本文件自包含。按下方步骤执行即可，无需重新调研。所有行号/路径/耦合事实均已核实(2026-05-29)。配套上游文档:
- 总优化方案: [`docs/plans/2026-05-29-codebase-optimization.md`](../plans/2026-05-29-codebase-optimization.md)
- 注释规范(Two-Axis): [`data/annotations/ANNOTATION_SPEC.md`](../../data/annotations/ANNOTATION_SPEC.md)
- 子仓模板参照: `/home/gaosh/projects/solar_backdating/`(已落地的 sibling-subrepo 范例)

---

## 0. 已锁定的决策

| # | 决策 | 选择 |
|---|---|---|
| Q1 | 分类器落点 | **兄弟子仓** `/home/gaosh/projects/solar_cls/`(镜像 solar_backdating,共享 ZAsolar `.venv` + PYTHONPATH,read-only import `core.*`) |
| Q2 | 训练集规范化范围 | **Phase 0–4**(含 DatasetSpec v2 spec 迁移) |
| Q3 | 正样本分桶 | **2 桶**:trusted / untrusted,唯一依据 `label_source→mask_trusted`;`quality_tier`(T1/T2) 与 `semantic_confidence`(A1/A2/A3) 作 join 列保留 |
| Q4 | pool / ledger 存放 | **索引进 git / 产物出 git**:manifest(CSV/YAML) + `configs/training_runs.yaml` 进 git;COCO chip 产物 + `runs/<id>/` 详档进 `~/zasolar_data/` |

两个小默认(已与用户确认随大流):① 子仓走新仓 `git init` + 主仓 `git rm`(不保 classifier 文件 commit 历史);② 趁搬迁顺手修 P0-4(classify_predictions mosaic/多城市感知 + thresholds_v2 消费)。

---

## 1. 执行模型(如何用 subagent)

新窗口作为 **driver**。原则:**有状态的 git/文件系统操作 driver 亲自串行做(并行会撞 git index);独立的代码生成/校验交给 subagent。**

**Track 1(分类器抽出)— 串行,driver 亲自做**,理由:涉及 `mv`、`git rm`、新建 repo、跨文件 import 重写,必须精确排序。完成后 dispatch **1 个验证 subagent**:
```
Agent(subagent_type="general-purpose", description="verify cls extraction", prompt=
"验证 /home/gaosh/projects/solar_cls 抽取无回归:
1. cd /home/gaosh/projects/ZAsolar && grep -rn 'from scripts.classifier\\|import.*build_cls_dataset\\|cls_nonpv_subtype\\|label_cls_nonpv' --include=*.py . | grep -v _archive — 应为空(分类器已不在主仓被引用)。
2. cd /home/gaosh/projects/ZAsolar && python -m pytest -q 2>&1 | tail -15 — 主仓测试应仍绿(分类器测试已移走)。
3. cd /home/gaosh/projects/solar_cls && source scripts/activate_env.sh && python -m pytest tests/classifier/ -q 2>&1 | tail -20 — 子仓 2 个测试应绿。
4. grep -rn 'relative_to(PROJECT_ROOT)' /home/gaosh/projects/solar_cls/scripts — 应为空(results_path 应改为 relative_to(ZASOLAR_ROOT))。
5. cd /home/gaosh/projects/solar_cls && source scripts/activate_env.sh && python scripts/classifier/build_cls_dataset.py --help 和 classify_predictions.py --help — 应无 ImportError。
逐条报告 pass/fail + 证据。")
```

**Track 2(训练集规范化)— 分 phase,每 phase: 实现 → 对抗式校验**。Phase 0/2/3 较小可 driver 直接做;Phase 1(build_training_pool.py)和 Phase 4(DatasetSpec v2)较大,建议各派 1 个实现 subagent + 1 个校验 subagent。若开 ultracode,可用 `Workflow` 把 Phase 1–4 串成 pipeline(实现→校验),但**Phase 间有严格依赖,必须顺序执行,不可并行**。

> ⚠️ 全程不要并行编辑同一文件;不要在 Track 1 完成并验证前开始 Track 2(train.py 在两个 track 都要改)。

---

## 2. Track 1 — 分类器抽出到 `solar_cls`

### 2.1 已核实的事实(无需重查)

**隔离性(三重确认)**: 分类器不在任何生产推理路径。`detect_and_evaluate.py:1914-2021` 的 `--classifier-*` flag 只是接收外部 GPKG 的 file-path passthrough,**不 import 任何 classifier 模块**;同名 `classify_predictions()` 函数(`detect_and_evaluate.py:1194`)是无关的 TP/FP-vs-GT labeler(命名碰撞)。**抽取后 `detect_and_evaluate.py` 原地不动。**

**要搬的文件**(全部 git-tracked):
| 源(ZAsolar) | 目标(solar_cls) |
|---|---|
| `scripts/classifier/*.py` (9 py + 2 sh: audit_cls_sources 423, build_cls_dataset 827, build_cls_dataset_v2 350, calibrate_v2_thresholds 291, classify_predictions 492, compare_results 255, dedup_cls_pool 261, propagate_subtype_to_v42 154, train_cls 596, run_cascade_sweep.sh, run_v2_training.sh) | `scripts/classifier/` |
| `scripts/analysis/cls_nonpv_subtype_audit.py` (334), `scripts/analysis/label_cls_nonpv_subtype.py` (475) | `scripts/analysis/` |
| `scripts/runpod_train_cls_queue.sh` (87), `scripts/runpod_setup_cls_env.sh` (57) | `scripts/runpod/` |
| `tests/classifier/{__init__.py, test_model_factory.py 114, test_registry_discovery.py 83}` | `tests/classifier/` |
| `configs/classifier/{cls_pv_thermal,convnext_tiny,dinov2_vits14,efficientnet_b0}.json, thresholds_v2.json` | `configs/classifier/` |
| `docs/experiments/exp_cls_{backbone_ablation,dataset_protocol,detector_integration}.md`, `docs/experiments/cls_training_registry.json` | `docs/` |

**留在主仓**: `scripts/training/negative_pool/bootstrap_from_cls_v2.py`(属 detector negative-pool 域;只读 cls CSV,见 2.5)。

**数据 / checkpoint**(~3.4 GB,全 gitignored;`~/zasolar_data` 与 repo 同一 ext4 文件系统 → `mv` 是瞬时 rename):
- `data/cls_pv_thermal` (187M), `data/cls_pv_thermal_v1` (688M), `data/cls_pv_thermal_v2` (35M), `data/cls_pv_nonpv_v42_cascade` (33M), `data/cls_pv_nonpv_v3c_v42_cascade` (372M)
- `checkpoints/cls_*`(7 个目录 ~2.1 GB)
- **git-tracked 小 manifest**(要在 solar_cls 里作为 real file 提交,保 provenance):`cls_pv_nonpv_v3c_v42_cascade/{build_meta.json, manifest.csv, labeler/v3c__both/*.csv(5 个)}`、`cls_pv_thermal_v1/dataset_manifest.json`、`cls_pv_thermal_v2/{dataset_manifest.json, subtype_labels.csv}`

**耦合(出向)**: 仅 3 文件 import core(已核实是 solar_backdating 已声明 surface 的子集):
- `build_cls_dataset.py:46-47`: `from core import region_registry` + `from core.grid_utils import TILES_ROOT, resolve_tiles_dir`
- `audit_cls_sources.py:36`: `from core import region_registry`
- `classify_predictions.py:42`: `from core.grid_utils import TILES_ROOT`

**耦合(入向)**: 2 个 analysis 文件 import `build_cls_dataset` 内部符号(随之搬走即解);1 个 CSV file-read(bootstrap,2.5 处理)。

### 2.2 Step A — 搭子仓骨架

```bash
mkdir -p /home/gaosh/projects/solar_cls/{scripts/{classifier,analysis,runpod},configs/classifier,tests/classifier,docs,data,checkpoints}
touch /home/gaosh/projects/solar_cls/scripts/__init__.py
```
从 solar_backdating **复制并改名** 这些骨架文件(逐个改 `solar_backdating`→`solar_cls`、`SOLAR_BACKDATING_ROOT`→`SOLAR_CLS_ROOT`、`scripts.temporal`→`scripts.classifier`):
- `scripts/activate_env.sh`(复制 `/home/gaosh/projects/solar_backdating/scripts/activate_env.sh`:改 ROOT 变量名;PYTHONPATH 行改为 `export PYTHONPATH="$SOLAR_CLS_ROOT:${PYTHONPATH}"`,**不需要 `/src`**,本子仓无 `src/` 包)
- `scripts/link_data_dirs.sh`(复制并改写,见 2.4 的 symlink 清单)
- `.gitignore`(复制 solar_backdating 的,把 `data/geid_*` 段替换为下方 2.4 的 cls 数据/symlink 规则 + `checkpoints/`)
- `pyproject.toml`(复制,改 name=`solar-cls`、description;`dependencies` 加 `torch, torchvision, timm, opencv-python-headless, scikit-learn, pycocotools`;`[tool.setuptools.packages.find]` 删掉 `where=["src"]` 或改为不打包)
- `README.md` / `CLAUDE.md` / `AGENTS.md` / `SHARED_FROM_ZASOLAR.md`(复制 solar_backdating 对应文件,**改写 identity 段**:见 2.7 的文案要点)

### 2.3 Step B — 搬代码 + 改 import

跨仓不能 `git mv` 保历史(决策不保历史)。做法:`cp` 进 solar_cls,稍后在主仓 `git rm`。
```bash
SRC=/home/gaosh/projects/ZAsolar ; DST=/home/gaosh/projects/solar_cls
cp -a $SRC/scripts/classifier/. $DST/scripts/classifier/
cp -a $SRC/scripts/analysis/cls_nonpv_subtype_audit.py $SRC/scripts/analysis/label_cls_nonpv_subtype.py $DST/scripts/analysis/
cp -a $SRC/scripts/runpod_train_cls_queue.sh $SRC/scripts/runpod_setup_cls_env.sh $DST/scripts/runpod/
cp -a $SRC/tests/classifier/. $DST/tests/classifier/
cp -a $SRC/configs/classifier/. $DST/configs/classifier/
cp -a $SRC/docs/experiments/exp_cls_*.md $SRC/docs/experiments/cls_training_registry.json $DST/docs/
```

**Import / 路径重写**(在 `$DST` 拷贝上做)。新增 `ZASOLAR_ROOT` 锚定主仓产物(results / annotations)。模式(每个改的文件顶部,在 `import sys`后加 `import os`,在 PROJECT_ROOT 定义后加):
```python
import os  # 若文件尚无
# PROJECT_ROOT = ...parent.parent.parent  → 现在 = solar_cls root(自有 data/configs/输出)
ZASOLAR_ROOT = Path(os.environ.get("ZASOLAR_ROOT", "/home/gaosh/projects/ZAsolar"))  # 主仓产物(results/annotations)
```
逐文件:
- **`scripts/classifier/build_cls_dataset.py`**(行号为搬前的): 加 `import os` + `ZASOLAR_ROOT`;`50` `RESULTS_DIR = PROJECT_ROOT/"results"`→`ZASOLAR_ROOT/"results"`;`229/318/400` `str(src.results_path.relative_to(PROJECT_ROOT))`→`relative_to(ZASOLAR_ROOT)`(replace_all,3 处同串);`338` `PROJECT_ROOT / config.paths.annotations_dir`→`ZASOLAR_ROOT / config.paths.annotations_dir`;`707/715` `PROJECT_ROOT / "results" / "analysis" / ...`→`ZASOLAR_ROOT / ...`;`690` `--output-dir` 默认 `PROJECT_ROOT/"data"/...` **保持不变**(自有 data)。
- **`scripts/classifier/audit_cls_sources.py`**: 加 `os`+`ZASOLAR_ROOT`;`39` RESULTS_DIR→ZASOLAR_ROOT;`142` `relative_to(PROJECT_ROOT)`→ZASOLAR_ROOT;`220/240` 读 main-repo gt_heater_audit / small_fp 的 `relative_to(PROJECT_ROOT)`→ZASOLAR_ROOT;`408` inventory 输出默认可留 `PROJECT_ROOT`(本地诊断输出)。CLI 默认指向 results/analysis 的**输入** CSV → ZASOLAR_ROOT。
- **`scripts/classifier/classify_predictions.py`**: 加 `os`+`ZASOLAR_ROOT`;`45` RESULTS_DIR→ZASOLAR_ROOT(读 detector 预测 gpkg);**P0-4 见 2.6**。
- **`scripts/analysis/cls_nonpv_subtype_audit.py`**: `36` `from scripts.classifier.build_cls_dataset import (..., PROJECT_ROOT, ...)` 改为同时 import `ZASOLAR_ROOT`;`46` small_fp 输入路径用 `ZASOLAR_ROOT`;`44` AUDIT_ROOT 输出留 `PROJECT_ROOT`(本地)。
- **`scripts/analysis/label_cls_nonpv_subtype.py`**: `44` import 加 `ZASOLAR_ROOT`;`109` `results_path = PROJECT_ROOT / r["results_path"]`→`ZASOLAR_ROOT / r["results_path"]`(r["results_path"] 是相对主仓存的);`53` AUDIT_ROOT 留本地。
- **tests**(`test_model_factory.py:18-21`, `test_registry_discovery.py:19-22`): import 路径 `from scripts.classifier import ...` **不用改**(solar_cls 里 `scripts/classifier` 仍在);`PROJECT_ROOT=parent.parent.parent`+`sys.path.insert` 仍指 solar_cls root,OK。`core` 经 activate_env.sh 的 PYTHONPATH 解析 → **测试前必须 `source scripts/activate_env.sh`**。
- **`calibrate_v2_thresholds.py`** / 其余脚本: 只 import 同包,无 core,基本不改;`thresholds_v2.json` 路径默认 `PROJECT_ROOT/"configs"/"classifier"` 保持(自有 configs)。

> 注: `region_registry` / `grid_utils` 的 `src.results_path`、`config.paths.annotations_dir` 由 region_registry 从**它自己的 __file__**(主仓)解析为绝对路径或相对值;改成 `relative_to(ZASOLAR_ROOT)` / `ZASOLAR_ROOT / ...` 即对。改完务必跑 2.1 验证 subagent 的第 4 条确认无残留 `relative_to(PROJECT_ROOT)`。

### 2.4 Step C — 搬数据 + symlink(产物出 git,manifest 进 git)

```bash
mkdir -p ~/zasolar_data/cls/{data,checkpoints}
# 整目录瞬时 rename(同一 ext4)
for d in cls_pv_thermal cls_pv_thermal_v1 cls_pv_thermal_v2 cls_pv_nonpv_v42_cascade cls_pv_nonpv_v3c_v42_cascade; do
  mv /home/gaosh/projects/ZAsolar/data/$d ~/zasolar_data/cls/data/$d
done
mv /home/gaosh/projects/ZAsolar/checkpoints/cls_* ~/zasolar_data/cls/checkpoints/
```
在 solar_cls 重建:**小 manifest = real file(进 git);大块子目录 = symlink(出 git)**。对每个数据目录:
```bash
DST=/home/gaosh/projects/solar_cls ; Z=~/zasolar_data/cls/data
# 例: cls_pv_thermal_v2
mkdir -p $DST/data/cls_pv_thermal_v2
cp -a $Z/cls_pv_thermal_v2/dataset_manifest.json $Z/cls_pv_thermal_v2/subtype_labels.csv $DST/data/cls_pv_thermal_v2/   # 进 git
ln -s $Z/cls_pv_thermal_v2/train $DST/data/cls_pv_thermal_v2/train                                                     # 出 git
ln -s $Z/cls_pv_thermal_v2/val   $DST/data/cls_pv_thermal_v2/val
```
按各目录的 tracked-manifest 清单(见 2.1)重复:`cls_pv_thermal_v1/dataset_manifest.json`、`cls_pv_nonpv_v3c_v42_cascade/{build_meta.json,manifest.csv,labeler/}` 进 git,其余子目录(`train/val/chips/manifest.gpkg`)symlink。`cls_pv_thermal`(只有未 tracked 的 dataset_meta.json)、`cls_pv_nonpv_v42_cascade`(无 manifest)整目录 symlink 即可。
checkpoints 全 symlink:`for c in ~/zasolar_data/cls/checkpoints/cls_*; do ln -s $c $DST/checkpoints/$(basename $c); done`。
solar_cls `.gitignore` 必须 ignore:`data/**/train`, `data/**/val`, `data/**/chips`, `*.gpkg`, `checkpoints/`, 以及所有 symlink 指向的大块(简单起见 ignore `*.png *.jpg *.tif *.pkl *.pth` + `checkpoints/`)。把上面 `cp` 的小 manifest 用 `git add -f` 强制纳入(若被通配 ignore)。

### 2.5 Step D — 主仓清理 + bootstrap 接缝

```bash
cd /home/gaosh/projects/ZAsolar
git rm scripts/classifier/* scripts/analysis/cls_nonpv_subtype_audit.py scripts/analysis/label_cls_nonpv_subtype.py \
       scripts/runpod_train_cls_queue.sh scripts/runpod_setup_cls_env.sh \
       tests/classifier/* configs/classifier/* \
       docs/experiments/exp_cls_*.md docs/experiments/cls_training_registry.json \
       $(git ls-files data/cls_pv_thermal data/cls_pv_thermal_v1 data/cls_pv_thermal_v2 data/cls_pv_nonpv_v42_cascade data/cls_pv_nonpv_v3c_v42_cascade)
```
**bootstrap 接缝**: `scripts/training/negative_pool/bootstrap_from_cls_v2.py:26,31-32` 读 `data/cls_pv_thermal_v2/{subtype_labels.csv,train/non_pv,val/non_pv}`。数据已搬走 → 在主仓建 gitignored symlink 让其仍可达:
```bash
ln -s ~/zasolar_data/cls/data/cls_pv_thermal_v2 /home/gaosh/projects/ZAsolar/data/cls_pv_thermal_v2
```
(主仓 `.gitignore:25 data/cls_pv_thermal_v*/` 已覆盖,symlink 不会被 track。)确认 `python scripts/training/negative_pool/bootstrap_from_cls_v2.py --help` 仍能找到 CSV。

### 2.6 Step E — P0-4 修复(在 solar_cls 的 classify_predictions.py)

当前 `_find_tile`(原 152-164)只 `grid_dir.glob(f"{grid_id}_*_*_geo.tif")`(chunked),mosaic 静默返回 None → 整分类静默 no-op。修:
1. **mosaic/多城市感知**: `_find_tile` 改用 `core.grid_utils.resolve_tiles_dir(grid_id, region=, imagery_layer=)`(build_cls_dataset 已这么用),按 `region_registry.get_imagery_layer(...).file_layout` 分支:`chunked`→现 glob;`mosaic`→单 `{grid}_mosaic.tif` + window read。给 CLI 加 `--region` / `--imagery-layer`(现在完全没 region,违反 rule-06「never pick region by grid id」)。
2. **消费 thresholds_v2.json**: `calibrate_v2_thresholds.py` 写它但无 reader;`classify_predictions.py:432` 只读单一标量 `pv_threshold`。加 per-imagery-layer 查表:从 `configs/classifier/thresholds_v2.json` 按 `imagery_layer` 取阈值,CLI `--pv-threshold` 仍可覆盖。否则 v2 calibration 永远是死代码。
3. 范围:单文件 + 少量 CLI,S 级。

### 2.7 Step F — 文档 + git

主仓文档更新:
- `CLAUDE.md`(L14 "Sibling subrepo" 段):新增 `solar_cls` 条目(与 solar_backdating 并列),声明 PV/thermal 分类器已抽出、本仓不再持有、新 classifier 工作去 `/home/gaosh/projects/solar_cls/`。
- `AGENTS.md`:同步。
- `ROADMAP.md`(约 L202/279/282 "PV vs thermal classifier" backlog):注明已抽到 solar_cls + P0-4 已修。
- `data/negative_pool/README.md`(约 L40/51/83)+ `data/negative_pool/archetype_taxonomy.yaml`(约 L140/246):`bootstrap_from_cls_v2.py` 对 cls_pv_thermal_v2 的引用路径更新为 symlink 说明。
- `docs/plans/2026-05-29-codebase-optimization.md`:标记 cls-1/cls-2(P0-4)+ 分类器抽出已执行。

solar_cls 写 `SHARED_FROM_ZASOLAR.md`(imported surface 表只需 `region_registry` 6 符号 + `grid_utils.{TILES_ROOT,resolve_tiles_dir}`,**不含** `core.annotation_loader`;单向契约;`ZASOLAR_ROOT` override 段)。

```bash
cd /home/gaosh/projects/solar_cls && git init && git add -A && git status   # 人工核对后 commit
```
**主仓 git rm 的改动:暂不 commit**(主仓在默认分支 main;按 repo 规则提交前先问用户/开分支)。Track 1 验证全绿后,把主仓改动整理成一个 commit 让用户确认。

### 2.8 Track 1 验收门(全绿才进 Track 2)
跑 2.1 的验证 subagent,5 条全 pass。重点:主仓 `pytest` 绿、主仓无 `from scripts.classifier` 残留、solar_cls 2 测试绿、无 `relative_to(PROJECT_ROOT)` 残留、`--help` 无 ImportError。

---

## 3. Track 2 — 训练集资源规范化 (Phase 0–4)

**贯穿原则**(rule-07): 2 桶(可信/不可信)是**叠加在**现有 A1/A2/A3 × H/R/S/G × T1/T2 之上的派生视图,**不替换**;唯一定义来源是 `label_source → mask_trusted`;`T1 ⊂ trusted`,任何 pool 字段不得用于 auto-promote T1。对 unknown/missing `label_source` **fail-closed → untrusted**。

**EXTEND 不重造**: `pipeline/specs.py`(`DatasetSpec` 严格校验)、`pipeline/manifests.py`(`write_build_manifest`/`generate_build_id`/`compute_file_sha256`/`_git_commit_hash`,目前 orphaned,行 41-227)、`data/negative_pool/`(已是理想 provenance manifest,直接当第 3 桶)、`train.py`(`training_history.json`,缺字段都是已知值没落盘 + **完全没 seed**)、`configs/model_registry.yaml`。

### Phase 0 — 抽映射,修漂移风险(零行为变化;driver 直接做)
现状 bug 风险: `export_coco_dataset.py:52-64 _MASK_TRUSTED` 与 `train.py:67-99 _LABEL_SOURCE_TO_MASK_TRUSTED`/`_LABEL_SOURCE_TO_BOUNDARY_W` 是**两份重复硬编码,会漂移**。
1. 建 `data/training_pool/boundary_trust_rules.yaml`(单一真相):
```yaml
schema_version: 1
fail_closed_default: untrusted          # unknown-but-present source
legacy_no_source_field: trusted         # 真 legacy 无 source 列(同 train.py:109-120 现行为)
map:
  human_manual:               { boundary_trust: trusted,   mask_trusted: true,  boundary_w: 1.0 }
  human_manual_sam_assisted:  { boundary_trust: trusted,   mask_trusted: true,  boundary_w: 1.0 }
  human_manual_qgis_geosam:   { boundary_trust: trusted,   mask_trusted: true,  boundary_w: 1.0 }
  sam_added_browser:          { boundary_trust: trusted,   mask_trusted: true,  boundary_w: 1.0 }
  reviewed_prediction:        { boundary_trust: untrusted, mask_trusted: false, boundary_w: 0.0 }
  gemini_reviewed_prediction: { boundary_trust: untrusted, mask_trusted: false, boundary_w: 0.0 }
  sam_refined_review:         { boundary_trust: untrusted, mask_trusted: false, boundary_w: 0.0 }
  sam_added_true_fn:          { boundary_trust: untrusted, mask_trusted: false, boundary_w: 0.0 }
  legacy_weak_supervision:    { boundary_trust: untrusted, mask_trusted: false, boundary_w: 0.0 }
```
> ⚠️ **先核对真实 enum**: `grep -rn label_source export_coco_dataset.py train.py data/annotations/ANNOTATION_SPEC.md .claude/rules/07-annotation-semantics.md`,以现有 `_MASK_TRUSTED`/`_LABEL_SOURCE_TO_*` 的**实际 key 与值**为准填表,不要照搬上面(上表是设计草案,可能有 enum 名出入)。
2. `export_coco_dataset._MASK_TRUSTED` 与 `train.py` 两处映射改为从该 YAML 加载,保留 fail-closed(`None`→raise;unknown→`fail_closed_default`)。
3. 加单测 `tests/test_boundary_trust_rules.py`:断言 YAML 加载值 == 重构前的旧硬编码值(逐 key);断言 unknown→untrusted、missing+legacy→trusted。
**验收**: `pytest tests/test_boundary_trust_rules.py` 绿;`export_coco_dataset.py`/`train.py` 行为不变(diff 一个小 COCO build 的 chip 标签分布)。

### Phase 1 — 物化 pool manifest(只读派生,不改训练;派实现 subagent)
正样本 pool **provenance-only manifest,不物化 chip**(chip 构建时现切,符合 rule-08)。
1. 写 `scripts/training/build_training_pool.py`:从 `core.annotation_loader.discover_annotations` + 各 region gpkg 读 `label_source`,join `data/annotations/annotation_manifest.csv` 拿 `quality_tier/semantic_confidence`,按 `boundary_trust_rules.yaml` 分流 → 生成 `data/training_pool/positive_trusted_manifest.csv` 与 `positive_untrusted_manifest.csv`。**region 一律经 `region_registry`(rule-06),imagery_layer 从 regions.yaml 取,禁止从 grid_id/经度推断。**
2. manifest schema(两份同列):`poly_id, region, grid_id, imagery_layer, source_file, source_layer, source_id, label_source, quality_tier, semantic_confidence, boundary_trust, mask_trusted, added_date, notes`。bucket 归属 ⇔ `boundary_trust_rules.map[label_source].mask_trusted`。
3. 补 `annotation_manifest.csv` 的 JHB Vexcel 行(现停在 2026-04-12)+ 加 `boundary_trust` 派生列。
**验收**: pool 行数与 `build_unified_reviewall.py` 当前实际选入的 polygon 数对得上(用旧脚本统计做 ground truth,确认没漏没串)。`data/training_pool/` 的 CSV 进 git(Q4)。

### Phase 2 — 生产 builder 先 emit build_manifest(先留痕;driver 直接做)
给 `scripts/training/build_unified_reviewall.py`(及 `export_v4_1_hn.py`)末尾加调用 `pipeline.manifests.write_build_manifest(...)` → 写 `build_manifest.json`(build_id + 逐源 sha256 + git provenance)。**语义不变**,只是开始留确定性 build ID。

### Phase 3 — train.py seed + run ledger(派实现 subagent)
1. **seed**(当前完全缺失,run 不可复现): `main()` 入口加 `--seed`(默认 42),设 `random/np.random/torch.manual_seed/torch.cuda.manual_seed_all`,可选 `--deterministic`。**这是 Track 2 唯一新增行为**(亦即优化方案 train-1)。
2. 扩 `training_history.json` 加 `run_manifest` 块,字段: `run_id`(复用 `generate_build_id`)、`dataset.{coco_dir, build_id, build_manifest_sha256, spec_path}`、`init_weights`+sha256、`seed`、`hyperparams`(全部 CLI:lr1/lr2/epochs1/epochs2/batch_size/chip_size/boundary_band_iters/reinit_*/diff_lr)、`boundary_aware`(per_instance_mask_trusted/per_source_mask_weight/freeze_mask_head/boundary_trust_rules_sha256)、`code_provenance`(git_commit/git_dirty,复用 `manifests._git_*`)、`metrics.chip_level`、`metrics.grid_level`(占位 null,由 run-evaluation 回写)、`output_checkpoints`。保留原有 `history`/`best_ap50`/`best_f1`(向后兼容)。
3. 新建顶层索引 `configs/training_runs.yaml`(每 run append 一行轻量索引)+ `runs/<run_id>/run_manifest.json`(详档,进 `~/zasolar_data` 或 gitignored `runs/`;索引进 git per Q4)。
4. run 收尾自动回写 `configs/model_registry.yaml`:`training_set_id = dataset.build_id`(必填化)。
5. 写 `scripts/training/diff_runs.py runA runB`:diff 两份 run_manifest 的 dataset.build_id/init_weights/seed/hyperparams/boundary_aware/git_commit 与 metrics → 输出「变了什么 → 指标怎么动」(用户要的追溯对比)。
**验收**: 跑一个 1-epoch smoke train,确认 `run_manifest` 落盘、`training_runs.yaml` 追加、seed 复现(同 seed 两次 run 前若干 step loss 一致)。

### Phase 4 — DatasetSpec v2 + spec 迁移(最大;派实现 subagent + 对抗校验 subagent)
1. `pipeline/specs.py` 加 schema_version=2 字段(默认值保 v1 兼容):
   - `PositiveSourceSpec{bucket, pool_manifest, regions, imagery_layers, tier_filter, label_sources, max_ratio}`
   - `MaskSupervisionSpec{per_instance_mask_trusted, boundary_band_iters, untrusted_max_x_trusted, freeze_mask_head}`
   - `HardNegativeEntry` 扩 type 枚举加 `negative_pool`(字段 archetypes/min_confidence/max_ratio)
   - `DatasetSpec` 顶层加 `positives:list, mask_supervision, init_weights, val_grids`
   - 校验:bucket∈{trusted,untrusted};pool_manifest 存在;regions 经 region_registry;`untrusted 总数 ≤ untrusted_max_x_trusted × trusted`(把 build_unified_reviewall 的硬编码 assertion 提升为 spec 校验);archetype 必须在 `archetype_taxonomy.yaml`;schema_version=2 才允许新 key。
2. `pipeline/dataset_builder.py` + `pipeline/hn_ops.py` 支持 `positives`/`negative_pool` HN/`val_grids`/`mask_supervision`。
3. 写 `configs/pipelines/datasets/unified_reviewall_v2.yaml`(示例见设计;positives 两桶 + negative_pool HN + val_grids 显式 holdout + mask_supervision + init_weights)。
4. **byte-diff 验收**: 用 spec build 跑一遍,build_manifest 的 `selected_annotations` 与 Phase 1 pool 行做 diff,确认与旧 `build_unified_reviewall.py` 路径产出一致 → 才把 bespoke builder 标 deprecated(保留作历史复现)。
5. 回填 `training_sets.yaml`:从此由 build_manifest 自动生成条目 + unknown-key 校验。
**对抗校验 subagent**: 独立验证「spec build 的 COCO 与 bespoke build 在 image/annotation 集合上 byte 等价」,默认怀疑不等价,逐项给证据。

> Phase 1–4 严格顺序;每 phase 验收门绿才进下一 phase。Phase 0/1 不改任何训练数据/权重;Phase 3 的 seed 是唯一行为变化;旧 checkpoint 复现性不受影响。

---

## 4. 关键文件清单(绝对路径,便于 grep)

**Track 1 改动**: `scripts/classifier/{build_cls_dataset,audit_cls_sources,classify_predictions}.py`(ZASOLAR_ROOT + P0-4)、`scripts/analysis/{cls_nonpv_subtype_audit,label_cls_nonpv_subtype}.py`、`scripts/training/negative_pool/bootstrap_from_cls_v2.py`(symlink 接缝)、`CLAUDE.md`/`AGENTS.md`/`ROADMAP.md`/`data/negative_pool/README.md`。
**Track 2 改动**: `pipeline/specs.py`(L… DatasetSpec v2)、`pipeline/manifests.py`(复用 41-227)、`pipeline/dataset_builder.py`、`pipeline/hn_ops.py`、`export_coco_dataset.py`(L52-67 `_MASK_TRUSTED`/`mask_trusted_for`)、`train.py`(L67-99/109-120 映射 + seed + L1500-1521 history payload)、`scripts/training/build_unified_reviewall.py`、`scripts/training/export_v4_1_hn.py`、新建 `scripts/training/build_training_pool.py`/`diff_runs.py`、`data/training_pool/`(NEW)、`configs/{training_runs.yaml(NEW),model_registry.yaml,datasets/training_sets.yaml,pipelines/datasets/unified_reviewall_v2.yaml(NEW)}`、`data/annotations/annotation_manifest.csv`(补行+列)。

## 5. 顺序总览
1. Track 1 Step A–F → 验证门(2.8)绿 → 整理主仓 commit 待用户确认 + commit solar_cls。
2. Track 2 Phase 0 → 1 → 2 → 3 → 4,逐 phase 验收。
3. 全部完成后:更新 `docs/plans/2026-05-29-codebase-optimization.md` 勾选 cls-1/2/train-1 等;写 daily-log;按需补 memory。
