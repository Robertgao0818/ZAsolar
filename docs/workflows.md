# Workflows — Cape Town Solar Panel Detection

> **V1.3 (2026-04-03)**: Task definition updated to reviewed prediction footprint segmentation. The `installation` evaluation profile is the default and evaluates reviewed predictions against installation-level GT. `legacy_instance` is available for historical comparisons.

## Environment Setup

```bash
./scripts/bootstrap_env.sh         # 首次创建/更新 .venv
source scripts/activate_env.sh     # 进入项目环境
./scripts/check_env.sh             # 检查关键依赖、运行时目录和 CUDA
```

- 虚拟环境固定在 `./.venv`
- 运行时缓存固定在仓库内：`.cache/`、`.config/`、`.local/`、`.tmp/`
- `requirements.lock.txt` 为环境快照，重建时优先使用
- 训练额外依赖：`torch`、`torchvision`、`opencv-python-headless`、`huggingface_hub`、`pycocotools`

## Inference / Evaluation

```bash
python building_filter.py          # 下载建筑轮廓
# VRT 拼接由 download pipeline 自动处理，无需手动执行
python detect_and_evaluate.py      # 检测 + 评估（需 GPU, 默认 installation profile, V1.3 reviewed prediction 语义）
```

### Evaluation Profiles

```bash
python detect_and_evaluate.py --evaluation-profile installation     # 默认: V1.3 三层指标（reviewed predictions vs installation-level GT）
python detect_and_evaluate.py --evaluation-profile legacy_instance  # 旧版兼容（V1.2 语义）
```

### Using Fine-Tuned Weights

```bash
python detect_and_evaluate.py --model-path checkpoints/best_model.pth --force
```

## Fine-Tuning

```bash
# 0. 生成标注 manifest（首次或标注变更后）
python3 scripts/annotations/bootstrap_manifest.py

# 1. 导出 COCO 数据集（排除 benchmark holdout，控制 neg 比例）
HOLDOUT="G1240 G1243 G1244 G1245 G1293 G1294 G1297 G1298 G1299 G1300 G1349 G1354 G1410 G1411 G1466 G1467 G1516 G1520 G1521 G1522 G1523 G1524 G1569 G1570 G1571 G1572"
python export_coco_dataset.py --output-dir data/coco --exclude-grids $HOLDOUT --neg-ratio 0.15 \
    --audit-csv results/analysis/gt_heater_audit/heater_20260405_2141/audit_labels_phase1.csv

# 1b. 合并 HN（batch 003 + batch 004）
python scripts/training/export_v4_1_hn.py --base-coco data/coco --output-dir data/coco_hn

# 2. 训练前检查依赖和 CUDA
./scripts/check_env.sh

# 3. 训练（需要 CUDA GPU）
python train.py --coco-dir data/coco_hn --output-dir checkpoints

# 4. 使用微调模型推理 + installation profile 评估 (V1.3)
python detect_and_evaluate.py --model-path checkpoints/best_model.pth --force
```

## Multi-Grid GPU Run

```bash
# 规范入口（支持任意 grid list + 并行度）
bash scripts/analysis/batch_inference.sh

# Legacy（仅 3-grid baseline）
./scripts/run_multigrid_gpu.sh
```

## Analysis Scripts

```bash
# 参数网格搜索
python scripts/analysis/param_search.py

# 后处理阈值校准扫描
python scripts/analysis/calibration_sweep.py --step a0   # 导出 pre-filter candidates
python scripts/analysis/calibration_sweep.py --step a1   # 运行 sweep

# 多 Grid baseline/泛化对比
python scripts/analysis/multi_grid_baseline.py
```

## Model Benchmark

训练后使用固定 benchmark 对比模型权重性能。Agent-first 设计：主输出为 `summary.json`（机器可读结论）。

```bash
# 默认 preset + 默认模型 (零参数启动)
python3 scripts/analysis/run_benchmark.py

# 对比两个模型
python3 scripts/analysis/run_benchmark.py --models v3c v3_cleaned

# 临时加入新 checkpoint
python3 scripts/analysis/run_benchmark.py --checkpoint checkpoints/exp005_v4_1_hn/best_model.pth --tag v4_1

# 只跑 smoke suite (快速回归检测)
python3 scripts/analysis/run_benchmark.py --suite cape_town_t1_smoke

# 只收集已有结果，重新生成报告
python3 scripts/analysis/run_benchmark.py --collect-only

# RunPod 上用并行加速 (默认 6 进程)
BENCHMARK_PARALLEL=6 python3 scripts/analysis/run_benchmark.py --models v3c v4_1
```

- **配置**: `configs/benchmarks/post_train.yaml` (preset) + `configs/model_registry.yaml` (模型注册表)
- **Suites**: `cape_town_t1_smoke` (smoke)、`cape_town_independent_26` (primary, 排名用)、`cape_town_batch003_diagnostic` (diagnostic)、`jhb_transfer_6` (secondary)
- **并行**: 环境变量 `BENCHMARK_PARALLEL` 控制推理并行度（默认 6，RTX 5090 最优）
- **输出**: `results/benchmark/<run_id>/` — `summary.json`、`summary.md`、`by_suite.csv`、`by_grid.csv`
- **Per-grid 产物**: `results/<GridID>/benchmark_<run_id>_<tag>/`
- **自动 verdict**: improved / regressed / flat / mixed / failed (基于 primary suite F1 delta)
- **标准后处理**: 所有 benchmark 必须使用 `configs/postproc/v4_canonical.json`（post_conf=0.85 + tiered）

## Google Colab

在 Colab 上运行本项目，参考 `notebooks/SA_Solar_Colab.ipynb`。

### 前置准备

1. **Runtime 设置**: Runtime → Change runtime type → T4 GPU
2. **Google Drive 数据目录**（大文件不入 Git）:
   ```
   MyDrive/SA_Solar_Data/
   ├── tiles/G1238/       # GeoTIFF 瓦片（~3 GB）
   ├── checkpoints/       # 模型权重（~170 MB/个）
   └── results/           # 检测输出（自动创建）
   ```
3. **一键安装**: `!bash scripts/colab_setup.sh`
4. **挂载 + 路径配置**: `from scripts.colab_config import setup_colab; setup_colab()`

### 快速流程

```python
# 在 Colab notebook 中
!git clone https://github.com/Robertgao0818/SA_Solar.git /content/SA_Solar
%cd /content/SA_Solar
!bash scripts/colab_setup.sh

from scripts.colab_config import setup_colab
setup_colab()

# 推理
!python detect_and_evaluate.py --grid-id G1238

# 训练
!python export_coco_dataset.py --output-dir data/coco
!python train.py --coco-dir data/coco --output-dir checkpoints/colab_run
```

### 注意事项

- Colab 使用 `requirements.txt`（非 lock 文件），依赖 Colab 自带的 CUDA 栈
- 通过 `scripts/colab_config.py` 自动将 `tiles/`、`checkpoints/`、`results/` 软链接到 Drive
- 环境变量 `SOLAR_TILES_ROOT` 可覆盖瓦片路径
- Colab 免费版 session 有时间限制，建议先用小 grid 验证

## GT Heater Audit (训练集热水器污染清除)

GT 中混入的太阳能热水器/泳池加热器会污染 PV vs heater 判别边界。审计流程在不修改 cleaned/ 原文件的前提下，从训练导出中隔离污染样本。

```bash
# 1. 构建审计队列 + 导出 400×400 chip
python scripts/analysis/build_gt_heater_audit.py
# 输出: results/analysis/gt_heater_audit/<run_id>/
#   audit_queue_full.csv    — 全量 GT 记录
#   audit_queue_phase1.csv  — 仅 tier A（compact, solidity>0.9, 5-30m²）
#   chips/rgb/, chips/overlay/

# 2. 生成 HTML 标注器（在 Windows 浏览器打开）
python scripts/analysis/label_gt_heater_audit.py \
    --run-dir results/analysis/gt_heater_audit/<run_id>
# 快捷键: 1=PV  2=热水器/非PV  3=不确定  S=跳过  B=回退
# 标完后点「导出 CSV」下载 gt_heater_audit_labeled.csv

# 3. 重新导出训练集（排除 heater_or_non_pv + uncertain）
python export_coco_dataset.py \
    --audit-csv results/analysis/gt_heater_audit/<run_id>/gt_heater_audit_labeled.csv

# 可自定义排除标签
python export_coco_dataset.py \
    --audit-csv <path> --exclude-audit-labels heater_or_non_pv
```

- Phase 1 仅审 tier A（~855 条），污染率显著时扩展到 tier B/C
- `--audit-csv` 过滤发生在 tier-filter 之后、tile 分配之前，不影响 benchmark GT

## Dataset Notes

- `export_coco_dataset.py` 导出带地理参考的 `400x400` chip、`train.json` / `val.json` 和 provenance CSV
- 导出采用 scan-then-write 策略：先扫描所有 chip 的 metadata，balance 采样后只写被选中的 chip 到磁盘，避免产生 orphaned files
- 默认 1:1 正负样本平衡（`--no-balance` 跳过），负样本为无标注 chip（hard negatives）
- COCO 数据集可通过 `--manifest` + `--tier-filter` 按标注质量等级过滤
- Targeted hard negatives（从审核 FP 提取）: `scripts/training/export_targeted_hn.py`

## RunPod Cloud Training

### 环境初始化（每次 Pod 启动后必须执行）

```bash
# 必装依赖（Pod 重启后丢失）
pip install --break-system-packages \
    pycocotools opencv-python-headless geopandas rasterio \
    huggingface_hub matplotlib seaborn geoai-py rasterstats

# 上传 task_grid.gpkg（推理评估需要）
scp data/task_grid.gpkg root@<pod-ip>:/workspace/ZAsolar/data/
```

### 数据加速

```bash
# 训练前：拷数据到 RAM disk（IO 提速 ~10x）
cp -r /workspace/coco_v3_no_hn /dev/shm/
# 训练时使用 /dev/shm 路径
python3 train.py --coco-dir /dev/shm/coco_v3_no_hn --output-dir /workspace/checkpoints/exp
```

### 训练（Spot 友好）

```bash
# 用 nohup 启动（防 SSH 断连）
nohup python3 train.py --coco-dir /dev/shm/coco_v3_no_hn \
    --output-dir /workspace/checkpoints/exp --batch-size 32 \
    > /workspace/train_log.txt 2>&1 &

# Spot 抢占后恢复（从最新 checkpoint 续训）
ls -t /workspace/checkpoints/exp/stage*_epoch*.pth | head -1
python3 train.py --coco-dir ... --output-dir ... --resume <checkpoint.pth>
```

### 批量推理

```bash
# 确认 rasterstats 已安装（否则 confidence=0 导致 0 检测）
python3 -c "import rasterstats; print('OK')"

# 单 grid 推理
python3 detect_and_evaluate.py --grid-id G1293 \
    --model-path /workspace/checkpoints/exp/best_model.pth \
    --evaluation-profile installation --force
```

### 注意事项

- **rasterstats 是关键依赖**：geoai 的 mask band2 存储 confidence，需要 rasterstats 回填。缺失时默认 confidence=0.5，被 `post_conf_threshold=0.7` 过滤后变成 0 检测
- `train.py` 默认 batch_size=32, num_workers=8, AMP 开启（适配 5090 32GB）
- 每 epoch 保存 checkpoint，保留最近 2 个，支持 Stage 1/2 断点续传
- Network volume 数据持久化，container 本地盘重启丢失
