# Workflows — Cape Town Solar Panel Detection

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
python tiles/build_vrt.py          # 瓦片配准 + VRT 拼接
python detect_and_evaluate.py      # 检测 + 评估（需 GPU, 默认 installation profile）
```

### Evaluation Profiles

```bash
python detect_and_evaluate.py --evaluation-profile installation     # 默认: 三层指标
python detect_and_evaluate.py --evaluation-profile legacy_instance  # 旧版兼容
```

### Using Fine-Tuned Weights

```bash
python detect_and_evaluate.py --model-path checkpoints/best_model.pth --force
```

## Fine-Tuning

```bash
# 0. 生成标注 manifest（首次或标注变更后）
python3 scripts/annotations/bootstrap_manifest.py

# 1. 导出 COCO 数据集（400×400 chips, 0.25 overlap, 80/20 split）
python export_coco_dataset.py --output-dir data/coco

# 1b. 仅用 T1 标注导出（可选）
python export_coco_dataset.py --output-dir data/coco_t1 \
  --manifest data/annotations/annotation_manifest.csv --tier-filter T1

# 2. 训练前检查依赖和 CUDA
./scripts/check_env.sh

# 3. 训练（需要 CUDA GPU）
python train.py --coco-dir data/coco --output-dir checkpoints

# 4. 使用微调模型推理 + installation profile 评估
python detect_and_evaluate.py --model-path checkpoints/best_model.pth --force
```

## Multi-Grid GPU Run

```bash
./scripts/run_multigrid_gpu.sh     # WSL 终端运行 3-grid baseline + 泛化验证
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

## Dataset Notes

- `export_coco_dataset.py` 导出带地理参考的 `400x400` chip、`train.json` / `val.json` 和 provenance CSV
- 训练集保留空标注 chip，负样本真正进入 Mask R-CNN 训练（hard negatives）
- COCO 数据集可通过 `--manifest` + `--tier-filter` 按标注质量等级过滤
