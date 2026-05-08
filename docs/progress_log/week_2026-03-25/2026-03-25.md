# 周报 2026-03-25

**项目**: Cape Town 屋顶太阳能安装检测 (ZAsolar)

---

## 本周进展

### 1. V3 模型训练完成

- **数据**: 43 grids, 2982 polygons (SAM2 cleaned), 74k chips (28GB COCO)
- **训练配置**: 两阶段 — Stage 1 heads-only (3 epochs, LR=1e-3) + Stage 2 full fine-tune (20 epochs, LR=1e-4, cosine decay)
- **最优结果**: val_AP50 = **0.8079** (epoch 7)，之后 plateau
- **V2→V3 提升**: AP50 0.689 → 0.808 (+0.119)
- 添加了 `--resume` 功能，支持从 checkpoint 断点续训

### 2. 泛化评估

在训练 grid (G1238) 和未见 grid (G1467) 上评估:

| Grid | Precision | Recall | F1 | Median IoU |
|------|-----------|--------|----|------------|
| G1238 (训练) | 0.812 | 0.714 | 0.760 | 0.801 |
| G1467 (未见) | 0.725 | 0.710 | 0.717 | 0.834 |

- 小面板 FN 分析：87% 的小 GT 多边形 (<5m²) 在 3m 内有预测覆盖，多为 SAM2 补充片段而非独立安装
- 修正后 installation-level recall ~0.79, F1 ~0.80

### 3. 半自动标注工作流

构建了完整的半自动标注流水线，大幅提升标注效率:

1. **Detection Review GUI** (`scripts/annotations/review_detections.py`)
   - Web 端逐 tile 审查模型预测（http://localhost:8766）
   - Canvas 缩放/平移，点击多边形选中分类 (correct/delete/edit)
   - FN 标记模式：点击标记漏检位置
   - 支持查看空 tile（无预测）防止漏标
   - 导出带 review_status 的 GPKG + QGIS QML 配色样式

2. **SAM 自动补标**
   - 对 FN 标记点调用 SAM-ViT-Large 点提示分割
   - 自动生成多边形并合并到 reviewed GPKG
   - Mask 选择策略: index 1 (中间粒度) 最适合光伏

3. **G1686 标注完成**
   - 131 个模型预测 → 102 correct + 32 delete + 3 SAM 补标 = **105 GT polygons**
   - 典型 hard negative: 透光棚架+阴影, 太阳能热水器 (非光伏)

### 4. Batch 003 Grid 扩展

- 范围: G1636 → G1847 (100 grids)
- 筛选: 61 个空白自动排除, 27 个 keep
- 发现 OSM 建筑数据在郊区不可靠 → 改为全量下载 keep grids
- 27 个 grid tiles 全量下载中 (D:\ZAsolar\tiles)

### 5. 云迁移 (RunPod)

- Network Volume 创建 (EU-RO-1, 100GB)
- Code + checkpoints 已上传 (522MB)
- Python 环境已验证 (torch 2.4.1+cu124, RTX A4500)
- COCO 数据集上传中 (28GB, 分卷 4GB×7)
- `cloud_setup.sh`: 本地 SSD 暂存 + 训练启动脚本
- `scripts/upload_to_runpod.sh`: 断点续传上传脚本

---

## 待讨论问题

### 1. 标注策略
- 当前 precision ~0.73 是主要瓶颈 (FP 多于 FN)
- 高置信度 FP (>95%) 集中在棚架+阴影、太阳能热水器 → 这些是高价值 hard negative
- **问题**: 是否应该优先标注更多 grid 来覆盖这类 hard case，还是先对现有数据做增量训练看效果？

### 2. 评估指标调整
- 小面板 (<5m²) 多为 SAM2 补充片段，非独立安装，拉低了 recall
- **问题**: 是否需要在评估时合并相邻小多边形，以更准确反映 installation-level 性能？

### 3. 云训练计划
- RunPod 环境已就绪，COCO 数据上传中
- A4500 20GB 可用于调试，正式训练需 A6000/A100
- **问题**: 下一轮训练是否等 batch 003 全部标完再启动，还是先用现有 43+1 grids 做增量训练？

### 4. OSM 过滤方案
- 郊区 OSM 建筑数据不完整，导致 tile 漏下载
- 已改为全量下载 keep grids（grid 级筛选仍然有效，减少 73% grids）
- **问题**: 后续是否需要引入 Microsoft Buildings 等补充数据源，还是全量下载已足够？

### 5. 跨 tile 边界问题
- 部分太阳能板跨 tile 边界被切断
- 当前后处理有 overlap 合并，但审查时仍可能看到半截多边形
- **问题**: 是否需要在 COCO 导出时处理跨 tile 合并？

---

## 下周计划

1. 完成 batch 003 剩余 grid 标注 (26 grids)
2. 导出新 COCO 数据集，启动 V4 增量训练
3. RunPod 云训练调试
4. 评估 V4 模型在 hard negative (棚架/热水器) 上的表现
