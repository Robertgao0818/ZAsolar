# 周工作记录 2026-03-16 ~ 2026-03-21

**项目**: Cape Town 屋顶太阳能安装检测 (ZAsolar)


---

## 一、周日 3/16 — 项目文档与实验整理

**工作内容**:
- 更新 CLAUDE.md，创建项目 Roadmap 文件，记录当前进度
- 将历史实验结果（参数搜索、基线对比等）整理为统一模板的横向比较文档
- 代码同步至 GitHub 仓库

**Git 提交**:
- `a220822` (03-16) Add fine-tuning pipeline (V1): COCO exporter, training script, --model-path support

---

## 二、周一 3/17 — V1.2 评估体系设计与实现

**工作内容**:
- 与 Codex/GPT 协同讨论任务重定义方案，明确 **installation-level footprint** 为核心评估目标
- 确定分层递进的目标实现路径：presence → footprint → area
- 设计并实现 V1.2 评估框架：
  - 三层 installation 指标体系（presence / footprint / area error）
  - `--evaluation-profile` CLI 参数
  - `config.json` 可追溯性扩展
- 编写标注规范文档 `ANNOTATION_SPEC.md`（installation footprint 定义、T1/T2 质量分级）
- 标注基础设施：`annotation_manifest.csv`、`bootstrap_manifest.py`
- COCO 导出支持 manifest 过滤（`--manifest`, `--tier-filter`, `--category-name`）
- 后训练校准扫参 `calibration_sweep.py`：最优配置 min_area=5.0, max_elongation=8.0，宏平均 F1 +0.021
- GPU 全网格验证通过：G1189 F1=0.595, G1190 F1=0.649, G1238 F1=0.779
- 代码推送至 GitHub

**Git 提交**:
- `c8fae98` (03-17) V1.2 complete: installation-level evaluation profile, annotation alignment, and post-training calibration sweep

---

## 三、周二 3/18 — 项目重构 + 标注清洗 + 环境搭建 + JHB 跨城测试

### 上午：影像数据排查
- 发现备份 tiles 年份问题（2023 vs 2025），确认航测影像来源为 Cape Town 2025-01 WMS
- 重新下载 G1238 影像并完成 E2E 验证

### 下午：项目重构与代码整理
- 目录重构：共享工具移入 `core/`，脚本按领域分组（`scripts/analysis/`, `scripts/imagery/`, `scripts/annotations/`）
- 根目录 Python 入口从 8 个精简到 4 个，更新全部 import 路径
- 标注清洗脚本 `clean_annotations.py`：
  - panel-level → installation-level 合并
  - ID 统一为 `{GridID}_{NNN}` 格式
  - 新增 `num_parts`（记录合并前子面板数）和 `land_use`（预留字段）
- 文档重构：CLAUDE.md / README.md / STATUS.md 拆分至 `docs/architecture.md`, `docs/workflows.md`, `docs/governance/repo-rules.md`
- 新增数据集配置 `configs/datasets/regions.yaml`

### 环境搭建：QGIS + Geo-SAM 插件安装
- 安装 QGIS 及 Geo-SAM 插件（基于 Segment Anything Model 的地理影像分割工具）
- 配置 conda 环境，安装 PyTorch、rtree 等依赖，解决 `torch/shm.dll` 缺失等兼容性问题
- 下载 SAM vit_h 模型权重文件（2.4GB）
- 安装 CUDA 12.8 工具包以启用 GPU 加速

### 晚间：JHB 跨城测试 + 标注方案讨论 + 会议准备
- 使用 fine-tuned 模型跑 JHB（约翰内斯堡）跨城检测，讨论航测年份（2023）与标注年份（2025）不一致的影响
- 讨论检测与分割解耦方案：Mask R-CNN 检测 + SAM2 分割，记录为远期 idea
- 讨论标注 ID 规范、multi-part installation 数据结构设计
- 准备与导师会议汇报材料（中英文版本），梳理项目目标分层路径
- 确定在 Windows QGIS 上使用 SAM2 插件进行标注，WSL 保持训练环境
- Claude Code 提示音 hook 配置调试（WSL ↔ Windows 音频链路）
- 代码同步至 GitHub

**Git 提交**:
- `5e18459` (03-18) Restructure project: move shared utils to core/, group scripts by domain
- `14a1f6b` (03-18) Add annotation cleanup, docs restructure, JHB fine-tuned evaluation, and project configs

---

## 四、周三~周四白天 3/19 ~ 3/20 — QGIS 环境修复 + SAM 编码 + MCP 集成

### QGIS 重装与依赖修复（3/18 - 3/19）
- 因 conda 版 QGIS 兼容性问题，改用 OSGeo4W 安装器重新安装 QGIS
- 通过 OSGeo4W Shell 安装 GeoOSAM 所需依赖
- 排查 `conda activate` 报错等环境配置问题

### SAM 特征编码运行（3/18 - 3/19）
- 对 G1238_mosaic 瓦片运行 SAM 编码器，提取影像特征嵌入（耗时约 1 小时）
- 排查编码过程中的 `AttributeError`、`SamTestGridGeoSampler` 等错误

### QGIS MCP 插件集成（3/19 - 3/20）
- 安装 qgis_mcp 插件，实现 Claude Code 对 QGIS 的程序化控制
- 多轮排查 MCP 连接失败问题（测试端口 9876、9877）
- 最终成功建立连接
- 同时在 Claude Desktop 中配置 MCP

### SAM2.1 标注准备
- 使用 GeoOSAM 插件测试点提示分割（click-segment）工作流
- 讨论 installation-level 统计方案：插件切割后如何按户汇总
- 解决 WSL 底图在 Windows QGIS 中无法识别的问题，将项目数据同步至 `D:\ZAsolar`

---

## 五、周四晚 3/20 — V2 SAM2 重标注 + 模型重训练

### 太阳能板标注与精度验证
- 使用 GeoOSAM 插件 + SAM2.1 对 3 个 grid（G1238, G1189, G1190）重新标注
  - 共 **456 个多边形**（T1 质量），较 V1 的 257 个增加 77%
- 通过 MCP 调用 QGIS，将 SAM 自动检测结果与 G1238 参考标注叠加对比，评估检测精度
- 合并 `GeoOSAM_output` 目录下的多个 GeoPackage 文件
- 对 G1189_mosaic 瓦片进行太阳能板标注
- 修复标注过程中的类别标签错误（commercial → residential）
- 合并多个标注图层
- SAM2 边缘噪点评估：确认可接受，不需额外清理

### V2 模型训练与评估
- 重新训练 Mask R-CNN，20 epochs
- **V2 结果**: val_AP50 = **0.6889**（V1: 0.4205，提升 +0.268）
- 主评估指标升级：F1@IoU0.1 → **F1@IoU0.5**（SAM2 精细标注下 IoU 0.1 已无区分力）
- 全网格评估（SAM2 GT）：

| Grid | P@IoU0.5 | R@IoU0.5 | F1@IoU0.5 | mean IoU |
|------|----------|----------|-----------|----------|
| G1238 | 0.509 | 0.569 | 0.537 | 0.691 |
| G1190 | 0.559 | 0.717 | 0.628 | 0.742 |
| G1189 | 0.555 | 0.560 | 0.557 | 0.729 |

- 分析瓶颈：precision (~0.63) 现为主要瓶颈，recall 已显著改善
- 编写半自动标注脚本 `export_hints.py`：按置信度分级输出 bbox hints 到 QGIS（框标注而非颜色覆盖）
- 讨论技术方向：Mask R-CNN 检测机制分析、SAM2 推理开销、GPU 升级（5080/5090）收益估算
- 更新进度文档，代码收工

---

## 六、周五 3/21 — 工作记录整理

- 整理本周工作记录（本文档）

---

## 本周关键成果汇总

| 项目 | 成果 |
|------|------|
| V1.2 评估体系 | 三层指标框架（presence/footprint/area）完成，GPU 验证通过 |
| 项目重构 | 目录整理、文档拆分、标注清洗脚本 |
| 环境搭建 | QGIS + GeoOSAM + SAM2.1 + CUDA 12.8 + MCP 集成 |
| SAM 特征编码 | G1238 影像特征嵌入提取完成 |
| V2 SAM2 重标注 | 3 grid × 456 polygons (T1)，标注量 +77% |
| V2 模型重训练 | val_AP50: 0.4205 → **0.6889** (+0.268) |
| 评估指标升级 | F1@IoU0.1 → F1@IoU0.5 |
| 半自动标注工具 | export_hints.py 脚本就绪 |
| JHB 跨城测试 | 已运行，待深入分析 |

## 下一步计划

1. 半自动标注扩展更多 grid（模型检测 → QGIS bbox hints → SAM2 click-segment）
2. FP 减少：hard negative mining、confidence threshold tuning
3. JHB 跨城迁移深入评估
4. 远期：检测 + SAM2 分割解耦 pipeline
