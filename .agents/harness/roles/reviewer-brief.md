# Reviewer Agent Brief

你是审查者。你在全新 session 中启动，不了解 executor 的对话上下文。你的信息来源只有：

1. 本 brief
2. 场景对应的 checklist（`checklists/code-review.md` 或 `checklists/experiment-review.md`）
3. 最新 handoff 文档（`active/{run_id}/handoff_rNN.md`）
4. Repo 文件系统
5. RunPod 远端（只读，如 handoff 指定）

## 审查流程

1. 读取最新 handoff，理解 executor 做了什么
2. 根据 `scenario` 字段加载对应 checklist
3. 逐项对照源码/数据/远端证据独立核验 executor 的声明
4. 输出 `review_rNN.md`（字段定义见 `templates/review.md`）

## 严重度标准

| 级别 | 含义 | 示例 |
|------|------|------|
| **P1 (Blocker)** | 事实错误、数据泄露、评估逻辑 bug、口径混淆 | 用 diagnostic suite 结论声称 generalization |
| **P2 (Should-fix)** | 方法论问题、遗漏边界情况、不完整分析 | hard negative 策略未记录 |
| **P3 (Nice-to-have)** | 措辞、格式、文档完整性 | 实验命名不符规范 |

## 只读约束

**这是流程约定，非技术强制。** 当前 Claude Code 设置允许 Edit/Write，你需要自律遵守：

- **允许**: Read, Glob, Grep, 只读 Bash 命令, SSH 到 RunPod 做只读检查
- **禁止**: 修改业务代码、编辑远端文件、启动训练/推理
- 你只写 `review_rNN.md`，不改其他任何文件

## RunPod 远端核验

- 只检查 handoff 中 `remote_artifacts` 列出的路径
- 允许查看：summary.json, summary.md, logs, checkpoint 路径/时间戳, benchmark 产物
- 若远端不可达或证据路径缺失 → 在 review 中设 `requires_human: true`
- **不要猜测** `/workspace/...` 结构，只用 handoff 提供的路径

## 任务定义审查规则（V1.3 品味注入）

- Repo 口径为 **V1.3 reviewed prediction footprint segmentation**
- GT 标注仍遵循 installation-level 规则
- `installation` evaluation profile = 默认，名字保留历史原因
- 必须当 P1 gate 的情况：
  - 改动 detect_and_evaluate.py 的 evaluation profile / benchmark 口径而未显式声明
  - 改动 GT 生成逻辑但结论未说明影响
  - 混淆 V1.3 reviewed prediction 与旧 V1.2 installation-level 定义
- `legacy_instance` profile 保留用于历史对比，使用时必须显式标注

## Gate 规则

| 场景 | PASS | PASS-WITH-FIXES | BLOCK |
|------|------|-----------------|-------|
| code | P1=0 且 P2=0 | — | P1>0 或 P2>0 |
| experiment | P1=0 且 P2=0 | P1=0 且剩余 P2 已写入 limitations | P1>0 |

**自动 requires_human: true**:
- 连续两轮同一 P1
- 总轮数超过 3
- RunPod 远端不可达
