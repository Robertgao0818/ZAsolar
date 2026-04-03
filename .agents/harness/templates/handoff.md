# Handoff Template

每轮 handoff 文档使用此模板。文件命名：`handoff_rNN.md`（NN = 轮次）。

## YAML Front Matter（必填字段）

```yaml
---
run_id: "YYYYMMDD-HHMM-{scenario}-{slug}"
scenario: code                # code | experiment
round: 1                     # 轮次，从 1 开始
evaluation_profile: installation  # installation | legacy_instance（须显式声明）
evidence_mode: runpod_remote  # runpod_remote | local_workspace
remote_host: "$RUNPOD_SSH_HOST"           # 仅 runpod_remote 模式
remote_workspace: "/workspace/ZAsolar"    # 仅 runpod_remote 模式
remote_artifacts:             # 远端证据路径列表（reviewer 不猜路径）
  - "/workspace/ZAsolar/results/benchmark/{run_name}_{timestamp}/summary.json"
changed_paths:                # 涉及变更的文件列表
  - "train.py"
commands_run:                 # 执行过的关键命令
  - "python train.py --coco-dir ..."
open_risks:                   # executor 自己不确定的地方
  - "..."
reviewer_focus:               # 建议 reviewer 重点核查
  - "..."
---
```

## Markdown Body

### 摘要
<!-- 1-3 句话说明做了什么 -->

### 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|

### 变更范围
<!-- 涉及的文件/数据/配置，reviewer 应该核查什么 -->

### 预期影响
<!-- 这些变更应该带来什么效果 -->

### 风险标注
<!-- 与 front matter 的 open_risks 对应，展开说明 -->

### 上下文索引
<!-- 告诉 reviewer 去哪里找详细信息 -->
- 配置：`configs/...`
- 结果：`results/...`
- 日志：`docs/...`
