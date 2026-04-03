# Cross-Review Harness

Agent 交叉审查协调层。将 review 与 execution 分离到不同 session，通过文件系统协调。

## 快速入口

- **Claude Code**: `/cross-review {code|experiment}`
- **Codex**: 读取本文件 + 对应 checklist，按 reviewer-brief 执行

## 目录结构

```
.agents/harness/
├── README.md              ← 你在这里
├── context-map.md         — Agent 导向的源码/数据地图
├── task-board.md          — 任务队列（含依赖和退出标准）
├── pattern-notes.md       — 探索笔记，供 clean agent 接手
├── roles/
│   ├── executor-brief.md  — Executor 角色定义
│   └── reviewer-brief.md  — Reviewer 角色定义
├── checklists/
│   ├── code-review.md     — 代码变更审查清单
│   └── experiment-review.md — 训练实验审查清单
├── templates/
│   ├── handoff.md         — Handoff 文档模板（YAML front matter）
│   └── review.md          — Review 输出模板（YAML front matter）
└── active/                — 运行态文件（gitignored）
    └── {run_id}/
        ├── handoff_r01.md
        └── review_r01.md
```

## 工作流程

1. **Executor** 完成任务后，在 `active/{run_id}/` 写 handoff 文档
2. **Reviewer** 在全新 session 中启动，读取 reviewer-brief + checklist + handoff
3. Reviewer 对照 repo 文件系统（+ RunPod 远端）独立核验，输出 review 文档
4. 若有 P1/P2 → Executor 修正 → 新一轮 handoff → 再 review
5. 收敛后 PASS / PASS-WITH-FIXES / BLOCK

## 协议

- **run_id**: `YYYYMMDD-HHMM-{code|experiment}-{slug}`
- **handoff/review**: Markdown + YAML front matter，字段定义见 `templates/`
- **严重度**: P1 (Blocker) / P2 (Should-fix) / P3 (Nice-to-have)
- **Gate**: code 要求 P1=0 且 P2=0; experiment 允许 P2 写入 limitations

## 设计原则

- 唯一事实源在此目录，Claude Code skill 和 Codex 入口只做指针
- Reviewer 只读约束是流程约定（brief 指令），非技术强制
- Harness 不内置 git 工作流，只负责 handoff / review / gate
- 详细设计方案见 `.claude/plans/flickering-conjuring-crane.md`
