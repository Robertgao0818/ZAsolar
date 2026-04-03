---
paths:
  - "CLAUDE.md"
  - "AGENTS.md"
  - "README.md"
  - "docs/**"
---

# Documentation Sync Rules

## docs/ 是唯一事实源
- `docs/architecture.md` 是目录结构和路径映射的唯一事实源
- `docs/workflows.md` 是工作流命令序列的唯一事实源
- `CLAUDE.md`、`AGENTS.md`、`README.md` 只包含摘要和指向 docs/ 的指针

## 入口文档三方同步
`CLAUDE.md`、`AGENTS.md`、`README.md` 有相同的"稳定骨架"（Key References、Working Constraints、Environment）。修改其中一个时，检查另外两个是否需要同步更新。

## 文件移动必须同 commit 更新 docs/
移动文件或新建目录时，必须在同一个 commit 中更新 `docs/architecture.md` 的目录结构。不允许出现"代码已移动，但文档仍指向旧路径"的中间状态。
