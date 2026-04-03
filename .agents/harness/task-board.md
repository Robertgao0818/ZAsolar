# Task Board — Cross-Review Harness

## Status Legend

- `DONE` — 完成
- `ACTIVE` — 当前进行中
- `NEXT` — 下一个
- `BLOCKED` — 被依赖阻塞

## Phase 0: 文档事实源迁移

| # | Task | Status | Exit Criteria |
|---|------|--------|---------------|
| 0a | V1.2 → V1.3 task definition across all docs | DONE | CLAUDE.md, AGENTS.md, README.md, ANNOTATION_SPEC.md, architecture.md, workflows.md 一致 |
| 0b | Profile comment in detect_and_evaluate.py | DONE | installation profile 定义处有 V1.3 alignment 注释 |
| 0c | Transition banners on stale sections | DONE | architecture.md, workflows.md 顶部有 V1.3 banner |
| 0d | Sync rules and entry docs | DONE | 02-evaluation-semantics.md 已更新；三方入口文档一致 |

## Phase 1: Harness 基础设施

| # | Task | Status | Exit Criteria |
|---|------|--------|---------------|
| 1a | Directory structure | DONE | .agents/harness/ 含 roles/, checklists/, templates/, active/ |
| 1b | README.md | DONE | 入口文档含目录结构、工作流程、协议说明 |
| 1c | context-map.md | DONE | 关键文件表 + CRS 速查 |
| 1d | task-board.md | DONE | 本文件 |
| 1e | pattern-notes.md | DONE | 探索笔记 |
| 1f | roles/ | DONE | executor-brief.md + reviewer-brief.md |
| 1g | checklists/ | DONE | code-review.md + experiment-review.md |
| 1h | templates/ | DONE | handoff.md + review.md |

## Phase 2: Skill 集成

| # | Task | Status | Depends On | Exit Criteria |
|---|------|--------|------------|---------------|
| 2a | .claude/skills/cross-review/SKILL.md | DONE | Phase 1 | Claude Code 入口包装 → harness README |
| 2b | .agents/skills/cross-review.md | DONE | Phase 1 | Codex 侧扁平指针 |
| 2c | .gitignore update | DONE | — | .agents/harness/active/ 被忽略 |
| 2d | CLAUDE.md Key References | DONE | 2a | 添加 harness 入口链接 |
