# Review Template

每轮 review 输出使用此模板。文件命名：`review_rNN.md`（NN = 轮次，与 handoff 对应）。

## YAML Front Matter（必填字段）

```yaml
---
run_id: "YYYYMMDD-HHMM-{scenario}-{slug}"
scenario: code                # code | experiment
round: 1                     # 轮次
verdict: PASS                 # PASS | PASS-WITH-FIXES | BLOCK
p1_count: 0
p2_count: 0
p3_count: 0
requires_human: false         # true 时需要人工介入
stale_if_executor_changes: true  # executor 再改动后此 review 过期
---
```

## Markdown Body

### 总结

- P1: {count} | P2: {count} | P3: {count}
- 结论: {verdict}

### 发现

#### P1 — Blockers

##### {issue-title}
- **位置**: {file:line 或 data path}
- **问题**: {描述}
- **证据**: {来自源码/数据/远端的具体引用}
- **建议修复**: {方向}

#### P2 — Should-fix

##### {issue-title}
- **位置**: ...
- **问题**: ...
- **证据**: ...
- **建议修复**: ...

#### P3 — Nice-to-have

- ...

### 收敛趋势

| Round | P1 | P2 | P3 | Verdict |
|-------|----|----|-----|---------|
| 1     |    |    |     |         |

### requires_human 原因（如适用）

<!-- 说明为什么需要人工介入 -->
