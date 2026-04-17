# save-conversation Skill

把当前对话导出成干净的 markdown 存到 `~/projects/agent_record/`，方便以后回顾或归档。

## When to use

用户说"保存对话"、"存一下这次对话"、"这次对话很重要存一下"、"导出对话"、"archive this conversation"等类似意思时触发。

也可以由 `/save-conversation` 这样的 slash 调用。

## Output location

固定为 `/home/gaosh/projects/agent_record/`（注意是 `~/projects/`，不是 ZAsolar 仓库内部）。

## Steps

### 1. 决定文件名

格式：`YYYY-MM-DD_<topic-slug>.md`

- 日期：当天 NZDT 日期（UTC+13）
- topic-slug：根据当前对话主题生成，3-6 个英文小写词，用连字符分隔
  - 看对话开头几条 user message 提炼主题
  - 例如：「逆向 GEID」→ `geid-reverse-engineering`，「跑 V4 训练」→ `v4-training-run`
  - 不要用 emoji、空格、中文、不要引号
- 完整例子：
  - `2026-04-08_geid-reverse-engineering.md`
  - `2026-04-09_v4-training-run.md`
  - `2026-04-10_cape-town-coco-export.md`

如果 `agent_record/` 里已有同名文件，在末尾加 `_2`、`_3` 后缀。

### 2. 写 TL;DR

根据当前对话的实际内容，写一段 1-2 段的中文摘要，放在导出 md 的最顶部。

要包含的关键信息（按重要性排序）：
- **目标**：用户最初想做什么
- **结果**：实际做成了什么（具体数字 / 文件 / commit hash 优先于抽象描述）
- **关键决策或转折**：如果对话中途换了方案，说明换的原因
- **遗留 / TODO**：还有什么没做完，下次接着做时要注意什么

约束：
- 不要复制粘贴对话原文
- 不要 emoji
- 不要写"我帮你做了 X"这种主语，用中性陈述
- 长度上限 ~150 字（中文字数），宁可短不要凑

把 TL;DR 写到一个 tmp 文件，避免 shell 转义问题：

```bash
cat > /tmp/tldr.md <<'EOF'
<你写的 TL;DR markdown 文本>
EOF
```

### 3. 调用导出脚本

```bash
python3 .claude/skills/save-conversation/export_session.py \
  /home/gaosh/projects/agent_record/<filename>.md \
  --tldr-file /tmp/tldr.md
```

脚本会：
- 自动找当前 session 的 jsonl（取 `~/.claude/projects/-home-gaosh-projects-ZAsolar/` 里最近修改的 `.jsonl`）
- 把 TL;DR 注入到标题下方（`## TL;DR` section）
- 解析 user / assistant turn
- 把 tool call + tool result 折叠成 `<details>` 块
- 滤掉 `<system-reminder>`、`<task-notification>` 等 harness 注入的噪音
- 写到目标 markdown 路径

如果用户想导出**之前**的某次对话（不是当前），加 `--session <session-id>` 参数。session-id 是 jsonl 文件名（不含 `.jsonl` 后缀）。这种情况下你看不到那次对话内容，无法写 TL;DR——告诉用户先打开那个 session 再触发 skill，或者让用户自己提供 TL;DR 文本。

### 3. 确认

- 给用户报：文件名、用户/助手 turn 数量、字节数
- 如果脚本输出有 warning，原样转给用户

## Constraints

- **绝对不要**用 `/export` 内置命令——那个产生的是带 ANSI 字符的 terminal dump，不是干净 markdown
- **绝对不要**手写 markdown 来"总结"对话——导出脚本读 jsonl 是为了精确，不是为了总结
- 不要把 `agent_record/` 放进 git——它在 ZAsolar 仓库外面，这是故意的（不想 commit 数十 MB 的对话历史）
- 文件名必须是合法 POSIX 文件名（不能有 `/`、`:`、空格、emoji）
- 跨 session：当前 session 的 jsonl 是 `~/.claude/projects/-home-gaosh-projects-ZAsolar/` 里 mtime 最新的那个；如果用户要导出历史 session，让他们指定 session ID
