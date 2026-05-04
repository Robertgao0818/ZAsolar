# weekly-summary Skill

把一周日报翻译成给经济学/政策同事看的 slide-style outline，仓库 + Dropbox 双份镜像。

整合自：`~/.hermes/skills/note-taking/economist-weekly-summary-from-progress-logs/SKILL.md`

## When to use

用户说"读本周日报给经济学同事讲进度 / 写非技术同事能听懂的周报 / 上周三到本周二 weekly outline / 解释本周技术工作对经济学分析的意义"等触发。

不是给 ML 同行的 technical recap — 那种用日报本身或另起一份。这个 skill 的输出是**数据产品就绪度**视角：模型估的 grid 总装机面积、安装数量、采用率是否稳定、是否系统性高/低估、能否对接 building denominator 与社经变量。

## Inputs to discover

- 起止日期（含两端）。常见说法"上周三到本周二"——按对话日期解析，跨周目录要读两个 `week_*`
- 缺日期的日子要在 outline 里明示"该日无 diary"，不要编
- 是否需要 Dropbox 镜像（默认是）
- 风格参考：`docs/progress_log/week_*/outline_*.md` 里最近的一份

## Date range rules

- 显式解析日期，确认上下界
- 双源搜索：
  - Repo: `docs/progress_log/week_*/YYYY-MM-DD.md`
  - Dropbox: `/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/week_*/YYYY-MM-DD.md`
- 两份都在时优先 Dropbox（除非用户明说要 repo-first 对账）
- 跨周目录时两个 folder 都读

## Workflow

### 1. 找最近一份 outline 当格式参照

例如 `docs/progress_log/week_2026-04-14/outline_0415-0421.md` 或 `outline_0409-0414.md`，对齐章节、表格写法、口吻。

### 2. 收集日报

按日期范围读所有匹配的 `YYYY-MM-DD.md`。每份只抽 week-level 信号：

- 当日主线
- 关键定量结果
- 路线/决策变化
- validation / GT 质量发现
- 风险与未决问题
- 下周 TODO

不要复述实现细节，除非它驱动决策。

### 3. 翻译技术语言为经济学/政策语言

| 技术概念 | 经济学语言 |
|---|---|
| false positive / FP | 高估 solar adoption / capacity |
| false negative / FN | 低估扩散 / 漏掉 adopters |
| precision | 已检出系统中残留多少 over-counting 风险 |
| recall | 实际 adoption / capacity 抓到多少 |
| F1 | 诊断指标，不必是最终政策指标 |
| area F1 / matched area | 估的面板面积是否能用于容量/强度估算 |
| grid-level R² | 空间差异是否在不同单元间保持 |
| bulk ratio | 总量是否系统性偏高/偏低 |
| GT omissions | 验证标签可能低估模型表现并污染训练 |
| classifier | 二级 filter，压低非 PV 物体造成的 over-count |
| SAM refinement | 边界清洗，提升面积/容量估算 |
| adoption rate | PV-building 数 ÷ building/parcel 分母 |
| installed capacity / area | PV 强度 / 数量，不是 adoption rate |

每个技术叙述用这个 frame 收：

1. 做了什么技术工作
2. 它解决什么测量问题
3. 数字说了什么
4. 对经济学分析的数据可用性意味着什么
5. 还剩什么风险

### 4. 推荐 slide 结构

slide-style markdown，每个 section 一张"slide"。一周很满可以超过 9 张：

1. **Slide 1 - Week overview**：一句话点出本周主线变化。例 "from model debugging to economics-ready aggregate inventory validation"。
2. **Slide 2 - Daily progress overview table**：列 = 日期 / 主线 / 关键产出 / 对经济分析的意义。
3. **Slide 3 - Aggregate metrics / validation framing**：为什么 polygon F1 不够；定义 grid total area / bulk ratio / MRE / R²；附最重要的数字表。
4. **Slide 4 - Model-route decision**：当前首选 detector / refinement / classifier 路线；小对比表 + 直白解释。
5. **Slide 5 - Classifier / FP work**：为什么过滤非 PV 对 over-counting 重要；有 subtype 分布就放上来。
6. **Slide 6 - GT quality and validation risk**：缺标 / 补标 / 内部 GT 不完美对结论的影响。
7. **Slide 7 - Economics variable bridge**：installed capacity vs adoption rate；building footprint denominator 进展。
8. **Slide 8 - Risks and mitigations**：风险 / 影响 / 当前应对三列表。
9. **Slide 9 - Next-week TODO and discussion questions**：决策导向 — 指标优先级、分母选择、GT 升级、validation 局限。

末尾留**一段直接可读给经济学同事的话**。

### 5. 表格示例（直接搬用）

```markdown
| 日期 | 主线 | 关键产出 | 对经济分析的意义 |
|---|---|---|---|
| 04-22 | DeepSolar-style 项目定位 + aggregate evaluation | 新增 area-aggregate 指标 | 把模型输出从"单个 polygon 是否对"转成"每个 grid 总装机面积是否可信" |
| 04-28 | adoption rate / plausibility validation | 区分 installed capacity 和 adoption rate | 开始把太阳能数据转成经济学更常用指标 |
```

```markdown
| 指标 | 简单解释 | 为什么经济学同事会关心 |
|---|---|---|
| pred_total_m² | 模型认为该 grid 内太阳能板总面积 | 可近似转成装机容量或光伏密度 |
| bulk_pred_gt_ratio | 加总后，模型总量 / GT 总量 | 看整体高估/低估 |
| R² | 预测总量与 GT 总量的线性一致性 | 看是否保留空间分布差异 |
```

```markdown
| 风险 | 影响 | 当前应对 |
|---|---|---|
| GT 本身漏标 | 会误判 FP，并污染 classifier | subtype audit + supplement GT |
| classifier 跨域迁移差 | CT-trained 不能直接用于 JHB GEID | 加 JHB hard cases，按 imagery 分域校准 |
```

### 6. 保存与镜像

```bash
WEEK_DIR=docs/progress_log/week_YYYY-MM-DD   # 该周周一
RANGE=outline_MMDD-MMDD.md

# repo
write to "$WEEK_DIR/$RANGE"

# Dropbox 镜像
mkdir -p "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/week_YYYY-MM-DD"
cp "$WEEK_DIR/$RANGE" "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/week_YYYY-MM-DD/$RANGE"

# 校验 byte-identical
cmp -s "$WEEK_DIR/$RANGE" \
       "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/week_YYYY-MM-DD/$RANGE" \
  && echo identical
```

读回 repo 副本确认标题/日期范围对。

## Quality checklist

最后回复前自查：

- [ ] 日期范围明示并正确
- [ ] 缺 diary 的日子明确说出，不静默虚构
- [ ] 用经济学语言而不是 raw implementation
- [ ] 重要定量对比都解释了"意味着什么"
- [ ] FP/FN/F1/R²/bulk ratio 都翻译成测量含义
- [ ] 写清本周工作如何推进数据产品就绪度（aggregate inventory / capacity / adoption / validation / bias）
- [ ] 下周 TODO 与讨论问题是决策导向的
- [ ] repo + Dropbox 双份 byte-identical

## Common pitfalls

1. **堆 filename / 脚本** — 非技术同事看不懂；只在标识可复用指标/流程时给脚本名
2. **把 F1 当终极目标** — 对经济学受众，F1 是诊断；强调 aggregate 准确性、系统性 bias、空间可比
3. **忘 denominator** — adoption rate 需要 buildings / parcels；面板面积/容量不需要
4. **隐藏 validation 不确定性** — 没有外部 rooftop PV GT 时直接说，并解释 multi-channel 替代方案
5. **假设 GT 完美** — 发现的漏标 / 补标要写出来，影响模型评估和 classifier 训练
6. **只存一份** — repo 和 Dropbox 必须同步，除非用户明说只要一份
