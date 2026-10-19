已只读抽查，未修改任何文件。**目前不能按原计划整套执行。**

**BLOCKERS**

- **`git mv` 不适用于这些源目录。** `zasolar_plans_20261017`、`zasolar_roadmap_20261017` 均在仓库之外，也不是 Git 仓库；直接执行会失败。应先复制到新分支、核对完整性，再决定何时处理原目录。[计划位置](/home/gao/projects/zasolar_plans_20261017/tidy_proposal_v1/PROPOSAL_v1.md:4)
- **文件拆分和旧路径映射尚未具体化。** 符号链接就是让旧路径指向新位置的文件系统入口；它不能自动修复目录结构变化。例如 [boot.py](/home/gao/projects/zasolar_plans_20261017/jhb25_analysis_v1/boot.py:3) 从运行目录读取多个 CSV，把代码和数据分开会破坏原运行方式。计划还漏列 `matched/*.py`；新内容若只在独立工作树（同一仓库的另一份检出目录），旧链接也不能指向尚未更新的主目录。忽略这些会产生失效路径或遗漏文件。
- **无法完成宿主机进程安全核验。** 本次 `ps` 只显示隔离环境内进程，`tmux` 访问被拒绝，不能确认计划中的进程编号仍有效。“先移动、后建链接”仍可能造成读取中断；目前应保留原目录，仅制作副本。
- **当前环境无法验证或完成推送。** 只读查询远端分支已因 SSH 配置文件权限错误失败；因此无法确认远端 `reorg-20261017` 是否存在，也不能认定现在可推送。

**Non-blocking issues**

- **Mask R-CNN 只能按计划“打标签、写说明”，不能实际搬走。** 已确认 Durban 驱动依赖 `detect_direct.py`、`finalize.py`，HN 审核依赖 `core.inference.presence_pregate`，JHB 脚本读取旧结果路径。保留这些依赖即可继续整理；搬走会破坏后续运行。
- **主目录状态需要明确保留。** 当前确为 `aa23c2f`，但有未提交的 `ROADMAP.md` 修改和未跟踪的 `weights/`。从该提交创建独立分支不会带入它们；忽略会遗漏最新决策和本地权重。`evalfix` 建议暂不移动；固定评分工作树的 `.git` 确实指回其主目录。
- **STATUS 已过时且需补限定。** HN 目录已有 `summary_v1.json`、`verdicts_v1.csv`，日志以 `GROK_EXIT 0` 结束，不能再写“结果尚未产出”。模型比较还应保留“参考标注源自 Mask R-CNN 检测结果”的限制，否则容易把该比较误读为可立即退役的依据。v2e 数量及 Group A 的 179 个评分单元与抽查文件一致。

VERDICT: BLOCKED