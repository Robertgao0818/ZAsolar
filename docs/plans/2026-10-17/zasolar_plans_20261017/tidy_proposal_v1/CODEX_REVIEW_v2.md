**剩余 BLOCKERS：无。v1 的方案性阻断已解决。**

- 改为整目录复制、保留原目录，并纳入 `matched/*.py`，解决了仓库外移动失败、代码与数据拆分、旧路径失效的问题。
- 不移动 evalfix 和仍被使用的模型代码，保留主目录未提交内容，避免影响现有任务。
- 主机进程与 SSH（远程连接通道）由执行者核查，仍是执行前提；沙箱无法核验不再单独构成方案阻断。
- STATUS 已补上 HN 审计完成情况及参考标注来源限制，避免把模型比较误当作退役依据。

**一处非阻断修正：**[STATUS 第 3 行](/home/gao/projects/zasolar_plans_20261017/tidy_proposal_v1/STATUS_draft_v2.md:3)仍称缓存“需重建”，但[现有记录](/home/gao/capetown_eval/ds3m_ft/hn_clean_v2e_20261017/README.md:2)表明已生成并核验。宜改为“训练前确认使用已核验的 v2e 缓存”，否则会重复工作并误报进度。

VERDICT: OK