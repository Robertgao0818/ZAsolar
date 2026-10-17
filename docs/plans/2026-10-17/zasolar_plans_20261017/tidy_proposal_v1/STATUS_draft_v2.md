# ZAsolar STATUS draft v2 — 2026-10-17 (NZDT)
- **Phase-1 模型对比**: MiT-b2 baseline / SegNeXt-B / ConvNeXt-T+UPerNet (b3 optional), cap ~26 GPU-h, trains on v2e. 未开始，等葉说开。[ROADMAP_v4.md]
- **v2e**: train 79 grids / 5,645 polygons, val 14, eval 5; 12,370 masks independently verified. G2028 留在 val; G1749/G1972/G2029 ignore 编辑保留; hn_clean v2e 缓存已生成并核验（~/capetown_eval/ds3m_ft/hn_clean_v2e_20261017/README.md），训练前确认用这一份。[v2e/DECISIONS_20261017.md, v2e/verify_v2e_independent.json]
- **Li test set**: 17 grids, 1,260 scoring polygon pieces (1,246 source polygons inside cells); Group A 8 grids / 179 units / 447 polygons. [cross_grid_summary_v1.md] (Group A/B/C/D split from summary; not re-counted here)
- **zaeval-2**: scorer commit 8044d14 (59911a8 invalid). r2 rules (TEST_RULES_frozen_20261017_r2.md) 未签，暂停；测试打分、相邻标注合并决定均暂停。
- **HN audit** (~/capetown_eval/ds3m_ft/hn_crop_audit_v1_20261017, finished): 214 HN samples, 22 pass, 192 flagged (pregate 145, both 47, gemini-only 0). Method is pregate + Gemini, **not SAM3** as described in the brief — to be confirmed with 葉. Codex review: zasolar_hn_review_20261017/CODEX_REVIEW.md.
- **Durban v4**: native 0.075 m. [UNVERIFIED in files this pass]
- **Backdating**: 暂停; 用 3.5。[ROADMAP_v4.md]
- **Mask R-CNN superseded**: JHB25 F1 0.775 (vector re-impl 0.771) vs MiT v2c 0.899 [0.865,0.922]. [ROADMAP_v4.md:128, jhb25_analysis_v1.md:6,12] Caveat: JHB25 GT is detector-seeded (from Mask R-CNN detections), so this is not a clean comparison [ROADMAP_v4.md:267].
- **Cross-grid edge**: cross_grid_summary_v1.md; 葉 keeps current approach; exp_edge_handling_v1.md deferred; at full-city inference recommend city-wide stitch with one owner per object (CPU).
