# Mask R-CNN superseded (2026-10-17)
SegFormer/MiT models outperform: JHB25 F1 Mask R-CNN 0.775 vs MiT v2c 0.899 (ROADMAP_v4.md:128; jhb25_analysis_v1.md). Caveat: JHB25 GT is detector-seeded.
Not moved: detect_direct.py, finalize.py, train.py, core/models/maskrcnn.py, core/inference/* stay — used by Durban v4 census, presence pregate (HN audit), JHB25 eval scripts in ~/capetown_eval.
Frozen reference: tag `maskrcnn-final-20261017` (aa23c2f). Env: requirements.lock.txt. Weights: see ARCHIVE_LOCATIONS.md (not in git).
