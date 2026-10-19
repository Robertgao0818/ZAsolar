You are independently reviewing the hard-negative (HN) approach for a Cape Town rooftop-solar segmentation model. READ-ONLY: do not edit, create or delete any file, do not run GPU jobs or training. CPU-only inspection commands (python reading npz/csv/json) are fine.

Files:
- HN cleaning: ~/capetown_eval/ds3m_ft/hn_clean/hn_clean.py, hn_clean_summary.json, hn_clean_per_grid.csv, hn_clean_per_tile.csv, hn_clean.log
- Training script consuming the cache: ~/capetown_eval/ds3m_ft/train2_v2d2.py (study exactly how HN coords are sampled, at what ratio, crop size, and any loss weighting)
- Training sets: ~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017 and ct_seg_trainset_v2e_20261017 (read READMEs)
- HN points came from detector false-positive review (v3c_targeted_hn_aerial_2025 review gpkgs), stored in coord_cache.npz.

Claimed facts (verify against code): cleaning drops points where mask != 0, within 0.3 m of PV, within 0.3 m of the owner's trusted hand labels (only 8 grids). In v2d2, 27,855 of 145,998 points dropped, all due to mask not background.
Known issues: v2d1 regressed vs v2c (precision 0.892 -> 0.864; extra FPs on unchanged tiles); small panels <4 m2 weak; most grids lack trusted hand labels; Durban Gemini HN set unaudited and unused. Project is in exploration mode.

Assess:
1. Is the HN source trustworthy, given risk that HN points sit on real unlabelled PV?
2. Are the drop rules and 0.3 m radius right? Is dropping on mask==255 (ignore) correct?
3. HN sampling ratio and weight in training; could it hurt small-panel recall or explain the v2d1 regression?
4. Should HN be refreshed by mining the current model's FPs?
5. Concrete cheap CPU checks (e.g. sample N HN points per grid for visual or Gemini audit).
Give a verdict (sound / sound with changes / unsound) and a prioritized change list. Cite file:line for every code claim. Mark anything you could not verify.
