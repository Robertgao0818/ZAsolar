# G14 building association restore — decision rule (written before any run, 光伏·运行, 2026-10-19 NZDT)
Source: ~/projects/zasolar_plans_20261017/G14_building_association_restore_design_v1.md (主管). Approval 葉 23:05 NZDT "恢复 G14 建筑关联".
- H1 supported iff, on BOTH sets (CT val 14 grids, JHB CBD25): building-level installed (≥2 m²) F1 ≥ object-level F1 of the same model
  (reference object F1 = zaeval-2 IoU≥0.5 h0.9_0.5: JHB from jhb_cbd25_zaeval2 RESULTS_v1; CT from ct_val_group_diag results_v1 merge),
  AND JHB GT installed-building count changes < 2% when row labels are merged at 1 m gap.
- H0 iff count change ≥ 5% under row merge, OR > 10% of GT PV area is not assigned to any building (off-building share).
- Between 2% and 5%: inconclusive; reported as such.
- Count agreement: |pred−GT|/GT installed buildings checked against ±5% (set level) and ±10% (stratum/H3-7 with n_gt ≥ 20). Indicative only (G4 tolerance applies to audit-calibrated totals).
- Model differences are diagnostic only (grid bootstrap, 2000 reps, seed 0); no model choice is made here.
- Stop: 2 consecutive failures (crash/OOM/mismatch) → stop and hand off.
