# RUN: Durban city-wide rule D — Dropbox upload (2026-10-13)

Decision (葉): no label override; rule-D results uploaded as-is.

- Pod: RTX 4090 (EU-RO-1, $0.89/h), 20:38–~22:26 NZDT, terminated; est. total ≈ $1.3–1.6.
- Scoring 30,604/30,604 ok; 6,124 nominated crops; Gemini (fpclean_v2_p3, cfg 6dbc5221c760df93, fp_conf 0.8):
  fp_candidate 3,276 / review 703 / keep 2,145. Only fp_candidate removed.
- BEFORE 30,604 blocks, 1,889,920 m²; AFTER 27,328 blocks, 1,635,102 m² (−3,276 blocks, −254,819 m²).
- Schema matches merged_iou010 (25 cols); 6 multipart blocks kept as MultiPolygon.

Uploaded (rclone copyto --ignore-existing; existing files untouched) to Dropbox `/RA_Solar/Gao/durban_census/`:

| file | bytes | sha256 |
|---|---|---|
| durban_census_inventory_stitch025_holefill_v3_before_20261012.gpkg | 172429312 | bdbf1736e83a83f4c11c7efee3fe6ed339bfe9a08bd7f22dd28a495635568260 |
| durban_census_inventory_stitch025_holefill_v3_ruleD_after_20261012.gpkg | 155103232 | 44d2bf9c9e19916eddf0252e0225ba599f64761c46133ab90e2cf1059f8771f5 |

Verified: sizes match; `rclone check --one-way` (Dropbox content hash) 0 differences, 2 matching.
Internal audit files (summary md/json, removed_blocks csv) not uploaded.
