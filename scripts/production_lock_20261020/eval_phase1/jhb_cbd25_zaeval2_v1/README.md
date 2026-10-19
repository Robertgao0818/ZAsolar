# Joburg CBD25 zaeval-2@8044d14 re-eval (光伏·运行, 2026-10-19)
- tmux 光伏运行-jhb_zaeval2. GPU infer 18:24–18:56 NZDT (run_infer.sh → infer_save_v1.py), CPU score 18:56–19:16 NZDT (run_score.sh → score_z2_v1.py), table_v1.py → RESULTS_v1.csv/md.
- probs/<model>/<grid>/<tile>.npz key p = uint8 rint(prob*255); official cell pregate (joburg25_vexcel/pregate/scores.json, 0.45) applied; 529/2500 tiles fully gated (zeros).
- GT clean_gt (buffer0 + dedupe as G6), mask 1/255 like G6 evalobj, gt_id pixel-centre, ignore mode exclude-pixels, ObjectAccumulator iou (0.3,0.5,0.7). avg3 = mean of 3 uint8/255 maps.
- Parity: legacy_cov columns ≈ G6 official (e.g. convnext h0.9 P .808/.808, R .928/.922, px IoU .832/.832); small diffs from exclude-pixels + pixel-centre gt_id + uint8 probs.
- scores_smoke/ = one-grid smoke test, not a result.
