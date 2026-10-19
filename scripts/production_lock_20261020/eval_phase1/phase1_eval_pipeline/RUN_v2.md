# Phase-1 follow-up run record

Status: complete after full GPU evaluation, independent checks, coverage and portable bundle verification.

Source of truth: TEST_REPORT_v2.md, visual_qa_v2.md, full_gpu_driver_status_v2.json, coverage_eval_orig10.csv, pod_bundle/README_v2.md and pod_bundle/RELEASE_NOTES_v2.md.

Decisions: only one GPU run; run_v2c/best.pt strictly compatible; actual training model source is unchanged train2_v2e.py because local claimed pinned file is an audit helper with no build_model. Claimed commit 2b2fdea remains unverified. Full eligible val=1703, eval=632; no Li/cache access; pod entry uses explicit bundled MSCAN init and separate PV/SWH recomputation. All original task files preserved; LOG.md appended by authorization. No uploads.
