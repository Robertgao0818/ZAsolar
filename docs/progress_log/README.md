# Progress Log

This file is updated by `scripts/progress_tracker.py` and the optional git `post-commit` hook.

## Current Focus
- Semi-automatic annotation: expand to more grids via export_hints.py + SAM2 click-segment workflow
- FP reduction: hard negative mining, confidence threshold tuning

## Entries
- 2026-03-25T22:00:00Z | manual | V3 retrain: 43 grids, 2982 polygons, val_AP50=0.788 at epoch 4/20 (training paused). export_coco_dataset.py auto-discovers cleaned/ annotations.
- 2026-03-25T22:00:00Z | manual | Project cleanup: deleted old COCO datasets (coco_sam2_260320, coco_v1_*), intermediate epoch weights (~6G freed). Cloud migration plan: RunPod Pod + Network Volume.
- 2026-03-24T12:00:00Z | manual | Annotation expansion: 43 grids in data/annotations/cleaned/, cumulative 2982 installations (SAM2 cleaned)
- 2026-03-22T12:00:00Z | manual | Day annotation: 26 grids (batch 001×14 + batch 002×12), 1148 panels / 644 installations. Cumulative: 29 grids, 1623 panels / 876 installations
- 2026-03-22T12:00:00Z | manual | OSM building filter: tile-level selective download saving 82%; identified OSM lag issue in suburban expansion zones
- 2026-03-22T04:59:52Z | commit:0c75640f | Add OSM building filter for tile-level selective download
- 2026-03-22T04:59:35Z | commit:f91c42cd | V2 annotation expansion: SAM2 batch labeling pipeline, manifest update, and weekly report
- 2026-03-20T22:00:00Z | manual | V2 SAM2 annotation + retrain: 456 polygons (SAM2.1), val_AP50=0.6889 (+0.268), primary metric → F1@IoU0.5
- 2026-03-20T22:00:00Z | manual | Model-assisted annotation script (export_hints.py): confidence-tiered bbox hints for QGIS
- 2026-03-20T22:00:00Z | manual | GT resolution updated to SAM2 annotations; tiles synced to D:\ZAsolar for Windows QGIS
- 2026-03-18T10:33:15Z | commit:14a1f6b9 | Add annotation cleanup, docs restructure, JHB fine-tuned evaluation, and project configs
- 2026-03-18T01:14:38Z | commit:5e184594 | Restructure project: move shared utils to core/, group scripts by domain
- 2026-03-18T00:04:58Z | manual | Added low-resolution grid preview batching for G1240+ screening and contact-sheet generation.
- 2026-03-18T00:04:58Z | manual | Added browser review UI for keep/exclude/review decisions with WSL-friendly LAN access hints.
- 2026-03-18T00:04:58Z | manual | Completed the first 100-grid preview batch from G1240 and finished manual screening for that batch.
