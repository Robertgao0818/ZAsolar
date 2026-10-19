# Phase-1 val / eval metrics

Scorer: zaeval-2 `8044d1447a92cc100bbc01976bf66cb4245c89bb`. Device: CPU float32.

PV = photovoltaic panels; SWH = solar water / pool heaters. P/R/F1 are precision/recall/harmonic mean.
All rates below are computed from summed counts, never averaged tile rates. No test data.

Partial run: **True**. Do not use smoke subsets to rank architectures.

Object metrics use raw polygons rasterized at pixel centres, exclude-pixels (255 ignored),
8-connected cross-tile prediction components, minimum prediction area 0.25 m² and one-to-one IoU matching.
Pixel IoU uses the v2f class masks, so the eroded label band is not counted.
Boundary F uses raw polygon outlines, excludes region255, includes the polygon band, and uses neighbour halos.
Missing cardinal neighbours exclude boundaries within 1 m of the core tile edge.

| Split / metric | PV | SWH |
|---|---:|---:|
| eval: scored tiles | 2 | 2 |
| eval: pixel IoU | 0.86202355 | 0.00000000 |
| eval: object P / R / F1, IoU 0.3 | 0.35714286 / 0.71428571 / 0.47619048 | 0.00000000 / 0.00000000 / 0.00000000 |
| eval: object P / R / F1, IoU 0.5 | 0.35714286 / 0.71428571 / 0.47619048 | 0.00000000 / 0.00000000 / 0.00000000 |
| eval: object P / R / F1, IoU 0.7 | 0.35714286 / 0.71428571 / 0.47619048 | 0.00000000 / 0.00000000 / 0.00000000 |
| eval: boundary F, 0.25 m | 0.48408018 | 0.00000000 |
| eval: boundary F, 0.5 m | 0.77450838 | 0.00000000 |
| eval: recall lt4 m² @ IoU 0.3 | N/A (0 GT) | N/A (0 GT) |
| eval: recall 4_10 m² @ IoU 0.3 | 1.00000000 (1/1) | N/A (0 GT) |
| eval: recall ge10 m² @ IoU 0.3 | 0.66666667 (4/6) | 0.00000000 (0/2) |
| eval: recall lt4 m² @ IoU 0.5 | N/A (0 GT) | N/A (0 GT) |
| eval: recall 4_10 m² @ IoU 0.5 | 1.00000000 (1/1) | N/A (0 GT) |
| eval: recall ge10 m² @ IoU 0.5 | 0.66666667 (4/6) | 0.00000000 (0/2) |
| eval: recall lt4 m² @ IoU 0.7 | N/A (0 GT) | N/A (0 GT) |
| eval: recall 4_10 m² @ IoU 0.7 | 1.00000000 (1/1) | N/A (0 GT) |
| eval: recall ge10 m² @ IoU 0.7 | 0.66666667 (4/6) | 0.00000000 (0/2) |
| val: scored tiles | 2 | 2 |
| val: pixel IoU | 0.67838563 | 0.00000000 |
| val: object P / R / F1, IoU 0.3 | 0.53846154 / 0.87500000 / 0.66666667 | 0.00000000 / 0.00000000 / 0.00000000 |
| val: object P / R / F1, IoU 0.5 | 0.46153846 / 0.75000000 / 0.57142857 | 0.00000000 / 0.00000000 / 0.00000000 |
| val: object P / R / F1, IoU 0.7 | 0.38461538 / 0.62500000 / 0.47619048 | 0.00000000 / 0.00000000 / 0.00000000 |
| val: boundary F, 0.25 m | 0.51752677 | 0.00000000 |
| val: boundary F, 0.5 m | 0.74443616 | 0.00000000 |
| val: recall lt4 m² @ IoU 0.3 | N/A (0 GT) | N/A (0 GT) |
| val: recall 4_10 m² @ IoU 0.3 | 1.00000000 (3/3) | N/A (0 GT) |
| val: recall ge10 m² @ IoU 0.3 | 0.80000000 (4/5) | 0.00000000 (0/2) |
| val: recall lt4 m² @ IoU 0.5 | N/A (0 GT) | N/A (0 GT) |
| val: recall 4_10 m² @ IoU 0.5 | 0.66666667 (2/3) | N/A (0 GT) |
| val: recall ge10 m² @ IoU 0.5 | 0.80000000 (4/5) | 0.00000000 (0/2) |
| val: recall lt4 m² @ IoU 0.7 | N/A (0 GT) | N/A (0 GT) |
| val: recall 4_10 m² @ IoU 0.7 | 0.33333333 (1/3) | N/A (0 GT) |
| val: recall ge10 m² @ IoU 0.7 | 0.80000000 (4/5) | 0.00000000 (0/2) |

Zero-denominator object rates retain zaeval defaults (0); use the accompanying counts to distinguish absence from failure.
SWH is scored only where swh_labelled=True. Unlabelled grids are not evidence of absence.
The imported training hit-rule diagnostic uses argmax and raw polygons; it is not the training best_objf1 metric.
