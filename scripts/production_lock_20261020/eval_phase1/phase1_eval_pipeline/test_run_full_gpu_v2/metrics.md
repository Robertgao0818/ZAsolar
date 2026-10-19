# Phase-1 val / eval metrics

Scorer: zaeval-2 `8044d1447a92cc100bbc01976bf66cb4245c89bb`. Device: cuda bfloat16.

PV = photovoltaic panels; SWH = solar water / pool heaters. P/R/F1 are precision/recall/harmonic mean.
All rates below are computed from summed counts, never averaged tile rates. No test data.

Partial run: **False**. Do not use smoke subsets to rank architectures.

Object metrics use raw polygons rasterized at pixel centres, exclude-pixels (255 ignored),
8-connected cross-tile prediction components, minimum prediction area 0.25 m² and one-to-one IoU matching.
Pixel IoU uses the v2f class masks, so the eroded label band is not counted.
Boundary F uses raw polygon outlines, excludes region255, includes the polygon band, and uses neighbour halos.
Missing cardinal neighbours exclude boundaries within 1 m of the core tile edge.

| Split / metric | PV | SWH |
|---|---:|---:|
| eval: scored tiles | 632 | 505 |
| eval: pixel IoU | 0.91124778 | 0.20271872 |
| eval: object P / R / F1, IoU 0.3 | 0.87537994 / 0.87272727 / 0.87405159 | 0.22000000 / 0.31428571 / 0.25882353 |
| eval: object P / R / F1, IoU 0.5 | 0.86018237 / 0.85757576 / 0.85887709 | 0.20000000 / 0.28571429 / 0.23529412 |
| eval: object P / R / F1, IoU 0.7 | 0.77203647 / 0.76969697 / 0.77086495 | 0.10000000 / 0.14285714 / 0.11764706 |
| eval: boundary F, 0.25 m | 0.86593428 | 0.19331940 |
| eval: boundary F, 0.5 m | 0.92576509 | 0.29912174 |
| eval: recall lt4 m² @ IoU 0.3 | 1.00000000 (1/1) | N/A (0 GT) |
| eval: recall 4_10 m² @ IoU 0.3 | 0.73972603 (54/73) | 0.00000000 (0/12) |
| eval: recall ge10 m² @ IoU 0.3 | 0.91015625 (233/256) | 0.47826087 (11/23) |
| eval: recall lt4 m² @ IoU 0.5 | 1.00000000 (1/1) | N/A (0 GT) |
| eval: recall 4_10 m² @ IoU 0.5 | 0.73972603 (54/73) | 0.00000000 (0/12) |
| eval: recall ge10 m² @ IoU 0.5 | 0.89062500 (228/256) | 0.43478261 (10/23) |
| eval: recall lt4 m² @ IoU 0.7 | 1.00000000 (1/1) | N/A (0 GT) |
| eval: recall 4_10 m² @ IoU 0.7 | 0.69863014 (51/73) | 0.00000000 (0/12) |
| eval: recall ge10 m² @ IoU 0.7 | 0.78906250 (202/256) | 0.21739130 (5/23) |
| val: scored tiles | 1703 | 985 |
| val: pixel IoU | 0.89533607 | 0.08806063 |
| val: object P / R / F1, IoU 0.3 | 0.82509158 / 0.88333333 / 0.85321970 | 0.05521472 / 0.33333333 / 0.09473684 |
| val: object P / R / F1, IoU 0.5 | 0.79945055 / 0.85588235 / 0.82670455 | 0.03067485 / 0.18518519 / 0.05263158 |
| val: object P / R / F1, IoU 0.7 | 0.75457875 / 0.80784314 / 0.78030303 | 0.01840491 / 0.11111111 / 0.03157895 |
| val: boundary F, 0.25 m | 0.79480095 | 0.06365916 |
| val: boundary F, 0.5 m | 0.89089730 | 0.14106512 |
| val: recall lt4 m² @ IoU 0.3 | 0.54545455 (6/11) | N/A (0 GT) |
| val: recall 4_10 m² @ IoU 0.3 | 0.74561404 (170/228) | 0.22222222 (2/9) |
| val: recall ge10 m² @ IoU 0.3 | 0.92829706 (725/781) | 0.38888889 (7/18) |
| val: recall lt4 m² @ IoU 0.5 | 0.54545455 (6/11) | N/A (0 GT) |
| val: recall 4_10 m² @ IoU 0.5 | 0.71052632 (162/228) | 0.11111111 (1/9) |
| val: recall ge10 m² @ IoU 0.5 | 0.90268886 (705/781) | 0.22222222 (4/18) |
| val: recall lt4 m² @ IoU 0.7 | 0.54545455 (6/11) | N/A (0 GT) |
| val: recall 4_10 m² @ IoU 0.7 | 0.66228070 (151/228) | 0.11111111 (1/9) |
| val: recall ge10 m² @ IoU 0.7 | 0.85403329 (667/781) | 0.11111111 (2/18) |

Zero-denominator object rates retain zaeval defaults (0); use the accompanying counts to distinguish absence from failure.
SWH is scored only where swh_labelled=True. Unlabelled grids are not evidence of absence.
The imported training hit-rule diagnostic uses argmax and raw polygons; it is not the training best_objf1 metric.
