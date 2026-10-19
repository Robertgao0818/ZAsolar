## Building level
| city   | model    |   n_bld_eval |   gt_installed |   pred_installed |   tp |   fp |   fn |     P | P_ci             |     R | R_ci             |    F1 | F1_boot_ci                             |   count_err | within_5pct   | within_10pct   |   obj_F1_ref | bld_F1_ge_obj_F1   |   erf_n |   erf_gt |   erf_pred |   erf_P | erf_P_ci         |   erf_R | erf_R_ci         |   erf_F1 |   erf_count_err |
|:-------|:---------|-------------:|---------------:|-----------------:|-----:|-----:|-----:|------:|:-----------------|------:|:-----------------|------:|:---------------------------------------|------------:|:--------------|:---------------|-------------:|:-------------------|--------:|---------:|-----------:|--------:|:-----------------|--------:|:-----------------|---------:|----------------:|
| CT     | convnext |         7906 |            574 |              628 |  563 |   65 |   11 | 0.896 | (0.8702, 0.918)  | 0.981 | (0.966, 0.9893)  | 0.937 | (np.float64(0.916), np.float64(0.95))  |      0.0941 | False         | True           |        0.836 | True               |    3769 |      501 |        553 |   0.893 | (0.8648, 0.9164) |   0.986 | (0.9714, 0.9932) |    0.937 |          0.1038 |
| CT     | b2       |         7906 |            574 |              630 |  552 |   78 |   22 | 0.876 | (0.8482, 0.8997) | 0.962 | (0.9427, 0.9746) | 0.917 | (np.float64(0.889), np.float64(0.937)) |      0.0976 | False         | True           |        0.829 | True               |    3769 |      501 |        553 |   0.877 | (0.847, 0.9018)  |   0.968 | (0.9488, 0.9802) |    0.92  |          0.1038 |
| CT     | avg2     |         7906 |            574 |              601 |  552 |   49 |   22 | 0.918 | (0.8938, 0.9378) | 0.962 | (0.9427, 0.9746) | 0.94  | (np.float64(0.921), np.float64(0.953)) |      0.047  | True          | True           |        0.85  | True               |    3769 |      501 |        528 |   0.919 | (0.8921, 0.939)  |   0.968 | (0.9488, 0.9802) |    0.943 |          0.0539 |
| JHB    | convnext |         9996 |            570 |              694 |  556 |  138 |   14 | 0.801 | (0.7698, 0.8292) | 0.975 | (0.9592, 0.9853) | 0.88  | (np.float64(0.823), np.float64(0.915)) |      0.2175 | False         | False          |        0.679 | True               |     nan |      nan |        nan | nan     | nan              | nan     | nan              |  nan     |        nan      |
| JHB    | b2       |         9996 |            570 |              691 |  556 |  135 |   14 | 0.805 | (0.7734, 0.8325) | 0.975 | (0.9592, 0.9853) | 0.882 | (np.float64(0.827), np.float64(0.916)) |      0.2123 | False         | False          |        0.709 | True               |     nan |      nan |        nan | nan     | nan              | nan     | nan              |  nan     |        nan      |
| JHB    | v2c      |         9996 |            570 |              659 |  552 |  107 |   18 | 0.838 | (0.8075, 0.8638) | 0.968 | (0.9506, 0.9799) | 0.898 | (np.float64(0.851), np.float64(0.929)) |      0.1561 | False         | False          |        0.723 | True               |     nan |      nan |        nan | nan     | nan              | nan     | nan              |  nan     |        nan      |
| JHB    | avg2     |         9996 |            570 |              652 |  549 |  103 |   21 | 0.842 | (0.812, 0.868)   | 0.963 | (0.9443, 0.9758) | 0.899 | (np.float64(0.852), np.float64(0.926)) |      0.1439 | False         | False          |        0.71  | True               |     nan |      nan |        nan | nan     | nan              | nan     | nan              |  nan     |        nan      |

## Strata n_gt>=20
| city   | model    | stratum         |   n_gt |   pred |   count_err | within_10pct   |
|:-------|:---------|:----------------|-------:|-------:|------------:|:---------------|
| CT     | convnext | H_R-formal      |    102 |    107 |      0.049  | True           |
| CT     | convnext | L_R-formal      |    270 |    288 |      0.0667 | True           |
| CT     | convnext | M_R-formal      |    178 |    205 |      0.1517 | False          |
| CT     | convnext | Sparse          |     24 |     28 |      0.1667 | False          |
| CT     | b2       | H_R-formal      |    102 |    108 |      0.0588 | True           |
| CT     | b2       | L_R-formal      |    270 |    293 |      0.0852 | True           |
| CT     | b2       | M_R-formal      |    178 |    202 |      0.1348 | False          |
| CT     | b2       | Sparse          |     24 |     27 |      0.125  | False          |
| CT     | avg2     | H_R-formal      |    102 |    104 |      0.0196 | True           |
| CT     | avg2     | L_R-formal      |    270 |    279 |      0.0333 | True           |
| CT     | avg2     | M_R-formal      |    178 |    191 |      0.073  | True           |
| CT     | avg2     | Sparse          |     24 |     27 |      0.125  | False          |
| JHB    | convnext | 87bcc3528ffffff |     53 |     64 |      0.2075 | False          |
| JHB    | convnext | 87bcc3529ffffff |    199 |    207 |      0.0402 | True           |
| JHB    | convnext | 87bcc352cffffff |     67 |     89 |      0.3284 | False          |
| JHB    | convnext | 87bcc352dffffff |     96 |    120 |      0.25   | False          |
| JHB    | convnext | 87bcc3cd3ffffff |     65 |     94 |      0.4462 | False          |
| JHB    | convnext | 87bcc3cdeffffff |     67 |     94 |      0.403  | False          |
| JHB    | b2       | 87bcc3528ffffff |     53 |     64 |      0.2075 | False          |
| JHB    | b2       | 87bcc3529ffffff |    199 |    208 |      0.0452 | True           |
| JHB    | b2       | 87bcc352cffffff |     67 |     86 |      0.2836 | False          |
| JHB    | b2       | 87bcc352dffffff |     96 |    119 |      0.2396 | False          |
| JHB    | b2       | 87bcc3cd3ffffff |     65 |     93 |      0.4308 | False          |
| JHB    | b2       | 87bcc3cdeffffff |     67 |     93 |      0.3881 | False          |
| JHB    | v2c      | 87bcc3528ffffff |     53 |     60 |      0.1321 | False          |
| JHB    | v2c      | 87bcc3529ffffff |    199 |    206 |      0.0352 | True           |
| JHB    | v2c      | 87bcc352cffffff |     67 |     78 |      0.1642 | False          |
| JHB    | v2c      | 87bcc352dffffff |     96 |    110 |      0.1458 | False          |
| JHB    | v2c      | 87bcc3cd3ffffff |     65 |     87 |      0.3385 | False          |
| JHB    | v2c      | 87bcc3cdeffffff |     67 |     90 |      0.3433 | False          |
| JHB    | avg2     | 87bcc3528ffffff |     53 |     60 |      0.1321 | False          |
| JHB    | avg2     | 87bcc3529ffffff |    199 |    202 |      0.0151 | True           |
| JHB    | avg2     | 87bcc352cffffff |     67 |     80 |      0.194  | False          |
| JHB    | avg2     | 87bcc352dffffff |     96 |    111 |      0.1562 | False          |
| JHB    | avg2     | 87bcc3cd3ffffff |     65 |     89 |      0.3692 | False          |
| JHB    | avg2     | 87bcc3cdeffffff |     67 |     84 |      0.2537 | False          |

## Row split
| city   | model    |   gap_m |   F1_orig |   F1_merged |     dF1 |
|:-------|:---------|--------:|----------:|------------:|--------:|
| CT     | convnext |    0.25 |     0.937 |       0.937 |  0      |
| CT     | convnext |    1    |     0.937 |       0.936 | -0.0008 |
| CT     | b2       |    0.25 |     0.917 |       0.917 |  0      |
| CT     | b2       |    1    |     0.917 |       0.916 | -0.0008 |
| CT     | avg2     |    0.25 |     0.94  |       0.94  |  0      |
| CT     | avg2     |    1    |     0.94  |       0.939 | -0.0008 |
| JHB    | convnext |    0.25 |     0.88  |       0.88  |  0      |
| JHB    | convnext |    1    |     0.88  |       0.88  |  0      |
| JHB    | b2       |    0.25 |     0.882 |       0.882 |  0      |
| JHB    | b2       |    1    |     0.882 |       0.882 |  0      |
| JHB    | v2c      |    0.25 |     0.898 |       0.898 |  0      |
| JHB    | v2c      |    1    |     0.898 |       0.898 |  0      |
| JHB    | avg2     |    0.25 |     0.899 |       0.899 |  0      |
| JHB    | avg2     |    1    |     0.899 |       0.899 |  0      |

## Area
| city   | source   |   n_polys |   pv_m2 |   assigned_to_eval_bld_m2 |   off_bld_share |   off_carport_candidate |   off_ground_mount_candidate |   off_misaligned_candidate |   off_missing_building_candidate |   off_small_or_ambiguous |
|:-------|:---------|----------:|--------:|--------------------------:|----------------:|------------------------:|-----------------------------:|---------------------------:|---------------------------------:|-------------------------:|
| CT     | gt0.0    |       967 |   36492 |                     35396 |          0.0192 |                  0.0011 |                       0.0108 |                     0.0058 |                           0.0014 |                 nan      |
| CT     | gt0.25   |       896 |   36682 |                     35589 |          0.0193 |                  0.0011 |                       0.0108 |                     0.0059 |                           0.0014 |                 nan      |
| CT     | gt1.0    |       796 |   37451 |                     36253 |          0.019  |                  0.0011 |                       0.0107 |                     0.0058 |                           0.0014 |                 nan      |
| CT     | convnext |      1021 |   37187 |                     35727 |          0.0294 |                  0.0031 |                       0.0109 |                     0.0072 |                           0.0082 |                   0.0001 |
| CT     | b2       |       984 |   38118 |                     35763 |          0.0502 |                  0.0018 |                       0.0397 |                     0.0048 |                           0.004  |                 nan      |
| CT     | avg2     |       955 |   36464 |                     35230 |          0.0233 |                  0.0028 |                       0.0103 |                     0.0063 |                           0.0039 |                 nan      |
| JHB    | gt0.0    |      2159 |  159371 |                    156641 |          0.0029 |                nan      |                       0.0013 |                     0.0012 |                           0.0004 |                 nan      |
| JHB    | gt0.25   |      1974 |  154543 |                    152812 |          0.0032 |                nan      |                       0.0016 |                     0.0012 |                           0.0004 |                 nan      |
| JHB    | gt1.0    |      1447 |  158708 |                    156991 |          0.003  |                nan      |                       0.0016 |                     0.001  |                           0.0004 |                 nan      |
| JHB    | convnext |      1770 |  174463 |                    172362 |          0.0051 |                nan      |                       0.0016 |                     0.0022 |                           0.0012 |                   0.0001 |
| JHB    | b2       |      1808 |  172290 |                    170499 |          0.0036 |                nan      |                       0.0015 |                     0.0015 |                           0.0007 |                   0      |
| JHB    | v2c      |      1693 |  167076 |                    165241 |          0.0038 |                nan      |                       0.0015 |                     0.0014 |                           0.0009 |                 nan      |
| JHB    | avg2     |      1632 |  169502 |                    167736 |          0.0035 |                nan      |                       0.0015 |                     0.0013 |                           0.0006 |                   0      |

## extra
```
{
 "CT_boot_b2-convnext": {
  "diff": -0.0201,
  "ci95": [
   -0.0314,
   -0.0102
  ]
 },
 "CT_boot_avg2-convnext": {
  "diff": 0.0028,
  "ci95": [
   -0.0028,
   0.0102
  ]
 },
 "CT_rowsplit_gt": {
  "0.0": {
   "gt_installed": 574,
   "d_pct": 0.0
  },
  "0.25": {
   "gt_installed": 574,
   "d_pct": 0.0
  },
  "1.0": {
   "gt_installed": 575,
   "d_pct": 0.17
  },
  "n_polys_gap0.0": 967,
  "n_polys_gap0.25": 896,
  "n_polys_gap1.0": 796
 },
 "JHB_boot_b2-convnext": {
  "diff": 0.0017,
  "ci95": [
   -0.0153,
   0.0185
  ]
 },
 "JHB_boot_v2c-convnext": {
  "diff": 0.0181,
  "ci95": [
   0.0046,
   0.0347
  ]
 },
 "JHB_boot_avg2-convnext": {
  "diff": 0.0187,
  "ci95": [
   0.0077,
   0.0344
  ]
 },
 "JHB_rowsplit_gt": {
  "0.0": {
   "gt_installed": 570,
   "d_pct": 0.0
  },
  "0.25": {
   "gt_installed": 570,
   "d_pct": 0.0
  },
  "1.0": {
   "gt_installed": 570,
   "d_pct": 0.0
  },
  "n_polys_gap0.0": 2159,
  "n_polys_gap0.25": 1974,
  "n_polys_gap1.0": 1447
 },
 "CT_gt_off_bld_share": 0.0192,
 "JHB_gt_off_bld_share": 0.0029
}
```
