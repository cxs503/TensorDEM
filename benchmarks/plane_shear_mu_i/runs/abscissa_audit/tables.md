# Round 10 — abscissa / window / gate matrix (generated, do not hand-edit)

band = ±10 % on `mu`; `nan` never counts as a pass.

## full_cell  (n = 4, gate-passing = 2)

| I_target | I_imposed_measP | I_vol | I_bulk | rate ratio | profile R2 | servo |
|---|---|---|---|---|---|---|
| 0.01 | 0.0102 | 0.0039 | 0.0059 | 0.38 | 0.57 | True |
| 0.03 | 0.0314 | 0.0305 | 0.0350 | 0.97 | 0.88 | False |
| 0.1 | 0.1038 | 0.0695 | 0.0736 | 0.67 | 0.90 | False |
| 0.3 | 0.3033 | 0.2753 | 0.3454 | 0.91 | 0.97 | True |

### ordinate `mu_full` — signed error [%] vs `mu_ref` at each abscissa

| I_target | I_target | I_imposed_measP | I_vol | I_bulk |
|---|---|---|---|---|
| 0.01 |    -6.5 |    -6.5 |    -4.1 |    -4.9 |
| 0.03 |   -14.4 |   -14.8 |   -14.5 |   -15.7 |
| 0.1 |   -23.0 |   -23.6 |   -18.0 |   -18.7 |
| 0.3 |   -27.2 |   -27.3 |   -25.9 |   -29.2 |

### ordinate `mu_full_final` — signed error [%] vs `mu_ref` at each abscissa

| I_target | I_target | I_imposed_measP | I_vol | I_bulk |
|---|---|---|---|---|
| 0.01 |    -0.1 |    -0.2 |    +2.4 |    +1.6 |
| 0.03 |    +1.3 |    +0.9 |    +1.2 |    -0.3 |
| 0.1 |   -21.4 |   -22.0 |   -16.2 |   -17.0 |
| 0.3 |   -27.8 |   -27.9 |   -26.5 |   -29.8 |

### ordinate `mu_full_zz` — signed error [%] vs `mu_ref` at each abscissa

| I_target | I_target | I_imposed_measP | I_vol | I_bulk |
|---|---|---|---|---|
| 0.01 |   -28.0 |   -28.0 |   -26.1 |   -26.8 |
| 0.03 |   -18.9 |   -19.3 |   -19.0 |   -20.2 |
| 0.1 |   -31.1 |   -31.5 |   -26.5 |   -27.2 |
| 0.3 |   -41.3 |   -41.4 |   -40.3 |   -42.9 |

### ordinate `mu_full_zz_final` — signed error [%] vs `mu_ref` at each abscissa

| I_target | I_target | I_imposed_measP | I_vol | I_bulk |
|---|---|---|---|---|
| 0.01 |   -23.6 |   -23.7 |   -21.6 |   -22.3 |
| 0.03 |    -1.0 |    -1.4 |    -1.1 |    -2.5 |
| 0.1 |   -29.3 |   -29.8 |   -24.6 |   -25.3 |
| 0.3 |   -40.8 |   -40.9 |   -39.8 |   -42.4 |

### abscissa budget (`mu_ref(I_vol)/mu_ref(I_target) - 1`)

| I_target | delta_ref [%] | explained (tail) | explained (final) |
|---|---|---|---|
| 0.01 |    -2.5 |   +0.39 |     nan |
| 0.03 |    +0.2 |   -0.01 |   +0.13 |
| 0.1 |    -6.2 |   +0.27 |   +0.29 |
| 0.3 |    -1.7 |   +0.06 |   +0.06 |

### pass counts (cells inside ±10 %)

| cell | all points | gate only |
|---|---|---|
| I_target|mu_full | 1/4 | 1/2 |
| I_target|mu_full_final | 2/4 | 1/2 |
| I_target|mu_full_zz | 0/4 | 0/2 |
| I_target|mu_full_zz_final | 1/4 | 0/2 |
| I_imposed_measP|mu_full | 1/4 | 1/2 |
| I_imposed_measP|mu_full_final | 2/4 | 1/2 |
| I_imposed_measP|mu_full_zz | 0/4 | 0/2 |
| I_imposed_measP|mu_full_zz_final | 1/4 | 0/2 |
| I_vol|mu_full | 1/4 | 1/2 |
| I_vol|mu_full_final | 2/4 | 1/2 |
| I_vol|mu_full_zz | 0/4 | 0/2 |
| I_vol|mu_full_zz_final | 1/4 | 0/2 |
| I_bulk|mu_full | 1/4 | 1/2 |
| I_bulk|mu_full_final | 2/4 | 1/2 |
| I_bulk|mu_full_zz | 0/4 | 0/2 |
| I_bulk|mu_full_zz_final | 1/4 | 0/2 |

## bulk_rheometry  (n = 4, gate-passing = 2)

| I_target | I_imposed_measP | I_vol | I_bulk | rate ratio | profile R2 | servo |
|---|---|---|---|---|---|---|
| 0.01 | 0.0102 | 0.0039 | 0.0059 | 0.38 | 0.57 | True |
| 0.03 | 0.0314 | 0.0305 | 0.0350 | 0.97 | 0.88 | False |
| 0.1 | 0.1038 | 0.0695 | 0.0736 | 0.67 | 0.90 | False |
| 0.3 | 0.3033 | 0.2753 | 0.3454 | 0.91 | 0.97 | True |

### ordinate `mu_wall` — signed error [%] vs `mu_ref` at each abscissa

| I_target | I_target | I_imposed_measP | I_vol | I_bulk |
|---|---|---|---|---|
| 0.01 |   -27.8 |   -27.9 |   -26.0 |   -26.6 |
| 0.03 |   -17.6 |   -17.9 |   -17.7 |   -18.9 |
| 0.1 |   -31.2 |   -31.6 |   -26.6 |   -27.3 |
| 0.3 |   -41.4 |   -41.5 |   -40.3 |   -43.0 |

### ordinate `mu_wall_final` — signed error [%] vs `mu_ref` at each abscissa

| I_target | I_target | I_imposed_measP | I_vol | I_bulk |
|---|---|---|---|---|
| 0.01 |   -23.3 |   -23.4 |   -21.4 |   -22.0 |
| 0.03 |    +2.1 |    +1.6 |    +1.9 |    +0.5 |
| 0.1 |   -31.1 |   -31.5 |   -26.5 |   -27.2 |
| 0.3 |   -40.9 |   -41.0 |   -39.8 |   -42.5 |

### abscissa budget (`mu_ref(I_vol)/mu_ref(I_target) - 1`)

| I_target | delta_ref [%] | explained (tail) | explained (final) |
|---|---|---|---|
| 0.01 |    -2.5 |   +0.09 |   +0.11 |
| 0.03 |    +0.2 |   -0.01 |   +0.08 |
| 0.1 |    -6.2 |   +0.20 |   +0.20 |
| 0.3 |    -1.7 |   +0.04 |   +0.04 |

### pass counts (cells inside ±10 %)

| cell | all points | gate only |
|---|---|---|
| I_target|mu_wall | 0/4 | 0/2 |
| I_target|mu_wall_final | 1/4 | 0/2 |
| I_imposed_measP|mu_wall | 0/4 | 0/2 |
| I_imposed_measP|mu_wall_final | 1/4 | 0/2 |
| I_vol|mu_wall | 0/4 | 0/2 |
| I_vol|mu_wall_final | 1/4 | 0/2 |
| I_bulk|mu_wall | 0/4 | 0/2 |
| I_bulk|mu_wall_final | 1/4 | 0/2 |
