## Table — five-case ladder (program-generated from verdict.json)

| case | a_nom | a_meas | gpr | N | $r_\infty/r_i$ | ref(a_meas) | err runout | $t_{95}$ [s] | ref $t$ [s] | err $t_{95}$ | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `R_a1_g8` | 1 | 1.0783 | 8 | 1955 | 2.1060 | 2.3370 | -9.88 % | 0.6684 | 0.6838 | -2.26 % | at_rest |
| `R_a1_g12` | 1 | 0.9885 | 12 | 6412 | 2.0782 | 2.2257 | -6.63 % | 0.6737 | 0.6588 | +2.26 % | at_rest |
| `R_a1_g16` | 1 | 1.0028 | 16 | 16030 | 2.0410 | 2.2434 | -9.02 % | 0.6584 | 0.6634 | -0.75 % | at_rest |
| `R_a3_g8` | 3 | 3.0866 | 8 | 5673 | 4.1193 | 3.8110 | +8.09 % | 1.1410 | 1.1570 | -1.39 % | at_rest |
| `R_a3_g12` | 3 | 3.0071 | 12 | 19687 | 4.3559 | 3.7746 | +15.40 % | 1.1285 | 1.1491 | -1.79 % | at_rest |
| `w5b_confirm_wall0.8_a2` | 2 | 2.0296 | 8 | 3718 | 3.4118 | 3.2794 | +4.04 % | 1.0591 | 0.9454 | +12.03 % | at_rest |
| `w5e_gpr12_a2` | 2 | 2.0326 | 12 | 13275 | 3.4313 | 3.2811 | +4.58 % | 1.3104 | 0.9461 | +38.51 % | at_rest |

## Pre-registered verdicts

| criterion | result | measured | threshold |
|---|---|---|---|
| `P1_A1_G16_BAND` | **pass** | 9.02% | 15% |
| `P2_A1_DRIFT` | **pass** | 2.40 pp | 5 pp |
| `P3_A3_G12_BAND` | **fail** | 15.40% | 15% |
| `P4_A3_DRIFT` | **pass** | 7.31 pp | 15 pp |
| `P5_TIME_A3_BAND` | **pass** | 1.79% | 15% |
| `P6_DEEP_FAMILY` | **fail** | 15.40% | 15% |
| `P7_INTEGRITY` | **pass** | stop_reason all `at_rest`; overlap clause `not_run (protocol archives no overlap count)` | — |
