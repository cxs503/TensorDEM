# Round 13 - provenance / consistency adjudication

Archival only: no new simulation. Sources:
* `runs/full_cell/point_0.01.json`
* `runs/full_cell/point_0.03.json`
* `runs/full_cell/point_0.1.json`
* `runs/full_cell/point_0.3.json`

## Table A - the disputed reference envelope

| I | PRIMARY_REPO (0.36/0.84/0.30) | TEST_GDR_MIDI (0.379/0.641/0.279) | envelope min | envelope max | rel. spread |
|---|---|---|---|---|---|
| 0.001 | 0.3616 | 0.3799 | 0.3609 | 0.3806 | 5.5% |
| 0.003 | 0.3648 | 0.3818 | 0.3628 | 0.3839 | 5.8% |
| 0.01 | 0.3755 | 0.3881 | 0.3691 | 0.3950 | 7.0% |
| 0.03 | 0.4036 | 0.4044 | 0.3855 | 0.4238 | 9.9% |
| 0.1 | 0.4800 | 0.4481 | 0.4303 | 0.5006 | 16.4% |
| 0.3 | 0.6000 | 0.5148 | 0.5005 | 0.6179 | 23.4% |

## Table B - is the archived (mu, I) pair law-consistent?

| I_target | rate_ratio | profile R^2 | servo | I archived | I(imposed, sigma_zz) | I(meas rate, p) | p/sigma_zz | spurious mu inflation | admissible |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.380 | 0.571 | PASS | 0.0102 | 0.0100 | 0.0044 | 0.766 | +30.6% | no |
| 0.03 | 0.971 | 0.878 | FAIL | 0.0314 | 0.0300 | 0.0312 | 0.902 | +10.8% | no |
| 0.1 | 0.669 | 0.898 | FAIL | 0.1038 | 0.1000 | 0.0728 | 0.882 | +13.4% | no |
| 0.3 | 0.908 | 0.969 | PASS | 0.3033 | 0.3000 | 0.3060 | 0.787 | +27.1% | yes |

Errors are **measured / reference - 1** (the sign convention of rounds 8-12).

## Table C - pass matrix (band 10 %), per arm x per candidate law

| arm | candidate | I=0.01 | I=0.03 | I=0.1 | I=0.3 | n pass | max |err| |
|---|---|---|---|---|---|---|---|
| zz-consistent, nominal (imposed) rate  [archived convention] | PRIMARY_REPO | -23.6% | -1.1% | -29.6% | -40.8% | 1/4 | 40.8% |
| zz-consistent, nominal (imposed) rate  [archived convention] | TEST_GDR_MIDI | -26.1% | -1.3% | -24.5% | -31.0% | 1/4 | 31.0% |
| zz-consistent, measured bulk rate | PRIMARY_REPO | -21.6% | -0.9% | -24.4% | -39.6% | 1/4 | 39.6% |
| zz-consistent, measured bulk rate | TEST_GDR_MIDI | -25.0% | -1.1% | -21.2% | -30.1% | 1/4 | 30.1% |
| p-consistent, nominal (imposed) rate | PRIMARY_REPO | -0.7% | +0.6% | -22.7% | -29.4% | 2/4 | 29.4% |
| p-consistent, nominal (imposed) rate | TEST_GDR_MIDI | -3.7% | +0.7% | -16.6% | -17.1% | 2/4 | 17.1% |
| p-consistent, measured bulk rate  [law-consistent] | PRIMARY_REPO | +2.2% | +0.9% | -16.9% | -28.1% | 2/4 | 28.1% |
| p-consistent, measured bulk rate  [law-consistent] | TEST_GDR_MIDI | -2.1% | +0.9% | -12.9% | -16.0% | 2/4 | 16.0% |
| wall traction, measured bulk rate | PRIMARY_REPO | -21.4% | +1.9% | -26.5% | -39.8% | 1/4 | 39.8% |
| wall traction, measured bulk rate | TEST_GDR_MIDI | -24.8% | +1.8% | -23.3% | -30.3% | 1/4 | 30.3% |

## Table D - can ANY Jop law fit the measured curve? (falsification oracle)

| arm | min worst-point err over the whole law family | argmin (mu_s, mu_2, I_0) | argmin on box edge? | #param sets inside 10 % |
|---|---|---|---|---|
| zz-consistent, nominal (imposed) rate  [archived convention] | **21.7%** | (0.340, 0.550, 0.380) | mu_s, mu_2, I_0 | 0 |
| zz-consistent, measured bulk rate | **20.3%** | (0.340, 0.550, 0.380) | mu_s, mu_2, I_0 | 0 |
| p-consistent, nominal (imposed) rate | **7.7%** | (0.365, 0.550, 0.380) | mu_2, I_0 | 120 |
| p-consistent, measured bulk rate  [law-consistent] | **6.1%** | (0.370, 0.550, 0.370) | mu_2 | 272 |
| wall traction, measured bulk rate | **20.7%** | (0.340, 0.550, 0.380) | mu_s, mu_2, I_0 | 0 |

An argmin sitting on the search-box edge means the optimum is **not identified**: the measurement is fitted only by pushing the parameters outside anything either candidate paper claims.  Read Table D together with Table D2.

## Table D2 - within the DISPUTED envelope only (8 corners)

| arm | #corners passing all points | best #points any corner passes | smallest worst-point err over the 8 corners |
|---|---|---|---|
| zz-consistent, nominal (imposed) rate  [archived convention] | 0/8 | 1/4 | 29.0% |
| zz-consistent, measured bulk rate | 0/8 | 1/4 | 28.0% |
| p-consistent, nominal (imposed) rate | 0/8 | 2/4 | 14.8% |
| p-consistent, measured bulk rate  [law-consistent] | 0/8 | 3/4 | 13.7% |
| wall traction, measured bulk rate | 0/8 | 1/4 | 28.3% |

## Table E - reverse calibration (what law DOES the device produce?)

| arm | mu_s | mu_2 | I_0 | rms | R^2 |
|---|---|---|---|---|---|
| zz-consistent, nominal (imposed) rate  [archived convention] | -0.390 | 0.371 | 0.001 | 0.0308 | +0.42 |
| zz-consistent, measured bulk rate | -0.007 | 0.370 | 0.001 | 0.0255 | +0.60 |
| p-consistent, nominal (imposed) rate | 0.381 | 1.838 | 10.000 | 0.0158 | +0.57 |
| p-consistent, measured bulk rate  [law-consistent] | 0.382 | 2.058 | 10.000 | 0.0143 | +0.65 |
| wall traction, measured bulk rate | -0.003 | 0.370 | 0.001 | 0.0327 | +0.47 |

## Headline

* law-consistent arm: `p-consistent, measured bulk rate  [law-consistent]`
* points inside the 10 % band vs the primary candidate: 2/4 (max |err| 28.1 %)
* corners of the disputed envelope passing every point: 0/8 (best corner reaches 3/4 points)
* best possible worst-point error over the **entire** Jop family: 6.1 % at (0.370, 0.550, 0.370), argmin on box edge: mu_2
* provenance can rescue the acceptance (identified fit): **NO**
* kinematically admissible points (rate_ratio>=0.9, R^2>=0.9, servo pass): 1/4
* worst spurious ordinate inflation from mixing pressure normalisers: +30.6 %
