### Table 1 - arm design

| arm | kn (N/m) | box W x D x H (m) | d_layers | P*d/kn | step scale | N_p | settle compression | p/p_target (worst) |
|---|---|---|---|---|---|---|---|---|
| baseline | 6.25e+04 | 0.12 x 0.05 x 0.06 | 6 | 0.0100 | 1.000 | 294 | 18.1% | +28.1% |
| stiff | 6.25e+05 | 0.12 x 0.05 x 0.06 | 6 | 0.0010 | 3.162 | 294 | 17.2% | +130.8% |
| thick | 6.25e+04 | 0.12 x 0.08 x 0.09 | 9 | 0.0100 | 1.000 | 924 | 3.0% | +57.8% |

### Table 2 - mu(I) vs reference, per arm

| I_target | mu[baseline] | err[baseline] | mu[stiff] | err[stiff] | mu[thick] | err[thick] |
|---|---|---|---|---|---|---|
| 0.001 | 0.3295 | -8.9% | 0.3126 | -13.6% | 0.2778 | -23.2% |
| 0.003 | 0.3289 | -9.9% | 0.3309 | -9.3% | 0.2430 | -33.4% |
| 0.01 | 0.3850 | +2.2% | 0.2957 | -21.4% | 0.2773 | -26.0% |
| 0.03 | 0.4074 | -0.5% | 0.3474 | -10.9% | 0.1980 | -50.5% |
| 0.1 | 0.4302 | -13.2% | 0.3727 | -17.7% | 0.3868 | -17.4% |
| 0.3 | 0.4916 | -19.5% | 0.4019 | -27.0% | 0.5686 | -0.7% |

### Table 3 - phi_bed_steady(I) vs reference, per arm

| I_target | phi[baseline] | err[baseline] | phi[stiff] | err[stiff] | phi[thick] | err[thick] | ref_phi |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.5356 | -7.7% | 0.5183 | -10.7% | 0.5769 | -0.6% | 0.5802 |
| 0.003 | 0.5342 | -8.0% | 0.5151 | -11.3% | 0.5765 | -0.7% | 0.5806 |
| 0.01 | 0.5287 | -9.2% | 0.5144 | -11.6% | 0.5765 | -0.9% | 0.5821 |
| 0.03 | 0.5241 | -10.6% | 0.5135 | -12.0% | 0.5763 | -1.5% | 0.5862 |
| 0.1 | 0.5231 | -12.4% | 0.5103 | -13.7% | 0.5750 | -3.1% | 0.5969 |
| 0.3 | 0.5197 | -15.0% | 0.5031 | -16.7% | 0.5736 | -5.4% | 0.6114 |

### Table 4 - arm-level aggregates

| arm | phi range | phi slope/decade | mean phi err | worst phi err | mean mu err | worst mu err | mu within 10% | phi within 10% | Z | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.5197-0.5356 | -6.590e-03 (+-5.8e-04) | -10.5% | 15.0% | -8.3% | 19.5% | 4/6 | 3/6 | 3.55 | 286 |
| stiff | 0.5031-0.5183 | -5.766e-03 (+-1.1e-03) | -12.7% | 16.7% | -16.6% | 27.0% | 1/6 | 0/6 | 2.67 | 687 |
| thick | 0.5736-0.5769 | -1.252e-03 (+-3.1e-04) | -2.0% | 5.4% | -25.2% | 50.5% | 1/6 | 6/6 | 4.10 | 433 |

### Table 5 - dilatancy (h_steady/h_settled - 1), per arm

| I_target | dilatancy[baseline] | dilatancy[stiff] | dilatancy[thick] |
|---|---|---|---|
| 0.001 | -2.58% | -0.39% | +0.06% |
| 0.003 | -2.32% | +0.23% | +0.14% |
| 0.01 | -1.31% | +0.36% | +0.12% |
| 0.03 | -0.44% | +0.54% | +0.17% |
| 0.1 | -0.25% | +1.17% | +0.39% |
| 0.3 | +0.40% | +2.63% | +0.64% |

### Table 6 - hypothesis adjudication

| hypothesis | statistic | value | verdict |
|---|---|---|---|
| H1_soft_contact_artefact | mean phi error, stiff minus baseline | -2.21% | worsened |
| H2_thin_bed_artefact | mean phi error, thick minus baseline | +8.43% | supported |

Best arm after the controls: **thick**, residual mean phi error -2.04%.

