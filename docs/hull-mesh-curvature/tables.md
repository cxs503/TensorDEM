# Generated tables and figures

All numbers below are written by `scripts/audit_hull_mesh_curvature_sagitta.py`; nothing is transcribed by hand.

## Table 1 - mid-chord relative force error (fraction of the true penalty load)

Negative = systematic underload. `drop` = contact lost entirely.

| delta/rp | N=4 | N=8 | N=16 | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---|---|---|---|---|---|---|---|
| 0.002 (R/rp=2) | drop | drop | drop | drop | drop | -30.12% | -7.53% | -1.88% |
| 0.002 (R/rp=5) | drop | drop | drop | drop | drop | -75.30% | -18.82% | -4.71% |
| 0.002 (R/rp=10) | drop | drop | drop | drop | drop | drop | -37.65% | -9.41% |
| 0.002 (R/rp=20) | drop | drop | drop | drop | drop | drop | -75.30% | -18.82% |
| 0.005 (R/rp=2) | drop | drop | drop | drop | -48.18% | -12.05% | -3.01% | -0.75% |
| 0.005 (R/rp=5) | drop | drop | drop | drop | drop | -30.12% | -7.53% | -1.88% |
| 0.005 (R/rp=10) | drop | drop | drop | drop | drop | -60.24% | -15.06% | -3.76% |
| 0.005 (R/rp=20) | drop | drop | drop | drop | drop | drop | -30.12% | -7.53% |
| 0.01 (R/rp=2) | drop | drop | drop | -96.31% | -24.09% | -6.02% | -1.51% | -0.38% |
| 0.01 (R/rp=5) | drop | drop | drop | drop | -60.23% | -15.06% | -3.76% | -0.94% |
| 0.01 (R/rp=10) | drop | drop | drop | drop | drop | -30.12% | -7.53% | -1.88% |
| 0.01 (R/rp=20) | drop | drop | drop | drop | drop | -60.24% | -15.06% | -3.76% |
| 0.02 (R/rp=2) | drop | drop | drop | -48.15% | -12.05% | -3.01% | -0.75% | -0.19% |
| 0.02 (R/rp=5) | drop | drop | drop | drop | -30.11% | -7.53% | -1.88% | -0.47% |
| 0.02 (R/rp=10) | drop | drop | drop | drop | -60.23% | -15.06% | -3.76% | -0.94% |
| 0.02 (R/rp=20) | drop | drop | drop | drop | drop | -30.12% | -7.53% | -1.88% |
| 0.05 (R/rp=2) | drop | drop | -76.86% | -19.26% | -4.82% | -1.20% | -0.30% | -0.08% |
| 0.05 (R/rp=5) | drop | drop | drop | -48.15% | -12.05% | -3.01% | -0.75% | -0.19% |
| 0.05 (R/rp=10) | drop | drop | drop | -96.31% | -24.09% | -6.02% | -1.51% | -0.38% |
| 0.05 (R/rp=20) | drop | drop | drop | drop | -48.18% | -12.05% | -3.01% | -0.75% |
| 0.1 (R/rp=2) | drop | drop | -38.43% | -9.63% | -2.41% | -0.60% | -0.15% | -0.04% |
| 0.1 (R/rp=5) | drop | drop | -96.07% | -24.08% | -6.02% | -1.51% | -0.38% | -0.09% |
| 0.1 (R/rp=10) | drop | drop | drop | -48.15% | -12.05% | -3.01% | -0.75% | -0.19% |
| 0.1 (R/rp=20) | drop | drop | drop | -96.31% | -24.09% | -6.02% | -1.51% | -0.38% |
| 0.2 (R/rp=2) | drop | -76.12% | -19.21% | -4.82% | -1.20% | -0.30% | -0.08% | -0.02% |
| 0.2 (R/rp=5) | drop | drop | -48.04% | -12.04% | -3.01% | -0.75% | -0.19% | -0.05% |
| 0.2 (R/rp=10) | drop | drop | -96.07% | -24.08% | -6.02% | -1.51% | -0.38% | -0.09% |
| 0.2 (R/rp=20) | drop | drop | drop | -48.15% | -12.05% | -3.01% | -0.75% | -0.19% |

## Table 2 - dropout threshold vs closed-form sagitta s = R (1 - cos(pi/N))

| R/rp | facets | measured dropout delta [m] | sagitta s [m] | relative difference |
|---|---|---|---|---|
| 2 | 8 | 1.522409350e-01 | 1.522409350e-01 | 1.28e-15 |
| 2 | 16 | 3.842943919e-02 | 3.842943919e-02 | 5.78e-15 |
| 2 | 32 | 9.630546656e-03 | 9.630546656e-03 | 2.31e-14 |
| 2 | 64 | 2.409087590e-03 | 2.409087590e-03 | 9.22e-14 |
| 5 | 8 | 3.806023374e-01 | 3.806023374e-01 | 7.29e-16 |
| 5 | 16 | 9.607359798e-02 | 9.607359798e-02 | 4.62e-15 |
| 5 | 32 | 2.407636664e-02 | 2.407636664e-02 | 4.61e-14 |
| 5 | 64 | 6.022718974e-03 | 6.022718974e-03 | 3.69e-14 |
| 10 | 8 | 7.612046749e-01 | 7.612046749e-01 | 1.75e-15 |
| 10 | 16 | 1.921471960e-01 | 1.921471960e-01 | 4.62e-15 |
| 10 | 32 | 4.815273328e-02 | 4.815273328e-02 | 9.37e-15 |
| 10 | 64 | 1.204543795e-02 | 1.204543795e-02 | 3.69e-14 |
| 20 | 8 | 1.522409350e+00 | 1.522409350e+00 | 1.75e-15 |
| 20 | 16 | 3.842943919e-01 | 3.842943919e-01 | 4.62e-15 |
| 20 | 32 | 9.630546656e-02 | 9.630546656e-02 | 9.37e-15 |
| 20 | 64 | 2.409087590e-02 | 2.409087590e-02 | 3.69e-14 |

## Table 3 - facets required for a 3% force error

| R/rp | delta/rp | closed-form facets | brute-force facets | error at N | error at N-1 | 3% met |
|---|---|---|---|---|---|---|
| 2 | 0.002 | 406 | 406 | -2.994% | -3.01% | yes |
| 2 | 0.005 | 257 | 257 | -2.989% | -3.01% | yes |
| 2 | 0.01 | 182 | 182 | -2.980% | -3.01% | yes |
| 2 | 0.02 | 129 | 129 | -2.965% | -3.01% | yes |
| 2 | 0.05 | 82 | 82 | -2.935% | -3.01% | yes |
| 2 | 0.1 | 58 | 58 | -2.933% | -3.04% | yes |
| 2 | 0.2 | 41 | 41 | -2.934% | -3.08% | yes |
| 5 | 0.002 | 642 | 642 | -2.993% | -3.00% | yes |
| 5 | 0.005 | 406 | 406 | -2.994% | -3.01% | yes |
| 5 | 0.01 | 287 | 287 | -2.996% | -3.02% | yes |
| 5 | 0.02 | 203 | 203 | -2.994% | -3.02% | yes |
| 5 | 0.05 | 129 | 129 | -2.965% | -3.01% | yes |
| 5 | 0.1 | 91 | 91 | -2.979% | -3.05% | yes |
| 5 | 0.2 | 65 | 65 | -2.919% | -3.01% | yes |
| 10 | 0.002 | 907 | 907 | -2.999% | -3.01% | yes |
| 10 | 0.005 | 574 | 574 | -2.996% | -3.01% | yes |
| 10 | 0.01 | 406 | 406 | -2.994% | -3.01% | yes |
| 10 | 0.02 | 287 | 287 | -2.996% | -3.02% | yes |
| 10 | 0.05 | 182 | 182 | -2.980% | -3.01% | yes |
| 10 | 0.1 | 129 | 129 | -2.965% | -3.01% | yes |
| 10 | 0.2 | 91 | 91 | -2.979% | -3.05% | yes |
| 20 | 0.002 | 1283 | 1283 | -2.998% | -3.00% | yes |
| 20 | 0.005 | 812 | 812 | -2.994% | -3.00% | yes |
| 20 | 0.01 | 574 | 574 | -2.996% | -3.01% | yes |
| 20 | 0.02 | 406 | 406 | -2.994% | -3.01% | yes |
| 20 | 0.05 | 257 | 257 | -2.989% | -3.01% | yes |
| 20 | 0.1 | 182 | 182 | -2.980% | -3.01% | yes |
| 20 | 0.2 | 129 | 129 | -2.965% | -3.01% | yes |

## Table 4 - hull_radius = s correction across one facet

Selected cases; `error over facet` is the RMS-free max/min over five positions.

| R/rp | facets | delta/rp | mid-chord before | mid-chord after | vertex after | max abs after | mean after |
|---|---|---|---|---|---|---|---|
| 10 | 32 | 0.05 | -96.31% | -0.000% | +96.31% | 96.31% | +49.07% |
| 10 | 64 | 0.02 | -60.23% | +0.000% | +60.23% | 60.23% | +30.70% |
| 10 | 128 | 0.01 | -30.12% | +0.000% | +30.12% | 30.12% | +15.36% |
| 20 | 128 | 0.02 | -30.12% | +0.000% | +30.12% | 30.12% | +15.21% |
| 2 | 16 | 0.05 | -76.86% | -0.000% | +76.86% | 76.86% | +42.11% |

## Checks

| check | value | pass |
|---|---|---|
| kernel_geometry_matches_closed_form_penetration | max_abs_deviation_m=3.678e-15, tolerance_m=1.000e-09 | PASS |
| 3d_mesh_kernel_equals_2d_polyline_kernel | max_abs_difference_N=3.665e-15, tolerance_N=1.000e-09 | PASS |
| dropout_threshold_equals_sagitta | max_relative_difference=9.217e-14, tolerance=0.000 | PASS |
| worst_mid_chord_underestimate_over_sweep | value=-1.000, note=most negative relative force error found (systematic underload) | PASS |
| hull_radius_s_offset_cancels_mid_chord_bias | max_abs_mid_chord_error_after_offset=6.115e-13, tolerance=1.000e-09 | PASS |
| required_facets_criterion_delivers_3pct | cases=28.000, max_abs_error_at_required=0.030 | PASS |
| analytic_reference_consistency | note=delta_facet = delta_true - s (cases with delta > s), max_abs_deviation_m=4.330e-15, tolerance_m=1.000e-12 | PASS |

## Figures

- **fig1_error_vs_facets.png** - Mid-chord relative force error vs facet count for delta/rp = 0.05; markers = engine, dashed = closed form s/delta.
- **fig2_angular_profile.png** - Error across one facet before/after the constant sagitta offset: a one-sided underload becomes a symmetric ripple.
- **fig3_required_facets.png** - Facets required for a 3% force error versus the design overlap.
