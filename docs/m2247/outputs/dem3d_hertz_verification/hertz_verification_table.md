| case | error metric | error | tolerance | verdict |
|---|---|---|---|---|
| `hertz_force_vs_independent_closed_form` | worst relative error vs closed form | 0.000000e+00 | 1.0e-12 | PASS |
| `reduced_modulus_mapping` | relative error of E* mapping | 0.000000e+00 | 1.0e-12 | PASS |
| `hertz_exponent_three_halves` | absolute error of log-log slope vs 1.5 | 2.220446e-16 | 1.0e-10 | PASS |
| `contact_potential_matches_force_integral` | relative error, potential vs integral | 3.122762e-11 | 1.0e-10 | PASS |
| `secant_stiffness_identity` | relative error of secant stiffness identity | 0.000000e+00 | 1.0e-12 | PASS |
| `solver_static_hertz_indentation` | worst relative error, solver vs closed form | 1.500057e-12 | 1.0e-09 | PASS |
| `solver_dynamic_hertz_impact_force` | relative error of peak force vs closed form | 3.882674e-06 | 3.0e-02 | PASS |
| `legacy_linear_penalty_vs_hertz` | max relative error of the legacy linear law vs Hertz | 9.997842e-01 | 3.0e-02 | PASS |

### Force-law sweep (equal spheres, E = 1 GPa, nu = 0.3, R = 25 mm)

| overlap (m) | implemented F (N) | independent closed form F (N) | relative error |
|---|---|---|---|
| 1e-06 | 8.190725e-02 | 8.190725e-02 | 0.00e+00 |
| 1e-05 | 2.590135e+00 | 2.590135e+00 | 0.00e+00 |
| 1e-04 | 8.190725e+01 | 8.190725e+01 | 0.00e+00 |
| 1e-03 | 2.590135e+03 | 2.590135e+03 | 0.00e+00 |
| 5e-03 | 2.895859e+04 | 2.895859e+04 | 0.00e+00 |
| 1e-02 | 8.190725e+04 | 8.190725e+04 | 0.00e+00 |

### Solver force path under a prescribed rigid indenter

| overlap (m) | solver reaction (N) | closed form F (N) | relative error |
|---|---|---|---|
| 1e-05 | 7.326007e+00 | 7.326007e+00 | 1.50e-12 |
| 1e-04 | 2.316687e+02 | 2.316687e+02 | 1.65e-13 |
| 1e-03 | 7.326007e+03 | 7.326007e+03 | 1.37e-15 |
| 5e-03 | 8.190725e+04 | 8.190725e+04 | 1.24e-15 |
| 1e-02 | 2.316687e+05 | 2.316687e+05 | 1.38e-15 |

Simultaneous indenter contacts during the sweep: 1

### Elastic impact on a rigid Hertzian platen

| quantity | DEM3D | closed form | relative error |
|---|---|---|---|
| peak force (N) | 2.040902e+02 | 2.040909e+02 | 3.883e-06 |
| impulse (N s) | 6.001751e-02 | 6.001751e-02 | 8.699e-09 |

dt = 1.115e-06 s (0.4 x recommended 2.788e-06 s), 1346 steps, impact speed 0.5 m/s.

### Legacy linear penalty vs Hertz (verification debt, k = 5000 N/m)

| overlap (m) | Hertz F (N) | linear F (N) | relative error | Hertz secant k (N/m) |
|---|---|---|---|---|
| 1e-06 | 2.316687e-01 | 5.000000e-03 | 0.978 | 2.316687e+05 |
| 1e-05 | 7.326007e+00 | 5.000000e-02 | 0.993 | 7.326007e+05 |
| 1e-04 | 2.316687e+02 | 5.000000e-01 | 0.998 | 2.316687e+06 |
| 1e-03 | 7.326007e+03 | 5.000000e+00 | 0.999 | 7.326007e+06 |
| 5e-03 | 8.190725e+04 | 2.500000e+01 | 1.000 | 1.638145e+07 |
| 1e-02 | 2.316687e+05 | 5.000000e+01 | 1.000 | 2.316687e+07 |

Overlap at which 5000 N/m reproduces the Hertz force exactly: 4.658e-10 m.

| ice modulus | k_eq at 1e-4 m | k_eq at 1e-3 m | k_eq at 1e-2 m |
|---|---|---|---|
| E_1e+09 | 2.3167e+06 | 7.3260e+06 | 2.3167e+07 |
| E_5e+09 | 1.1583e+07 | 3.6630e+07 | 1.1583e+08 |
| E_1e+10 | 2.3167e+07 | 7.3260e+07 | 2.3167e+08 |
