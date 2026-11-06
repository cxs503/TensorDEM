# Ice-sheet plate-on-elastic-foundation audit (programmatic output)

- rigidity D = 4.57875e+08 N m, foundation k = 10055.2 N/m^3, characteristic length l = 14.6079 m
- exact point-load identity w(0) = P/(8 sqrt(kD)): relative error 1.49e-16
- point-load reaction / load = 0.999999996 (relative error 4.09e-09)
- max centre-moment difference, closed form vs Hankel route: 3.38e-07
- max |M_t(0)/M_r(0) - 1| (equibiaxial): 0.00e+00
- published Westergaard constant 0.6159 vs extracted 0.615932 at a/l = 1e-3 (absolute error 3.16e-05)
- worst published-vs-exact peak-stress difference over a/l in [0.005, 0.2]: 0.176 %

| h (m) | l (m) | P_allow (kN) | sigma per N (Pa/N) | Westergaard rel. error |
| ---: | ---: | ---: | ---: | ---: |
| 0.3 | 5.921 | 16.9 | 29.598623 | -2.20e-05 |
| 0.5 | 8.686 | 43.1 | 11.606636 | -1.30e-05 |
| 1.0 | 14.608 | 155.1 | 3.224327 | -8.06e-06 |
| 1.5 | 19.800 | 329.6 | 1.516924 | -6.76e-06 |
| 2.0 | 24.568 | 563.9 | 0.886751 | -6.16e-06 |
