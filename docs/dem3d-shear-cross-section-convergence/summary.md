# DEM3D direct-shear cross-section convergence (audit summary)

Analytic reference `C44 = 7k/(3a) = 93333.3 Pa`, `a = 0.0500 m`, shear `gamma = 1e-06` (static solves, no dynamics).

| specimen | N | A_eff/A_nom | G_nom/C44 | G_eff/C44 | G_win/C44 | window margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5x5x3 | 75 | 0.7086 | 0.4869 | 0.6872 | 0.7327 | 1 |
| 9x9x3 | 243 | 0.8325 | 0.7040 | 0.8457 | 0.9751 | 3 |
| 13x13x3 | 507 | 0.8825 | 0.7921 | 0.8976 | 0.9920 | 3 |
| 17x17x3 | 867 | 0.9095 | 0.8398 | 0.9234 | 0.9951 | 3 |
| 21x21x3 | 1323 | 0.9265 | 0.8697 | 0.9388 | 0.9964 | 3 |
| 25x25x3 | 1875 | 0.9381 | 0.8902 | 0.9490 | 0.9972 | 3 |
| 33x33x3 | 3267 | 0.9529 | 0.9165 | 0.9618 | 0.9981 | 3 |
| 41x41x3 | 5043 | 0.9620 | 0.9326 | 0.9695 | 0.9985 | 3 |
| 9x5x3 | 135 | 0.7873 | 0.6646 | 0.8442 | 0.9029 | 1 |
| 17x9x3 | 459 | 0.8814 | 0.8134 | 0.9229 | 0.9955 | 3 |
| 25x13x3 | 975 | 0.9178 | 0.8708 | 0.9488 | 0.9974 | 3 |
| 33x21x3 | 2079 | 0.9433 | 0.9072 | 0.9617 | 0.9981 | 3 |
| 41x25x3 | 3075 | 0.9533 | 0.9242 | 0.9694 | 0.9985 | 3 |

* Square family, quadratic extrapolation in `1/n`: limit `0.9986 C44` (deviation 0.14 %); linear fit `1.0147` (1.47 %).
* Thin-specimen interior window (margin 3 sites): `0.9981 C44` (deficit 0.19 %), exponential decay length `0.63 a` (r2 0.9521).
* Thick specimens, whole layer 33x33: nz=3: 0.962, nz=5: 0.906, nz=7: 0.848, nz=9: 0.790.
* Mid-layer interior window, relaxation length of the deficit (bootstrap, h/L = height over width): nz=3 (h/L=0.06): no positive deficit inside margin 5, nz=5 (h/L=0.12): 1.07 a, nz=7 (h/L=0.18): 2.28 a, nz=9 (h/L=0.24): 3.66 a;
* Combined square+rect quadratic fit: r2 0.942, limit 0.9783 -> the single perimeter-fraction law is *not* adequate across aspect ratios.

- [PASS] square-family extrapolated limit within 3% of C44: 0.99865 (|limit-1| < 0.03)
- [PASS] linear-fit extrapolated limit within 3% of C44: 1.01474 (|limit-1| < 0.03)
- [PASS] thin specimen interior window (margin 3) within 1% of C44: 1.94e-03 (deficit < 0.01)
- [PASS] interior window measure is exactly C44 under affine shear: 2.22e-16 (max |G/C44-1| < 1e-5)
- [PASS] thick specimen modulus is NOT height independent (honest limitation): 0.1716 (reported as specimen property)
- [n/a] single 1/nx+1/ny law across aspect ratios (reported, not required): r2=0.9418, limit=0.9783 (informative only)
