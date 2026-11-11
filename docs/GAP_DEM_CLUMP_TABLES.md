# GAP-DEM-CLUMP — program-generated tables

Generated from `artifacts/GAP-DEM-CLUMP/results.json` (campaign run 2026-11-11T10:30:26Z, wall 462.8 s, loongarch64, numpy 2.4.6).

## C1 — analytic inertia kernel vs independent references

| check | value | acceptance | verdict |
|---|---|---|---|
| single sphere (exact $2/5 m r^2$) | 0 | <1e-12 | PASS |
| tangent dumbbell (textbook tensor) | 0 | <1e-12 | PASS |
| overlapping pair vs 384³ voxel (inertia) | 6.5383e-05 | <1e-3 | PASS |
| overlapping pair vs 384³ voxel (volume) | 4.3542e-05 | <1e-3 | PASS |
| three-sphere cluster, 3rd order vs 384³ voxel | 3.8675e-05 | <1e-3 | PASS |
| three-sphere cluster, 2nd order vs 384³ voxel | 6.6275e-04 | reference | — |
| triple-intersection volume fraction | 0.0107 | >0 (non-trivial) | PASS |

Third order improves the triple case: **True**; kernel criterion C1 = **PASS**.

## C2 — production clump inertia vs analytic (acceptance 1 %)

| a/b | n members | achieved AR | sphericity | volume [m³] | mass [kg] | I_analytic (ax, t1, t2) [kg·m²] | I_production | rel.err production | rel.err fine voxel (192³) |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 1 | 1.000 | 1.0000 | 7.2382e-09 | 1.8096e-05 | 1.0423e-11, 1.0423e-11, 1.0423e-11 | 1.0423e-11, 1.0423e-11, 1.0423e-11 | **1.5500e-16** | 8.3809e-05 |
| 1.50 | 3 | 1.420 | 0.9733 | 1.0867e-08 | 2.7167e-05 | 1.5957e-11, 2.5194e-11, 2.5194e-11 | 1.5973e-11, 2.5234e-11, 2.5234e-11 | **0.0016** | 2.9769e-04 |
| 2.00 | 4 | 1.973 | 0.9287 | 1.4441e-08 | 3.6104e-05 | 2.0906e-11, 5.2181e-11, 5.2181e-11 | 2.0900e-11, 5.2265e-11, 5.2265e-11 | **0.0016** | 1.3108e-04 |
| 3.00 | 6 | 2.908 | 0.8461 | 2.1822e-08 | 5.4555e-05 | 3.1618e-11, 1.5822e-10, 1.5822e-10 | 3.1761e-11, 1.5871e-10, 1.5871e-10 | **0.0031** | 2.5114e-04 |

Worst production-vs-analytic component error over the scan: **0.0031** (acceptance 1 %) → C2 = **PASS**.

## C3 — repose angle vs aspect ratio (2 seeds, mu_roll = 0.3)

| a/b | seed | repose angle [deg] | peak height [m] | flank radius [m] | n particles | steps | wall [s] |
|---|---|---|---|---|---|---|---|
| 1.00 | 0 | 25.573 | 0.0177 | 0.0465 | 648 | 3800 | 38.6 |
| 1.00 | 1 | 26.064 | 0.0191 | 0.0503 | 648 | 3800 | 39.9 |
| 1.50 | 0 | 27.999 | 0.0162 | 0.0372 | 260 | 3800 | 50.0 |
| 1.50 | 1 | 27.985 | 0.0183 | 0.0343 | 260 | 3800 | 47.8 |
| 2.00 | 0 | 19.314 | 0.0162 | 0.0378 | 243 | 3800 | 49.6 |
| 2.00 | 1 | 15.621 | 0.0156 | 0.0451 | 243 | 3800 | 48.7 |
| 3.00 | 0 | 26.602 | 0.0142 | 0.0316 | 104 | 3800 | 24.1 |
| 3.00 | 1 | 18.364 | 0.0134 | 0.0372 | 104 | 3800 | 26.1 |

| a/b | mean [deg] | std over seeds | Δ vs sphere [deg] |
|---|---|---|---|
| 1.0 | 25.819 | 0.347 | 0 |
| 1.5 | 27.992 | 0.010 | 2.174 |
| 2.0 | 17.467 | 2.611 | -8.351 |
| 3.0 | 22.483 | 5.825 | -3.335 |

Monotone non-decreasing: **False**; max Δ vs sphere: **2.174°** (need ≥ 3°); sphere baseline inside the 22–25° literature band: **False** → C3 = **FAIL**.

## C3 — rolling-resistance mechanism sweep (sphere vs a/b = 3, seed 0)

| mu_roll | sphere [deg] | a/b = 3 [deg] | Δ [deg] |
|---|---|---|---|
| 0.0 | 11.179 | 20.639 | 9.460 |
| 0.6 | 32.052 | 24.884 | -7.169 |

## Criteria summary

| id | criterion | verdict |
|---|---|---|
| C1_analytic_kernel | analytic inertia kernel vs closed forms / fine voxel | **PASS** |
| C2_inertia_vs_analytic_under_1pct | production clump inertia vs analytic < 1 % | **PASS** |
| C3_repose_trend_matches_literature | repose-angle trend vs aspect ratio | **FAIL** |
