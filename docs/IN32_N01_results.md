# IN32-N01 — results tables (generated)

Source: `artifacts/in32_n01_polygon_floe_field.json` · git `d12d016` · python 3.11.6 · numpy 2.4.6 · arch loongarch64 · wall 467.8 s · 6/6 criteria PASS (all_pass = yes)

## (a) No overlap — 4 FSD families x 12 seeds x 5 concentrations

| field | value |
|---|---|
| fields generated | 240 |
| violations (gap < -1e-9 m) | 0 |
| minimum SAT signed gap [m] | 0.0842 |
| max area-solver relative error | 0.0017 |
| max |cover - 1| (tessellation identity) | 3.71e-07 |
| wall time [s] | 227.8335 |
| criterion met | yes |

Per concentration (minimum signed gap over 48 fields each):

| phi | min gap [m] |
|---|---|
| 0.60 | 0.4532 |
| 0.70 | 0.3040 |
| 0.80 | 0.1838 |
| 0.85 | 0.1318 |
| 0.90 | 0.0842 |

Per FSD family:

| family | fields | min gap [m] | max |coverage - phi| | min polygon vertices |
|---|---|---|---|---|
| power_law_2.5 | 60 | 0.2519 | 2.22e-16 | 4 |
| log_normal_0.5 | 60 | 0.1246 | 2.22e-16 | 3 |
| bimodal | 60 | 0.0842 | 2.22e-16 | 3 |
| delta | 60 | 0.3387 | 3.34e-07 | 4 |

## (b) Area moments vs the analytic FSD moments

| family | max m1 rel. error | max m2 rel. error | sampling floor m1 | sampling floor m2 | N |
|---|---|---|---|---|---|
| power_law_2.5 | 1.28e-06 | 5.24e-07 | 1.45e-07 | 5.24e-07 | 1200 |
| log_normal_0.5 | 1.74e-04 | 0.0015 | 1.74e-04 | 0.0015 | 1200 |
| bimodal | 4.12e-05 | 2.01e-04 | 4.12e-05 | 2.01e-04 | 1200 |
| delta | 1.56e-09 | 2.84e-16 | 0 | 0 | 1200 |

Worst relative error over all families: **0.0015** (criterion < 1e-2): **yes**.

The *sampling floor* is the error of the stratified FSD sample before any tessellation: it shows that the residual moment error is the finite-sample floor of the quantile sampling, not a geometry error.

## (c) Contact-set completeness at N = 2000

| quantity | value |
|---|---|
| pairs from the KD-tree search | 6061 |
| pairs from the O(N^2) enumeration | 6061 |
| false positives | 0 |
| false negatives | 0 |
| max |gap difference| between the two paths [m] | 3.82e-14 |
| cutoff [m] | 1.2500 |
| criterion met | yes |

## (d) Force/torque bookkeeping of a static field

| quantity | value |
|---|---|
| floes | 600 |
| free-field contacts | 0 |
| free-field centroid drift [m] | 0 |
| loaded contacts | 1824 |
| |sum F| / max|F| | 2.26e-15 |
| pair torque identity tau_i+tau_j = d x f (max rel.) | 2.39e-16 |
| |tau_i - r_i x f| (independent recomputation) | 0 |
| mean lever arm [m] | 8.0918 |
| max lever arm [m] | 31.1050 |
| max |contact torque| [N m] | 2.72e+05 |
| criterion met | yes |

## (e) Monodisperse limit vs the circumscribed-disk field

Alignment: equal coordination Z=6.0.

| phi | seed | IoU (equal Z) | IoU (tau = 0.5 r_eq) | tau polygon [m] | tau disc [m] | first-shell pairs | min gap [m] |
|---|---|---|---|---|---|---|---|
| 0.6000 | 0 | 0.9884 | 0.1772 | 3.8907 | 3.1992 | 2400 | 1.8729 |
| 0.6000 | 1 | 0.9942 | 0.1660 | 3.9135 | 3.1344 | 2400 | 2.0105 |
| 0.6000 | 2 | 0.9917 | 0.1783 | 3.7915 | 3.1853 | 2400 | 2.0493 |
| 0.7000 | 0 | 0.9859 | 0.9848 | 2.6317 | 2.0024 | 2400 | 1.2565 |
| 0.7000 | 1 | 0.9934 | 0.9888 | 2.6307 | 1.9425 | 2400 | 1.3489 |
| 0.7000 | 2 | 0.9909 | 0.9856 | 2.5720 | 1.9302 | 2400 | 1.3749 |

Minimum IoU: **0.9859** (criterion >= 0.98): **yes**. The absolute-threshold IoU column documents the shape offset between a polygon and its circumscribed disk.

## (f) Generation budget at N = 4000

| quantity | value |
|---|---|
| run times [s] | 29.90, 31.76 |
| best [s] | 29.8972 |
| median [s] | 30.8300 |
| criterion (< 30 s) | yes |
| tessellation share [s] | 31.4499 |
| assembly share [s] | 0.2997 |
| homotopy/Newton iterations | 15 |
| neighbour planes needed | 128 |
| final area error | 1.78e-06 |
| min signed gap [m] | 0.1440 |

## (g) Test suite

`PYTHONPATH=src python -m pytest tests/test_polygon_floe_field.py -q` → 52 collected cases from 33 test functions, 20 of them negative controls, covering tessellation validity, the prescribed-area solve, SAT geometry, torques, KD-tree vs brute force, field assembly, determinism and input validation.

