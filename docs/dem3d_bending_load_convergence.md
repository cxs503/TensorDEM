# DEM3D three-point-bending load-increment convergence campaign

## Purpose

This campaign reruns the existing three-point-bending numerical fixture with geometrically refined load increments (default: 30, 60, and 120 increments) while holding the nominal peak nodal load fixed. It reports changes in peak support reaction, center deflection, and broken-bond count.

This is **load-path discretization sensitivity**, not physical-time timestep convergence. The current fixture uses fixed support particles and a distributed centerline load; it is not a calibrated roller-supported beam test.

## Run

```bash
python scripts/benchmark_dem3d_bending_load_convergence.py --output results-dem3d-bending-load-convergence
python -m unittest discover -s tests -p 'test_dem3d_bending_load_convergence.py'
```

For a quick smoke run:

```bash
python scripts/benchmark_dem3d_bending_load_convergence.py --base-steps 8 --levels 2 --peak-load-N 0.01
```

Outputs:
- bending_load_convergence.csv: one row per refinement level.
- bending_load_convergence.json: input parameters, adjacent-level relative changes, and interpretation limits.
- One subdirectory per level, containing the underlying bending history and report.

## Reading the results

Relative change is calculated as abs(fine - coarse) / max(abs(fine), abs(coarse), 1e-30). It is a descriptive measure, not an acceptance threshold. A change in broken-bond count flags a discontinuous fracture response; close reaction/deflection values alone do not establish fracture convergence.

Before quantitative ice-flexural-strength claims, add source-traceable experimental geometry and loading rate, replace fixed supports with validated roller/contact boundaries, establish physical-time timestep and particle-resolution convergence, calibrate to one data subset, and validate against an independent hold-out case. Do not invent missing experimental targets.
