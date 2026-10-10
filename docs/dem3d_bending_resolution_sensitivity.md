# DEM3D three-point-bending particle-resolution sensitivity

## Purpose

This campaign complements the load-increment study by quantifying how the
existing bonded-sphere three-point-bending fixture changes as its particle
lattice is refined. It writes JSON, a summary CSV, and a per-step history CSV.

Run:

```bash
python scripts/benchmark_dem3d_bending_resolution.py --output results-dem3d-bending-resolution
python -m unittest tests.test_dem3d_bending_resolution -v
```

## Protocol

The default factors are 1 and 2. For factor (r), the particle radius is
divided by (r), and each lattice interval count is multiplied by (r);
the specimen envelope is therefore approximately preserved. The load ramp
step count is increased as (r^{3/2}) to compensate for the nominal explicit
stable-step scaling of sphere mass, keeping the physical ramp duration roughly
comparable. The actual computed time step and duration are included in output.

The summary records particle/bond counts, time step, duration, peak support
reaction, peak midspan deflection, broken-bond count, finite-history status,
and relative differences against the finest tested resolution. The history
CSV preserves force, deflection, energy, and fracture count at every step.

## Interpretation and limitations

A PASS only means all tested resolutions produced finite, nonzero numerical
load transfer. Sensitivity deltas are diagnostics, not experimental errors.
Bond/contact stiffness is deliberately not recalibrated with resolution, so
this is not a mesh-objective material comparison and must not be called
physical convergence. Fracture counts can jump because the discrete crack path
changes. The fixture also uses fixed-particle supports and a nodal load on the
top centerline; no experimental flexural strength is inferred. Calibration
requires traceable specimen geometry, loading rate, material parameters,
uncertainty, and independent test data.
