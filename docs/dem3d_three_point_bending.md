# DEM3D three-point-bending benchmark

## Purpose and limits

This fixture exercises load transfer through the 3-D bonded-sphere lattice: two fixed support rows near the lower surface, a central downward load applied to upper-surface particles, support-reaction and deflection histories, finite-state checks, and irreversible fracture bookkeeping.

**This is a numerical integration fixture, not calibrated sea-ice flexural-strength validation.** The current model uses fixed particle supports rather than physical rollers and distributes the force over centerline top particles. Both choices can bias stiffness and stress concentration.

## Run

```bash
python scripts/benchmark_dem3d_three_point_bending.py --output results-dem3d-three-point-bending
python -m unittest discover -s tests -p 'test_dem3d_three_point_bending.py'
```

The outputs are a CSV history and JSON report. A deterministic signature helps detect unintended output changes on the same software/hardware environment; it does not certify physical validity.

## Requirements before quantitative comparison

Add a source-traceable experimental case with beam length, width, thickness, support span, loading rate, density, modulus, flexural strength, and uncertainty. Do not fill missing measurements with assumptions. Then run timestep/resolution convergence, quasi-static-rate checks, boundary sensitivity, and a calibration/hold-out split. The current manifest intentionally contains no invented experimental target values.

Reference context: Ji, Di and Long (2017), DOI [10.1061/(ASCE)EM.1943-7889.0000996](https://doi.org/10.1061/(ASCE)EM.1943-7889.0000996).
