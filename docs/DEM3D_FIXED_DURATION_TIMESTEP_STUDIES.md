# Fixed-duration DEM3D time-step studies

The three-point-bending numerical fixture supports explicit time-step refinement while
holding the simulated duration and peak load fixed. This is a prerequisite for a
time-step sensitivity study; it is not itself proof of convergence.

The solver computes a conservative stability bound from the model configuration.
Any explicit `--dt-s` must be positive and no larger than that bound. When
`--duration-s` is supplied, the duration must be an integer multiple of the effective
time step, and the number of integration steps is derived from that ratio.

Example:

```bash
python scripts/benchmark_dem3d_three_point_bending.py \
  --output outputs/bending-dt-coarse \
  --dt-s 8e-5 --duration-s 0.00192 --peak-load-N 0.02
python scripts/benchmark_dem3d_three_point_bending.py \
  --output outputs/bending-dt-fine \
  --dt-s 4e-5 --duration-s 0.00192 --peak-load-N 0.02
```

Use the actual `dt_s` reported by the first run as the reference value; the numbers
above are illustrative and may exceed the model's stability bound. To halve the
step while holding duration fixed, also double the step count. The report now records
`dt_s` and `simulated_duration_s`.

Compare identical observables at the same physical time and load, including
midspan deflection, support reaction, broken-bond count and energy history. Report
relative differences between refinements, but do not label the method converged
until predeclared tolerances are satisfied across more than one refinement pair.
The present fixture's support and loading idealizations remain unvalidated.
