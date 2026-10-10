# TensorDEM 3-D Mechanics Benchmark Protocol

Protocol: `tensordem-dem3d-mechanics-v1`

## Purpose and scope

This protocol is the first mechanics-level layer below the named indenter regression
suite. It checks the laws actually implemented by `IceDEM3D`, using analytic
expectations and invariants. It is deliberately narrower than an experimental
validation campaign.

The current 3-D implementation uses:

- linear central-force bonds, with force magnitude `bond_stiffness * extension`;
- a linear penalty normal contact law, with force magnitude
  `max(contact_stiffness * overlap - contact_damping * normal_velocity, 0)`;
- irreversible tensile/shear bond failure;
- a prescribed spherical indenter and explicit time integration.

It does **not** currently implement Hertz-Mindlin contact, tangential friction
history, particle rotation, rolling resistance, or calibrated macroscopic ice
constitutive response. The benchmark labels must not imply those capabilities.

## Run

From the repository root:

```bash
python scripts/run_dem3d_mechanics_benchmarks.py --output results-dem3d-mechanics
python -m unittest tests.test_dem3d_mechanics_benchmarks -v
```

The runner returns a nonzero exit code if any invariant fails and writes:

- `mechanics_benchmark_report.json`: machine-readable protocol, verdict and observations;
- `mechanics_benchmark_report.csv`: one row per check for quick review.

## Cases and acceptance criteria

| Case | Analytic / invariant expectation | Current acceptance |
|---|---|---|
| `bond_axial_spring` | Isolated force equals (k_b\,\Delta l) | Absolute and relative error each (le 10^{-10}) |
| `linear_contact_penalty` | Zero-velocity normal force equals (k_c\,\delta) | Absolute and relative error each (le 10^{-10}) |
| `force_balance` | Sum of particle forces plus indenter reaction is zero, with active tool contact | Residual (le 10^{-9}) N |
| `single_bond_shear_failure` | Affine simple shear triggers the objective shear criterion while tensile threshold is held high | At least one broken bond, shear mode code 2 |
| `irreversible_tensile_failure` | Above-threshold bond fails and does not heal on unloading | At least one failure, zero healing |
| `rigid_rotation_objectivity` | Rigid rotation creates no internal force or damage | Maximum force (le 10^{-7}) N; zero broken bonds |
| `timestep_guard` | An explicit step above the configured conservative bound is rejected | Rejection required |

The tolerances above are software regression tolerances for deterministic CPU tests,
not material-error tolerances or evidence of physical accuracy.

## What PASS does and does not mean

A PASS establishes that these implemented laws satisfy their selected analytic
checks for the exercised states. It does not establish experimental validity,
material calibration, convergence, general numerical stability, or engineering
qualification. The contact test is specifically a **linear penalty contact** test;
it must not be reported as a Hertz contact test.

## Next layers

### Layer 2 — bonded specimen response

Add specimen generators and loading fixtures with controlled geometry and
boundary conditions. The planned cases are:

1. uniaxial compression (UCS): stress-strain curve, tangent modulus, peak stress,
   damage onset and failure localization;
2. Brazilian splitting: force-displacement curve, peak load and tensile crack path;
3. three-point bending with a notch: load-deflection curve, crack trajectory and
   fracture-energy estimate;


Each material-level case needs a published or laboratory reference dataset and a
declared calibration/validation split. A numerically repeatable result alone is
not enough.

### Layer 3 — ice mechanics

Use plate/beam bending and local indentation/crushing experiments to calibrate
bond-network parameters. Document temperature, strain rate, salinity, specimen
geometry and loading rate when those data are available. Compare force history,
failure pattern and energy dissipation, not just peak force.

### Layer 4 — ship–ice engineering cases

Only after the material-level layer is validated should TensorDEM add wedge-bow
or ship-bow traversal, ice-piece contact/rearrangement, and continuous-navigation
resistance. Report peak and mean resistance, impulse, damage distribution,
fragment-size statistics, runtime, time-step sensitivity and particle-resolution
sensitivity. Validate against an independently held-out model test or trusted
published dataset.

## Reproducibility record

For every published benchmark run, preserve the Git commit, Python/PyTorch versions,
device, complete configuration, case manifest, output report, and raw force/damage
history. CPU deterministic signatures are useful regression guards but are not a
substitute for physical validation.
