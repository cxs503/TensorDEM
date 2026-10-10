# DEM3D mechanics and time-step sensitivity validation

This validation layer is intentionally DEM-only. It does not modify TensorLBM or
TensorFEM and does not change the existing solver's integration scheme.

## Mechanical regression checks

`tests/test_dem3d_mechanics_validation.py` adds checks for:
- force assembly and global force/reaction balance in the undeformed equilibrium;
- exact position/velocity constraints on fixed particles under external loading;
- a convergence-study smoke test that verifies equal physical end time, the expected
  step-count ratio, finite force/energy metrics, and invalid-schedule rejection.

These checks complement the existing 3-D rigid-rotation, irreversible-fracture,
cell-list, checkpoint and energy-audit tests. They are regression checks, not a
substitute for analytical contact-impact validation or experimental calibration.

## Run the time-step study

From the repository root:

```bash
python scripts/validate_dem3d_convergence.py --steps 1000 --factors 1 2 4
```

The coarsest level uses the configured `dt` (or the solver's recommended `dt`).
Each refinement divides `dt` by its integer factor and adjusts the step count so
all levels reach the same physical final time. Outputs are:
- `convergence_3d.csv`: one row per time-step level;
- `convergence_3d.json`: configuration, protocol, end time and interpretation;
- `reaction_history_3d.csv`: initial state plus every step for every refinement,
  including all reaction components and magnitude, broken-bond count, mechanical
  energy, and energy-balance residual.

Metrics include peak absolute vertical indenter reaction, final reaction, broken
bond count, final mechanical energy, maximum absolute energy-balance residual,
and force/damage deltas relative to the finest level. The finest run is a numerical
reference, not an exact solution. In particular, fracture is discontinuous, so
bond counts can vary with time-step size even when reaction histories look similar.

## Engineering interpretation

1. Inspect force history and failure-event timing, not only a single peak value.
2. Refine until the force response and damage pattern are stable for the intended
   quantity of interest; do not infer convergence from a single pair of levels.
3. Record the chosen `dt`, particle spacing, stiffness, damping, boundary conditions,
   and fracture thresholds alongside any result.
4. Treat the energy ledger as a diagnostic. The current semi-implicit Euler update,
   discrete work estimates and extension-based fracture-release estimate mean the
   residual is not an exact conservation proof.
5. Before engineering predictions, validate contact impact against analytical
   limiting cases and calibrate bond/contact parameters against measured material data.
