# DEM3D dynamic uniaxial compression benchmark

## Purpose

The UCS-style benchmark exercises the DEM3D solver under opposing, prescribed-
velocity top and bottom plane platens. Unlike the affine compression patch, the
particle positions evolve dynamically under the solver's bond, contact, damping,
and platen forces. The x/y edge constraints and bottom-particle fixing are
disabled so the specimen is free to deform laterally.

Run it from the repository root:

```bash
python scripts/benchmark_dem3d_ucs.py --output results-dem3d-ucs
```

The runner executes the configured case twice, compares deterministic history
signatures, and writes:

- `ucs_report.json`: protocol metadata, acceptance checks, peak load/stress,
  strain at peak stress, lateral span changes, bond damage and full history.
- `ucs_history.csv`: per-step platen closure, engineering strain, reaction
  forces, engineering stress, broken bonds, and energy terms.

Engineering stress uses the mean of the absolute top and bottom platen
reactions divided by the initial gross rectangular cross-sectional area.
Engineering axial strain is total platen closure divided by initial specimen
height. Both conventions are stated explicitly in the output.

## Scope and limitations

This is a **software-level dynamic compression regression**, not yet a
laboratory-equivalent or calibrated ice UCS test:

- The specimen is a small simple-cubic bonded-sphere prism, not a cylindrical
  core; lattice direction and finite particle count can bias response.
- Platens are infinite, frictionless planes moving at constant prescribed speed.
  There is no force servo, platen friction, finite platen geometry, or confining
  pressure.
- Peak stress is configuration-dependent and must not be presented as a
  measured or calibrated ice compressive strength.
- A credible material test programme still needs timestep/loading-rate studies,
  particle-resolution convergence, directional/lattice-bias checks, and
  calibration against independent experiments.

The next mechanics step after this regression is to improve specimen geometry
and loading controls, then add independent Brazilian-splitting and three-point
bending fixtures using the same output and acceptance conventions.
