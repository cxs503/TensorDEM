# DEM3D material calibration and fracture-validation roadmap

## Current baseline

TensorDEM has deterministic numerical checks for dynamic UCS-style compression,
loading-rate and particle-resolution sensitivity, and tensile bond-break threshold
screening. These are regression and sensitivity tools, not proof of physical ice
properties.

## Calibration loop added in this milestone

1. Preserve a measured engineering stress-strain curve in CSV with metadata.
2. Define candidate parameter ranges and objective weights before running.
3. Execute a repeatable parameter grid over bond stiffness and tensile breaking
   strain while holding all other inputs fixed.
4. Rank candidates against the full curve and peak metrics, not peak stress alone.
5. Retain full histories, configuration values, deterministic signatures and
   machine-readable checks.
6. Validate the selected parameter set against independent experiments before
   using it in a ship-ice calculation.

## Remaining fracture-validation work

- Add independent tensile/Brazilian splitting and shear specimen protocols.
- Check objectivity under rigid-body rotation and mixed-mode loading.
- Audit bond energy before failure, released energy at failure, and post-failure
  energy bookkeeping under controlled single-bond tests.
- Establish fracture-energy calibration from measured crack-growth or fracture
  work; do not equate breaking strain with fracture energy.
- Quantify uncertainty and sensitivity to particle resolution and loading rate.
- Add ice-layer indentation and moving-hull cases only after material calibration
  has independent validation evidence.

## Progress against the remaining fracture-validation list

- **Brazilian (indirect tensile) splitting protocol**: apparatus and case manifest
  added in `scripts/benchmark_dem3d_brazilian.py` and
  `benchmarks/dem3d/brazilian_cases.json` (`tensordem-dem3d-brazilian-v1`). The
  protocol invariants (reaction symmetry, split-stress definition, contact-arc
  width, repeatability) are green; the comparison of the lattice centre field
  against the Hondros closed form still requires an equilibration campaign of
  order 2e4 steps and is reported as WITHHELD, not passed. See
  `docs/dem3d_platen_contact_mask_audit.md`.
- **Platen boundary defect fixed**: the plane boundaries applied their damping
  term to particles that never intersected the plane, adding a fictitious load
  `n gamma_p |v_p|` to every platen benchmark. Fixed and covered by
  `tests/test_dem3d_platen_contact_mask.py`.
- Still open: shear specimen protocol, objectivity under rigid-body rotation,
  fracture-energy calibration from measured crack growth.

## Acceptance philosophy

Numerical repeatability, numerical stability, parameter fit and physical
validation are separate claims. A green CI run supports software regression only.
It does not establish laboratory agreement, mesh independence, or engineering
predictive capability. Report the experimental conditions and the domain of
validation alongside every calibrated parameter set.
