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
- **Shear specimen protocol added** (`src/tensordem/shear_specimen.py`,
  `scripts/benchmark_dem3d_shear_specimen.py`,
  `tests/test_dem3d_shear_specimen.py`, `docs/dem3d_shear_specimen.md`): the
  `shear_breaking_strain` threshold is now calibrated to a macroscopic
  engineering shear strain by closed form (`gamma_c = sqrt(1.5*(sqrt(1+4s^2)-1))`,
  verified against the solver to 4e-12), the criterion is shown to be
  *deviatoric* rather than mode-II specific (uniaxial onset `sqrt(1+3s)-1`,
  verified to 7e-14), and a driven block is taken to its first mode-II bond event
  (local invariant = threshold to 0.09 %; macroscopic onset 0.77x the affine
  value, explained by a measured 1.305 strain concentration). The driven
  specimen's *apparent modulus* is recorded as WITHHELD: the ramp is still 20 %
  from rate convergence. Mode-II fracture energy remains not representable in a
  bond law with no tangential stiffness.
- Objectivity under rigid-body rotation is covered by
  `tests/test_dem3d_fracture_objectivity.py` and by the rotation probe in
  `docs/dem3d_shear_specimen.md`.
- Quasi-static apparent modulus of a driven block: **closed as a rate-free static
  reference** in `docs/dem3d_shear_static_reference.md` (sparse equilibrium of the
  same boundary conditions, `G_eff/C44 = 0.8442` for the default 9x5x3 block,
  reproduced by the solver's dynamics to 2.6 %). The driven-ramp modulus is shown
  to be protocol-limited (cross-section census bias `0.7873`, transient hump), not
  rate-limited; a material shear modulus still needs a cross-section convergence
  study or periodic boundaries.
- **Material shear modulus identified** from the direct-shear geometry by a
  cross-section convergence study: `docs/dem3d_shear_cross_section_convergence.md`.
  Extrapolating the free-surface bias in `1/nx+1/ny` over `5x5x3 … 41x41x3`
  recovers `C44 = 7k/(3a)` to 0.14 % (quadratic) / 1.47 % (linear); an interior
  window at margin 3 sites recovers it to 0.19 %, with a measured edge-relaxation
  length of `0.63 a`. Conditional result: for `h/L >= 0.12` the shear lag fills
  the cross-section (relaxation length grows with `h/L`, up to `3.66 a`), so thick
  specimens report a geometry-dependent modulus (down to `0.790 C44`) and must not
  be read as a material constant.
- Still open: mode-II energy release rate (model change), fracture-energy
  calibration from measured crack growth, independent shear-specimen data.

## Acceptance philosophy

Numerical repeatability, numerical stability, parameter fit and physical
validation are separate claims. A green CI run supports software regression only.
It does not establish laboratory agreement, mesh independence, or engineering
predictive capability. Report the experimental conditions and the domain of
validation alongside every calibrated parameter set.
