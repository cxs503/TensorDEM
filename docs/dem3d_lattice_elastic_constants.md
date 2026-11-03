# DEM3D lattice elastic constants: analytic verification and non-affine audit

## Why this audit exists

`tensordem/dem3d.py` states that its lattice is a simple-cubic network with face
and body diagonals inside the initial cutoff and that its *directional stiffness
must be calibrated before engineering use*. This page replaces that advice with
closed-form elastic constants, a machine-precision verification against the
solver's own bond network, and a quantitative audit of the non-affine part of the
response. It is linear-elasticity verification of the intact bond network: no
fracture, no contact, no dynamics and no ice-material claim.

Run:

```bash
python scripts/audit_dem3d_lattice_elastic.py --output results-dem3d-lattice-elastic
python -m unittest tests.test_dem3d_lattice_elastic -v
```

The script writes to `results-dem3d-lattice-elastic/` (a scratch results directory, git
ignored by convention). The archived copies used for this page, including the SHA-256
signature over the report body, are in [`docs/dem3d-lattice-elastic/`](dem3d-lattice-elastic/):
`lattice_elastic_audit.json`, `lattice_elastic_table.csv` and the three figures.

## 1. Bond topology

The interior coordination number of the solver is **Z = 26**: every
offset in the 26-neighbour box carries a bond. The builder emits
**13 half-space representatives** (one per undirected pair), so the
lattice sum is taken over those representatives and each counts twice. The audit
verifies the set identity directly against the solver's bond list
(`topology_check.direction_sets_identical = True`).

| class | half-space offsets | share of the C11 lattice sum |
|---|---|---|
| axial | 3 | 23.08% |
| face diagonal | 6 | 46.15% |
| body diagonal | 4 | 30.77% |

## 2. Closed-form (affine / Born) elastic constants

For a central-force network under affine strain,

```
U/V = 0.5 * (k/a^3) * sum_S |r|^2 (n.E.n)^2
```

with `k` the bond stiffness, `a = 2 * radius` the spacing and `S` the 13
half-space offsets. Hence

| constant | closed form | value for k = 2000 N/m, a = 0.050 m |
|---|---|---|
| C11 | 13 k / (3 a) = 4.333333 k/a | 173333.3 Pa |
| C12 | 7 k / (3 a) = 2.333333 k/a | 93333.3 Pa |
| C44 | 7 k / (3 a) = 2.333333 k/a | 93333.3 Pa |
| E[100] | 2.7 k / a | 108000.0 Pa |
| nu[100] | 0.35 (independent of k and a) | 0.3500 |
| Zener A = 2 C44 / (C11 - C12) | 7/3 | 2.3333 |

Cubic symmetry residuals and the Cauchy relation `C12 = C44` hold to machine
precision (`cubic_symmetry_relative_deviation` max
0.0e+00,
`cauchy_relation_relative_deviation` 0.0e+00).
The scalar `bond_stiffness` therefore sets a modulus, but the Poisson ratio is
locked by topology and the shear modulus is tied to C12.

## 3. Verification against the solver's own bond network

`affine_patch` deforms a finite specimen affinely and measures the constants from
the solver's bond list and positions, using only particles with a complete
26-bond shell. It must reproduce the closed form exactly:

| constant | solver (affine patch, n = 8 core particles) | closed form | relative error |
|---|---|---|---|
| C11 | 173333.333333 Pa | 173333.333333 Pa | 0.00e+00 |
| C12 | 93333.333333 Pa | 93333.333333 Pa | 4.44e-16 |
| C44 | 93333.333333 Pa | 93333.333333 Pa | 2.22e-16 |

The harmonic operator used for the relaxed solve is cross-checked against the
solver's force law with a pseudo-random displacement field: energy relative
difference 1.4e-06, force relative difference
1.5e-05 (`pass = True`).

## 4. Non-affine (relaxed) response

Freeing the interior degrees of freedom while the driven faces follow the affine
field changes the answer substantially. Tension: x faces driven, lateral faces
free. Shear: y faces driven, other faces free. The constant uses the same core
estimator as the affine patch, and `ratio_to_affine_same_specimen` compares it
with the affine value on the *same* specimen, which is the correct variational
bound.

| n (edge particles) | N | relaxed E [Pa] | relaxed nu | relaxed C44 [Pa] | E / affine(same specimen) | C44 / affine(same specimen) | free-DOF residual [N] |
|---|---|---|---|---|---|---|---|
| 4 | 64 | 165869.5 | 0.1758 | 69349.1 | 0.9569 | 0.7430 | 6.5e-19 |
| 6 | 216 | 142038.6 | 0.2126 | 57377.1 | 0.8195 | 0.6148 | 2.6e-18 |
| 8 | 512 | 134283.4 | 0.2296 | 53494.4 | 0.7747 | 0.5732 | 4.3e-18 |
| 10 | 1000 | 130529.1 | 0.2394 | 51768.3 | 0.7531 | 0.5547 | 7.8e-18 |
| 12 | 1728 | 128351.2 | 0.2458 | 50871.7 | 0.7405 | 0.5451 | 1.0e-17 |

Extrapolating `value = intercept + slope / n`:

| quantity | 1/n extrapolation | fit residual |
|---|---|---|
| tension modulus | 107328.9 Pa | 3089.7 Pa |
| shear stiffness | 40304.4 Pa | 1625.9 Pa |
| Poisson ratio | 0.2817 | 0.0011 |

Interpretation:

- The **tension modulus is essentially affine**: the extrapolated relaxed value is
  107329 Pa against the closed-form E[100] = 108000 Pa,
  a relative difference of 0.62%
  (< 3 % criterion: True).
- The **shear stiffness is strongly non-affine**: at the largest specimen the relaxed
  value is 0.545 of the affine C44, and the
  extrapolation gives 40304 Pa
  (0.432 of affine).
- Consequently the affine Zener anisotropy A = 2.33 is
  mostly an artifact of the affine constraint: the relaxed shear stiffness is within
  3.74% of the isotropic value implied by the relaxed E and nu.
  Report shear stiffness from a relaxed calculation, not from the affine formula.

## 5. Calibration rule and what a scalar stiffness cannot do

Calibrating E[100] to a target modulus inverts the closed form:

```
k = E_target * a / 2.7 = E_target * radius / 1.35
```

For E = 9.0 GPa and r = 0.025 m:
k = 1.6667e+08 N/m, recovered modulus relative error
2.2e-16. But the same k forces nu = 0.35 (the
coefficient is scale free, so nu does not depend on k or a) and gives an *affine*
shear modulus of 7.778 GPa, i.e.
124.7% above the isotropic value
3.462 GPa required by a target
(E, nu = 0.3). Section 4 shows the relaxed shear sits near the
isotropic value, so the practical limitation is the locked Poisson ratio and the
absence of tangential/rotational stiffness, not a spurious shear anisotropy.

## 6. Caveats and scope

- **Correction (see [the periodic k = 0 audit](dem3d_born_huang_internal_strain.md)):** the
  relaxed/affine ratios of section 4 are a *free-surface specimen* response under mixed
  boundary conditions, not a bulk non-affine correction. For the periodic lattice the k = 0
  internal-strain tensor vanishes identically (one atom per primitive cell, inversion
  symmetry) and the relaxed constants equal the affine ones to 2.2e-16, so the Zener ratio
  A = 7/3 is exact in the bulk. The 1/n extrapolation of section 4 is a specimen quantity and
  2 of its 5 points sit on a singular linear system.
- Linear small-strain elasticity of the intact lattice; no fracture, contacts, drag
  or free-surface reconstruction.
- Relaxed values come from free-surface specimens of 64 to
  1728 particles with a dense harmonic solve
  (limit 4000 particles). The 1/n extrapolation is an estimate with the fit residual
  quoted above, not a converged bulk modulus. The rigorous alternative is a
  periodic k = 0 Born-Huang internal-strain calculation; **now implemented in
  [dem3d_born_huang_internal_strain.md](dem3d_born_huang_internal_strain.md)**, where the
  internal-strain tensor is shown to vanish identically for this lattice.
- The Poisson ratio reported for the relaxed specimen is an effective finite-specimen
  quantity and is not the same object as the affine nu[100].
- These are lattice-mechanics numbers. They do not calibrate ice, do not validate a
  material model and say nothing about ship-ice resistance.

## 7. Reproduce

```bash
python scripts/audit_dem3d_lattice_elastic.py --output results-dem3d-lattice-elastic
python -m unittest tests.test_dem3d_lattice_elastic
```

Archived artifacts of this run: `docs/dem3d-lattice-elastic/`. Figures: `fig1_lattice_elastic_constants.png` (affine vs relaxed, size trend),
`fig2_bond_direction_classes.png` (which bond directions carry C11),
`fig3_stiffness_calibration.png` (shear consequence and the k = E a / 2.7 rule).
