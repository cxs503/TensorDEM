# 2-D bonded-disk lattice: exact elasticity, the anisotropy nobody had quantified, and an exactly isotropic calibration

Status: analytic derivation + four-route numerical verification + a measured
engineering consequence in the indentation case. Verified for
`tensordem.dem.IceDEM` (eight-neighbour square + face-diagonal central-force
lattice) and for the opt-in per-bond backend
`tensordem.calibrated_dem.CalibratedIceDEM`.

Artifacts: `docs/lattice_elasticity2d/`
(`lattice_elasticity2d_report.json`, `directional_moduli.csv`,
`indentation_impact.csv`, `figures/`).
Reproduce:

```bash
python scripts/validate_lattice_elasticity2d.py --steps 420
python -m pytest tests/test_lattice_elasticity2d.py -q
```

## 1. Why this exists (the verification debt)

`IceDEM` is the two-dimensional engine used by the hull/ridge/wedge benchmarks
in this repository. It bonds particles that are within `sqrt(2)` particle
diameters, i.e. four axis neighbours plus four face diagonals, and gives all of
them **one** `bond_stiffness`. The three-dimensional sibling already carried an
affine-elasticity audit (`docs/dem3d_lattice_elasticity.md`,
`docs/dem3d_lattice_elastic_constants.md`), which converted `bond_stiffness`
into engineering moduli and quantified the specimen-direction bias. The **2-D**
engine had none of that:

* `bond_stiffness` was a free number with no link to a Young's modulus;
* the directional bias was mentioned qualitatively ("simple-cubic lattice
  directional bias", `docs/DEM3D_BENCHMARK_PAPER.md` §8) but never measured for
  the 2-D lattice;
* nothing recorded the stronger structural fact that four-fold bond symmetry is
  **not** sufficient for an isotropic 2-D continuum — it admits a cubic-type
  anisotropy with an independent shear constant (Hexagonal/6-fold symmetry
  would be needed).

## 2. Method

Cauchy–Born (affine) stiffness of a central-force bond network
(Born & Huang 1954):

    C_ijkl = (1/A) * sum_b k_b l_b^2 n_i n_j n_k n_l ,   A = a^2 = (2r)^2.

With the engine's bond set (four axis bonds of length `a`, four face diagonals of
length `a*sqrt(2)`, `k_a` and `k_d` their stiffnesses) the sum collapses to

    C11 = C22 = k_a + k_d ,      C12 = C66 = k_d .

Isotropy for a central-force 2-D network needs the Cauchy relation `C12 = C66`
(automatic here) **and** `C11 - C12 = 2 C66`. The engine default `k_d = k_a`
violates the second identity by a factor of two, so

    k_d = k_a / 2   ==>   C11 = 1.5 k_a, C12 = C66 = 0.5 k_a,
                          E = 4 k_a / 3, nu = 1/3 (any in-plane direction).

Four independent routes are used so a single algebra slip cannot pass:

| route | what it uses | where |
|---|---|---|
| 1 closed form | the algebra above | `stiffness_components`, `affine_stiffness_tensor` |
| 2 Voigt summation | the definition summed over the ideal bond set | `voigt_sum_stiffness` |
| 3 solver topology | `IceDEM.bond_indices` / `reference_bond_vectors`, interior average | `bulk_stiffness_from_sim` |
| 4 solver energy | `IceDEM.mechanical_energy()` under affine strain, free-surface bond deficit added back | `stiffness_from_strain_energy` |

Route 4 is the only one that exercises the engine's own energy routine; the
deficit correction is exact (each missing neighbour slot is half a bond, so its
energy enters with weight 1/2).

Parameters: `bond_stiffness = 2000 N/m`, `radius = 0.025 m`,
`thickness = 0.2 m`, `density = 917 kg/m3`, `contact_stiffness = 2000 N/m`,
patch `21 x 21` for the impact case and `11 x 11` for the modulus routes;
finite-difference strain step `1e-6` (far below `breaking_strain = 0.015`).

## 3. Results

### 3.1 Four routes agree

| route | C11 (N/m) | C12 (N/m) | C66 (N/m) | relative Frobenius error vs closed form |
|---|---:|---:|---:|---:|
| 1 closed form (`k_d = k_a`) | 4000 | 2000 | 2000 | 0 (definition) |
| 2 Voigt summation | 4000 | 2000 | 2000 | 0.0e+00 |
| 3 solver topology (`11 x 11`, 81 interior particles, coordination 8) | 4000 | 2000 | 2000 | 6.5e-16 |
| 4 solver energy (`11 x 11`) | 4000 | 2000 | 2000 | 1.9e-07 |
| 1 closed form (`k_d = k_a/2`) | 3000 | 1000 | 1000 | 0 (definition) |
| 3 solver topology, calibrated backend | 3000 | 1000 | 1000 | 6.3e-16 |
| 4 solver energy, calibrated backend | 3000 | 1000 | 1000 | 1.4e-07 |

Route 4's `1.9e-07` is the central-difference truncation error at step `1e-6`
(`O(h^2)` times the third-order bond term), not a model error; the truncation
scales as `h^2` and is invariant under patch size.

### 3.2 The engine default is 60 % anisotropic

| quantity | engine default `k_d = k_a` | isotropic `k_d = k_a/2` |
|---|---:|---:|
| `C11` | 2.000 `k_a` | 1.500 `k_a` |
| `C12 = C66` | 1.000 `k_a` | 0.500 `k_a` |
| Zener-type ratio `2C66/(C11-C12)` | **2.000** | 1.000 |
| `E(0 deg)` | 1.500 `k_a` | 1.333 `k_a` |
| `E(45 deg)` | 2.400 `k_a` | 1.333 `k_a` |
| `E(45)/E(0)` | **1.600** | 1.000 |
| `nu(0 deg)` / `nu(45 deg)` | 0.500 / 0.200 | 1/3 / 1/3 |
| max coupling term `C16` | 0.250 `k_a` | 0 (1e-16) |
| Pa-equivalent `E(0)` at `thickness = 0.2 m` | 15 000 Pa | 13 333 Pa |

`E(45 deg)/E(0 deg) = 8/5` exactly: a specimen loaded along the face diagonals
is 60 % stiffer than one loaded along the grid axes, and the Poisson ratio is
not a material constant but a direction-dependent quantity running from 0.20 to
0.50. Every 2-D benchmark in `docs/` that reports force, stress or fracture
pattern therefore carries an unstated 0–60 % specimen-orientation dependence.

Figures: `figures/directional_modulus.png` (polar `E(theta)/k_a` and `nu(theta)`
for both calibrations).

### 3.3 Consequence in a real indentation case

Same geometry, same `dt` (`6.8788e-05 s`), same 420 steps (`t_end = 0.02889 s`,
tool penetration `0.0289 m`), engine default versus isotropic calibration:

| observable | engine default | isotropic calibration | change |
|---|---:|---:|---:|
| peak |tool reaction| (N) | 58.15 | 57.75 | **-0.70 %** |
| mean |tool reaction| (N) | 18.16 | 18.07 | -0.49 % |
| final bond strain energy (J) | 2.657e-03 | 1.776e-03 | **-33.2 %** |
| broken bonds | 9 | 11 | +2 |
| final kinetic energy (J) | 0.1113 | 0.1152 | +3.5 % |

The tool reaction is a *weak* discriminator here and that must not be hidden:
at this shallow penetration the series stiffness is dominated by the contact
spring and by the inertia of the short dynamic indent (kinetic energy exceeds
bond energy by ~40x, so the response is not quasi-static). The observables that
do carry the lattice anisotropy are the stored bond energy (−33 %) and the
damage count. With the softer diagonals the specimen stores a third less
elastic energy at the same imposed displacement and breaks two more bonds —
i.e. the isotropic calibration is not a cosmetic relabelling of the default: it
changes damage evolution.

## 4. Error judgement against the 3 % criterion

| key quantity | reference | measured error | below 3 %? |
|---|---|---|---|
| `C11`, `C12`, `C66`, engine default | closed form `k_a + k_d`, `k_d` | 0.0e+00 (Voigt), 6.5e-16 (topology), 1.9e-07 (energy) | yes |
| `C11`, `C12`, `C66`, isotropic calibration | closed form `1.5 k_a`, `0.5 k_a` | 6.3e-16 (topology), 1.4e-07 (energy) | yes |
| `E(45)/E(0)` engine default | 8/5 exactly | < 1e-12 | yes |
| `nu` isotropic calibration | 1/3 exactly | < 1e-9 | yes |
| `C16` isotropic calibration | 0 | 1.2e-16 `k_a` | yes |
| tool reaction, default vs isotropic | *no independent reference exists* — this is a model-choice comparison, not a validation | -0.70 % peak | not applicable (see §5) |
| bond strain energy, default vs isotropic | same, model choice | -33.2 % | not applicable (see §5) |

The elastic constants are verified to machine precision against an exact
analytic reference; the engineering-impact numbers have **no experimental
benchmark** and are reported as a model-choice sensitivity, not as validation.

## 5. Caveats and limitations

1. **Affine/Cauchy–Born only.** The moduli above are the response to a
   *uniform* strain field. They are exact for the lattice's small-strain
   constitutive law but say nothing about non-uniform fields: bending, crack-tip
   stress concentration and Saint-Venant decay need their own benchmark (the
   lattice is rotationally invariant under central forces but not
   Cauchy-continuum-equivalent, so `C12 = C66` and `nu = 1/3` are *kinematic
   consequences*, not fitted materials).
2. **No manufacturing/numerical-error bar is claimed for the impact case.** It
   is a short dynamic indent with large kinetic energy; the sampling of the peak
   is step-size dependent and the case is not converged in resolution. Its role
   here is relative comparison (identical setup, one scalar changed), not an
   absolute prediction.
3. **`k_d = k_a/2` is a calibration choice, not measured ice physics.** The
   repository holds no experimental data that fixes the diagonal/axis stiffness
   ratio. What is proven is that `k_d = k_a/2` makes the 2-D continuum limit
   isotropic, while `k_d = k_a` does not. Changing it slightly (say `0.45` or
   `0.55`) buys back most of the isotropy; users who need a specific Poisson
   ratio should invert `nu = k_d/(k_a + k_d)` instead.
4. **Plane stress only.** The engine is a sheet of unit-cell area `(2r)^2` with
   no out-of-plane coupling; `C` is therefore a traction-per-width (N/m) and the
   Pa-equivalent value shown above divides by `thickness`, which is a
   reporting convention, not an independent 3-D modulus.
5. **No engine source change was needed.** The isotropic variant is reached
   through the existing opt-in per-bond backend `CalibratedIceDEM`, whose
   invariant (`every coefficient <= config.bond_stiffness`) the calibration
   satisfies by construction. The default `IceDEM` path and every existing
   regression artifact are therefore bit-identical to before.
6. **Not covered:** wave speeds / phonon dispersion of the 2-D lattice (the 3-D
   analogue is `docs/dem3d_lattice_wave_propagation.md`), torsional and frictional
   contact, and particle rotations — none of which exist in this engine.

## 6. Answer to "does it pass 3 %?"

Yes for every quantity that has an analytic reference (≤ 2e-07 relative, and
≤ 7e-16 for the topology route). No claim is made for the indentation
comparison, which has no external reference: it is reported as a −0.70 %
(peak reaction) / −33.2 % (bond strain energy) model-choice sensitivity.