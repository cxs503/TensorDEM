# Periodic k = 0 Born-Huang internal strain of the DEM3D bond lattice

## Why this page exists

`docs/dem3d_lattice_elastic_constants.md` derives the affine (Born) cubic constants of the
`IceDEM3D` bond lattice, measures a *relaxed* response on finite specimens with free faces,
concludes that the relaxed shear stiffness is "strongly non-affine" and that the affine Zener
anisotropy is "mostly an artifact", and closes with an explicit gap:

> The rigorous alternative is a periodic k = 0 Born-Huang internal-strain calculation;
> it is not implemented here.

This page implements that calculation. The result is that for the solver's lattice the
internal-strain tensor is **identically zero** and the relaxed bulk constants **equal** the
affine ones to machine precision. The free-surface numbers of the earlier audit are a
boundary-condition quantity (a specimen stiffness), not a bulk non-affine correction.
Section 5 states the correction and what remains open.

## 1. Method

A periodic (Born-von Karman) cell with bond list $(i,j)$, rest vectors $r$, unit normals $n$ and
stiffness $k$ has bond energy $\Phi = \tfrac12\sum k(|r+u_j-u_i|-|r|)^2$. Expanded around the
affine state $u_i = E\,x_i$:

```
Phi(u)   = Phi_affine + 1/2 u^T K u - f^T u
K        = sum_bonds k n n            (k = 0 dynamical matrix, 3x3 blocks)
f_i      = sum_{bonds at i} k |r| (n . E . n) n
u*       = K^+ f                      (non-affine relaxation)
Xi       = du*/dE                     (internal-strain tensor)
Phi_relaxed = Phi_affine - 1/2 f^T u*
```

Two consequences are used below: the affine constants are an upper bound on the relaxed ones,
and the bound is saturated exactly when the affine state is force balanced. Implementation:
`src/tensordem/born_huang.py` (NumPy/SciPy float64). `bonded_lattice_analysis` accepts any
periodic bond list, so the same engine is applied to the solver lattice and to two-atom-basis
control lattices with known closed forms.

## 2. Case definition

| item | value |
|---|---|
| particle radius | 0.025 m |
| lattice spacing a = 2r | 0.05 m |
| bond stiffness k | 2000.0 N/m |
| topology | 13 half-space offsets / 26 neighbours, one atom per primitive cell |
| periodic cell | 3 x 3 x 3 primitive cells, 27 atoms, 351 bonds, 13 bonds per cell |
| strain probes | eps_xx (unit), biaxial diag(1,1,0), engineering shear gamma = 1 |
| scope | linear small-strain elasticity of the intact bond network; no fracture, contact, dynamics |

## 3. Result: the internal strain of the solver lattice is zero

| constant | affine closed form [Pa] | affine from solver bond list [Pa] | periodic k = 0 relaxed [Pa] | relaxed vs closed form |
|---|---|---|---|---|
| C11 | 173333.333333 | 173333.333333 | 173333.333333 | +2.22e-16 |
| C12 | 93333.333333 | 93333.333333 | 93333.333333 | +2.22e-16 |
| C44 | 93333.333333 | 93333.333333 | 93333.333333 | +2.22e-16 |

- Affine force balance: largest relative residual 1.64e-14 (xx 1.64e-14, biaxial 8.21e-15, shear 0.00e+00).
  This is the numerical image of the exact cancellation `sum_26 k|r|(n.E.n)n = 0` produced by
  inversion symmetry of the 26-shell; the 26 bonds are enumerated separately, so it is a
  cancellation and not an identity built into the code.
- Internal strain: max |u*| = 3.87e-18 m (xx), 2.87e-18 m (biaxial), 0.00e+00 m (shear), per unit strain. Relaxed/affine = 1.000000000000000 for all three
  probes.
- Hence the affine Zener anisotropy A = 2 C44/(C11 - C12) = 7/3 = 2.3333 and the Cauchy relation
  C12 = C44 are **exact bulk properties** of this lattice, not constraint artifacts.

Why it must vanish: every atom of a one-atom-basis Bravais lattice is a centre of inversion, so
the `+n` and `-n` bonds of the 26-shell exert opposite forces of equal magnitude. A non-zero
`Xi` requires more than one atom per primitive cell with inequivalent bonding.

## 4. Negative controls: the engine does report internal strain when it exists

Diatomic chain (two atoms per cell, springs k1 and k2) against its closed form: internal strain
$u_A/\varepsilon = L(k_1-k_2)/2(k_1+k_2)$, relaxed $C_{11} = 2k_1k_2/((k_1+k_2)L)$ (harmonic
mean), affine $C_{11} = (k_1+k_2)/2L$ (arithmetic mean).

| k1 [N/m] | k2 [N/m] | affine engine / closed [Pa] | relaxed engine / closed [Pa] | Xi engine / closed [m] | relaxed/affine |
|---|---|---|---|---|---|
| 1.0 | 1.0 | 1.000000000000 / 1.000000000000 | 1.000000000000 / 1.000000000000 | 0.000000000000 / 0.000000000000 | 1.000000 |
| 4.0 | 3.0 | 3.500000000000 / 3.500000000000 | 3.428571428571 / 3.428571428571 | 0.071428571429 / 0.071428571429 | 0.979592 |
| 2.0 | 1.0 | 1.500000000000 / 1.500000000000 | 1.333333333333 / 1.333333333333 | 0.166666666667 / 0.166666666667 | 0.888889 |
| 3.0 | 1.0 | 2.000000000000 / 2.000000000000 | 1.500000000000 / 1.500000000000 | 0.250000000000 / 0.250000000000 | 0.750000 |
| 5.0 | 1.0 | 3.000000000000 / 3.000000000000 | 1.666666666667 / 1.666666666667 | 0.333333333333 / 0.333333333333 | 0.555556 |
| 5.0 | 0.2 | 2.600000000000 / 2.600000000000 | 0.384615384615 / 0.384615384615 | 0.461538461538 / 0.461538461538 | 0.147929 |

Max absolute error over all controls: **2.2e-16** (machine precision). The engine
reproduces a *non-zero* internal strain (up to 0.462 m per unit strain at k1:k2 = 5:0.2, where
relaxed/affine = 0.148) to the same accuracy as it reports zero for the solver lattice, so the
zero in section 3 is a property of the lattice and not of the code.

Three-dimensional two-basis control (same axial springs plus same-species y/z springs, k = 0.7):
eps_xx affine 1.500000 Pa -> relaxed 1.333333 Pa (ratio 0.888889), Xi = 0.166667 m; eps_yy affine 0.700000 Pa -> relaxed 0.700000 Pa (ratio 1.000000).
The transverse springs are inert under eps_xx and affine under eps_yy, as predicted.

## 5. What the free-surface ladder of the earlier page actually measures

The finite free-surface protocol (`relaxed_patch`, shear: drive the y faces, free interior, core
energy density) is re-implemented with a sparse solve so the ladder can be extended:

| n | N | relaxed/affine (this audit) | documented | deviation | solve path | core/surface RMS |
|---|---|---|---|---|---|---|
| 4 | 64 | 0.743026 | 0.7430 | +3.5e-05 | spsolve | 0.870 |
| 5 | 125 | 0.658186 | — | — | spsolve | 0.896 |
| 6 | 216 | 0.687167 | 0.6148 | +1.2e-01 | lsqr | 0.997 |
| 7 | 343 | 0.589326 | — | — | spsolve | 0.918 |
| 8 | 512 | 0.573155 | 0.5732 | -7.9e-05 | spsolve | 0.924 |
| 10 | 1000 | 0.191581 | 0.5547 | -6.5e-01 | lsqr | 0.961 |
| 12 | 1728 | 0.545054 | 0.5451 | -8.4e-05 | spsolve | 0.936 |

**(a) The free-surface block is singular for n = 2 (mod 4).** At n = 6 the free block has
432 degrees of freedom and a 17-dimensional zero-energy space (min eigenvalue
-2.8e-12 against max 3.6e+04). The load is compatible (|z^T rhs|/||rhs|| = 4.2e-17),
so in exact arithmetic the energy is invariant along those directions; a factorisation-based
solve and a minimum-norm solve still return different fields, and only the first lands on the
smooth branch. This audit uses the minimum-norm path, which reproduces n = 4, 8, 12 to
4: +3.5e-05, 8: -7.9e-05, 12: -8.4e-05 but deviates at
n = 6: +1.2e-01, n = 10: -6.5e-01.
Those documented entries are solver-path dependent and are **not reproduced here**; the odd
sizes added in this audit fill the sequence smoothly (n = 5: 0.658, n = 7: 0.589) and bracket the documented values, so the trend is not in doubt, only its interpretation.

**(b) The non-affine field is global, not a surface skin.** The quotient of core RMS displacement
to near-surface RMS displacement is 0.87–1.00 at every size: the relaxation is not confined to
the missing-bond layers (that would give a small quotient) but is a long-wavelength relaxation
of the whole specimen that the mixed boundary condition permits. It is not the short-range
internal-strain object that `Xi` measures.

**(c) The 1/n extrapolation is not a bulk estimate.** Fitting `ratio = 1 - c1/n` to the well-posed
sizes gives intercept 0.4295 (max residual 0.0156), close to the earlier
page's 0.432; `1 - c1/n - c2/n^2` gives 0.5345 (max residual 7.5e-04). The data cannot
separate the two models, while the rigorous periodic answer is exactly 1.000. Extrapolating a
mixed-boundary specimen quantity in 1/n reproduces the earlier number without validating it as a
bulk constant.

## 6. Correction to docs/dem3d_lattice_elastic_constants.md

- Section 4's "the shear stiffness is strongly non-affine" and "the affine Zener anisotropy is
  mostly an artifact" hold only for that free-surface specimen under that mixed boundary
  condition. For the bulk lattice the k = 0 relaxed constants equal the affine ones to 2.2e-16
  and the Zener ratio 7/3 stands.
- Section 6's gap ("periodic k = 0 Born-Huang internal-strain calculation; it is not implemented
  here") is closed by this page: `Xi = 0` identically for the one-atom-basis 26-neighbour lattice.
- The five-point 1/n extrapolation (intercept 0.432 of affine C44) should not be quoted as the
  bulk relaxed C44: it extrapolates a specimen stiffness and 2 of its 5 points sit on a singular
  linear system.

## 7. Reproduce

```bash
python scripts/audit_dem3d_born_huang.py --output docs/dem3d-born-huang
python scripts/make_dem3d_born_huang_doc.py
python -m pytest tests/test_dem3d_born_huang.py -q
```

Artifacts in `docs/dem3d-born-huang/`: `born_huang_audit.json` (SHA-256 of the body embedded),
`free_surface_ladder.csv`, `fig1_born_huang_solver_lattice.png`,
`fig2_diatomic_control_validation.png`, `fig3_free_surface_ladder.png`.

## 8. Known limits

- Linear small-strain elasticity of the intact network. Bond breaking, contacts, damping and ice
  material data are outside the model; nothing here calibrates ice or ship-ice resistance.
- `Xi` is reported in displacement form per unit strain. This lattice needs no internal-strain
  parameter, so no Born-Huang `zeta` is quoted.
- The free-surface investigation covers n <= 12 with one boundary protocol (y faces driven).
  Driving both lateral faces gives 0.847 at n = 4 against 0.743 for y only; the merged routine's
  `yz_faces` label pins axis 1 only, which is recorded as a trap in the module.
- Zero-energy modes of the free block were counted only at n = 6 (dense eigensolve); larger
  sizes were classified by the direct-solve failure instead of an eigensolve.

