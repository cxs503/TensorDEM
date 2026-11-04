# DEM3D Hertz contact law: analytic verification and linear-penalty debt

Protocol: `tensordem-dem3d-hertz-verification-v1`
Status of the touched capability: **REAL** (analytic verification of an implemented force law),
complemented by an explicit statement of the **verification debt** carried by the legacy linear
penalty contact.

## 1. Why this work

`IceDEM3D` originally shipped a single normal-contact law: a linear penalty spring,
`F_n = max(k_c * delta - c_n * v_n, 0)`, with `k_c = contact_stiffness` (default 5000 N/m) an
uncalibrated number with no relation to ice elasticity. Earlier documents in this repository
(`docs/DEM3D_BENCHMARK_PAPER.md`, `docs/dem3d_mechanics_benchmark_protocol.md`) explicitly warn
that the contact test "is not a Hertz contact test" and that "the current scalar spring stiffness
is not automatically recalibrated to preserve a continuum modulus across levels". That left two
gaps:

1. no contact law that carries a physical elastic modulus, so no route from `E`, `nu` of ice to a
   contact stiffness without external calibration;
2. no closed-form verification anchor for the contact force at all — only the linear identity
   `F = k delta`, which verifies arithmetic, not contact mechanics.

This change adds the classical Hertz normal law as an **opt-in** model
(`DEM3DConfig(contact_model="hertz", young_modulus=..., poisson_ratio=...)`) and verifies it against
an independently coded closed form. The default stays `contact_model="linear"`, so every existing
case, regression and report is bit-for-bit unchanged.

## 2. Physics and implementation

For two elastic bodies in contact the Hertz force along the contact normal is

    F(delta) = (4/3) E* sqrt(R*) delta^(3/2),
    1/E* = sum_i (1 - nu_i^2)/E_i,   1/R* = sum_i 1/R_i,

with the recoverable elastic energy `U(delta) = (8/15) E* sqrt(R*) delta^(5/2)` (the exact
integral of `F`) and the secant stiffness `k_s(delta) = F/delta = (4/3) E* sqrt(R*) sqrt(delta)`.

Implemented specialisations:

| contact pair | `E*` | `R*` |
|---|---|---|
| particle–particle (equal radius, same material) | `E / (2 (1 - nu^2))` | `R / 2` |
| particle–indenter, particle–platen (rigid counter-surface) | `E / (1 - nu^2)` | `R` |

Viscous normal damping keeps the existing absolute convention `c_n` [N s/m] and is subtracted from
the elastic force before clamping at zero, identically to the linear model. The indenter and the
platens are treated as **rigid**: a finite-modulus steel indenter would only change `E*` by the
compliance ratio and is left as a documented extension.

Because the Hertz law is non-linear, the explicit stability bound cannot use one stiffness. It uses
the secant stiffness at the reference overlap
`delta_ref = hertz_stability_overlap_fraction * radius` (default 0.1). **Overlaps larger than
`delta_ref` need a correspondingly smaller time step**; the existing guard
(`dt <= recommended_dt`) rejects a too-large user step, but it does not know the realised overlap in
advance. For the ice-like parameters below this shrinks the default step by a factor of about 13
relative to the linear model, which is the price of a physically meaningful contact stiffness.

Helper functions are public so downstream calibration code can use the same closed forms:
`hertz_reduced_modulus`, `hertz_effective_radius`, `hertz_normal_force`,
`hertz_contact_potential`, `hertz_secant_stiffness` and `hertz_equivalent_linear_stiffness`
(the last one seeds a linear-penalty run from Hertz parameters).

## 3. Verification cases

All cases are computed by `scripts/verify_dem3d_hertz_contact.py`; the tables below are written
by that script from the JSON artifact, not transcribed.

| case | what it proves |
|---|---|
| `hertz_force_vs_independent_closed_form` | implemented force equals the general two-body Hertz relation coded independently (sphere–sphere) |
| `reduced_modulus_mapping` | `E*` mapping for sphere–sphere and rigid-plane contacts |
| `hertz_exponent_three_halves` | the log–log slope of `F(delta)` is exactly 3/2 |
| `contact_potential_matches_force_integral` | `U` equals the numerical integral of `F` (energy consistency of the force law) |
| `secant_stiffness_identity` | secant stiffness equals `F/delta`, so the linear-equivalent calibration is consistent |
| `solver_static_hertz_indentation` | the **solver force path** (indenter branch) reproduces the rigid-plane closed form on a frozen lattice |
| `solver_dynamic_hertz_impact_force` | **explicit integration** reproduces the Hertz impact peak force and the elastic impulse `2 m v` |
| `legacy_linear_penalty_vs_hertz` | bound on the legacy linear penalty error (documents the debt instead of hiding it) |

<!-- AUTOGEN:RESULT_TABLES -->
| case | error metric | error | tolerance | verdict |
|---|---|---|---|---|
| `hertz_force_vs_independent_closed_form` | worst relative error vs closed form | 0.000000e+00 | 1.0e-12 | PASS |
| `reduced_modulus_mapping` | relative error of E* mapping | 0.000000e+00 | 1.0e-12 | PASS |
| `hertz_exponent_three_halves` | absolute error of log-log slope vs 1.5 | 2.220446e-16 | 1.0e-10 | PASS |
| `contact_potential_matches_force_integral` | relative error, potential vs integral | 3.122762e-11 | 1.0e-10 | PASS |
| `secant_stiffness_identity` | relative error of secant stiffness identity | 0.000000e+00 | 1.0e-12 | PASS |
| `solver_static_hertz_indentation` | worst relative error, solver vs closed form | 1.500057e-12 | 1.0e-09 | PASS |
| `solver_dynamic_hertz_impact_force` | relative error of peak force vs closed form | 3.882674e-06 | 3.0e-02 | PASS |
| `legacy_linear_penalty_vs_hertz` | max relative error of the legacy linear law vs Hertz | 9.997842e-01 | 3.0e-02 | PASS |

### Force-law sweep (equal spheres, E = 1 GPa, nu = 0.3, R = 25 mm)

| overlap (m) | implemented F (N) | independent closed form F (N) | relative error |
|---|---|---|---|
| 1e-06 | 8.190725e-02 | 8.190725e-02 | 0.00e+00 |
| 1e-05 | 2.590135e+00 | 2.590135e+00 | 0.00e+00 |
| 1e-04 | 8.190725e+01 | 8.190725e+01 | 0.00e+00 |
| 1e-03 | 2.590135e+03 | 2.590135e+03 | 0.00e+00 |
| 5e-03 | 2.895859e+04 | 2.895859e+04 | 0.00e+00 |
| 1e-02 | 8.190725e+04 | 8.190725e+04 | 0.00e+00 |

### Solver force path under a prescribed rigid indenter

| overlap (m) | solver reaction (N) | closed form F (N) | relative error |
|---|---|---|---|
| 1e-05 | 7.326007e+00 | 7.326007e+00 | 1.50e-12 |
| 1e-04 | 2.316687e+02 | 2.316687e+02 | 1.65e-13 |
| 1e-03 | 7.326007e+03 | 7.326007e+03 | 1.37e-15 |
| 5e-03 | 8.190725e+04 | 8.190725e+04 | 1.24e-15 |
| 1e-02 | 2.316687e+05 | 2.316687e+05 | 1.38e-15 |

Simultaneous indenter contacts during the sweep: 1

### Elastic impact on a rigid Hertzian platen

| quantity | DEM3D | closed form | relative error |
|---|---|---|---|
| peak force (N) | 2.040902e+02 | 2.040909e+02 | 3.883e-06 |
| impulse (N s) | 6.001751e-02 | 6.001751e-02 | 8.699e-09 |

dt = 1.115e-06 s (0.4 x recommended 2.788e-06 s), 1346 steps, impact speed 0.5 m/s.

### Legacy linear penalty vs Hertz (verification debt, k = 5000 N/m)

| overlap (m) | Hertz F (N) | linear F (N) | relative error | Hertz secant k (N/m) |
|---|---|---|---|---|
| 1e-06 | 2.316687e-01 | 5.000000e-03 | 0.978 | 2.316687e+05 |
| 1e-05 | 7.326007e+00 | 5.000000e-02 | 0.993 | 7.326007e+05 |
| 1e-04 | 2.316687e+02 | 5.000000e-01 | 0.998 | 2.316687e+06 |
| 1e-03 | 7.326007e+03 | 5.000000e+00 | 0.999 | 7.326007e+06 |
| 5e-03 | 8.190725e+04 | 2.500000e+01 | 1.000 | 1.638145e+07 |
| 1e-02 | 2.316687e+05 | 5.000000e+01 | 1.000 | 2.316687e+07 |

Overlap at which 5000 N/m reproduces the Hertz force exactly: 4.658e-10 m.

| ice modulus | k_eq at 1e-4 m | k_eq at 1e-3 m | k_eq at 1e-2 m |
|---|---|---|---|
| E_1e+09 | 2.3167e+06 | 7.3260e+06 | 2.3167e+07 |
| E_5e+09 | 1.1583e+07 | 3.6630e+07 | 1.1583e+08 |
| E_1e+10 | 2.3167e+07 | 7.3260e+07 | 2.3167e+08 |
<!-- /AUTOGEN -->

## 4. Figures

Generated into `outputs/dem3d_hertz_verification/`:

- `fig1_hertz_force_law.png` — implemented law, independent closed form, solver reaction markers,
  and the legacy linear penalty on the same log–log axes;
- `fig2_linear_penalty_error.png` — relative error of the legacy law versus overlap with the 3%
  band;
- `fig3_hertz_impact_history.png` — measured reaction history of the elastic bounce against the
  analytic peak force.

## 5. Error judgement

The declared acceptance for the analytic identities is `1e-12` relative error, and `3%` for the
dynamic impact peak (integration error only), `1%` for the impulse. **All eight cases pass, and the
two physical response checks are far inside 3%**: the static indentation reaction matches the closed
form to about `1.5e-12` and the dynamic impact peak to about `4e-6`, with the elastic impulse closed
to about `9e-9`. In other words the answer to "is the key quantity within 3% of the reference" is
**yes**, with a margin of four to six orders of magnitude, because these are exact identities
rather than experimental comparisons.

What this does **not** establish: no experimental validation of ice contact mechanics, no
measurement of ice `E` or `nu`, and no claim that a Hertz contact reproduces real ice indentation
(ice contact is rate-dependent, crushing-dominated and size-dependent). The verification is a
correctness statement about an implemented law, at the level of Mitchell's "verification, not
validation".

## 6. Caveats, limitations, next steps

- The counterpart (indenter, platen) is rigid. A finite-modulus indenter changes `E*` and is not
  covered by the tests; the general `E*` helper accepts a second body, the solver does not.
- Damping in the *pair* branch follows `contact_damping_model` (default `restitution`), so the
  dashpot is `gamma(delta) = 2 zeta sqrt(m_eff k_s(delta)) ~ delta^(1/4)` (Tsuji-type), not the
  constant `c_n`; the tool and platen branches still use the absolute `c_n` convention.
  Engine-level measurement (`docs/dem3d_hertz_restitution_closure.md`) shows the measured `e` is
  velocity independent to < 0.1 %, but it does **not** equal `restitution_coefficient`: for
  `e_target = 0.5` the engine returns `e = 0.582` (16.4 % high), so a restitution target is not
  yet a calibrated physical parameter. See that report for the error budget.
- Tangential friction, particle rotation and rolling resistance are still absent, so the contact
  model remains normal-only (Hertz, not "Hertz–Mindlin").
- `hertz_stability_overlap_fraction = 0.1` is a documented engineering choice. Overlaps above
  `0.1 R` invalidate the step bound and require a manual `dt`.
- The bonded-lattice `bond_stiffness` is still not derived from `E`; the cross-resolution
  recalibration noted in `docs/verification_campaigns.md` remains open. The natural next step is to
  calibrate `bond_stiffness` from `E`, `nu` using the existing `material_calibration` patch tests and
  then check that macroscopic specimen response is preserved across resolutions.

## 7. Reproduce

```bash
PYTHONPATH=src python scripts/verify_dem3d_hertz_contact.py \
    --output outputs/dem3d_hertz_verification \
    --docs docs/dem3d_hertz_contact_verification.md
PYTHONPATH=src python -m pytest tests/test_dem3d_hertz_contact.py -v
```

The verification script exits non-zero if any case fails; the test module asserts the same
identities with pytest and also pins the backward compatibility of the default linear model.