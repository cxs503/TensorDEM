# Legacy propeller–ice milling module (IN27-S4): analytic verification and defect report

**Protocol** `tensordem-propeller-ice-milling-legacy-verification-v1` · **module** `src/tensordem/propeller_ice_milling.py` · **campaign node** IN27-S4
**Tested twin** src/tensordem/propeller_ice/ (IN34-W3-N05, 28 tests) · **gates** 17/25 PASS · **defects** 3
**Artifacts** `artifacts/propeller_ice_milling_legacy/` · **Harness** `scripts/verify_propeller_ice_milling_legacy.py` · **Gates as tests** `tests/test_propeller_ice_milling_legacy_verification.py`

---

## 1. Case description

Ice-going propellers do not merely push broken ice aside: a slab drawn into the
disc is **milled** by the blades, the shaft sees a pulsating torque at the
blade-passing frequency, and the crushed debris is ejected and transported by the
propeller race.  Two independent implementations of this physics live in this
repository:

| implementation | file | campaign node | tests | driver | documentation |
|---|---|---|---|---|---|
| legacy | `src/tensordem/propeller_ice_milling.py` (1140 lines) | IN27-S4 | **0** | none | none |
| package | `src/tensordem/propeller_ice/` (7 modules) | IN34-W3-N05 | 28 | `scripts/run_propeller_ice_milling.py` | `docs/PROPELLER_ICE_MILLING_TRANSIENT.md` |

The legacy module ships in the public `src` tree and declares a *plastic-capped
Hertz* contact, blade-element/momentum open-water hydrodynamics, a
*“momentum-conserving”* self-similar race and a debris transport model, but no
test, driver or document ever exercises it.  This mission closes that verification
debt: every closed-form claim is pinned to an independent reference, the module is
cross-checked against its tested twin on a matched geometry, and the defects found
are registered here with their evidence.

Geometry used throughout (legacy defaults): `Z=4 D=3.0 EAR=0.700 P/D=0.95`, `c(0.7R)` from the Beta-normalised
planform; the package geometry is matched to `A_E/A_0 = 0.700000` (`chord_ratio = 0.538998`, `skew_tip = 0`).

## 2. Method

Six gate families, each against a reference that is *not* the code under test:

| gate | quantity | independent reference |
|---|---|---|
| G1 | planform scale, expanded-area closure, pitch law | closed-form Beta integral `B(1+e,1+e)`, `A_E = EAR·πD²/4`, `atan(P/2πr)` |
| G2 | propeller-race momentum flux, induced velocity | `∫ρu²2πr dr = T`; actuator-disk `½(−V+√(V²+2T/ρA))` |
| G3 | debris quadratic-drag transport | closed form `|w|(t) = |w0|/(1+k|w0|t)` + step-refinement study |
| G4 | torque spectrum | synthetic BPF harmonic train with known amplitudes; white-noise negative control |
| G5 | open-water BEM | efficiency identity `η = J K_T/(2πK_Q)`, monotone `K_T(J)`, zero-thrust `J0` |
| G6 | cross-implementation | tested `tensordem.propeller_ice` BEM on a geometry matched in `D, Z, r_h/R, P/D, A_E/A_0` |

Compute: all gates are seconds-scale (single process, no cluster), so the run is
repeatable on the control node.  Nothing here is a physical validation: no
experimental open-water chart or ice-milling torque measurement exists in this
repository, so absolute magnitudes stay **WITHHELD** by construction.

## 3. Results

### 3.1 Gate table (programmatically injected)

| gate | what | measured | reference | rel. error | tol | verdict |
|---|---|---|---|---|---|---|
| `G1a_chord_scale_e0.20` | chord_scale vs closed-form Beta integral (planform exponent 0.2) | 1.2455 | 1.2455 | 1.46e-05 | 0.0001 | **PASS** |
| `G1b_panel_area_e0.20` | sum of blade-panel areas vs A_E = EAR pi D^2 / 4 (e=0.2) | 5.0014 | 4.948 | 0.0108 | 0.02 | **PASS** |
| `G1a_chord_scale_e0.35` | chord_scale vs closed-form Beta integral (planform exponent 0.35) | 1.6441 | 1.6441 | 7.89e-06 | 0.0001 | **PASS** |
| `G1b_panel_area_e0.35` | sum of blade-panel areas vs A_E = EAR pi D^2 / 4 (e=0.35) | 5.0142 | 4.948 | 0.0134 | 0.02 | **PASS** |
| `G1a_chord_scale_e0.50` | chord_scale vs closed-form Beta integral (planform exponent 0.5) | 2.1525 | 2.1525 | 3.47e-06 | 0.0001 | **PASS** |
| `G1b_panel_area_e0.50` | sum of blade-panel areas vs A_E = EAR pi D^2 / 4 (e=0.5) | 5.0147 | 4.948 | 0.0135 | 0.02 | **PASS** |
| `G1c_pitch_angle_0.7R` | pitch angle at 0.7 R vs atan(P/(2 pi r)) | 0.40778 | 0.40778 | 1.36e-16 | 1e-12 | **PASS** |
| `G2_momentum_flux_x0.0m` | propeller-race momentum flux at x = 0.0 m | 9999.9 | 20000 | 0.5 | 1e-05 | **FAIL** |
| `G2_momentum_flux_x1.0m` | propeller-race momentum flux at x = 1.0 m | 9999.9 | 20000 | 0.5 | 1e-05 | **FAIL** |
| `G2_momentum_flux_x4.0m` | propeller-race momentum flux at x = 4.0 m | 9999.9 | 20000 | 0.5 | 1e-05 | **FAIL** |
| `G2b_induced_velocity` | race induced velocity vs actuator-disk closed form | 0.95113 | 0.95113 | 0 | 1e-12 | **PASS** |
| `G3_drag_relax_finest` | legacy RK2 debris exit speed vs closed form in the RaceModel carrier | 5.0493 | 7.4977 | 0.327 | 0.02 | **FAIL** |
| `G3b_convergence_order` | observed step-refinement order of the debris integrator (self-convergence) | 2.8036 | 2 | 0.402 | 0.6 | **PASS** |
| `G4a_bpf_amplitude` | recovered BPF torque amplitude vs the synthetic amplitude | 120 | 120 | 5.46e-07 | 0.005 | **PASS** |
| `G4b_peak_frequency` | spectral peak location vs f_BPF | 6 | 6 | 0 | 0.005 | **PASS** |
| `G4c_harmonic_2` | recovered 2 x BPF harmonic amplitude | 40 | 40 | 1.03e-06 | 0.02 | **PASS** |
| `G4c_harmonic_3` | recovered 3 x BPF harmonic amplitude | 15 | 15 | 2.82e-06 | 0.02 | **PASS** |
| `G4d_white_noise_control` | white-noise record must not concentrate its peak on the BPF | 163 | 6 | 0 | — | **PASS** |
| `G5a_efficiency_identity` | max over J of |eta - J K_T/(2 pi K_Q)| / eta | 0 | 0 | 0 | 1e-10 | **PASS** |
| `G5b_KT_monotone` | K_T decreases monotonically with advance ratio | 1.3355 | 1.3355 | 0 | — | **PASS** |
| `G5c_zero_thrust_advance_ratio` | zero-thrust advance ratio J0 exists inside the operating envelope | 0.83652 | 0.7 | 0.195 | 0.9 | **PASS** |
| `G6_K_T_divergence` | agreement 1 - worst |K_T_legacy - K_T_package| / K_T_package over J = 0.2..0.8 | 0.27518 | 1 | 0.725 | 0.25 | **FAIL** |
| `G6_K_Q_divergence` | agreement 1 - worst |K_Q_legacy - K_Q_package| / K_Q_package over J = 0.2..0.8 | 0.25696 | 1 | 0.743 | 0.25 | **FAIL** |
| `G6_eta_divergence` | agreement 1 - worst |eta_legacy - eta_package| / eta_package over J = 0.2..0.8 | 0.70131 | 1 | 0.299 | 0.25 | **FAIL** |

The three failing families are the three defects below; every other claim holds to
the stated tolerance (the planform scale to 1.5e-5, the pitch law and the
efficiency identity to machine precision, the spectral amplitudes to <0.5 % and
<2 % on the BPF and its harmonics).

### 3.2 Cross-implementation table (matched geometry)

Package note: `Z=4 D=3.000 m r_h/R=0.40 P/D=0.95 A_E/A_0=0.700 c(0.7R)=1.617 m skew_tip=0.0 deg phi(0.7R)=23.3 deg`

| J | K_T legacy | K_T package | K_Q legacy | K_Q package | η legacy | η package | strips at induction cap (package) |
|---|---|---|---|---|---|---|---|
| 0.20 | 0.3149 | 0.7046 | 0.0244 | 0.0661 | 0.4103 | 0.3395 | 24/24 |
| 0.40 | 0.1634 | 0.3476 | 0.0191 | 0.0514 | 0.5454 | 0.4305 | 21/24 |
| 0.60 | 0.0984 | 0.1812 | 0.0135 | 0.0322 | 0.6977 | 0.5373 | 0/24 |
| 0.80 | 0.0215 | 0.0781 | 0.0047 | 0.0185 | 0.5768 | 0.5386 | 0/24 |

Figures: `fig1_open_water_crossval.png` (characteristics of both implementations),
`fig2_debris_convergence.png` (debris transport vs closed form under step
refinement), `fig3_spectrum_recovery.png` (torque-spectrum harmonic recovery).

## 4. Defects

### DEFECT-D1 — high — `propeller_ice_milling.RaceModel.centreline`

the Gaussian profile is normalised with u_c = sqrt(M/(2 pi sigma^2)), which delivers int rho u^2 2 pi r dr = T/2 for momentum flux M = T/rho; the class docstring claims the profile 'conserves momentum exactly'.

*Impact.* propeller-race speeds (and hence chip transport) are sqrt(2) low.

*Evidence.* `{"momentum_flux_ratio": 0.4999946666325328, "gate": "G2_momentum_flux_*"}`

*Note.* the induced_velocity closed form uses the correct T/(2 rho A) form, so the defect is confined to centreline/axial_velocity

### DEFECT-D2 — high — `propeller_ice_milling.integrate_debris (water_velocity closure)`

the debris advection uses sigma = 0.5 * diameter (= R) while RaceModel.sigma() returns 0.25 * diameter (= R/2); the two conventions differ by 2 in width, i.e. 4 in u_c, so the chips see a jet that is 4x weaker than the race the milling torque was computed with.

*Impact.* IN27-S4's own acceptance observable (EJECTION_BAND 0.5-2.0 m/s) is evaluated against an under-driven jet; instability of the band is not evidence of physical agreement.

*Evidence.* `{"exit_speed_rel_err_vs_module_carrier": 0.3265555331830977, "gate": "G3_drag_relax_finest"}`

### DEFECT-D3 — medium — `propeller_ice.hydro.solve_open_water`

at the matched 3 m geometry and J <= 0.2 every strip sits on the induction cap a = 0.85: the blade-element load is then no longer balanced by the annulus momentum, yet the returned residual (max |delta a|) is ~1e-8, so a residual-based test cannot detect the violation. The announced Buhl high-induction branch (_buhl) is never called.

*Impact.* absolute open-water coefficients of BOTH implementations are WITHHELD (no experimental chart in the repository); the package coefficients are cap-limited below J ~ 0.3, the legacy ones are loss/polar-limited.

*Evidence.* `{"K_T_legacy": [0.3148580360583247, 0.16343922224843874, 0.09840154338088734, 0.02150346586852765], "K_T_package": [0.70461730634673, 0.3475753539005899, 0.18117462632963352, 0.07814205831298512], "strips_at_cap_per_J": [24, 21, 0, 0], "worst_K_T_divergence": 0.2751842776190905, "min_strips_at_cap": 0}`

## 5. Points to note / limitations

* **The defects were found by cross-checking, not by reading the prose.** The
  module's own docstring asserts momentum conservation; quadrature shows a factor
  2. An untested 1140-line module in a shipped `src` tree is a standing risk, and
  the same lesson applies to the package: its BEM accepts cap-saturated strips
  because the residual is defined as `max|Δa|` rather than as a blade/momentum
  imbalance.
* **No absolute coefficient is claimed.** Without a Wageningen-B style table or a
  measured open-water point, `K_T`, `K_Q` of *both* implementations remain
  WITHHELD; only internal identities, trends and J0 are gateable.
* **Debris transport is dilute and gravity-free** in both implementations (chip
  Froude number ≫ 1 over the 1 m discharge length); the closed-form reference
  therefore validates the integrator, not the physical model.
* **The contact law is not isolated.** The plastic-capped Hertz branch lives inside
  `MillingSimulation` and cannot be exercised without a full transient; it is
  covered indirectly by the package's `BladeIceContact` tests and remains an open
  item for the legacy module.

## 6. Error judgement (explicit)

| key quantity | reference | error | < 3 % ? |
|---|---|---|---|
| planform scale `c_max` | Beta integral | 1.46e-05 | yes |
| blade area vs `A_E` | `EAR·πD²/4` | 0.0108 | yes |
| BPF torque amplitude | analytic 120 N·m | 5.46e-07 | yes |
| efficiency identity | `J K_T/(2πK_Q)` | 0 | yes |
| race momentum flux | `T` | 0.5 | **no — factor 2 (DEFECT-D1)** |
| debris exit speed | closed form in the module's own carrier | 0.327 | **no (DEFECT-D2)** |
| `K_T` legacy vs package | matched geometry | 0.725 | **no — absolute value WITHHELD (DEFECT-D3)** |

## 7. Disposition

**request changes on the legacy module; do not delete it yet.** Concretely:

1. `RaceModel.centreline` must use `u_c = sqrt(2 M/(π σ²))`… i.e. the normalisation
   that delivers `∫ρu²2πr dr = T`; `integrate_debris` must consume
   `RaceModel.sigma/centreline` instead of its own `σ = 0.5 D` closure.
2. The fixed module must pass the two `xfail(strict=True)` gates in
   `tests/test_propeller_ice_milling_legacy_verification.py` unchanged — that is
   the acceptance criterion of the repair.
3. Until then the module's ejection-speed observable and every number derived from
   its race are **not admissible evidence**; the tested IN34-W3-N05 package remains
   the canonical propeller–ice engine.
4. Follow-up (registered, not fixed here — out of this mission's scope): the package
   BEM's cap-saturation branch and its unused `_buhl` correction need either a
   blade/momentum imbalance residual or a documented cap-validity limit `J`.

## 8. Evidence pointers

* platform report **id 15253** (`scenario_key = tensordem-propeller-ice-milling-legacy-verification`)
* gates as tests: 14 passed, 2 xfailed (`xfail(strict)`) in
  `tests/test_propeller_ice_milling_legacy_verification.py` (7.3 s)
* machine-readable evidence: `artifacts/propeller_ice_milling_legacy/verification.json`

## 9. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_propeller_ice_milling_legacy_verification.py -q
PYTHONPATH=src python scripts/verify_propeller_ice_milling_legacy.py   # writes artifacts + 3 figures
PYTHONPATH=src python scripts/render_propeller_ice_milling_legacy_doc.py
```

