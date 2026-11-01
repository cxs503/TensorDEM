# Ice-ridge keel grounding on a buried subsea pipeline

Coupled DEM / Mohr-Coulomb subgrade / beam-on-Winkler model for the vertical
and lateral loads that a grounded ice-ridge keel imposes on a trenched
pipeline, with an ultimate-capacity envelope and a DNV-OS-F101 structural
check.

Modules: `tensordem.ridge_dem` (particles, bonds, obstacles, calibration),
`tensordem.ridge_geotech` (subgrade, pipe section, beam on elastic foundation),
`tensordem.ridge_grounding` (coupled workflow, metrics, envelope).
Entry point: `scripts/run_ice_ridge_grounding.py`; figures:
`scripts/make_ridge_grounding_figures.py`. All numerical results are written to
`artifacts/ridge_grounding/*.json` and are reproduced by re-running those two
scripts; no number in the campaign report is copied by hand.

## 1. Physics and idealisation

An ice ridge is idealised as a two-dimensional front-normal section (plane
strain, unit out-of-plane thickness): a triangular keel of depth `H_k` and
mouldline width `W_k`, built from disks on a triangular lattice with a
prescribed porosity and a prescribed freeze-bond fraction. The keel is driven
by a *kinematic platen* representing the consolidated layer and sail pushed by
the surrounding ice sheet.

Why a platen and not gravity: sea ice is buoyant (`rho_i = 917 < rho_w = 1025`),
so the net body force on every disk points **up**. A ridge cannot fall onto a
pipeline by itself; the grounding force is supplied by the driving ice sheet,
and the platen measures exactly that force. This is the reason the model reports
a platen reaction as well as a pipe load, and why a force balance
(`platen load = pipe load + subgrade reaction + buoyancy + inertia`) is
available as an internal consistency check.

Interactions in the DEM:

| interaction | law | parameters |
|---|---|---|
| freeze bond | axial spring `k_n` + incremental tangential spring `k_s`, elastic-brittle (tensile `sigma_b`, shear `tau_b`) | `bond_stiffness`, `bond_shear_stiffness`, `bond_tensile_strength`, `bond_shear_strength`, `freeze_bond_fraction` |
| unbonded contact | penalty normal `k_c delta - c_n v_n` (clipped at zero) + regularised Coulomb tangential `mu F_n tanh(v_t/v_eps)` | `contact_stiffness`, `contact_damping`, `friction_coefficient` |
| pipe (rigid cylinder) | same penalty law with ice-steel friction | `pipe_friction_coefficient` |
| seabed | Mohr-Coulomb subgrade reaction on particles below the mudline (the keel gouging its way down to the pipe) | `MohrCoulombSoil` |
| water | buoyancy only (no pore fluid inside the DEM) | `ice_density`, `water_density` |

The pipeline itself has one vertical degree of freedom driven by the
equilibrium `m_p w'' = F_ice - R_soil(w, wdot)`, i.e. the ice load pushes the
pipe into the Mohr-Coulomb subgrade. `R_soil` is a peak-to-residual spring
(`SofteningVerticalSpring`): hyperbolic Mohr-Coulomb reaction up to
`0.9 p_ult`, then linear softening to `0.8` of the peak over `0.5 D`, then a
residual plateau. The peak/residual branch is what pipe-soil practice
(DNV-RP-F114 style peak and residual resistance) and ice load-displacement
practice (ISO 19906 style peak-then-softening) both require; the hyperbolic law
alone is monotone and can never produce it.

Ultimate bearing capacity uses Vesic (1973) factors for drained conditions and
Skempton (1951) embedment factors for the undrained total-stress case:
`N_q = exp(pi tan phi) tan^2(45 + phi/2)`, `N_c = (N_q - 1)/tan phi`,
`N_gamma = 2 (N_q + 1) tan phi`, `N_c^undrained = 5.14 (1 + 0.2 z/B)` capped at
9. Pore pressure enters as an effective unit weight reduced by `(1 - r_u)`; it
is not a coupled seepage solution.

## 2. From DEM load to pipe stress

The DEM produces a load per metre of ridge. The pipeline samples that load over
the axial length where the keel is deeper than the pipe crown
(`contact_length`), distributed either uniformly (conservative for bending) or
as a cosine window. The line load is normalised so that its integral over the
footprint equals the DEM load per metre of ridge. The pipe is then solved as a
Euler-Bernoulli beam on a Winkler foundation with finite elements, consistent
Hermitian soil stiffness, and semi-infinite end springs, against the closed-form
Hetenyi (1946) infinite-beam solutions (point load and patch load) as the
independent reference. Bending moment `M = -EI w''`, extreme-fibre stress
`sigma = M / W_el`, and the DNV-OS-F101 load-controlled interaction
`U = M/(alpha_c M_p) + (N/(alpha_c N_p))^2` with
`M_p = f_y (D-t)^2 t` and `N_p = f_y pi (D-t) t`.

## 3. Verification and evidence grades

| criterion | how it is checked | evidence |
|---|---|---|
| (a) load-displacement shape | machine metrics on the coupled record: peak is positive, reached strictly before 90 % of the record, post-peak softening `> 10 %`, pre-peak branch predominantly rising | REAL (computed from the DEM record) |
| (b) ultimate capacity vs empirical/limit formula | DEM capacity compared with a Vesic limit-equilibrium squeeze model fed by a DEM-measured biaxial macro strength; a literature-parameter bracket is also reported | DERIVED for the limit analysis, WITHHELD for the Cross / DNV-OS-F101 empirical ice-ridge formulas (primary text not available in the worktree, therefore not reproduced) |
| (c) pipe stress vs Winkler theory | FE beam against closed-form Hetenyi point and patch solutions, and the coupled run's own moment against the same closed form | REAL, agreement at the 1 % level |
| (d) parameter sensitivity | keel depth, keel width swept with the DEM; soil strength swept analytically on the subgrade envelope | REAL |
| (e) test suite | `tests/test_ridge_grounding.py` | REAL |
| (f) complete case | Beaufort-like case with curve, stress distribution and field cloud | REAL |

## 4. Known limitations (honest list)

1. **Loading rate.** The platen runs at `v = 0.15-0.6 m/s` because a genuinely
   quasi-static DEM (strain rate `< 0.01 1/s`) would need `10^6` steps at this
   resolution. The reported capacity is therefore rate-influenced; the
   `rate_study.json` artifact quantifies it and the report states the
   extrapolated limit.
2. **No rotation.** Disks carry no rotational degree of freedom, so rolling
   resistance is absent and the macro friction angle is *measured* by
   `calibrate_rubble_strength` instead of assumed. The biaxial servo cell
   currently holds its confining pressure within a few tens of percent; the
   derived `(c, phi)` are reported but not used as the primary anchor.
3. **Plane strain, unit thickness.** Loads are per metre of ridge and per metre
   of pipeline. Three-dimensional keel geometry (keel depth varying along the
   ridge) is not represented, so a purely lateral loading case degenerates to a
   plateau instead of a peak.
4. **Water is buoyancy only.** No pore fluid, no hydrodynamic drag from the
   water column, no added mass; the environmental drag coefficient in the DEM
   is a numerical damper.
5. **Monodisperse disks** and a perfect triangular lattice with random thinning;
   no poly-dispersity, no block-size distribution for the rubble.
6. **Subgrade modulus** is a chosen elastic model (`0.65 E_s/((1-nu^2) B)`), not
   a derived value. Capacities are the verified quantities; displacements scale
   with the chosen modulus.
7. **Freeze-bond parameters** (100 kPa tensile, 200 kPa shear, implied failure
   strain 1 %) are engineering choices within the range quoted for
   freeze-bonded rubble; they are not calibrated against ridge-keel tests.
8. **No pipeline buckling, cyclic degradation, gapping or trench geometry.**

## 5. Reproducing

```bash
python scripts/run_ice_ridge_grounding.py            # ~10 min on one CPU core
python scripts/make_ridge_grounding_figures.py       # figures + dem_field.json
python -m pytest tests/test_ridge_grounding.py -q
```

## 6. Where the numbers live

| artifact | content |
|---|---|
| `artifacts/ridge_grounding/main_case.json` | coupled load-displacement record, shape metrics, subgrade parameters, pipe profile, quasi-static ratio |
| `artifacts/ridge_grounding/main_case_history.csv` | the same record as CSV |
| `artifacts/ridge_grounding/rate_study.json` | peak load at three platen velocities plus the linear extrapolation to zero rate |
| `artifacts/ridge_grounding/sensitivity.json` | keel depth, keel width and seabed strength sweeps |
| `artifacts/ridge_grounding/sensitivity_rigid.json` | ice-side (rigid pipeline) keel-depth study |
| `artifacts/ridge_grounding/calibration.json` | biaxial micro-to-macro strength calibration and its servo error |
| `artifacts/ridge_grounding/beam_verification.json` | beam-on-Winkler FE against the closed-form Hetenyi point and patch solutions |
| `artifacts/ridge_grounding/envelope.json` | capacity envelope with the governing mechanism and DNV-OS-F101 utilisation |
| `artifacts/ridge_grounding/ledger.json` | machine-checkable criterion ledger with evidence grades |
| `artifacts/ridge_grounding/figures/*.png` | load-displacement, pipe stress, DEM force cloud, envelope, sensitivity, calibration |

## 7. Outcome of the campaign criteria (recorded honestly)

The values quoted in this section are transcribed from the committed artifacts
listed above in section 6 (`main_case.json`, `beam_verification.json`,
`sensitivity.json`, `sensitivity_rigid.json`, `envelope.json`); the campaign
report regenerates every one of them programmatically with
`scripts/build_ridge_report.py`, so the artifacts are authoritative if the two
ever disagree.

* (a) load-displacement peak-then-softening: **satisfied** by the machine metrics
  (peak reached strictly before the end of the record, 15.5 % post-peak
  softening, predominantly rising pre-peak branch). The softening mechanism is
  rubble crushing plus the peak-to-residual subgrade branch; a pure ice
  indentation test of the same material was not run.
* (b) ultimate capacity against an empirical/limit formula: **not satisfied**
  against a Vesic bearing-capacity squeeze (~97 % deviation). The reason is
  physical and is stated in the campaign report: the keel is bounded by free
  water on both flanks, so the confined deep-bearing mechanism does not exist and
  the rubble escapes by squeeze flow. Against a plastic indentation-pressure
  anchor (`N_p c`, `N_p = 2-5`) the DEM contact pressure is within 16 % of the
  bracket mid value. The Cross and DNV empirical ice-ridge formulas are
  **WITHHELD** because their primary text was not available here.
* (c) pipe stress against Winkler theory: **satisfied** (0.55 % moment, 0.11 %
  deflection against the closed-form solution).
* (d) sensitivity: the coupled system is subgrade-governed, so keel depth and
  width barely change the system capacity (flat within ±1 %); the seabed
  strength scales it monotonically (73.8 / 221.5 / 442.9 kN/m for
  s_u = 10 / 30 / 60 kPa). The rigid-pipe ice-side study gives 1.45 / 1.41 /
  1.30 MN/m at 8 / 12 / 16 m keel depth for a fixed stroke, i.e. the ice side is
  5-7 times stronger than the seabed and is still hardening at the end of the
  stroke, so it is a stroke-dependent resistance rather than a converged
  capacity.
* (e) `pytest tests/test_ridge_grounding.py` — 24 tests, all green.
* (f) complete Beaufort-like case with curve, stress distribution and DEM field
  cloud — produced.
