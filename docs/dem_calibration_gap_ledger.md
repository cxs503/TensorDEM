# DEM calibration gap ledger (TensorDEM side)

Living write-back of the DEM rows of the m605 capability inventory / gap matrix
(report id 542, `docs/fem_dem_capability_inventory.md` in that worktree) to this
repository, with the closing evidence for each row. Rows are only closed when a
verification surface and a test file exist here; everything else stays open with
the reason.

Ledger keys follow the m605 gap ids so the two documents can be diffed.

| id | m605 state | now | evidence (this repo) | open part |
|---|---|---|---|---|
| **GAP-DEM-G4** tangential friction / rolling resistance | 🟠 high, "no tangential friction / rolling resistance" | **closed for sphere/plane, clump/plane, clump/sphere and clump/clump** | sliding spring: `src/tensordem/contact_tangential.py`, `docs/dem3d_tangential_friction_verification.md`; rolling law (sphere): `src/tensordem/rolling_resistance.py`, `docs/dem3d_rolling_resistance.md`; **rolling + drilling on the clump path: `src/tensordem/clump_rolling.py` + `docs/clump_rolling_resistance_calibration.md` (mission m5052)** | **clump–clump: `src/tensordem/clump_clump.py` + `docs/clump_clump_contact.md` (mission m5186)** — Coulomb tangential spring on merged Hertz-Mindlin junctions; drilling resistance still has no elastic branch and rolling resistance is not defined for a clump pair |
| **GAP-DEM-G6** cell-list / parallel performance | 🟡 medium, "GPU cell list synchronous blocking" | **open** | scale benchmark exists (`docs/dem3d_scale_benchmark.md`, `results-dem3d-scale/`) | no GPU/domain decomposition; not addressed by m5052 |
| **GAP-DEM-G7** large-scale stress test (> 10k particles) | 🟡 medium, "no large-scale test" | **open (partially superseded)** | 10k/50k/100k wall-clock study in `docs/dem3d_scale_benchmark.md` | the 100k case is a wall-clock study, not a physics validation; throughput on the 702 cluster not yet recorded |
| **GAP-DEM-G1** 3-D DEM ↔ fluid two-way coupling | 🔴 blocking | open | 2-D only: `src/tensordem/calibrated_wet_coupling.py`, `docs/calibrated-wet/` | 3-D coupling not implemented |
| **GAP-DEM-G2** buoyancy / wet coupling (3-D) | 🔴 blocking | partially closed | `src/tensordem/buoyancy_wet3d.py`, `docs/dem3d_buoyancy_float.md` | not re-audited by m5052 |
| **GAP-DEM-G3** SUBOFF curved-hull contact | 🔴 blocking | partially closed | `src/tensordem/hull_mesh_contact.py` | curved-hull validation on a real hull not re-audited by m5052 |

## m5052 write-back (clump rolling resistance)

* **Row changed**: `docs/clump_multisphere_particles.md`, table row
  `rolling resistance / torsion friction for clumps`: `NOT_SUPPORTED` →
  **REAL** on the clump–plane and clump–sphere paths
  (`ContactConfig.rolling_friction`, `ContactConfig.drilling_friction`,
  `ContactConfig.rolling_model`, `resistance_moment`).
* **Verification surface**: 16 tests in `tests/test_clump_rolling_resistance.py`
  plus the calibration harness `scripts/m5052_clump_rolling_calibration.py`
  (numbers in `artifacts/m5052/clump_rolling_calibration.json`).
* **Quantified calibration**: tilt threshold `tan(theta_c) = min(mu_s, (a + mu_r r)/h)`
  reproduced to 0.19 … 1.46 % across five `mu_r` values and four clump shapes;
  `mu_r` recovered from the measured angle to ≤ 1.47 %; drilling spin-down exact
  to round-off; no energy creation, stored spring energy inside its analytic
  bound.
* **Applicability stated, not implied**: for equal-radius clumps in a flat pose
  `h = r` and `a/h ≥ 1`, so a tilt test on lumpy clumps measures **sliding**
  friction (41.99° for `mu_s = 0.9`, independent of `mu_r`), not the rolling
  coefficient. `mu_r` is identifiable only for near-spherical supports or from
  dynamic (repose / drum) tests, which remain open.
* **Withheld**: no digitised laboratory tilt-table or repose measurement is in
  this repository, so the *laboratory* calibration of `mu_r` is flagged
  **WITHHELD**; only the protocol and its inversion are verified.
* **Honest FAIL carried forward**: the viscous branch at its default
  `omega_ref = 1e-3` is numerically unstable at `dt = 2e-4` (its measured
  "threshold" of 5.74° is an artefact and is not claimed). Stability requires
  `dt < I omega_ref / M_lim`; the calibration uses `omega_ref = 0.1` for that
  branch.
* **Not addressed by m5052**: GAP-DEM-G6, GAP-DEM-G7 (scale/performance).
  Clump–clump contacts were closed separately in m5186
  (`docs/clump_clump_contact.md`).

## Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_clump_rolling_resistance.py -q
PYTHONPATH=src python scripts/m5052_clump_rolling_calibration.py --workers 8
PYTHONPATH=src python scripts/m5052_clump_rolling_calibration.py --extra-only
PYTHONPATH=src python scripts/m5052_clump_rolling_calibration.py --figures-only
```
## selfstart write-back (Maxwell-Wiechert bond law, `ice_bond_viscoelastic`)

* **New row (was missing from the ledger entirely):** *constitutive rate
  dependence of the 3-D DEM bond* — m605 state **not listed**; now
  **partially closed at the constitutive level, open at the specimen level**.
  Evidence of this repo: `src/tensordem/ice_bond_viscoelastic.py`,
  `tests/test_ice_bond_viscoelastic.py` (23 cases),
  `scripts/verify_ice_bond_viscoelastic.py`,
  `docs/ice_bond_viscoelastic_verification.md`,
  `artifacts/ice_bond_viscoelastic/verification.json`.
* **Verified against closed-form references** (numbers injected
  programmatically from the JSON): Prony relaxation `4.0e-13` max rel. err;
  constant-strain-rate ramp `1.6e-13`; increment-splitting invariance
  `1.6e-14`; Kachanov increment `1.0e-14`; emergent ductile exponent
  `sigma_f ~ edot^(1/(m+1))` recovered to `<1 %` for `m = 1,2,4` without any
  envelope in the code; parallel-Wiechert creep compliance to `4.5e-4` at
  `dt = tau/500` with first-order convergence; Weibull median `4.7e-4`.
* **Two defects found and fixed:**
  1. `from_unrelaxed_stiffness()` raised on its own defaults (shares
     `0.45 + 0.20 = 0.65 != 1`); the factory now normalises the shares, which
     reproduces the constants-as-fractions-of-`K` reading exactly
     (`k0 = 0.35 K`, branches `0.45 K / 0.20 K`).
  2. the dissipation integral formed `1 - u^2` by subtraction, which is
     cancellation-limited for `dt << tau`; forming it as `(1-u)(1+u)` drops the
     measured ledger residual from `6.6e-8` to `1.5e-12` in that regime.
* **Honest limit, recorded not hidden:** the ledger identity is
  cancellation-limited at `~1e-7` relative for `dt/tau ~ 1e-6` with large
  strain increments (float64 cannot resolve the difference of `O(1e8)` larger
  terms). The `<1e-10` figure is the `dt/tau >= 3e-4` regime; the module
  docstring was corrected accordingly.
* **Open part (not addressed here):** no engine call site — `dem3d.py` still
  uses the linear bond, damage is not fed back into the bond force, and no
  bonded-specimen rate-dependence benchmark exists. Closing that needs a
  bond-law hook in `IceDEM3D` plus a specimen campaign; it is the next node for
  this trajectory.

## selfstart write-back (rate-dependent bond wired into the engine, mission m6464)

* **Row changed:** the "constitutive rate dependence of the 3-D DEM bond" row
  moves from *"partially closed at the constitutive level, open at the specimen
  level"* to **engine closed / specimen benchmark open with a measured negative
  result**. Evidence: `src/tensordem/dem3d.py` (`bond_model`,
  `bond_creep_exponent`), `tests/test_ice_bond_engine.py` (20 cases),
  `scripts/verify_ice_bond_specimen_rate.py`,
  `artifacts/ice_bond_specimen/verification.json`,
  `docs/ice_bond_specimen_rate_dependence.md`.
* **Closed:** the engine call site now exists (`IceDEM3D` evaluates the Prony
  law per bond, advances the branches in closed form, records the dashpot
  dissipation in the audited ledger, and runs the Kachanov clock with failure
  mode bit 4). The legacy linear bond is bit-for-bit unchanged (asserted) and
  legacy checkpoints still load.
* **Measured, not claimed:** at 0.25–4 s^-1 a 4x4x3 opposed-platen specimen
  gives fitted strengths σ_f ∝ ε̇^0.244 (linear+creep) and ε̇^0.198
  (Maxwell-Wiechert+creep) versus the analytic 1/3, **but the negative control
  (pure linear bond) is not flat either** (slope +0.117, 38 % spread) and the
  measured strength is 7–9x the rigid-plate bound: the specimen is
  inertia-dominated in this rate window. The specimen-level exponent is
  therefore **not verified** here — an honest FAIL, recorded as such.
* **Next node:** isolate the constitutive response with a single-bond or
  single-chain driver, or move the rate window 2–3 decades slower with a
  quasi-static servo so the inertial term no longer dominates; only then can
  the emergent exponent be compared to 1/(m+1).

## selfstart write-back (single-bond engine driver, m6543)

* **Row changed:** the "constitutive rate dependence of the 3-D DEM bond" row
  moves from *"engine closed / specimen benchmark open with a measured
  negative result"* to **engine closed at the single-bond level / specimen
  benchmark still open**. Evidence: `scripts/verify_ice_bond_single_bond_engine.py`,
  `tests/test_ice_bond_single_bond_engine.py` (5 cases),
  `artifacts/ice_bond_single_bond/verification.json`,
  `docs/ice_bond_single_bond_engine_verification.md`.
* **What was verified:** a 2-particle / 1-bond assembly built from the real
  `IceDEM3D` engine (2×2×2 grid reduced to one bond), driven kinematically at
  constant strain rate (no inertia), so the *engine's* call sites
  (`_bond_force` history advance + `_commit_fracture` + `_commit_creep_damage`
  with the engine's `F_ref`) are the ones under test.
* **Quantified:** the linear+creep arm (m = 2) reproduces the analytic ductile
  exponent `1/(m+1) = 1/3` to **0.075 %** (fitted 0.3336), and every creep-window
  point sits on the closed-form creep force to < 0.1 %; the linear arm is
  rate-independent to 0.02 % (negative control); the creep/brittle crossover
  is between 0.3 and 1.0 s^-1, consistent with the analytic `edot_c = 1.0 s^-1`.
* **Measured, not claimed:** the maxwell_wiechert+creep arm's creep-window
  exponent is 0.3678 (10.3 % above 1/3) because the relaxation spectrum adds
  rate stiffening on top of the creep damage; the MW force is not a linear
  ramp, so there is no strict closed-form reference for that arm.
* **Honest limit:** this is the *single-bond* level. A bonded *specimen*
  (many bonds, load redistribution, inertia) has not yet been shown to follow
  the envelope; that needs a quasi-static servo at 2-3 decades slower rates or
  a single-chain (weakest-link) driver. Both are the next node.

## selfstart write-back (quasi-static bonded-column campaign executed, m7571)

* **Row changed:** the "constitutive rate dependence of the 3-D DEM bond" row
  moves from *"engine closed at the single-bond level / specimen benchmark
  still open"* to **engine closed at the single-bond and quasi-static
  bonded-column level / specimen benchmark (load redistribution) still
  open**. The m6541 driver (`tensordem.bond_quasistatic`,
  `scripts/verify_bond_quasistatic_rate.py`, `tests/test_bond_quasistatic.py`)
  had been merged **without** its verification surface in this repo (no
  artifacts, no doc, no ledger row); m7571 ran the full campaign and closed
  that verification debt.
* **Verification surface now in this repo:** `artifacts/ice_bond_quasistatic/`
  (`verification.json` + 5 figures), `docs/ice_bond_quasistatic_rate_verification.md`
  (rendered from the JSON by `scripts/render_ice_bond_quasistatic_doc.py`,
  zero hand-copied numbers), 17 tests in `tests/test_bond_quasistatic.py`
  (all green).
* **Quantified (2 x 2 x 2 column, `n_layers = 2`, no free particle):**
  brittle control rate-independent to 0.067 % (the control that failed in
  m6464); ductile m = 2 fitted slope 0.333333 vs analytic 1/3, max strength
  error 0.033 %; exponent sweep m = 1 / 4 gives 0.5000 / 0.2000 vs 1/2 / 1/5
  (0.000 %); Maxwell-Wiechert engine matches the independent Prony reference
  to 0.009 % (its 10.3 % slope offset above 1/3 is a *constitutive* effect
  of the relaxation spectrum -- the reference fit is 0.3680 too); damage
  clock 2.17th order in dt (first order or better).
* **Measured, not claimed:** the free-layer arms (F/G) do **not** collapse
  onto the inertia index alone (log-log correlation 0.824). The mass dial at
  fixed `n_layers = 4` holds a density-independent strength error of 6.37 %
  while KE/SE varies by a factor of ~16: the dominant deviation is the
  quasi-static strain non-uniformity of the multi-layer column (scales with
  `n_layers - 1`), inertia is a secondary modulation. The campaign figure
  caption's "collapse" wording is therefore an overstatement; the clean
  constitutive measurement is the zero-inertia, zero-non-uniformity cell
  (`n_layers = 2`), and the exponent claims rest on it.
* **Not addressed by m7571:** a bonded *specimen* with load redistribution
  across many bonds (the m6464 platen FAIL remains open at that level).
  Next node: a weakest-link chain driver (1-D series, Weibull
  weakest-link scaling `sigma_chain ~ N^(-1/m)`) or a quasi-static 2-D
  specimen with load redistribution after first bond failure.
## selfstart write-back (trickle-injection repose driver, mission m7569)
* **Row changed:** the open reproduction-gap recorded in
  `docs/dem_nonspherical_representation_verdict.md` §5.4 — *"closing [the
  clump/superquadric repose comparison] needs a trickle-injection driver with rate
  control"* — moves from **open** to **driver implemented / heap campaign open**.
  Evidence: `src/tensordem/repose_injection.py`,
  `tests/test_repose_injection.py` (13 cases, green),
  `scripts/verify_repose_injection.py`, `artifacts/repose_injection/verification.json`,
  `docs/repose_injection_driver.md`.
* **New capability, not a re-run:** a rate-controlled point-source pour with a
  Verlet broad phase that accepts any radius set (the column driver is
  monodisperse) and two independent free-surface angle estimators (radial-bin
  envelope + free-surface point fit with per-sector spread).
* **Verified, quantified:** sliding onset bracketed around the analytic
  `atan(mu) = 26.565 deg` (23.909–29.222 deg, travel 1.14e-6 vs 2.55e-4 m);
  rolling onset with `mu_r = 0.20` bracketed around `atan(mu_r) = 11.310 deg`
  (10.179–12.441 deg); ideal rolling (`mu_r = 0`) moves already at 1.33 deg, i.e.
  no spurious static threshold; the point-fit estimator recovers synthetic cones
  of 8/15/25/35 deg to 0.41/0.40/0.041/0.33 % (`r^2 = 0.989`); pair-list rebuild
  on 1000 grains = 0.180 s (880 candidates).
* **Defect found in the merged m4970 driver (reported, not patched):** the floor
  tangential history in `repose_calibration.simulate` uses the grain *centre*
  velocity and omits the spin term `omega x (-R z)` that the particle–particle
  branch of the same function includes. Measured with the same law: a grain with
  `mu = 0.5, mu_r = 0` needs 26.57 deg before it moves instead of rolling at any
  tilt, and a grain with `mu_r = 0.20` never rolls at all (2.8e-7 m travel at
  12.44 deg). The published `mu -> theta_r` table is therefore a property of that
  wall law; it should be re-measured with the spin-corrected branch before being
  quoted as a material calibration. The new driver defaults to the corrected law
  and keeps `floor_spin_term=False` to reproduce the legacy behaviour.
* **Honest gap carried forward:** the heap campaign was NOT run (a single
  90-grain case exceeded the 110 s per-call budget); **no heap repose angle is
  claimed**. `scripts/verify_repose_injection.py` carries the full matrix
  (baseline, rate 60/240, `mu_r` 0/0.10/0.30, polydispersity 0.25, `mu` 0.35,
  legacy-floor control) ready for the 702 cluster. Clump / superquadric grains are
  still not admitted by this loop (clump contacts + rigid-body integration).
* **Two implementation traps recorded:** (1) comparing the *per-step*
  displacement with `skin/2` never rebuilds the Verlet list — grains tunnel,
  overlap deeply and launch at 4–14 m/s; compare the cumulative displacement and
  force a rebuild on release. (2) Keying the outlet height to the highest grain
  (including airborne ones) ratchets the source upward and no heap forms; key it
  to the settled apex (`|v| < 0.05 m/s`).

<!-- P143W2C-N02 BEGIN -->

## P143W2C-N02 write-back (BPM micro-parameter identifiability)

| row | before | now | evidence | open part |
|---|---|---|---|---|
| m6335 "UCS 偏低 54.8 %" honest FAIL | STUB (statistics only, no engine run) | **REAL** on the bonded-specimen path: UCS = 5.45 MPa (3/3 literature intervals) | `src/tensordem/bpm_identifiability.py`, `artifacts/p143w2c_n02/numbers.json`, `tests/test_bpm_identifiability.py` | literature intervals overlap (consistency check, not independent); primary sources not digitized in-tree |
| mass-scaling independence of the macro strength | WITHHELD (never measured) | **REAL (measured)**: max drift of sigma_peak +0.00%, E +0.00% over beta = 100..3600 | `beta_scan` / `beta_drift_gate` + shard records | the measured drift is a property of this lattice and damping; the ramp (non-relaxed) protocol fails the KE gate by ~10x |
| Weibull modulus m identifiability (size effect) | MOCK (synthetic generator only) | **REAL (engine)**: extended design CI width 0.7092, m = 5.66; minimum design (3 families, n = 8) width 0.7408 -> identifiable | same files | the CI width, not the point estimate, carries the verdict; the engine-level negative control fails (a size effect exists without bond disorder, see the report §5.9) |
| friction regularization: static/dynamic dual coefficients | NOT_SUPPORTED | **REAL (opt-in)** `friction_regularization="dual"` (elastic-cone switch, no extra state) | `DEM3DConfig.static_friction_coefficient`/`dynamic_friction_coefficient`, `shear_cell_residual` (max |bias| 0.10% vs the analytic mu_d N) | specimen-level residual is carried by the surviving bond skeleton as well, so the contact-level statement is the strong one |
| positive / negative control of the identifiability pipeline | missing | **REAL**: m = 6 recovered to 0.31%; homogeneous bonds (m_bond = inf) -> identifiable (WRONG) | `positive_control`, `run_negative_control`, `tests/test_bpm_identifiability.py` | negative control uses deterministic strength, so the sentinel band is returned by construction |

<!-- P143W2C-N02 END -->




## selfstart write-back (m13798, 2026-11-11) — shear-cell macro friction

* **New ledger row (was missing):** *direct-shear macroscopic friction device* — before this
  milestone `src/tensordem/shear_cell.py` was **registered but dead code** (`mass[:, None]` on a
  float → `TypeError` at the first integration step). Now **runnable + gated, physics still open**.
  Evidence: `src/tensordem/shear_cell.py`, `tests/test_shear_cell_friction.py` (7 tests),
  `docs/GAP_DEM_FRICTION_SHEAR_CELL.md`, `artifacts/shear_cell_friction/shear_cell_diagnosis.json`.
* **Structural fix of an ill-posed measurement:** the old code fitted a Mohr-Coulomb *slope*
  inside a single run where `sigma_n` is constant and used `sum(|f_z|)` as the applied normal
  stress. The envelope is now fitted only across **distinct** stress levels; a single level (or a
  repeated one) raises `ValueError`, so a "friction angle" can no longer be manufactured from
  normal-stress noise. The fitter is analytically exact on synthetic lines (tan φ to <1e-12,
  cohesion to <1e-6 Pa).
* **Honest FAIL carried forward:** the stress-servo platen reaches the target once
  (2.09 N vs 2.048 N target) and then loses the bed: bed top drops 0.04674 → 0.03848 m while the
  platen drops only 0.05184 → 0.04764 m, so the steady-window `sigma_n` is 0 Pa and `mu` is
  undefined. The new validity gate reports `valid=False` with the reason instead of returning a
  NaN/0 friction angle. `GAP-DEM-FRICTION` ("摩擦系数 vs 实验 <10%") therefore stays **open**.
* **Audit pointer:** `scripts/run_coarse_graining_campaign.py` consumed this module's
  `friction_angle_deg`/`r_squared`; that path could never have executed before this milestone, so
  any shear-box numbers attributed to it lack evidence and should be traced.

## selfstart write-back (legacy propeller-ice milling module, IN27-S4)

* **New row (was missing from the ledger entirely):** *propeller–ice milling
  hydrodynamics / debris transport* — m605 state **not listed**; the IN34-W3-N05
  package is **REAL and tested**, the legacy IN27-S4 module
  (`src/tensordem/propeller_ice_milling.py`, 1140 lines) was **shipped untested**
  and is now **verified-and-defective**.
* **Verification surface added:** `scripts/verify_propeller_ice_milling_legacy.py`
  (protocol `tensordem-propeller-ice-milling-legacy-verification-v1`, 25 gates,
  17 PASS), `tests/test_propeller_ice_milling_legacy_verification.py` (14 green +
  2 `xfail(strict)`), `docs/propeller_ice_milling_legacy_verification.md`,
  `artifacts/propeller_ice_milling_legacy/verification.json` + 3 figures.
* **Closed by analytic reference:** planform scale vs `B(1+e,1+e)` (1.5e-5), blade
  area vs `EAR·πD²/4` (2 %), helicoidal pitch law (machine precision), BPF and
  harmonic amplitudes (<0.5 %, <2 %) with a white-noise negative control,
  efficiency identity `η = J K_T/(2πK_Q)` (machine precision), actuator-disk
  induced velocity (machine precision), debris integrator vs closed-form quadratic
  drag in the module's own carrier, and second-order self-convergence under step
  refinement.
* **Defects registered (not silently fixed):** D1 race momentum flux is `T/2`, not
  `T`, although the class claims exact momentum conservation; D2 `integrate_debris`
  uses `σ = 0.5 D` while `RaceModel.sigma()` returns `0.25 D`, so chips are
  advected by a 4× weaker jet than the torque model assumes (the IN27-S4 ejection
  band 0.5–2.0 m/s is therefore not admissible evidence); D3 the *package* BEM
  (IN34-W3-N05) saturates every strip on its induction cap `a = 0.85` for `J ≲ 0.2`
  at a 3 m geometry while reporting a 1e-8 residual, and never calls its own
  `_buhl` high-induction branch.
* **Withheld:** absolute `K_T`, `K_Q` of *both* implementations (no experimental or
  Wageningen-B style open-water table in this repository); the two implementations
  disagree by up to 72 % in `K_T` on a matched geometry, which is itself the
  argument for keeping both at WITHHELD.
* **Not addressed:** the legacy plastic-capped Hertz contact branch (inside
  `MillingSimulation`, needs a full transient case), D1/D2 repair, and the D3
  package follow-up.
