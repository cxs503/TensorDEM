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
