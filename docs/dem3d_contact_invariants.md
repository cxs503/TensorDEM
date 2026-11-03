# DEM3D contact-kernel invariance audit (metamorphic verification)

Status: **kernel invariants PASS, one real defect found and fixed** (monitor purity).
Modules: `tensordem.contact_invariants`, `tensordem.dem3d`.
Scripts: `scripts/audit_dem3d_contact_invariants.py`.
Tests: `tests/test_dem3d_contact_invariants.py`.
Artifacts: `results-dem3d-contact-invariants/` (`audit_report.json`, `audit_report.md`,
`invariant_results.csv`, `invariant_summary.csv`, `contact_invariants.png`).

## 1. Why this audit exists

Every DEM number produced in this repository passes through one routine,
`IceDEM3D.forces()`. The constitutive pieces are each verified separately —
Hertz law (`docs/dem3d_hertz_contact_verification.md`), restitution
(`docs/dem3d_contact_restitution.md`), Coulomb cone
(`docs/dem3d_coulomb_friction_cone_verification.md`), rolling resistance
(`docs/dem3d_rolling_resistance_verification.md`), energy ledger
(`docs/dem3d_energy_ledger_analytic_verification.md`). What was *not* audited is
the **bookkeeping around those laws**: cell-list neighbour detection, the
bond/contact mutual-exclusion mask, `index_add_` accumulation, pair-order and
label independence, and whether a *probe* is side-effect free.

Bookkeeping defects are the worst class in a research code: they produce finite,
plausible, smooth, wrong results, and no single-case test catches them because
each case can be reproduced by the same broken code path. This audit attacks
them with properties that must hold for **any** constitutive law.

## 2. Method

Randomised lattices (`nx=4, ny=4, nz=3`, 48 particles, 326 bonds, radius 25 mm,
simple-cubic plus diagonals) are displaced by a uniform random perturbation of
`0.5 R` (intact lattice) or `1.2 R` (heavily packed) and given random velocities
of ±0.05 m/s from a private `torch.Generator`. Four seeds `{0,1,2,3}` × five
cases = 132 invariance samples.

The configuration is deliberately *closed*: `drag=0`, no fixed edges, no platens,
and the indenter parked 5 m above the specimen so its branch contributes
identically zero. Each case is instantiated twice — once intact, once
**pre-fractured** (one committed fracture pass, then the resulting `alive` mask
is replayed into every fresh instance). The pre-fracture step is essential: in an
intact perturbed lattice *every* overlapping pair is still bonded, the contact
law is masked to zero, and a brute-force comparison would silently test only the
bond term (the audit reports `unbonded_overlaps = 0` for the intact case, which
is exactly how this trap was found — see §5).

| invariant | statement |
| --- | --- |
| `momentum_closure` | closed system ⇒ `Σ F_i = 0` and tool reaction `= 0` (Newton III) |
| `translation` | `F(x + t) = F(x)` for a rigid offset `t` |
| `rotation` | `F(Rx, Rv) = R F(x, v)`, `R` a random proper rotation from QR of a Gaussian |
| `brute_force` | engine force equals an independent `O(N²)` double loop over all pairs, skipping live bonds |
| `determinism` | two fresh instances give bitwise identical forces; a probe on a reused instance is bitwise identical |
| `load_superposition` | `F(loads) − F(0) = loads` exactly |
| `reciprocity` | `K = ∂F/∂x` symmetric by central differences (`h = 10⁻⁶ R`) on the elastic path |

The brute-force reference shares no code with the solver's spatial hashing: it is
a naive double loop implementing the documented pair law
(`linear: F = k δ`, `hertz: F = (4/3) E* √R* δ^{3/2}`) plus the documented bond
law, with velocities zeroed so the dashpots are idle.

## 3. Results (4 seeds × 5 cases = 132 samples)

Worst relative residual per invariant, versus the declared tolerance
(regenerate with `python scripts/audit_dem3d_contact_invariants.py --figure`):

| invariant | worst residual | tolerance | verdict |
| --- | --- | --- | --- |
| `momentum_closure` | 6.0e-16 | 1e-12 | PASS |
| `translation` | 7.5e-14 | 1e-12 | PASS |
| `rotation` | 1.7e-15 | 1e-12 | PASS |
| `brute_force` | 1.6e-16 | 1e-12 | PASS |
| `determinism` | 0.0 (bitwise) | 0 | PASS |
| `load_superposition` | 2.4e-16 | 1e-12 | PASS |
| `reciprocity` | 2.4e-10 | 1e-8 | PASS |

Coverage evidence (so the audit cannot be vacuous): the fractured cases carry
`broken_bonds = 326/326` with `unbonded_overlaps = 40` genuine contact pairs
(`contact_scale = 175 N` linear, `4.67e5 N` Hertz), the intact case carries
`bonded_pairs = 326` with `contact_scale = 0` by construction, and the tangent
stiffness scale is `1.1e4 – 1.1e5 N/m`. The Hertz case stresses the kernel at a
contact magnitude four orders larger than the linear case, and the rotation
determinant is `1 − 2e-16`, i.e. the test rotation is proper.

Figure `contact_invariants.png`: left panel — every invariant/case residual on a
log axis against its tolerance (blue = pass); right panel — monitor-purity
divergences, all at the float64 noise floor after the fix.

## 4. Defect found and fixed: a probe changed the friction history

`IceDEM3D.diagnostics()` evaluates forces to report the reaction. On the
frictional path, `forces()` also integrated the Cundall–Strack tangential
displacement (`tang_disp += v_tangent · dt`) and *stored* it. So every diagnostic
call inserted one extra slip increment into the friction history — measuring a
run changed it. The CLI and every plotting script call `diagnostics()` once per
output interval, so this was on the default usage path.

Measured on `linear_friction_fractured` (μ = 0.4, 60 steps of
`dt = 3.4e-5 s`), pre-fix: **max ‖Δx‖ = 1.53e-4 m** (0.6 % of the particle
radius) between a monitored and an unmonitored twin, and a probe-level force
discrepancy of `1.7e-4` relative (the `determinism` invariant was the trigger).
Post-fix both are **exactly 0.0**.

Fix (minimal, default-preserving): `forces(..., update_history=True)` keyword;
the tangential increment, the tangential state write-back and the diagnostic
rolling-torque accumulator are guarded by it; `diagnostics()` passes
`update_history=False`. Step semantics are unchanged, so every existing result is
bit-for-bit reproducible — the fix only stops *probes* from perturbing state.
Regression: `tests/test_dem3d_contact_invariants.py::MonitorPurityTests` asserts
zero divergence, an untouched tangential state under probing, *and* that the
step-owner path still integrates slip (the fix must not silently disable
friction). `tests/test_dem3d.py`, `test_contact_tangential.py`,
`test_dem3d_pair_contact_damping.py`, `test_dem3d_energy_audit.py`,
`test_dem3d_hertz_contact.py`, `test_dem3d_external_load_balance.py`: 75 passed.

## 5. Notes, traps, limits

* **Vacuous-audit trap.** The first version of the brute-force comparison passed
  with residual 0.0 — because all contacts were masked by live bonds. The
  invariant now reports `unbonded_overlaps` and the tests assert it is `> 10` for
  fractured cases and `= 0` for the intact one. Any future edit that stops
  exercising contacts fails the suite instead of passing silently.
* **Friction history is stateful by design.** Calling `forces()` twice with
  `update_history=True` legitimately adds two slip increments; `determinism` is
  therefore asserted with probe semantics (`update_history=False`) for the reused
  instance, and with bitwise equality for two independent fresh instances.
* **Stale history keys.** `_tangential_state` is keyed by particle pair and never
  pruned when a bond breaks or particles separate, so memory grows with the
  number of pairs that ever touched and a re-contacting pair resumes its old slip.
  Not fixed here (it changes friction physics); recorded as a known limitation.
  The rolling-torque accumulator is diagnostic-only and never reset by a step,
  so it is a running sum; probes no longer inflate it.
* **Scope.** Only particle–particle bond/contact terms are covered. Tool, platen
  and load branches are covered only by `load_superposition` and by the
  zero-reaction requirement. The brute-force reference does not reproduce the
  history-dependent tangential spring, so friction cases are audited by the
  closure/symmetry invariants only. `reciprocity` is meaningful only for the
  hyperelastic path and is skipped for dissipative cases.
* All fracture-pass cases end with a fully broken topology (`326/326` bonds), so
  mixed bonded/unbonded topologies are covered only indirectly (intact = fully
  bonded, fractured = fully unbonded). A mixed case is the natural next increment.

## 6. Error verdict (explicit)

The audited quantities are **floating-point identities, not physical
predictions**, so the platform's 15 % engineering band does not apply and there
is no experimental reference value to compare against. The declared tolerances
are instead:

* `1e-12` relative for the exact identities — measured worst case `7.5e-14`
  (translation), i.e. **~13 orders of magnitude inside tolerance**;
* `1e-8` relative for the finite-difference reciprocity check — measured
  `2.4e-10`, **~40× inside tolerance**;
* `0` for determinism — measured exactly `0.0` after the fix.

Answer to "is the key quantity within 3 %?": **yes, by ~13 orders of magnitude**,
but the honest reading is that this table certifies *kernel bookkeeping*, not
ice-physics accuracy. The monitor-purity defect was a real physics error
(0.6 % of a particle radius after 60 steps, growing in time); it is now exactly
zero.