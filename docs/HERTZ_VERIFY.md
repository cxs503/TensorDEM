# Hertz contact contract re-verification (TensorDEM, selfstart tp-doc-3 / m6518)

Supersedes the previous contents of this file. The earlier version quoted six result
numbers (force error `0.0012`, restitution error `0.0000`, dissipated energy `1.2e-11`,
"monotone convergence yes") with **no script, no artifact, no test and no path token**
behind any of them. That is why the repository-wide pointer audit
(`docs/dem3d_doc_evidence_pointer_audit.md`) never flagged it: the audit resolves
pointers that exist, so a document written with zero pointers is invisible to it.
Every number below is produced by `scripts/verify_hertz_contract.py` into
`artifacts/hertz_contract/verification.json`; nothing is transcribed by hand.

## 1. Case

| parameter | value |
|---|---|
| material | E = 70 GPa, nu = 0.23 (the constants the old document used) |
| particle | R = 0.01 m, rho = 2500 kg/m3, m = 1.047198e-02 kg |
| pair law | `contact_model="hertz"`, `contact_damping=0`, e = 1 (static) |
| reduced constants (module) | E* = 3.695492e+10 Pa, R* = 5.000e-03 m (sphere-sphere) |
| rigid constants (module) | E* = 7.390983e+10 Pa, R* = 1.000e-02 m |

## 2. Method

* **Q1 static law**: a single contact pair `(0,1)` carved out of a `2x2x2` `IceDEM3D`
  box, bonds severed, the six other particles parked 10 m away (a default lattice
  overlaps, and a third body contaminates the measured pair force - this was a bug in
  the first version of the harness, found and fixed before any number was kept).
  Particle 0 is left free so its force is not masked; the pair separation is prescribed
  and `IceDEM3D.forces()` is read once, no stepping.
* **Q2 rigid constants**: `hertz_reduced_modulus(..., rigid_counterpart=True)` and
  `hertz_effective_radius(..., rigid_counterpart=True)` against the closed forms
  `E/(1-nu^2)` and `R`.
* **Q3 restitution**: the shipped pair law is
  `F = max(k_s(delta) delta - 2 zeta sqrt(k_s(delta) m_eff) v_n, 0)` with
  `zeta = -ln e / sqrt(pi^2 + ln^2 e)` (the **linear** dashpot mapping) and
  `v_n` the separation rate. Because `k_s` follows the overlap, the law is not the
  linear oscillator the mapping was derived for, so the realised `e` is not the
  requested `e`. Quantified by RK4 integration of that law (step = period/400 at the
  module's own `delta_ref = 0.1 R` secant stiffness) and cross-checked on the solver.
  A first version integrated `v_n = +v` (penetration rate) and produced `e > 1`; at
  that sign the dashpot pumps energy during restitution. The sign error was mine, is
  documented here because it is a live trap, and is fixed: with `v_n = -v` the law is
  dissipative as shipped.

## 3. Results

### 3.1 Static force-indentation law (Q1)

| delta/R | F measured [N] | F Hertz closed form [N] | measured/closed form |
|---|---|---|---|
| 0.01 | 4.9273e+03 | 3.4841e+03 | 1.414214 |
| 0.05 | 5.5089e+04 | 3.8954e+04 | 1.414214 |
| 0.10 | 1.5582e+05 | 1.1018e+05 | 1.414214 |
| 0.20 | 4.4071e+05 | 3.1163e+05 | 1.414214 |

The overlap exponent is right (`F proportional to delta^1.5` to the last digit) and the
**magnitude is 41.42 % high at every overlap**. The ratio is exactly `sqrt(2)` =
`sqrt(R / R*)`: the shipped pair magnitude behaves as if it used the rigid (plane)
effective radius `R` while keeping the sphere-sphere modulus `E*`. Reproduced with a
direct in-process probe (`_hertz_pair_params` = (3.6955e+10, 5.0e-03), single contact
pair `[0,1]`, ratio 1.4142135623730971). **Status: open finding**, not fixed here - the
localisation needs one more pass through the pinned-pair branch, and a fix would change
every Hertz trajectory, so it is filed rather than quietly patched.
`tests/test_hertz_contract_verification.py` locks the *observed* ratio so the fix is
visible when it lands.

### 3.2 Rigid (plane) constants (Q2)

`E* = 7.390983e+10 Pa` vs `E/(1-nu^2) = 7.390983e+10 Pa` (relative error 0.0);
`R* = 1.000e-02 m` vs `R` (0.0). The previous document's limitation notes ("sphere-sphere
requires modified E*, R*") are correct: the sphere-sphere constants are exactly half of
the rigid ones in both factors.

### 3.3 Restitution mapping (Q3)

| requested e | realised e (RK4, v0 = 0.05 m/s) | realised e (RK4, v0 = 0.5 m/s) | error vs requested |
|---|---|---|---|
| 0.90 | 0.911744 | 0.911780 | **+1.30 %** |
| 0.70 | 0.742299 | 0.742398 | **+6.04 %** |
| 0.50 | 0.582397 | 0.582544 | **+16.48 %** |

* The solver reproduces the RK4 reference: `e = 0.9117199` vs `0.911780` (6.0e-05,
  0.007 %), i.e. the bias is a property of the law, not of the integrator.
* The bias is **velocity independent to 0.004 %** between 0.05 and 0.5 m/s: with
  `gamma proportional to sqrt(k_s(delta))` the law is amplitude self-similar, so the
  error is a pure *mapping* error `zeta -> e`, not a step-size or velocity artefact.
* Consequences: the `<3 %` bar is met only for `e >= 0.85`-ish; at `e = 0.5` the
  requested value is missed by **16.5 %**, so the old document's
  `error = 0.0000, PASS` line was wrong by more than an order of magnitude in the
  useful range. No energy-creation claim is made or needed: the law is dissipative at
  the shipped sign (Q3 above).
* Applying the linear closed form to a Hertz law is the defect class. A corrected
  inversion is a one-dimensional root find on this same reference and is **not**
  included here (it changes trajectories); filed as follow-up.

## 4. Notes / limitations

* Static Q1 covers the particle-particle branch only; the tool/platen branch uses the
  rigid constants of §3.2, which are verified as constants but not as a force path.
* Q3 covers a head-on binary impact, both free (`m_eff = m/2`), zero friction. Tangential
  and rolling branches are out of scope.
* The previous document's "convergence study (coarse dt = 1e-6, fine dt = 5e-7)" is
  dropped, not re-run: for this configuration the solver's own `recommended_dt` is
  3.78e-07 s, i.e. both "levels" of that study were **above** the stability bound the
  module itself recommends, so its "fine" column cannot be a converged reference.
* Honest FAIL carried forward: the 41.4 % static-force offset (§3.1) and the >5 %
  restitution mapping bias at `e <= 0.7` (§3.3) are open. Neither is papered over.

## 5. Reproduction

```bash
PYTHONPATH=src python scripts/verify_hertz_contract.py     # writes the JSON + figure
PYTHONPATH=src python -m pytest tests/test_hertz_contract_verification.py -q
```
Figure: `docs/figures/hertz_contract.png` (left: measured vs closed-form F(delta);
right: realised vs requested e).
