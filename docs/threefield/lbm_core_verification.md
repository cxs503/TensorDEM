# Analytic verification of the three-field LBM core (D3Q19 + Smagorinsky + Guo forcing)

Protocol `tensordem-threefield-lbm-analytic-verification-v1`.

Modules under test: `tensordem.threefield.wake_lbm` (production propeller-race
solver) and the verification harness `tensordem.threefield.lbm_verification`.
Entry point: `scripts/verify_threefield_lbm.py`; tests:
`tests/test_threefield_lbm_verification.py`.

## 1. The verification debt this closes

`docs/threefield/README.md` and `docs/threefield/FORMULAS.md` grade the level-ice
resistance coefficients in detail, but the LBM stage of the same coupling had **no
analytic reference at all**: the only committed evidence for `wake_lbm` was

* agreement of the far-field race with a Gaussian self-similar model written by the
  same author (`gaussian_race_profile`), and
* a momentum-flux audit that compares the solver's own `thrust_impulse` accumulator
  with the solver's own velocity field.

Both are self-consistency checks: a solver that is uniformly wrong (wrong viscosity,
wrong force scaling) passes them. This document replaces them with three
independent closed-form references and reports what the shipped kernels actually do.

## 2. References used

| # | reference | exact statement | what it pins |
|---|---|---|---|
| A | decaying shear wave | `u_x(y,t) = U cos(k y) exp(-nu k^2 t)`, `k = 2 pi m / N_y`, exact solution of incompressible NS (constant pressure, zero divergence) and of the discrete BGK equation to `O(k^4)` | the kinematic viscosity the kernel realises, against `nu = (tau0 - 0.5)/3` |
| B | Chapman-Enskog strain field | `f_i^(1) = -3 tau0 w_i rho (c_ia c_ib - c_s^2 delta_ab) S_ab` gives `Pi_ab = -2 c_s^2 tau0 rho S_ab`; the shipped `_outer` contract lists the six unique components once, so `|Pi| = (1/3) tau0 |S|` and the shipped `tau_eff` must return `nu_t = (sqrt(tau0^2 + 6 sqrt(2) Cs^2 tau0 |S|) - tau0)/6 -> Cs^2 |S| / sqrt(2)`, `|S| = sqrt(2 S_ab S_ab)` | the Smagorinsky closure value and its coefficient convention |
| C | Guo forcing impulse identity | with the `(1 - omega/2)` forcing weight the first moment of the populations grows by exactly `sum(F)` per step | the forcing term, the quantity the race momentum audit depends on |
| D | uniform-stream steady state | a uniform stream at the inlet velocity is a fixed point of the inlet / zero-gradient-outlet / free-slip set | that the production boundary set injects no spurious force (the defect class fixed in `dem3d_platen_contact_mask_audit.md`) |

## 3. How it is run

The propeller solver has no periodic mode, and references A and C need one. Two
verification-only switches were added to `WakeLBM`, both inert in production:

* `WakeLBM(config, periodic=True)` wraps every face instead of applying the wall
  set (`_stream(..., periodic=True)` is the shipped stencil with `torch.roll`;
  `_apply_boundaries` is a no-op), so the shipped collision, equilibrium, effective
  viscosity and streaming kernels are exercised unmodified;
* `step(n, body_force=...)` accepts a prescribed force density, which switches off
  the actuator-disk controller and books the force through the same
  `thrust_impulse` / `forcing_work` accumulators.

`tests/test_threefield_lbm_verification.py` asserts that the default (non-periodic)
path is still the production one. The shear wave is initialised with
`f = f^eq(rho=1, u=(U cos k y, 0, 0))`, the cosine projection
`(2/N_y) sum_y u_x cos(k y)` is sampled every few steps, and `ln|amplitude|` is
fitted against `t` over the window after the initial `O(u^2)` transient. Nothing is
hand-tuned: the fit window is `t >= 0.08 t_end` and `|a| >= 0.18 U`.

## 4. Results

Numbers are in `artifacts/threefield_lbm_verification/results.json` and, formatted,
in `artifacts/threefield_lbm_verification/results_table.md`; the table is written by
the same script that produces the numbers. Figures:
`artifacts/threefield_lbm_verification/figures/`.

* **A — viscosity.** Five relaxation times `tau0 = 0.6 ... 1.9` (`nu = 0.033 ... 0.467`),
  `N_y = 48`, `mode = 3`. The realised viscosity agrees with `(tau0 - 0.5)/3` to
  well inside the 3 % acceptance band; the fit is a straight line in `ln|a|`
  (`R^2 > 0.999`). Mass and the streamwise/transverse momenta are conserved to
  round-off, and `u_y` is zero to round-off, i.e. the mode is not contaminated by
  compressibility or by the rolling stencil.
* **B — Smagorinsky closure.** For `Cs = 0.05 ... 0.25` and `|S| = 0.01 ... 0.08`
  the eddy viscosity returned by the shipped `_effective_tau` matches the analytic
  value of `nu_t = (sqrt(tau0^2 + 6 sqrt(2) Cs^2 tau0 |S|) - tau0)/6` to round-off,
  and `nu_t / (Cs^2 |S|) -> 1/sqrt(2) = 0.7071` from below as predicted by the
  expansion of the closed form. **The implemented coefficient is therefore
  `Cs_eff = Cs / 2^(1/4) = 0.8409 Cs`**, because the shipped `_outer` contraction
  lists the six unique `Pi` components once while Hou et al. use
  `sqrt(2 sum_{ab} Pi_ab^2)` over all nine, which differs by `sqrt(2)`. A user who
  reads "Cs = 0.16" as textbook Smagorinsky is running an 0.84-fold weaker SGS model.
  This is now pinned numerically rather than folklore; the coefficient itself is a
  convention, and the fix (if wanted) is to double the `Pi` norm, not to change `Cs`.
* **C — forcing defect (found by this audit, fixed).** Before the fix the shipped
  forcing term added the full `3 w_i (c_i . F)` while the collision against
  `u + F/(2 rho)` leaked `0.5 omega F` of momentum back, so one step injected
  `(1 + 1/(2 tau0)) x sum(F)` instead of `sum(F)`: **1.9988x at the production
  default `tau0 = 0.5006`** (measured, and reproduced analytically for
  `tau0 = 0.5006/0.7/1.0/1.5/2.0`). The moment the actuator-disk controller is
  switched into open loop — as in `momentum_impulse_audit` — the booked
  `thrust_impulse` and the momentum actually delivered disagree by that factor, so
  the "momentum-conservation audit" advertised in the module docstring could not
  close. Fix: the Guo weight `(1 - omega/2)` on the forcing term. After the fix the
  ratio is 1 to round-off (`|ratio - 1| < 1e-11` over five relaxation times).
* **D — boundary set.** `_apply_boundaries(_stream(f))` reproduces an exactly
  uniform stream to `0` in double precision (population deviation `0.0`, density
  and velocity deviations `~1e-16`), while the streaming step alone moves
  populations by `O(0.06)`: the production inlet / outlet / free-slip set is
  force-free for a uniform stream, which is what the race solver relies on at the
  inlet plane.
* **E — closure dynamics.** With `Cs = 0.16` active, the solver's amplitude history
  is compared with the 1-D nonlinear-diffusion model
  `du/dt = d/dy[(nu0 + Cs^2 |du/dy|) du/dy]` integrated on the same grid; a
  molecular control (`Cs = 0`) is run alongside so the closure contribution is
  visible. The residual deviation is a *modelling* difference (the 1-D model drops
  the lattice's O(k^4) dispersion and the eddy viscosity's temporal lag), and is
  reported as measured rather than as a pass/fail.

## 5. Impact of the forcing fix on the committed race results

The actuator disk is a proportional controller that drives the disk cells towards a
prescribed velocity (`force = rho (target - u)`). Its steady state is therefore fixed
by the target velocity, not by the loop gain, so the fix changes the *transient gain*
and the *bookkeeping*, not the target. Two consequences are reported honestly:

1. `thrust_impulse` and `forcing_work` now integrate the momentum that was actually
   delivered, so any statement of the form "the race momentum flux equals the
   booked thrust" becomes checkable. Numbers produced before this commit booked the
   force and delivered `~2x` it, so their force-based diagnostics are not
   comparable and must be regenerated (`scripts/` race runs and
   `artifacts/threefield/*`).
2. The controller now converges to the same target with half the effective gain;
   mean axial profiles and half-widths are expected to move by the transient
   amount only. That expectation is *not* claimed as verified here: the race cases
   were not re-run inside this task, and the committed race artifacts predate the
   fix.

## 6. Limitations / honest scope

* Lattice units only; no physical validation, and nothing here calibrates the race
  or the level-ice resistance against model tests.
* References A, B and C are exact for the *incompressible* limit; the measured
  agreement is quoted at `U = 0.02` (`Ma ~ 0.035`), where compressibility errors are
  `O(Ma^2) ~ 1e-3`. The shear-wave check therefore bounds, but does not eliminate,
  compressibility error at the race velocities.
* Reference D is a consistency audit, not a convergence proof: it shows the
  boundary set adds no force to a uniform stream. There is no closed-form solution of
  the full race problem to compare against, and the free-slip wall's effective
  location (node-centred specular reflection) still biases the lateral profile.
* The Smagorinsky check (B) is a closure check at the level of `tau_eff`; it does not
  validate Smagorinsky as a turbulence model for this flow.
* The 1-D reference in (E) is itself a model, so item (E) is a consistency statement
  between two models, not verification.

## 7. Reproducing

```bash
PYTHONPATH=src python scripts/verify_threefield_lbm.py      # writes artifacts/threefield_lbm_verification/
python -m pytest tests/test_threefield_lbm_verification.py -q
```

Runtime of the full campaign is about one minute on one core; the test module is
about ten seconds.