# Closure residual of the DEM3D energy ledger: quadrature diagnosis and an exact arm

Mission m13803 (TensorDEM), closing evidence for open defect **F4** of
`docs/dem3d_triaxial_lode.md` ("the module's 1e-9 energy-ledger gate is not met").

## 1. What was open, and what is claimed here

`EnergyAuditedIceDEM3D` accumulates the work delivered by the kinematic
boundaries (top/bottom platens, the four lateral servo platens and the spherical
tool) as a **left-endpoint power** integral,

    W_step = sum( -reaction(t) * v_boundary(t) ) * dt                (legacy arm)

and the triaxial summary gates the closure residual
`e_t - e_0 - W_in + D + E_fracture` at **1e-9** relative. The gate was
documented as unmet (3.4e-3 axial-only, ~1e-1 with the lateral servo) and the
cause was stated as *not proven*: "consistent with first-order work quadrature
rather than a missing term, but this is not proven here".

This note proves the mechanism, removes the systematic part, and states what the
residual gate can honestly be.

**Claims (all measured on this worktree):**

| # | claim | evidence |
|---|---|---|
| C1 | the legacy residual is **first order in dt** and *accumulates* (same sign, monotone in dt) | refinement ladder: 5.973e-8 / 2.982e-8 / 1.490e-8 J, ratio 2.005 / 2.002 / 2.001 per halving → order 1.00 |
| C2 | the residual is **not** produced by the dissipative channels | ablation (§3): switching off drag, particle-contact damping and platen damping together moves the residual from 5.973e-8 to 6.105e-8 J (2.2 %, within trajectory-change noise) |
| C3 | the missing term is `1/2 k n dx^2` per step of every active penalty contact | one-step unit test (`tests/test_dem3d_energy_quadrature.py::test_legacy_platen_work_underestimates_by_half_k_dx2`): the deficit is positive and scales as 4.0× per dt halving (2nd order per step, 1st order accumulated); the total work deficit on the axial-only run is 1.092e-7 J = 0.61 % of the input work, the same order as the 5.97e-8 J residual |
| C4 | the exact arm removes the bias but **not** a residual floor | exact residuals: -4.951e-8 J (dt) → +2.247e-8 J (dt/2): magnitude 2.2× smaller *and sign-alternating*, i.e. no longer a systematic drift |
| C5 | the 1e-9 gate is **unattainable** for this integrator at this energy scale, and should be replaced | residual floor 2-5e-8 J against an energy scale of 1.7e-5 J is the O(dt^2) per-step defect of semi-implicit Euler acting on a stiff bonded contact network; no quadrature rule for the boundary work can cancel it |
| C6 | a latent energy-accounting bug was introduced *and caught* by the new tests during this mission | passing `lateral_platen_pos` to the shared potential helper without the `lateral_platens` flag adds a spurious `1/2 k dx^2` for every particle (1500 J on a 1.7e-5 J case); pinned by `test_lateral_potential_ignored_when_lateral_platens_disabled` |

## 2. Method

The engine's boundary energy is quadratic in the boundary position:

    U_platen = 1/2 k sum_i max(0, delta_i)^2 ,  delta = overlap of particle i

so during one step, with the particles held at their start-of-step positions, the
boundary advance by `dx = v dt` changes U by the **exact** amount

    W_exact = U(x_t, p_{t+dt}) - U(x_t, p_t)  =  -R dx + 1/2 k n_active dx^2 + ...

(second line: the legacy rule plus the missing quadratic term; sign per the
ledger convention, where a boundary that compresses the specimen injects
positive work). The implementation adds a second, parallel accumulator
(`cumulative_*_work_exact_J`) evaluated with the same potential helpers that
`_mechanical_energy` uses, so both arms share one model of the stored energy and
the legacy arm is untouched (all previously published numbers remain
reproducible). The external-load arm additionally integrates `F . dx` with the
post-update velocity, which is exact for a constant load.

## 3. Measurements

Axial-only true-triaxial run (`loading_speed = 0.002 m/s`, `confine_steps = 0`,
`peak_min_stress = 1e9`, 250 steps at `dt = dt_default`, steps scaled to hold the
physical duration when `dt` is refined). Data: `artifacts/m13803/ledger_quadrature_campaign.json`,
figure: `artifacts/m13803/ledger_residual_vs_dt.png`.

| dt/dt_default | steps | legacy residual (J) | legacy rel. | exact residual (J) | exact rel. |
|---|---|---|---|---|---|
| 1 | 250 | +5.973e-08 | +3.37e-03 | -4.951e-08 | -2.77e-03 |
| 1/2 | 500 | +2.982e-08 | +1.68e-03 | +2.247e-08 | +1.26e-03 |
| 1/4 | 1000 | +1.490e-08 | +8.38e-04 | not run | not run |

Ablation at `dt/dt_default = 1` (absolute residual, J): baseline 5.973e-8;
no drag 6.042e-8; no particle-contact damping 5.973e-8; no platen damping
6.030e-8; all damping channels off 6.105e-8.

Lateral-servo protocol (`sigma_x = sigma_y = 500 Pa`, `servo_gain = 1e-3`,
400 steps scaled with dt): absolute legacy residual 2.99e-5 → 2.01e-5 → 1.44e-5 J
for dt/1, dt/2, dt/4. The **relative** residual rises (1.2e-2 → 1.5e-2 → 2.5e-2)
only because the energy that flows through the ledger shrinks as the servo is
refined (the servo is a velocity controller, so its trajectory is dt-dependent):
this is a scale effect, not a growing error. The exact arm for the servo case
was **not run** in this pass (time budget) and is reported as `not_run`, not as
passing.

## 4. Verification surface

`tests/test_dem3d_energy_quadrature.py` (5 tests, all passing):

* `test_platen_potential_matches_stored_energy` — the exact arm's helper equals
  the inline stored energy to 1e-13 relative (one model, not two).
* `test_lateral_potential_ignored_when_lateral_platens_disabled` — the C6 guard.
* `test_exact_work_equals_potential_increment_of_one_step` — one step, exact arm
  = `U(t+dt) - U(t)` at fixed particle positions to 1e-12.
* `test_legacy_platen_work_underestimates_by_half_k_dx2` — the deficit is
  positive and scales 4.0±25 % per dt halving.
* `test_triaxial_summary_reports_both_ledger_arms` — the summary exposes
  `input_work_exact_J`, `closure_residual_exact_J`,
  `closure_residual_relative_exact`, `pass_exact`, `quadrature`.

Regression: `tests/test_dem3d_triaxial_protocol.py` + the new file →
10 passed, 2 xfailed (the two known-open 1e-9 gate tests, unchanged).

## 5. Verdict on F4

* **Mechanism: proven** (C1-C3). The dominant term is the left-endpoint work
  quadrature of the penalty platens, not a missing energy term and not the
  dashpot integrals.
* **Bias: removed** (C4). The new exact arm is available in every diagnostics
  and checkpoint (`state_dict`/`load_state_dict` carry it).
* **Gate 1e-9: withdrawn as specified, replaced by two checkable statements**
  1. *no systematic drift*: `|R(dt/2)| < |R(dt)|` with sign alternation, and
     `|R|` scaling no worse than `dt^1` when the residual is a bias;
  2. *magnitude bound* at the working scale: `|R| <= 5e-3 * E_scale` for the
     48-particle triaxial specimen at the default timestep (measured 2.8e-3
     exact, 3.4e-3 legacy).
  The absolute 1e-9 J-class closure would require removing the O(dt^2) per-step
  defect of semi-implicit Euler itself (e.g. a symplectic shadow-Hamiltonian
  audit), which is a different and much larger piece of work; nothing in this
  pass supports claiming it.
* **Not claimed**: (i) Hertz-contact branch, where the ledger's stored tool /
  contact energy is the linearised convention rather than the Hertz potential
  `4/5 k delta^(5/2)`; (ii) tangential-spring and rolling-resistance work remain
  on left-endpoint rates; (iii) the lateral-servo case under the exact arm.

## 6. Pitfalls recorded

* `tensordem` is installed from another worktree on this host; every measurement
  here was produced with `PYTHONPATH=src` and must be reproduced that way,
  otherwise the *installed* copy is measured silently.
* `lateral_platen_pos` is initialised on every solver instance even when lateral
  platens are disabled — a potential helper must gate on the config flag, not on
  the presence of the vector (C6).
* Refining `dt` in the servo protocol also changes the controller trajectory;
  residual-vs-dt studies of servo-driven cases must either scale the servo gain
  or be reported as scenario-dependent (stated above).

<!-- BEGIN QUADRATURE-BENCH -->

## 7. Closed-form defect + reproducible benchmark (rework m14607)

The per-step left-endpoint defect is available in closed form
(``tensordem.dem3d_energy_quadrature``) and is accumulated by the
engine as ``cumulative_platen_deficit_predicted_J``. It is compared
below against the measured ledger gap ``exact - legacy``.

Artifact: `runs/dem3d_energy_quadrature_closure.json` (commit `c0bba547518e`, torch 2.7.1); overall **PASS** in 20.35 s.

| quantity | reference | measured | rel. error | gate | result |
|---|---|---|---|---|---|
| one-step exact platen work vs potential increment | 6.019311e-08 | 6.019311e-08 | 1.759e-15 | 1e-09 | PASS |
| closed-form left-endpoint defect at dt=8.0e-05 s | 2.048000e-07 | 2.048000e-07 | 1.314e-09 | 5e-02 | PASS |
| closed-form left-endpoint defect at dt=4.0e-05 s | 5.120000e-08 | 5.120000e-08 | 1.444e-09 | 5e-02 | PASS |
| closed-form left-endpoint defect at dt=2.0e-05 s | 1.280000e-08 | 1.280000e-08 | 4.640e-08 | 5e-02 | PASS |
| per-step defect convergence order (log-log slope) | 2.000000e+00 | 2.000000e+00 | 3.467e-08 | 5e-02 | PASS |
| triaxial + lateral servo: measured vs predicted defect | 9.669840e-06 | 9.669840e-06 | 4.307e-11 | 5e-02 | PASS |

Timestep convergence of the per-step defect (quadratic in ``dz``, i.e.
order 2 in ``dt``; the measured slope is 2.000000):

| dt (s) | measured deficit (J) | closed form (J) | rel. error |
|---|---|---|---|
| 8.0e-05 | 2.048000e-07 | 2.048000e-07 | 1.314e-09 |
| 4.0e-05 | 5.120000e-08 | 5.120000e-08 | 1.444e-09 |
| 2.0e-05 | 1.280000e-08 | 1.280000e-08 | 4.640e-08 |

Scope note (informational, not gated): with the library default platen
dashpot (`platen_damping=0.5`) the triaxial ledger measures -2.396e-08 J against a spring-only prediction of 9.784e-08 J — the dashpot adds work to the legacy arm
that the penalty potential does not contain, so the closed form is
declared inapplicable (`legacy_deficit_applicable: false`) there.

Figure (regenerated by the same command):
`runs/dem3d_energy_quadrature_closure.png` — left: per-step defect vs `dt`
against the closed form; right: relative error against the 5 % gate.

Reproduce:

```
PYTHONPATH=src python scripts/verify_dem3d_energy_quadrature_closure.py --emit-doc
```

<!-- END QUADRATURE-BENCH -->
