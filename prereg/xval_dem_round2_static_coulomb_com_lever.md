# Pre-registration — XVAL-DEM-ROUND2-STATIC-COULOMB-PLUS-COM-LEVER (m17674)

Written **before** any cross-validation number was produced. Thresholds below are
module constants of `benchmarks/xval_round2_static_coulomb/xval_spec.py` and are
not re-fitted after inspecting results.

## 1 Decision to support

Does the TensorDEM static-Coulomb + COM-lever rolling law (`rolling_model="coulomb"`,
`rolling_lever="com"`, `src/tensordem/contact/rolling_static.py`) reproduce the
*reference* behaviour of the same named做法 — i.e.

* **spec leg**: the closed-form/analytic law written in the reference description
  (`docs/GAP_DEM_CLUMP_ROUND2.md` §2.1 + the module docstring of
  `rolling_static.py`, both read from the OSS TensorDEM/self artifact), and
* **reproduction leg**: the published round-2 repose angles (17.47° / 14.86° /
  15.09° for `viscous/member`, `coulomb/member`, `coulomb/com` at ar=1.0, seed=0)
  —

within the owner precision standard **< 15 %**?

## 2 Alternatives (mutually exclusive)

* **H1 conform**: implementation == reference within the predeclared tolerances on
  every leg ⇒ the做法 is ported faithfully and the published numbers are
  reproducible in this worktree.
* **H0 deviate**: at least one leg exceeds tolerance ⇒ the做法 differs from its
  reference description; the deviation must be localised (spec / lever geometry /
  engine coupling) and reported as honest FAIL with `fail_stage`.

## 3 Metrics (unit of analysis = one experiment cell)

| id | metric | predeclared threshold | rationale |
|---|---|---|---|
| P1 | `max_rel_err` of delivered torque vs the independent oracle on a 3×2×2×2×4×7 grid (μ_r, R, Fn, I_eff, dt, ω/ω_eps) | ≤ 1e-12 | both are closed-form; anything above round-off is a real difference |
| P2 | saturation vs analytic `min(1, ω/ω_eps)` | ≤ 1e-10 abs | same |
| P3 | one-step arrest residual vs analytic `max(0, |ω| − ω_eps)` | ≤ 1e-12 rel | exact discrete identity |
| P4 | single-sphere `com` vs `member` lever | ≤ 1e-15 abs | bit-level sphere-baseline invariance |
| P5 | ellipsoid tip/flank lever ratio vs analytic `(a+r_mem)/(b+r_mem)` | ≤ 1e-12 rel | geometry, no fitting |
| P6 | engine spin-down: steps to arrest vs closed-form `max(1, ceil(ω₀ dt/ω_eps))`; energy monotone | ≤ 1 step; ΔE ≤ 1e-12 | engine coupling check |
| P7 | repose angle reproduction (3 configs) | **< 15 %** vs published; ≤ 1e-9 ° for the audit's bit-level `coulomb/member` value | owner precision standard |

## 4 Design, controls, confounders

* **Controls**: legacy `viscous/member` (pre-round-2 law) runs the same cells and
  must reproduce its *own* published number (17.47°) — it isolates "engine drift"
  from "new-law behaviour"; single sphere (ar=1.0, `com`≡`member`) isolates the
  lever change from the static-branch change.
* **Confounders neutralised**: same worktree, same seed (0), same mass-matched fill
  protocol, same `MU_ROLL=0.3`; the three configs differ only in the two
  `ContactParams` switches. Wall clock is *not* a metric.
* **Sensitivity (≥2)**: (i) time step dt ∈ {2e-5, 5e-5, 2e-4} s at fixed
  ω/ω_eps — the delivered saturation must be dt-invariant when scaled by ω_eps;
  (ii) spin rate sweep ω/ω_eps ∈ [1e-3, 1e4] must show the plastic→sticking
  transition at 1; (iii) element order none (pure functions + one engine run).
* **Stopping rule**: run all predeclared cells once; no early stopping, no cell
  dropped after inspection. A crashed cell is reported, not replaced.

## 5 Environment / records

worktree `/root/tp-agent/work/m17674`, branch `agent/m17674`, commit at run start
recorded in `artifacts/XVAL-DEM-R2-COULOMB-COM/results.json:env`; Python 3.11.6,
numpy 2.4.6; single process on the control plane (no cluster needed, wall < 10 min
per cell); raw JSON per leg next to the aggregate.

## 6 Interpretation limits

* The reference is the OSS artifact `TensorDEM/self` (this repository's own
  published source + round-2 report). `oss_inventory`/`oss_solutions` hold **no**
  third-party DEM implementation and the network egress is blocked (see friction
  log in the report), so the "OSS" comparison cannot be an independent
  third-party code comparison; the independent element here is the **oracle**
  (implemented from the documented equations only) plus the **closed forms**.
* A PASS means "faithful port + reproducible artifact", **not** "physically
  correct rolling friction": the round-2 report itself records the sphere
  baseline below the 22–25° literature band. Causality is not inferred beyond
  the identity of the equations.