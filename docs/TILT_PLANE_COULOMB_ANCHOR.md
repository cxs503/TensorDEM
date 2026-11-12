# Tilted-plane Coulomb friction anchor (`tensordem.tilt_plane_friction`)

**Status:** benchmark surface implemented and measured; campaign artifact
`artifacts/tilted_plane/results.json` (numbers, thresholds, sha256) +
`artifacts/tilted_plane/provenance.json` (versions, repro command).
**Owner module:** `src/tensordem/tilt_plane_friction.py`; contact-law extension:
`src/tensordem/friction_polyhedron.py` (`mu_dynamic`, `slip_stick_speed`,
`friction_cone_limit`). **Tests:** `tests/test_tilt_plane_coulomb_anchor.py`.

## 1. Physics and what is measured

A body on a plane inclined at `theta` is held by static friction iff
`tan(theta) <= mu_s`; once sliding, its down-slope acceleration is
`a = g (sin theta - mu_k cos theta)`. Both are closed forms, so the module
measures **two independent branches** from the same DEM configuration:

| branch | measurement | inversion |
|---|---|---|
| static | bisect `theta` for first motion (generic bracket `[0, 60] deg`) | `mu_s = tan(theta_c)` |
| dynamic | least-squares acceleration of the centroid on the late window (40–100 % of the run), `x = x0 + v0 t + a t^2/2` | `mu_k = tan(theta) - a / (g cos theta)`, reported as `atan(mu_k)` |

The static/dynamic pair is carried by the **contact law**, not by switching
coefficients between runs: `FrictionPolyhedronConfig.mu_dynamic` selects the
residual cone the moment the elastic tangential force exceeds the static cone,
and the contact stays latched at `mu_d` until the slip speed falls below
`slip_stick_speed` (re-stick). Without the latch a contact would ratchet between
the two cones and the mean friction would depend on the time step.

## 2. Configuration and why it is the right one

| choice | reason |
|---|---|
| **rotation locked** (`inertia = 1e12 I`) | a freely rotating body measures rolling, not Coulomb sliding: a sphere below `tan(theta) = 3.5 mu` rolls without slipping and the naive inversion returns `(2/7) tan(theta)` (measured, `sphere_regime_probe`); a freely rotating flat-faced cube creeps/rolls below `atan(mu)` because the collapsed face contact cannot shift its centre of pressure (measured, `free_cube_probe`). Rotation locking is a **numerical isolation experiment** and is documented as such. |
| **static-equilibrium initial state** | the block is placed with wall overlap `m g cos(theta)/k_n` and pre-loaded tangential spring `m g sin(theta)/k_t`, so step 1 is already the steady state. A free-fall gap or an empty spring makes the onset angle a start-up transient: the elastic Cundall–Strack spring overshoots the static cone on its first oscillation and the block trips into the residual cone at *any* angle above `atan(mu_k)` (measured: 28.5 deg for `mu_s = 0.6, mu_k = 0.4` before this fix, 30.96 deg after). |
| **micro disturbance** `kick_speed = 1e-6 m/s` | required by the benchmark protocol; its kinetic energy is orders of magnitude below the static barrier except within ~0.02 deg of the limit, and the test suite shows the bracket does not move when the kick is varied. |
| **first-motion criterion** `slide > 0.01 * particle extent` (1 % of the 12 mm cube edge = 0.12 mm over `t_end = 0.30 s`) | the elastic stick displacement at onset is `mu_s m g cos(theta)/k_t ~ 1e-6 m`, i.e. two orders below the tolerance; near the limit the acceleration needed to reach the tolerance is `a > 2 tol / t_end^2 ~ 2.7e-3 m/s^2`, i.e. a 0.016 deg angle bias — well below the 0.05 deg bisection tolerance. |
| **time step** `dt = 0.01 * 2 pi sqrt(m / k_max)` (= 100 steps per contact period) and levels `dt, dt/2, dt/4` | the stability bound is explicit; the convergence table reports the critical angle at all three levels. |

## 3. Predeclared thresholds (also written into `results.json`)

* `REL_ERR_LIMIT_PCT = 5.0` — static inversion error for `mu in {0.2, 0.4, 0.6}` (acceptance gate).
* `REPORT_STRICT_PCT = 3.0` — stricter level answered explicitly in the report; missed levels are reported, never re-tuned.
* `DUAL_MIN_SEPARATION_DEG = 2.0` — required static-vs-dynamic friction-angle separation in the dual case `(0.6, 0.4)` (analytic 9.16 deg).
* `DT_LEVELS = (0, 1, 2)` — convergence levels; gate = absolute angle error does not increase with refinement.
* Secondary gates: dynamic inversion ≤ 5 %, 3×3 layer load sharing ≤ 5 % plus no motion below the analytic angle, `F_n / W = cos(theta)` within 1 %.

## 4. Worked example

```bash
# full campaign (29 tasks, deterministic, ~15 min on 16 workers)
PYTHONPATH=src python -m tensordem.tilt_plane_friction campaign \
    --out artifacts/tilted_plane --workers 16 --figures
# figures only (regenerated from results.json, no hand-typed numbers)
PYTHONPATH=src python -m tensordem.tilt_plane_friction figures \
    --results artifacts/tilted_plane/results.json
# single measurement (debug)
PYTHONPATH=src python -m tensordem.tilt_plane_friction probe --mu 0.4 --theta 30
# end-to-end pipeline smoke test (numbers are NOT the benchmark)
PYTHONPATH=src python -m tensordem.tilt_plane_friction campaign --out /tmp/smoke --smoke
```

## 5. Known limitations (measured, not assumed)

1. **Rotation-locked bodies are an isolation experiment.** The anchor measures
   the translational Coulomb cap. The physically free configurations measured in
   this campaign (sphere, flat-faced cube) do *not* return `mu` below their
   rolling/tip limits — that is the documented reason for the frozen rotation.
2. **The 3×3 layer is non-contacting** (`raft_gap = 0.02`, spheres). A touching
   layer is unaffordable with the current pair kernel: a *penetrating*
   polytope–polytope pair costs ~17 ms (sphere–sphere ~90 ms) against ~0.6 ms
   for a separated pair, and `PolyhedronDEMSystem` rebuilds the convex hull of
   every polytope at every step (~130 ms per step for nine cubes). The layer
   task therefore verifies load sharing over nine simultaneous wall contacts
   (rest below the limit, steady acceleration inverts to `mu`), not
   inter-particle coupling. A coupled-layer anchor is deferred.
3. **2-D-free geometry:** the tilt is a rigid rotation of the gravity vector
   relative to a plane, so no finite-size/edge effects of a real tilting rig
   (vibration, table compliance) are represented.
4. Dynamic inversion assumes steady sliding on the late window; it is invalid
   near the threshold, where `mu_app = tan(theta) - a/(g cos(theta))` is
   ill-conditioned (`a -> 0`). Dynamic probes are therefore placed at
   `atan(mu) + 8/12/16 deg`.

## 6. Result tables

The tables are in `artifacts/tilted_plane/results.json`
(`static_inversion`, `dynamic_inversion`, `convergence`, `static_dynamic`,
`raft_static`, `sphere_regime_probe`, `free_cube_probe`, `verdicts`) and are
reproduced verbatim in the mission report; every number there is read from the
artifact (no hand copying), and the sha256 of the payload is stored both in the
file itself and in `provenance.json`.