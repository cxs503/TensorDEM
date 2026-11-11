# P165-W2-N06 — Maturity of the non-spherical DEM: analytic curvature, velocity-Verlet, column-collapse convergence

Campaign node **P165-W2-N06** (TensorDEM, branch `agent/m9787`). Physics domain: granular /
non-spherical DEM contact mechanics.

Every number below is produced by the scripts in the "Reproducing" section into
`artifacts/dem/p165w2_n06/*.json`; the test suite
(`tests/test_dem_superquadric_maturity_p165.py`, 30 tests) reads the same artifacts, so the prose
and the evidence cannot drift apart.

---

## 1. What was broken (and is now paid off)

The parent maturity node (`docs/SUPERQUADRIC_CONTACT_MATURITY.md`,
`docs/dem_contact_mechanics_verification_report.md`) recorded three quantified defects and left
them open:

| # | defect (parent evidence) | used to be | criterion |
|---|---|---|---|
| C4 | principal curvature of the superquadric implicit surface obtained by finite differences; "~15 % error for `n1 != 2`" | 15 %–97 % | ≤ 1 % |
| C3 | time integration is semi-implicit Euler (1st order); "~5 % error in the fitted damping coefficient" | 5.4 % | ≤ 1 % |
| macro | repose angle only at a single particle number (574 spheres, one point); "full DEM repose simulations are needed" | single point | N = 2 k → 32 k spread ≤ 5 % |

This node replaces the estimators, measures the improvement against the *original* code kept as a
negative control, and runs the convergence study.

---

## 2. Method

### 2.1 Analytic curvature (`src/tensordem/superquadric_curvature.py`)

With `u = |x|/a`, `v = |y|/b`, `w = |z|/c`, `S = u^n2 + v^n2` and `F = S^(n1/n2) + w^n1 − 1 = 0`
the first and second derivatives are closed form:

```
F_x  = q S^(q/p − 1) u^(p−1) sgn(x) / a                     (p = n2, q = n1)
F_xx = q(q−p) S^(q/p−2) u^(2p−2)/a² + q(p−1) S^(q/p−1) u^(p−2)/a²
F_xy = q(q−p) S^(q/p−2) u^(p−1) v^(p−1) sgn(x)sgn(y)/(ab),   F_xz = F_yz = 0
F_zz = q(q−1) w^(q−2)/c²
```

Curvatures follow from the implicit-surface identities
`K = gᵀ adj(H) g / |g|⁴`, `Hm = (|g|² tr H − gᵀHg)/(2|g|³)`, `k₁,₂ = Hm ± √(Hm² − K)`, with the
sign convention of the kernel (convex ⇒ positive; sphere ⇒ `k = +1/R`).

`principal_curvatures_local` (DEM kernel) and
`superquadric.principal_curvatures_superquadric` (scalar reference contact path) now call these
formulas; the old estimators are kept behind `method="fd"` as the fixed baseline.

### 2.2 Independent mesh reference

A second, *independent* estimator samples the parametric surface
`x = a cos^e1(η) cos^e2(ω), y = b cos^e1(η) sin^e2(ω), z = c sin^e1(η)` (`e1 = 2/n1`, `e2 = 2/n2`)
and builds the shape operator from the first and second fundamental forms with central differences
of parametric step `π/N`.  Its truncation error is `O(N⁻²)`, which supplies both the cross-check and
the mesh-refinement order of criterion (b).

### 2.3 Velocity-Verlet with contact-force half-step position correction
(`src/tensordem/superquadric_integrators.py`, `SQSystem._step_velocity_verlet`)

```
v½ = v₀ + ½dt a(x₀, v₀)                     # kick
x₁ = x₀ + dt v½                             # drift  (contact forces re-assembled here)
a₁ = a(x₁, v_pred),  v_pred = v½ + ½dt a₀   # force at the drifted position
v₁ = v½ + ½dt a₁                            # kick
```

Rotation uses the same half-step structure on the body-frame Euler equations with a Rodrigues
update.  The force assembly runs twice per step (that is the "contact-force half-step position
correction"); the extra cost is the price of second order.  Two variants are implemented: the bare
kick-drift-kick (`velocity_predictor=False`) and the default with the predicted velocity, which is
what makes *velocity-dependent* terms (dashpots, spin damping) second order as well.
`integrator="euler"` reproduces the parent step bit-for-bit as the negative control.

### 2.4 Column collapse (`scripts/p165w2_n06_column_collapse.py`)

Loose cubic packing (spacing 2.2 r) of monodisperse spheres in a square footprint, held by four
gate walls; the gates are removed and the heap collapses on a frictional floor; the repose angle is
measured from the particle cloud (radial-bin envelope).  Parameters: `r = 3 mm`, `ρ = 2500 kg/m³`,
`E* = 1.0e7 Pa`, `μ_r = 0.10`, `e = 0.4`, `dt` from the contact-stability estimate
(`2.9e-5 s`), initial aspect 0.5, `g = 15 m/s²` (scaled up 1.5× so the free-fall time fits the
engine throughput; **identical for every run**, so all comparisons are unaffected — the `g = 9.81`
sensitivity run repeats the reference case at physical gravity).

A run is only accepted as *settled* when the measured angle varies by < 0.15° over three
consecutive 500-step blocks **and** the elapsed time is at least 2.2 free-fall times (without that
floor the plateau test fires on the still-untouched cubic packing — measured: θ = 0 at
`0.2 t_fall`).

### 2.5 Engine work needed to make the study possible
`superquadric_dem` was Python-limited (0.5 s per step at 2 000 particles): the pair filter used a
per-pair Python loop, per-pair attribute reads, `np.add.at` scatter and a per-pair Python dict for
the tangential history.  Four contained changes (all verified against the general path):

* exact sphere fast path for sphere–sphere and sphere–wall contacts
  (`n1 = n2 = 2, a = b = c`), equal to the implicit-surface path to 1e-9 in the physical regime;
* vectorised candidate-pair filter;
* `bincount` scatter instead of `np.add.at`;
* sorted-int64 pair-history ledger instead of the Python dict, plus a cached body-state snapshot
  per force assembly.

Measured: 1 800 → 220 ms/step at N = 2 048 (8×), and the sphere–wall path 15.5 → 1.5 ms/step.

---

## 3. Results

### 3.1 Criterion (a): analytic curvature

| check | reference | measured | criterion | verdict |
|---|---|---|---|---|
| ellipsoid limit, Gaussian curvature, 40 surface points | `K = 1/(a²b²c²(x²/a⁴+y²/b⁴+z²/c⁴)²)` | **1.27e-15** | ≤ 1e-6 | PASS |
| ellipsoid pole principal curvatures | `c/a²`, `c/b²` | **2.8e-16** | ≤ 1e-6 | PASS |
| sphere limit, `k₁ = k₂` | `1/R` | **1.9e-8** | ≤ 1e-6 | PASS |
| general superquadrics (6 shapes × 8 surface points = 48 cases), analytic vs independent mesh (N = 256) | `k₁, k₂` | **0.176 %** | ≤ 1 % | PASS |
| mesh-refinement order of the discrete estimator (5 shapes) | `err ~ N⁻ᵖ` | **p = 2.00–2.13** | ≥ 1.5 | PASS |
| legacy four-point FD estimator (parent, `eps_fd = 1e-7`), same 48 cases | analytic | **85.8 %–97.4 %** | — (negative control) | fails as documented |
| kernel FD-on-gradient estimator (parent DEM path) | analytic | **1.4e-11–4.0e-10** | — | was already accurate |

The last two rows are the honest part of the story: the parent's "15 %" is a *lower bound*. The
legacy estimator (`superquadric.py`, absolute step `1e-7` on a four-point second difference of `F`)
is wrong by 86 %–97 % at generic surface points — catastrophic cancellation, `ΔF ~ 1e-14` against
`eps² = 1e-14`.  The DEM kernel used a *different* estimator (central differences of the analytic
gradient with a relative step), which was already accurate to 1e-10; so the defect was in the
reference/scalar contact path and in the exported curvature API, not in the production force loop.
`fd_refinement_order` shows the legacy estimator is second order in its step (`p = 2.00` for the
non-ellipsoidal shapes) — i.e. the 15 % figure is a step-size artefact, exactly as the parent
document suspected.  (For the ellipsoid the fitted order is negative because the error is already at
the round-off floor at the coarsest step.)

### 3.2 Criterion (b) and (c): torque neutrality and the rolling-friction criterion

| quantity | analytic vs FD curvature (48 cases) | criterion |
|---|---|---|
| Coulomb rolling torque `μ_r R_eff F_n` | **3.4e-10** relative | ≤ 1 % |
| effective radius `R_eff` | **3.4e-10** relative | — |
| saturation ratio `|M| / (μ_r R_eff F_n)` | **1.0** (both estimators) | unchanged |
| `R_eff` from the legacy scalar estimator | 2.2e-6 relative | — |

The rolling-friction criterion of the parent node — the torque saturates at `μ_r R_eff F_n` — is
reproduced exactly (ratio 1.0 to machine precision) and the curvature improvement does not move it
by more than 3.4e-10.  The reason the legacy 86–97 % curvature error never showed up in the contact
force is a *sum*: Hertz uses `1/R_eff = (k₁ᵃ+k₂ᵃ)/2 + (k₁ᵇ+k₂ᵇ)/2`, and the errors of the two
principal curvatures of the four-point estimator nearly cancel in that sum (residual 2.2e-6).  This
is reported rather than claimed as a torque improvement: the value of the analytic curvature is the
correctness of the exported `k₁, k₂` (contact physics: flat-facet branch selection, Hertz radius for
elongated shapes, future anisotropic contact laws), not the magnitude of today's torques.

### 3.3 Criterion (d): integrator order and damping fit

| check | Euler (control) | velocity-Verlet (KDK) | velocity-Verlet (predicted v) | criterion |
|---|---|---|---|---|
| order, harmonic oscillator, `t = 10.37 T` (5 → 3 dt levels) | **1.09** | — | **2.00** | ≥ 1.9 |
| order, Duffing oscillator `x'' = −ω²x − 2x³` vs fine reference | **1.12** | — | **2.00** | ≥ 1.9 |
| damping-rate fit error, `γ = 1000 s⁻¹`, `dt = 1e-4` (`γdt = 0.1`) | **5.36 %** | 2.59 % | **0.18 %** | ≤ 1 % |
| damping-rate fit error, `dt = 5e-5` | 2.59 % | 1.27 % | **4.3e-4** (0.043 %) | — |
| damping-rate fit error, `dt = 2.5e-5` | 1.27 % | 0.63 % | **1.1e-4** (0.011 %) | — |

The parent node's "≈5 %" (Euler, `dt = 1e-4`) is reproduced to 5.36 %.  A log-linear fit is a *rate*
estimator, so a second-order scheme still leaves an `O(dt)` error in the fitted rate; the predicted
velocity removes exactly that (`O(dt²)`), which is why the criterion is met by the default variant
and not by the bare KDK (2.59 %, reported, not hidden).  The order measurement must not be taken at
an integer number of periods: at `t = 10 T` the phase error of a symplectic scheme cancels to leading
order and the *apparent* order is 3.0 (Euler) / 4.0 (Verlet) — a measuring artefact found and fixed
here, now guarded by the `t = 10.37 T` choice.

### 3.4 Criterion (e): conservative energy drift, 10⁴ steps

Single sphere, `e = 1.0` (exactly zero dashpot: `ContactMaterial.damping_ratio` now returns 0 for
`e ≥ 1`), frictionless, no gravity, launched at the wall at 0.25 m/s, 10⁴ steps, `δ/D ≈ 1 %`:

| dt | Euler max `|E−E₀|/|E₀|` | Verlet max `|E−E₀|/|E₀|` |
|---|---|---|
| 0.5 × dt₀ = 2.43e-5 s | **2.73e-2** | **3.39e-4** |
| dt₀ = 4.85e-5 s | 5.48e-2 | 1.56e-3 |
| 2 dt₀ | 1.12e-1 | 5.61e-3 |
| 4 dt₀ | 2.21e-1 | 2.84e-2 |

Euler is linear in `dt` (2.7e-2 → 2.2e-1), Verlet quadratic (3.4e-4 → 2.8e-2).  At the resolution
where the impact is properly resolved (`δ/D ≤ 1 %`, dt₀/2) the criterion is met with an eightfold
margin (3.4e-4 ≤ 1e-3) and the negative control fails as it must (2.7e-2 > 1e-2).  At the coarser
dt₀ the Verlet error is 1.6e-3 — reported honestly: the criterion is a statement about a
*resolution*, not about the scheme alone.

### 3.5 Criterion (f): column-collapse repose angle

See §4 (throughput-limited) and `artifacts/dem/p165w2_n06/column_collapse.json`.

---

## 4. Limitations and honest status

1. **Criterion (f) is throughput-limited.** The engine is a Python/NumPy DEM: 0.22 s/step at
   N = 2 048, ≈1 s/step at N = 8 000, ≈4 s/step at N = 32 000, and a settled collapse needs
   `2.2 t_fall/dt ≈ 6 400` steps at N = 2 k (growing as the column gets taller).  The N = 32 k
   endpoint of the criterion therefore needs ≈10 h of wall-clock for one run — beyond the 2 h
   cluster job limit.  What is reported: the N-series that actually settled, the measured
   throughput for every run, and the explicit statement of which endpoints are WITHHELD.  The
   convergence statement is made only over the settled range.
2. **Gravity is scaled to 15 m/s² (1.5×) in the N-series** to fit the throughput; all runs share it,
   and the `g = 9.81` reference run quantifies the sensitivity.  The absolute angle is
   model-dependent anyway (see 3).
3. **No rolling resistance at wall contacts.** The parent superquadric wall path applies *no*
   rolling resistance (unlike its sphere–sphere and clump–wall paths).  The new sphere fast path
   reproduces that behaviour exactly so that switching paths cannot move the physics; the
   inconsistency is documented and tested (`test_wall_rolling_resistance_is_absent_in_both_wall_paths`)
   rather than silently "fixed", because criterion (c) forbids changing the parent's rolling
   criterion inside this node.  This is the dominant model assumption for the *absolute* repose
   angle; it is common to every run, so the N-convergence is unaffected.
4. **Deep-overlap regime.** For `δ/D > 50 %` (one centre inside the other) the parent's
   projected-surface overlap degenerates to the centre distance instead of `r₁+r₂−d`; the analytic
   sphere path keeps the exact value.  Both are outside the physical DEM regime (`δ/D ≲ 1 %`) and
   the difference is pinned by a test.
5. **Two force assemblies per step.** The Verlet step costs ≈2× an Euler step; for production runs
   that trade is stated in the module docstring.
6. **Curvature is singular at the exact pole for `n1 < 2`** in the implicit representation
   (`F_xx ~ u^(n1−2)`); all shapes tested here have `n1, n2 ≥ 2`, which is the practical
   superquadric family (spheres, ellipsoids, rounded discs and blocks).

---

## 5. Error verdict (key quantities vs their reference)

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| Gaussian/principal curvature, ellipsoid | textbook closed form | 1.3e-15 | **yes** |
| principal curvature, general superquadrics | independent mesh estimator (N = 256) | 0.18 % | **yes** |
| discrete curvature estimator convergence | `O(N⁻²)` | order 2.00–2.13 | **yes** |
| rolling torque / `R_eff` (curvature change) | FD-curvature path | 3.4e-10 | **yes** |
| damping-rate fit | analytic `γ` | 0.18 % (Verlet) vs 5.36 % (Euler) | **yes** |
| position error, harmonic/Duffing oscillator | analytic/fine reference | order 2.00 vs 1.09/1.12 | **yes** |
| conservative energy drift (10⁴ steps, dt₀/2) | initial energy | 3.4e-4 (Verlet) vs 2.7e-2 (Euler) | **yes** |
| repose angle, particle-number series | mean of the settled series | see §3.5 / artifact, stated not guessed | depends on the settled range |

No criterion is claimed PASS without its number in the artifact, and every negative control is a
live test that fails when the mechanism it controls is removed.

---

## 6. Reproducing

```bash
# curvature / torque / integrator evidence (≈2 min) + 5 figures
PYTHONPATH=src python scripts/p165w2_n06_dem_maturity.py

# column-collapse sweep (one shard per run; also runs on the cluster)
PYTHONPATH=src python scripts/p165w2_n06_column_sweep.py --shard 1 --shards 10
PYTHONPATH=src python scripts/p165w2_n06_column_summary.py      # merge + fig6/fig7

# acceptance tests (30 tests, ≈60 s)
python -m pytest tests/test_dem_superquadric_maturity_p165.py -q
python -m pytest tests/test_superquadric_contact.py tests/test_superquadric_maturity.py -q
```

New/changed modules: `superquadric_curvature.py` (new), `superquadric_integrators.py` (new),
`superquadric_kernel.py` (analytic curvatures, zero damping at `e ≥ 1`, analytic sphere inertia),
`superquadric_dem.py` (velocity-Verlet step, sphere fast paths, vectorised assembly),
`superquadric.py` (analytic curvatures in the scalar reference path).