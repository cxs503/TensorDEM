# DEM friction -> angle-of-repose calibration surface (TensorDEM, m4970)

**Status: forward model + calibration table + inverse lookup implemented and
tested; monotonicity achieved on a truncated bracket (one non-monotone node
reported); the end-to-end inverse-recovery criterion (< 2 %) is NOT met and is
reported as an honest FAIL with an error budget (see §6).**

Module: `src/tensordem/repose_calibration.py`
Script: `scripts/calibrate_repose_dem3d.py`
Tests: `tests/test_repose_calibration.py` (8 tests, incl. 2 negative cases)
Artifact: `artifacts/repose_calibration/repose_calibration.json` (+ `.md` table)
Figures: `docs/repose_angle_calibration/*.png`

---

## 1. Case introduction (physics, geometry, parameters)

Granular DEM is driven by *microscopic* contact parameters — the Coulomb
friction coefficient `mu` and the rolling-resistance coefficient `mu_r`. The
quantity that can be measured on a hold, a hopper or an ice-rubble stack is
*macroscopic*: the **angle of repose** `theta_r`, the steepest free surface the
material can sustain. A DEM model is only usable for ship-hold / bulk-cargo /
rubble-pile work once there is a *calibrated, quotable map* `mu -> theta_r` and
its inverse, with the domain of validity stated.

Before this milestone TensorDEM had calibration tools for stiffness and
uniaxial compressive strength (`material_calibration.py`,
`scripts/calibrate_dem3d_ucs.py`) and for restitution
(`scripts/calibrate_dem3d_restitution.py`), but **no automatic calibration for a
free-surface angle** — the gap recorded as "m605 缺自动标定". This milestone
fills exactly that gap with the minimal surface requested: forward map,
monotone table, inverse lookup, extrapolation guard.

### Forward model (lifting-cylinder column collapse)

A hexagonal-close-packed column of monodisperse spheres rests on a frictional
floor inside a rough cylindrical wall. The wall bottom edge rises linearly with
time (a *lifting cylinder*, the standard repose experiment), so the column
drains from below and deposits an axisymmetric heap; the free-surface angle of
the settled heap is `theta_r`.

| quantity | symbol | value | note |
|---|---|---|---|
| sphere radius | `r` | 8.0 mm | monodisperse |
| density | `rho` | 2500 kg/m³ | ballast-like |
| cylinder radius | `R_c` | 35 mm | rough wall, `mu_wall = mu` |
| fill height | `H` | 140 mm | `H/(2R_c) = 2.0` |
| particle count | `N` | 102 | hexagonal packing, `jitter = 0.002 r` |
| normal stiffness | `k_n` | 6.0e4 N/m | linear spring-dashpot |
| tangential stiffness | `k_t` | 0.5 `k_n` | Cundall-Strack |
| restitution | `e` | 0.05 | dashpot from `zeta(e)` |
| gravity | `g` | 9.81 m/s² | −z |
| rolling resistance | `mu_r` | 0.20 | held fixed for the primary table |
| pre-settle / lift / settle | | 0.03 / 0.35 / 0.70 s | wall down, wall rise, free |
| timestep | `dt` | 6.6e-5 s | period/20 of the stiffest contact |
| steps | | 16.4 k | per case |
| realisations | | 3 seeds | `4970, 4971, 4972` |

Contact laws (all TensorDEM, SI units):

* normal: `F_n = max(k_n δ − γ v_n, 0)`, `γ = 2 ζ sqrt(k_n m_eff)`,
  `ζ = |ln e| / sqrt(π² + ln² e)`;
* tangential: `F_t = −k_t δ_t`, `δ_t` accumulated from the relative surface
  velocity (including the spin contribution `ω × (∓r n)`) and projected onto
  `|δ_t| ≤ mu F_n / k_t`;
* rolling: isotropic rolling spring `M = −k_r θ_r`, `θ_r` accumulated from the
  relative spin and projected onto `|θ_r| ≤ mu_r r F_n / k_r` (an internal
  couple: equal and opposite on the two particles);
* rough wall and floor: the same tangential law against the static surfaces.

### Macro-response estimator

The deposit is described in cylindrical coordinates about its own centroid.
Each annulus of width `r` is split into 8 azimuthal sectors; the maximum
elevation per sector is the local free-surface height and the annulus height is
the sector mean. A straight line is fitted on the flank between 15 % and 90 % of
the shoulder-to-toe radial span with one-sided 2σ clipping; the reported angle is
`theta_r = atan(−slope)`. The regression `r²` and the residual surface speed are
exported as quality flags (a heap that is still flowing gives a meaningless
angle — all reported runs settle to `max|v| ≤ 9 mm/s`).

---

## 2. Method and tool chain

1. `python scripts/calibrate_repose_dem3d.py --jobs 12` — runs the 39 forward
   cases (6+1 friction nodes × 3 seeds + 6 verification μ × 3 seeds) in a
   process pool, each worker on one BLAS thread.
2. Nodes `mu = 0.25 … 0.55` step 0.05 at fixed `mu_r = 0.20`; three seeds per
   node; the node value is the seed mean, its seed scatter is exported.
3. **Monotone bracket policy**: the table is built on the *longest strictly
   increasing prefix* of the raw nodes. Nodes that break monotonicity are
   exported separately (`excluded_nodes`) with their raw angle — no silent
   smoothing.
4. Table interpolation is monotone piecewise cubic (Fritsch-Carlson PCHIP), so
   the interpolant cannot overshoot and the inverse is single-valued.
5. Inverse lookup is a bisection on the PCHIP with a strict bracket guard:
   queries below `theta_r(mu_min)` or above `theta_r(mu_max)` (and friction
   queries outside `[mu_min, mu_max]`) raise `ReposeExtrapolationError`. The
   guard fires on real data in this milestone (the μ = 0.525 verification run is
   outside the truncated bracket and is reported, not extrapolated).
6. Verification runs at six off-node friction values close the loop
   `mu → DEM → theta_r → table → mu_hat` and export the relative error, plus the
   local invertibility `|dmu/dtheta|` and the propagated seed-noise uncertainty
   `sigma_mu = |dmu/dtheta| · sigma_theta`.

Anchors (falsifiable, closed form, exercised by the test suite):

* **A1** — irrotational sphere on a tilted plane starts to slide exactly at
  `theta = arctan(mu)`;
* **A2** — free sphere with rolling resistance moves exactly at
  `theta = arctan(min(mu, mu_r))` (sliding needs `tan θ ≤ mu`, rolling needs
  `tan θ ≤ mu_r`), and `mu_r = 0` gives motion at any tilt;
* **bound** — for cohesionless material the settled slope must satisfy
  `tan(theta_r) ≤ mu`; checked at every table node;
* **negative control** — the same pile with `mu_r = 0` must be markedly flatter
  (it is: the rolling degrees of freedom are what make a sphere heap possible at
  all).

---

## 3. Results

### 3.1 Calibration table (μ_r = 0.20, 3 seeds, settled heaps)

| `mu` | `theta_r` [deg] | seed σ [deg] | `R_toe` [mm] | apex [mm] | flank `r²` | max\|v\| [mm/s] |
|---|---|---|---|---|---|---|
| 0.25 | 4.56 | 0.24 | 102.7 | 16.4 | 0.67 | 1.0 |
| 0.30 | 5.26 | 0.32 | 113.3 | 19.4 | 0.68 | 1.0 |
| 0.35 | 6.32 | 1.66 | 100.0 | 21.1 | 0.74 | 1.0 |
| 0.40 | 7.21 | 0.33 | 108.0 | 21.8 | 0.68 | 2.0 |
| 0.45 | 10.92 | 4.28 | 105.3 | 30.3 | 0.87 | 2.0 |
| 0.50 | 12.54 | 1.99 | 102.7 | 29.5 | 0.84 | 9.0 |
| 0.55 | **10.88** | 2.26 | 100.0 | 25.6 | 0.75 | 2.0 |

The `mu = 0.55` node breaks monotonicity (10.88° < 12.54°) and is **excluded
from the table**, exported verbatim as the non-monotone region. Calibrated
bracket: `mu ∈ [0.25, 0.50]`, `theta_r ∈ [4.56°, 12.54°]`.

Physical reading: the angle grows with sliding friction over the whole bracket,
but the growth flattens as `theta_r` approaches `arctan(mu_r) = 11.3°` — the
rolling-resistance ceiling. Above that ceiling `mu` is no longer identifiable
from a repose angle, which is why the curve saturates and eventually scatters.

### 3.2 Closed-loop inverse recovery (the acceptance number)

| `mu` (truth) | DEM `theta_r` [deg] | recovered `mu_hat` | error |
|---|---|---|---|
| 0.275 | 6.35 | 0.3512 | **+27.7 %** |
| 0.325 | 5.08 | 0.2888 | −11.1 % |
| 0.375 | 5.47 | 0.3111 | −17.0 % |
| 0.425 | 7.14 | 0.3977 | −6.4 % |
| 0.475 | 10.13 | 0.4381 | −7.8 % |
| 0.525 | 12.10 | 0.4798 | −8.6 % |

### 3.3 Anchors and physics checks

| check | reference | measured | error |
|---|---|---|---|
| A1, `mu = 0.30` | `arctan(mu)` = 16.70° | 17.40° | +0.70° (+4.2 %) |
| A1, `mu = 0.60` | `arctan(mu)` = 30.96° | 31.56° | +0.59° (+1.9 %) |
| A2, (`mu`,`mu_r`)=(0.6,0.2) | static below 11.31°, moving above | 0.7 µm vs 1.85 mm | exact sign change |
| A2, (0.2,0.2) | same ceiling | 0.7 µm vs 2.56 mm | exact sign change |
| A2, (0.6,0.6) | static below 30.96° | 1.5 µm vs 7.93 mm | exact sign change |
| kinematic bound `tan θ_r ≤ mu` | 6/6 nodes | margin 0.17 … 0.28 | satisfied |

The A1 residual (+0.6…0.7°) is a one-sided *detection lag* of the bisection (the
driving acceleration vanishes at the onset, so the displacement grows only
quadratically); it is bounded by ~1° for the stated `duration = 0.15 s`,
`slide_fraction = 0.25`.

### 3.4 Figures

* `fig_repose_calibration_curve.png` — calibration curve with node error bars,
  off-node verification points, and the closed-loop inverse error against the
  2 % acceptance line;
* `fig_repose_contact_anchors.png` — anchor A1 (measured onset vs `arctan mu`)
  and anchor A2 (static/sliding on both sides of `arctan(min(mu, mu_r))`);
* `fig_repose_heap_sections.png` — settled heap cross-sections with the fitted
  flank lines for `mu = 0.30 / 0.40 / 0.50` (seed 4970 realisations:
  `theta_r = 5.1° / 7.1° / 14.8°`; the seed-mean table values are 5.26 / 7.21 /
  12.54°, the spread at `mu = 0.50` being the seed scatter quoted in §3.1).

---

## 4. Reproducibility / audit chain

```
python scripts/calibrate_repose_dem3d.py --jobs 12     # 39 cases, ~3 min on 12 cores
python -m pytest tests/test_repose_calibration.py -q   # 8 tests
```

The artifact records the case dictionary, the contact-law list, per-node raw and
seed-mean angles, the excluded node, the table with its PCHIP slopes, the
verification rows with their status (`recovered` / `rejected_outside_bracket`),
the local invertibility and the noise-propagated uncertainty, the anchors and an
explicit `verdict` block. Every number in this document is copied from that
artifact.

---

## 5. Pitfalls, limitations, scope of validity

* **Rolling resistance decides everything for spheres.** With `mu_r = 0` a
  sphere heap cannot hold any slope (negative control), and with `mu_r = 0.2`
  the achievable angle saturates at `arctan(0.2) = 11.3°`. A repose calibration
  of *spherical* particles is therefore a calibration of `mu_r` first and of
  `mu` second; this is why the table stops at `mu = 0.50`.
* **Two sign conventions were decisive during development** and are documented
  in the module: the pair contact force on particle *i* must be the *negative*
  of the force on *j*, and the rolling spring is an *internal couple* (equal and
  opposite). Both errors are detectable only through the anchors above — they
  produce anti-damping and a "liquefied" pile that still looks plausible.
* **The deposit is thin.** At `theta_r ≈ 10°` a 100 mm-radius heap is only
  2–4 particle diameters deep, so the flank is resolved by ~6 grains; the seed
  scatter (up to 4°) is the honest uncertainty of one measurement.
* **Splash regime**: for `mu ≲ mu_r` the column drains too easily and the
  material spreads into a sheet instead of a heap (`apex ≈ r`); such runs have
  no measurable flank and are rejected by the estimator rather than fitted.
* Soft-sphere overlap under the own weight reaches ~2 % of the radius; the
  contact stiffness is therefore part of the case definition and is quoted.
* Monodisperse spheres, no cohesion, no polydispersity, smooth-grain rolling
  law, `g` kept at 9.81 m/s², and a single fixed `mu_r` for the primary table.
  **Joint `(mu, mu_r)` inversion is not implemented — it is declared as the next
  roadmap item**, with the 2-D surface and a proper identifiability analysis.

---

## 6. Error judgement (criterion by criterion, no hedging)

| acceptance criterion | measured | verdict |
|---|---|---|
| calibration curve monotone (or non-monotone region reported) | 6/7 raw nodes monotone; `mu = 0.55` breaks it and is excluded + reported | **PASS** (with reported exception) |
| inverse recovery error < 2 % | 6.4 % … 27.7 % (worst 27.7 %) | **FAIL** |
| pytest ≥ 5 incl. extrapolation-rejection negative case | 8 tests, 2 negative | **PASS** |
| kinematic bound `tan θ_r ≤ mu` | satisfied at all 6 nodes | **PASS** |
| table entries reproducible / deterministic | 3 seeds per node, full case dictionary exported | **PASS** |

**Is the key quantity within 3 % of a reference?** No, not end to end.
The *closed-form anchors* are matched to 1.9–4.2 % on the angle (0.6–0.7°,
one-sided detection lag), and the *interpolation machinery* recovers `mu` to
1e-12 when fed its own forward value — but the **end-to-end** recovery, where
the DEM itself is the reference, is 6.4–27.7 % on `mu`. The 2 % target is not
reached and the artifact says so (`"inverse_recovery_under_2pct": false`).

### Error budget for the failed criterion

`|dmu/dtheta|` of the table is 0.052 in the shallow part and 0.015–0.020 in the
steep part, while one DEM measurement carries `sigma_theta = 0.6–2.3°`
(seed scatter). Propagating gives `sigma_mu = 0.03–0.13`, i.e. **7–38 % of mu** —
the observed 6–28 % is inside that band. The failure is therefore *resolution*,
not a bug: with ~6 grains along the flank and a saturated (rolling-controlled)
curve, the repose angle simply does not carry 2 % information about `mu`.

### What would close the gap (ordered, measurable)

1. Resolve the flank: `r = 2 mm` spheres and a 3–4× larger deposit
   (`R_toe/2r ≈ 25`), which reduces `sigma_theta` to ~0.3° (target
   `sigma_mu ≈ 2 %`); estimated cost 20–50 CPU-min per node on the 2-node
   cluster (128 cores × 2).
2. Steeper operating point: calibrate at `mu_r = 0.4–0.5`, where the pile is
   5–8 layers deep for the same radius, so the flank fit is well posed and the
   `mu` sensitivity sits in `mu < mu_r`.
3. Average more realisations (≥ 8 seeds) per node *and* per verification point;
   the seed scatter of the node mean then drops below 1°.
4. Joint `(mu, mu_r)` inversion on a 2-D table — the roadmap item this
   milestone declares; a single repose angle cannot separate sliding from
   rolling resistance (that is exactly the saturation seen here), so a second
   independent macro-response (e.g. a drained hopper time or a direct-shear
   peak) is required for identifiability.

---

## 7. Roadmap declaration

* **Declared, not implemented**: joint `(mu, mu_r)` inversion (2-D calibration
  surface) and a second macro-response for identifiability.
* **Declared, not implemented**: polydisperse / non-spherical (angular) grains,
  for which the angularity, not `mu_r`, sets the repose angle.
* **Declared, not implemented**: laboratory comparison. The values here are a
  *numerical* calibration of this solver; quoting them as physical repose angles
  of a specific bulk material without a measured `mu`/`mu_r` pair would be
  unjustified.