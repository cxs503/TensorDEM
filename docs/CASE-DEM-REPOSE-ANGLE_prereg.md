# CASE-DEM-REPOSE-ANGLE — pre-registration (frozen before any run)

Mission 19975 · branch `agent/m19975` · TensorDEM · domain: DEM granular mechanics

This file is written **before** the sweep is submitted. It fixes the decision,
the competing alternatives, the metrics, the acceptance thresholds, the run list
and the analysis, so that the numbers produced later cannot be used to move the
goal posts.

## 1. Decision

> Does TensorDEM's existing lifting-cylinder angle-of-repose protocol, driven by
> its existing Cundall–Strack + EPSD-rolling contact with **laboratory glass-bead
> contact parameters**, reproduce the experimental angle of repose of glass
> beads, **22°** (tolerance **±3°**)?

## 2. Alternatives (both pre-declared)

| id | statement | how it is tested |
|---|---|---|
| **H1 — predictive** | The pile slope is controlled by *sliding* friction alone: with `mu = tan(22°) = 0.404` (the friction angle the reference itself quotes) and a small, independently-motivated glass-bead rolling coefficient (`mu_r = 0.05`), the simulated heap reproduces `22° ± 3°`. | Leg **P** (3 held-out seeds, no tuning) |
| **H0 — rolling-limited** | The monodisperse smooth-sphere pile cannot hold a slope unless the *rolling* branch binds: the measured law of this solver is `theta_r ≈ arctan(min(mu, mu_r))`, so `mu_r >= tan(22°)` is required. The case is then *calibratable* but **not predictive** from independent glass parameters. | Leg **S** (`mu_r` scan) + Leg **C** (calibrated, held-out seeds) + Leg **N** (frictionless control) |

Both outcomes are publishable results of the case; neither is "the wrong
answer". H0 is the alternative the previous milestones' negative knowledge
predicts (one repose angle cannot separate sliding from rolling resistance).

## 3. Reference value (zero hand-copy)

B. T. Brzinski III, M. Mayor, D. J. Durian, *Depth-dependent resistance of
granular media to vertical penetration*, **Phys. Rev. Lett. 111, 168002 (2013)**;
the text version held in this repository is **arXiv:1307.4638v2**.

| quantity | value | original location |
|---|---|---|
| angle of repose of the medium | **22°** | arXiv:1307.4638v2, **p. 1, col. 2** |
| internal friction coefficient | **mu = 0.40** | same line |
| bulk density | **rho_bulk = 1.48 g/cm³** | same line |
| packing fraction | **phi = 0.59** | same line |
| grain size | **d = 300 ± 50 µm** (glass beads) | same line |

Derived in code, not typed: `rho_particle = rho_bulk / phi` (2508 kg/m³),
`mu_friction_angle = tan(22°) = 0.4040` (cross-check against the quoted `0.40`,
1.0 % consistency). The coarsening factor `d_DEM/d_exp = 26.7` is stated, not
hidden (Leg **E** measures its effect).

## 4. Outcome metrics and unit of analysis

* Primary metric: `theta_r` of the settled heap, degrees, from the **existing**
  `tensordem.repose_calibration.repose_angle` estimator (annulus x sector
  free-surface fit; not re-implemented here).
* Unit of analysis: one DEM realisation (one seed). Legs aggregate as
  mean ± 1 sigma over seeds; the seed is the replication unit.
* Error metric: `dtheta = theta_got - 22.0` deg, and `|dtheta|` in degrees
  (the mission tolerance is absolute, ±3°, so no percentages are used for the
  verdict; percentages are reported for information only).

## 5. Pre-declared acceptance criteria

| id | criterion | verdict form |
|---|---|---|
| **AC-1** | Leg P (predictive, `mu=0.404, mu_r=0.05`, 3 held-out seeds): `abs(mean(theta) - 22) <= 3 deg` | PASS / FAIL |
| **AC-2** | Leg C (calibrated `mu_r*`, 3 held-out seeds **not** used in the scan): `abs(mean(theta) - 22) <= 3 deg` | PASS / FAIL |
| **AC-3** | Leg N (frictionless control `mu = mu_r = 0`): `mean(theta) < 5 deg` — the angle must come from friction, not from packing or estimator | PASS / FAIL |
| **AC-4** | Leg S monotonicity: `theta(mu_r)` between 0.05 and 0.35 must be non-decreasing within `2 sigma_seed` at every node pair (the calibration is invertible) | PASS / FAIL |
| **AC-5** | Convergence / scale robustness (informational, not gating): particle radius `r ∈ {3, 4, 6} mm` and container `R_c/r ∈ {5, 8}` must not move the angle by more than the case tolerance 3° | PASS / FAIL, reported |

**Overall case verdict** = `AC-3 AND (AC-1 OR AC-2)`: the case is a *validated
benchmark* if the model can hit 22° ± 3° by at least one of the two routes while
the frictionless control flattens. Which route was needed is the *result*, not a
post-hoc choice.

## 6. Quality flags (reported per run, exclusion declared in advance)

A run is flagged (and reported in a separate "flagged" column, still shown in
the table) if any of:
`speed_max > 0.05 m/s` (heap still flowing), `r2 < 0.80` (flank not a line),
`n_fit_bins < 5` (flank shorter than 5 grain diameters), or
`discharge_ratio = toe_radius / R_c < 1.30` (stub that never left the cylinder).
Means are reported **both** over all seeds and over unflagged seeds; the verdict
uses the unflagged mean. Flat piles (Leg N) are expected to trip `r2`, which is
why AC-3 is stated on the angle only.

## 7. Controlled / changed variables

| variable | role | values |
|---|---|---|
| `mu_r` (rolling) | **changed** (scan) | 0.05 … 0.50, 9 nodes |
| `mu` (sliding) | **changed** (sensitivity) | 0.30, 0.404, 0.50 |
| seed | **changed** (replication) | scan seeds 19975/19976; held-out 19981/19982/19983 |
| grain radius | **changed** (scale) | 3, 4, 6 mm |
| `R_c/r` | **changed** (confinement) | 5, 8 |
| lift time | **changed** (strain rate) | 0.90, 1.20, 1.50 s |
| restitution | **changed** (quasi-static check) | 0.30, 0.70, 0.90 |
| stiffness `k_n` | fixed | 2000 N/m (already stiffness-converged per `docs/repose_material_calibration.md` §3.6) |
| `k_t/k_n`, wall law, floor law, geometry ratios, `dt = T_stiff/20`, `H/R_c = 9`, hold/settle times, density | fixed | as in `repose_material_calibration` defaults |
| estimator, evaluation code, machine, thread count | fixed | repo `main` + this branch, 1 torch thread per realisation |

Nuisance variables identified: seed-to-seed packing noise (dominant, quantified
by replication), container confinement (Leg W), coarsening (Leg E), lift rate
(Leg L), restitution (Leg R). No blinding is possible (deterministic
simulation, single analyst), so the thresholds were frozen in this file first.

## 8. Repetitions and rationale

2 seeds per scan node (to expose scatter without inflating cost), 3 held-out
seeds for the two verdict legs. With `sigma_seed ≈ 1-2°` known from
`docs/repose_material_calibration.md` §3.5, 3 seeds give a standard error of
~0.6-1.2°, i.e. the ±3° band is resolved to about one standard error — enough to
decide AC-1/AC-2 without claiming a precise mean. This is stated as a limit, not
a strength.

## 9. Analysis plan (frozen)

1. Aggregate per leg: mean, sigma, min, max of `theta_r`, plus flagged/unflagged
   split.
2. Invert `theta(mu_r)` at `theta_target = 22°` by bracket interpolation on the
   monotone part of the scan; refuse extrapolation (report `unresolved` if the
   target is outside the bracket).
3. Leg C = independent runs at `mu_r*` with held-out seeds; AC-2 uses their mean.
4. Report `dtheta` per leg with the tolerance band, all five criteria, and the
   sensitivity deviations. No outlier removal beyond the pre-declared flags; no
   threshold changes after seeing the data.
5. Raw data: `artifacts/CASE-DEM-REPOSE-ANGLE/results.json` (every run, every
   diagnostic), figures in the same directory, markdown table generated from the
   JSON (no hand-typed numbers in the report).

## 10. Environment / resource limits

Single repo checkout of `agent/m19975`; `torch` 2.7.1 CPU; one torch thread per
realisation (bit-reproducible); the sweep is submitted to the compute queue
(8 workers, single command, no node pinning), wall-clock budget 7200 s.
Reproduce with:

```
python3 benchmarks/case_dem_repose_angle/run_case.py --out artifacts/CASE-DEM-REPOSE-ANGLE --jobs 8
```

## 11. Interpretation limits

* Monodisperse perfect spheres: real glass beads are slightly irregular, so the
  calibrated `mu_r*` is a **numerical equivalent** of the missing angularity, not
  a measured material property.
* One repose angle cannot identify `(mu, mu_r)` separately (documented negative
  knowledge); the scan therefore quotes a *locus*, not a unique calibration.
* The laboratory grain is 300 µm and the DEM grain 4 mm (coarsening 26.7x);
  Leg E bounds the resulting bias, it does not remove it.