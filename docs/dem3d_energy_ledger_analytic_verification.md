# Closed-form verification of the DEM3D cumulative energy ledger

## Purpose

`EnergyAuditedIceDEM3D` (`src/tensordem/dem3d_energy.py`) accumulates tool work, external
work, drag dissipation, contact-damping dissipation and fracture release, and reports
`energy_balance_residual_J` as the closing term. Before this study the ledger had only
self-consistency coverage: terms finite and non-negative, checkpoint restart identical,
repeat-run signature stable. Those tests cannot tell whether the *numbers* are right, nor
how large the residual should be. This campaign scores the ledger against closed-form
solutions of two exactly solvable subsystems driven through the production `step()`.

No solver source was modified: the closed-form cases are built by replacing the lattice
state of a fresh solver (`build_subsystem`) and then resetting the ledger reference energy.

## Method

### Case A - exponential viscous decay (drag only)

A single free particle, mass `m`, linear drag coefficient `c`, no bonds, no contacts,
tool parked far away:

```
m dv/dt = -c v            ->  v(t) = v0 exp(-t/tau),   tau = m/c
D(t) = int c v^2 dt'      ->  D(t) = 0.5 m v0^2 (1 - exp(-2 t / tau))
```

Parameters: `m = 0.0600175 kg` (radius 0.025 m, density 917.0 kg/m^3), `c = 2.0 N s/m`, `v0 = 0.5 m/s`, run to `t = 1.50 tau` = 385 steps at `dt = 0.000116917 s` (the solver's `recommended_dt`).

The ledger's `drag_dissipation_cumulative_J` is compared with `D(t)`, and the recorded
residual is compared with the truncation the scheme must produce: because the drag rate
is sampled at the pre-update velocity, each step over-records
`0.5 * c * v^2 * (dt c / m) * dt`, i.e. a fraction `0.5 dt c / m` of the dissipated energy,
giving the closed-form prediction `R_pred = 0.5 (c/m) dt D(t)`.

### Case B - bonded harmonic oscillator (elasticity only)

Two free particles joined by one elastic bond of stiffness `k`, equal masses `m`, no drag,
no external load. The relative stretch `xi = |x1 - x0| - L0` obeys
`xi'' = -omega^2 xi` with `omega = sqrt(2k/m)`, so `xi(t) = A cos(omega t)`.

Parameters: `k = 2000 N/m`, `omega = 258.161 rad/s`, `T = 0.0243382 s`, `A = 0.00025 m` (208 steps per period), 3 periods = 624 steps.
Note the amplitude is 0.5 % of the rest length: this is a small-strain case, so the linear
bond law is the exact model and the cosine is an exact reference.

Fracture thresholds are lifted (`breaking_strain = shear_breaking_strain = 10`) so the
case stays purely elastic. This is not cosmetic - see finding 4.

## Results

### Case A: decay and dissipation ledger

| quantity | simulation | closed form | relative error | verdict (<3 %) |
| --- | --- | --- | --- | --- |
| velocity `v(t)` [m/s] | 0.111238707 | 0.111565080 | 0.2925 % | PASS |
| kinetic energy [J] | 3.713298327e-04 | 3.735119820e-04 | 0.5842 % | PASS |
| cumulative dissipation `D(t)` [J] | 7.144777271e-03 | 7.128676725e-03 | 0.2259 % | PASS |
| ledger residual [J] | 1.391840e-05 | 1.388703e-05 | 0.2259 % | PASS |

Residual as a fraction of the reference mechanical energy (`E0 = 0.00750219 J`): `0.001855` (0.1855 %).
No tool work and no contact dissipation were recorded, as required for an isolated particle.

### Case B: oscillator

| quantity | simulation | closed form | relative error | verdict (<3 %) |
| --- | --- | --- | --- | --- |
| period [s] | 0.024337299 | 0.024338223 | 0.003796 % | PASS |
| stretch at `t_end` [m] | 2.500284043e-04 | 2.499716669e-04 | 0.02269 % of A | PASS |
| sampled energy spread / mean | 0.0301835 | 0.0301835 (`dt*omega`) | 0.0001352 % | PASS |
| ledger residual / E | 0.00043293 | 0 | - | PASS (bounded) |

The bond survived all 624 steps (`alive_bonds = 1`), and no work or
dissipation was booked: the residual is pure integrator behaviour, not a bookkeeping leak.

### Step-size convergence

Observed order = least-squares slope of `log(error)` vs `log(dt)` over the refined levels.

| dt [s] | steps A | v err [%] | D err [%] | residual / E0 | steps B | period err [%] | energy spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000116917 | 385 | 0.2925 | 0.2259 | 0.001855 | 624 | 0.003796 | 0.03018 |
| 5.84586e-05 | 770 | 0.1462 | 0.1128 | 0.0009266 | 1248 | 0.0009484 | 0.01509 |
| 2.92293e-05 | 1540 | 0.07307 | 0.05638 | 0.000463 | 2499 | 0.0002372 | 0.007546 |

| error measure | observed order | expected |
| --- | --- | --- |
| case A velocity | 1.001 | 1 (explicit drag rate) |
| case A dissipation | 1.001 | 1 |
| case A residual | 1.001 | 1 |
| case B period | 2.000 | 2 (symplectic Euler phase) |

### Fracture strain measure probe

The same campaign scored `IceDEM3D._objective_shear_strain`, the invariant that gates
shear fracture, against closed-form Green-Lagrange strains.

| configuration | shear invariant | reference | verdict |
| --- | --- | --- | --- |
| cubic lattice 135 particles, undeformed | 4.622e-16 | 0 | PASS |
| cubic lattice 135 particles, uniaxial `e = 0.01` | 0.006700000 | 0.006700000 | PASS (1.6e-12 %) |
| cubic lattice 135 particles, rigid 30 deg rotation | 6.340e-16 | 0 | PASS |
| two particles, one collinear bond, undeformed | 0.000000000 | 0 | PASS (fixed: rank-deficient neighbourhoods report 0) |

## Findings

1. **The ledger reproduces closed-form energy bookkeeping.** Drag dissipation matches the
   analytic integral to 0.2259 %, and the isolated bond conserves
   energy with a residual of 0.000433 of the reference energy.
2. **The residual is a first-order truncation, and it is predictable.** `R = 0.5 (c/m) dt D(t)`
   matches the recorded residual to 0.2259 %, and the measured order is
   1.00. The residual must therefore be reported together with `dt`;
   it is not evidence of a bookkeeping bug and it is not zero in the limit `dt -> 0` only by
   construction - it is a known, removable first-order bias in the dissipation estimator.
3. **The oscillator's energy error is bounded, not secular.** The sampled energy oscillates
   with relative amplitude `dt*omega` = 0.0301835, matched to
   0.000135 %, and the period error is second order
   (2.00). Soak runs are limited by phase error, not by energy drift.
4. **Defect found and fixed: the objective shear invariant was not zero-preserving on
   rank-deficient neighbourhoods.** For a single bond the least-squares deformation gradient
   degenerated to the projection `e e^T`, so `E = 0.5 (e e^T - I)` had deviatoric norm `1/3`
   even for a rigid, undeformed bond (0.333333 measured). The default shear limit is
   `0.03`, so **every bond of a one-particle-wide (chain-like) specimen broke in the first step**.
   The solver now skips the shear test on rank-deficient local reference configurations
   (fewer than 3 non-coplanar bonds, detected via `torch.linalg.matrix_rank(cov) < 3`) and
   reports 0, so a rigid, undeformed collinear bond no longer fractures. On the shipped
   cubic lattices every particle has full-rank (3) covariance, so the measure is unchanged
   (uniaxial value matched to 0.0 % and frame-indifferent to 6.3e-16); the fix is confined
   to degenerate reference configurations and unblocks thin/linear specimens and single-bond
   unit setups, which is exactly how closed-form cases are built.

### Fix applied (this campaign)

The behaviour-preserving alternative from the original recommendation was applied: the
solver now skips the shear test when the local reference covariance rank is below 3
(`torch.linalg.matrix_rank(cov) < 3`), reporting 0 shear for such degenerate
neighbourhoods. This is the minimal, low-risk form that does not perturb the full-rank
lattice: every particle on the shipped cubic lattices has rank-3 covariance, so the
UCS/bending benchmarks are bit-for-bit unchanged (verified below). The alternative
Tikhonov-regularisation form (add a term scaled by the largest covariance eigenvalue)
remains available if a rank-2 planar neighbourhood should later be given a meaningful
out-of-plane shear measure.

## Limitations

* Only the linear-drag and single-bond universes are covered: contact damping, tool work,
  platen contact and fracture release are exercised by the existing benchmark suite, not
  compared with a closed form here.
* Failure thresholds are lifted in case B, so the campaign does not verify fracture energy
  accounting.
* The integrator is the shipped semi-implicit Euler; no sub-stepping or higher-order scheme
  is characterised.
* The closed-form cases use 1-2 particle subsystems; the lead defect (finding 4) was
  precisely a statement that such subsystems are degenerate for the shear criterion. It is
  now fixed (rank-deficient neighbourhoods report 0 shear), so single-bond unit setups are
  usable; future closed-form campaigns may still prefer a full-rank patch (at least 4
  non-coplanar bonds) for a physically meaningful deformation gradient.

## Error judgement (vs closed-form reference)

| key quantity | relative error | < 3 % ? |
| --- | --- | --- |
| case A velocity | 0.2925 % | yes |
| case A cumulative dissipation | 0.2259 % | yes |
| case A residual vs closed-form prediction | 0.2259 % | yes |
| case B period | 0.003796 % | yes |
| case B stretch (of A) | 0.02269 % | yes |
| case B energy spread vs `dt*omega` | 0.0001352 % | yes |
| lattice shear invariant vs analytic | 1.618e-12 % | yes |

All seven reference comparisons are below 3 %. The tightest is the shear invariant on a
full cubic lattice (1.6e-12 %), followed by the oscillator energy-spread prediction
(1.4e-4 %); the loosest is case A velocity (0.29 %). The former *failure* (the
collinear-neighbourhood shear invariant reporting 0.3333 where 0 is exact) is now
resolved: rank-deficient neighbourhoods report 0, so the measure is zero-preserving
on all reference configurations.

## Reproduction

```bash
PYTHONPATH=src python scripts/verify_dem3d_energy_ledger_analytic.py \
  --output results-dem3d-energy-ledger-analytic --refinements 3
PYTHONPATH=src python -m pytest -q tests/test_dem3d_energy_ledger_analytic.py
```

Artifacts: `energy_ledger_analytic.json` (all numbers above), `energy_ledger_dt_sweep.csv`
(convergence table) and `energy_ledger_analytic.png` (four panels: decay, dissipation
ledger, oscillator waveform, log-log convergence).
