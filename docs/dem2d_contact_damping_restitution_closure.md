# 2-D engine contact-damping → restitution closure

Evidence grade before this work: **MOCK** for the 2-D contact-damping capability.
`DEMConfig.contact_damping` was a raw dashpot coefficient in N·s/m with no
measurement of the coefficient of restitution it produces, and no test that
`tensordem.restitution`'s closure (which the 3-D path owns) transfers to the
2-D solver, although `src/tensordem/restitution.py` already asserted in prose that
it does. After this work: **REAL** — the transfer is measured on the engine's
own contact branch, with a stated error budget.

## 1. The gap

`tensordem.dem` (2-D bonded disks) applies the same tensionless linear
spring-dashpot as the 3-D solver (`src/tensordem/dem.py`, pair and tool branches):

```
F_n = max(0, k_n · δ − c_n · v_n)        δ > 0 overlap, v_n = normal separation rate
```

integrated with semi-implicit Euler. The force is **clipped at zero**, so the
law is unilateral: the dashpot cannot pull, the contact opens as soon as the
spring force is overcome, and the rebound is *faster* than the bilateral
textbook result `e = exp(−πζ/√(1−ζ²))`, `ζ = c/(2√(k m))`. The bilateral formula
even predicts complete sticking, whereas the clipped law still rebounds.

Nothing in the 2-D test suite measured the restitution actually produced by a
given `contact_damping` (`tests/test_dem.py` has no restitution probe; the only
restitution evidence in the repository is the 3-D platen probe in
`src/tensordem/restitution.py`). A user who wanted "ice with e = 0.7" had no
verified way to choose the dashpot.

## 2. Method

* Harness: `src/tensordem/dem2d_restitution.py`. The lattice is built with the
  production constructor (`IceDEM(DEMConfig(nx=2, ny=2))`) so the solver state
  has its normal layout; every particle except one probe is parked far from the
  tool and clamped, the lattice bonds are broken by evaluating the force field
  once, and the free probe is released at `impact_speed` towards the kinematic
  circular tool. The 2-D solver has no rotational degrees of freedom and the
  tool is kinematic, so this is exactly one degree of freedom against an
  infinite-mass wall: reduced mass = particle mass.
* References: the continuous **unilateral** (clipped-law) solution
  `restitution.unilateral_restitution` and the **bilateral** text-book formula
  `restitution.classical_restitution`. Both are reused, not re-derived — that
  module remains the single source of truth.
* Base case: `k_n = 2000 N/m`, disk `r = 25 mm`, `t = 0.2 m`, `ρ = 917 kg/m³`
  (`m = 0.360105 kg`), `impact_speed = 1 m/s`, `Δt = DEMConfig.recommended_dt`.
  The recommended step tightens automatically with damping, which is why the
  coarse-damping rows below run at finer `Δt`.
* Reproduce:

```bash
python scripts/verify_dem2d_restitution_closure.py        # writes outputs/
python -m pytest tests/test_dem2d_restitution_closure.py   # 14 tests + 5 subtests
```

## 3. Results

### 3.1 Measured restitution vs the two closed forms

| c (N·s/m) | ζ | e measured | e unilateral | e classical | err vs unilateral | err vs classical | Δt (s) |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 1.0007 | 1.0000 | 1.0000 | +0.07% | +0.07% | 1.014e-03 |
| 1 | 0.0186 | 0.9452 | 0.9438 | 0.9431 | +0.15% | +0.22% | 1.014e-03 |
| 2 | 0.0373 | 0.8950 | 0.8919 | 0.8895 | +0.34% | +0.62% | 1.014e-03 |
| 5 | 0.0932 | 0.7634 | 0.7584 | 0.7453 | +0.65% | +2.42% | 1.014e-03 |
| 8 | 0.1490 | 0.6579 | 0.6515 | 0.6228 | +0.97% | +5.63% | 1.014e-03 |
| 10 | 0.1863 | 0.5983 | 0.5918 | 0.5512 | +1.11% | +8.56% | 1.014e-03 |
| 15 | 0.2795 | 0.4773 | 0.4726 | 0.4008 | +0.99% | +19.09% | 6.859e-04 |
| 20 | 0.3726 | 0.3883 | 0.3849 | 0.2832 | +0.89% | +37.10% | 5.144e-04 |
| 30 | 0.5589 | 0.2696 | 0.2677 | 0.1203 | +0.72% | +124.06% | 3.430e-04 |
| 50 | 0.9316 | 0.1493 | 0.1486 | 0.0003 | +0.50% | +46644.06% | 2.058e-04 |
| 80 | 1.4905 | 0.0766 | 0.0763 | 0.0000 | +0.32% | +7.66% | 1.286e-04 |

Two readings of the same table:

* the engine tracks the **clipped-law** prediction across four decades of
  restitution, worst case **1.11%** (ζ = 0.186), i.e. inside the 3% engineering
  gate everywhere, including the overdamped regime (`c = 80 N·s/m` extrapolates
  to `e = 0.077` while the bilateral formula already reports 0);
* the **bilateral** text-book formula is only adequate below ζ ≈ 0.1 (+2.4% at
  ζ = 0.093) and unusable above ζ ≈ 0.2 (+37% at ζ = 0.37, +124% at ζ = 0.56).
  The last column of "err vs classical" for `c = 80` is small only because the
  classical value is itself ~1e-4: the absolute discrepancy is 0.0766.

### 3.2 Inverse mapping — `recommended_damping`

| target e | inverse branch | recovered c | measured e | signed error | <3% |
|---|---|---|---|---|---|
| 0.90 | unilateral | 1.8391 | 0.9028 | +0.32% | YES |
| 0.90 | classical | 1.7991 | 0.9048 | +0.54% | YES |
| 0.70 | unilateral | 6.5597 | 0.7060 | +0.86% | YES |
| 0.70 | classical | 6.0548 | 0.7240 | +3.42% | NO |
| 0.40 | unilateral | 19.0276 | 0.4037 | +0.92% | YES |
| 0.40 | classical | 15.0285 | 0.4767 | +19.18% | NO |

`dem2d_restitution.recommended_damping` (bisection on the clipped law) hits all
three targets inside 3%; `recommended_classical_damping` fails from `e = 0.70`
downwards and misses by 19% at `e = 0.40`. Both branches agree at `e = 0.90`,
which is the regime where the two laws coincide.

### 3.3 Residual error is the integrator, and it is first order

| c | Δt (s) | e measured | err vs unilateral |
|---|---|---|---|
| 5 | 1.014e-03 | 0.76338 | +0.650% |
| 5 | 2.536e-04 | 0.75962 | +0.154% |
| 5 | 6.340e-05 | 0.75873 | +0.037% |
| 20 | 5.144e-04 | 0.38832 | +0.894% |
| 20 | 1.286e-04 | 0.38570 | +0.213% |
| 20 | 3.215e-05 | 0.38506 | +0.049% |

Refining `Δt` by 4 cuts the closure error by 4.2× and 4.2× (c = 5) and 4.2×/4.3×
(c = 20): semi-implicit Euler bias, `O(Δt)`, always on the *bouncy* side (the
scheme injects energy into a linear oscillator, visible as `e = 1.0007` at
`c = 0`). At the solver's own recommended step the bias is ≤ 1.1%, so no
correction term is needed for engineering use; users who need tighter `e`
should refine `Δt` rather than de-tune `c`.

## 4. Error judgement

| quantity | reference | error | < 3% |
|---|---|---|---|
| e(ζ = 0.037) | continuous clipped law | +0.34% | YES |
| e(ζ = 0.093) | continuous clipped law | +0.65% | YES |
| e(ζ = 0.186) | continuous clipped law | +1.11% | YES |
| e(ζ = 0.373) | continuous clipped law | +0.89% | YES |
| e(ζ = 0.932) | continuous clipped law | +0.50% | YES |
| damping for e = 0.70 (unilateral inverse) | measured restitution | +0.86% | YES |
| damping for e = 0.40 (unilateral inverse) | measured restitution | +0.92% | YES |
| damping for e = 0.70 (classical inverse) | measured restitution | +3.42% | NO — documented trap, not a fix target |
| Δt-convergence order (c = 5) | first order in Δt | observed 4.2× per 4× | YES (qualitative) |

The last "NO" is the *finding*, not a defect: the classical inverse is retained
only as the audit branch.

## 5. Pitfalls and limits

* The probe measures the **contact-law** restitution of one degree of freedom.
  A bonded multi-particle ice block has a different, configuration-dependent
  effective restitution (bond compliance plus staggered contact onset); do not
  quote these numbers for a block impact. The 3-D module documents the same
  caveat (block deviation > 10% at moderate damping).
* `DEMConfig.tool_speed` must be positive, so the "static" counterface is
  modelled with `1e-9 m/s`; tool travel during a contact is ~1e-10 m, i.e.
  seven decades below the particle radius.
* The single-dof assumption is *asserted*, not assumed: `measure_restitution`
  raises if any bond survives the parking step.
* `recommended_dt` depends on `contact_damping` (`damping_dt = 0.2 m / c`), so
  a damping sweep silently changes `Δt`. Rows in the tables above therefore
  carry their own `Δt`; comparisons at *fixed* `Δt` are made only in §3.3.
* Impulse/energy bookkeeping is not audited here — the tool-reaction ledger is
  a separate contract (`docs/dem3d_energy_ledger_analytic_verification.md` is
  the 3-D analogue; no 2-D equivalent exists yet). That is the natural next
  verification debt for this engine.
* Rebound is speed independent to < 1% for a linear law (asserted in the test
  suite at 0.1 and 5 m/s), which also bounds the regime where these numbers
  apply: strain-rate dependent ice contact needs a different law, not a
  different restitution.

## 6. Pointers

| artefact | path |
|---|---|
| closure harness | `src/tensordem/dem2d_restitution.py` |
| tests (14 + 5 subtests) | `tests/test_dem2d_restitution_closure.py` |
| evidence generator | `scripts/verify_dem2d_restitution_closure.py` |
| measured table | `outputs/dem2d_restitution_closure/closure_table.csv` |
| inverse map | `outputs/dem2d_restitution_closure/inverse_map.json` |
| Δt convergence | `outputs/dem2d_restitution_closure/timestep_convergence.json` |
| figure | `outputs/dem2d_restitution_closure/closure.png` |
| 2-D contact law | `src/tensordem/dem.py` |
| continuous closed forms (reused) | `src/tensordem/restitution.py` |