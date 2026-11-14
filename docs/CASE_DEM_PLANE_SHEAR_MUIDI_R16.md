# CASE-DEM-PLANE-SHEAR-MUIDI — Round 16: strain-convergence measurement, and why half of it is void

**Mission** m20855 · **anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` · **ref** da Cruz et al. 2005,
Phys. Rev. E **72**:021309 (`10.1103/physreve.72.021309`); GDR MiDi 2004, Eur. Phys. J. E
**14**:341 · **acceptance** μ(I) and φ(I) within 10 % over I = 1e-3 … 0.3.

---

## ① Case and background

Plane shear of a dry granular bed between two platens at fixed normal stress P: the
canonical device for the local rheology

```
μ(I) = τ/P ,   I = γ̇ d / sqrt(P/ρ)
```

with `μ = μ_s + (μ_2−μ_s)/(I_0/I + 1)` and a *dilatant* `φ(I)` that falls with `I`.
The case has run 15 rounds; round 15 closed with a single explicit open item:

> "The archived protocol cannot separate 'loose initial fill still compacting' from a
> genuine steady state, because the shear budget is `max(min_shear_steps, γ_target/(γ̇Δt))`
> … A strain-convergence sweep … was launched … it had not returned within this session,
> so no claim about the steady-state φ(I) is made here. **That is the next round's first
> measurement.**"

Device (`src/tensordem/mu_i_rheology.py`, `ServoShearCase`): periodic in x, rough or smooth
platens, servo-controlled normal stress, r = 0.005 m, d = 0.01 m, ρ = 2500 kg/m³,
P = 62.5 kPa, k_n = 6.25e4, μ = 0.5, μ_roll = 0.05, e = 0.3, N = 294, dt = 3.215e-5 s,
seed 5279, smooth walls. Six inertial numbers, one run each.

## ② Method

The six round-15 cluster jobs `cj2950 … cj2955` (`phi_estimator_audit.py --live --strain
2.0`) **all succeeded**; only their collection was missed. This round:

1. `archive_strain_sweep.py` — pulls the six artifacts out of the cluster tarballs, keeps
   every scalar verbatim, records job provenance, and thins the time series to ≤ 500 points
   (factors 1, 1, 1, 1, 3, 41 — see ⑤ for why one run is 4.7 MB raw).
2. `strain_convergence_audit.py` — reconstructs the step budget from the device's own
   formula, tests stationarity, and compares to `reference_law.py` on the `I_bulk`
   abscissa fixed in round 10.
3. `test_strain_convergence_audit.py` — 20 assertions on the invariants (110 passed,
   3 skipped for the whole benchmark suite).

No new physics was run; the round is a re-derivation and a measurement of artifacts that
already existed.

## ③ Results

### 3.1 The requested strain was not delivered above I = 0.01

<!-- table 1 -->
| I_target | γ̇ | planned floor | executed | recover steps | strain req | strain ach | overrun | budget respected |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.5 | 124522 | 124721 | 199 | 2.00 | 2.00 | 1.00x | yes |
| 0.003 | 1.5 | 41574 | 41773 | 199 | 2.00 | 2.01 | 1.01x | yes |
| 0.01 | 5.0 | 12543 | 12742 | 199 | 2.00 | 2.05 | 1.02x | yes |
| 0.03 | 15.0 | 4248 | 5922 | 1674 | 2.00 | 2.86 | 1.43x | **NO** |
| 0.1 | 50.0 | 1345 | 3892 | 2547 | 2.00 | 6.26 | 3.13x | **NO** |
| 0.3 | 150.0 | 515 | 20515 | 20000 | 2.00 | 98.93 | 49.46x | **NO** |

The mechanism is arithmetic, not statistical. `mu_i_rheology.py:1226`

```python
shear_steps = int(max(case.min_shear_steps,
                      math.ceil(case.shear_strain / (gamma_dot * dt))))
...
max_shear_steps = shear_steps + max(0, int(case.max_recover_steps))   # = +20000
```

and the **constant-stress validity gate** at the end of the loop keeps shearing while the
servo has not re-established the applied pressure, up to `max_recover_steps = 20000`
(default). `shear_steps` is therefore a *floor, never a ceiling*: nothing bounds the
accumulated strain. The reconstruction reproduces the archived strain with bookkeeping
error ≤ 1.4e-4 at every point, and at I = 0.3 the recover gate spent **exactly 20000
steps** — the fingerprint of a budget exhausted rather than a converged state.

### 3.2 φ *is* converged at the three honoured points

Stationarity is judged on the **terminal** drift (last block vs second-to-last of four
equal blocks of the shear series), threshold 0.5 %:

| I_target | φ_container | whole-series drift | terminal drift | φ converged |
|---|---|---|---|---|
| 0.001 | 0.5613 | +0.17 % | +0.50 % | yes |
| 0.003 | 0.5563 | +1.89 % | −0.05 % | yes |
| 0.01 | 0.5501 | +2.27 % | −0.21 % | yes |

The whole-series drift is positive and the terminal drift is ≈ 0: the bed *does* compact
from a loose fill and then settles. Strain 2.0 is sufficient — **round 15's central worry is
resolved, and in the direction that removes it as an excuse.**

### 3.3 … but μ is *not* converged anywhere in this sweep

| I_target | μ terminal drift |
|---|---|
| 0.001 | −13.2 % |
| 0.003 | −8.1 % |
| 0.01 | +14.1 % |

The virial stress needs roughly an order of magnitude more strain than the packing
fraction. These runs therefore **cannot adjudicate the μ half of the acceptance**; the
μ numbers in 3.4 are reported only to show they must not be used.

### 3.4 Position against the reference law

<!-- table 2 -->
| I_target | I_bulk | μ | μ_ref | μ err % | φ | φ_ref | φ err % |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.00133 | 0.2487 | 0.3621 | −31.3 | 0.5613 | 0.6397 | −12.3 |
| 0.003 | 0.00239 | 0.2894 | 0.3638 | −20.4 | 0.5563 | 0.6395 | −13.0 |
| 0.01 | 0.00496 | 0.3059 | 0.3678 | −16.8 | 0.5501 | 0.6390 | −13.9 |
| 0.03 | 0.01490 | 0.3033 | 0.3827 | −20.8 | 0.5528 | 0.6372 | −13.2 |
| 0.1 | 0.07809 | 0.3592 | 0.4591 | −21.8 | 0.5529 | 0.6276 | −11.9 |
| 0.3 | 0.32883 | 0.4385 | 0.6110 | −28.2 | 0.5776 | 0.6086 | −5.1 |

Only I ≤ 0.01 is quotable for φ. There, the steady φ is **0.550 – 0.561, i.e. 12.3 – 13.9 %
below the reference 0.639 – 0.640**, and the direction (φ falling with I) is correct.

## ④ Figures

| file | caption |
|---|---|
| `runs/round16/fig_strain_budget.png` | (a) planned shear-step floor vs steps actually executed; (b) strain achieved / strain requested against I, with the 1.05 tolerance line. The budget is honoured below I = 0.01 and fails by up to 49× at I = 0.3. |
| `runs/round16/fig_phi_mu_vs_reference.png` | measured φ(I) (container definition) and μ(I) (full-cell virial) against `reference_law.py` on the I_bulk abscissa. Filled markers = strain budget respected and therefore quotable; open squares = budget violated, shown for completeness only. |
| `runs/round16/fig_phi_convergence.png` | φ_extent along the accumulated strain for all six runs; dotted = budget violated. The three solid curves flatten (terminal drift ≤ 0.5 %), the dotted ones are still sheared when they stop. |

## ⑤ Pitfalls and limitations

* **A strain request is not a strain delivery.** The device's `shear_steps` is a lower
  bound; the pressure-recovery gate can multiply the sheared strain by 50 without
  warning. Any future round quoting a high-`I` point from a `--strain`-limited run is
  quoting a state sheared far past its budget. `strain_convergence_audit.verdict()` now
  classifies the points automatically (`n_strain_respected` / `n_strain_violated`) so this
  cannot silently recur.
* **A run can be valid for φ and useless for μ** — they have very different settling
  times. The audit reports both terminal drifts separately and sets
  `mu_usable_for_acceptance: False` for this sweep.
* **The φ deficit is not a transient.** That was round 15's working hypothesis and this
  round falsifies it: with the container definition *and* confirmed stationarity the gap is
  still ≈ 13 %. The remaining candidates are the loose initial preparation (a settled loose
  fill never reaches 0.64) and the reference magnitudes themselves, which are
  **PROVENANCE-UNVERIFIED** (0.58/0.64 inherited from round 1, no archived reading of da
  Cruz 2005 / GDR MiDi in-repo; da Cruz 2005 is a 2D-disk study with φ ≈ 0.84 → 0.79).
* **I=0.3 archived at 500 points, not 20309.** `sample_stride = max(1, steps//400)` uses
  the *planned* budget; at I = 0.3 that budget is 515 steps, so the stride collapsed to 1
  and the run emitted one sample per step (4.7 MB). This is collateral damage of the same
  defect. The archive keeps all scalars verbatim and records the thinning factor (41) and
  original length.
* Single seed per point (5279). No ensemble here; round 12 established the seed spread for
  the sweep protocol, but not for these strain-limited runs.

## ⑥ Error verdict (honest)

| quantity | criterion | measured | verdict |
|---|---|---|---|
| strain budget honoured | ≤ 1.05× requested | 3/6 points (1.00–1.02×); 1.43×, 3.13×, **49.46×** | **FAIL** |
| φ stationary at honoured points | terminal drift ≤ 0.5 % | +0.50 %, −0.05 %, −0.21 % | **PASS** |
| μ stationary at honoured points | terminal drift ≤ 0.5 % | −13.2 %, −8.1 %, +14.1 % | **FAIL** (μ not usable here) |
| φ(I) vs reference, I ≤ 0.01 | < 10 % | −12.3 %, −13.0 %, −13.9 % | **FAIL** |
| φ(I) trend | decreasing in I | 0.5613 → 0.5563 → 0.5501 | **PASS** (dilatancy) |
| μ(I) vs reference | < 10 % | no quotable point in this sweep | **WITHHELD** |

**Is any key quantity within 3 % of its reference? No.** The closest is φ at I = 0.3
(−5.1 %), which is exactly the point with a 49× strain overrun and is therefore not
quotable. Among quotable points the best φ error is −12.3 %.

**Ledger verdict: still FAIL, but for a narrower and now precisely stated reason.** The
φ gap is not a strain-convergence artifact — that hypothesis is falsified. What remains is
(a) the high-`I` μ deficit, which this sweep cannot address at all, and (b) a ≈ 13 % φ
offset whose status cannot be settled until the reference magnitudes are anchored to an
archived reading.

### Recommended next steps

1. Anchor φ_max/φ_min to an archived reading of da Cruz 2005 / GDR MiDi 2004 (or Chialvo &
   Sundaresan) **before** any further φ comparison — this is the blocking item.
2. Fix the protocol so a strain request is a ceiling: either terminate on
   `strain >= shear_strain` with the pressure gate reinterpreted as a *validity flag*, or
   raise/parametrise `max_recover_steps` and record `strain_budget_respected` in every run.
3. Re-run I = 0.03 … 0.3 with the fixed budget to get quotable φ points, and with ≥ 10×
   the strain for μ.
4. Test the "loose initial fill" candidate directly: shear from a pre-compacted state at
   I = 0.001 and see whether φ approaches 0.64.