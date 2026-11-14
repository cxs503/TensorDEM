# CASE-DEM-PLANE-SHEAR-MUIDI — Round 17: a strain request becomes a ceiling

**Mission** m21066 · **anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` · **ref** da Cruz et al. 2005,
Phys. Rev. E **72**:021309 (`10.1103/physreve.72.021309`); GDR MiDi 2004, Eur. Phys. J. E
**14**:341 · **acceptance** μ(I) and φ(I) within 10 % over I = 1e-3 … 0.3.

---

## ① Case and background

Plane shear of a dry granular bed between two platens at fixed normal stress P is the
canonical device for the local rheology

```
μ(I) = τ/P ,   I = γ̇ d / √(P/ρ) ,   μ = μ_s + (μ_2 − μ_s)/(I_0/I + 1)
```

with a *dilatant* φ(I) that falls as I grows. The device is
`src/tensordem/mu_i_rheology.py` (`ServoShearCase`): periodic in x, walled in y/z, rough or
smooth platens, servo-controlled normal stress, r = 0.005 m, d = 0.01 m, ρ = 2500 kg/m³,
P = 62.5 kPa, μ_contact = 0.5, μ_roll = 0.05, e = 0.3, dt = 3.2149e-5 s, seed 5279.

Round 16 closed with the anchor still FAIL and named the remaining blockers in priority
order. Its **item 2** is the subject of this round:

> "Fix the protocol so a strain request is a ceiling: either terminate on
> `strain >= shear_strain` with the pressure gate reinterpreted as a *validity flag*, or
> raise/parametrise `max_recover_steps` and record `strain_budget_respected` in every run."

That is a *correctness* defect, not an accuracy question: round 16 measured runs sheared up
to **49.5× past the strain they asked for** while still reporting `servo_valid: True`, so
every high-I point in the archive is silently mislabelled. No amount of re-analysis of
those artifacts can fix it — the device has to stop doing it. This round does that.

## ② Method

Scope was kept to the protocol, not to new physics. Nothing in the contact law, the servo,
or the initial preparation was touched; the six archived round-15 runs were **not**
re-run (they belong to the next, compute-heavy round).

1. **Device fix** (`src/tensordem/mu_i_rheology.py`).
   * New case field `max_recover_strain: Optional[float] = None` — a **strain-unit** ceiling
     on the constant-stress recovery phase. Strain (not steps) is the correct currency
     because the overrun depends on the step floor, which scales with 1/γ̇, i.e. with I.
     `None` keeps the historical step-only behaviour **bit-for-bit** so every archived run
     stays reproducible.
   * The recovery gate now records *why* it stopped: `recover_termination ∈
     {pressure_recovered, max_recover_strain, max_recover_steps, not_entered}`.
   * New helper `strain_budget_status(achieved, requested, tol)` — the **single** place that
     decides whether a delivered strain honours the request. Every result dict now carries
     `shear_strain_requested`, `strain_overrun_ratio`, `strain_budget_respected`,
     `strain_budget_status`, `recover_termination`, `max_recover_strain`, `strain_budget_tol`.
     A point can no longer be quoted without its budget status next to it.
2. **Driver fix** (`benchmarks/plane_shear_mu_i/phi_estimator_audit.py`) — the script that
   produced the round-15 archivals. Two changes, both in strain units:
   * `min_shear_steps=0`: the old `ceil(strain/(γ̇ dt)) + 100` fudge re-made the request a
     floor. At I = 0.3 one step is ≈ 0.96 % of a 0.5 request, so the `+100` alone delivered
     **1.93×** the budget before the recovery gate even opened. The step count must follow
     from strain alone.
   * `MAX_RECOVER_STRAIN_RATIO = 0.025`: the recovery phase may spend at most **half** of
     the 5 % classification tolerance, leaving the other half for the gate's one-step
     discretisation overshoot (the ceiling is tested after the strain increment).
   * The sampling stride is now sized against the *worst case* the gate can reach
     (`strain_steps + max_recover_steps`) instead of the planned floor, which is what made
     the I = 0.3 archival emit one sample per step over a 20 015-step run (4.7 MB).
3. **Evidence** — `run_strain_ceiling_r17.py` runs the identical configuration
   (I = 0.3, request 0.5, 210 particles, same seed) in three arms and saves the strain
   trajectories verbatim; `figures_strain_ceiling_r17.py` plots them from that JSON.
4. **Tests** — `tests/test_strain_budget_ceiling.py`, 15 assertions.

## ③ Results

### 3.1 The defect re-derived from the device's own arithmetic

The step budget is `shear_steps = max(min_shear_steps, ceil(strain/(γ̇dt)))` and the gate
may then run up to `+max_recover_steps = 20000` more. Reproducing that for the six nominal
I (round-16 parameters, driver floor `ceil(strain/(γ̇dt))+100`):

| I | γ̇ [1/s] | driver floor | `shear_steps` | worst-case strain | worst-case overrun |
|---|---|---|---|---|---|
| 0.001 | 0.500 | 124522 | 124522 | 2.32 | 1.16× |
| 0.003 | 1.500 | 41574 | 41574 | 2.97 | 1.48× |
| 0.01 | 5.000 | 12543 | 12543 | 5.23 | 2.62× |
| 0.03 | 15.000 | 4248 | 4248 | 11.69 | 5.85× |
| 0.1 | 50.000 | 1345 | 1345 | 34.31 | 17.16× |
| 0.3 | 150.000 | 515 | 515 | **98.93** | **49.46×** |

Every entry reproduces round 16's table-1 "executed"/"strain achieved" columns to the digit
(124522, 41574, 12543, 4248, 1345, 515; 98.93 at I = 0.3), independently of the archived
artifacts — the defect is arithmetic, not statistics. The overrun is **monotone in I**
because the floor is `1/γ̇` while the recovery allowance is a fixed step count.

### 3.2 The fix, measured on the shipped device

Same configuration, three arms, all at I = 0.3 with a 0.5 strain request:

| arm | `max_recover_strain` | `min_shear_steps` | delivered strain | overrun | status | `recover_termination` | `servo_valid` |
|---|---|---|---|---|---|---|---|
| historical | None | 200 | 16.6225 | **33.25×** | violated | pressure_recovered | **True** |
| ceiling only | 0.05 | 200 | 0.9645 | 1.93× | violated | max_recover_strain | False |
| ceiling + strain-only budget | 0.0125 | 0 | **0.5160** | **1.03×** | **respected** | max_recover_strain | False |

* Row 1 is the bug in one line: a run sheared 33× past its own request, reported
  `servo_valid: True`, and said nothing about it.
* Row 2 isolates the second facet: the ceiling bounds the *recovery* overrun, but the
  `min_shear_steps=200` floor alone still delivers 0.964 for a 0.5 request. This is why the
  driver change is required and why the flag matters — row 2 is honestly still `violated`.
* Row 3 is the round-17 protocol: delivered strain lands **3.2 % above the request**, inside
  the 5 % band, and the run *declares* that it is not a constant-stress measurement
  (`servo_valid: False`) instead of over-shearing until the servo settles. That is the
  intended trade: an honest "this point is not at constant stress" beats a silent 49× run.

Reduction in delivered-strain error against the request: **3325 % → 3.2 %**.

## ④ Figures

| file | caption |
|---|---|
| `runs/round17/fig_strain_budget_defect.png` | (a) worst-case strain overrun vs I, re-derived from the device arithmetic, with the step floor annotated on each point and the 1.05× acceptance band; the round-16 observed 49.5× sits on the analytic bound at I = 0.3. (b) accumulated strain vs shear step for the three arms: the historical run leaves the plot (log scale, 16.62), the capped arms stop. |
| `runs/round17/fig_ceiling_verdict.png` | delivered/requested strain per arm with the 5 % band: 33.25×, 1.93×, 1.03×, annotated with the `strain_budget_status` each run now reports. |

## ⑤ Pitfalls and limitations

* **A capped run is not automatically a good run.** Row 2 shows the ceiling alone leaves a
  1.93× overrun, because `min_shear_steps` is *also* a floor. The flag caught it
  (`violated`) — that is the flag working, and it is why the fix is two-sided.
* **The ceiling overshoots by one step.** The gate is evaluated after `strain += γ̇dt`, so a
  capped run can exceed the ceiling by up to one step of strain (4.82e-3 at I = 0.3, ≈ 0.96 %
  of a 0.5 request). `MAX_RECOVER_STRAIN_RATIO = 0.025` budgets for this explicitly.
* **`max_recover_strain=None` is deliberate.** Making the ceiling the default would have
  silently changed the meaning of every archived run. The default preserves bit-for-bit
  reproducibility; the archived round-15 runs were verified to reproduce to the last digit
  (`16.622514640912446`, 3447 steps, before and after the patch). The *drivers* opt in.
* **Only `phi_estimator_audit.py` was converted.** The other drivers
  (`full_cell_rheometry.py`, `bulk_stress_rheometry.py`, `rough_wall_sweep.py`,
  `round7_protocol.py`, `run_sweep.py`) still pass `min_shear_steps` without a ceiling.
  They now inherit the **flags** (so no silent invalid point can escape), but their
  *behaviour* is unbounded until the same two-line change is applied. That is the honest
  scope of this round.
* **No accuracy claim.** This round does not move μ(I) or φ(I) toward the reference. The
  ≈ 13 % φ offset and the unquotable high-I μ of round 16 are untouched, and the reference
  magnitudes remain PROVENANCE-UNVERIFIED (round 16's blocking item 1).
* **Not re-run at scale.** The six-point sweep with the fixed budget (round 16's item 3) and
  the pre-compacted preparation test (item 4) were not run here.

## ⑥ Error verdict (honest)

| quantity | criterion | measured | verdict |
|---|---|---|---|
| delivered strain vs request, historical path | ≤ 1.05× | 33.25× | **FAIL** (the defect, faithfully reproduced) |
| delivered strain vs request, round-17 protocol | ≤ 1.05× | **1.03×** (+3.2 %) | **PASS** |
| worst-case overrun, analytic vs round-16 archive | exact | 6/6 points identical (…, 98.93, 49.46×) | **PASS** |
| `None` path bit-for-bit | identical | 16.622514640912446 / 3447 steps both sides | **PASS** |
| new test module | all green | 15 passed | **PASS** |
| wider μ(I)/servo regression suite | all green | **not_run** at write time (cluster job `cj3163` still queued) | **WITHHELD** |
| μ(I), φ(I) vs reference (< 10 %) | < 10 % | unchanged from round 16 (−12.3 … −13.9 % φ; μ unquotable) | **FAIL** (anchor unchanged) |

**Is any key quantity within 3 % of its reference?** No accuracy quantity is quoted by this
round, so the 3 % question does not apply to the physics — and the anchor's own verdict is
unchanged at FAIL. The one quantity here with an external criterion is the delivered-strain
error, which is **3.2 % against a 5 % band**: it passes the band this round defines, and it
would *marginally fail* a 3 % bar. Stated that way deliberately, rather than rounding 3.2 %
down.

**Ledger verdict: still FAIL for the anchor, and that is correct.** This round removes one
named blocker (a protocol that could not deliver the strain it claimed) and replaces it with
a device that cannot produce a silently over-sheared point. The remaining blockers are
unchanged and are all *measurement* items: anchor the φ reference magnitudes, re-run
I = 0.03 … 0.3 at a budget that is now enforceable, and shear from a pre-compacted state.

### Recommended next steps

1. Apply the same two-line conversion (`min_shear_steps=0`, `max_recover_strain=ratio×strain`)
   to the other five drivers, then re-run the six-point sweep — the budget is now
   enforceable, which is what round 16 was missing.
2. Anchor φ_max/φ_min to an archived reading of da Cruz 2005 / GDR MiDi 2004 before any
   further φ comparison (round 16 item 1, still blocking).
3. μ needs ~10× the strain of φ (round 16 §3.3); with the ceiling in place this is now a
   cost decision rather than a correctness hazard.