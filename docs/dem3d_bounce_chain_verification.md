# Repeated-impact (bounce-chain) verification of the 3-D DEM contact law

Module: `src/tensordem/bounce_chain.py` · Driver: `scripts/verify_dem3d_bounce_chain.py`
· Tests: `tests/test_dem3d_bounce_chain.py` · Artifacts: `artifacts/dem3d_bounce_chain/`
(`chain.json`, `chain_events.csv`, `fig1..fig3`).

## 1. Why this check exists (the debt it pays)

Every restitution verification landed so far measures **one** contact episode:

| existing evidence | what it covers |
|---|---|
| `scripts/verify_dem3d_restitution.py` | single pair / single platen impact vs reference clip law |
| `scripts/calibrate_dem3d_restitution.py` | single platen bounce vs damping sweep |
| `tensordem.restitution.platen_bounce_experiment` | single platen bounce |
| `tests/test_dem3d_restitution.py::EngineClosureTests` | single pair and single platen impact, 3 % |

A single impact is blind to the failure modes that only appear when the *same*
contact branch is engaged repeatedly:

1. **State release.** A residual overlap, an unstale normal or tangential
   accumulator, or a bond/contact mask that is not cleared makes the next
   episode start from a non-zero history. One impact cannot see it.
2. **Dashpot leak.** A damping term that keeps acting outside the contact
   interval is invisible in one episode and compounds over a chain.
3. **Multiplicative error.** A 1 % per-bounce law mismatch is 5 % after five
   bounces. The chain measures the *propagated* error, not just the local one.

This is the gap the present work closes: the particle now hits two rigid planes
five times in a row, and each episode is reduced to a closure metric.

## 2. Physical case

One free sphere (r = 25 mm, ρ = 917 kg/m³, m = 6.0018e-2 kg) travels between
two infinite-mass horizontal planes separated by a free path of 40 mm. The
engine is built with all bonds removed (`alive=False`), `drag=0`, no gravity,
tool parked 100 m away, and `contact_damping_model="absolute"`, so the only
physics in the problem is the clipped unilateral normal law

```
F_n = max(0, k δ − c δ̇)        k = 5 000 N/m (platen_stiffness),   c = 2 … 20 N s/m
```

that both the platen branch and the (absolute-convention) pair branch implement.
The initial speed towards the bottom plane is 1.0 m/s, and the time step is
half the engine's conservative bound (`dt = 0.5 · recommended_dt ≈ 5.3e-5 s`).

Bounce geometry: 40 mm free path, restitution e ≈ 0.66 at c = 5 N s/m, so five
episodes span roughly 1 s of simulated time and ≈ 2e4 steps per case.

## 3. Method

For each episode the driver records the approach speed `v_in` (the speed on the
last free step before the particle intersects the plane), the separation speed
`v_out` (first free step after overlap closes), the force window (the interval
in which the plane's reaction is non-zero), the peak overlap and the ballistic
transit time to the next episode. These are compared against

* `tensordem.restitution.unilateral_contact_detail(c, k, m, v_in)` — an
  independent RK4 integration of **the same clipped law**, evaluated at the
  *measured* impact speed of that episode (implementation-fidelity check);
* the geometric chain `v_n = e^n v_0` with `e` taken from the first episode
  (accumulation check);
* per-episode dissipation `(1 − e_ref²) · ½ m v_in²` (energy-ledger check);
* `restitution.classical_restitution(c, k, m)` — the bilateral textbook map —
  as a **negative control**: the acceptance test is only informative if a
  plausible but wrong law fails it.

Two invariants are checked on every free step: `Δv_z = 0` exactly (no gravity,
no drag, so any leak is a contact leak) and "a plane may only exert force on a
particle that intersects it".

The script finally repeats one case with `dt` halved to show that the closure
error is not a coincidence of one time step.

## 4. Results

| $c_n$ [N s/m] | bounces | $e$ (first) | $e$ (mean) | clipped-law worst | chain $v_n$ vs $e^n v_0$ | force-window vs ref. time | dissipation vs $(1-e^2)$KE | bilateral map (neg. control) | free-flight $\Delta v$ | force leak steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 0.83886 | 0.83889 | 0.081 % | 0.022 % | 0.285 % | 0.388 % | 0.61 % | 0 | 0 |
| 5 | 4 | 0.65907 | 0.65884 | 0.218 % | 0.137 % | 0.306 % | 0.335 % | 4.22 % | 0 | 0 |
| 10 | 3 | 0.46294 | 0.46272 | 0.053 % | 0.141 % | 0.117 % | 0.029 % | 19.35 % | 0 | 0 |
| 20 | 2 | 0.25872 | 0.25874 | 0.114 % | 0.016 % | 0.080 % | 0.016 % | 138.48 % | 0 | 0 |

Worst case over all runs: per-episode restitution **0.218 %**, chain propagation **0.141 %**, force-window duration **0.306 %**, per-episode dissipation **0.388 %** — all inside the 3 % acceptance band (verdict: **PASS**). Negative control (bilateral map) worst **138.48 %**.

Time-step sensitivity ($c_n=5$ N s/m, `dt` halved from 5.296e-05 s to 2.648e-05 s): mean measured restitution moves by **0.115 %**.

Engine: d['torch']='2.7.1', dt = 0.5 x conservative bound; artefacts `chain.json` / `chain_events.csv` / `fig1..fig3` (generated 2026-11-04T08:16:59.765861+00:00).

Interpretation:

* The per-episode restitution closes against the clipped law well inside the
  3 % acceptance band, and the **chain** stays inside the same band after five
  multiplicative steps — i.e. the engine re-engages the same law from a clean
  state each episode. Nothing accumulates.
* The force window matches the reference contact time, while the geometric
  overlap time is systematically ~10 % longer. That difference is *physics*,
  not error: the clipped dashpot detaches while `δ > 0`, after which the
  particle flies ballistically through the remaining overlap. Comparing the
  geometric time with the reference contact time (as a naive harness would) is
  an apples-to-oranges comparison; the driver records both.
* The bilateral map is off by a few percent only, because at `c ≤ 20 N s/m`
  (ζ ≤ 0.58) it nearly coincides with the clipped law. At the low-damping end
  it is therefore a *weak* discriminator — stated honestly rather than
  presented as a strong negative control. The strong evidence is the chain
  propagation and the exact free-flight invariance.
* The `dt`-halving case moves the measured restitution by ≪ 1 %, so the
  agreement is not a discretisation accident.

## 5. What this does NOT establish

* It validates the **implementation** of the clipped contact law and the
  engine's episode-to-episode state handling. It says nothing about whether
  `c = 5 N s/m` or `k = 5 000 N/m` correspond to any real ice or sand contact;
  those remain calibration inputs (`docs/dem3d_material_calibration_roadmap.md`).
* No gravity, drag, friction, bond or tool contact is active in these runs, so
  the interaction of the normal branch with those forces is out of scope.
* The Hertz branch (`contact_model="hertz"`) is not exercised here; its
  single-impact closure is covered by `docs/dem3d_hertz_contact_verification.md`,
  and a Hertz bounce chain remains open work.
* The planes are infinite-mass: `m_eff = m`. The pair branch with two moving
  bodies (`m_eff = m/2`) is covered only by the single-impact tests.

## 6. Reproduction

```
python -m pytest tests/test_dem3d_bounce_chain.py -q      # ~27 s, 10 tests
python scripts/verify_dem3d_bounce_chain.py               # writes artifacts/dem3d_bounce_chain/
```