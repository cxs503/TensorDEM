# Ice-ridge grounding: regenerated artifacts after the Skempton capacity fix

**Status of the debt closed here.** `docs/ridge_capacity_skempton_audit.md` (m2983)
found that the undrained subgrade capacity had been overstated by a double-counted
depth/shape factor, fixed it, and left two items `WITHHELD`:

1. the coupled DEM / subgrade / beam artifacts in `artifacts/ridge_grounding/` were
   **stale** — generated with the pre-fix capacity and never regenerated, and
2. the three seabed-strength numbers quoted in
   `docs/ice_ridge_pipeline_grounding.md` §7(d) (`73.8 / 221.5 / 442.9 kN/m`) were
   flagged as *"must be re-derived, not scaled by a constant factor"*.

This ticket re-executes the campaign runner on the fixed code and closes both.

## 1. Case (unchanged from the published campaign)

Beaufort-like ridge keel grounding on a buried pipeline, plane strain, unit
out-of-plane thickness; the case deck is the one hard-coded in
`scripts/run_ice_ridge_grounding.py` (it is *not* re-specified here, so the only
difference between the published numbers and these is the capacity fix and run
non-determinism of order 0):

| item | value |
|---|---|
| keel | depth 12.0 m, width 24.0 m, triangular, prescribed porosity, freeze-bond fraction |
| DEM | 159 disks on a triangular lattice, 419 initial bonds, 336 bonds broken by the end of the stroke |
| subgrade | `MohrCoulombSoil` undrained, `s_u = 30 kPa`, `B = 0.914 m`, cover 0.5 m (`z/B = 0.547`) |
| pipe | `BEAUFORT_SECTION`, one vertical DOF, peak-to-residual softening spring (0.9 peak, 0.8 residual) |
| platen | `v = 0.15 m/s`, stroke 0.882 m, `dt = 2.150e-4 s`, 40000 steps (step cap reached) |
| solver | `tensordem.ridge_grounding.run_grounding`, torch CPU, deterministic, single core |

## 2. What was done

```
PYTHONPATH=src:scripts OMP_NUM_THREADS=1 nice -n 5 python3 scripts/run_ice_ridge_grounding.py
```

Stages `beam,main,rate,sensitivity,calibration` (~10 CPU-minutes on one core, one
process — see §5 for the capacity-discipline note). Everything below is read back
from the machine artifacts (`artifacts/ridge_grounding/*.json`) and re-emitted as
`docs/artifacts/ridge_grounding_refresh/summary.json`; **no number in this document
is transcribed by hand**. The seabed-strength row of the sweep is a closed-form
re-evaluation of `SofteningVerticalSpring` (the same object the runner builds), not
a DEM run, which is what the published number always was.

## 3. Results

### 3.1 Seabed-strength sweep — the published numbers were 4/3 too high

| `s_u` | published peak reaction [kN/m] | regenerated peak reaction [kN/m] | regenerated ultimate [kN/m] | published / regenerated |
|---|---|---|---|---|
| 10 kPa | 73.8 | **55.36** | 61.52 | 1.3330 |
| 30 kPa | 221.5 | **166.09** | 184.55 | 1.3336 |
| 60 kPa | 442.9 | **332.18** | 369.09 | 1.3333 |

The ratio is the same to four significant digits at all three strengths — i.e. the
published row is a pure capacity-factor error, and the *shape* of the published
trend (linear in `s_u`) survives. It is not a rounding or a scaling artefact of the
DEM: `q ∝ s_u N_c` with `N_c` independent of `s_u`.

### 3.2 Reference coupled case

| quantity | published (doc §7) | regenerated | change |
|---|---|---|---|
| coupled peak capacity, 30 kPa @ 0.15 m/s [kN/m] | 221.5 (as quoted for 30 kPa) | **174.44** | published +27.0 % |
| subgrade ultimate `q_ult` [kN/m] | 221.5 (audit, pre-fix) | **184.55** | +20.0 % |
| subgrade peak reaction [kN/m] | — | 166.09 | — |
| residual [kN/m] | — | 140.15 | — |
| post-peak softening | 15.5 % | 19.65 % | shape criterion still green |
| rising-increment fraction | — | 0.717 | criterion `> 0.6` green |
| steps / particles / broken bonds | — | 40000 / 159 / 336 | step cap reached |

### 3.3 Loading-rate sensitivity and the zero-rate limit

| platen velocity [m/s] | coupled capacity [kN/m] | steps |
|---|---|---|
| 0.60 | 204.50 | 10853 |
| 0.30 | 184.65 | 21707 |
| 0.15 | 174.44 | 40000 |

Linear fit in velocity: `P(v) = 164.51 + 66.5 v` kN/m (the three points are
collinear to <0.1 %), so the **zero-rate limit is 164.5 kN/m** and the reported
`174.44` is **+6.0 % rate-influenced**. That extrapolation lands **0.95 % below the
subgrade peak reaction (166.09 kN/m)**, which is the physically expected closure:
at zero rate the coupled system is limited by the subgrade's peak branch, not by
the ice side. This is the first quantitative statement of that closure in the
repository, and it is a genuine constraint: the DEM cannot deliver more than the
subgrade peak at vanishing rate, and it does not.

### 3.4 Geometry sweeps — the published "flat within ±1 %" claim holds

| keel depth [m] | capacity [kN/m] | | keel width [m] | capacity [kN/m] |
|---|---|---|---|---|
| 8.0 | 192.36 | | 16.0 | 190.76 |
| 12.0 | 191.28 | | 24.0 | 191.28 |
| 16.0 | 190.18 | | 32.0 | 192.06 |

Depth: −0.57 % over 8→16 m; width: +0.34 % over 16→32 m. Both inside the published
±1 % statement, so that claim is reproduced rather than corrected. (These runs are
at `v = 0.40 m/s`, hence above the 0.15 m/s reference case; they are internally
consistent with each other, not with §3.2.)

### 3.5 Winkler beam against the Hetenyi closed form — the published 1 % claim does not survive

`beam_verification.json` is regenerated by the same run:

| load case | moment rel. error | deflection rel. error | profile moment rel. error | profile deflection rel. error |
|---|---|---|---|---|
| point load | **0.116 %** | 0.00002 % | 0.116 % | 0.00019 % |
| patch load | **1.63 %** | **4.47 %** | 4.30 % | 4.47 % |

The published claim (`docs/ice_ridge_pipeline_grounding.md` §3(c), §7(c)) was
"0.55 % moment, 0.11 % deflection ... agreement at the 1 % level". That reproduces
for the **point** load (and better: 0.12 %) but **not** for the **patch** load,
where the deflection error is 4.47 % and the profile moment error is 4.30 %. Both
stay inside the 15 % acceptance threshold but miss the 3 % target, so the honest
grade of criterion (c) is *point load REAL (<3 %), patch load REAL but 3-15 %* —
not "the 1 % level" as published. Whether the 4.5 % is a discretisation effect of
the 289-node FE mesh or a genuinely non-consistent patch-load distribution is not
settled here (a mesh refinement of the beam model alone would settle it cheaply;
it is not a DEM run) — recorded as an open, bounded follow-up.

## 4. Error verdict against the acceptance thresholds

Platform standard: relative error `> 15 %` is unacceptable, `< 3 %` is the target for
a *verified* quantity. Here the "reference" is the quantity recomputed on the fixed
physics; for the closure in §3.3 the reference is independent (the subgrade peak).

| key quantity | reference | published value | regenerated | error of the published value | < 3 % ? |
|---|---|---|---|---|---|
| subgrade peak, `s_u = 10 kPa` | 55.36 kN/m | 73.8 | 55.36 | **+33.3 %** | no — published value refuted |
| subgrade peak, `s_u = 30 kPa` | 166.09 kN/m | 221.5 | 166.09 | **+33.4 %** | no — refuted |
| subgrade peak, `s_u = 60 kPa` | 332.18 kN/m | 442.9 | 332.18 | **+33.3 %** | no — refuted |
| subgrade ultimate, 30 kPa | 184.55 kN/m | 221.5 | 184.55 | **+20.0 %** | no — refuted |
| coupled peak, 30 kPa @0.15 m/s | 174.44 kN/m | 221.5 | 174.44 | **+27.0 %** | no — refuted |
| zero-rate capacity vs subgrade peak | 166.09 kN/m | — | 164.51 | **0.95 %** | **yes** |
| doc §7(d) numbers vs artifacts (this revision) | `summary.json` | — | — | 0 % by construction (programmatic injection) | yes |
| patch-load Winkler deflection | 4.47 % (vs regenerated FE) | 0.11 % / 1 % level | 4.47 % | n/a (own FE vs closed form) | no — misses 3 % target, inside 15 % |
| depth/width flatness claim (±1 %) | measured 0.57 / 0.34 % | ±1 % | — | within claim | yes |

**Bottom line answered explicitly:** every number the campaign published for the
seabed-strength trend and the 30 kPa reference case was wrong by 20–33 %, i.e. above
the 15 % acceptance threshold and *not* acceptable. They are replaced above by the
regenerated values; the only <3 % agreement claimed here is the zero-rate/subgrade
closure (0.95 %).

## 5. Notes, limitations, and what is still not verified

* **Capacity discipline.** The box reported `load15 ≈ 128–155` on 128 cores for the
  whole window, so this ticket ran exactly **one** nice'd single-core process
  (~10 min), which is what the published reproduction recipe prescribes. The
  delivery cluster is **not usable for TensorDEM**: both loong64 nodes ship a bare
  CPython with no `numpy`/`torch` (`ModuleNotFoundError`), so the torch-based DEM
  cannot be shipped there. This is a platform-level finding, reported, not fixed
  (TensorPlatform is frozen).
* **Step cap.** The reference run hits the 40000-step cap at `v = 0.15 m/s`
  (`stroke/dt/v` would need more), so the record ends during post-peak softening.
  All shape criteria are evaluated on the record actually produced; the capacity is
  a peak value, which is reached before the truncation.
* **Rate extrapolation is linear in `v` only.** Three velocities, all collinear to
  <0.1 %, so the zero-rate limit is well determined *within this model family*, but
  the DEM's damping model is linear in velocity, which is exactly why a straight
  line is the right ansatz here and not evidence of quasi-static convergence.
* **`envelope.json` / `ledger.json` were regenerated by the same run** (stage
  `ledger`): the capacity envelope is soil-governed at every keel depth, with
  `peak_utilization = 0.0657` and `acceptable = true` for all three depths, and the
  machine ledger keeps `a_curve_shape = REAL`,
  `b_capacity_vs_limit_analysis = DERIVED + WITHHELD (Cross/DNV text)`. The
  published envelope numbers quoted in
  `docs/ice_ridge_pipeline_grounding.md` §7 are superseded by these.
* **Not regenerated:** `sensitivity_rigid.json` (ice-side rigid-pipe study,
  `scripts/run_ridge_rigid_sensitivity.py`, a separate multi-run campaign). The
  published `1.45 / 1.41 / 1.30 MN/m` ice-side numbers and the "5-7x stronger than
  the seabed" ratio are therefore still pre-this-ticket numbers and remain
  **WITHHELD**.
* **The `N_p = 2 + 1.5 z/D` lateral limit pressure** remains `WITHHELD` (no offline
  reference available in the worktree) — unchanged by this ticket.
* `artifacts/ridge_grounding/` is a build output and is not committed; the
  committed, citable copy of the numbers used in this document is
  `docs/artifacts/ridge_grounding_refresh/summary.json` (+ `ridge_refresh.png`),
  regenerated by the commands in §2.