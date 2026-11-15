# Preregistration — m20055 dt-convergence ladder (D5)

Written **before** any ladder run. This is an *additive* study on top of the
closed-out CASE-DEM-SILO-BEVERLOO (m20055) deliverable. It does not re-litigate
the filed honest FAIL; it asks a narrower, answerable question about it.

## Why (the observation that motivates the study)

The filed matrix (docs/CASE-DEM-SILO-BEVERLOO_m20055.md) contains one
integration-sensitivity arm:

| case | dt | W [kg/s] | rel. err vs Beverloo (C=0.58, k=1.4) |
|---|---|---|---|
| D5        | dt        | 3.4069 | +20.6 % |
| D5_dthalf | dt/2      | 3.1036 |  +8.9 % |

Halving the time step moved the answer by 8.9 % of the reference — in the
direction of the literature value. With only two levels the size and even the
existence of a converged limit is unknown, so the filed verdict ("+20.6 %,
honest FAIL") and the tolerance judgement on it both rest on an unquantified
discretisation error.

## Decision this study supports

**Can the m20055 verdict be stated at the discretisation limit?** Concretely,
for the primary case D5 (D/d = 5, mu_p = 0.30, smooth wall, settled bed) we
report the discharge rate *extrapolated to dt -> 0* and its error against the
same fixed reference (C = 0.58, k = 1.4 read from
`benchmarks/m20055_beverloo/ref_beverloo_1961.json`), and decide:

* **PASS-CONVERGED-IN-TOL**: extrapolated error < 15 % (platform precision
  standard) -> the filed +20.6 % is a discretisation artefact, not a physics
  defect, and the case is admissible for the platform.
* **FAIL-BUT-QUANTIFIED**: extrapolated error >= 15 % -> the overprediction is
  physical (contact model / bed preparation), and its size is now separated from
  the dt contribution.
* **INCONCLUSIVE-NO-ORDER**: the ladder is not monotone, or the observed order
  p lies outside (0, 4] -> no extrapolation is claimed; only the ladder itself
  is reported.

## Competing alternatives / hypotheses

* **H1 (numerical)**: the +20.6 % is dominated by temporal discretisation of the
  contact law (dt/T_contact = 0.0354 at the default step). Prediction: W(dt)
  decreases monotonically with decreasing dt and approaches the literature value.
* **H2 (physical)**: the offset is caused by the DEM contact/bed model (loose
  settled bed, phi_bed = 0.519 on the filed D4 arm; mu_rolling, restitution),
  and does not vanish as dt -> 0. Prediction: extrapolated W stays ~15-25 %
  above Beverloo.
* The two are separable by the ladder: H1 is supported iff the extrapolated
  error collapses well below the dt-level error, with a consistent positive
  observed order.

## Outcome metric and unit of analysis

* Outcome metric: `W` = steady mass flow rate [kg/s] from
  `tensordem.silo_discharge.steady_rate` with the **predeclared** quantile band
  0.20-0.80 of the discharge stream, `n_windows = 5` (unchanged from the filed
  case, so the ladder is comparable with it).
* Unit of analysis: one case run per level, identical case configuration except
  `dt`. Particle count, seed (20055), bed preparation, settle time, discharge
  time, wall condition, friction, plate thickness: all unchanged.
* Derived: relative error vs the fixed Beverloo prediction,
  observed order p, Richardson extrapolation, and the k-uncertainty band.

## Ladder (3 levels, one shard each)

| level | dt | dt / T_contact | expected steps |
|---|---|---|---|
| `dt`        | 7.188680532649903e-05 (default) | 0.0354 | ~50 k |
| `dthalf`    | dt/2 | 0.0177 | ~100 k |
| `dtquarter` | dt/4 | 0.0088 | ~200 k |

`dteighth` (dt/8, ~400 k steps) is *not* predeclared: it is only run if the
observed order from three levels is in (0,4] **and** the two-level extrapolation
with p = 1 and p = 2 disagree by more than 5 % (i.e. the limit is not yet pinned).
Stopping rule: stop the ladder after three levels unless that condition holds.

## Analysis fixed in advance

1. Standardised rate differences: `a_i = (W_i - W_{lit}) / W_{lit}`, levels
   i = 0 (dt), 1 (dt/2), 2 (dt/4).
2. Observed order from the three-level Richardson estimator with refinement
   ratio r = 2:
   `p = log2( (W_0 - W_1) / (W_1 - W_2) )`, defined only if the two differences
   have the same sign and neither is zero.
3. Extrapolated limit `W_inf = W_2 + (W_2 - W_1) / (2^p - 1)`.
4. Robustness band: the same extrapolation with p fixed at 1 and at 2, plus the
   coarser `W_1 + (W_1 - W_0)/(2^1 - 1)` (p = 1, two levels) — the decision is
   taken on the *band*, not on the single p-from-data value.
5. Error of the limit: `(W_inf - W_lit)/W_lit`. Decision thresholds as above
   (15 % = platform standard; 10 % = the case's own filed tolerance; 3 % is
   reported for the paper-grade column but is **not** a pass criterion here,
   because the reference pair (C, k) is cited, not re-digitised — see the
   provenance note below).
6. Parameter-identifiability check (arithmetic, no extra runs): because a single
   orifice diameter is compared, the literature k-range [1.2, 1.6] maps to a
   W-range through `(D - k d)^{5/2}`. This band is reported for every level, so
   the reader can see how much of any residual "error" is the literature's own
   k-uncertainty rather than DEM.
7. Per-sub-window coefficient of variation (CV) is reported as the steadiness
   check; a level whose CV exceeds 0.30 is flagged as a noisy measurement and is
   not used for the order estimate.

## Exclusions / missing data

* A level that fails to run (DEM blow-up, empty run, fewer than 5 sub-windows)
  is recorded as `INVALID` in the summary and the order is computed from the
  valid subset only (two levels -> p = 1 extrapolation, labelled as such).
* No result is discarded for being inconvenient; the raw per-level JSON is
  committed under `runs/m20055_beverloo_dt_ladder/`.

## Reference provenance and its limit (unchanged from m20055)

`ref_beverloo_1961.json` carries C = 0.58, k = 1.4 with
`verification_status = "cited, NOT re-digitized"` (no copy of the 1961 paper and
no outbound network in this worktree). A residual few-% bias in C is therefore
*inside* the reference itself and cannot be separated from the DEM error. This
is why the pass criterion for this study is stated in terms of the 15 % platform
standard and the k-band, never as a "< 3 %" claim.

## Reproducibility record

Each level records git sha, host, platform, python/numpy versions, dt,
steps, seed, the exact command line, the full discharge event record and the
steady-rate sub-windows. Long runs are submitted through the scheduler
(`run_compute submit`, `stage_dir` = this worktree), one shard per level.

    PYTHONPATH=src python benchmarks/m20055_beverloo/dt_ladder.py --level dthalf
    PYTHONPATH=src python benchmarks/m20055_beverloo/dt_ladder.py --analyse