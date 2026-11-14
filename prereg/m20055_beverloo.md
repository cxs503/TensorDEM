# Preregistration — CASE-DEM-SILO-BEVERLOO (m20055)

Written **before** any case in the matrix was run. (The only pre-registration
runs were two mechanics pilots used to size the case — a 205-particle and a
553-particle flat-bottom silo — whose sole purpose was to check that a
*settled* bed discharges at all; their numbers are reported as pilots and do not
enter the verdict table.)

## Decision

Does TensorDEM's flat-bottom silo DEM (`tensordem.silo_discharge`) reproduce the
Beverloo (1961) steady discharge law within the case tolerance of **< 10 %**?

## Competing alternatives (for the *measurement*, not for the physics)

* **A (baseline, already on record — m6296)**: ordered HCP bed, orifice open from
  step 0, rate = mean of five fixed 0.2 s windows starting at `t_settle`.
  Result on record: −77 … −86 % vs Beverloo, flow not steady (arch + start-up
  burst). Re-used here only as the control arm.
* **B (this case)**: disordered bed created by dropping a loose lattice
  (`settle_loose_gap = 0.3 r`, i.e. non-touching) into a **plugged** silo,
  settling it under gravity for `settle_time`, then opening the orifice; rate
  taken over the middle 60 % of the discharge stream (mass quantiles 0.2–0.8 of
  the discharged count) and split into 5 sub-windows for a steadiness check.
* **C (rejected)**: reuse of the conical-hopper kernel m15259 (`verification/
  granular/hopper_discharge/kernel.py`) — different geometry (hopper, not a
  flat-bottom silo), its run artifact is not on disk and its cluster job cj1037
  never executed, so it cannot serve as evidence; also rejected because the case
  asks for 料仓 (silo) discharge.

**Hypothesis**: on a disordered bed the discharge is steady and the measured
`W` agrees with Beverloo within 10 % for `D/d = 5`, μ_p = 0.30, smooth walls.

## Reference (fixed before running)

`benchmarks/m20055_beverloo/ref_beverloo_1961.json` — read by every script, no
number typed by hand into code or report:

    W = C · rho_b · sqrt(g) · (D − k·d)^(5/2),  C = 0.58, k = 1.4

Beverloo, Leniger & van de Velde, *Chem. Eng. Sci.* **15**(3-4), 260–269 (1961),
correlation Eq. (4), p. 265 (spherical particles). Provenance/limitation: the
1961 paper is not in this worktree and there is no outbound network, so the
original table could not be re-digitized — the coefficients are the published
pair as documented in the repository; this is recorded in the JSON
`provenance.verification_status`.

## Case matrix (6 runs, one shard each, single core)

| id | D/d | μ_p | dt | discharge window | purpose |
|---|---|---|---|---|---|
| D4 | 4 | 0.30 | default | 4.5 s | law check point |
| **D5** | **5** | **0.30** | default | 3.0 s | **primary case (verdict)** |
| D6 | 6 | 0.30 | default | 2.0 s | law check point |
| D5_mu15 | 5 | 0.15 | default | 3.0 s | sensitivity axis 1 |
| D5_mu45 | 5 | 0.45 | default | 3.0 s | sensitivity axis 1 |
| D5_dthalf | 5 | 0.30 | dt/2 | 3.0 s | sensitivity axis 2 (integration) |

Shared: d = 20 mm, ρ_p = 2500 kg/m³, μ_r = 0.02, e = 0.30, kn = 1e5 N/m,
kt/kn = 0.5, R_silo = 0.130 m (silo/D = 2.6 at D/d = 5), loose column height
0.40 m, plate thickness 0.5 d, wall μ = 0, seed 20055, settle 0.60 s.

## Metrics

* `W` = steady mass flow rate [kg/s] from the discharge-event record
  (`steady_rate`, mass quantiles 0.2–0.8, 5 sub-windows).
* `rho_b` = **bed** bulk density measured at the instant the orifice opens.
* `W_ref` = Beverloo at the same `D`, `d`, `rho_b`, `g`.
* primary error = (W − W_ref)/W_ref; secondary = recovered `C` at k = 1.4.
* steadiness = CV of the 5 steady sub-windows.

## Predeclared thresholds and stopping rule

* **PASS**: |error| < 10 % *and* CV < 0.15 *and* ≥ 300 particles discharged.
* **MARGINAL**: 10–15 %.
* **FAIL**: > 15 %, or the flow is not steady (CV ≥ 0.15), or the bed arches
  (no discharge).
* Stopping rule: no case is re-run with different parameters until the full
  6-case matrix has been analysed; the verdict is the primary case D5. No
  parameter is changed after seeing a result — the matrix above is frozen.

## Variables / confounders

* Controlled: particle size, density, contact law, wall friction, geometry
  ratios, initial bed construction, seed, integration step.
* Varied (one at a time): D/d, μ_p, dt.
* Confounders: bed-density drift with settling time (measured, not assumed —
  ρ_b used in the reference is the measured bed value); wall effect at
  silo/D = 2.6 (bounded by the D/d sweep); Poisson noise of window counts
  (3.5–3.7 k discharged particles, noise ≈ 2 %); rolling friction (fixed at
  0.02, not swept).

## Analysis (fixed in advance)

Aggregation = steady-window mean; uncertainty = 1σ across the 5 sub-windows and
the Poisson floor; outliers = none removed; comparison = relative error vs the
same-parameter Beverloo value, plus a log–log slope of W vs (D − k d) compared
with the 5/2 exponent of the law.

## Environment / resource limits

702 cluster (loong64, scheduler-assigned node), 1 core per shard, `timeout`
3600 s per shard, TensorDEM at branch `agent/m20055` (git sha recorded in each
JSON), NumPy as pinned by the repository.

## Interpretation limits

Empirical correlation, not a first-principles law: ±10 % scatter is normal even
between two experiments. Monodisperse spheres, smooth walls, flat bottom,
`D/d ≤ 6`; no polydispersity, no wall friction, no humidity/cohesion. Agreement
within 10 % supports the contact law + geometry model, it does **not** prove
Beverloo's C and k are universal.

## Baseline arm

The m6296 control arm (ordered bed, fixed windows) is not re-run: its recorded
artefacts are in `runs/m6296_silo/` and `docs/silo_discharge_beverloo_verification.md`,
and re-running it cannot change the conclusion (−77 … −86 %).