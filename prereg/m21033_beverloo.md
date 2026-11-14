# Preregistration — CASE-DEM-SILO-BEVERLOO (m21033)

Written **before** any case in the matrix was run.

## Decision

Does TensorDEM's flat-bottom silo DEM (`tensordem.silo_discharge`) reproduce the
Beverloo (1961) steady discharge law within **< 10 %** (ledger tolerance for
`CASE-DEM-SILO-BEVERLOO`) once the contact stiffness is raised to the value
that already passed on the hopper kernel (m15259, −0.4 %)?

## Background on record (no re-derivation)

* m20055: settled+plug protocol, `kn = 1e5 N/m`, μ = 0.30 → D4 **+25.0 %** FAIL
  (`docs/CASE-DEM-SILO-BEVERLOO_m20055.md`).
* m20585: root-cause diagnosis — soft contacts (kn = 1e5) enlarge the effective
  orifice; the hopper kernel at `kn = 1e6 N/m` reached **−0.4 %** (m15259).
  Recommendation: re-run the silo matrix at `kn = 1e6`.
* m15259 (hopper kernel, conical 30°): `kn = 1e6`, μ = 0.15, e = 0.5,
  μ_rolling = 0, wall smooth, D/d = 5 → **−0.4 %** PASS.

## Protocol (identical to m20055 except where marked)

* Geometry: flat-bottom circular silo, centred orifice, smooth side wall
  (wall_mu = 0, glass-bead condition), plate thickness 0.5 d.
* Bed: loose HCP lattice (clearance 0.30 r, jitter 0.05 r) dropped into a
  **plugged** silo, gravity-settled for `settle_time = 0.60 s`; bed height and
  bed bulk density ρ_b measured at the instant the orifice opens.
* Rate: middle 60 % of the discharge stream (mass quantiles 0.2–0.8), split
  into 5 sub-windows; CV reported as a steadiness check.
* Reference: `W_ref = C ρ_b √g (D − k d)^(5/2)` with C = 0.58, k = 1.4 read
  from `benchmarks/m20055_beverloo/ref_beverloo_1961.json` (machine-readable,
  zero hand-typed numbers). ρ_b = measured bed density of the *same* case.

## Predeclared case matrix

Shared: d = 20 mm, ρ_p = 2500 kg/m³, e = 0.30, μ_r = 0.02, wall_mu = 0,
silo R = 0.130 m, fill = 0.40 m, seed = 21033, **kn = 1e6 N/m** (changed from
m20055's 1e5 — the single hypothesis under test), **μ = 0.15** (changed from
0.30 to the glass-bead value that passed on the hopper kernel).

| id         | D/d | μ_p  | kn [N/m] | dt        | purpose                     |
|------------|-----|------|----------|-----------|-----------------------------|
| D5         | 5   | 0.15 | 1e6      | default   | **primary (verdict)**       |
| D4         | 4   | 0.15 | 1e6      | default   | law check point             |
| D6         | 6   | 0.15 | 1e6      | default   | law check point             |
| D5_mu30    | 5   | 0.30 | 1e6      | default   | friction sensitivity        |
| D5_kn1e5   | 5   | 0.15 | 1e5      | default   | kn attribution arm          |
| D5_dthalf  | 5   | 0.15 | 1e6      | dt/2      | time-step convergence       |

## Verdict rules (fixed in advance)

* PASS iff |W − W_ref| / W_ref < 10 % for the primary case **D5**.
* D4/D6 are law-check points (reported, not gating).
* D5_kn1e5 is an attribution arm: if it reproduces the m20055-style
  over-prediction while D5 passes, the stiffness hypothesis is confirmed.
* D5_dthalf: |W(dt/2) − W(dt)| / W(dt) < 5 % required for the dt to be
  declared converged.
* Any case that does not reach a steady state (CV > 0.30 or < 40 % of the bed
  discharged) is reported as a protocol failure, not a law failure.
