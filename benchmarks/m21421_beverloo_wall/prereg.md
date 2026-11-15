# Pre-registration — CASE-DEM-SILO-BEVERLOO: wall-proximity scan (m21421)

Written **before** any case of this matrix was run. Anchor slug:
`CASE-DEM-SILO-BEVERLOO` (ledger status `open`, tolerance **< 10 %**).

## State of the record on entry (no re-derivation)

| kernel / case | condition | error vs Beverloo | source |
|---|---|---|---|
| hopper (conical 30°) | kn = 1e6, μ = 0.15, D/d = 5 | **−0.4 %** PASS | m15259 |
| silo_flow (d = 3 mm) | kn = 5e3, μ = 0.50, silo_D/D = 3.33 | D5 **−6.9 %**, D6 **+5.9 %** PASS | m20707 |
| silo_discharge (d = 20 mm) | kn = 1e5, μ = 0.30, silo_D/D = 2.0 | D4 **+25.0 %** FAIL | m20055 |
| silo_discharge (d = 20 mm) | kn = 1e6, μ = 0.15, silo_D/D = 2.6 | D5 **+20.2 %** FAIL | m21033 (cj3201/n315) |

m21274 falsified the "soft contact (kn) is the first-order root cause" hypothesis
(+25 % at kn = 1e5 → +20 % at kn = 1e6) and ranked **wall proximity** first:
the failing cases sit at/just above the Beverloo applicability boundary
`silo_D/D ≥ 2.5` (ref JSON field `validation_domain.silo_D_over_D_min`, read
from `benchmarks/m20055_beverloo/ref_beverloo_1961.json`), whereas both passing
kernels sit at `silo_D/D ≥ 3.3`. When the annular stagnant zone no longer fits
between the orifice and the side wall the whole bottom participates in the flow,
the effective aperture exceeds `D − k·d`, and `C_recovered` is inflated.

## Question

Is wall proximity the cause of the +20 % over-prediction of the
`tensordem.silo_discharge` kernel, i.e. does the kernel reproduce Beverloo
within the ledger tolerance once the case is placed **inside** the applicability
domain (`silo_D/D ≥ 3.5`)?

## Design

Everything except `silo_radius` is held fixed, so the scan is a clean one-variable
experiment: bed height stays at **H = 0.400 m (H/D = 4.0)** in every case, so the
column of material above the orifice is identical and only the distance from the
orifice rim to the side wall changes. This removes the third candidate root
cause of m21274 (bed height / system scale) as a confound. The particle count
necessarily grows with the silo cross-section (~1525 / ~2765 / ~3610), which is a
statistical benefit paid for in wall-clock.

Shared: d = 20 mm, ρ_p = 2500 kg/m³, μ_p = 0.15, μ_r = 0.02, e = 0.30,
kn = 1e6 N/m, kt/kn = 0.5, g = 9.81 m/s², D = 0.100 m (**D/d = 5**),
plate thickness 0.010 m (0.5 d), wall_mu = 0 (smooth, glass-bead condition),
settled-lattice + plugged-then-opened protocol (settle 0.60 s),
discharge 3.0 s, 5 windows of 0.50 s, steady window = mass quantiles 0.20–0.80,
seed = 21033 (**same random draw for every case**, so the scan is paired).

| id | silo R [m] | silo_D/D | fill H [m] | H/D | n_bed (expected) | purpose |
|---|---|---|---|---|---|---|
| S26 | 0.130 | 2.6 | 0.400 | 4.0 | ~1525 | baseline: reproduces m21033 D5 (expect +20 %) |
| S35 | 0.175 | 3.5 | 0.400 | 4.0 | ~2765 | inside the applicability domain |
| S40 | 0.200 | 4.0 | 0.400 | 4.0 | ~3610 | deep inside the applicability domain |
| F26 | 0.130 | 2.6 | 0.400 | 4.0 | ~1525 | protocol control: orifice open during settle (no plug precompaction), otherwise identical to S26 |

Reference: `W_ref = C ρ_b √g (D − k d)^(5/2)`, `C = 0.58`, `k = 1.4` read from
`benchmarks/m20055_beverloo/ref_beverloo_1961.json`; `ρ_b` = bulk density of the
settled bed of the *same* case (`bulk_density_bed_kg_m3`), unchanged from
m21033 — **no reference redefinition is allowed by this pre-registration**.

## Second, independent flux estimator (new in this shift)

`SiloCase.measure_orifice_flux` meters the Eulerian mass flux through the
mid-plate aperture plane (`z = −t/2`, `radial < a`). `W_flux` (crossings of the
plane) and `W_count` (`steady_flow_rate_kg_s`, particles that ended below the
plate) measure the same steady flow by independent bookkeeping.

Pre-registered consistency condition: `|W_flux − W_count| / W_count < 3 %` in
every case. A larger gap invalidates `W_count` and must be reported as a
kernel defect rather than a law failure.

## Pre-registered predictions and verdict rules

* **H1 (wall proximity is the root cause).** `|err|` decreases monotonically
  with `silo_D/D`, and the primary acceptance `|err(S40)| < 10 %` holds, i.e.
  `C_recovered(S40) ∈ [0.522, 0.638]`.
* **H1 strong form.** `C_recovered(S40) ∈ [0.55, 0.60]` (inside Beverloo's own
  quoted band for spheres).
* **Falsification of H1.** `|err(S40)| ≥ 10 %`. Then wall proximity is *not* the
  first-order cause, the anchor stays FAIL, and the next candidate (density
  definition / flowing-state dilatancy, or the plate geometry) is promoted —
  reported honestly, no redefinition of `ρ_b`.
* **H2 (the settled+plug protocol is not an artifact).** `|W(F26) − W(S26)| /
  W(S26) < 5 %`.
* A case is a **protocol failure** (not a law failure) if the steady window
  CV > 0.30 or fewer than 40 % of the bed is discharged.
* Ledger acceptance for the anchor is declared **met** only if the primary
  case inside the applicability domain passes **and** H2 holds and the flux
  consistency condition holds.

## Honest limits fixed in advance

* `C`, `k` are cited (not re-digitized: no network, no 1961 original).
* One seed per geometry; scatter assessed by the 5 sub-windows only.
* `silo_D/D` is scanned by *growing the silo*, not by shrinking the orifice;
  keeping D/d = 5 means the particle size relative to the orifice is fixed
  (which is what the law scales with), while the number of particles grows with
  the silo cross-section — a statistical benefit, but the runs are ~2× costlier.