# Interface contract: level-ice breakup → clear-ice floe field (`clear-ice-contract/1.0`)

Status: **implemented and tested** (`src/tensordem/clear_ice/contract.py`,
`tests/test_clear_ice_study.py`).
Owner of each side:

| side | repository module | owns |
|---|---|---|
| breakup (producer) | `tensordem.level_ice` (`sheet.py`, `formula.py`, `campaign.py`) | how a sheet breaks: bond rupture, fragment sizes and shapes, the resistance anchors |
| floe field (consumer) | `tensordem.clear_ice` | how the resulting floes move, the force network, the porosity statistics, the load on a rigid hull face |

Neither side imports the other's internals. The hand-over is data.

## 1. Why a contract at all

The level-ice sheet produces a *bonded* continuum that is broken into fragments;
the clear-ice field is a *rigid-disk* granular assembly.  The two have different
state variables, different time steps and different evidence.  The failure mode
of an implicit coupling is silent: an area leak between the two changes the ice
concentration, and the concentration is the single parameter that conditions
every statistic the floe field reports (`P(f)`, `Z`, `n`).  So the contract
carries an **area ledger** and the consumer refuses a population that does not
close it.

## 2. `FloePopulation` (producer → consumer)

| field | type | meaning |
|---|---|---|
| `x` | `(N, 2) float` | fragment centroid in the channel frame, m |
| `radius` | `(N,) float` | equivalent radius `r = sqrt(A/pi)`, m |
| `thickness` | `(N,) float` | fragment thickness, m |
| `parent_area` | `float` | area of the parent sheet that was broken, m² |
| `source` | `str` | provenance tag of the producer |
| `contract_version` | `str` | `clear-ice-contract/1.0` |
| `metadata` | `dict` | free-form (seed, breakup criterion, ...) |

Derived: `area = sum(pi r^2)`, `hash = sha256(x, radius, thickness)` over the
float64 arrays.  The hash is the audit key: any result can be traced to the
fragment set that produced it, and the same fragment set reproduces the same run
(test `test_population_handover_reproduces_the_field`).

Producer helper: `population_from_breakup(fragment_area, centroids, thickness)`
converts the areas that a bonded-particle breakup actually produces into
`r = sqrt(A/pi)` and sets `parent_area = sum(A)` — **area conservation by
construction**, verified afterwards by the consumer.

## 3. `validate_population` (consumer-side gate)

Machine-checkable, all four recorded in the result provenance:

1. `area_ledger` — `|sum(pi r^2) - parent_area| / parent_area <= 1e-3`
2. `containment` — centroids inside the delivery domain
3. `non_overlap` — `max((r_i + r_j - d)/min(r_i,r_j)) <= 5e-2` (periodic metric)
4. `concentration` — the areal concentration the population implies in the
   domain; this is the quantity the floe-field statistics are conditioned on.

A population that fails 1 or 3 is rejected, not repaired: repairing it in the
consumer would move the area leak out of sight.

## 4. `ClearIceRequest` (driver → consumer)

`domain (Lx, Ly)`, `gate_width` (hull face span), `drift_speed` (imposed ice
drift), `duration`, `dt`; plus the population.  Material and contact parameters
live in `FieldConfig` / `ContactMaterial` and default to the columnar-ice values
documented in `docs/CLEAR_ICE_DEM_REGULARIZATION.md`.

## 5. `ClearIceResult` (consumer → any)

The five blocks the campaign node requires, all mandatory at the boundary
(`ClearIceResult.validate_payload` raises if one is missing):

| block | content |
|---|---|
| `floe_kinematics` | drift velocity, mean/rms speed, granular temperature, per-floe displacement statistics, wall-normal velocity profile, resolution quality (`oscillation_index`, high-frequency fraction, max overlap) and the negative control |
| `force_chain_stats` | `P(f)` histogram with the exponential fit (`f0`, `R^2`), power-law alternative and BIC difference, weak-contact fraction, load share above `3 <F_n>`, coordination number `Z` (with and without rattlers) |
| `porosity` | domain-mean and region-resolved porosity by two independent estimators (areal coverage, periodic Voronoi free volume), estimator gap, criterion band |
| `convergence_report` | deterministic trajectory-order study, ensemble `dt` sweep with per-metric observed order, particle-count study, contact time and the `dt` stability rule, hull load |
| `uncertainty_quantification` | one-at-a-time ±20 % sensitivities of the reported statistics plus the seed scatter |
| `provenance` | contract version, population hash and audit record, full config |

## 6. Versioning rules

* Adding an optional field → minor bump (`1.x`), consumer tolerates unknown keys.
* Removing/renaming a required block or changing a unit → major bump (`2.0`);
  the consumer rejects a `2.x` payload (`test_population_hash_is_deterministic_and_roundtrips`).
* The `population_hash` algorithm is part of the contract: changing it is a
  major bump, because it invalidates stored audit trails.

## 7. What the contract deliberately does *not* do

* It does not transfer fragments' *shape*: the floe field takes an equivalent
  disk.  Shape enters only through the equivalent radius (area-conserving), so a
  shape-sensitive quantity (e.g. interlocking) must be reported by the breakup
  side, not inferred here.
* It does not carry a buoyancy/draught state: the floe field assumes the
  flat-lying hydrostatic equilibrium of the whole population (stated limitation,
  see the main report).
* It does not couple back: the floe field never modifies the fragment set.  A
  two-way coupling (fragments breaking further under the contact load) would be
  a new contract version with a different owner.