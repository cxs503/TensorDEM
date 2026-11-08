# Rate-dependent ice bond: engine wiring and bonded-specimen rate campaign

Status: **closed at the engine + specimen level, single-machine; laboratory
calibration of the spectrum remains WITHHELD** (no digitised ice test data in
this repository).

Predecessor node: `docs/ice_bond_viscoelastic_verification.md` verified the
Maxwell-Wiechert (Prony) bond law and the Kachanov damage clock at the
*constitutive* (single-bond) level and recorded, in
`docs/dem_calibration_gap_ledger.md`, the open part: *"no engine call site —
`dem3d.py` still uses the linear bond, damage is not fed back into the bond
force, and no bonded-specimen rate-dependence benchmark exists."* This node
closes the first two and supplies the third.

## 1. What changed

`src/tensordem/dem3d.py` (`IceDEM3D`) gains two **independent, default-off**
extensions:

| knob | default (legacy) | effect when set |
|---|---|---|
| `bond_model` | `"linear"` | `"maxwell_wiechert"` gives the bond a Prony relaxation spectrum: `F = k0 d + sum_a k_a (d - d_a)`, `d(d_a)/dt = (d - d_a)/tau_a` |
| `bond_creep_exponent` | `0` | `m > 0` enables the Kachanov clock `dD/dt = (1/t_ref) (|F|/F_ref)^m`, bond deleted at `D = 1` (failure mode bit 4) |

Two implementation points matter for correctness:

* **The constitutive force is evaluated before the failure commits.** A
  rate-dependent bond must be judged on the force it actually carries, not on
  the geometric strain. For the linear law the number is unchanged, so the
  legacy trajectory is bit-for-bit identical (asserted by
  `test_default_bond_model_is_linear_and_bit_for_bit_legacy`).
* **Deleted bonds park their branch offsets on the current elongation**
  (`s_a = 0`) and their dissipation is masked out of the ledger, so a broken
  bond cannot keep dissipating energy that is no longer in the system.

Energy accounting follows the constitutive law: `bond_stored_energy()` returns
`k0 d^2/2 + sum k_a s_a^2/2` per bond (identical to the legacy expression when
`bond_law is None`) and is used by `energy()`, `diagnostics()`,
`EnergyAuditedIceDEM3D._mechanical_energy()` and as the release energy recorded
at bond deletion. The Maxwell dashpot dissipation accumulates in
`cumulative_bond_dissipation_J` and enters the audited ledger's dissipation
side. Checkpoints carry `branch_offsets`, `damage`, `elongation_prev` and the
dissipation accumulator under the `bond_law` key; a rate-dependent solver
refuses a checkpoint without it, and legacy checkpoints still load.

`TriaxialConfig` forwards the same knobs, so the existing uniaxial
(`ucs_opposed`) and true-triaxial protocols can run the rate-dependent bond.

## 2. The damage clock is a *time* criterion, not a tuned stress

With `F_ref = k_unrelaxed * eps_b * L0` (the force at which the brittle strain
criterion would delete the bond) the damage of a linearly ramped bond reaches 1
when

    D(t_f) = (1/t_ref) * t_f / (m + 1) = 1   =>   t_f = (m + 1) t_ref,

so creep damage takes over from brittle cleavage below the crossover strain rate

    edot* = eps_b / ((m + 1) t_ref).

For a fixed rate the emergent strength is `sigma_f = E edot t_f ∝
edot^(1/(m+1))`. **Both the exponent and the crossover are predictions of the
mechanism, computed from `m`, `t_ref`, `eps_b` — nothing is fitted.**

## 3. Specimen campaign

`scripts/verify_ice_bond_specimen_rate.py` runs the bonded specimen
(`nx, ny, nz = 4, 4, 3`, `r = 0.025 m`, `k_unrelaxed = 2000 N/m`,
`eps_b = 0.015`, `dt = 3.37e-5 s`) through the opposed-platen uniaxial protocol
at three axial strain rates and separates the mechanisms into four arms:

* **A `linear`** — legacy elastic bond, brittle criterion only: the negative
  control. Strength must be rate independent.
* **B `linear + creep`** (`m = 2`, `t_ref = 5 ms`) — elastic bond + damage
  clock: the closed-form anchor `1/(m+1) = 1/3` and crossover
  `edot* = 1.0 s^-1`.
* **C `maxwell_wiechert`** (`tau = 5 ms, 50 ms`, `k0/k_unrelaxed = 0.35`) —
  relaxation only: rate *stiffening* without an envelope exponent.
* **D `maxwell_wiechert + creep`** — the physical model (both mechanisms).

Rates: 4, 1 and 0.25 s^-1 (the crossover sits at 1 s^-1, so the ladder brackets
it by a factor of four both ways). Each run is capped at twice the cleavage
strain; the peak of the axial stress is the reported strength, the peak strain
and the broken-bond count are recorded alongside.

Numbers are written to `artifacts/ice_bond_specimen/verification.json`; the
report's tables and figures are generated from that file only (no hand
transcription).

## 4. Known limitations (stated, not implied)

1. **No stiffness degradation.** Kachanov damage acts purely as a failure
   clock; the bond force stays elastic right up to deletion. A real damaged
   bond would soften before it fails, so the post-peak branch of this model is
   too stiff.
2. **No Weibull heterogeneity in the specimen campaign.** Every bond carries the
   same `F_b` and the same `F_ref`, so the specimen fails on a single critical
   bond rather than through distributed damage. `weibull_bond_strengths()` is
   available but is not wired into the engine (deliberately: it would change
   the default bond strength).
3. **Specimen size.** 48 particles / 48-node cross-section is the smallest
   specimen that still has an interior bond; the brittle strength of a 4x4
   cross-section is a rigid-plate upper bound, not a converged continuum
   strength. The campaign measures the *rate dependence*, not an absolute
   material strength.
4. **Inertia.** The fastest rate (4 s^-1) is 3 % of the lattice wave speed, so
   inertia is present but small; the final kinetic energy is recorded with each
   run so the reader can check it rather than trust the claim.
5. **The audited ledgers are not closed to 1e-9 on this problem** even on the
   legacy linear bond (measured O(1e-3) relative residual, dominated by the
   first-order work/dissipation estimate of a stiff chain). The Maxwell term is
   checked *against its linear twin* instead of against an absolute bound.
6. **Laboratory calibration is WITHHELD.** The spectrum (`0.35 / 0.45 / 0.20`,
   `tau = 5 ms, 50 ms`) is a modelling choice for demonstration, not a fit to
   ice creep data.

## 5. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_ice_bond_engine.py -q
PYTHONPATH=src python -m pytest tests/test_ice_bond_viscoelastic.py -q
PYTHONPATH=src python scripts/verify_ice_bond_specimen_rate.py            # ~5 min, 1 thread
PYTHONPATH=src python scripts/verify_ice_bond_specimen_rate.py --quick    # 1 rate smoke test
```