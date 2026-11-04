# Undrained bearing capacity of the ridge-grounding subgrade: Skempton embedment counted twice

Protocol `tensordem-ridge-capacity-skempton-audit-v1`. Audit of the merged geotechnical chain in
`src/tensordem/ridge_geotech.py` against closed-form and tabulated references.
Scope: code-versus-reference verification, not a physical validation.

## 1. Case, physics and parameters

A first-year ice ridge keel is pushed down onto a buried subsea pipeline. The
seabed is represented by `MohrCoulombSoil`: a subgrade reaction `R(w)` that rises
elastically, saturates at an ultimate vertical capacity `p_ult` and (through
`SofteningVerticalSpring`) softens to a residual plateau. The pipeline is an
Euler-Bernoulli beam on that Winkler foundation and the DEM keel supplies the load
per metre of ridge (`docs/ice_ridge_pipeline_grounding.md`).

Ultimate capacity is the quantity that sets how far the keel can drive the pipe
into the mudline and how much of the ice load reaches the pipe wall, so a biased
capacity propagates directly into the reported ice load, bending moment and
DNV-OS-F101 utilisation. The audited reference case is the one the coupled
campaign uses:

| item | value |
|---|---|
| pipeline | 36 in, `D = 0.914 m`, `t = 0.0159 m` |
| cover depth above mudline | `0.50 m` |
| subgrade model | undrained total stress, `phi = 0` |
| undrained shear strength | `s_u = 30 kPa` |
| saturated / water unit weight | 18.0 / 10.06 kN/m^3 |

## 2. What was done

1. Read the merged capacity path: `bearing_capacity_factors` (Vesic 1973),
   `MohrCoulombSoil.vertical_pressure_capacity` (drained and undrained branches)
   and `vertical_capacity_per_length` (pipe invert as the footing base at
   `embedment + D`).
2. Built an independent reference set: the Prandtl (1920) slip-line solution
   `N_c = 2 + pi` for a weightless strip on `phi = 0` clay; the tabulated Vesic
   (1973) factor triple at `phi = 0..40 deg`; the Skempton (1951) strip embedment
   sequence `N_c = 5.14 (1 + 0.2 z/B)` with its `7.5` strip limit (`9.0` for a
   square/circular footing after the `1 + 0.2 B/L` shape factor).
3. Wrote `scripts/verify_ridge_geotech_capacity.py` (JSON artifact + three-panel
   figure), derived the defect and fixed it in `skempton_undrained_nc` /
   `MohrCoulombSoil.vertical_pressure_capacity`.
4. Added `tests/test_ridge_geotech_capacity_verification.py` (table-driven) and
   updated the single assertion in `tests/test_ridge_grounding.py` that had
   encoded the inflated value.

## 3. Results

### 3.1 Drained branch: Vesic factors reproduce the published table

Max absolute deviation over `phi = 0..40 deg` is
**0.0051**, i.e. the two-decimal rounding of the table
entries themselves. `N_c(0) = 5.14` is the Prandtl value `2 + pi = 5.1416`
truncated (`+0.03 %`).

| phi (deg) | Nc table | Nc code | Nq table | Nq code | Ng table | Ng code | max abs err |
|---|---|---|---|---|---|---|---|
| 0 | 5.14 | 5.1400 | 1.00 | 1.0000 | 0.00 | 0.0000 | 0.0000 |
| 10 | 8.35 | 8.3449 | 2.47 | 2.4714 | 1.22 | 1.2242 | 0.0051 |
| 20 | 14.83 | 14.8347 | 6.40 | 6.3994 | 5.39 | 5.3863 | 0.0047 |
| 25 | 20.72 | 20.7205 | 10.66 | 10.6621 | 10.88 | 10.8763 | 0.0037 |
| 30 | 30.14 | 30.1396 | 18.40 | 18.4011 | 22.40 | 22.4025 | 0.0025 |
| 35 | 46.12 | 46.1236 | 33.30 | 33.2961 | 48.03 | 48.0288 | 0.0039 |
| 40 | 75.31 | 75.3131 | 64.20 | 64.1952 | 109.41 | 109.4105 | 0.0048 |

### 3.2 Undrained branch: the defect

The merged implementation computed

`N_c = min(5.14 (1 + 0.2 z/B), 9.0) * (1 + 0.2 k)`, `k = min(z/B, 1)`,

so the Skempton `0.2 z/B` embedment term -- which *is* the depth effect, derived by
Skempton from deep-mechanism solutions for increasing `z/B` -- was multiplied by a
second Meyerhof/Hansen depth factor `(1 + 0.2 k)`, and the `9.0` cap belonging to
a square/circular footing was applied to a model that treats the pipe as a strip
footing of width `D` per metre of length (strip limit `7.5`). Both effects push
the capacity up.

Current implementation (`skempton_undrained_nc`) against the Skempton strip
sequence:

| z/B | Nc reference | Nc code | relative error |
|---|---|---|---|
| 0.00 | 5.14 | 5.1400 | +0.000 |
| 0.50 | 5.65 | 5.6540 | +0.071 |
| 1.00 | 6.17 | 6.1680 | -0.032 |
| 2.00 | 7.20 | 7.1960 | -0.056 |
| 2.50 | 7.50 | 7.5000 | +0.000 |
| 4.00 | 7.50 | 7.5000 | +0.000 |

Error of the pre-fix formula against the same reference:

| z/B | Nc reference | Nc pre-fix | error |
|---|---|---|---|
| 0.00 | 5.140 | 5.140 | +0.0 |
| 0.50 | 5.654 | 6.219 | +10.0 |
| 1.00 | 6.168 | 7.402 | +20.0 |
| 1.50 | 6.682 | 8.018 | +20.0 |
| 2.00 | 7.196 | 8.635 | +20.0 |
| 2.50 | 7.500 | 9.252 | +23.4 |
| 3.00 | 7.500 | 9.869 | +31.6 |
| 3.50 | 7.500 | 10.486 | +39.8 |
| 4.00 | 7.500 | 10.800 | +44.0 |

### 3.3 Reference case

| quantity | pre-fix | fixed | change |
|---|---|---|---|
| `N_c` at `z/B = 1.547` | 8.076 | 6.730 | +20.0 % |
| ultimate pressure `q_ult` (kPa) | 242.3 | 201.9 | +20.0 % |
| capacity per metre of pipe (kN/m) | 221.5 | 184.5 | +20.0 % |

The single assertion in the old suite that pinned this branch
(`30.0e3 * min(5.14 * 1.2, 9.0) * 1.2`) is exactly the double-counted value, so
the inflated number was locked in by the tests rather than caught by them.

## 4. Fix

* New module-level helper `skempton_undrained_nc(depth, width, *, length_ratio=0.0)`
  returning `min(5.14 (1 + 0.2 B/L)(1 + 0.2 z/B), 7.5 (1 + 0.2 B/L))`. The strip
  case (`length_ratio = 0`) reproduces the `5.14 -> 7.5` Skempton sequence; the
  square case reproduces `6.17 -> 9.0`.
* `MohrCoulombSoil.vertical_pressure_capacity(..., length_ratio=0.0)` returns
  `s_u N_c` in the undrained branch: one depth factor, no Meyerhof term on top.
* `vertical_capacity_per_length` forwards `length_ratio`. The drained branch and
  the lateral limit pressure are untouched.

Behavioural consequences: undrained capacities drop by 20 % at the audited `z/B`
(typically 1-2 for buried pipelines) and by up to
44 % once the old code hit both its
`9.0` cap and the extra `1.2` (i.e. `z/B >= 2.3`). Drained soils and all
drained-only workflows are numerically unchanged, which
`test_drained_branch_unchanged_by_undrained_fix` pins to `1e-12`.

## 5. Evidence grades and open items

| criterion | evidence | grade |
|---|---|---|
| drained Vesic factors vs published table | `vesic_table` in the JSON, max abs err 0.0051 | REAL |
| undrained `N_c` vs Skempton sequence | `skempton_table`, max err 0.07 % | REAL |
| `q/s_u` linear in `s_u`, analytic `dq/dz` | `test_undrained_pressure_slope_matches_skempton_gradient` (1e-7) | REAL |
| lateral limit pressure `N_p = 2 + 1.5 z/D` | test only pins the implemented formula | WITHHELD: no offline reference for this pipe-soil practice could be checked in the worktree |
| coupled DEM + capacity-envelope artifacts | **stale**: generated with the pre-fix capacity and not regenerated here (capacity discipline: `load15 ~119` of 128 cores, so no new heavy runs in this ticket) | WITHHELD |

Stale artifacts to regenerate in a later ticket, in this order:
`scripts/run_ice_ridge_grounding.py` (main case, soil-strength sweep, capacity
envelope) and `scripts/make_ridge_grounding_figures.py`. The DEM seabed contact in
`src/tensordem/ridge_dem.py` uses the same `vertical_pressure_capacity`, so keel
penetration depth and the sensitivity numbers quoted in
`docs/ice_ridge_pipeline_grounding.md` (`73.8 / 221.5 / 442.9 kN/m`) predate this
fix and must be re-derived, not scaled by a constant factor.

Unrelated red test found while running the suite (pre-existing; untouched by this
ticket): `tests/test_fracture_gen_mass_footprint.py` cannot be collected because
it imports `MASS_RADIUS_FACTOR`, which no longer exists in
`src/tensordem/fracture_gen.py` (lost when a gated commit filtered its source
files). Left for a dedicated ticket.

## 6. Error verdict against the acceptance threshold

Acceptance threshold used here: `< 15 %` relative error of a capacity factor
against its reference (platform standard), with `< 3 %` as the target for a
verified quantity.

| key quantity | reference | value before fix | value after fix | error after fix | < 3 %? |
|---|---|---|---|---|---|
| `N_q` (phi = 30 deg) | Vesic table 18.40 | 18.4011 | 18.4011 | 0.0061 % | yes |
| `N_c` (phi = 30 deg) | Vesic table 30.14 | 30.1396 | 30.1396 | 0.0012 % | yes |
| `N_gamma` (phi = 30 deg) | Vesic table 22.40 | 22.4025 | 22.4025 | 0.0111 % | yes |
| `N_c` (phi = 0, surface) | Prandtl 2 + pi | 5.1400 | 5.1400 | 0.0310 % | yes |
| `N_c` undrained, `z/B = 1` | Skempton table 6.17 | 7.4016 | 6.1680 | 0.0324 % | yes |
| `N_c` undrained, `z/B = 2.5` | Skempton table 7.50 | 9.2520 | 7.5000 | 0.0000 % | yes |
| `q_ult`, audited case (`z/B = 1.547`) | Skempton 201.9 kPa | 242.2931 kPa | 201.9109 kPa | 0.0000 % | yes |

**Largest pre-fix error over the listed factors: 23.36 %; largest
error after the fix: 0.032 %.** The pre-fix undrained branch violated
both the platform's 15 % acceptance limit and the 3 % target; after the fix every
listed quantity is within 0.032 % of its reference. The withheld items
in section 5 are not covered by this verdict.

## 7. Reproduce

```bash
PYTHONPATH=src python scripts/verify_ridge_geotech_capacity.py   # JSON + PNG
PYTHONPATH=src python scripts/build_ridge_capacity_audit.py      # this document
python -m pytest tests/test_ridge_geotech_capacity_verification.py tests/test_ridge_grounding.py -q
```

Figure `artifacts/ridge_capacity_verification/capacity_verification.png`:
(a) `N_c(z/B)` reference, pre-fix curve and table points; (b) pre-fix relative
error with the 15 % acceptance line and the audited case marked; (c) capacity per
metre of pipe versus cover depth for `D = 0.914 m` and
`s_u = 30 kPa`.
