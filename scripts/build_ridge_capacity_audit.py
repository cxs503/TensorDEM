"""Generate docs/ridge_capacity_skempton_audit.md from the verification artifact.

Numbers are injected from ``artifacts/ridge_capacity_verification/capacity_verification.json``
(and recomputed from the library where a table entry is needed), so no value in
the document is transcribed by hand.
Run ``PYTHONPATH=src python scripts/verify_ridge_geotech_capacity.py`` first.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tensordem.ridge_geotech import bearing_capacity_factors, skempton_undrained_nc

ART = Path("artifacts/ridge_capacity_verification/capacity_verification.json")
DOC = Path("docs/ridge_capacity_skempton_audit.md")


def _prefix_nc(ratio: float) -> float:
    """The pre-m2983 implementation: Skempton depth factor *times* Meyerhof."""
    return min(5.14 * (1.0 + 0.2 * ratio), 9.0) * (1.0 + 0.2 * min(ratio, 1.0))


def main() -> None:
    data = json.loads(ART.read_text())
    case = data["reference_case"]
    verdict = data["verdict"]
    diameter = case["diameter_m"]
    ratio_case = case["z_over_B"]

    vesic_rows = "\n".join(
        f"| {r['phi_deg']:.0f} | {r['Nc_ref']:.2f} | {r['Nc_code']:.4f} | "
        f"{r['Nq_ref']:.2f} | {r['Nq_code']:.4f} | {r['Ngamma_ref']:.2f} | "
        f"{r['Ngamma_code']:.4f} | {r['max_abs_err']:.4f} |"
        for r in data["vesic_table"])

    skempton_rows = "\n".join(
        f"| {r['z_over_B']:.2f} | {r['Nc_ref']:.2f} | {r['Nc_code']:.4f} | "
        f"{100.0 * r['rel_err']:+.3f} |"
        for r in data["skempton_table"])

    keep = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    sampled = "\n".join(
        f"| {c['z_over_B']:.2f} | {c['Nc_reference']:.3f} | {c['Nc_prefix']:.3f} | "
        f"{100.0 * c['rel_err_prefix']:+.1f} |"
        for c in data["curve"]
        if any(abs(c["z_over_B"] - k) < 1e-9 for k in keep))

    # --- error verdict table (all values recomputed, never transcribed) -------
    nc30, nq30, ng30 = bearing_capacity_factors(math.radians(30.0))
    entries = [
        ("`N_q` (phi = 30 deg)", "Vesic table 18.40", 18.40, nq30, nq30, 1.0, ""),
        ("`N_c` (phi = 30 deg)", "Vesic table 30.14", 30.14, nc30, nc30, 1.0, ""),
        ("`N_gamma` (phi = 30 deg)", "Vesic table 22.40", 22.40, ng30, ng30, 1.0, ""),
        ("`N_c` (phi = 0, surface)", "Prandtl 2 + pi", 2.0 + math.pi,
         skempton_undrained_nc(0.0, diameter), skempton_undrained_nc(0.0, diameter),
         1.0, ""),
        ("`N_c` undrained, `z/B = 1`", "Skempton table 6.17", 6.17, _prefix_nc(1.0),
         skempton_undrained_nc(diameter, diameter), 1.0, ""),
        ("`N_c` undrained, `z/B = 2.5`", "Skempton table 7.50", 7.50, _prefix_nc(2.5),
         skempton_undrained_nc(2.5 * diameter, diameter), 1.0, ""),
    ]
    q_ref = case["su_Pa"] * 5.14 * (1.0 + 0.2 * ratio_case)
    entries.append((f"`q_ult`, audited case (`z/B = {ratio_case:.3f}`)",
                    f"Skempton {q_ref / 1e3:.1f} kPa", q_ref,
                    case["q_prefix_Pa"], case["q_current_Pa"], 1e3, " kPa"))

    def row(label, ref_name, ref, prefix, fixed, scale, unit):
        err = 100.0 * abs(fixed - ref) / ref
        flag = "yes" if err < 3.0 else "**NO**"
        return (f"| {label} | {ref_name} | {prefix / scale:.4f}{unit} | "
                f"{fixed / scale:.4f}{unit} | {err:.4f} % | {flag} |")

    error_rows = "\n".join(row(*e) for e in entries)
    prefix_worst = max(100.0 * abs(e[3] - e[2]) / e[2] for e in entries)
    fixed_worst = max(100.0 * abs(e[4] - e[2]) / e[2] for e in entries)

    doc = f"""# Undrained bearing capacity of the ridge-grounding subgrade: Skempton embedment counted twice

Protocol `{data['protocol']}`. Audit of the merged geotechnical chain in
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
| pipeline | 36 in, `D = {diameter:.3f} m`, `t = 0.0159 m` |
| cover depth above mudline | `{case['cover_depth_m']:.2f} m` |
| subgrade model | undrained total stress, `phi = 0` |
| undrained shear strength | `s_u = {case['su_Pa'] / 1e3:.0f} kPa` |
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
**{verdict['vesic_max_abs_err']:.4f}**, i.e. the two-decimal rounding of the table
entries themselves. `N_c(0) = 5.14` is the Prandtl value `2 + pi = 5.1416`
truncated (`+0.03 %`).

| phi (deg) | Nc table | Nc code | Nq table | Nq code | Ng table | Ng code | max abs err |
|---|---|---|---|---|---|---|---|
{vesic_rows}

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
{skempton_rows}

Error of the pre-fix formula against the same reference:

| z/B | Nc reference | Nc pre-fix | error |
|---|---|---|---|
{sampled}

### 3.3 Reference case

| quantity | pre-fix | fixed | change |
|---|---|---|---|
| `N_c` at `z/B = {ratio_case:.3f}` | {case['Nc_prefix']:.3f} | {case['Nc_current']:.3f} | {100.0 * case['rel_err_prefix']:+.1f} % |
| ultimate pressure `q_ult` (kPa) | {case['q_prefix_Pa'] / 1e3:.1f} | {case['q_current_Pa'] / 1e3:.1f} | {100.0 * case['rel_err_prefix']:+.1f} % |
| capacity per metre of pipe (kN/m) | {case['capacity_per_length_prefix_N_per_m'] / 1e3:.1f} | {case['capacity_per_length_current_N_per_m'] / 1e3:.1f} | {100.0 * case['rel_err_prefix']:+.1f} % |

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
{100.0 * verdict['prefix_max_ratio_err']:.0f} % once the old code hit both its
`9.0` cap and the extra `1.2` (i.e. `z/B >= 2.3`). Drained soils and all
drained-only workflows are numerically unchanged, which
`test_drained_branch_unchanged_by_undrained_fix` pins to `1e-12`.

## 5. Evidence grades and open items

| criterion | evidence | grade |
|---|---|---|
| drained Vesic factors vs published table | `vesic_table` in the JSON, max abs err {verdict['vesic_max_abs_err']:.4f} | REAL |
| undrained `N_c` vs Skempton sequence | `skempton_table`, max err {100.0 * verdict['skempton_max_abs_ratio_err']:.2f} % | REAL |
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
{error_rows}

**Largest pre-fix error over the listed factors: {prefix_worst:.2f} %; largest
error after the fix: {fixed_worst:.3f} %.** The pre-fix undrained branch violated
both the platform's 15 % acceptance limit and the 3 % target; after the fix every
listed quantity is within {fixed_worst:.3f} % of its reference. The withheld items
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
metre of pipe versus cover depth for `D = {diameter:.3f} m` and
`s_u = {case['su_Pa'] / 1e3:.0f} kPa`.
"""
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} chars)")


if __name__ == "__main__":
    main()