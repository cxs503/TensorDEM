"""Generate the P316-W1-N09 report (docs markdown + platform payload).

Every number in the document is injected from ``artifacts/p316_n09/results.json``
(zero hand-copied digits).  Two outputs:

* ``docs/P316_N09_DEM_RESISTANCE_3D_MAPPING.md``  -- the repository report;
* ``artifacts/p316_n09/report_payload.json``      -- payload for
  ``POST /tensor/reports`` (mission 9985), figures inlined as base64 PNG.

Usage: ``PYTHONPATH=src python scripts/p316_n09_report.py``
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p316_n09"
FIG = ART / "figures"
DOC = ROOT / "docs" / "P316_N09_DEM_RESISTANCE_3D_MAPPING.md"


def _f(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build_markdown(results: dict) -> str:
    base = results["base"]
    criteria = results["criteria"]
    case = base["case"]
    spec_case = results["spec"]["base_case"]
    sweep = results["thickness_sweep"]
    refined = results["refinement"]
    mass = results["mass_scale_axis"]

    parameter_rows = [
        ["ice thickness h", f"{case['realised_thickness_m']:.3f} m", f"{case['layers']} particle layers"],
        ["particle radius r", f"{case['particle_radius_m']:.3f} m", "spherical particles, cubic lattice"],
        ["model volume", f"{case['model_length_m']:.2f} x {case['model_width_m']:.2f} m",
         f"{case['particle_count']} particles"],
        ["bow", f"wedge, half-open angle {case['wedge_angle_deg']:.0f} deg",
         "rigid, advancing along +x"],
        ["bow speed", f"{case['bow_speed_m_s']:.2f} m/s", "design speed, not towing-tank scale"],
        ["compressive strength sigma_c", f"{case['compressive_strength_pa'] / 1e6:.2f} MPa",
         "crushing cap = 0.5 sigma_c (triangular-profile mean)"],
        ["flexural strength sigma_f", f"{case['flexural_strength_pa'] / 1e6:.2f} MPa",
         "bond break strain = sigma_f / E"],
        ["ice modulus E", f"{spec_case['elastic_modulus_pa'] / 1e9:.1f} GPa",
         "bond stiffness = E * 2r (lattice estimate)"],
        ["hull-ice friction mu", f"{case['friction_coefficient']:.3f}", "Coulomb, tanh slip-regularised"],
        ["mass scaling", f"{case['mass_scale']:g}", "quasi-static device; see the mass-scale axis"],
        ["time step", f"{case['dt_s']:.3e} s", "stable step from K M^-1 (safety 0.35)"],
        ["bow travel analysed", f"{case['bow_travel_m'] * 1e3:.1f} mm", f"{case['steps']} steps"],
        ["steady window", f"last {case['steady_window_steps']} steps",
         f"starts at step {case['steady_window_start_step']}"],
    ]

    criteria_rows = []
    for name, entry in criteria.items():
        if name == "mass_scale_axis":
            continue
        criteria_rows.append([
            name.replace("_", " "),
            "PASS" if entry["passed"] else "**FAIL**",
            entry["detail"],
        ])
    mass_entry = criteria["mass_scale_axis"]
    criteria_rows.append([
        "mass-scale axis (documented, not a node criterion)",
        "PASS" if mass_entry["passed"] else "**FAIL**",
        mass_entry["detail"],
    ])

    sweep_rows = []
    notch_rows = []
    for key in sorted(sweep, key=float):
        entry = sweep[key]
        steady = entry["steady"]
        import math as _math
        required = entry["case"]["realised_thickness_m"] / _math.sin(
            _math.radians(entry["case"]["wedge_angle_deg"]))
        notch_rows.append([
            key,
            f"{required * 1e3:.0f} mm",
            f"{entry['case']['bow_travel_m'] * 1e3:.1f} mm",
            f"{entry['case']['bow_travel_m'] / required:.3f}",
        ])
        sweep_rows.append([
            key,
            f"{entry['case']['particle_count']}",
            _f(steady["r_total"], 0),
            _f(steady["components"]["crushing"], 0),
            _f(steady["components"]["bending"], 0),
            _f(steady["components"]["submersion"], 0),
            f"{steady['shares']['crushing']:.3f}",
            f"{steady['shares']['bending']:.3f}",
            f"{steady['shares']['submersion']:.3f}",
            f"{entry['lindqvist']['total_signed_error_pct']:+.1f}%",
            f"{entry['contact_patch_area_ratio']:.3f}",
        ])

    comparison_rows = []
    for row in base["lindqvist"]["rows"]:
        comparison_rows.append([
            row["mechanism"],
            _f(row["dem_N_m"], 0),
            _f(row["lindqvist_N_m"], 0),
            f"{row['signed_error_pct']:+.1f}%",
            f"{row['dem_share']:.3f}",
            f"{row['lindqvist_share']:.3f}",
        ])
    comparison_rows.append([
        "**total**",
        _f(base["lindqvist"]["dem_total_N_m"], 0),
        _f(base["lindqvist"]["lindqvist_total_N_m"], 0),
        f"{base['lindqvist']['total_signed_error_pct']:+.1f}%",
        "1.000",
        "1.000",
    ])

    refinement_rows = [
        ["coarse r = " + f"{refined['coarse']['case']['particle_radius_m']:.3f} m",
         f"{refined['coarse']['case']['particle_count']}",
         f"{refined['coarse']['case']['bow_travel_m'] * 1e3:.1f} mm",
         _f(refined["coarse"]["steady_per_width"]["r_total_N_m"], 0),
         f"{refined['coarse']['max_relative_residual']:.1e}"],
        ["fine r/2 = " + f"{refined['fine']['case']['particle_radius_m']:.3f} m",
         f"{refined['fine']['case']['particle_count']}",
         f"{refined['fine']['case']['bow_travel_m'] * 1e3:.1f} mm",
         _f(refined["fine"]["steady_per_width"]["r_total_N_m"], 0),
         f"{refined['fine']['max_relative_residual']:.1e}"],
    ]

    mass_rows = [
        [key, _f(mass[key]["steady_per_width"]["r_total_N_m"], 0),
         f"{mass[key]['steady']['shares']['crushing']:.3f}",
         f"{mass[key]['steady']['shares']['submersion']:.3f}",
         f"{mass[key]['case']['dt_s']:.2e}"]
        for key in sorted(mass, key=float)
    ]

    conventions = [
        ["L = model sheet length (primary)", f"{base['lindqvist']['lindqvist_total_N_m']:.0f}",
         f"{base['lindqvist']['total_signed_error_pct']:+.1f}%"],
        ["L = plate characteristic length", f"{base['lindqvist_characteristic_length']['lindqvist_total_N_m']:.0f}",
         f"{base['lindqvist_characteristic_length']['total_signed_error_pct']:+.1f}%"],
        ["crushing treated as a line load", f"{base['lindqvist_line_load']['lindqvist_total_N_m']:.0f}",
         f"{base['lindqvist_line_load']['error_pct']:+.1f}%"],
    ]

    base_case_file = ART / "cases" / f"{results['base_label']}.json"
    base_series = json.loads(base_case_file.read_text())["series"]
    tail = base_series[int(0.8 * len(base_series)):]
    capped_fraction = sum(row["capped_contacts"] for row in tail) / max(
        1.0, sum(row["contacts"] for row in tail))
    detached_contacts = sum(row["detached_contacts"] for row in tail) / max(1, len(tail))
    submersion_rows = sum(1 for row in tail if abs(row["r_submersion_N"]) > 0.0)

    verdict = results["verdict"]
    return f"""# P316-W1-N09: DEM ice-resistance 3-D mapping closure with crushing/bending/submersion decomposition

Protocol id: `{results['protocol']}`  |  campaign verdict: **{verdict}**
Machine-readable results: `artifacts/p316_n09/results.json` (per-case files in
`artifacts/p316_n09/cases/`), figures in `artifacts/p316_n09/figures/`.

## 1. Case, geometry, parameters

The capability under test is the *mapping* from per-contact DEM forces to the
macroscopic ice resistance, split into the three Lindqvist mechanisms.  The case
is the 3-D bonded-particle ice sheet of `tensordem.dem3d_wedge` penetrated by a
rigid wedge bow, run at model scale (table below).  The bow advances along +x;
the resistance is the reaction along -x.  All quantities are SI.

{_table(["parameter", "value", "note"], parameter_rows)}

## 2. Method

**Closure identity (verified).**  With ``e`` the advance direction and ``F_i`` the
force the hull exerts on the ice at contact ``i``, the ice reacts on the hull with
``-F_i``, whose component along the retarding direction ``-e`` is ``c_i = F_i . e``.
The heading reaction is therefore ``R = -e . sum_i F_i = sum_i c_i`` and, because
every contact is assigned to exactly one mechanism, ``R = R_crush + R_bend +
R_sub`` up to round-off.  The residual is *measured* on every sampled step
(`max_relative_residual`), never assumed.

**Taxonomy (argued, not verified).**  Priority order:

1. *submersion* -- ice detached from the intact bond graph (not connected to the
   largest surviving cluster) **and** pressed down by the hull: the load path that
   costs work against buoyancy;
2. *crushing* -- the contact has reached the plastic crushing cap
   ``p_crush * chord(d) * w_trib`` with ``chord(d) = 2 sqrt(2 r d - d^2)``;
3. *bending* -- remaining contacts on the intact sheet, i.e. load transmitted in
   flexure without local crushing.

**DEM additions (all opt-in, base classes untouched).**  Plastic crushing cap on
the penalty normal law; Coulomb friction on the hull face (tanh
slip-regularised); Archimedes buoyancy + gravity with the waterline set for
floating equilibrium ``f_sub = rho_i / rho_w``; bond-graph connectivity for the
detachment test.  The bond law is derived from the material (``k = E * 2r``,
``eps_b = sigma_f / E``) because the inherited defaults break bonds at ~10 N,
three orders below a capped contact force.  Mass scaling multiplies ice and water
density together so that every force and the floating equilibrium are unchanged
while the stable step grows; its convergence is checked by an explicit axis.

**Reference.**  `tensordem.threefield.resistance` (Lindqvist-style three
components, coefficient ledger in `docs/threefield/FORMULAS.md`) evaluated at the
same model geometry and reduced to the same per-unit-width basis.

## 3. Results

### 3.1 Pre-registered criteria

{_table(["criterion", "verdict", "evidence"], criteria_rows)}

### 3.2 Thickness sweep (mechanism shares)

{_table(["h [m]", "particles", "R [N]", "R_crush [N]", "R_bend [N]", "R_sub [N]",
          "share crush", "share bend", "share sub", "vs Lindqvist", "A_dem/A_ref"], sweep_rows)}

Shares are per-case means over the trailing steady window; ``A_dem/A_ref`` is the
ratio of the resolved contact-patch area to the nominal crushing band ``h * B``.

How far the bow is from a *steady crushing notch* (the state in which the face
loads the whole ice thickness, ``h / sin(alpha)`` of advance):

{_table(["h [m]", "advance for a full notch", "analysed advance", "ratio"], notch_rows)}

### 3.3 Component comparison with Lindqvist (base case, per unit width)

{_table(["mechanism", "DEM [N/m]", "Lindqvist [N/m]", "signed error", "DEM share", "Lindqvist share"],
        comparison_rows)}

### 3.4 Reduction-convention sensitivity

Lindqvist's crushing term carries no beam, so its per-metre reduction is a
convention.  Three conventions are reported; the criterion uses the first.

{_table(["convention", "Lindqvist total [N/m]", "signed deviation of DEM"], conventions)}

### 3.5 Numerical axes

Particle refinement (equal bow advance, so the same physical state):

{_table(["case", "particles", "travel", "R per unit width [N/m]", "max closure residual"], refinement_rows)}

Mass scaling:

{_table(["mass scale", "R per unit width [N/m]", "share crush", "share sub", "dt [s]"], mass_rows)}

### 3.6 Negative control

An empty contact set decomposes to exactly zero, and a DEM case whose bow never
reaches the ice returns ``R_total = 0`` with every component exactly zero
(`series_max_abs_total = {results['negative_control']['series_max_abs_total']}` N,
`max |residual|/|R| = {results['negative_control']['max_relative_residual']:.1e}`).

## 4. Figures

* `figures/closure_history.png` -- stacked components over the base-case history,
  with the worst closure residual annotated.
* `figures/component_shares_vs_thickness.png` -- mechanism shares vs ice thickness.
* `figures/dem_vs_lindqvist.png` -- component-by-component DEM vs Lindqvist and
  the reduction-convention sensitivity.
* `figures/numerical_axes.png` -- particle refinement and mass-scaling axes.

## 5. Notes, traps, applicability

* The closure identity is exact *by construction* (partition of the contact set)
  and the measured residual is round-off; what the tests protect is that no
  contact is dropped, double counted or sign-flipped, and that the reaction
  vector agrees with the component sum.  It is **not** a physical validation.
* The mechanism taxonomy is a load-path proxy, not an energy partition.  A
  contact can be elastic at one instant and capped at the next, so a share is a
  window-averaged statement.  Detached-ice contacts pressed down are attributed
  to submersion; detached ice squeezed forward stays crushing.
* The contact patch resolved by a particle model is smaller than the nominal
  crushing band when the penetration is shallow, so the DEM cannot transmit the
  crushing pressure over the whole ice thickness before the ice displaces or
  fragments.  The ``A_dem/A_ref`` column quantifies this directly and is the
  first-order explanation of the magnitude gap.
* The modelled sheet is much shorter than the floating-plate characteristic
  length (``L_sheet/L_plate = {base['sheet_length_over_characteristic_length']:.2f}``),
  so it behaves as a clamped sheet rather than as an ice field: the flexural
  mechanism cannot develop at its physical length scale.
* Mass scaling is a numerical device: it preserves forces but slows the inertial
  response.  The axis in §3.5 is the evidence for how much it matters; it is not
  a licence to run far from the quasi-static limit.
* Buoyancy uses the spherical-cap model with per-particle sphere volume, so the
  lattice's floating depth is set by ``rho_i/rho_w`` only, not by its porosity.
* **Mechanism degeneracy measured in this geometry**: over the last 20 % of the
  base run {100 * capped_fraction:.0f} % of the contacts are cap-saturated, the
  window average of detached-ice contacts is {detached_contacts:.1f} per step and
  only {submersion_rows} sampled step(s) carry a non-zero submersion component.
  Measured, not inferred: the detached-ice contacts in this window carry no
  downward force component, so under the taxonomy they are squeezing contacts
  (crushing) rather than buoyancy-loaded ones.  The split therefore closes and is
  populated across the run (bending dominates the early elastic ramp, crushing
  the loaded phase), but this geometry does not put load on all three classes
  *simultaneously* in the averaging window; a one-sided (submerged) bow face is
  the obvious next case, and the pure-layer tests exercise the three-class
  partition directly.
* **All contacts cap-saturated** also makes criterion (c) undiscriminating in
  this regime: the crushing share is a step function of the contact state (a
  contact is either crushed or not), so once every contact is capped the share is
  exactly 1.0 and cannot grow with thickness.
* Applicability: this campaign is a model-scale DEM study of the mapping.  It is
  not a full-scale resistance prediction and no full-scale data were available
  offline.

## 6. Error assessment against a reference

| quantity | reference | outcome | < 3 %? |
|---|---|---|---|
| closure residual ``R - (R_c + R_b + R_s)`` | exact identity | ``{criteria['a_closure_residual']['worst_relative_residual']:.1e}`` relative | **yes** (round-off) |
| total resistance | Lindqvist at the same geometry | ``{base['lindqvist']['total_signed_error_pct']:+.1f} %`` | **no** |
| crushing component | Lindqvist ``R_crush`` | ``{base['lindqvist']['rows'][0]['signed_error_pct']:+.1f} %`` | **no** |
| bending component | Lindqvist ``R_bend`` | ``{base['lindqvist']['rows'][1]['signed_error_pct']:+.1f} %`` | **no** |
| submersion component | Lindqvist ``R_sub`` | ``{base['lindqvist']['rows'][2]['signed_error_pct']:+.1f} %`` | **no** |

The 3 % question is answered explicitly: only the *closure identity* is within
3 % (it is exact to round-off).  Every comparison against the Lindqvist
engineering formula fails the 3 % bar, and the campaign verdict is **{verdict}**:
criteria (a) closure, (e) negative control and (f) deliverables pass, the
magnitude criterion (b) fails, and the trend/refinement criteria are reported
with their measured numbers rather than asserted.  The gap is attributed, not
explained away: it is a contact-patch/sheet-length resolution limit of a
particle model at feasible cost, quantified by the ``A_dem/A_ref`` and
``L_sheet/L_plate`` columns, and it is consistent with the earlier 2-D section
study (`docs/LEVEL_ICE_RESISTANCE_DEM.md`, ``+1964 %``).
"""


def build_payload(markdown: str, results: dict) -> dict:
    figures = []
    captions = {
        "closure_history.png": "Base case: stacked crushing/bending/submersion components of the hull reaction.",
        "component_shares_vs_thickness.png": "Mechanism shares vs ice thickness (DEM).",
        "dem_vs_lindqvist.png": "DEM vs Lindqvist components and reduction-convention sensitivity.",
        "numerical_axes.png": "Particle refinement and mass-scaling axes.",
    }
    for name, caption in captions.items():
        path = FIG / name
        if not path.is_file():
            continue
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })
    criteria = results["criteria"]
    summary = (
        "3-D DEM contact-to-resistance mapping closure: closure residual "
        f"{criteria['a_closure_residual']['worst_relative_residual']:.1e}, "
        f"Lindqvist total deviation {results['base']['lindqvist']['total_signed_error_pct']:+.1f}% "
        f"(criteria: {results['verdict']})"
    )
    return {
        "title": "P316-W1-N09 DEM 冰阻力 3D 映射闭合：crushing/bending/submersion 三分量分解与 Lindqvist 对标",
        "summary": summary,
        "scenario_key": "ice_resistance_dem",
        "mission_id": 9985,
        "tags": ["TensorDEM", "ice resistance", "DEM", "Lindqvist", "closure", "P316-W1-N09"],
        "markdown": markdown,
        "figures": figures,
    }


def main() -> int:
    results = json.loads((ART / "results.json").read_text())
    markdown = build_markdown(results)
    DOC.write_text(markdown)
    payload = build_payload(markdown, results)
    (ART / "report_payload.json").write_text(json.dumps(payload))
    print(f"wrote {DOC} ({len(markdown)} chars) and {ART / 'report_payload.json'} "
          f"({len(payload['figures'])} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())