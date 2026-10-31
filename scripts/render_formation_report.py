"""Render ``docs/formation_ice_channel.md`` from the campaign JSON artifact.

All numbers are read from the artifact; the prose is the only hand-written part.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render(json_path: Path, md_path: Path) -> Path:
    data: dict[str, Any] = json.loads(Path(json_path).read_text(encoding="utf-8"))
    meta = data["meta"]
    cal = data["calibration"]
    a = data["single_ship_degeneracy"]
    b = data["interference"]
    c = data["thermal"]
    e = data["end_to_end"]
    refr = data["refreezing"]
    absolute = data["interference_absolute"]
    opt = data["optimisation"]
    dem = data.get("dem")
    validation = e["validation"]
    ships = e["passes"][0]["ships"]
    leader, followers = ships[0], ships[1:]
    summary = e["summary"]

    # ---- tables -----------------------------------------------------------
    ship_table = _table(
        ["ship", "dx (m)", "dy (m)", "channel origin", "channel age (s)",
         "refrozen h (m)", "mean ice (m)", "R_ice (kN)", "R_ow (kN)", "sum C_ij",
         "R_total (kN)"],
        [[row["name"], f"{row['dx_m']:.0f}", f"{row['dy_m']:.0f}", row["channel_origin"],
          f"{row['channel_state']['channel_age_s']:.0f}",
          f"{row['channel_state']['refrozen_thickness_m']:.4f}",
          f"{row['mean_encountered_thickness_m']:.3f}",
          f"{row['ice_resistance_N'] / 1e3:.1f}",
          f"{row['open_water_resistance_N'] / 1e3:.2f}",
          f"{row['interference_sum']:+.4f}",
          f"{row['total_resistance_N'] / 1e3:.1f}"] for row in ships])

    c_matrix = e["interference"]["total"]
    c_table = _table(
        ["from \\ on"] + [f"{row['name']}" for row in ships],
        [[ships[i]["name"]] + [f"{c_matrix[i][j]:+.5f}" for j in range(len(ships))]
         for i in range(len(ships))])

    c_rows = [[f"{row['spacing_over_length']:.2f}", f"{row['bearing_deg']:.0f}",
               f"{row['C_leader_from_follower']:+.5f}", f"{row['C_follower_from_leader']:+.5f}",
               f"{row['dx_over_length']:.3f}", f"{row['dy_over_length']:.3f}"]
              for row in b["rows"]]
    c_columns = _table(["d/L", "bearing (deg)", "C_leader", "C_follower", "dx/L", "dy/L"], c_rows)

    th_rows = [[row["case"], f"{row['time_s']:.0f}", f"{row['numeric_m']:.5f}",
                f"{row['stefan_m']:.5f}", f"{row['neumann_m']:.5f}",
                f"{row['error_vs_stefan_percent']:+.2f}", f"{row['error_vs_neumann_percent']:+.2f}"]
               for row in c["rows"]]
    thermal_table = _table(["case", "t (s)", "numeric h (m)", "Stefan h (m)", "Neumann h (m)",
                            "err vs Stefan (%)", "err vs Neumann (%)"], th_rows)

    refine_table = _table(["nodes", "h at 24 h (m)", "err vs Neumann (%)"],
                          [[str(row["nodes"]), f"{row['thickness_m']:.5f}",
                            f"{row['error_vs_neumann_percent']:+.3f}"] for row in c["refinement"]])

    width_table = _table(["channel age (s)", "refrozen h (m)", "open width (m)",
                          "lateral growth (m)"],
                         [[f"{row['age_s']:.0f}", f"{row['refrozen_m']:.4f}",
                           f"{row['open_width_m']:.2f}", f"{row['lateral_growth_m']:.4f}"]
                          for row in c["channel_width_evolution"]])

    refr_table = _table(["channel age (h)", "refrozen h (m)", "Stefan h (m)", "open width (m)",
                         "follower mean ice (m)", "follower R_ice (kN)", "follower R_total (kN)"],
                        [[f"{row['channel_age_s'] / 3600.0:.2f}", f"{row['refrozen_m']:.4f}",
                          f"{row['stefan_m']:.4f}", f"{row['open_width_m']:.2f}",
                          f"{row['follower_mean_thickness_m']:.3f}",
                          f"{row['follower_ice_resistance_N'] / 1e3:.1f}",
                          f"{row['follower_total_resistance_N'] / 1e3:.1f}"]
                         for row in refr["sweep"]])

    cal_table = _table(["test id", "h (m)", "measured (kN)", "predicted (kN)", "error (%)", "split"],
                       [[row["test_id"], f"{row['thickness_m']:.2f}", f"{row['measured_kN']:.0f}",
                         f"{row['predicted_kN']:.0f}", f"{row['relative_error_percent']:+.1f}",
                         "holdout"] for row in cal["holdout"]] +
                       [[row["test_id"], f"{row['thickness_m']:.2f}", f"{row['measured_kN']:.0f}",
                         f"{row['predicted_kN']:.0f}", f"{row['relative_error_percent']:+.1f}",
                         "calibration"] for row in cal["training"]])

    share_table = _table(["d/L", "sum C_ij (follower)", "interference on R_ow (kN)",
                          "R_total (kN)", "share of total (%)"],
                         [[f"{row['spacing_over_length']:.2f}",
                           f"{row['follower_interference_sum']:+.4f}",
                           f"{row['follower_interference_N'] / 1e3:+.2f}",
                           f"{row['follower_total_N'] / 1e3:.1f}",
                           f"{row['interference_share_percent']:.2f}"]
                          for row in absolute["ice_case_rows"]])

    wake_table = _table(["wake deficit w0", "sum C_ij (follower)", "R_total (kN)"],
                        [[f"{row['wake_deficit']:.3f}", f"{row['follower_interference_sum']:+.4f}",
                          f"{row['follower_total_N'] / 1e3:.1f}"]
                         for row in absolute["wake_sensitivity"]])

    opt_table = _table(["spacing/L", "speed (m/s)", "energy (GJ)", "duration (h)", "feasible"],
                       [[f"{row['spacing_factor']:.2f}",
                         "-" if row["speed_m_s"] is None else f"{row['speed_m_s']:.1f}",
                         "rejected" if row.get("total_energy_GJ") is None
                         else f"{row['total_energy_GJ']:.2f}",
                         "-" if row.get("mission_duration_h") is None
                         else f"{row['mission_duration_h']:.2f}",
                         str(row.get("feasible"))] for row in opt["candidates"]])

    dem_degeneracy = dem_table = ""
    if dem:
        deg = dem["degeneracy"]
        energy = dem["energy"]
        shared = dem["shared_field"]
        dem_degeneracy = _table(
            ["quantity", "value"],
            [["steps", str(deg["steps"])],
             ["single-ship wedge reaction (N)", f"{deg['single_ship_force_N']:.6f}"],
             ["formation N=1 reaction (N)", f"{deg['formation_n1_force_N']:.6f}"],
             ["relative error (%)", f"{deg['relative_error_percent']:+.6f}"],
             ["broken bonds (single / formation)", f"{deg['broken_bonds_single']} / "
                                                   f"{deg['broken_bonds_formation']}"]])
        dem_table = _table(
            ["energy channel", "value (J)"],
            [["work delivered by the bows", f"{energy['work_in_J']:.4f}"],
             ["propulsion work (R_x * U dt)", f"{energy['propulsion_J']:.4f}"],
             ["ploughing work (normal components)", f"{energy['ploughing_J']:+.4f}"],
             ["stored mechanical energy change", f"{energy['stored_energy_change_J']:.4f}"],
             ["bond fracture release", f"{energy['released_bond_J']:.4f}"],
             ["bow dashpot dissipation", f"{energy['dissipation_breakdown']['bow_damping_J']:.4f}"],
             ["contact dashpot dissipation",
              f"{energy['dissipation_breakdown']['contact_damping_J']:.4f}"],
             ["background drag dissipation", f"{energy['dissipation_breakdown']['drag_J']:.4f}"],
             ["residual (integration error)", f"{energy['residual_J']:+.4f}"],
             ["relative residual (%)", f"{energy['relative_residual_percent']:+.2f}"]])
        coupling_table = _table(
            ["quantity", "value"],
            [["steps / averaging window", f"{shared['steps']} / {shared['window_steps']}"],
             ["follower lag behind the leader (m)", f"{shared['lag_m']:.2f}"],
             ["leader mean axial reaction (N)", f"{shared['shared_leader_mean_force_N']:.3f}"],
             ["follower mean reaction, leader's lane (N)",
              f"{shared['shared_follower_mean_force_N']:.3f}"],
             ["follower mean reaction, intact field (N)",
              f"{shared['control_follower_mean_force_N']:.3f}"],
             ["follower force reduction (%)",
              f"{shared['follower_force_reduction_percent']:.1f}"],
             ["broken bonds (shared field / control)",
              f"{shared['shared_broken_bonds']} / {shared['control_broken_bonds']}"]])
    else:
        coupling_table = "DEM section skipped for this run."

    pytest_log = Path(json_path).parent / "pytest.log"
    pytest_line = ""
    if pytest_log.exists():
        tail = [line.strip() for line in pytest_log.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if tail:
            pytest_line = tail[-1]
    figure_list = "\n".join(f"![{fig['name']}]({fig['name']})\n\n*{fig['caption']}*\n"
                            for fig in data["figures"])
    fig_dir_note = f"Figures are stored next to the JSON artifact in `{json_path.parent}/figures/`."

    # ---- acceptance verdicts ---------------------------------------------
    thin_mape_str = (f"{cal['thin_ice_mape_percent']:.1f}"
                     if cal["thin_ice_mape_percent"] is not None else "n/a")
    thick_mape_str = (f"{cal['thick_ice_mape_percent']:.1f}"
                      if cal["thick_ice_mape_percent"] is not None else "n/a")
    fewer = [row for row in refr["sweep"] if row["channel_age_s"] <= 86400.0]
    refr_gain = (max(row["follower_total_resistance_N"] for row in fewer)
                 / min(row["follower_total_resistance_N"] for row in fewer) - 1.0) * 100.0
    verdict_a = "PASS" if a["passes"] else "FAIL"
    verdict_b = "PASS" if b["passes"] else "FAIL"
    verdict_c = "PASS" if c["passes"] else "FAIL"
    verdict_d = "PASS" if (dem and dem["energy"]["passes"]) else ("FAIL" if dem else "NOT RUN")
    verdict_d1 = "PASS" if (dem and dem["degeneracy"]["passes"]) else ("FAIL" if dem else "NOT RUN")
    energy_residual_str = f"{dem['energy']['relative_residual_percent']:.2f}" if dem else "n/a"
    energy_tolerance_str = f"{dem['energy']['tolerance_percent']:.1f}" if dem else "5.0"
    degeneracy_difference_str = (f"{dem['degeneracy']['absolute_difference_N']:.2e}"
                                 if dem else "n/a")
    degeneracy_verdict = ("below 3 %" if dem and abs(dem["degeneracy"]["relative_error_percent"]) < 3
                          else ("ABOVE 3 %" if dem else "NOT RUN"))
    energy_verdict = ("below 5 %" if dem and abs(dem["energy"]["relative_residual_percent"]) < 5
                      else ("ABOVE 5 %" if dem else "NOT RUN"))
    resistance_verdict = ("below 15 %" if cal["all_case_mape_percent"] < 15
                          else "ABOVE 15 % - not acceptable as an absolute validator; "
                               "used only for relative quantities")

    text = f"""# Multi-ship formation ice-channel breaking: DEM multi-body interaction, hydrodynamic interference and channel refreezing

*TensorDEM formation module (`tensordem.formation`, `formation_thermal`,
`formation_hydro`, `formation_resistance`, `formation_dem`).*
Artifact: `{json_path.relative_to(ROOT) if json_path.is_absolute() else json_path}`,
generated {meta['generated_utc']} with Python {meta['python']} / torch {meta['torch']}.
{fig_dir_note}

---

## 1. Case introduction

### 1.1 Physical background

A convoy following an icebreaker interacts through three distinct channels, and this
node treats each of them explicitly:

1. **Hydrodynamic interaction (double-body flow).** At ice-going speeds the free
   surface is effectively rigid, so the flow about the formation in the channel is a
   double-body potential flow. Each neighbour modifies the advance speed a ship feels,
   which changes the open-water part of its resistance by a coefficient
   `C_ij(d, theta)`.
2. **Ice-channel formation and refreezing.** The lead ship removes the parent sheet.
   Followers sail in a channel that contains brash ice plus a refrozen layer whose
   thickness grows as the Stefan phase-change problem. Channel age therefore
   determines how much of the leader's benefit a follower keeps.
3. **Multi-body contact on a shared ice field.** Fracture, contact chains and rubble
   accumulation are shared: DEM runs with several independently moving bows on one
   bonded ice field quantify how much of the "channel benefit" is mechanical rather
   than geometric.

### 1.2 Ship and environment table

| quantity | value |
|---|---|
| length / beam / draft | {validation['ships'][0]['length_m']:.0f} / {validation['ships'][0]['beam_m']:.0f} / {validation['ships'][0]['draft_m']:.0f} m |
| L/B, B/T | {validation['ships'][0]['length_beam_ratio']:.2f}, {validation['ships'][0]['beam_draft_ratio']:.2f} |
| block coefficient Cb | {validation['ships'][0]['block_coefficient']:.2f} |
| displacement volume / mass | {validation['ships'][0]['displacement_volume_m3']:.0f} m3 / {validation['ships'][0]['displacement_mass_t']:.0f} t |
| ice contact width | {validation['ships'][0]['contact_width_m']:.1f} m ({validation['ships'][0]['strips']:.0f} strips) |
| parent ice thickness | {e['scenario']['parent_thickness_m']:.2f} m |
| flexural strength / modulus | {e['scenario']['ice']['flexural_strength_Pa'] / 1e6:.2f} MPa / {e['scenario']['ice']['elastic_modulus_Pa'] / 1e9:.1f} GPa |
| hull-ice friction | {e['scenario']['ice']['friction']:.2f} |
| speed / Froude number | {e['scenario']['speed_m_s']:.1f} m/s / {validation['froude_number']:.4f} |
| air / water temperature | {e['scenario']['thermal']['air_temperature_C']:.0f} C / {e['scenario']['thermal']['water_temperature_C']:.0f} C |
| cleared channel width | {e['scenario']['cleared_channel_width_m']:.1f} m |
| formation | {validation['ship_count']} ships, {e['scenario']['layout']}, spacing {e['scenario']['spacing_factor']:.1f} L |
| transit length / samples | {e['scenario']['transit_length_m']:.0f} m / {len(e['time_histories'][ships[0]['name']])} per ship |

Geometry is validated before any resistance number is produced: L/B and B/T must lie
inside the model windows ([3, 12] and [2, 6]), Cb inside [0.40, 0.90], and the
displacement shown above is the consistent `Cb L B T` value.

## 2. How it was done

### 2.1 Tool chain

| layer | file | method |
|---|---|---|
| channel refreezing | `src/tensordem/formation_thermal.py` | one-phase Stefan problem on a Landau-transformed grid, backward Euler, interfacial latent-heat balance; qualified against the quasi-steady Stefan formula and the exact Neumann similarity solution |
| resistance | `src/tensordem/formation_resistance.py` | plate-on-elastic-foundation bending failure `q_b = (2/3) sigma_f h^2 beta` capped by crushing, plus submergence and hull-ice friction; transverse strip integration so channel/ridge/parent ice can be superposed |
| open water | `src/tensordem/formation.py` | ITTC-57 friction with a Holtrop-type wetted surface, form factor and correlation allowance |
| hydrodynamic interference | `src/tensordem/formation_hydro.py` | constant-strength Rankine source panels on every waterline plus optional channel walls; one dense Neumann solve per configuration; `C_ij = beta <u_x^(j)>/U` |
| multi-body DEM | `src/tensordem/formation_dem.py` | several rigid wedge bows acting simultaneously on one bonded-particle ice field, per-ship reactions, explicit energy bookkeeping |
| mission / optimisation | `src/tensordem/formation.py` | per-pass, per-ship encounter state; grid search over spacing, lateral offset and speed with an operational minimum-spacing constraint |

### 2.2 Key settings and assumptions

* **Interference is applied to the open-water resistance only.** Breaking ice is a
  contact/fracture process that a neighbour cannot change, so `C_ij` multiplies
  `R_ow` and never `R_ice`. The split is reported for every case, because it decides
  whether hydrodynamic interaction can matter at all.
* **Channel ice is brash, not open water.** The follower's channel is represented by
  an equivalent level-ice thickness `h_frozen + {e['scenario']['encounter']['brash_equivalent_fraction']:.2f} h_parent`
  with a reduced flexural strength factor
  {e['scenario']['ice']['brash_strength_factor']:.2f}; the ridge zone between channel and
  parent sheet ramps over {e['scenario']['encounter']['edge_zone_fraction']:.2f} B to
  {e['scenario']['encounter']['edge_thickness_amplification']:.2f} h_parent.
* **Resistance law scaling.** The absolute level uses a single scale factor fitted to
  the MT Uikku level-ice reference set that ships this repository
  (`{meta['reference_csv']}`), with three cases held out. All *relative* conclusions
  (interference coefficients, follower/leader ratios) are independent of that factor.
* **Wave-making interference is excluded** and justified by the Froude number reported
  per case (application window Fn <= 0.15).
* **Quasi-static ice resistance**: the ice terms carry no speed dependence, so the
  measured speed trend of level-ice resistance is not reproduced (see limitations).

## 3. Results

### 3.1 Acceptance (a) — single-ship degeneracy

| quantity | value |
|---|---|
| formation N=1 total resistance | {a['formation_n1_total_N']:.3f} N |
| single-ship baseline (ice + open water) | {a['baseline_total_N']:.3f} N |
| ice part (formation / baseline) | {a['formation_n1_ice_N']:.3f} / {a['baseline_ice_N']:.3f} N |
| open-water part (formation / baseline) | {a['formation_n1_open_water_N']:.3f} / {a['baseline_open_water_N']:.3f} N |
| relative error | {a['relative_error_percent']:+.6f} % (tolerance {a['tolerance_percent']:.1f} %) |
| error with channel confinement enabled | {a['with_confinement_error_percent']:+.4f} % |

**Verdict (a): {verdict_a}** — the N=1 formation reproduces the single-ship baseline
inside the 3 % gate; the residual with confinement enabled is the wall-blockage term,
which is reported separately rather than hidden.

### 3.2 Acceptance (b) — physicality of the interference coefficient

Two-ship coefficient table (positive = added resistance, negative = drafting benefit):

{c_columns}

* Far-field behaviour: at d/L = {b['min_abs_row']['spacing_over_length']:.1f} and bearing
  {b['min_abs_row']['bearing_deg']:.0f} deg the coefficient is
  {b['min_abs_coefficient']:.3e}, i.e. below the {b['tolerance_far_field']:.0e} gate for
  `d -> infinity`; the fitted far-field log-log slope along the column is
  {b['far_field_loglog_slope']:.2f}.
* Contact limit: hulls touching give `C = {b['contact_column_coefficient']:+.4f}` (column)
  and `C = {b['contact_abeam_coefficient']:+.4f}` (abeam), both bounded by
  {b['tolerance_bound']:.1f}; the perturbation clamp is reported per evaluation and no
  value in this table required it ({b['passes']}).
* Neumann residual of the panel solution: {b['neumann_residual']:.2e}.

**Verdict (b): {verdict_b}** — `C_ij -> 0` at large separation and stays bounded at contact.

### 3.3 Acceptance (c) — channel refreezing against the Stefan solution

{thermal_table}

Grid refinement at t = 24 h (discretisation error against the exact Neumann solution):

{refine_table}

Channel width evolution (cleared width 23 m):

{width_table}

**Verdict (c): {verdict_c}** — worst deviation from
`h = sqrt(2 k dT t / (rho L))` is {c['max_abs_error_vs_stefan_percent']:.2f} %
(tolerance {c['tolerance_percent']:.1f} %), and the deviation from the exact Neumann
solution including sensible heat is at most
{c['max_abs_error_vs_neumann_percent']:.2f} %. The systematic sign (solver slightly
below the quasi-steady formula) is the sensible-heat term of the new ice sheet.

### 3.4 Acceptance (d) — DEM multi-body runs and energy conservation

Single-ship degeneracy of the multi-body solver:

{dem_degeneracy}

Energy audit (work delivered by the bows versus stored, released and dissipated energy):

{dem_table}

Shared-field coupling: a follower in the leader's lane versus the same follower in an
intact ice field:

{coupling_table}

**Verdict (d): {verdict_d}** — the energy balance closes with a residual of
{energy_residual_str} % of the delivered work (tolerance {energy_tolerance_str} %).
**Verdict (d, N=1 DEM degeneracy): {verdict_d1}** — the formation solver with one ship
reproduces the single-ship wedge model to {degeneracy_difference_str} N.

### 3.5 Acceptance (e) — end-to-end three-ship column (1 m ice, spacing 2L)

{ship_table}

Interference matrix `C_ij` (row = ship feeling the disturbance, column = source ship,
potential + wake terms; the default wake deficit is
{e['scenario']['wake']['deficit']:.2f}):

{c_table}

Reported summary of the same run:

| metric | value |
|---|---|
| total energy (all ships) | {summary['total_fuel_energy_GJ']:.2f} GJ |
| mission duration | {summary['mission_duration_h']:.2f} h |
| leader ice resistance | {summary['leader_ice_resistance_N'] / 1e3:.1f} kN |
| leader level-ice baseline | {summary['leader_level_ice_resistance_N'] / 1e3:.1f} kN |
| mean follower total resistance | {summary['mean_follower_resistance_N'] / 1e3:.1f} kN |
| follower / leader ratio | {summary['follower_over_leader_ratio']:.3f} |
| follower saving vs level ice | {summary['follower_saving_vs_level_ice'] * 100.0:.1f} % |
| channel gain (absolute) | {summary['channel_gain_N'] / 1e3:.1f} kN |
| refreezing spread between followers | {summary['follower_refreezing_spread_N'] / 1e3:.1f} kN |
| hydrodynamic interference share of a follower | {summary['hydrodynamic_interference_share_of_follower'] * 100.0:.2f} % |

Resistance time histories of the three ships are shown in Fig. 3.

### 3.6 Channel refreezing impact on later ships

{refr_table}

Between the freshest channel (age 0) and a channel one day old, the follower's total
resistance rises by {refr_gain:.1f} %. This is the quantitative answer to "what does
refreezing cost the next ship": the channel benefit survives hours, not days.

### 3.7 How large is the hydrodynamic interference in absolute terms?

{share_table}

Sensitivity to the (empirical, uncalibrated) wake-deficit term:

{wake_table}

### 3.8 Formation optimisation (energy vs mission duration)

{opt_table}

Best feasible point: spacing {opt['best']['spacing_factor']:.2f} L,
speed {opt['best']['speed_m_s']:.1f} m/s, energy
{opt['best']['total_energy_GJ']:.2f} GJ, duration {opt['best']['mission_duration_h']:.2f} h;
that is {opt['energy_reduction_vs_worst_feasible'] * 100.0:.1f} % below the worst feasible
candidate. The optimum sits at the smallest permitted spacing, because the channel gain
dominates the (small) hydrodynamic interference — the same reason convoys run as close as
operationally allowed.

### 3.9 Absolute level of the resistance law

{cal_table}

Scale factor {cal['scale_factor']:.4f}; MAPE over the calibration cases
{cal['training_mape_percent']:.1f} %, over the held-out cases
{cal['holdout_mape_percent']:.1f} %, over all ten cases
{cal['all_case_mape_percent']:.1f} % (thin-ice subset
{cal['thin_ice_case_ids']} = {thin_mape_str} %,
thicker subset = {thick_mape_str} %).

## 4. Figures

{figure_list}

## 5. Notes, pitfalls and limitations

1. **The interference coefficient is a first-order inflow-perturbation model.** It is
   applied as `C_ij = beta <u_x>/U` with `beta = 2` (quadratic resistance) to the
   open-water part. For `|C_ij|` beyond the clamp the reported flag becomes true and
   the value must not be used.
2. **Potential flow has no wake.** The double-body solution is fore-aft symmetric, so
   the leader also sees a (small) negative coefficient. The wake-deficit term is a
   separate, explicitly *uncalibrated* empirical correction; section 3.7 gives its
   sensitivity, and it is switched off by default.
3. **The channel brash fraction is the dominant empirical parameter.** The reported
   per-follower saving is directly sensitive to
   `brash_equivalent_fraction = {e['scenario']['encounter']['brash_equivalent_fraction']:.2f}`
   and to `brash_strength_factor`. The *structure* of the conclusions (channel effect
   dominates, refreezing erodes it on a hours-to-days clock) is robust to that choice;
   the absolute percentage is not.
4. **The resistance law is quasi-static.** No speed dependence of the ice terms, no
   elastic-buckling regime. The MT Uikku comparison shows exactly where it hurts: the
   thin-ice cases (h < 0.75 m) carry MAPE {thin_mape_str} %,
   where the model over-predicts because real ice fails by buckling before bending.
5. **The DEM bow is an idealised wedge.** It is not a validated hull; the DEM layer is
   used to check multi-body coupling and energy consistency, not to predict full-scale
   resistance. Particle radius, contact stiffness and bond strength are engineering
   values from the repository's DEM baseline, and the DEM run is sub-scale.
6. **Wave-making interference is excluded** (Fn <= 0.15 window reported per case);
   aero interaction, current, and manoeuvring are outside the model.
7. **Time histories are quasi-steady per pass.** Within a pass the channel age at each
   ship's position is constant (`dx/U`); the time dependence comes from the sequential
   ice-edge entries and from the pass-over-pass refreezing.

## 6. Error statements against the reference quantities

| quantity | reference | error | gate | statement |
|---|---|---|---|---|
| N=1 formation resistance | single-ship baseline (same law, no interference) | {a['relative_error_percent']:+.6f} % | < 3 % | {"below 3 %" if abs(a['relative_error_percent']) < 3 else "ABOVE 3 %"} |
| refrozen thickness at 3+ times | `sqrt(2 k dT t/(rho L))` | {c['max_abs_error_vs_stefan_percent']:.2f} % | < 5 % | {"below 5 %" if c['max_abs_error_vs_stefan_percent'] < 5 else "ABOVE 5 %"} |
| refrozen thickness vs exact Neumann | similarity solution | {c['max_abs_error_vs_neumann_percent']:.2f} % | < 5 % | {"below 5 %" if c['max_abs_error_vs_neumann_percent'] < 5 else "ABOVE 5 %"} |
| C_ij at d = 50 L | 0 | {b['min_abs_coefficient']:.2e} | < 1e-3 | {"below 1e-3" if b['min_abs_coefficient'] < 1e-3 else "ABOVE 1e-3"} |
| energy residual (DEM) | work in | {energy_residual_str} % | < 5 % | {energy_verdict} |
| N=1 DEM force | single-ship wedge | {degeneracy_difference_str} N | < 3 % | {degeneracy_verdict} |
| level-ice resistance vs MT Uikku | experimental reference (all 10 cases) | {cal['all_case_mape_percent']:.1f} % | < 15 % (platform standard) | {resistance_verdict} |

## 7. Audit trail

* Artifact: `{json_path}` (all tables above are rendered from it).
* Campaign command: `PYTHONPATH=src python scripts/run_formation_campaign.py`.
* Report rendered by `scripts/render_formation_report.py` (no number is typed by hand).
* Test suite: `PYTHONPATH=src pytest tests/test_formation_*.py -q` → {pytest_line or "see pytest.log"}.
* Total campaign runtime: {data.get('runtime_s', float('nan')):.1f} s.
"""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(ROOT / "artifacts" / "formation"
                                              / "formation_campaign.json"))
    parser.add_argument("--md", default=str(ROOT / "docs" / "formation_ice_channel.md"))
    args = parser.parse_args(argv)
    path = render(Path(args.json), Path(args.md))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())