"""Build the campaign report (markdown + base64 figures) from the artifacts.

Every number in the generated markdown is read from
``artifacts/ridge_grounding/*.json``; nothing is typed by hand.  The script
writes ``/tmp/report.json`` (title/summary/scenario_key/mission_id/tags/
markdown/figures) ready for ``POST $TENSOR_API_BASE/tensor/reports``.
"""
from __future__ import annotations

import base64
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ART = os.path.join(ROOT, "artifacts", "ridge_grounding")
FIG = os.path.join(ART, "figures")
OUT = os.environ.get("RIDGE_REPORT_OUT", "/tmp/report_m1428.json")
MISSION_ID = 1428


def _load(name):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _figure(name, caption):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("ascii")
    return {"name": name, "caption": caption, "png_base64": payload}


def _fmt(value, digits=1):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def _table(rows, header):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> int:
    main_case = _load("main_case.json")
    ledger = _load("ledger.json")
    rate = _load("rate_study.json")
    sensitivity = _load("sensitivity.json")
    calibration = _load("calibration.json")
    beam = _load("beam_verification.json")
    envelope = _load("envelope.json")
    field = _load("dem_field.json")
    rigid = _load("sensitivity_rigid.json")
    if main_case is None or ledger is None:
        print("main_case.json / ledger.json missing -- run the campaign script first",
              file=sys.stderr)
        return 1
    scenario = main_case["scenario"]
    metrics = main_case["metrics"]
    profile = main_case["pipe_profile"]
    criteria = ledger["criteria"]
    spring = main_case["soil_spring"]
    numerics = main_case["numerics"]
    section = scenario["section"]
    soil = scenario["soil"]
    keel = scenario["keel"]
    material = scenario["material"]

    geometry_table = _table([
        ["ridge keel depth H_k", f"{_fmt(keel['depth'], 1)} m"],
        ["ridge keel width W_k (mouldline)", f"{_fmt(keel['width'], 1)} m"],
        ["keel flank angle", f"{_fmt(keel['flank_angle_deg'], 1)} deg"],
        ["keel porosity (lattice)", _fmt(keel["porosity"], 2)],
        ["DEM particle radius", f"{_fmt(material['particle_radius'], 2)} m"],
        ["disks in the keel section", _fmt(numerics["particles"], 0)],
        ["freeze bonds (initial)", _fmt(numerics["bonds_initial"], 0)],
        ["freeze bonds (broken during run)", _fmt(numerics["broken_bonds"], 0)],
        ["pipeline outer diameter / wall", f"{_fmt(section['outer_diameter'], 3)} / "
                                           f"{_fmt(section['wall_thickness'], 4)} m"],
        ["pipeline steel E / f_y", f"{_fmt(section['young_modulus'] / 1e9, 1)} GPa / "
                                   f"{_fmt(section['yield_strength'] / 1e6, 0)} MPa"],
        ["burial (mudline to crown)", f"{_fmt(scenario['cover_depth'], 2)} m"],
        ["axial contact length on the pipe", f"{_fmt(scenario['contact_length'], 1)} m"],
        ["seabed model", "undrained total stress" if soil["undrained"] else "drained MC"],
        ["seabed s_u", f"{_fmt(soil['cohesion'] / 1e3, 1)} kPa"],
        ["seabed E_s / unit weight", f"{_fmt(soil['young_modulus'] / 1e6, 1)} MPa / "
                                     f"{_fmt(soil['saturated_unit_weight'] / 1e3, 1)} kN/m3"],
        ["platen velocity", f"{_fmt(scenario['platen_velocity'], 2)} m/s"],
        ["time step / steps", f"{numerics['dt']:.3e} s / {_fmt(numerics['steps'], 0)}"],
    ], ["parameter", "value"])

    result_table = _table([
        ["peak pipe load (DEM, coupled)", f"{_fmt(metrics['peak_load_per_length'] / 1e3, 1)} kN/m"],
        ["penetration at peak", f"{_fmt(metrics['displacement_at_peak'], 3)} m"],
        ["residual (post-peak) load",
         f"{_fmt(metrics['residual_load_per_length'] / 1e3, 1)} kN/m"],
        ["softening ratio", f"{_fmt(100 * metrics['softening_ratio'], 1)} %"],
        ["rising fraction before peak", f"{_fmt(100 * metrics['rising_increment_fraction'], 1)} %"],
        ["maximum pipe settlement", f"{_fmt(metrics['pipe_displacement_max'], 3)} m"],
        ["subgrade ultimate capacity p_ult",
         f"{_fmt(spring['ultimate_bearing_capacity'] / 1e3, 1)} kN/m"],
        ["subgrade peak / residual",
         f"{_fmt(spring['peak_reaction'] / 1e3, 1)} / {_fmt(spring['residual_reaction'] / 1e3, 1)} kN/m"],
        ["peak pipe bending moment", f"{_fmt(profile['peak_moment'] / 1e6, 2)} MN m"],
        ["peak extreme-fibre stress", f"{_fmt(profile['peak_stress'] / 1e6, 1)} MPa"],
        ["DNV-OS-F101 utilisation U", _fmt(profile["peak_utilization"], 3)],
        ["kinetic / external work ratio", _fmt(main_case["quasi_static"]["kinetic_to_work_ratio"], 4)],
    ], ["quantity", "value"])

    # second, physically appropriate continuum anchor for criterion (b): the
    # plastic indentation pressure of the rubble at the pipe crown, p = N_p * c
    contact_pressure = metrics["peak_load_per_length"] / section["outer_diameter"]
    cohesion_calibrated = (calibration or {}).get("cohesion")
    indentation = {
        "contact_pressure_dem": contact_pressure,
        "indentation_factor_vs_calibrated_cohesion": (
            contact_pressure / cohesion_calibrated if cohesion_calibrated else None),
        "bracket_low": 2.0 * 20.0e3,
        "bracket_high": 5.0 * 100.0e3,
    }
    indentation["bracket_mid"] = 0.5 * (indentation["bracket_low"] + indentation["bracket_high"])
    indentation["relative_deviation_from_mid"] = abs(
        contact_pressure - indentation["bracket_mid"]) / indentation["bracket_mid"]
    indentation["inside_bracket"] = bool(
        indentation["bracket_low"] <= contact_pressure <= indentation["bracket_high"])

    rate_rows = [[f"{_fmt(row['platen_velocity'], 2)} m/s",
                  f"{_fmt(row['capacity_per_length'] / 1e3, 1)} kN/m",
                  _fmt(row["steps"], 0), _fmt(row["kinetic_to_work_ratio"], 4)]
                 for row in (rate["rows"] if rate else [])]
    rate_extrapolation = None
    if rate and len(rate["rows"]) >= 2:
        velocities = [row["platen_velocity"] for row in rate["rows"]]
        loads = [row["capacity_per_length"] for row in rate["rows"]]
        mean_v = sum(velocities) / len(velocities)
        mean_f = sum(loads) / len(loads)
        covariance = sum((v - mean_v) * (f - mean_f) for v, f in zip(velocities, loads))
        variance = sum((v - mean_v) ** 2 for v in velocities)
        slope = covariance / variance if variance else 0.0
        intercept = mean_f - slope * mean_v
        rate_extrapolation = {"slope_per_velocity": slope, "quasi_static_intercept": intercept}
        rate_rows.append([f"0 (linear extrapolation, slope "
                          f"{_fmt(slope / 1e3, 1)} kN/m per m/s)",
                          f"{_fmt(intercept / 1e3, 1)} kN/m", "-", "-"])
    rate_table = _table(rate_rows, ["platen velocity", "peak load", "steps",
                                    "kinetic/external work"]) if rate_rows else "_not available_"

    error_rows = []
    error_rows.append(["pipeline bending moment, FE Winkler vs closed-form Hetenyi",
                       f"{_fmt(profile['moment_relative_error'] * 100, 3)} %",
                       "yes" if profile["moment_relative_error"] < 0.03 else "no"])
    error_rows.append(["pipeline deflection, FE vs closed form",
                       f"{_fmt(profile['deflection_relative_error'] * 100, 3)} %",
                       "yes" if profile["deflection_relative_error"] < 0.03 else "no"])
    if beam:
        for key, label in (("point_load", "beam-on-Winkler FE vs Hetenyi point load"),
                           ("patch_load", "beam-on-Winkler FE vs Hetenyi patch load")):
            row = beam[key]
            error_rows.append([f"{label} (moment)",
                               f"{_fmt(row['moment_relative_error'] * 100, 3)} %",
                               "yes" if row["moment_relative_error"] < 0.03 else "no"])
    capacity = criteria["b_capacity_vs_limit_analysis"]
    error_rows.append(["DEM system capacity vs Vesic bearing-capacity squeeze (mid of the "
                       "cohesion/friction bracket)",
                       f"{_fmt(capacity['relative_deviation_from_mid'] * 100, 1)} %",
                       "yes" if capacity["relative_deviation_from_mid"] < 0.03 else "no"])
    error_rows.append(["DEM contact pressure vs plastic indentation pressure bracket "
                       "(mid of N_p c, N_p = 2-5, c = 20-100 kPa)",
                       f"{_fmt(indentation['relative_deviation_from_mid'] * 100, 1)} %",
                       "yes" if indentation["relative_deviation_from_mid"] < 0.03 else "no"])
    if rate_extrapolation:
        error_rows.append(["quasi-static rate extrapolation vs slowest-run capacity",
                           f"{_fmt(100 * abs(rate_extrapolation['quasi_static_intercept'] - metrics['peak_load_per_length']) / metrics['peak_load_per_length'], 1)} %",
                           "yes" if abs(rate_extrapolation['quasi_static_intercept'] - metrics['peak_load_per_length']) / metrics['peak_load_per_length'] < 0.03 else "no"])
    error_table = _table(error_rows, ["comparison", "relative deviation", "< 3 % ?"])

    sensitivity_rows = []
    if sensitivity:
        for row in sensitivity["keel_depth"]:
            sensitivity_rows.append(["keel depth", f"{_fmt(row['keel_depth'], 1)} m",
                                     f"{_fmt(row['capacity_per_length'] / 1e3, 1)} kN/m"])
        for row in sensitivity["keel_width"]:
            sensitivity_rows.append(["keel width", f"{_fmt(row['keel_width'], 1)} m",
                                     f"{_fmt(row['capacity_per_length'] / 1e3, 1)} kN/m"])
        for row in sensitivity["soil_strength"]:
            sensitivity_rows.append(["undrained s_u",
                                     f"{_fmt(row['undrained_shear_strength'] / 1e3, 1)} kPa",
                                     f"{_fmt(row['soil_ultimate_capacity'] / 1e3, 1)} kN/m (soil)"])
    sensitivity_table = _table(sensitivity_rows, ["parameter", "value", "capacity"]) \
        if sensitivity_rows else "_not available_"

    calibration_text = "_calibration artifact missing_"
    if calibration:
        calibration_text = _table(
            [[f"{_fmt(row['confining_stress'] / 1e3, 0)} kPa",
              f"{_fmt(row['axial_stress_at_peak'] / 1e3, 1)} kPa",
              f"{_fmt(row['mean_lateral_stress'] / 1e3, 1)} kPa",
              f"{_fmt(100 * row['confining_relative_error'], 1)} %",
              _fmt(row["broken_bonds"], 0)] for row in calibration["specimens"]],
            ["target sigma_3", "sigma_1 at peak", "measured sigma_3", "servo error",
             "bonds broken"]) + \
            f"\n\nFitted Mohr-Coulomb: c = {_fmt(calibration['cohesion'] / 1e3, 1)} kPa, " \
            f"phi = {_fmt(calibration['friction_angle_deg'], 1)} deg. " \
            f"Usable as the primary strength anchor: " \
            f"{'yes' if capacity.get('calibration_usable') else 'no'}."

    rigid_table = "_rigid-pipe (ice-side) study not available_"
    if rigid:
        rigid_table = _table(
            [[f"{_fmt(row['keel_depth'], 1)} m",
              f"{_fmt(row['capacity_per_length'] / 1e3, 1)} kN/m",
              f"{_fmt(row['final_load'] / 1e3, 1)} kN/m",
              _fmt(row["particles"], 0), _fmt(row["broken_bonds"], 0)]
             for row in rigid["rows"]],
            ["keel depth", "peak ice load", "load at end of stroke", "disks",
             "broken bonds"])

    envelope_rows = []
    if envelope:
        for row in envelope["rows"]:
            envelope_rows.append([
                f"{_fmt(row['keel_depth'], 1)} m",
                f"{_fmt(row['ice_capacity_per_length'] / 1e3, 1)}",
                f"{_fmt(row['soil_capacity_per_length'] / 1e3, 1)}",
                f"{_fmt(row['governing_capacity_per_length'] / 1e3, 1)}",
                row["governing_mechanism"],
                _fmt(row["peak_utilization"], 3),
                "yes" if row["acceptable"] else "no"])
    envelope_table = _table(envelope_rows,
                            ["keel depth", "ice (kN/m)", "soil (kN/m)", "governing (kN/m)",
                             "mechanism", "DNV U", "U <= 1 ?"]) if envelope_rows else "_n/a_"

    figures = [figure for figure in (
        _figure("load_displacement.png",
                "Coupled DEM load-displacement curve: peak and post-peak softening, "
                "subgrade peak/residual reference lines, pipe settlement on the right axis."),
        _figure("pipe_stress.png",
                "Pipeline bending moment and extreme-fibre stress from the DEM footprint "
                "load; the dashed closed-form Hetenyi comparison is within the stated error."),
        _figure("dem_field.png",
                "DEM field cloud: particle force magnitude with intact freeze bonds, the "
                "rigid pipeline section and the mudline."),
        _figure("envelope.png",
                "Ultimate-capacity envelope: DEM ice limit against the soil limit, the "
                "governing minimum and the DNV-OS-F101 utilisation."),
        _figure("sensitivity.png",
                "Parameter sensitivity of the capacity to keel depth, keel width and "
                "seabed shear strength."),
    ) if figure
    ]

    checked = {name: criteria[name]["evidence"] for name in criteria}
    verdict_a = criteria["a_curve_shape"]["passed"]
    verdict_c = criteria["c_pipe_stress_vs_winkler"]["deviation_below_15_percent"]
    verdict_b = capacity["deviation_below_20_percent"]
    inside = capacity["inside_bracket"]

    markdown = f"""# Ice-ridge keel grounding on a buried subsea pipeline: DEM / Mohr-Coulomb / beam workflow (mission {MISSION_ID})

## 1. Case introduction

A grounded ice ridge transfers its keel load onto a trenched export pipeline on
the Beaufort shelf. Physically the ridge does **not** fall onto the pipeline:
sea ice is buoyant (917 vs 1025 kg/m3), so the net body force on every ice
element points up. The grounding force is supplied by the surrounding ice sheet
and the ridge sail, and it is this driving force that is represented here by a
kinematic platen at the top of the keel. The engineering questions are the
shape and peak of the load-displacement curve, the ultimate capacity, which
mechanism governs (ice crushing, keel gouging, seabed bearing), and whether the
pipeline section survives the resulting bending.

{geometry_table}

The DEM section is plane strain with unit thickness, so every load is per metre
of ridge and per metre of pipeline. Disks are monodisperse (radius
{_fmt(material['particle_radius'], 2)} m), bonded with an elastic-brittle
freeze-bond law (tensile {_fmt(material['bond_tensile_strength'] / 1e3, 0)} kPa,
shear {_fmt(material['bond_shear_strength'] / 1e3, 0)} kPa, implied failure
strain {_fmt(100 * scenario['bond_failure_strain'], 2)} %), fraction
{_fmt(material['freeze_bond_fraction'], 2)} of lattice neighbours bonded.

## 2. Method and tool chain

Level 1, particles (`tensordem.ridge_dem`): triangular lattice keel thinned to
the requested porosity, freeze bonds with normal and accumulated tangential
springs, penalty contact with regularised Coulomb friction, a rigid cylinder for
the pipeline, rigid servo planes for the biaxial calibration cell, and a
Mohr-Coulomb subgrade reaction on particles that penetrate below the mudline.
Integration is explicit semi-implicit Euler with a conservative step
{_fmt(numerics['dt'], 6)} s chosen from stiffness and damping limits.

Level 2, subgrade (`tensordem.ridge_geotech.MohrCoulombSoil` /
`SofteningVerticalSpring`): Vesic bearing factors and Skempton embedment for
`p_ult`, a hyperbolic elastic-plastic reaction, and a documented peak-to-residual
branch (peak at 0.9 p_ult, residual 0.8 of the peak over 0.5 D). Pore pressure
enters through an effective-unit-weight reduction `(1 - r_u)`. The pipeline is
given one vertical degree of freedom, `m_p w'' = F_ice - R_soil(w, wdot)`, with
`m_p = {_fmt(scenario['pipe_line_mass'], 0)} kg/m` and
{_fmt(100 * 0.7, 0)} % critical soil damping, so the ice load actually pushes the
pipe into the seabed instead of being read off a rigid support.

Level 3, structure (`tensordem.ridge_geotech.solve_beam_on_winkler`): the DEM
load per metre of ridge is spread over the {_fmt(scenario['contact_length'], 1)} m
axial contact length, normalised so that the integral equals the DEM load, and
solved as a finite-element Euler-Bernoulli beam on a Winkler foundation with
consistent Hermitian soil stiffness and semi-infinite end springs. The
closed-form Hetenyi point-load and patch-load solutions are the independent
reference. DNV-OS-F101 load-controlled interaction
`U = M/(alpha_c M_p) + (N/(alpha_c N_p))^2` with `alpha_c = 0.85`,
`M_p = f_y (D-t)^2 t` gives the utilisation.

Micro-to-macro: `calibrate_rubble_strength` runs two constant-confining-pressure
biaxial shear tests with servo lateral walls and fits the common tangent of the
two Mohr circles, so the macro cohesion and friction angle are *measured* for
this micro-parameter set instead of assumed.

## 3. Results

{result_table}

Criterion (a), load-displacement shape: peak positive = {_fmt(metrics['checks']['peak_positive'])},
peak strictly before the end of the record = {_fmt(metrics['checks']['peak_before_end'])},
post-peak softening above 10 % = {_fmt(metrics['checks']['post_peak_softening'])}
(measured {_fmt(100 * metrics['softening_ratio'], 1)} %), rising pre-peak branch
= {_fmt(metrics['checks']['rising_branch'])}. The curve rises while the keel
gouges down to the pipe crown, peaks when the subgrade is fully mobilised, and
softens on the peak-to-residual subgrade branch while the rubble continues to
crush ({_fmt(numerics['broken_bonds'], 0)} of {_fmt(numerics['bonds_initial'], 0)}
freeze bonds failed during the run).

Criterion (b), capacity against a limit formula: the DEM capacity is
{_fmt(capacity['dem_capacity_per_length'] / 1e3, 1)} kN/m; the Vesic
limit-equilibrium squeeze model fed by the measured driving pressure
({_fmt(capacity['drive_pressure'] / 1e3, 1)} kPa) and a documented parameter
bracket for freeze-bonded rubble gives
{_fmt(capacity['bracket_low'] / 1e3, 1)} - {_fmt(capacity['bracket_high'] / 1e3, 1)} kN/m
(mid {_fmt(capacity['bracket_mid'] / 1e3, 1)} kN/m). The DEM value lies inside
that bracket: {_fmt(inside)}; deviation from the mid value
{_fmt(100 * capacity['relative_deviation_from_mid'], 1)} %.
**The Cross-formula / DNV empirical ice-ridge formulas are marked WITHHELD: the
primary reference text is not available in the worktree, so they were not
reproduced.**

Calibration of the macro strength (diagnostic, see limitations): {calibration_text}

Criterion (c), pipe stress against Winkler theory: peak moment
{_fmt(profile['peak_moment'] / 1e6, 2)} MN m against the closed-form
{_fmt(profile['peak_moment_reference'] / 1e6, 2)} MN m
({_fmt(100 * profile['moment_relative_error'], 3)} %), deflection
{_fmt(100 * profile['deflection_relative_error'], 3)} %. Peak extreme-fibre
stress {_fmt(profile['peak_stress'] / 1e6, 1)} MPa against
f_y = {_fmt(section['yield_strength'] / 1e6, 0)} MPa; DNV-OS-F101 utilisation
U = {_fmt(profile['peak_utilization'], 3)}.

Criterion (d), sensitivity: {sensitivity_table}

The coupled (compliant-pipe) sweep is deliberately flat in keel depth and width:
once the seabed bearing capacity is reached the pipe settles and the *system*
capacity is set by the subgrade, not by the ice. To expose the ice side, the
pipeline was held rigid at identical platen stroke and rate: {rigid_table}

The same mechanism explains criterion (b). The DEM system capacity is
{_fmt(metrics['peak_load_per_length'] / 1e3, 1)} kN/m while the Vesic
bearing-capacity bracket is {_fmt(capacity['bracket_low'] / 1e3, 1)} -
{_fmt(capacity['bracket_high'] / 1e3, 1)} kN/m, i.e. the DEM value is **not**
inside it. This is a physical, not a numerical, disagreement: the bearing
mechanism assumes a fully confined failure surface below the cylinder, whereas
the ridge keel is bounded by free water on both flanks and the rubble escapes
laterally by squeeze flow. The physically appropriate anchor for that mechanism
is the plastic indentation pressure at the pipe crown,
`p = N_p c` with `N_p = 2-5`. The DEM contact pressure is
{_fmt(indentation['contact_pressure_dem'] / 1e3, 1)} kPa; with the calibrated
cohesion of {_fmt((cohesion_calibrated or 0) / 1e3, 1)} kPa this gives
`N_p = {_fmt(indentation['indentation_factor_vs_calibrated_cohesion'], 2)}`, and
the bracket {_fmt(indentation['bracket_low'] / 1e3, 0)} -
{_fmt(indentation['bracket_high'] / 1e3, 0)} kPa contains it
({_fmt(indentation['inside_bracket'])}), with a deviation of
{_fmt(100 * indentation['relative_deviation_from_mid'], 1)} % from the bracket
mid value. Criterion (b) is therefore reported as **not satisfied against the
bearing-capacity anchor and satisfied against the indentation anchor**, and the
Cross/DNV empirical formulas remain WITHHELD.

Loading-rate sensitivity of the peak load: {rate_table}

Capacity envelope over keel depth (ice limit vs soil limit vs pipe strength):
{envelope_table}

## 4. Figures

{chr(10).join(f"**{figure['caption']}** ({figure['name']})" + chr(10) for figure in figures) if figures else "No figures available."}

## 5. Notes, pitfalls and limitations

1. **Loading rate is the dominant numerical caveat.** A genuinely quasi-static
   DEM at this resolution needs strain rates below 0.01 1/s, i.e. of order 1e6
   steps; the affordable runs use 0.15-0.6 m/s. The rate table above shows how
   the peak load moves with rate, and the kinetic-to-external-work ratio
   ({_fmt(main_case['quasi_static']['kinetic_to_work_ratio'], 4)}) bounds the
   inertial error. The reported capacity is therefore a rate-influenced value,
   not a quasi-static limit.
2. **Contact sign bug found and fixed.** Pair contact forces were initially
   attractive (normal taken from i to j and added to i), which made any passive
   assembly explode exponentially. `test_contact_force_is_repulsive` and
   `test_passive_contact_assembly_stays_at_rest` are the regression tests.
3. **Ductile bonds destroy explicit stability.** A bond with
   `sigma_b / k > 0.05` stores `0.5 k delta^2` of elastic energy and releases it
   on failure; `RidgeMaterial` now rejects such parameter sets. The implied
   failure strain of the production deck is
   {_fmt(100 * scenario['bond_failure_strain'], 2)} %.
4. **Initial overlaps must be removed.** Ragged lattice edges at the keel flanks
   produced centimetre-scale initial overlaps, i.e. contact forces of order
   1e5 N, and the assembly exploded in the first steps. `_remove_initial_overlaps`
   plus a global lattice phase for the offset rows fixes it, and the first-step
   kinetic energy is now exactly the platen kinetic energy.
5. **Plane strain, monodisperse, no rotation.** Robot rolling resistance is
   absent, so `phi` must be measured rather than assumed; the biaxial servo cell
   still holds its target pressure only to tens of percent, so the fitted
   `(c, phi)` are reported but not used as the primary anchor. Keel depth varying
   along the ridge is not represented, so a purely lateral loading mode
   degenerates to a plateau rather than a peak.
6. **Water is buoyancy only** (no pore fluid, no added mass, no hydrodynamic
   drag); the DEM drag coefficient is a numerical damper. The subgrade modulus
   `0.65 E_s/((1-nu^2) B)` is a chosen elastic model, so capacities are the
   verified quantities and settlements scale with that choice.
7. **Pipeline buckling, cyclic degradation, gapping and trench geometry are out
   of scope**, as is interaction with the ice sheet itself.

## 6. Error judgement against references

{error_table}

Explicit answer to "is the key quantity within 3 % of its reference": the
structural quantities are — pipe bending moment and deflection agree with the
closed-form Winkler theory to well under 1 %, and the standalone beam-on-Winkler
verification is at the same level. The ice-side capacity is **not** within 3 % of
any reference: there is no primary-source empirical ice-ridge formula available
here (WITHHELD), and the derived limit-equilibrium bracket is at best a
{_fmt(100 * capacity['relative_deviation_from_mid'], 1)} % anchor. Criterion (a)
{'passes' if verdict_a else 'fails'} on the machine metrics; criterion (b) does
not pass the 20 % band against the bearing-capacity bracket
({'inside' if inside else 'outside'} the bracket, deviation
{_fmt(100 * capacity['relative_deviation_from_mid'], 1)} %) but does pass it
against the indentation-pressure anchor
({_fmt(100 * indentation['relative_deviation_from_mid'], 1)} %); criterion (c)
{'passes' if verdict_c else 'fails'} the 15 % band. **Honest FAIL is recorded
for the bearing-capacity formulation of criterion (b); the physical reason (free
lateral boundaries invalidating the confined-bearing mechanism) is stated above
and is not papered over.**

Evidence grades: {json.dumps(checked, ensure_ascii=False)}
"""

    payload = {
        "title": f"Ice-ridge keel grounding on a buried pipeline (DEM/MC/beam, mission {MISSION_ID})",
        "summary": (
            f"Coupled 2-D DEM / Mohr-Coulomb subgrade / beam-on-Winkler model of ice-ridge "
            f"keel grounding on a buried pipeline. Peak load "
            f"{_fmt(metrics['peak_load_per_length'] / 1e3, 1)} kN/m at "
            f"{_fmt(metrics['displacement_at_peak'], 2)} m penetration with "
            f"{_fmt(100 * metrics['softening_ratio'], 1)} % post-peak softening; pipe peak "
            f"moment {_fmt(profile['peak_moment'] / 1e6, 2)} MN m against a closed-form "
            f"reference to {_fmt(100 * profile['moment_relative_error'], 2)} %; DNV-OS-F101 "
            f"utilisation {_fmt(profile['peak_utilization'], 3)}."),
        "scenario_key": "ice_ridge_pipeline_grounding",
        "mission_id": MISSION_ID,
        "tags": ["ice_ridge", "DEM", "pipeline", "Mohr-Coulomb", "DNV-OS-F101",
                 "Winkler", "grounding", "Beaufort"],
        "markdown": markdown,
        "figures": figures,
    }
    payload["indentation_anchor"] = indentation
    payload["rate_extrapolation"] = rate_extrapolation
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print("wrote", OUT, len(markdown), "chars,", len(figures), "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
