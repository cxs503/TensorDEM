"""Formation ice-channel campaign: acceptance evidence, figures and report.

Every number quoted in ``docs/formation_ice_channel.md`` is produced here and
written to ``artifacts/formation/formation_campaign.json``; the report is rendered
from that file by ``scripts/render_formation_report.py``, so no value is
transcribed by hand.

Usage
-----
    PYTHONPATH=src python scripts/run_formation_campaign.py [--skip-dem] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from tensordem.formation import (FormationScenario, WakeInterferenceConfig,
                                 _channel_state, _ship_encounter,
                                 open_water_resistance, optimize_formation,
                                 simulate_formation)
from tensordem.formation_hydro import (FormationFlow, FormationFlowConfig,
                                       HullWaterline, froude_number)
from tensordem.formation_resistance import (IcebreakerShip, IceResistanceModel,
                                            IceSheet, calibrate_ice_resistance,
                                            level_ice_resistance)
from tensordem.formation_thermal import (ChannelRefreezing, ChannelThermalConfig,
                                         neumann_thickness, stefan_thickness)

REFERENCE_CSV = (ROOT / "benchmarks" / "dem3d" / "ship_ice_validation"
                 / "mt_uikku_level_ice.csv")
SHIP_LENGTH = 100.0
SHIP_BEAM = 20.0
SHIP_DRAFT = 8.0


def _ships(count: int) -> list[IcebreakerShip]:
    return [IcebreakerShip(name=f"P{index + 1}", length=SHIP_LENGTH, beam=SHIP_BEAM,
                           draft=SHIP_DRAFT) for index in range(count)]


# --------------------------------------------------------------------------- (a)
def single_ship_degeneracy(scale: float) -> dict[str, Any]:
    ship = _ships(1)[0]
    scenario = FormationScenario(
        ships=[ship], speed=2.0, parent_thickness=1.0, ice=IceSheet(),
        resistance_model=IceResistanceModel(scale_factor=scale), include_confinement=False)
    result = simulate_formation(scenario)
    row = result["passes"][0]["ships"][0]
    ice = level_ice_resistance(ship, scenario.resistance_model, scenario.parent_thickness)
    ow = open_water_resistance(ship, scenario.speed)
    baseline = ice["total_resistance_N"] + ow["total_resistance_N"]
    confined = simulate_formation(
        FormationScenario(ships=[ship], speed=2.0, parent_thickness=1.0, ice=IceSheet(),
                          resistance_model=IceResistanceModel(scale_factor=scale)))
    confined_total = confined["passes"][0]["ships"][0]["total_resistance_N"]
    error = (row["total_resistance_N"] - baseline) / baseline * 100.0
    return {
        "formation_n1_total_N": row["total_resistance_N"],
        "baseline_total_N": baseline,
        "formation_n1_ice_N": row["ice_resistance_N"],
        "baseline_ice_N": ice["total_resistance_N"],
        "formation_n1_open_water_N": row["open_water_resistance_N"],
        "baseline_open_water_N": ow["total_resistance_N"],
        "relative_error_percent": error,
        "with_confinement_total_N": confined_total,
        "with_confinement_error_percent": (confined_total - baseline) / baseline * 100.0,
        "tolerance_percent": 3.0,
        "passes": abs(error) < 3.0,
    }


# --------------------------------------------------------------------------- (b)
def _loglog_slope(points: list[tuple[float, float]]) -> float:
    usable = [(x, y) for x, y in points if x >= 2.0 and y > 0.0]
    if len(usable) < 2:
        return float("nan")
    n = len(usable)
    sx = sum(math.log(x) for x, _ in usable)
    sy = sum(math.log(y) for _, y in usable)
    sxx = sum(math.log(x) ** 2 for x, _ in usable)
    sxy = sum(math.log(x) * math.log(y) for x, y in usable)
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def interference_asymptotics() -> dict[str, Any]:
    hull = HullWaterline(length=SHIP_LENGTH, beam=SHIP_BEAM, panels_per_side=28)
    flow = FormationFlow([hull, hull], FormationFlowConfig(speed=2.0, channel_half_width=None))
    spacings = [1.02, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    bearings = [0.0, 15.0, 30.0, 60.0, 90.0]
    rows = flow.interference_tables(spacings, bearings)
    column = [row for row in rows if row["bearing_deg"] == 0.0]
    far_row = min(rows, key=lambda row: abs(row["C_follower_from_leader"]))
    contact_column = flow.interference([(0.0, 0.0), (SHIP_LENGTH + 1.0e-3, 0.0)])
    contact_abeam = flow.interference([(0.0, 0.0), (0.0, SHIP_BEAM + 1.0e-3)])
    c_contact = float(contact_column["coefficients"][1, 0])
    c_abeam = float(contact_abeam["coefficients"][1, 0])
    return {
        "rows": rows,
        "column_rows": column,
        "max_abs_coefficient": max(abs(row["C_follower_from_leader"]) for row in rows),
        "min_abs_coefficient": abs(far_row["C_follower_from_leader"]),
        "min_abs_row": far_row,
        "far_field_loglog_slope": _loglog_slope(
            [(row["spacing_over_length"], abs(row["C_follower_from_leader"])) for row in column]),
        "contact_column_coefficient": c_contact,
        "contact_abeam_coefficient": c_abeam,
        "contact_bounded": abs(c_contact) <= 1.0 and abs(c_abeam) <= 1.0,
        "neumann_residual": float(contact_column["neumann_residual"]),
        "tolerance_far_field": 1.0e-3,
        "tolerance_bound": 1.0,
        "passes": (abs(far_row["C_follower_from_leader"]) < 1.0e-3
                   and abs(c_contact) <= 1.0 and abs(c_abeam) <= 1.0),
    }


# --------------------------------------------------------------------------- (c)
def thermal_qualification() -> dict[str, Any]:
    configs = [("reference", ChannelThermalConfig()),
               ("cold", ChannelThermalConfig(air_temperature=-30.0)),
               ("warm", ChannelThermalConfig(air_temperature=-5.0))]
    times = [600.0, 3600.0, 21600.0, 86400.0, 604800.0]
    rows: list[dict[str, Any]] = []
    for label, config in configs:
        model = ChannelRefreezing(config)
        numeric = model.thickness_history(times)
        for t, value in zip(times, numeric):
            stefan = stefan_thickness(t, config)
            neumann = neumann_thickness(t, config)
            rows.append({"case": label, "time_s": t, "numeric_m": value, "stefan_m": stefan,
                         "neumann_m": neumann,
                         "error_vs_stefan_percent": (value - stefan) / stefan * 100.0,
                         "error_vs_neumann_percent": (value - neumann) / neumann * 100.0})
    refinement = []
    for nodes in (20, 40, 80, 160):
        config = ChannelThermalConfig(nodes=nodes)
        value = ChannelRefreezing(config).thickness(86400.0)
        reference = neumann_thickness(86400.0, config)
        refinement.append({"nodes": nodes, "thickness_m": value,
                           "error_vs_neumann_percent": (value - reference) / reference * 100.0})
    widths = []
    canal = ChannelRefreezing(ChannelThermalConfig())
    for age in (0.0, 3600.0, 21600.0, 86400.0, 259200.0, 604800.0):
        widths.append({"age_s": age, "refrozen_m": canal.thickness(age),
                       "open_width_m": canal.channel_width(age, 23.0),
                       "lateral_growth_m": canal.lateral_growth(age)})
    default = ChannelThermalConfig()
    worst = max(abs(row["error_vs_stefan_percent"]) for row in rows)
    return {
        "rows": rows,
        "max_abs_error_vs_stefan_percent": worst,
        "max_abs_error_vs_neumann_percent": max(abs(row["error_vs_neumann_percent"])
                                               for row in rows),
        "tolerance_percent": 5.0,
        "passes": worst < 5.0,
        "refinement": refinement,
        "channel_width_evolution": widths,
        "parameters": {"stefan_number": default.stefan_number,
                       "thermal_diffusivity_m2_s": default.thermal_diffusivity,
                       "freezing_range_K": default.freezing_range},
    }


# ---------------------------------------------------------------------------
def resistance_calibration() -> dict[str, Any]:
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ship = IcebreakerShip(name="MT Uikku equivalent", length=105.0, beam=19.2, draft=7.0,
                          block_coefficient=0.75)
    result = calibrate_ice_resistance(rows, ship, holdout=3)
    everything = result["training"] + result["holdout"]
    thick = [row for row in everything if row["thickness_m"] >= 0.75]
    thin = [row for row in everything if row["thickness_m"] < 0.75]
    result["all_case_mape_percent"] = (sum(abs(row["relative_error_percent"])
                                          for row in everything) / len(everything))
    result["thick_ice_mape_percent"] = (sum(abs(row["relative_error_percent"])
                                            for row in thick) / len(thick)) if thick else None
    result["thin_ice_mape_percent"] = (sum(abs(row["relative_error_percent"])
                                           for row in thin) / len(thin)) if thin else None
    result["thin_ice_case_ids"] = [row["test_id"] for row in thin]
    result["source"] = str(REFERENCE_CSV.relative_to(ROOT))
    return result


# --------------------------------------------------------------------------- (e)
def end_to_end_case(scale: float) -> dict[str, Any]:
    scenario = FormationScenario(
        ships=_ships(3), speed=2.0, parent_thickness=1.0, spacing_factor=2.0,
        ice=IceSheet(), resistance_model=IceResistanceModel(scale_factor=scale),
        transit_length=2.0e4, time_samples=25)
    return simulate_formation(scenario)


def refreezing_study(scale: float) -> dict[str, Any]:
    thermal = ChannelRefreezing(ChannelThermalConfig())
    scenario = FormationScenario(
        ships=_ships(3), speed=2.0, parent_thickness=1.0, spacing_factor=2.0,
        ice=IceSheet(), resistance_model=IceResistanceModel(scale_factor=scale),
        reuse_previous_channel=True, pass_interval_s=3600.0)
    sweep = []
    for age in (0.0, 300.0, 3600.0, 21600.0, 43200.0, 86400.0, 172800.0, 604800.0):
        state = _channel_state(scenario, thermal, age)
        encounter = _ship_encounter(scenario, scenario.ships[-1], 0.0, state)
        ice = encounter["resistance"]
        ow = open_water_resistance(scenario.ships[-1], scenario.speed)
        sweep.append({
            "channel_age_s": age,
            "refrozen_m": state["refrozen_thickness_m"],
            "stefan_m": state["stefan_reference_m"],
            "open_width_m": state["open_width_m"],
            "follower_mean_thickness_m": ice["mean_thickness_m"],
            "follower_ice_resistance_N": ice["total_resistance_N"],
            "follower_total_resistance_N": ice["total_resistance_N"] + ow["total_resistance_N"],
        })
    return {"sweep": sweep}


def interference_absolute_effect(scale: float) -> dict[str, Any]:
    rows = []
    coefficients = []
    for spacing in (1.5, 2.0, 3.0, 5.0):
        scenario = FormationScenario(
            ships=_ships(3), speed=2.0, parent_thickness=1.0, spacing_factor=spacing,
            ice=IceSheet(), resistance_model=IceResistanceModel(scale_factor=scale))
        result = simulate_formation(scenario)
        follower = result["passes"][0]["ships"][1]
        rows.append({
            "spacing_over_length": spacing,
            "follower_interference_sum": follower["interference_sum"],
            "follower_open_water_N": follower["open_water_resistance_N"],
            "follower_interference_N": (follower["open_water_effective_N"]
                                        - follower["open_water_resistance_N"]),
            "follower_total_N": follower["total_resistance_N"],
            "interference_share_percent": abs(follower["open_water_effective_N"]
                                              - follower["open_water_resistance_N"])
            / follower["total_resistance_N"] * 100.0,
        })
        table = scenario.flow().interference(scenario.offsets())
        coefficients.append({
            "spacing_over_length": spacing,
            "C_follower_from_leader": float(table["coefficients"][1, 0]),
            "C_leader_from_follower": float(table["coefficients"][0, 1]),
            "confinement_leader": float(table["confinement"][0]),
            "confinement_follower": float(table["confinement"][1]),
        })
    wakes = []
    for deficit in (0.0, 0.075, 0.15, 0.30):
        scenario = FormationScenario(
            ships=_ships(3), speed=2.0, parent_thickness=1.0, spacing_factor=2.0,
            ice=IceSheet(), resistance_model=IceResistanceModel(scale_factor=scale),
            wake=WakeInterferenceConfig(deficit=deficit))
        follower = simulate_formation(scenario)["passes"][0]["ships"][1]
        wakes.append({"wake_deficit": deficit,
                      "follower_interference_sum": follower["interference_sum"],
                      "follower_total_N": follower["total_resistance_N"]})
    return {"ice_case_rows": rows, "open_water_coefficients": coefficients,
            "wake_sensitivity": wakes}


def optimisation_study(scale: float) -> dict[str, Any]:
    scenario = FormationScenario(
        ships=_ships(3), speed=2.0, parent_thickness=1.0, spacing_factor=2.0,
        ice=IceSheet(), resistance_model=IceResistanceModel(scale_factor=scale),
        transit_length=2.0e4)
    return optimize_formation(scenario,
                              spacing_candidates=(1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0),
                              speed_candidates=(1.5, 2.0, 2.5),
                              max_duration_h=48.0, min_spacing_factor=1.5)


# --------------------------------------------------------------------------- (d)
def dem_campaign(energy_steps: int = 3200, coupling_steps: int = 1200) -> dict[str, Any]:
    """Multi-body DEM runs: N=1 degeneracy, energy audit, shared-field channel effect.

    The wedge bow is placed so that the initial penetration is exactly zero
    (``tip_x0 = x_min - radius / sin(theta)``); otherwise the run starts from a
    pre-stressed ice state and the energy audit cannot close.
    """
    from tensordem.dem3d import DEM3DConfig
    from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D
    from tensordem.formation_dem import FormationIceDEM3D, ShipBow

    radius = 0.05
    theta = math.radians(45.0)
    slow_speed = 0.1
    slow = DEM3DConfig(nx=8, ny=5, nz=3, radius=radius, tool_speed=slow_speed,
                       bond_stiffness=1200.0, contact_stiffness=3500.0,
                       breaking_strain=0.01, shear_breaking_strain=0.03,
                       drag=5.0, contact_damping=0.5, fix_x_edges=False, fix_bottom=False)
    probe = FormationIceDEM3D(slow, [ShipBow(name="P1", speed=slow_speed, center_y=0.0,
                                             half_width=0.1, tip_x0=0.0)])
    x_min = float(probe.initial_positions[:, 0].min())
    zero_overlap_tip = x_min - radius / math.sin(theta) - 1.0e-9
    gap_for_wedge_class = radius / math.sin(theta) - 0.5 * radius

    single = MovingWedgeBowIceDEM3D(slow, bow_speed=slow_speed, wedge_angle_deg=45.0,
                                    bow_half_width_m=0.1, initial_gap=gap_for_wedge_class)
    formation_one = FormationIceDEM3D(
        slow, [ShipBow(name="P1", speed=slow_speed, center_y=float(single.bow_center_y_m),
                       half_width=0.1, tip_x0=zero_overlap_tip, wedge_angle_deg=45.0,
                       tip_height_fraction=single.tip_height_fraction)])
    started = time.time()
    single_history = []
    formation_history = []
    for _ in range(energy_steps):
        single.step()
        formation_one.step()
        single_history.append(float(single.diagnostics()["bow_reaction_x_N"]))
        formation_history.append(formation_one.last_ship_diagnostics["P1"]["reaction_x_N"])
    degeneracy_runtime = time.time() - started
    single_force = single_history[-1]
    formation_force = formation_history[-1]
    audit = formation_one.audit()
    tolerance = 0.03 * abs(single_force) + 1.0e-12
    head = max(5, energy_steps // 2)
    trailing = max(5, energy_steps // 4)
    degeneracy = {
        "steps": energy_steps,
        "initial_gap_used_m": gap_for_wedge_class,
        "tip_x0_m": zero_overlap_tip,
        "single_ship_force_N": single_force,
        "formation_n1_force_N": formation_force,
        "relative_error_percent": ((formation_force - single_force) / single_force * 100.0
                                   if single_force else 0.0),
        "absolute_difference_N": formation_force - single_force,
        "max_abs_force_difference_head_N": max(
            abs(a - b) for a, b in zip(single_history[:head], formation_history[:head])),
        "max_abs_force_difference_trailing_N": max(
            abs(a - b) for a, b in zip(single_history[-trailing:], formation_history[-trailing:])),
        "broken_bonds_single": int(single.broken_bonds),
        "broken_bonds_formation": int(formation_one.broken_bonds),
        "tolerance_percent": 3.0,
        "passes": abs(formation_force - single_force) <= tolerance,
        "runtime_s": degeneracy_runtime,
        "dt_s": float(single.dt),
    }
    energy = {
        "steps": energy_steps, "work_in_J": audit["work_in_J"],
        "stored_energy_change_J": audit["stored_energy_change_J"],
        "released_bond_J": audit["released_bond_J"],
        "dissipation_J": audit["dissipation_J"],
        "dissipation_breakdown": audit["dissipation_breakdown"],
        "propulsion_J": audit["propulsion_J"], "ploughing_J": audit["ploughing_J"],
        "residual_J": audit["residual_J"],
        "relative_residual_percent": audit["relative_residual"] * 100.0,
        "tolerance_percent": 5.0,
        "passes": abs(audit["relative_residual"]) * 100.0 < 5.0,
        "broken_bonds": int(formation_one.broken_bonds), "dt_s": audit["dt_s"],
    }

    # --- shared ice field: a follower in the leader's lane vs an intact field ----
    fast_speed = 0.5
    fast = DEM3DConfig(nx=10, ny=5, nz=3, radius=radius, tool_speed=fast_speed,
                       bond_stiffness=1200.0, contact_stiffness=3500.0,
                       breaking_strain=0.01, shear_breaking_strain=0.03,
                       drag=2.0, contact_damping=0.5, fix_x_edges=False, fix_bottom=False)
    fast_probe = FormationIceDEM3D(fast, [ShipBow(name="P1", speed=fast_speed, center_y=0.0,
                                                 half_width=0.1, tip_x0=0.0)])
    fast_x_min = float(fast_probe.initial_positions[:, 0].min())
    fast_tip = fast_x_min - radius / math.sin(theta) - 1.0e-9
    lag = 0.05  # half a particle spacing behind the leader
    lane_ships = [
        ShipBow(name="P1", speed=fast_speed, center_y=0.0, half_width=0.1, tip_x0=fast_tip),
        ShipBow(name="P2", speed=fast_speed, center_y=0.0, half_width=0.1, tip_x0=fast_tip - lag),
    ]
    shared = FormationIceDEM3D(fast, lane_ships)
    control = FormationIceDEM3D(fast, [ShipBow(name="P2", speed=fast_speed, center_y=0.0,
                                               half_width=0.1, tip_x0=fast_tip - lag)])
    rollout = time.time()
    shared_forces = {"P1": [], "P2": []}
    control_forces: list[float] = []
    for _ in range(coupling_steps):
        shared.step()
        control.step()
        shared_forces["P1"].append(shared.last_ship_diagnostics["P1"]["reaction_x_N"])
        shared_forces["P2"].append(shared.last_ship_diagnostics["P2"]["reaction_x_N"])
        control_forces.append(control.last_ship_diagnostics["P2"]["reaction_x_N"])
    rollout_runtime = time.time() - rollout
    window = max(5, coupling_steps // 4)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    follower_shared = mean(shared_forces["P2"][-window:])
    follower_control = mean(control_forces[-window:])
    shared_field = {
        "steps": coupling_steps,
        "lag_m": lag,
        "particles": int(len(shared.positions)),
        "bonds": int(len(shared.pairs)),
        "window_steps": window,
        "shared_leader_mean_force_N": mean(shared_forces["P1"][-window:]),
        "shared_follower_mean_force_N": follower_shared,
        "control_follower_mean_force_N": follower_control,
        "follower_force_reduction_percent": ((follower_control - follower_shared)
                                             / follower_control * 100.0
                                             if follower_control else 0.0),
        "shared_broken_bonds": int(shared.broken_bonds),
        "control_broken_bonds": int(control.broken_bonds),
        "runtime_s": rollout_runtime,
        "follower_shared_history": shared_forces["P2"],
        "follower_control_history": control_forces,
        "leader_history": shared_forces["P1"],
    }
    return {
        "config": {key: value for key, value in slow.__dict__.items() if key != "dt"},
        "fast_config": {key: value for key, value in fast.__dict__.items() if key != "dt"},
        "dt_s": float(formation_one.dt),
        "fast_dt_s": float(shared.dt),
        "particles_single": int(len(formation_one.positions)),
        "bonds_single": int(len(formation_one.pairs)),
        "degeneracy": degeneracy,
        "energy": energy,
        "shared_field": shared_field,
    }


# --------------------------------------------------------------------------- figures
def make_figures(campaign: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[dict[str, str]] = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    thermal = campaign["thermal"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for case in ("reference", "cold", "warm"):
        rows = [row for row in thermal["rows"] if row["case"] == case]
        ax.plot([row["time_s"] for row in rows], [row["numeric_m"] for row in rows], "o-",
                ms=4, label=f"numeric ({case})")
        ax.plot([row["time_s"] for row in rows], [row["stefan_m"] for row in rows], "k--", lw=1.0)
        ax.plot([row["time_s"] for row in rows], [row["neumann_m"] for row in rows], "r:", lw=1.0)
    ax.plot([], [], "k--", lw=1.0, label="Stefan sqrt(2 k dT t / rho L)")
    ax.plot([], [], "r:", lw=1.0, label="Neumann similarity")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("channel age since clearance (s)"); ax.set_ylabel("new ice thickness (m)")
    ax.set_title("Channel refreezing: solver vs analytical Stefan solutions")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_stefan_growth.png", dpi=140); plt.close(fig)
    figures.append({"name": "fig1_stefan_growth.png",
                    "caption": "Fig. 1 - Refrozen channel thickness from the finite-volume Stefan "
                               "solver against the quasi-steady Stefan and exact Neumann solutions "
                               "for three air temperatures."})

    rows = campaign["interference"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for bearing in sorted({row["bearing_deg"] for row in rows}):
        subset = [row for row in rows if row["bearing_deg"] == bearing]
        ax.plot([row["spacing_over_length"] for row in subset],
                [row["C_follower_from_leader"] for row in subset], "o-", label=f"{bearing:.0f} deg")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("separation d / L"); ax.set_ylabel("C_ij (follower) [-]")
    ax.set_title("Hydrodynamic interference coefficient, two-ship formation")
    ax.grid(alpha=0.3); ax.legend(title="bearing", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_interference_matrix.png", dpi=140); plt.close(fig)
    figures.append({"name": "fig2_interference_matrix.png",
                    "caption": "Fig. 2 - Interference coefficient C_ij of the follower versus "
                               "separation and bearing (0 deg = column, 90 deg = abeam)."})

    e2e = campaign["end_to_end"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for ship in e2e["passes"][0]["ships"]:
        history = e2e["time_histories"][ship["name"]]
        ax.plot([row["time_s"] / 3600.0 for row in history],
                [row["resistance_N"] / 1000.0 for row in history], "-",
                label=f"{ship['name']} (dx = {ship['dx_m']:.0f} m)")
    ax.set_xlabel("mission time (h)"); ax.set_ylabel("total resistance (kN)")
    ax.set_title("Three-ship column, spacing 2L, parent ice 1 m")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_resistance_histories.png", dpi=140); plt.close(fig)
    figures.append({"name": "fig3_resistance_histories.png",
                    "caption": "Fig. 3 - Resistance time history of every ship in the three-ship "
                               "column case (1 m ice, spacing 2L, 2 m/s)."})

    sweep = campaign["refreezing"]["sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([row["channel_age_s"] / 3600.0 for row in sweep],
            [row["refrozen_m"] * 100.0 for row in sweep], "o-", color="tab:blue",
            label="refrozen thickness (cm)")
    ax.set_xscale("log")
    ax.set_xlabel("channel age (h)"); ax.set_ylabel("refrozen ice thickness (cm)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot([row["channel_age_s"] / 3600.0 for row in sweep],
             [row["follower_total_resistance_N"] / 1000.0 for row in sweep], "s-", color="tab:red")
    ax2.set_ylabel("follower total resistance (kN)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("Refreezing penalty on a following ship")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_refreezing_penalty.png", dpi=140); plt.close(fig)
    figures.append({"name": "fig4_refreezing_penalty.png",
                    "caption": "Fig. 4 - Refrozen channel thickness and the resulting follower "
                               "resistance versus channel age."})

    energy = campaign["dem"]["energy"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = ["stored", "bond release", "bow damping", "contact damping", "drag", "residual"]
    values = [energy["stored_energy_change_J"], energy["released_bond_J"],
              energy["dissipation_breakdown"]["bow_damping_J"],
              energy["dissipation_breakdown"]["contact_damping_J"],
              energy["dissipation_breakdown"]["drag_J"], energy["residual_J"]]
    ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:purple",
                                  "tab:brown", "tab:red"])
    ax.set_ylabel("energy (J)")
    ax.set_title(f"DEM multi-body energy audit (work in = {energy['work_in_J']:.3f} J)")
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig5_energy_audit.png", dpi=140); plt.close(fig)
    figures.append({"name": "fig5_energy_audit.png",
                    "caption": "Fig. 5 - Energy audit of the DEM multi-body run. Each dissipation "
                               "channel is integrated from its own force law; the residual is "
                               "time-integration error."})

    calibration = campaign["calibration"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    everything = calibration["training"] + calibration["holdout"]
    ax.scatter([row["measured_kN"] for row in calibration["training"]],
               [row["predicted_kN"] for row in calibration["training"]], c="tab:blue",
               label="calibration cases")
    ax.scatter([row["measured_kN"] for row in calibration["holdout"]],
               [row["predicted_kN"] for row in calibration["holdout"]], c="tab:red",
               marker="s", label="held-out cases")
    limit = max(row["measured_kN"] for row in everything) * 1.15
    ax.plot([0, limit], [0, limit], "k--", lw=1.0, label="1:1")
    for factor in (0.8, 1.25):
        ax.plot([0, limit], [0, factor * limit], ":", color="grey", lw=0.9)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel("measured resistance (kN)"); ax.set_ylabel("predicted resistance (kN)")
    ax.set_title("Level-ice resistance law vs MT Uikku reference cases")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig6_resistance_calibration.png", dpi=140)
    plt.close(fig)
    figures.append({"name": "fig6_resistance_calibration.png",
                    "caption": "Fig. 6 - Predicted versus measured level-ice resistance (dotted "
                               "lines: +/-25 %); squares are the held-out cases."})
    shared = (campaign.get("dem") or {}).get("shared_field")
    if shared:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        steps = range(len(shared["leader_history"]))
        ax.plot(list(steps), shared["leader_history"], lw=1.0, label="leader (P1)")
        ax.plot(list(steps), shared["follower_shared_history"], lw=1.0,
                label="follower in leader's lane (P2)")
        ax.plot(list(steps), shared["follower_control_history"], lw=1.0,
                label="same follower in intact ice")
        ax.set_xlabel("DEM step"); ax.set_ylabel("axial reaction (N)")
        ax.set_title("Shared ice field: channel effect on the following bow")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / "fig7_shared_field.png", dpi=140)
        plt.close(fig)
        figures.append({"name": "fig7_shared_field.png",
                        "caption": "Fig. 7 - DEM shared-field coupling: the follower's axial "
                                   "reaction in the leader's broken lane versus the same bow in an "
                                   "intact ice field."})
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "formation"))
    parser.add_argument("--skip-dem", action="store_true")
    parser.add_argument("--dem-steps", type=int, default=3200)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    calibration = resistance_calibration()
    scale = float(calibration["scale_factor"])
    print(f"[formation] resistance scale factor {scale:.4f}", flush=True)
    campaign: dict[str, Any] = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "reference_csv": calibration["source"],
        },
        "calibration": calibration,
        "single_ship_degeneracy": single_ship_degeneracy(scale),
        "interference": interference_asymptotics(),
        "thermal": thermal_qualification(),
        "end_to_end": end_to_end_case(scale),
        "refreezing": refreezing_study(scale),
        "interference_absolute": interference_absolute_effect(scale),
        "optimisation": optimisation_study(scale),
    }
    print("[formation] framework sections done", flush=True)
    if not args.skip_dem:
        campaign["dem"] = dem_campaign(energy_steps=args.dem_steps)
    else:
        campaign["dem"] = None
    figures = make_figures_only(campaign, out_dir)
    campaign["figures"] = figures
    campaign["runtime_s"] = time.time() - started
    (out_dir / "formation_campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[formation] wrote {out_dir / 'formation_campaign.json'}", flush=True)
    try:
        from scripts.render_formation_report import render
    except Exception:  # pragma: no cover - report rendering is optional
        from render_formation_report import render  # type: ignore
    render(out_dir / "formation_campaign.json", ROOT / "docs" / "formation_ice_channel.md")
    print("[formation] report rendered", flush=True)
    return 0


def make_figures_only(campaign: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    if campaign.get("dem") is None:
        # Figures 5 needs the DEM energy block; keep the others.
        campaign = dict(campaign)
        campaign["dem"] = {"energy": {"work_in_J": 0.0, "stored_energy_change_J": 0.0,
                                      "released_bond_J": 0.0,
                                      "dissipation_breakdown": {"bow_damping_J": 0.0,
                                                                "contact_damping_J": 0.0,
                                                                "drag_J": 0.0},
                                      "residual_J": 0.0}}
    return make_figures(campaign, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())