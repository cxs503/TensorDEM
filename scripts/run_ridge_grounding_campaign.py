"""W5-N03 campaign: ice-ridge keel grounding (Kerr foundation + bonded-particle keel).

Runs every machine-checkable acceptance criterion and writes a single JSON
artifact (all numbers programmatic, nothing hand-copied) plus the report figures.

Usage::

    PYTHONPATH=src python scripts/run_ridge_grounding_campaign.py [--fast]

Outputs
-------
``artifacts/w5_n03_ridge_grounding.json``  -- criteria, tables, profiles
``docs/figures/w5_n03_*.png``              -- figures used by the report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem import kerr_foundation as kf          # noqa: E402
from tensordem import keel_lattice as kl            # noqa: E402
from tensordem import local_crushing as lc          # noqa: E402
from tensordem import ridge_grounding as rg         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _round_floats(obj, digits=10):
    if isinstance(obj, dict):
        return {k: _round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, digits) for v in obj]
    if isinstance(obj, float):
        return round(obj, digits)
    return obj


def stage_kerr(scenario: rg.RidgeGroundingScenario, fast: bool) -> dict:
    t0 = time.time()
    foundation = rg.kerr_foundation_for_keel(scenario)
    winkler = kf.KerrFoundation(foundation.upper, foundation.lower, 0.0)
    ei = scenario.section.ei
    load = 3.0e6
    comparison = kf.compare_analytic_fem(scenario.keel_length, ei, foundation, load,
                                         elements=240 if not fast else 96)
    comparison_winkler = kf.compare_analytic_fem(scenario.keel_length, ei, winkler, load,
                                                 elements=240 if not fast else 96)
    convergence = kf.convergence_study(scenario.keel_length, ei, foundation, load,
                                       element_counts=(24, 48, 96, 192, 384) if not fast
                                       else (24, 48, 96))
    analytic = kf.kerr_beam_analytic(scenario.keel_length, ei, foundation, load)
    fem = kf.kerr_beam_fem(scenario.keel_length, ei, foundation, load, elements=240 if not fast else 96)
    positions = [1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0]
    scan = kf.moving_load_scan(scenario.keel_length, ei, foundation, load, positions,
                               elements=240 if not fast else 96)
    response = kf.load_deflection_response(
        scenario.keel_length, ei, foundation,
        [0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6] if not fast else [1e6, 3e6, 5e6],
        elements=240 if not fast else 96)
    analysis = rg.kerr_design_summary(scenario, load)
    return {
        "foundation": foundation.as_dict(),
        "winkler_equivalent": foundation.equivalent_winkler,
        "characteristic_length_m": kf.characteristic_length(ei, foundation),
        "analytic_vs_fem": comparison,
        "analytic_vs_fem_winkler": comparison_winkler,
        "mesh_convergence": convergence,
        "moving_load_scan": {
            "worst_deflection_m": scan["worst_deflection_m"],
            "worst_pressure_Pa": scan["worst_pressure_Pa"],
            "worst_position_m": scan["worst_position_m"],
            "x_m": scan["x_m"],
            "deflection_envelope_m": scan["deflection_envelope_m"],
        },
        "load_deflection_response": response,
        "design_summary": analysis,
        "profiles": {
            "x_m": analytic.x.tolist(),
            "analytic_deflection_m": analytic.deflection.tolist(),
            "fem_deflection_m": np.interp(analytic.x, fem.x, fem.deflection).tolist(),
            "analytic_pressure_Pa": analytic.pressure.tolist(),
            "analytic_shear_layer_m": analytic.shear_layer.tolist(),
        },
        "runtime_s": round(time.time() - t0, 2),
    }


def stage_calibration(scenario: rg.RidgeGroundingScenario, radius: float,
                      bond_stiffness: float | None, verify: bool = True,
                      do_cantilever: bool = True) -> dict:
    calibration = kl.calibrate_bond_stiffness(scenario.keel_length, scenario.keel_thickness,
                                              radius, scenario.target_e2d)
    if bond_stiffness is not None:
        pos, s = kl.triangular_lattice(scenario.keel_length, scenario.keel_thickness, radius)
        pairs = kl.neighbour_pairs(pos, s)
        uniaxial = kl.uniaxial_strain_test(pos, pairs, bond_stiffness) if verify else {}
        cantilever = kl.cantilever_test(pos, pairs, bond_stiffness) if do_cantilever else None
        calibration = {
            "test": "reused_bond_stiffness",
            "bond_length_m": s,
            "particles": int(pos.shape[0]),
            "bonds": int(pairs.shape[0]),
            "target_E_2D_N_per_m": scenario.target_e2d,
            "bond_stiffness_N_per_m": float(bond_stiffness),
            "realised_E_2D_N_per_m": uniaxial.get("E_2D_N_per_m"),
            "relative_error": abs(uniaxial["E_2D_N_per_m"] - scenario.target_e2d)
            / scenario.target_e2d if uniaxial else None,
            "cantilever": cantilever,
        }
    return calibration


def stage_quasi_static(scenario: rg.RidgeGroundingScenario, calibration: dict,
                       depths, max_active_iterations: int) -> dict:
    t0 = time.time()
    result = kl.quasi_static_indentation(
        scenario.keel_length, scenario.keel_thickness, calibration["bond_length_m"] / 2.0,
        calibration["bond_stiffness_N_per_m"], scenario.foundation_modulus,
        punch_width=scenario.punch_width, depths=depths,
        max_active_iterations=max_active_iterations)
    result["runtime_s"] = round(time.time() - t0, 2)
    for step in result["steps"]:
        step["keel_length_m"] = scenario.keel_length
    return result


def stage_keel_vs_kerr(scenario: rg.RidgeGroundingScenario, calibration: dict,
                       quasi_static: dict) -> dict:
    foundation = rg.kerr_foundation_for_keel(scenario)
    winkler = kf.KerrFoundation(foundation.upper, foundation.lower, 0.0)
    ei = calibration["cantilever"]["EI_Nm2"]
    kerr = rg.compare_keel_with_kerr(quasi_static["steps"], foundation, ei)
    wink = rg.compare_keel_with_kerr(quasi_static["steps"], winkler, ei)
    monotonic = rg.spearman_monotonic([s["punch_depth_m"] for s in quasi_static["steps"]],
                                      [s["reaction_N_per_m"] for s in quasi_static["steps"]])
    monotonic_w = rg.spearman_monotonic([s["punch_depth_m"] for s in quasi_static["steps"]],
                                        [s["beam_deflection_m"] for s in quasi_static["steps"]])
    breaking = scenario.breaking_strain
    onset = next((s["punch_depth_m"] for s in quasi_static["steps"]
                  if s["max_bond_strain"] > breaking), None)
    return {
        "breaking_strain": breaking,
        "fracture_onset_punch_depth_m": onset,
        "max_bond_strain_at_last_step": quasi_static["steps"][-1]["max_bond_strain"],
        "ei_calibrated_Nm2": ei,
        "ei_section_formula_Nm2": scenario.section.ei,
        "ei_relative_difference": abs(ei - scenario.section.ei) / scenario.section.ei,
        "kerr": kerr,
        "winkler": wink,
        "spearman_force_vs_penetration": monotonic,
        "spearman_deflection_vs_penetration": monotonic_w,
        "criterion_b_met": kerr["criterion_15_percent_met"],
        "criterion_c_met": monotonic["spearman_rho"] > 0.95,
    }


def stage_crushing(scenario: rg.RidgeGroundingScenario, pressures, fast: bool) -> dict:
    panel = scenario.hull_panel
    rows = []
    for pressure in pressures:
        numerical = lc.membrane_panel_energy(
            panel, pressure, grid=(13, 17) if fast else (15, 19),
            max_iterations=400 if fast else 2500)
        jones = lc.rigid_plastic_membrane_deflection(pressure, panel)
        variational_deflection = pressure / lc.variational_membrane_pressure(1.0, panel)
        variational = lc.variational_membrane_pressure(variational_deflection, panel)
        rows.append({
            "pressure_Pa": pressure,
            "analytic_deflection_m": jones,
            "variational_deflection_m": variational_deflection,
            "numerical_deflection_m": numerical["central_deflection_m"],
            "relative_error_vs_closed_form": numerical["relative_error"],
            "variational_pressure_at_numerical_Pa": variational,
            "relative_error_vs_variational": abs(
                numerical["central_deflection_m"] - variational_deflection) / variational_deflection,
            "max_effective_strain": numerical["max_effective_strain"],
            "plastic_ratio": numerical["plastic_ratio"],
            "optimizer_converged": numerical["optimizer_converged"],
            "optimizer_iterations": numerical["optimizer_iterations"],
        })
    # design-point damage from the integrated grounding force of the Kerr solution
    foundation = rg.kerr_foundation_for_keel(scenario)
    design = kf.kerr_beam_analytic(scenario.keel_length, scenario.section.ei, foundation, 3.0e6)
    contact_pressure = rg.pressure_from_line_load(design.max_pressure, scenario.punch_width)
    damage = rg.contact_damage(contact_pressure, scenario)
    best = min(rows, key=lambda r: r["relative_error_vs_closed_form"])
    return {
        "panel": lc.crushing_damage(0.0, panel)["panel"],
        "rows": rows,
        "worst_error_vs_closed_form": max(r["relative_error_vs_closed_form"] for r in rows),
        "worst_error_vs_variational": max(r["relative_error_vs_variational"] for r in rows),
        "design_point": {
            "peak_foundation_pressure_Pa": design.max_pressure,
            "contact_pressure_Pa": contact_pressure,
            "damage": damage,
        },
    }


def stage_grounding(scenario: rg.RidgeGroundingScenario) -> dict:
    depths = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0]
    return rg.grounding_probability(depths, scenario)


def stage_fragment_interface(scenario: rg.RidgeGroundingScenario) -> dict:
    # representative crushed-rubble distribution: power-law tail, 10 mm .. 400 mm
    sizes = [0.010, 0.025, 0.050, 0.100, 0.200, 0.400]
    counts = [520.0, 210.0, 96.0, 38.0, 12.0, 3.0]
    distribution = rg.fragment_size_distribution(sizes, counts)
    mapping = rg.lattice_from_fragment_sizes(distribution, particle_radius_ratio=0.5)
    return {"distribution": distribution, "mapping": mapping}


def stage_dynamic_fracture(scenario: rg.RidgeGroundingScenario, calibration: dict,
                           increments: int, steps: int, depth_step: float = 0.0025) -> dict:
    """Dynamic DEM run with bond breaking: force history + break pattern."""
    import torch

    torch.set_num_threads(4)
    radius = 0.015
    length, thickness = 3.0, 0.7
    cal = kl.calibrate_bond_stiffness(length, thickness, radius, scenario.target_e2d)
    config = kl.KeelLatticeConfig(
        length=length, thickness=thickness, radius=radius,
        bond_stiffness=cal["bond_stiffness_N_per_m"],
        breaking_strain=scenario.breaking_strain,
        foundation_modulus=scenario.foundation_modulus,
        punch_width=0.3, mass_scale=10.0)
    dem = kl.KeelLatticeDEM(config)
    history = []
    t0 = time.time()
    for _ in range(increments):
        info = dem.advance_punch(depth_step, relax_steps=steps, energy_tolerance=1e-3)
        relax = info["relax"]
        history.append({
            "step": dem.step_count,
            "punch_depth_m": info["penetration_m"],
            "reaction_N_per_m": info["reaction_N_per_m"],
            "beam_deflection_m": dem.beam_deflection(dem.punch_center),
            "broken_bonds": info["broken_bonds"],
            "kinetic_over_elastic": relax["kinetic_energy_J"] / max(relax["elastic_energy_J"], 1e-30),
            "converged": relax["converged"],
        })
    fragments = dem.fragment_size_distribution(min_size=0.0)
    profile = dem.pressure_profile(bins=41)
    positions = dem.pos.cpu().numpy()
    broken = (~dem.alive).cpu().numpy()
    top = positions[:, 1] > 0.5 * thickness
    return {
        "config": config.__dict__ if hasattr(config, "__dict__") else {},
        "particles": dem.particles,
        "bonds": dem.bonds,
        "dt_s": dem.dt,
        "breaking_strain": scenario.breaking_strain,
        "history": history,
        "broken_bonds": dem.broken_bonds,
        "broken_fraction": dem.broken_bonds / max(dem.bonds, 1),
        "fragments": fragments,
        "pressure_profile": profile,
        "scatter": {
            "x_m": positions[top, 0].tolist(),
            "y_m": positions[top, 1].tolist(),
        },
        "elastic_energy_J": dem.elastic_energy(),
        "runtime_s": round(time.time() - t0, 2),
    }


def figures(artifact: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    made = []
    pro = artifact["kerr"]["profiles"]
    x = np.asarray(pro["x_m"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, np.asarray(pro["analytic_deflection_m"]) * 1e3, "k-", lw=2,
                 label="Kerr analytic")
    axes[0].plot(x, np.asarray(pro["fem_deflection_m"]) * 1e3, "r--", lw=1.4,
                 label="mixed FEM")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("deflection w [mm]")
    axes[0].legend()
    axes[0].set_title("(a) beam on Kerr foundation, P = 3 MN/m")
    axes[1].plot(x, np.asarray(pro["analytic_pressure_Pa"]) / 1e3, "k-", lw=2,
                 label="contact pressure p")
    twin = axes[1].twinx()
    twin.plot(x, np.asarray(pro["analytic_shear_layer_m"]) * 1e3, "b:", lw=1.4,
              label="shear-layer w_s")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("p [kPa]")
    twin.set_ylabel("w_s [mm]")
    axes[1].set_title("(b) foundation pressure and shear layer")
    fig.tight_layout()
    path = FIGURES / "w5_n03_kerr_beam.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    conv = artifact["kerr"]["mesh_convergence"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.loglog([c["element_size_m"] for c in conv],
              [c["deflection_profile_rel_error"] for c in conv], "o-", label="max profile error")
    ax.loglog([c["element_size_m"] for c in conv],
              [c["peak_deflection_rel_error"] for c in conv], "s--", label="peak error")
    ax.axhline(0.03, color="r", ls=":", label="3% criterion")
    ax.set_xlabel("element size [m]")
    ax.set_ylabel("relative error vs analytic")
    ax.legend()
    ax.set_title("Kerr analytic vs FEM mesh convergence")
    fig.tight_layout()
    path = FIGURES / "w5_n03_kerr_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    steps = artifact["keel_vs_kerr"]["kerr"]["steps"]
    w = np.asarray([s["beam_deflection_m"] for s in steps])
    p_dem = np.asarray([s["discrete_resistance_N_per_m"] for s in steps])
    p_kerr = np.asarray([s["kerr_resistance_N_per_m"] for s in steps])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(w * 1e3, p_dem / 1e6, "ko-", label="discrete bonded-particle keel")
    axes[0].plot(w * 1e3, p_kerr / 1e6, "r--", label="Kerr beam analytic")
    axes[0].set_xlabel("foundation-level deflection w [mm]")
    axes[0].set_ylabel("grounding resistance P [MN/m]")
    axes[0].legend()
    axes[0].set_title("(a) DEM keel vs Kerr foundation beam")
    axes[1].plot(w * 1e3, np.asarray([s["relative_error"] for s in steps]) * 100, "o-")
    axes[1].axhline(15, color="r", ls=":", label="15% criterion")
    axes[1].set_xlabel("deflection [mm]")
    axes[1].set_ylabel("relative error [%]")
    axes[1].legend()
    axes[1].set_title("(b) deviation (max %.1f%%)"
                      % (100 * artifact["keel_vs_kerr"]["kerr"]["max_relative_error"]))
    fig.tight_layout()
    path = FIGURES / "w5_n03_keel_vs_kerr.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    design = artifact["quasi_static"]["steps"][-1]
    bx = np.asarray(design["bottom_x_m"])
    bw = np.asarray(design["bottom_settlement_m"])
    foundation = kf.KerrFoundation(**{
        "upper": artifact["kerr"]["foundation"]["upper_N_per_m3"],
        "lower": artifact["kerr"]["foundation"]["lower_N_per_m3"],
        "shear": artifact["kerr"]["foundation"]["shear_N"]})
    ei = artifact["keel_vs_kerr"]["ei_calibrated_Nm2"]
    load = design["reaction_N_per_m"]
    continuum = kf.kerr_beam_fem(artifact["scenario"]["keel_length"], ei, foundation, load,
                                 elements=240)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(bx, bw * 1e3, "k.", ms=3, label="discrete keel (bottom row)")
    axes[0].plot(continuum.x, continuum.deflection * 1e3, "r-", lw=1.5, label="Kerr FEM")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("settlement [mm]")
    axes[0].legend()
    axes[0].set_title("(a) deflection profile, P = %.2f MN/m" % (load / 1e6))
    prof = artifact["dynamic_fracture"]["pressure_profile"]
    axes[1].plot(prof["x_m"], np.asarray(prof["pressure_Pa"]) / 1e6, "b-")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("contact pressure [MPa]")
    axes[1].set_title("(b) dynamic DEM contact pressure (fracture run)")
    fig.tight_layout()
    path = FIGURES / "w5_n03_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    crush = artifact["crushing"]["rows"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    pressures = np.linspace(0.2e6, 8e6, 50)
    panel = artifact["scenario"]["hull_panel"]
    pp = lc.PlatePanel(**panel)
    ax.plot(pressures / 1e6, [lc.rigid_plastic_membrane_deflection(p, pp) * 1e3 for p in pressures],
            "k-", label="rigid-plastic closed form (factor 8)")
    ax.plot(pressures / 1e6, [p / lc.variational_membrane_pressure(1.0, pp) * 1e3
                              for p in pressures], "b--",
            label="variational closed form (factor 6.4)")
    ax.plot([r["pressure_Pa"] / 1e6 for r in crush],
            [r["numerical_deflection_m"] * 1e3 for r in crush], "ro", label="numerical membrane")
    ax.axhline(pp.thickness * 1e3, color="gray", ls=":", label="plate thickness")
    ax.set_xlabel("contact pressure [MPa]")
    ax.set_ylabel("panel deflection [mm]")
    ax.legend()
    ax.set_title("Local crushing: membrane solutions")
    fig.tight_layout()
    path = FIGURES / "w5_n03_crushing.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    ground = artifact["grounding"]["rows"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot([r["water_depth_m"] for r in ground],
            [r["grounding_probability"] for r in ground], "o-")
    ax.set_xlabel("water depth [m]")
    ax.set_ylabel("P(keel touches seabed)")
    ax.set_title("Grounding probability (log-normal keel draft)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "w5_n03_grounding_probability.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    dyn = artifact["dynamic_fracture"]
    hist = dyn["history"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot([h["punch_depth_m"] * 1e3 for h in hist],
                 [h["reaction_N_per_m"] / 1e6 for h in hist], "o-")
    axes[0].set_xlabel("punch advance [mm]")
    axes[0].set_ylabel("reaction [MN/m]")
    axes[0].set_title("(a) force history (dynamic DEM, bonds may break)")
    ax2 = axes[0].twinx()
    ax2.plot([h["punch_depth_m"] * 1e3 for h in hist], [h["broken_bonds"] for h in hist],
             "r--")
    ax2.set_ylabel("broken bonds", color="r")
    sc = dyn["scatter"]
    axes[1].scatter(sc["x_m"], sc["y_m"], s=1.5, c="tab:blue")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title("(b) top-layer particles after %.2f%% broken bonds"
                      % (100 * dyn["broken_fraction"]))
    fig.tight_layout()
    path = FIGURES / "w5_n03_dynamic_fracture.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="reduced resolutions")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    scenario = rg.RidgeGroundingScenario()
    artifact: dict = {"scenario": scenario.as_dict(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def log(stage: str, payload: dict) -> None:
        artifact[stage] = payload
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACTS / "w5_n03_ridge_grounding.partial.json", "w") as handle:
            json.dump(_jsonable(_round_floats(artifact)), handle, indent=1)
        print(f"[{stage}] done", flush=True)

    print("stage: kerr", flush=True)
    log("kerr", stage_kerr(scenario, args.fast))

    print("stage: calibration (base)", flush=True)
    calibration = stage_calibration(scenario, 0.02, None)
    log("calibration", calibration)

    print("stage: quasi-static indentation (base)", flush=True)
    depths = [0.004, 0.008, 0.012, 0.016, 0.020, 0.025, 0.030] if not args.fast \
        else [0.008, 0.016, 0.024]
    quasi = stage_quasi_static(scenario, calibration, depths, 40)
    log("quasi_static", quasi)

    print("stage: keel vs kerr", flush=True)
    log("keel_vs_kerr", stage_keel_vs_kerr(scenario, calibration, quasi))

    print("stage: crushing", flush=True)
    log("crushing", stage_crushing(scenario, [2.0e6, 5.0e6] if not args.fast else [2.0e6], args.fast))

    print("stage: grounding probability", flush=True)
    log("grounding", stage_grounding(scenario))

    print("stage: fragment interface", flush=True)
    log("fragment_interface", stage_fragment_interface(scenario))

    print("stage: dynamic fracture demo", flush=True)
    log("dynamic_fracture", stage_dynamic_fracture(scenario, calibration,
                                                   increments=40 if not args.fast else 6,
                                                   steps=250 if not args.fast else 120,
                                                   depth_step=0.004 if not args.fast else 0.0025))

    print("stage: resolution convergence", flush=True)
    radii = [0.0102, 0.0072] if not args.fast else [0.0144, 0.0102]
    rows = []
    bond_stiffness = calibration["bond_stiffness_N_per_m"]
    for radius in radii:
        cal = stage_calibration(scenario, radius, bond_stiffness,
                                do_cantilever=(radius == 0.02))
        qs = stage_quasi_static(scenario, cal, [0.010, 0.020], 20)
        rows.append({
            "radius_m": radius,
            "particles": cal["particles"],
            "bonds": cal["bonds"],
            "bond_stiffness_N_per_m": cal["bond_stiffness_N_per_m"],
            "realised_E_2D_N_per_m": cal.get("realised_E_2D_N_per_m"),
            "EI_Nm2": (cal["cantilever"] or {}).get("EI_Nm2"),
            "depths": [s["punch_depth_m"] for s in qs["steps"]],
            "resistances_N_per_m": [s["reaction_N_per_m"] for s in qs["steps"]],
            "deflections_m": [s["beam_deflection_m"] for s in qs["steps"]],
            "runtime_s": qs["runtime_s"],
        })
        print(f"  radius {radius} particles {cal['particles']} P {rows[-1]['resistances_N_per_m']}", flush=True)
    low, high = rows[0], rows[-1]
    diffs = [abs(a - b) / b if b > 0.0 else float("nan")
             for a, b in zip(low["resistances_N_per_m"], high["resistances_N_per_m"])]
    convergence = {
        "rows": rows,
        "particle_counts": [low["particles"], high["particles"]],
        "relative_differences": diffs,
        "max_relative_difference": max(diffs),
        "criterion_5_percent_met": bool(max(diffs) < 0.05),
    }
    log("resolution_convergence", convergence)

    print("stage: figures", flush=True)
    artifact["figures"] = figures(artifact)
    artifact["criteria"] = {
        "a_kerr_analytic_vs_fem_below_3pct":
            artifact["kerr"]["analytic_vs_fem"]["deflection_profile_rel_error"] < 0.03,
        "a_value": artifact["kerr"]["analytic_vs_fem"]["deflection_profile_rel_error"],
        "b_dem_vs_kerr_below_15pct": artifact["keel_vs_kerr"]["criterion_b_met"],
        "b_value": artifact["keel_vs_kerr"]["kerr"]["max_relative_error"],
        "c_force_monotonic_spearman": artifact["keel_vs_kerr"]["criterion_c_met"],
        "c_value": artifact["keel_vs_kerr"]["spearman_force_vs_penetration"]["spearman_rho"],
        "d_crushing_vs_variational_membrane_below_10pct":
            artifact["crushing"]["worst_error_vs_variational"] < 0.10,
        "d_value_variational": artifact["crushing"]["worst_error_vs_variational"],
        "d_crushing_vs_factor8_closed_form_below_10pct":
            artifact["crushing"]["worst_error_vs_closed_form"] < 0.10,
        "d_value_factor8_closed_form": artifact["crushing"]["worst_error_vs_closed_form"],
        "e_particle_convergence_below_5pct": convergence["criterion_5_percent_met"],
        "e_value": convergence["max_relative_difference"],
    }
    artifact["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = ARTIFACTS / "w5_n03_ridge_grounding.json"
    with open(path, "w") as handle:
        json.dump(_jsonable(_round_floats(artifact)), handle, indent=1)
    partial = ARTIFACTS / "w5_n03_ridge_grounding.partial.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps(artifact["criteria"], indent=1), flush=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())