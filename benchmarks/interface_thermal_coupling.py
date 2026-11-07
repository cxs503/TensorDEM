"""Drive the ice-structure contact interface verification and write the record.

Run from the repository root::

    PYTHONPATH=src python3 benchmarks/interface_thermal_coupling.py

Outputs
-------
``artifacts/interface_thermal_coupling.json``
    Machine-readable record: every acceptance criterion with its measured
    number and verdict, the physical parameter tables, the per-case histories
    that the figures are drawn from, and an ``evidence`` field per block
    (REAL / MOCK / STUB / WITHHELD semantics of the platform audit).
``docs/interface-thermal/figures/*.png``
    Six figures: conduction profiles, Stefan front, Jaeger moving source,
    coupled histories, coupling convergence and the negative controls.
``runs/interface_thermal/summary.txt``
    One-screen digest of the numbers quoted in the report.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.interface_thermal import (                       # noqa: E402
    Conduction1D,
    CoupledInterface,
    EnthalpyStefan1D,
    IceSteelContact,
    MeltLubricationModel,
    blok_partition_coefficient,
    contact_heat_transfer_coefficient,
    jaeger_band_max_temperature,
    jaeger_band_surface_temperature,
    jaeger_high_speed_limit,
    jaeger_low_speed_limit,
    moving_band_fvm_solution,
    neumann_interface_position,
    neumann_lambda,
    neumann_stefan_number,
    neumann_temperature,
    semi_infinite_surface_temperature,
    semi_infinite_temperature_profile,
    thermal_diffusivity,
)

ARTIFACT = ROOT / "artifacts" / "interface_thermal_coupling.json"
FIGURE_DIR = ROOT / "docs" / "interface-thermal" / "figures"
RUN_DIR = ROOT / "runs" / "interface_thermal"

ICE = dict(k_w_m_k=2.2, rho_kg_m3=917.0, cp_j_kg_k=2100.0)
ALPHA_ICE = thermal_diffusivity(ICE["k_w_m_k"], ICE["rho_kg_m3"], ICE["cp_j_kg_k"])
WATER = dict(k_w_m_k=0.56, rho_kg_m3=1000.0, cp_j_kg_k=4200.0,
             cp_solid_j_kg_k=2100.0, latent_heat_j_kg=3.34e5)
ALPHA_WATER = thermal_diffusivity(WATER["k_w_m_k"], WATER["rho_kg_m3"],
                                  WATER["cp_j_kg_k"])

CRITERIA = {
    "a_conduction_profiles": "max |T_num - T_erfc| / (Ts - Ti) < 2 %",
    "a2_constant_flux_flash": "surface rise vs 2 q sqrt(alpha t/pi)/k < 5 %",
    "b_stefan_front": "melt front vs X = 2 lambda sqrt(alpha t) < 5 %",
    "c_friction_heat_transient": "flash temperature vs Carslaw-Jaeger < 5 %",
    "c2_jaeger_moving_source": "Jaeger integral vs independent FVM < 2 %",
    "d_coupling_convergence": "iterations <= 5 and residual < 1 %",
    "e_friction_monotonicity": "mu(T) monotone down, mu(Tmelt) = 0",
    "f_tests": ">= 15 tests including negative controls",
}


def celsius_pairs(history) -> dict[str, list]:
    """JSON-ready arrays, keeping the string-valued ``mode`` column as strings."""
    out: dict[str, list] = {}
    for key, value in history.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype.kind in "SU":
            out[key] = [str(item) for item in value]
        else:
            out[key] = np.asarray(value, dtype=float).tolist()
    return out


# ---------------------------------------------------------------------------
# (a) pure conduction
# ---------------------------------------------------------------------------
def conduction_case() -> dict:
    case = Conduction1D(length_m=0.06, cells=401, t_initial=-2.0, grading=1.0, **ICE)
    surface = 0.0
    history = case.run(60.0, 0.01, t_surface=surface,
                       record_times=[2.0, 10.0, 30.0, 60.0])
    rows = []
    for time, profile in zip(history["time"], history["profiles"]):
        exact = semi_infinite_temperature_profile(history["x"], time, ALPHA_ICE,
                                                  -2.0, surface)
        rows.append({
            "time_s": float(time),
            "max_abs_error_k": float(np.max(np.abs(profile - exact))),
            "relative_error": float(np.max(np.abs(profile - exact)) / 2.0),
        })
    flux = 2.0e4
    case.reset()
    flash = case.run(60.0, 0.01, q_surface=flux, record_times=[1.0, 10.0, 60.0])
    exact_surface = semi_infinite_surface_temperature(np.asarray(flash["time"]),
                                                      ALPHA_ICE, ICE["k_w_m_k"],
                                                      flux, -2.0)
    flash_rows = [{
        "time_s": float(t),
        "surface_numeric_c": float(num),
        "surface_exact_c": float(ex),
        "relative_error": float(abs(num - ex) / (ex + 2.0)),
    } for t, num, ex in zip(flash["time"], flash["surface"], exact_surface)]
    worst = max(row["relative_error"] for row in rows)
    worst_flash = max(row["relative_error"] for row in flash_rows)
    return {
        "evidence": "REAL",
        "criterion": "a",
        "geometry": "0.06 m slab, 401 nodes, surface held at 0 degC, initial -2 degC",
        "reference": "T = Ti + (Ts - Ti) erfc(x / (2 sqrt(alpha t)))",
        "alpha_m2_s": ALPHA_ICE,
        "profiles": rows,
        "flash_temperature": flash_rows,
        "worst_relative_error": worst,
        "worst_flash_relative_error": worst_flash,
        "pass": worst < 0.02 and worst_flash < 0.05,
        "history": celsius_pairs(history),
        "_figure": {"time": history["time"], "profiles": history["profiles"],
                    "x": history["x"]},
    }


# ---------------------------------------------------------------------------
# (b) Stefan / Neumann
# ---------------------------------------------------------------------------
def stefan_case() -> dict:
    superheat = 5.0
    ste = neumann_stefan_number(superheat, WATER["latent_heat_j_kg"],
                                WATER["cp_j_kg_k"])
    lam = neumann_lambda(ste)
    rows = []
    refinement = []
    for dT_melt in (0.02, 0.01, 0.005):
        solver = EnthalpyStefan1D(length_m=0.01, cells=1201, t_melt=0.0,
                                  t_initial=0.0, dT_melt=dT_melt, theta=1.0,
                                  **WATER)
        history = solver.run(10.0, 0.01, t_surface=superheat,
                             record_times=[1.0, 2.0, 5.0, 10.0])
        block = []
        for time, thickness in zip(history["time"], history["melt_thickness"]):
            exact = float(neumann_interface_position(time, lam, ALPHA_WATER))
            block.append({
                "time_s": float(time),
                "melt_thickness_m": float(thickness),
                "neumann_m": exact,
                "relative_error": float(abs(thickness - exact) / exact),
            })
        refinement.append({"dT_melt_k": dT_melt, "rows": block,
                           "worst_relative_error": max(r["relative_error"]
                                                       for r in block),
                           "history": celsius_pairs(history)})
        if dT_melt == 0.01:
            rows = block
    solver = EnthalpyStefan1D(length_m=0.01, cells=1201, t_melt=0.0, t_initial=0.0,
                              dT_melt=0.01, theta=1.0, **WATER)
    energy_rows = []
    flux_in = 0.0
    dt = 0.01
    initial = solver.capacity_scale * solver.enthalpy(solver.t)
    for step in range(1000):
        solver.step(dt, t_surface=superheat)
        flux_in += solver.conductance[0] * (solver.t[0] - solver.t[1]) * dt
        if (step + 1) % 100 == 0:
            stored = float(np.sum((solver.capacity_scale * solver.enthalpy(
                solver.t))[1:] - initial[1:]))
            energy_rows.append({
                "time_s": (step + 1) * dt,
                "flux_in_j_m2": flux_in,
                "latent_j_m2": float(solver.latent_energy()),
                "sensible_j_m2": float(solver.sensible_energy()),
                "stored_inner_j_m2": stored,
                "closure_error": (stored - flux_in) / flux_in,
            })
    worst = max(row["relative_error"] for row in rows)
    return {
        "evidence": "REAL",
        "criterion": "b",
        "stefan_number": ste,
        "neumann_lambda": lam,
        "lambda_small_ste_asymptote": math.sqrt(ste / 2.0),
        "superheat_k": superheat,
        "reference": "X(t) = 2 lambda sqrt(alpha t), lambda from lambda e^{lambda^2} erf(lambda) = Ste/sqrt(pi)",
        "rows": rows,
        "worst_relative_error": worst,
        "regularisation_refinement": [
            {"dT_melt_k": block["dT_melt_k"],
             "worst_relative_error": block["worst_relative_error"]}
            for block in refinement],
        "energy_ledger": energy_rows,
        "worst_energy_closure": max(abs(r["closure_error"]) for r in energy_rows),
        "pass": worst < 0.05 and max(abs(r["closure_error"]) for r in energy_rows) < 1e-3,
        "_figure": {"rows": rows, "lam": lam,
                    "history": refinement[1]["history"]},
    }


# ---------------------------------------------------------------------------
# (c) friction heat: transient flash + Jaeger moving source
# ---------------------------------------------------------------------------
def friction_case() -> dict:
    flux = 2.0e4
    conductor = Conduction1D(length_m=0.06, cells=401, t_initial=-5.0, **ICE)
    history = conductor.run(20.0, 0.01, q_surface=flux,
                            record_times=[0.5, 1.0, 5.0, 20.0])
    exact = semi_infinite_surface_temperature(np.asarray(history["time"]),
                                              ALPHA_ICE, ICE["k_w_m_k"], flux, -5.0)
    transient_rows = [{
        "time_s": float(t),
        "rise_numeric_k": float(num + 5.0),
        "rise_exact_k": float(ex + 5.0),
        "relative_error": float(abs((num + 5.0) - (ex + 5.0)) / (ex + 5.0)),
    } for t, num, ex in zip(history["time"], history["surface"], exact)]

    half_width = 5.0e-3
    band_flux = 1.0e6
    moving_rows = []
    for velocity in (0.005, 0.01, 0.02):
        quadrature = jaeger_band_max_temperature(half_width, velocity, ALPHA_ICE,
                                                ICE["k_w_m_k"], band_flux, samples=81)
        fvm = moving_band_fvm_solution(half_width, velocity, ALPHA_ICE,
                                       ICE["k_w_m_k"], band_flux, cells_x=1001,
                                       cells_z=100, grading_z=1.05)
        high = jaeger_high_speed_limit(half_width, velocity, ALPHA_ICE,
                                       ICE["k_w_m_k"], band_flux)
        moving_rows.append({
            "velocity_m_s": velocity,
            "peclet": velocity * half_width / (2.0 * ALPHA_ICE),
            "max_temperature_quadrature_k": float(quadrature["max_temperature_k"]),
            "max_temperature_fvm_k": float(fvm["max_temperature_k"]),
            "quadrature_vs_fvm_relative_error": float(
                abs(fvm["max_temperature_k"] - quadrature["max_temperature_k"])
                / quadrature["max_temperature_k"]),
            "high_speed_limit_k": float(high),
            "quadrature_vs_limit_relative_error": float(
                abs(quadrature["max_temperature_k"] - high) / high),
        })
    low_speed_rows = []
    for velocity in (1e-3, 5e-4, 2e-4, 1e-4):
        centre = float(jaeger_band_surface_temperature(0.0, half_width, velocity,
                                                       ALPHA_ICE, ICE["k_w_m_k"],
                                                       band_flux)[0])
        limit = jaeger_low_speed_limit(half_width, ALPHA_ICE, ICE["k_w_m_k"],
                                       band_flux, velocity)
        low_speed_rows.append({
            "velocity_m_s": velocity,
            "peclet": velocity * half_width / (2.0 * ALPHA_ICE),
            "centre_quadrature_k": centre,
            "log_limit_k": limit,
            "relative_error": abs(centre - limit) / limit,
        })
    partition = blok_partition_coefficient(
        ICE["k_w_m_k"], ICE["rho_kg_m3"], ICE["cp_j_kg_k"],
        45.0, 7850.0, 470.0)
    worst_transient = max(r["relative_error"] for r in transient_rows)
    worst_moving = max(r["quadrature_vs_fvm_relative_error"] for r in moving_rows)
    profile = jaeger_band_max_temperature(half_width, 0.01, ALPHA_ICE,
                                         ICE["k_w_m_k"], band_flux, samples=121)
    return {
        "evidence": "REAL",
        "criterion": "c",
        "flash_transient": {
            "flux_w_m2": flux,
            "reference": "Ts = Ti + 2 q sqrt(alpha t / pi) / k",
            "rows": transient_rows,
            "worst_relative_error": worst_transient,
        },
        "jaeger_moving_band": {
            "half_width_m": half_width,
            "flux_w_m2": band_flux,
            "reference": "T(xi) = (q a / pi k) int exp(-V(xi-u)) K0(V|xi-u|) du",
            "rows": moving_rows,
            "worst_relative_error_vs_fvm": worst_moving,
            "low_speed_log_limit": low_speed_rows,
        },
        "blok_partition": partition,
        "pass": worst_transient < 0.05 and worst_moving < 0.02,
        "_figure": {"rows": moving_rows, "profile": profile,
                    "half_width": half_width, "flux": band_flux,
                    "low_speed": low_speed_rows},
    }


# ---------------------------------------------------------------------------
# (d)(e) coupled contact
# ---------------------------------------------------------------------------
BASE = dict(normal_force_n=500.0, mu_cold=0.08, nominal_contact_area_m2=1.0e-4,
            initial_temperature_c=-10.0, cells=201, grading=1.05)
WET = MeltLubricationModel(mu_base=0.08, mu_wet=0.03)


def coupled_case(tag: str, kb: dict) -> dict:
    case = IceSteelContact(**BASE, **kb)
    interface = CoupledInterface(case)
    result = interface.run(4.0, 0.005)
    history = celsius_pairs(result)
    profile = result["temperature_profile_c"]
    return {
        "tag": tag,
        "case": {
            "normal_force_n": case.normal_force_n,
            "slip_velocity_m_s": case.slip_velocity_m_s,
            "nominal_contact_area_m2": case.contact_area_m2,
            "mean_pressure_pa": case.mean_pressure_pa,
            "contact_radius_m": case.contact_radius_m,
            "hertz_stiffness_n_m": case.contact_stiffness_n_m,
            "initial_temperature_c": case.initial_temperature_c,
            "mu_cold": case.mu_cold,
            "mu_wet": getattr(case.lubrication, "mu_wet", 0.0),
            "t_onset_c": case.lubrication.t_onset,
            "layer_thickness_m": case.layer_thickness_m,
            "cells": case.cells,
            "partition_ice": case.partition()["partition_1"],
            "partition_steel": case.partition()["partition_2"],
        },
        "max_iterations": result["max_iterations_used"],
        "max_residual": result["max_residual"],
        "all_converged": bool(result["all_converged"]),
        "final": {
            "interface_temperature_c": float(result["interface_temperature_c"][-1]),
            "mu": float(result["mu"][-1]),
            "friction_force_n": float(result["friction_force_n"][-1]),
            "film_thickness_m": float(result["film_thickness_m"][-1]),
            "normal_stiffness_n_m": float(result["normal_stiffness_n_m"][-1]),
            "load_bearing_fraction": float(result["load_bearing_fraction"][-1]),
        },
        "initial_friction_force_n": float(result["friction_force_n"][0]),
        "energy_ledger": result["energy_ledger"],
        "mechanical_work_j": result["mechanical_work_j"],
        "modes": sorted(set(str(mode) for mode in result["mode"])),
        "history": history,
        "_figure": {"history": history, "profile": profile},
    }


def coupling_case() -> dict:
    self_limiting = coupled_case("self_limiting (mu_wet = 0)",
                                 dict(slip_velocity_m_s=0.5))
    wet = coupled_case("boundary_lubrication (mu_wet = 0.03)",
                       dict(slip_velocity_m_s=1.0, lubrication=WET))
    zero_slip = CoupledInterface(IceSteelContact(**BASE, slip_velocity_m_s=0.0))
    zero = zero_slip.run(1.0, 0.005)
    coulomb = CoupledInterface(IceSteelContact(**BASE, slip_velocity_m_s=0.5))
    early = coulomb.run(0.02, 0.005)
    cases = [self_limiting, wet]
    worst_iterations = max(c["max_iterations"] for c in cases)
    worst_residual = max(c["max_residual"] for c in cases)
    worst_closure = max(abs(c["energy_ledger"]["closure_error"]) for c in cases)

    model = MeltLubricationModel(mu_base=0.08, t_melt=0.0, t_onset=-3.0, exponent=2.0)
    temperatures = np.linspace(-10.0, 0.0, 201)
    mu_curve = [model.mu(t) for t in temperatures]
    return {
        "evidence": "REAL",
        "criterion": "d",
        "cases": cases,
        "max_iterations": worst_iterations,
        "max_residual": worst_residual,
        "worst_energy_closure": worst_closure,
        "pass": worst_iterations <= 5 and worst_residual < 1e-2
                and worst_closure < 1e-4,
        "negative_controls": {
            "zero_slip": {
                "interface_temperature_c": float(zero["interface_temperature_c"][-1]),
                "max_heat_flux_w_m2": float(np.max(zero["heat_flux_w_m2"])),
                "film_thickness_m": float(zero["film_thickness_m"][-1]),
                "mu": float(zero["mu"][-1]),
                "pass": bool(np.allclose(zero["interface_temperature_c"], -10.0,
                                         atol=1e-9)),
            },
            "no_phase_change_coulomb": {
                "interface_temperature_c": float(early["interface_temperature_c"][-1]),
                "mu": float(early["mu"][-1]),
                "friction_force_n": float(early["friction_force_n"][-1]),
                "expected_friction_force_n": 0.08 * 500.0,
                "pass": bool(abs(float(early["mu"][-1]) - 0.08) < 1e-12),
            },
        },
        "friction_law": {
            "t_melt_c": model.t_melt,
            "t_onset_c": model.t_onset,
            "exponent": model.exponent,
            "temperatures_c": temperatures.tolist(),
            "mu": mu_curve,
            "monotone": bool(np.all(np.diff(mu_curve) <= 1e-15)),
            "mu_at_melt": model.mu(0.0),
            "pass": bool(np.all(np.diff(mu_curve) <= 1e-15) and model.mu(0.0) == 0.0),
        },
    }


def contact_conductance_block() -> dict:
    rows = []
    for pressure in (0.5e6, 1.0e6, 2.0e6, 5.0e6):
        h_c = contact_heat_transfer_coefficient(2.2, 45.0, 5e-5, 0.2, pressure, 1e7)
        rows.append({
            "pressure_pa": pressure,
            "h_c_w_m2_k": h_c,
            "conductance_w_m2_k": h_c,
            "resistance_m2_k_w": 1.0 / h_c,
        })
    return {
        "evidence": "REAL",
        "model": "Cooper-Mikic-Yovanovich, h_c = 1.25 k_h (m/sigma) (P/H)^0.95",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def write_figures(conduction: dict, stefan: dict, friction: dict,
                  coupling: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = []

    def save(name: str, caption: str, fig) -> None:
        path = FIGURE_DIR / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": name, "caption": caption, "path": str(path)})

    # 1 conduction profiles
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = conduction["_figure"]
    for time, profile in zip(data["time"], data["profiles"]):
        exact = semi_infinite_temperature_profile(data["x"], time, ALPHA_ICE, -2.0, 0.0)
        axes[0].plot(np.asarray(data["x"]) * 1e3, profile, "-", label=f"FVM {time:.0f} s")
        axes[0].plot(np.asarray(data["x"]) * 1e3, exact, "k--", lw=0.9)
    axes[0].set_xlim(0, 12)
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("T [degC]")
    axes[0].set_title("(a) step-heated ice: FVM vs erfc")
    axes[0].legend(fontsize=7)
    flashes = conduction["flash_temperature"]
    times = np.linspace(0.5, 60.0, 100)
    axes[1].plot(times, semi_infinite_surface_temperature(times, ALPHA_ICE,
                                                         ICE["k_w_m_k"], 2e4, -2.0) + 2.0,
                 "k--", label="exact 2q sqrt(alpha t/pi)/k")
    axes[1].plot([row["time_s"] for row in flashes],
                 [row["surface_numeric_c"] + 2.0 for row in flashes], "o",
                 label="FVM")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("surface rise [K]")
    axes[1].set_title("(a2) constant-flux flash temperature")
    axes[1].legend(fontsize=8)
    save("fig1_conduction.png",
         "FVM temperature fields against the exact semi-infinite solutions.", fig)

    # 2 Stefan front
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rows = stefan["rows"]
    axes[0].plot([r["time_s"] for r in rows], [r["melt_thickness_m"] * 1e3 for r in rows],
                 "o", label="enthalpy solver")
    axes[0].plot([r["time_s"] for r in rows], [r["neumann_m"] * 1e3 for r in rows],
                 "k--", label="Neumann $2\\lambda\\sqrt{\\alpha t}$")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("melt thickness [mm]")
    axes[0].set_title("(b) Stefan front vs similarity solution")
    axes[0].legend(fontsize=8)
    refinement = stefan["regularisation_refinement"]
    axes[1].semilogy([r["dT_melt_k"] for r in refinement],
                     [r["worst_relative_error"] * 100 for r in refinement], "o-")
    axes[1].set_xlabel("melting-band width dT_melt [K]")
    axes[1].set_ylabel("worst front error [%]")
    axes[1].set_title("(b2) regularisation refinement")
    save("fig2_stefan.png",
         "Melt-front position against Neumann's solution and its convergence "
         "with the melting-band regularisation.", fig)

    # 3 Jaeger
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = friction["_figure"]
    a = data["half_width"]
    profile = data["profile"]
    axes[0].plot(profile["profile_x"] * 1e3, profile["profile_t"], "-",
                 label="Jaeger integral")
    fvm = moving_band_fvm_solution(a, 0.01, ALPHA_ICE, ICE["k_w_m_k"], data["flux"],
                                   cells_x=1001, cells_z=100, grading_z=1.05)
    axes[0].plot(fvm["x"] * 1e3, fvm["surface_temperature"], "--",
                 label="2D FVM (Rosenthal)")
    axes[0].set_xlim(-3 * a * 1e3, 3 * a * 1e3)
    axes[0].set_xlabel("xi [mm] (band frame)")
    axes[0].set_ylabel("surface rise [K]")
    axes[0].set_title("(c) moving band source, V = %.1f" % (0.01 * a / (2 * ALPHA_ICE)))
    axes[0].legend(fontsize=8)
    rows = data["rows"]
    velocities = np.linspace(0.002, 0.03, 40)
    axes[1].loglog(velocities, [jaeger_band_max_temperature(a, v, ALPHA_ICE,
                                                            ICE["k_w_m_k"], data["flux"],
                                                            samples=41)["max_temperature_k"]
                                for v in velocities], "-", label="Jaeger integral")
    axes[1].loglog(velocities, [jaeger_high_speed_limit(a, v, ALPHA_ICE,
                                                        ICE["k_w_m_k"], data["flux"])
                                for v in velocities], "k--",
                   label="(2q/k) sqrt(alpha 2a/v / pi)")
    axes[1].loglog([r["velocity_m_s"] for r in rows],
                   [r["max_temperature_fvm_k"] for r in rows], "o", label="2D FVM")
    axes[1].set_xlabel("sliding speed [m/s]")
    axes[1].set_ylabel("peak rise [K]")
    axes[1].set_title("(c2) peak temperature vs speed")
    axes[1].legend(fontsize=8)
    save("fig3_jaeger.png",
         "Jaeger moving band source: Bessel integral, independent finite-volume "
         "solution and the high-Peclet transit asymptote.", fig)

    # 4 coupled histories
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for case in coupling["cases"]:
        history = case["_figure"]["history"]
        time = np.asarray(history["time"])
        style = "-" if "self" in case["tag"] else "--"
        label = case["tag"].split(" ")[0]
        axes[0, 0].plot(time, np.asarray(history["interface_temperature_c"]), style,
                        label=label)
        axes[0, 1].plot(time, np.asarray(history["mu"]), style, label=label)
        axes[1, 0].plot(time, np.asarray(history["friction_force_n"]), style, label=label)
        axes[1, 1].plot(time, np.asarray(history["film_thickness_m"]) * 1e6, style,
                        label=label)
    axes[0, 0].set_ylabel("interface T [degC]")
    axes[0, 1].set_ylabel("friction coefficient [-]")
    axes[1, 0].set_ylabel("friction force [N]")
    axes[1, 1].set_ylabel("melt film [micrometre]")
    for ax in axes.ravel():
        ax.set_xlabel("t [s]")
        ax.legend(fontsize=8)
    axes[0, 0].set_title("(d) interface temperature (capped at the melting point)")
    axes[0, 1].set_title("(e) mu(T) from the melt fraction")
    axes[1, 0].set_title("(d) friction force feedback")
    axes[1, 1].set_title("(b) melt film grown by the Stefan condition")
    save("fig4_coupled_history.png",
         "Two-way coupled histories for the self-limiting (dry, mu_wet = 0) and "
         "boundary-lubricated (mu_wet = 0.03) contacts.", fig)

    # 5 convergence
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for case in coupling["cases"]:
        history = case["_figure"]["history"]
        time = np.asarray(history["time"])
        style = "-" if "self" in case["tag"] else "--"
        axes[0].plot(time, np.asarray(history["iterations"]), style,
                     label=case["tag"].split(" ")[0])
        axes[1].semilogy(time, np.maximum(np.asarray(history["residual"]), 1e-16),
                         style, label=case["tag"].split(" ")[0])
    axes[0].axhline(5, color="r", ls=":", label="5-iteration gate")
    axes[1].axhline(1e-2, color="r", ls=":", label="1 % gate")
    axes[0].set_ylabel("Newton iterations per step")
    axes[1].set_ylabel("relative residual")
    for ax in axes:
        ax.set_xlabel("t [s]")
        ax.legend(fontsize=8)
    axes[0].set_title("(d) iterations of the DEM-FVM fixed point")
    axes[1].set_title("(d) residual history")
    save("fig5_convergence.png",
         "Coupling iterations and residuals per time step against the "
         "5-step / 1 % acceptance gate.", fig)

    # 6 negative controls and friction law
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    law = coupling["friction_law"]
    axes[0].plot(law["temperatures_c"], law["mu"], "-")
    axes[0].axvline(law["t_onset_c"], color="grey", ls=":")
    axes[0].axvline(law["t_melt_c"], color="r", ls=":")
    axes[0].set_xlabel("interface temperature [degC]")
    axes[0].set_ylabel("friction coefficient [-]")
    axes[0].set_title("(e) mu(T): monotone down to zero at melting")
    controls = coupling["negative_controls"]
    labels = ["zero slip", "no phase change"]
    values = [controls["zero_slip"]["interface_temperature_c"],
              controls["no_phase_change_coulomb"]["interface_temperature_c"]]
    mus = [controls["zero_slip"]["mu"], controls["no_phase_change_coulomb"]["mu"]]
    twin = axes[1].twinx()
    axes[1].bar(np.arange(2) - 0.18, values, width=0.35, label="interface T [degC]")
    twin.bar(np.arange(2) + 0.18, mus, width=0.35, color="tab:orange",
             label="mu [-]")
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("interface temperature [degC]")
    twin.set_ylabel("friction coefficient [-]")
    axes[1].set_title("(f) negative controls")
    save("fig6_controls.png",
         "Friction law and the two negative controls (no slip -> no heating; "
         "no phase change -> pure Coulomb).", fig)
    return figures


# ---------------------------------------------------------------------------
def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    conduction = conduction_case()
    stefan = stefan_case()
    friction = friction_case()
    coupling = coupling_case()
    figures = write_figures(conduction, stefan, friction, coupling)

    record = {
        "mission": "IN32-N04",
        "title": "Ice-structure contact interface: Stefan condition and "
                 "friction-heat melting in a two-way DEM-FVM coupling",
        "repository": "TensorDEM",
        "module": "src/tensordem/interface_thermal",
        "criteria": CRITERIA,
        "conduction": {k: v for k, v in conduction.items() if not k.startswith("_")},
        "stefan": {k: v for k, v in stefan.items() if not k.startswith("_")},
        "friction": {k: v for k, v in friction.items() if not k.startswith("_")},
        "coupling": {k: v for k, v in coupling.items() if not k.startswith("_")},
        "contact_conductance": contact_conductance_block(),
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figures],
        "verdicts": {
            "a_conduction": conduction["pass"],
            "b_stefan": stefan["pass"],
            "c_friction_heat": friction["pass"],
            "d_coupling_convergence": coupling["pass"],
            "e_friction_monotonicity": coupling["friction_law"]["pass"],
            "f_negative_controls": (coupling["negative_controls"]["zero_slip"]["pass"]
                                    and coupling["negative_controls"]
                                    ["no_phase_change_coulomb"]["pass"]),
        },
        "limitations": [
            "The film drainage (squeeze flow) is not modelled: the melt layer "
            "only grows by the Stefan condition, so its thickness is an upper "
            "bound once melting starts.",
            "Coupled cases assume the nominal contact area and the mean pressure "
            "are fixed (force-controlled); the displacement-controlled branch is "
            "reported as normal_force_at_approach and used only for the audit of "
            "the stiffness feedback.",
            "The ice side is a 1D FVM column: lateral conduction, surface "
            "convection to water and radiation are not represented.",
            "mu(T) collapses to zero at the melting point by construction "
            "(criterion e). Sustained melting therefore requires a residual "
            "boundary-lubrication friction, which is carried as mu_wet and "
            "reported as the second coupled case.",
        ],
    }
    def _coerce(value):
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"not JSON serialisable: {type(value)}")

    ARTIFACT.write_text(json.dumps(record, indent=2, default=_coerce))

    lines = [
        "IN32-N04 interface thermal coupling -- summary",
        "=" * 52,
        f"(a) conduction profile worst relative error : {conduction['worst_relative_error']*100:.3f} %  (gate 2 %)",
        f"(a2) constant-flux flash worst error       : {conduction['worst_flash_relative_error']*100:.3f} %  (gate 5 %)",
        f"(b) Stefan front worst relative error      : {stefan['worst_relative_error']*100:.3f} %  (gate 5 %)",
        f"(b) Stefan energy ledger worst closure     : {stefan['worst_energy_closure']:.3e}",
        f"(c) frictional flash worst error           : {friction['flash_transient']['worst_relative_error']*100:.3f} %  (gate 5 %)",
        f"(c) Jaeger integral vs 2D FVM worst error  : {friction['jaeger_moving_band']['worst_relative_error_vs_fvm']*100:.3f} %  (gate 2 %)",
        f"(d) coupling iterations (max)              : {coupling['max_iterations']}  (gate 5)",
        f"(d) coupling residual (max)                : {coupling['max_residual']:.3e}  (gate 1e-2)",
        f"(d) ice-side energy closure (worst)        : {coupling['worst_energy_closure']:.3e}",
        "(e) mu(T) monotone, mu(Tmelt) = 0          : "
        f"{coupling['friction_law']['pass']}",
        "",
        "coupled cases",
    ]
    for case in coupling["cases"]:
        final = case["final"]
        lines += [
            f"  {case['tag']}",
            f"    interface T = {final['interface_temperature_c']:.3f} degC, "
            f"mu = {final['mu']:.4f}, F_t = {final['friction_force_n']:.2f} N "
            f"(from {case['initial_friction_force_n']:.2f} N)",
            f"    film = {final['film_thickness_m']*1e6:.2f} um, "
            f"k_eff = {final['normal_stiffness_n_m']:.3e} N/m, "
            f"iterations <= {case['max_iterations']}",
        ]
    RUN_DIR.joinpath("summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nartifact: {ARTIFACT}")
    print(f"figures : {len(figures)} in {FIGURE_DIR}")
    ok = all(record["verdicts"].values())
    print("all criteria pass:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())