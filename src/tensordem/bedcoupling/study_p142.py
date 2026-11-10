"""P142W3-N03 acceptance study: Ergun closure, RZ scaling, three-force ledger, U_mf.

The five machine-checkable criteria of the node (thresholds pre-registered in
``prereg/P142W3-N03_fluidization_prereg.md``):

===  =====================================================================
1    fixed-bed ``dp/dz`` vs Ergun (1952), closure on local voidage,
     ``Re_s in [1, 1000]`` (13-point log sweep)                       <= 10 %
2a   single-particle settling vs Stokes terminal velocity            <= 5 %
2b   single-particle settling vs Schiller--Naumann terminal velocity <= 10 %
2c   hindered settling curve ``u_p(eps)`` vs ``u_t eps^n_RZ``
     at eps = 0.80/0.86/0.92/0.96, every point                       <= 10 %
3a   three-force ledger vs ``d(mv)/dt``, every Verlet step           <= 1 %
3b   three-force equilibrium of the steady window                    <= 1 %
4    onset of fluidisation ``U_mf`` vs Wen--Yu (1966)                <= 10 %
5    three resolutions (N = 600/1200/2400) agree on ``dp/dz``,
     ``U_mf`` and the settling speed                                 <= 5 %
===  =====================================================================

Negative controls are computed in the same pass and written next to the
criteria (never instead of them):

* N1 hydrostatic instead of mixture buoyancy in the RZ sweep -- expected to
  *fail* the RZ comparison, which is what makes the comparison discriminating;
* N2 drag dropped from the applied force while the ledger still records it --
  the ledger residual must blow up to O(1) or criterion 3 has no teeth;
* N3 Wen--Yu-only drag branch at ``eps = 0.38`` against Ergun;
* N4 timestep halving -- the dt-dependent discretisation residual must shrink.

Every number written here comes from a run of this function; nothing is
hard-coded except the closure and correlation constants of the published laws.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bed import FluidProperties, bed_pressure_drop
from .closures import (
    ergun_pressure_gradient,
    rz_exponent,
    terminal_velocity_sn,
    terminal_velocity_stokes,
)
from .fluidization import (
    column_dynamics,
    ergun_minimum_fluidization,
    resolution_convergence,
    simulated_minimum_fluidization,
    wen_yu_minimum_fluidization,
)
from .packing import (
    PackedBed,
    local_porosity,
    make_jittered_cubic_array,
    make_random_close_packing,
    porosity_statistics,
)

__all__ = [
    "run_acceptance_study",
    "write_results",
    "PACKED_FLUID",
    "SETTLING_FLUID",
    "DIAMETER",
    "PROBE_RADIUS_OVER_D",
    "RZ_POROSITIES",
    "REYNOLDS_SWEEP",
]

DIAMETER = 1.0e-3  # 1 mm monodisperse bed
PROBE_RADIUS_OVER_D = 1.5  # local-voidage kernel radius in particle diameters
STOKES_DIAMETER = 0.2e-3  # small particle -> Re_t << 1
PACKED_FLUID = FluidProperties(rho_fluid=998.0, rho_particle=2650.0, mu=1.0e-3)
SETTLING_FLUID = FluidProperties(rho_fluid=1200.0, rho_particle=2650.0, mu=0.05)
RZ_POROSITIES = (0.80, 0.86, 0.92, 0.96)
REYNOLDS_SWEEP = tuple(np.logspace(0.0, 3.0, 13))  # Re_s = 1 ... 1000


def _criterion(name: str, value: float, threshold: float, passed: bool, **extra) -> dict:
    out = {
        "name": name,
        "value": float(value),
        "threshold": float(threshold),
        "passed": bool(passed),
        "criterion": extra.pop("criterion", ""),
    }
    out.update(extra)
    return out


def _single_particle(diameter: float) -> PackedBed:
    """One particle in a cell 20 diameters wide: eps = 1 - (pi/6)/8000."""
    half = 10.0 * diameter
    return PackedBed(
        diameter=diameter,
        box_length=20.0 * diameter,
        positions=np.array([[half, half, half]]),
        kind="single-particle",
    )


# ---------------------------------------------------------------------------
# criterion 1: Ergun pressure drop over Re_s in [1, 1000]
# ---------------------------------------------------------------------------
def criterion_ergun(bed: PackedBed, eps_local: np.ndarray) -> dict:
    fluid = PACKED_FLUID
    d = bed.diameter
    rows = []
    for re_target in REYNOLDS_SWEEP:
        u_sup = re_target * fluid.mu / (fluid.rho_fluid * d)
        ref = ergun_pressure_gradient(
            u_sup,
            diameter=d,
            porosity=bed.porosity,
            rho_fluid=fluid.rho_fluid,
            mu=fluid.mu,
        )
        measured = float(bed_pressure_drop(bed, u_sup, fluid, eps_local=eps_local))
        measured_mean = float(bed_pressure_drop(bed, u_sup, fluid))
        wen_yu_branch = float(bed_pressure_drop(bed, u_sup, fluid, eps_local=eps_local, branch="wen_yu"))
        rows.append(
            {
                "re_superficial": fluid.rho_fluid * u_sup * d / fluid.mu,
                "u_superficial": u_sup,
                "ergun_dpdz": ref.total,
                "ergun_viscous": ref.viscous,
                "ergun_inertial": ref.inertial,
                "inertial_fraction": ref.inertial_fraction,
                "measured_local_voidage_dpdz": measured,
                "measured_mean_voidage_dpdz": measured_mean,
                "wen_yu_branch_dpdz": wen_yu_branch,
                "rel_error_local": measured / ref.total - 1.0,
                "rel_error_mean": measured_mean / ref.total - 1.0,
                "rel_error_wen_yu_branch": wen_yu_branch / ref.total - 1.0,
            }
        )
    worst = max(abs(r["rel_error_local"]) for r in rows)
    worst_wen_yu = max(abs(r["rel_error_wen_yu_branch"]) for r in rows)
    return {
        "fluid": asdict(fluid),
        "diameter": d,
        "porosity": bed.porosity,
        "n_particles": bed.n_particles,
        "probe_radius_over_d": PROBE_RADIUS_OVER_D,
        "rows": rows,
        "worst_abs_error_local_voidage": worst,
        "worst_abs_error_mean_voidage": max(abs(r["rel_error_mean"]) for r in rows),
        "worst_abs_error_wen_yu_branch": worst_wen_yu,
        "criterion": _criterion(
            "ergun_relative_error",
            worst,
            0.10,
            worst <= 0.10,
            criterion="max_U |dp/dz_closure(local voidage) - dp/dz_Ergun| / dp/dz_Ergun <= 10 % for Re_s in [1,1000]",
            re_range=[float(REYNOLDS_SWEEP[0]), float(REYNOLDS_SWEEP[-1])],
            negative_control_wen_yu_branch=worst_wen_yu,
        ),
    }


# ---------------------------------------------------------------------------
# criterion 2: terminal velocity (Stokes / Schiller-Naumann) and RZ curve
# ---------------------------------------------------------------------------
def criterion_terminal_velocity(*, n_steps: int = 3000) -> dict:
    # --- Stokes regime
    stokes_fluid = SETTLING_FLUID
    u_stokes = terminal_velocity_stokes(
        STOKES_DIAMETER,
        rho_particle=stokes_fluid.rho_particle,
        rho_fluid=stokes_fluid.rho_fluid,
        mu=stokes_fluid.mu,
    )
    stokes_run = column_dynamics(
        _single_particle(STOKES_DIAMETER), stokes_fluid, n_steps=n_steps, fluid_mode="imposed"
    )
    u_sim_stokes = -stokes_run["steady_mean_vz"]
    rel_stokes = u_sim_stokes / u_stokes - 1.0

    # --- transition regime
    sn = terminal_velocity_sn(
        DIAMETER,
        rho_particle=PACKED_FLUID.rho_particle,
        rho_fluid=PACKED_FLUID.rho_fluid,
        mu=PACKED_FLUID.mu,
    )
    transition_run = column_dynamics(
        _single_particle(DIAMETER), PACKED_FLUID, n_steps=n_steps, fluid_mode="imposed"
    )
    u_sim_transition = -transition_run["steady_mean_vz"]
    rel_transition = u_sim_transition / sn["u_terminal"] - 1.0

    return {
        "stokes": {
            "diameter": STOKES_DIAMETER,
            "fluid": asdict(stokes_fluid),
            "u_stokes": u_stokes,
            "u_simulated": u_sim_stokes,
            "rel_error": rel_stokes,
            "re_terminal": stokes_fluid.rho_fluid * u_stokes * STOKES_DIAMETER / stokes_fluid.mu,
            "dt_over_tau": stokes_run["dt_over_tau"],
            "criterion": _criterion(
                "stokes_terminal_velocity",
                abs(rel_stokes),
                0.05,
                abs(rel_stokes) <= 0.05,
                criterion="|u_sim - u_Stokes| / u_Stokes <= 5 % (Re_t < 0.01)",
            ),
        },
        "transition": {
            "diameter": DIAMETER,
            "fluid": asdict(PACKED_FLUID),
            "u_schiller_naumann": sn["u_terminal"],
            "u_simulated": u_sim_transition,
            "rel_error": rel_transition,
            "re_terminal": sn["re_terminal"],
            "dt_over_tau": transition_run["dt_over_tau"],
            "criterion": _criterion(
                "schiller_naumann_terminal_velocity",
                abs(rel_transition),
                0.10,
                abs(rel_transition) <= 0.10,
                criterion="|u_sim - u_SchillerNaumann| / u_SN <= 10 % (transition regime)",
            ),
        },
    }


def criterion_rz_curve(*, n_steps: int = 1500, n_side: int = 8) -> dict:
    """Hindered settling curve against ``u_t eps^n`` at four porosities."""
    fluid = SETTLING_FLUID
    ref = terminal_velocity_sn(
        DIAMETER, rho_particle=fluid.rho_particle, rho_fluid=fluid.rho_fluid, mu=fluid.mu
    )
    n_rz = float(rz_exponent(ref["re_terminal"]))
    rows = []
    for model in ("mixture", "hydrostatic"):
        for eps in RZ_POROSITIES:
            cell = make_jittered_cubic_array(n_side, DIAMETER, porosity=eps, seed=7)
            run = column_dynamics(
                cell,
                fluid,
                n_steps=n_steps,
                fluid_mode="closed_box",
                buoyancy_model=model,
            )
            u_p = -run["steady_mean_vz"]
            predicted = ref["u_terminal"] * eps**n_rz
            rows.append(
                {
                    "buoyancy_model": model,
                    "porosity": eps,
                    "n_particles": cell.n_particles,
                    "u_particle_simulated": u_p,
                    "u_particle_std": run["steady_std_vz"],
                    "u_rz_predicted": predicted,
                    "rel_error": u_p / predicted - 1.0,
                    "dt_over_tau": run["dt_over_tau"],
                    "ledger_equilibrium_residual": run["equilibrium_residual"],
                }
            )
    mixture = [r for r in rows if r["buoyancy_model"] == "mixture"]
    hydrostatic = [r for r in rows if r["buoyancy_model"] == "hydrostatic"]
    worst_mixture = max(abs(r["rel_error"]) for r in mixture)
    eps_arr = np.array([r["porosity"] for r in mixture])
    u_arr = np.array([r["u_particle_simulated"] for r in mixture])
    fit_exponent = float(np.polyfit(np.log(eps_arr), np.log(u_arr), 1)[0])
    with np.errstate(invalid="ignore"):
        hydro_worst = max(abs(r["rel_error"]) for r in hydrostatic)
    return {
        "u_terminal": ref["u_terminal"],
        "re_terminal": ref["re_terminal"],
        "n_rz": n_rz,
        "fit_exponent_mixture": fit_exponent,
        "fit_exponent_deviation": fit_exponent - n_rz,
        "rows": rows,
        "worst_abs_error_mixture": worst_mixture,
        "worst_abs_error_hydrostatic": hydro_worst,
        "criterion": _criterion(
            "rz_curve_relative_error",
            worst_mixture,
            0.10,
            worst_mixture <= 0.10,
            criterion="max_eps |u_p(eps) - u_t eps^n_RZ| / (u_t eps^n_RZ) <= 10 % (mixture buoyancy)",
            negative_control_hydrostatic=hydro_worst,
        ),
    }


# ---------------------------------------------------------------------------
# criterion 3: three-force ledger
# ---------------------------------------------------------------------------
def criterion_ledger(*, n_steps: int = 1500, n_side: int = 8, porosity: float = 0.80) -> dict:
    cell = make_jittered_cubic_array(n_side, DIAMETER, porosity=porosity, seed=7)
    primary = column_dynamics(cell, SETTLING_FLUID, n_steps=n_steps, fluid_mode="closed_box")
    dropped = column_dynamics(
        cell,
        SETTLING_FLUID,
        n_steps=n_steps,
        fluid_mode="closed_box",
        applied_terms=("pressure", "gravity"),
        seed=11,
    )
    coarse = column_dynamics(cell, SETTLING_FLUID, n_steps=n_steps // 2, fluid_mode="closed_box")
    fine = column_dynamics(
        cell, SETTLING_FLUID, n_steps=n_steps, dt=0.5 * primary["dt"], fluid_mode="closed_box"
    )
    book = primary["bookkeeping_residual"]
    equilibrium = primary["equilibrium_residual"]
    passed = book <= 0.01 and equilibrium <= 0.01
    return {
        "case": {
            "n_particles": cell.n_particles,
            "porosity": cell.porosity,
            "diameter": DIAMETER,
            "fluid": asdict(SETTLING_FLUID),
            "fluid_mode": "closed_box",
            "buoyancy_model": "mixture",
            "branch": "gidaspow",
            "n_steps": n_steps,
            "dt": primary["dt"],
            "dt_over_tau": primary["dt_over_tau"],
        },
        "bookkeeping_residual": book,
        "equilibrium_residual": equilibrium,
        "equilibrium_fluctuation": primary["equilibrium_residual_fluctuation"],
        "discretisation_residual": primary["discretisation_residual"],
        "steady_forces_N": {
            "drag": primary["steady_mean_drag_N"],
            "pressure": primary["steady_mean_pressure_N"],
            "gravity": primary["steady_mean_gravity_N"],
            "total": primary["steady_mean_ledger_N"],
        },
        "weight_total_N": primary["weight_total_N"],
        "finite": primary["finite"],
        "negative_controls": {
            "dropped_drag_bookkeeping": dropped["bookkeeping_residual"],
            "dropped_drag_equilibrium": dropped["equilibrium_residual"],
            "timestep_halving": {
                "dt_coarse": coarse["dt"],
                "discretisation_coarse": coarse["discretisation_residual"],
                "dt_fine": fine["dt"],
                "discretisation_fine": fine["discretisation_residual"],
                "ratio": fine["discretisation_residual"] / max(coarse["discretisation_residual"], 1e-300),
            },
        },
        "series": {
            "time_over_tau": (primary["time"] / primary["tau"]).tolist(),
            "f_drag": primary["f_drag"].tolist(),
            "f_pressure": primary["f_pressure"].tolist(),
            "f_gravity": primary["f_gravity"].tolist(),
            "f_total": (primary["f_drag"] + primary["f_pressure"] + primary["f_gravity"]).tolist(),
            "momentum_rate": (np.gradient(primary["momentum"], primary["dt"])).tolist(),
            "mean_vz": primary["mean_vz"].tolist(),
            "weight_total": primary["weight_total_N"],
        },
        "criterion": _criterion(
            "three_force_ledger",
            max(book, equilibrium),
            0.01,
            passed,
            criterion="|sum F - d(mv)/dt| / sum(m g) <= 1 % per step and in the steady window",
            bookkeeping=book,
            equilibrium=equilibrium,
            negative_control_value=dropped["bookkeeping_residual"],
        ),
    }


# ---------------------------------------------------------------------------
# criterion 4: onset of fluidisation vs Wen-Yu
# ---------------------------------------------------------------------------
def criterion_fluidization(bed: PackedBed, eps_local: np.ndarray) -> dict:
    fluid = PACKED_FLUID
    d = bed.diameter
    wen_yu = wen_yu_minimum_fluidization(
        d, rho_particle=fluid.rho_particle, rho_fluid=fluid.rho_fluid, mu=fluid.mu
    )
    ergun = ergun_minimum_fluidization(
        bed.porosity,
        diameter=d,
        rho_particle=fluid.rho_particle,
        rho_fluid=fluid.rho_fluid,
        mu=fluid.mu,
    )
    sim_local = simulated_minimum_fluidization(
        bed, fluid, eps_local=eps_local, description="closure on local voidage"
    )
    sim_mean = simulated_minimum_fluidization(bed, fluid, description="closure on mean voidage")
    rel = sim_local["u_mf"] / wen_yu["u_mf"] - 1.0
    return {
        "fluid": asdict(fluid),
        "diameter": d,
        "bed": {
            "n_particles": bed.n_particles,
            "porosity": bed.porosity,
            "local_porosity_mean": float(eps_local.mean()),
            "local_porosity_std": float(eps_local.std()),
        },
        "wen_yu": wen_yu,
        "ergun": ergun,
        "simulated_local": sim_local,
        "simulated_mean": sim_mean,
        "rel_error_vs_wen_yu": rel,
        "rel_error_ergun_vs_wen_yu": ergun["u_mf"] / wen_yu["u_mf"] - 1.0,
        "consistency_max_vs_ergun": abs(sim_mean["u_mf"] / ergun["u_mf"] - 1.0),
        "criterion": _criterion(
            "minimum_fluidization_velocity",
            abs(rel),
            0.10,
            abs(rel) <= 0.10,
            criterion="|U_mf,closure - U_mf,WenYu| / U_mf,WenYu <= 10 %",
            u_mf_wen_yu=wen_yu["u_mf"],
            u_mf_closure=sim_local["u_mf"],
            re_mf=sim_local["re_mf"],
        ),
    }


# ---------------------------------------------------------------------------
# the full study
# ---------------------------------------------------------------------------
def run_acceptance_study(
    *,
    n_particles: int = 1200,
    packing_seed: int = 20261110,
    terminal_steps: int = 3000,
    rz_steps: int = 1500,
    ledger_steps: int = 1500,
    convergence_counts: tuple[int, ...] = (600, 1200, 2400),
    include_convergence: bool = True,
) -> dict:
    """Run the five criteria (and their negative controls) once."""
    bed = make_random_close_packing(n_particles, DIAMETER, 0.38, seed=packing_seed)
    probe = PROBE_RADIUS_OVER_D * DIAMETER
    if probe + DIAMETER >= 0.5 * bed.box_length:
        raise ValueError(
            f"probe radius {probe:.3g} too large for a {n_particles}-particle cell of "
            f"{bed.box_length:.3g} m; raise n_particles"
        )
    eps_local = local_porosity(bed.positions, bed.diameter, bed.box_length, probe)

    sections = {
        "packing": {
            "summary": bed.summary(),
            "porosity_statistics": porosity_statistics(eps_local, bed.porosity),
            "probe_radius_over_d": PROBE_RADIUS_OVER_D,
            "minimum_separation_over_d": bed.minimum_separation() / DIAMETER,
        },
        "ergun": criterion_ergun(bed, eps_local),
        "terminal_velocity": criterion_terminal_velocity(n_steps=terminal_steps),
        "rz_curve": criterion_rz_curve(n_steps=rz_steps),
        "ledger": criterion_ledger(n_steps=ledger_steps),
        "fluidization": criterion_fluidization(bed, eps_local),
    }
    criteria = {
        "C1_ergun": sections["ergun"]["criterion"],
        "C2a_stokes_terminal": sections["terminal_velocity"]["stokes"]["criterion"],
        "C2b_transition_terminal": sections["terminal_velocity"]["transition"]["criterion"],
        "C2c_rz_curve": sections["rz_curve"]["criterion"],
        "C3_ledger": sections["ledger"]["criterion"],
        "C4_fluidization": sections["fluidization"]["criterion"],
    }
    if include_convergence:
        conv = resolution_convergence(
            PACKED_FLUID,
            particle_counts=convergence_counts,
            diameter=DIAMETER,
            target_porosity=0.38,
            probe_radius_over_d=PROBE_RADIUS_OVER_D,
            settling_porosity=0.80,
            settling_steps=rz_steps,
        )
        sections["convergence"] = conv
        criteria["C5_convergence"] = _criterion(
            "resolution_convergence",
            conv["worst_spread"],
            0.05,
            conv["worst_spread"] <= 0.05,
            criterion="(max-min)/mean across N=600/1200/2400 <= 5 % for dp/dz, U_mf, settling speed",
            per_quantity={k: v["spread"] for k, v in conv["quantities"].items()},
            resolutions=list(convergence_counts),
        )
    else:
        sections["convergence"] = None

    failed = {k: v for k, v in criteria.items() if not v["passed"]}
    return {
        "node": "P142W3-N03",
        "module": "tensordem.bedcoupling.fluidization",
        "criteria": criteria,
        "failed_criteria": list(failed),
        "all_criteria_passed": not failed,
        "sections": sections,
        "settings": {
            "diameter_m": DIAMETER,
            "probe_radius_over_d": PROBE_RADIUS_OVER_D,
            "reynolds_sweep": [float(r) for r in REYNOLDS_SWEEP],
            "rz_porosities": list(RZ_POROSITIES),
            "n_particles": n_particles,
            "packing_seed": packing_seed,
            "terminal_steps": terminal_steps,
            "rz_steps": rz_steps,
            "ledger_steps": ledger_steps,
            "convergence_counts": list(convergence_counts),
        },
    }


def write_results(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    return path


def _main() -> None:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="P142W3-N03 acceptance study")
    parser.add_argument("--out", default="artifacts/P142W3-N03/results.json")
    args = parser.parse_args()
    results = run_acceptance_study()
    write_results(results, args.out)
    print(json.dumps({k: v["passed"] for k, v in results["criteria"].items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _main()