"""Acceptance study: Ergun pressure drop, static balance and RZ settling scaling.

Six machine-checkable criteria are evaluated here and written to JSON:

===  ======================================================================
1    random-packing porosity ``eps in [0.36, 0.40]`` with ``std <= 0.02``
2    fixed-bed ``dp/dz`` vs the Ergun relation, laminar and transition, <= 15 %
3    single-particle terminal velocity vs the Schiller--Naumann balance, <= 5 %
4    batch-settling index ``n`` vs the Richardson--Zaki correlation, |Delta|<=0.3
5    static-bed net-force residual, <= 1e-6 of the total gravity
6    time-step halving changes ``dp/dz`` by <= 3 %
===  ======================================================================

Criterion 4 is evaluated on the **mixture-buoyancy** formulation (see
:func:`tensordem.bedcoupling.bed.body_force_per_particle`), which is the one
consistent with the Anderson--Jackson inter-phase coupling; the
hydrostatic-buoyancy variant is reported alongside, because the difference
between the two is exactly one power of the voidage and hence one unit of the
Richardson--Zaki index.  Nothing is tuned to hit a threshold: every number is
produced by the run and reported with its case definition.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bed import (
    FluidProperties,
    bed_force_assembly,
    bed_pressure_drop,
    body_force_per_particle,
    fit_power_law,
    simulate_settling,
    settling_sweep,
    static_bed_force_audit,
)
from .closures import (
    ergun_pressure_gradient,
    gidaspow_branch_jump,
    rz_exponent,
    terminal_velocity_sn,
)
from .packing import (
    PackedBed,
    local_porosity,
    make_jittered_cubic_array,
    make_random_close_packing,
    porosity_statistics,
)

__all__ = ["run_acceptance_study", "write_results", "PACKING_DIAMETER"]

PACKING_DIAMETER = 1.0e-3  # 1 mm monodisperse bed
PROBE_RADIUS_OVER_D = 3.0  # top-hat probe radius of the local-voidage kernel
PACKED_BED_FLUID = FluidProperties(rho_fluid=998.0, rho_particle=2650.0, mu=1.0e-3)
SETTLING_FLUID = FluidProperties(rho_fluid=1200.0, rho_particle=2650.0, mu=0.05)
RZ_POROSITIES = (0.80, 0.86, 0.92, 0.96)


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


# ---------------------------------------------------------------------------
def criterion_packing(n_particles: int = 1500, seed: int = 20261108) -> dict:
    bed = make_random_close_packing(n_particles, PACKING_DIAMETER, 0.38, seed=seed)
    probe = PROBE_RADIUS_OVER_D * PACKING_DIAMETER
    eps_local = local_porosity(bed.positions, bed.diameter, bed.box_length, probe)
    stats = porosity_statistics(eps_local, bed.porosity)
    sensitivity = []
    for ratio in (2.0, 2.5, 3.0, 3.5):
        if (ratio + 1.0) * PACKING_DIAMETER >= 0.5 * bed.box_length:
            continue  # probe would break the minimum-image construction
        e = local_porosity(bed.positions, bed.diameter, bed.box_length, ratio * PACKING_DIAMETER)
        s = porosity_statistics(e, bed.porosity)
        sensitivity.append(
            {"probe_radius_over_d": ratio, "mean": s["mean"], "std": s["std"], "min": s["min"], "max": s["max"]}
        )
    # a one-diameter-thick slab, for the report's voidage field map
    z0 = 0.5 * bed.box_length
    slab_mask = np.abs(bed.positions[:, 2] - z0) <= 1.5 * bed.diameter
    slab = [
        {"x": float(x), "y": float(y), "eps": float(e)}
        for x, y, e in zip(bed.positions[slab_mask, 0], bed.positions[slab_mask, 1], eps_local[slab_mask])
    ]
    passed = (0.36 <= stats["mean"] <= 0.40) and stats["std"] <= 0.02 and bed.max_overlap <= 1e-3 * bed.diameter
    return {
        "packing": bed.summary(),
        "probe_radius": probe,
        "porosity_statistics": stats,
        "probe_sensitivity": sensitivity,
        "local_voidage": eps_local.tolist(),
        "slab_z": z0,
        "slab": slab,
        "criterion": _criterion(
            "porosity_in_window_and_spread",
            stats["mean"],
            0.02,
            passed,
            criterion="eps in [0.36,0.40], std <= 0.02, max overlap <= 1e-3 d",
            std=stats["std"],
            max_overlap_over_d=bed.max_overlap / bed.diameter,
        ),
    }


def criterion_ergun(bed: PackedBed, eps_local: np.ndarray) -> dict:
    fluid = PACKED_BED_FLUID
    d = bed.diameter
    cases = []
    rows = []
    for u_sup in (5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 8e-2):
        ref = ergun_pressure_gradient(
            u_sup, diameter=d, porosity=bed.porosity, rho_fluid=fluid.rho_fluid, mu=fluid.mu
        )
        measured_local = bed_pressure_drop(bed, u_sup, fluid, eps_local=eps_local)
        measured_mean = bed_pressure_drop(bed, u_sup, fluid, eps_local=None)
        wen_yu_branch = bed_pressure_drop(bed, u_sup, fluid, eps_local=None, branch="wen_yu")
        rows.append(
            {
                "u_superficial": u_sup,
                "ergun_dpdz": ref.total,
                "ergun_viscous": ref.viscous,
                "ergun_inertial": ref.inertial,
                "inertial_fraction": ref.inertial_fraction,
                "regime": ref.regime,
                "measured_local_voidage_dpdz": measured_local,
                "measured_mean_voidage_dpdz": measured_mean,
                "wen_yu_branch_dpdz": wen_yu_branch,
                "rel_error_local": measured_local / ref.total - 1.0,
                "rel_error_mean": measured_mean / ref.total - 1.0,
                "rel_error_wen_yu_branch": wen_yu_branch / ref.total - 1.0,
            }
        )
    for target_regime, u_sup in (("laminar", 2e-3), ("transition", 5e-2)):
        row = next(r for r in rows if r["u_superficial"] == u_sup)
        cases.append(
            {
                "regime": target_regime,
                "u_superficial": u_sup,
                "inertial_fraction": row["inertial_fraction"],
                "ergun_dpdz": row["ergun_dpdz"],
                "measured_dpdz": row["measured_local_voidage_dpdz"],
                "rel_error": row["rel_error_local"],
            }
        )
    worst = max(abs(c["rel_error"]) for c in cases)
    return {
        "fluid": asdict(fluid),
        "rows": rows,
        "cases": cases,
        "criterion": _criterion(
            "ergun_relative_error",
            worst,
            0.15,
            worst <= 0.15,
            criterion="|dp/dz measured - Ergun| / Ergun <= 15 % in laminar and transition regimes",
            per_case=[{"regime": c["regime"], "rel_error": c["rel_error"]} for c in cases],
        ),
        "branch_jump": gidaspow_branch_jump(
            5e-2, diameter=d, rho_fluid=fluid.rho_fluid, mu=fluid.mu
        ),
    }


def criterion_terminal_velocity() -> dict:
    fluid = PACKED_BED_FLUID
    d = PACKING_DIAMETER
    bed = PackedBed(
        diameter=d,
        box_length=20.0 * d,
        positions=np.array([[10.0 * d, 10.0 * d, 10.0 * d]]),
        kind="single-particle",
    )
    ref = terminal_velocity_sn(
        d, rho_particle=fluid.rho_particle, rho_fluid=fluid.rho_fluid, mu=fluid.mu
    )
    run = simulate_settling(bed, fluid, n_steps=4000, fluid_mode="at_rest")
    u_sim = -run["steady_mean_vz"]
    rel = u_sim / ref["u_terminal"] - 1.0
    return {
        "u_terminal_sim": u_sim,
        "u_terminal_reference": ref["u_terminal"],
        "re_terminal": ref["re_terminal"],
        "rel_error": rel,
        "dt_over_tau": run["dt_over_tau"],
        "criterion": _criterion(
            "terminal_velocity_error",
            abs(rel),
            0.05,
            abs(rel) <= 0.05,
            criterion="|u_sim - u_SchillerNaumann| / u <= 5 %",
        ),
    }


def criterion_rz(*, n_steps: int = 2000, include_sweep: bool = True) -> dict:
    fluid = SETTLING_FLUID
    d = PACKING_DIAMETER
    ref = terminal_velocity_sn(
        d, rho_particle=fluid.rho_particle, rho_fluid=fluid.rho_fluid, mu=fluid.mu
    )
    n_rz = float(rz_exponent(ref["re_terminal"]))
    packings = {e: make_jittered_cubic_array(6, d, porosity=e) for e in RZ_POROSITIES}
    out = {"u_terminal": ref["u_terminal"], "re_terminal": ref["re_terminal"], "n_rz": n_rz}
    payload = {}
    for model in ("mixture", "hydrostatic"):
        sweep = settling_sweep(packings, fluid, n_steps=n_steps, buoyancy_model=model)
        n_sim = sweep["fit"]["exponent"]
        payload[model] = {
            "runs": sweep["runs"],
            "fit": {k: v for k, v in sweep["fit"].items() if k not in ("x", "y")},
            "exponent": n_sim,
            "deviation": n_sim - n_rz,
            "deviation_over_threshold": abs(n_sim - n_rz) / 0.3,
        }
    out["variants"] = payload
    primary = payload["mixture"]
    out["acceptance"] = {
        "buoyancy_model": "mixture",
        "n_sim": primary["exponent"],
        "n_rz": n_rz,
        "deviation": primary["deviation"],
    }
    out["criterion"] = _criterion(
        "rz_index_deviation",
        abs(primary["deviation"]),
        0.3,
        abs(primary["deviation"]) <= 0.3,
        criterion="|n_sim - n_RZ| <= 0.3 (mixture buoyancy; hydrostatic variant reported too)",
        n_sim=primary["exponent"],
        n_rz=n_rz,
        hydrostatic_deviation=payload["hydrostatic"]["deviation"],
    )
    # Re_t sweep: where does the Gidaspow blend stop tracking RZ?
    sweep_rows = []
    if not include_sweep:
        out["re_sweep"] = sweep_rows
        return out
    for rho_p, rho_f, mu, label in (
        (2650.0, 1200.0, 0.30, "glycerin-water 30 P"),
        (2650.0, 1200.0, 0.05, "glycerin-water 5 P"),
        (2500.0, 998.0, 1.0e-3, "sand-water"),
        (7850.0, 998.0, 1.0e-3, "steel-water"),
        (2500.0, 1.2, 1.8e-5, "sand-air"),
    ):
        f = FluidProperties(rho_fluid=rho_f, rho_particle=rho_p, mu=mu)
        r = terminal_velocity_sn(d, rho_particle=rho_p, rho_fluid=rho_f, mu=mu)
        row = {"label": label, "re_terminal": r["re_terminal"], "n_rz": float(rz_exponent(r["re_terminal"]))}
        for model in ("mixture", "hydrostatic"):
            sw = settling_sweep(packings, f, n_steps=n_steps, buoyancy_model=model)
            row[f"n_sim_{model}"] = sw["fit"]["exponent"]
            row[f"dev_{model}"] = sw["fit"]["exponent"] - row["n_rz"]
        sweep_rows.append(row)
    out["re_sweep"] = sweep_rows
    return out


def criterion_static_residual(bed: PackedBed) -> dict:
    audit = static_bed_force_audit(bed, PACKED_BED_FLUID)
    return {
        **audit,
        "criterion": _criterion(
            "static_bed_residual",
            audit["residual_over_total_gravity"],
            1e-6,
            audit["residual_over_total_gravity"] <= 1e-6,
            criterion="|sum F_net - expected| / sum(m g) <= 1e-6",
            max_per_particle_over_weight=audit["max_per_particle_residual_over_weight"],
        ),
    }


def criterion_timestep() -> dict:
    """Halve the time step and compare the drag pressure drop of the same run."""
    fluid = SETTLING_FLUID
    d = PACKING_DIAMETER
    bed = make_jittered_cubic_array(6, d, porosity=0.80)
    base = simulate_settling(bed, fluid, n_steps=600, fluid_mode="closed_box")
    dt = base["dt"]
    coarse = simulate_settling(bed, fluid, n_steps=600, dt=dt, fluid_mode="closed_box")
    fine = simulate_settling(bed, fluid, n_steps=1200, dt=dt / 2.0, fluid_mode="closed_box")
    dp_coarse = coarse["steady_mean_dpdz"]
    dp_fine = fine["steady_mean_dpdz"]
    rel = abs(dp_coarse - dp_fine) / abs(dp_fine)
    # transient comparison at an early common physical time
    t_star = 2.0 * coarse["tau"]
    i_coarse = int(np.argmin(np.abs(coarse["time"] - t_star)))
    i_fine = int(np.argmin(np.abs(fine["time"] - t_star)))
    transient_rel = abs(coarse["dpdz"][i_coarse] - fine["dpdz"][i_fine]) / abs(fine["dpdz"][i_fine])
    return {
        "dt": dt,
        "dt_over_tau": coarse["dt_over_tau"],
        "dpdz_dt": dp_coarse,
        "dpdz_dt_half": dp_fine,
        "rel_change": rel,
        "t_star": t_star,
        "dpdz_transient_dt": float(coarse["dpdz"][i_coarse]),
        "dpdz_transient_dt_half": float(fine["dpdz"][i_fine]),
        "transient_rel_change": float(transient_rel),
        "history_dt": {"time": coarse["time"].tolist(), "dpdz": coarse["dpdz"].tolist()},
        "history_dt_half": {"time": fine["time"].tolist(), "dpdz": fine["dpdz"].tolist()},
        "criterion": _criterion(
            "timestep_halving_dpdz_change",
            rel,
            0.03,
            rel <= 0.03,
            criterion="|dp/dz(dt) - dp/dz(dt/2)| / dp/dz <= 3 %",
            transient_rel_change=float(transient_rel),
        ),
    }


# ---------------------------------------------------------------------------
def run_acceptance_study(
    *,
    n_particles: int = 1500,
    seed: int = 20261108,
    n_steps: int = 2000,
    include_sweeps: bool = True,
) -> dict:
    packing = criterion_packing(n_particles=n_particles, seed=seed)
    bed = make_random_close_packing(n_particles, PACKING_DIAMETER, 0.38, seed=seed)
    eps_local = local_porosity(
        bed.positions, bed.diameter, bed.box_length, PROBE_RADIUS_OVER_D * PACKING_DIAMETER
    )
    ergun = criterion_ergun(bed, eps_local)
    terminal = criterion_terminal_velocity()
    rz = criterion_rz(n_steps=n_steps, include_sweep=include_sweeps)
    static = criterion_static_residual(bed)
    timestep = criterion_timestep()

    criteria = {
        "1_packing_porosity": packing["criterion"],
        "2_ergun_pressure_drop": ergun["criterion"],
        "3_terminal_velocity": terminal["criterion"],
        "4_rz_index": rz["criterion"],
        "5_static_bed_residual": static["criterion"],
        "6_timestep_halving": timestep["criterion"],
    }
    all_passed = all(c["passed"] for c in criteria.values())
    return {
        "node": "IN143-N09",
        "repository": "TensorDEM",
        "module": "tensordem.bedcoupling",
        "case_definition": {
            "packing": {
                "kind": "random close packing, quasi-static compression",
                "diameter_m": PACKING_DIAMETER,
                "target_porosity": 0.38,
                "n_particles": n_particles,
                "seed": seed,
            },
            "packed_bed_fluid": asdict(PACKED_BED_FLUID),
            "settling_fluid": asdict(SETTLING_FLUID),
            "rz_porosities": list(RZ_POROSITIES),
            "local_voidage_probe_radius_over_d": PROBE_RADIUS_OVER_D,
        },
        "criteria": criteria,
        "all_criteria_passed": all_passed,
        "sections": {
            "packing": packing,
            "ergun": ergun,
            "terminal_velocity": terminal,
            "rz_settling": rz,
            "static_bed": static,
            "timestep": timestep,
        },
    }


def write_results(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    return path


def _main() -> None:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/IN143-N09/results.json")
    parser.add_argument("--n-particles", type=int, default=1500)
    parser.add_argument("--n-steps", type=int, default=2000)
    args = parser.parse_args()
    res = run_acceptance_study(n_particles=args.n_particles, n_steps=args.n_steps)
    write_results(res, args.out)
    for name, c in res["criteria"].items():
        print(f"{name:28s} value={c['value']:.6g} threshold={c['threshold']:.6g} passed={c['passed']}")
    print("all passed:", res["all_criteria_passed"])


if __name__ == "__main__":  # pragma: no cover
    _main()
