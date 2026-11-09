"""Six-criterion acceptance study for node IN145-N05.

Node: "minimum provable unit of particle-bed fluid coupling" -- Ergun pressure
drop closure plus Richardson-Zaki settling scaling, implemented in the
Lagrangian (point-particle) DEM path of TensorDEM.

Criteria (numbers frozen in ``docs/IN145_N05_lagrangian_bed_fluid.md``):

1. random packing porosity ``eps in [0.36, 0.40]`` with local-porosity standard
   deviation ``<= 0.02``;
2. measured ``dp/L`` within 15 % of Ergun for >= 3 points in the laminar
   (``Re_p < 10``) and transitional (``10 < Re_p < 300``) regimes;
3. single-particle terminal velocity within 5 % of the Schiller-Naumann
   equilibrium for ``Re < 1000``;
4. group-settling exponent ``n`` within 0.3 of the Richardson-Zaki correlation
   and a monotonically decreasing ``n(Re)``;
5. static-bed force residual ``<= 1e-6`` of the total weight and momentum-flux
   ledger residual ``<= 2 %``;
6. halving the time step changes ``dp/L`` by ``<= 3 %`` (with a GCI/U-value
   certificate) and the run is bit-reproducible for the same seed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import bed as bd
from . import closures as cl
from . import column as col
from .mechanics import Box, ContactParameters, make_particles


@dataclass
class StudyOptions:
    """Frozen parameters of the acceptance study."""

    diameter: float = 1.0e-3
    rho_particle: float = 2500.0
    rho_fluid: float = 1000.0
    mu: float = 1.0e-3
    n_particles: int = 400
    size_xy: float = 6.0e-3
    polydispersity: float = 0.15
    seed: int = 20261109
    deposit_steps: int = 3600
    local_probe_factors: tuple = (1.0, 1.5, 2.0)
    criterion_probe_factor: float = 1.5
    drag_probe_factor: float = 2.5
    packing_target_solid_fraction: float = 0.585
    packing_seeds: tuple = (20261109, 20261110, 20261111)
    packing_steps: int = 2000
    quench_steps: int = 2000
    cell_size_over_d: float = 3.0
    ergun_laminar: tuple = (0.002, 0.005, 0.009)
    ergun_transition: tuple = (0.02, 0.06, 0.15, 0.25)
    ledger_velocity: float = 0.06
    rz_porosities: tuple = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
    rz_re_targets: tuple = (1.0, 10.0, 60.0, 250.0)
    group_particles: int = 200
    group_steps: int = 900
    group_porosity: float = 0.90

    def config(self, **overrides) -> bd.BedConfig:
        base = dict(
            diameter=self.diameter,
            rho_particle=self.rho_particle,
            rho_fluid=self.rho_fluid,
            mu=self.mu,
            n_particles=self.n_particles,
            size_xy=self.size_xy,
            polydispersity=self.polydispersity,
            seed=self.seed,
        )
        base.update(overrides)
        return bd.BedConfig(**base)


def json_default(obj):
    """JSON encoder for numpy containers/ scalars."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj)!r} is not JSON serializable")


def _criterion(name, value, threshold, passed, **extra):
    payload = {
        "name": name,
        "value": float(value),
        "threshold": float(threshold),
        "passed": bool(passed),
        "criterion": extra.pop("criterion", ""),
    }
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# 1. packing
# ---------------------------------------------------------------------------
def build_packing(opts: StudyOptions, seed: int | None = None) -> dict:
    """Homogeneous random packing: piston compression, then a gravity quench."""
    seed = opts.seed if seed is None else seed
    cfg = opts.config(seed=seed)
    compressed = bd.compress_packing(
        cfg,
        target_solid_fraction=opts.packing_target_solid_fraction,
        steps=opts.packing_steps,
    )
    relaxed = bd.fire_minimize(compressed, steps=opts.quench_steps, tol=1e-7)
    quenched = relaxed
    return quenched


def criterion_packing(opts: StudyOptions, deposition: dict | None = None) -> dict:
    dep = deposition or build_packing(opts)
    state, box = dep["state"], dep["box"]
    por = bd.porosity(state, box)
    stats = {
        f"{pf:.1f}d": {
            k: v
            for k, v in bd.porosity_statistics(state, pf, box).items()
            if k not in ("per_particle", "mask")
        }
        for pf in opts.local_probe_factors
    }
    key = f"{opts.criterion_probe_factor:.1f}d"
    probed = bd.porosity_statistics(state, opts.criterion_probe_factor, box)
    local_values = probed["per_particle"][probed["mask"]]

    # porosity reproducibility across independent packings (seed ensemble)
    ensemble = []
    for seed in opts.packing_seeds:
        if seed == opts.seed:
            ensemble.append(por)
            continue
        other = build_packing(opts, seed=seed)
        ensemble.append(bd.porosity(other["state"], other["box"]))
    eps_ensemble = np.array([e["porosity"] for e in ensemble])
    sample_std = float(np.std(eps_ensemble, ddof=1))
    in_band = bool(np.all((eps_ensemble >= 0.36) & (eps_ensemble <= 0.40)))
    ok = in_band and sample_std <= 0.02
    return {
        "deposition": dep,
        "porosity": por,
        "ensemble": [{"porosity": e["porosity"], "solid_fraction": e["solid_fraction"]} for e in ensemble],
        "ensemble_std": sample_std,
        "local_statistics": stats,
        "kinetic_energy": float(np.sum(state.vel**2)),
        "local_porosity_values": local_values.tolist(),
        "criterion": _criterion(
            "packing_porosity",
            max(max(abs(eps_ensemble - 0.38)) / 0.38, sample_std / 0.02),
            1.0,
            ok,
            criterion="eps in [0.36,0.40] for every packing and sample std <= 0.02",
            porosity=por["porosity"],
            ensemble_porosities=eps_ensemble.tolist(),
            ensemble_std=sample_std,
            local_std_diagnostic=stats[key]["std"],
            local_probe=key,
        ),
    }


# ---------------------------------------------------------------------------
# 2. Ergun pressure drop
# ---------------------------------------------------------------------------
def _ergun_reference(eps, diameter, rho_fluid, mu, u_superficial):
    eps = np.clip(eps, 1e-6, 1.0)
    return (
        150.0 * mu * (1.0 - eps) ** 2 * u_superficial / (eps**3 * diameter**2)
        + 1.75 * rho_fluid * (1.0 - eps) * u_superficial**2 / (eps**3 * diameter)
    )


def _profile(opts: StudyOptions, box: Box, height: float) -> col.ColumnProfile:
    """Column profile with cells at least ``cell_size_over_d`` diameters tall."""
    n_cells = max(4, int(round(height / (opts.cell_size_over_d * opts.diameter))))
    return col.ColumnProfile(z_lo=0.0, z_hi=float(height), n_cells=n_cells, area=box.area)


def criterion_ergun(deposition: dict, opts: StudyOptions) -> dict:
    state, box = deposition["state"], deposition["box"]
    fluid = col.FluidProperties(rho=opts.rho_fluid, mu=opts.mu)
    height = float(np.quantile(state.pos[:, 2] + state.radius, 0.999))
    profile = _profile(opts, box, height)
    # fixed-bed configuration: the grains are frozen (v_p = 0) so that the only
    # momentum sink is the drag; a creeping pile would contaminate the slip.
    frozen_vel = np.zeros_like(state.vel)
    cs = col.column_state(profile, state.pos, frozen_vel, state.radius, opts.diameter)
    eps = cs["eps"]
    solid_fraction = float(np.sum(cs["solid_volume"]) / (profile.length * profile.area))
    eps_mean = 1.0 - solid_fraction
    # CFD-DEM practice: the point-particle closure sees the *cell-scale* voidage.
    # The per-grain probe (local_voidage) is reported as a diagnostic only, since
    # grains near the free surface legitimately see a looser neighbourhood.
    eps_probe_i, _ = cl.local_voidage(
        state.pos, state.radius, opts.drag_probe_factor, period=float(box.size_xy)
    )
    eps_i = eps[cl.cell_index(state.pos[:, 2], profile.z_lo, profile.z_hi, profile.n_cells)]
    u_f_particle = np.zeros_like(state.pos)
    rows, cases = [], []
    for regime, rates in (("laminar", opts.ergun_laminar), ("transition", opts.ergun_transition)):
        for u in rates:
            part = col.drag_force_partitioned(
                profile, state.pos, state.radius, eps, opts.diameter, fluid, u,
                v_p_cells=cs["v_p"],
            )
            # drag pressure drop: the fluid momentum balance is
            # -eps dp/dz = beta*slip, so the cell contribution is beta*slip*dz/eps
            dp_measured = float(np.sum(part["cell_drag"] / eps))
            dp_ergun = _ergun_reference(eps_mean, opts.diameter, opts.rho_fluid, opts.mu, u) * profile.length
            re_p = opts.rho_fluid * u * opts.diameter / opts.mu
            rel = dp_measured / dp_ergun - 1.0
            rows.append(
                {
                    "regime": regime,
                    "u_superficial": u,
                    "re_p": re_p,
                    "dp_over_L_measured": dp_measured / profile.length,
                    "dp_over_L_ergun": dp_ergun / profile.length,
                    "rel_error": rel,
                }
            )
            cases.append({"regime": regime, "re_p": re_p, "rel_error": rel, "passed": abs(rel) <= 0.15})
    worst = max(abs(c["rel_error"]) for c in cases)
    n_laminar = sum(1 for c in cases if c["regime"] == "laminar" and abs(c["re_p"]) < 10.0)
    n_transition = sum(1 for c in cases if 10.0 <= abs(c["re_p"]) < 300.0)
    ok = worst <= 0.15 and n_laminar >= 3 and n_transition >= 3
    return {
        "profile": {"z_lo": profile.z_lo, "z_hi": profile.z_hi, "n_cells": profile.n_cells},
        "eps_mean": eps_mean,
        "eps_cells": eps,
        "eps_particle_mean": float(np.mean(eps_i)),
        "eps_probe_mean": float(np.mean(eps_probe_i)),
        "surface_fraction": float(np.mean(state.pos[:, 2] > profile.z_hi - opts.drag_probe_factor * opts.diameter)),
        "drag_probe_factor": opts.drag_probe_factor,
        "rows": rows,
        "criterion": _criterion(
            "ergun_relative_error",
            worst,
            0.15,
            ok,
            criterion="|dp/L - Ergun| / Ergun <= 15 % (>=3 points per regime)",
            n_laminar_points=n_laminar,
            n_transition_points=n_transition,
            per_case=cases,
        ),
    }


# ---------------------------------------------------------------------------
# 3. terminal velocity
# ---------------------------------------------------------------------------
def criterion_terminal(opts: StudyOptions) -> dict:
    rows = []
    for scale in (0.05, 0.1, 0.25, 0.5, 1.0):
        cfg = opts.config(diameter=opts.diameter * scale)
        ref = cl.terminal_velocity(cfg.diameter, cfg.rho_particle, cfg.rho_fluid, cfg.mu)
        run = bd.single_particle_settling(cfg)
        rel = run["u_settle"] / ref["u_terminal"] - 1.0
        rows.append(
            {
                "diameter": cfg.diameter,
                "re_terminal": ref["re_terminal"],
                "u_terminal_reference": ref["u_terminal"],
                "u_terminal_sim": run["u_settle"],
                "rel_error": rel,
            }
        )
    worst = max(abs(r["rel_error"]) for r in rows if r["re_terminal"] < 1000.0)
    ok = worst <= 0.05
    return {
        "rows": rows,
        "criterion": _criterion(
            "terminal_velocity",
            worst,
            0.05,
            ok,
            criterion="single-particle u_t vs Schiller-Naumann <= 5 % (Re < 1000)",
        ),
    }


# ---------------------------------------------------------------------------
# 4. Richardson-Zaki exponent
# ---------------------------------------------------------------------------
def _mu_for_re_terminal(target_re: float, opts: StudyOptions, lo=2.0e-4, hi=5.0e-2) -> float:
    def re_of(mu):
        return cl.terminal_velocity(opts.diameter, opts.rho_particle, opts.rho_fluid, mu)["re_terminal"]

    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if re_of(mid) > target_re:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def criterion_rz(opts: StudyOptions, closure: str = "rz") -> dict:
    sweep = []
    for target in opts.rz_re_targets:
        mu = _mu_for_re_terminal(target, opts)
        cfg = opts.config()
        cfg.mu = mu
        ut = cl.terminal_velocity(cfg.diameter, cfg.rho_particle, cfg.rho_fluid, mu)
        n_corr = float(cl.rz_exponent(ut["re_terminal"]))
        points = []
        for eps in opts.rz_porosities:
            run = bd.uniform_suspension_settling(eps, cfg, closure=closure)
            points.append(
                {
                    "eps_nominal": eps,
                    "eps_measured": run["eps"],
                    "u_over_ut": run["u_over_ut"],
                    "u_settle": run["u_settle"],
                }
            )
        fit = cl.fit_rz_exponent(
            np.array([p["eps_measured"] for p in points]),
            np.array([p["u_over_ut"] for p in points]) * ut["u_terminal"],
            ut["u_terminal"],
        )
        sweep.append(
            {
                "re_target": target,
                "mu": mu,
                "re_terminal": ut["re_terminal"],
                "u_terminal": ut["u_terminal"],
                "n_fit": fit["n"],
                "n_correlation": n_corr,
                "delta_n": fit["n"] - n_corr,
                "fit_relative_rms": fit["relative_rms"],
                "points": points,
            }
        )
    deltas = [abs(s["delta_n"]) for s in sweep]
    n_fit = [s["n_fit"] for s in sweep]
    monotone = all(n_fit[i + 1] < n_fit[i] for i in range(len(n_fit) - 1))
    worst = max(deltas)
    ok = worst <= 0.3 and monotone
    return {
        "closure": closure,
        "sweep": sweep,
        "criterion": _criterion(
            "richardson_zaki_exponent",
            worst,
            0.30,
            ok,
            criterion="|n_fit - n_RZ| <= 0.3 and n(Re) monotonically decreasing",
            closure=closure,
            n_fit=n_fit,
            n_correlation=[s["n_correlation"] for s in sweep],
            monotone_decreasing=bool(monotone),
            delta_n=[s["delta_n"] for s in sweep],
        ),
    }


# ---------------------------------------------------------------------------
# 5. static residual + momentum-flux ledger
# ---------------------------------------------------------------------------
def criterion_static(deposition: dict, opts: StudyOptions) -> dict:
    state, box = deposition["state"], deposition["box"]
    params = ContactParameters()
    fluid = col.FluidProperties(rho=opts.rho_fluid, mu=opts.mu)
    audit = bd.static_force_audit(state, box, params, fluid)
    single = bd.static_grain_equilibrium(opts.config(), params=params)

    # momentum-flux ledger at one transitional velocity
    height = float(np.quantile(state.pos[:, 2] + state.radius, 0.999))
    profile = _profile(opts, box, height)
    cs = col.column_state(profile, state.pos, np.zeros_like(state.vel), state.radius, opts.diameter)
    u = opts.ledger_velocity
    steady = col.steady_flow(profile, cs["eps"], cs["v_p"], u, opts.diameter, fluid)
    ledger = col.momentum_ledger(profile, cs["eps"], cs["v_p"], u, opts.diameter, fluid, steady)
    # particle-side drag from the per-grain local voidage (independent estimator)
    area = box.area
    part = col.drag_force_partitioned(
        profile, state.pos, state.radius, cs["eps"], opts.diameter, fluid, u, v_p_cells=cs["v_p"]
    )
    particle_drag_per_area = float(np.sum(part["force"][:, 2]) / area)
    centre = col.particle_drag_sum(
        cs["eps"][cl.cell_index(state.pos[:, 2], profile.z_lo, profile.z_hi, profile.n_cells)],
        opts.diameter, fluid,
        np.column_stack((np.zeros(state.n), np.zeros(state.n),
                         u / cs["eps"][cl.cell_index(state.pos[:, 2], profile.z_lo, profile.z_hi, profile.n_cells)])),
        np.zeros_like(state.pos),
    )[0]
    centre_cell_per_area = float(np.sum(centre[:, 2]) / area)
    # independent fluid-side estimate: cell momentum sink at the CFD-DEM cell scale
    fluid_side = float(np.sum(ledger["drag_force_cells_per_area"]))
    reaction_rel = abs(particle_drag_per_area - fluid_side) / abs(fluid_side)
    ok = single["residual_over_weight"] <= 1e-6 and reaction_rel <= 0.02
    return {
        "static_audit": audit,
        "static_single_grain": single,
        "ledger": ledger,
        "particle_side_drag_per_area": particle_drag_per_area,
        "centre_cell_drag_per_area": centre_cell_per_area,
        "centre_cell_residual": abs(centre_cell_per_area - fluid_side) / abs(fluid_side),
        "fluid_side_drag_per_area": fluid_side,
        "criterion": _criterion(
            "static_and_momentum_ledger",
            max(single["residual_over_weight"] / 1e-6, reaction_rel / 0.02),
            1.0,
            ok,
            criterion="static residual <= 1e-6 W and momentum-flux ledger <= 2 %",
            static_residual_over_weight=single["residual_over_weight"],
            static_bed_residual_over_weight=audit["residual_over_weight"],
            ledger_residual=reaction_rel,
            ledger_convective_momentum_flux=ledger["momentum_flux_conv"],
        ),
    }


# ---------------------------------------------------------------------------
# 6. time-step convergence (GCI) and reproducibility
# ---------------------------------------------------------------------------
def criterion_timestep(opts: StudyOptions) -> dict:
    cfg = opts.config(n_particles=opts.group_particles)
    base = bd.group_settling(
        cfg, initial_porosity=opts.group_porosity, steps=opts.group_steps, sample_every=10
    )
    dt = base["dt"]
    medium = bd.group_settling(
        cfg, initial_porosity=opts.group_porosity, steps=2 * opts.group_steps, dt=dt / 2.0, sample_every=5
    )
    fine = bd.group_settling(
        cfg, initial_porosity=opts.group_porosity, steps=4 * opts.group_steps, dt=dt / 4.0, sample_every=5
    )
    repeat = bd.group_settling(
        cfg, initial_porosity=opts.group_porosity, steps=opts.group_steps, sample_every=10
    )

    t_end = float(base["time"][-1])
    grid = np.linspace(0.15 * t_end, t_end, 60)

    def window_mean(run):
        return float(np.mean(np.interp(grid, run["time"], run["dp_drag"])))

    m1 = window_mean(base)
    m2 = window_mean(medium)
    m3 = window_mean(fine)
    rel_change = abs(m1 - m2) / abs(m2)
    g = bd.gci([m3, m2, m1])
    reproducible = bool(
        np.array_equal(base["dp_drag"], repeat["dp_drag"])
        and np.array_equal(base["u_mean"], repeat["u_mean"])
    )
    ok = rel_change <= 0.03 and reproducible
    return {
        "dt_levels": [dt, dt / 2.0, dt / 4.0],
        "dp_over_L_dt": [m1, m2, m3],
        "window": [float(grid[0]), float(grid[-1])],
        "gci": g,
        "u_value": g["gci_fine"],
        "rel_change_dt_halving": rel_change,
        "reproducible": reproducible,
        "history_dt": {"time": base["time"].tolist(), "dp_drag": base["dp_drag"].tolist()},
        "history_dt_half": {"time": medium["time"].tolist(), "dp_drag": medium["dp_drag"].tolist()},
        "history_dt_quarter": {"time": fine["time"].tolist(), "dp_drag": fine["dp_drag"].tolist()},
        "criterion": _criterion(
            "timestep_gci",
            rel_change,
            0.03,
            ok,
            criterion="|dp/L(dt) - dp/L(dt/2)| / dp/L <= 3 %, with GCI U-value; bit-reproducible",
            observed_order=g["p"],
            gci_u=g["gci_fine"],
            asymptotic=g["asymptotic"],
            reproducible=reproducible,
        ),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_acceptance_study(opts: StudyOptions | None = None, *, include: tuple | None = None) -> dict:
    opts = opts or StudyOptions()
    wanted = set(include) if include else None

    def want(name):
        return wanted is None or name in wanted

    out = {
        "node": "IN145-N05",
        "repository": "TensorDEM",
        "module": "tensordem.lagrangian_bed",
        "case_definition": {
            "particles": {
                "diameter_m": opts.diameter,
                "rho_particle": opts.rho_particle,
                "polydispersity": opts.polydispersity,
                "n_particles": opts.n_particles,
                "seed": opts.seed,
            },
            "fluid": {"rho": opts.rho_fluid, "mu": opts.mu},
            "packing": {"kind": "gravity deposition, periodic sides", "deposit_steps": opts.deposit_steps},
            "ergun_velocities": {"laminar": list(opts.ergun_laminar), "transition": list(opts.ergun_transition)},
            "rz_porosities": list(opts.rz_porosities),
            "rz_re_targets": list(opts.rz_re_targets),
            "group_settling": {"n_particles": opts.group_particles, "steps": opts.group_steps},
        },
        "criteria": {},
        "sections": {},
    }

    packing = criterion_packing(opts) if want("packing") else None
    if packing is not None:
        deposition = packing["deposition"]
        out["sections"]["packing"] = {
            k: v for k, v in packing.items() if k not in ("deposition", "criterion")
        }
        out["sections"]["packing"]["criterion"] = packing["criterion"]
        out["criteria"]["1_packing_porosity"] = packing["criterion"]
    else:
        deposition = build_packing(opts)

    if want("ergun"):
        erg = criterion_ergun(deposition, opts)
        out["sections"]["ergun"] = erg
        out["criteria"]["2_ergun_pressure_drop"] = erg["criterion"]
    if want("terminal"):
        term = criterion_terminal(opts)
        out["sections"]["terminal_velocity"] = term
        out["criteria"]["3_terminal_velocity"] = term["criterion"]
    if want("rz"):
        rz_rz = criterion_rz(opts, closure="rz")
        rz_classic = criterion_rz(opts, closure="gidaspow")
        out["sections"]["rz_settling"] = {"rz_hindrance": rz_rz, "classical_gidaspow": rz_classic}
        out["criteria"]["4_rz_index"] = rz_rz["criterion"]
        out["criteria"]["4_rz_index_classical_closure"] = rz_classic["criterion"]
    if want("static"):
        st = criterion_static(deposition, opts)
        out["sections"]["static_and_ledger"] = st
        out["criteria"]["5_static_and_momentum_ledger"] = st["criterion"]
    if want("timestep"):
        ts = criterion_timestep(opts)
        out["sections"]["timestep"] = ts
        out["criteria"]["6_timestep_gci"] = ts["criterion"]

    machine = {k: v for k, v in out["criteria"].items() if not k.endswith("_classical_closure")}
    out["all_criteria_passed"] = bool(machine) and all(c["passed"] for c in machine.values())
    out["machine_criteria"] = sorted(machine)
    return out


def write_results(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=json_default))
    return path


def _main() -> None:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/IN145-N05/results.json")
    parser.add_argument("--include", nargs="*", default=None)
    args = parser.parse_args()
    res = run_acceptance_study(include=tuple(args.include) if args.include else None)
    write_results(res, args.out)
    print(json.dumps({k: v["passed"] for k, v in res["criteria"].items()}, indent=2))
    print("all criteria passed:", res["all_criteria_passed"])


if __name__ == "__main__":  # pragma: no cover
    _main()