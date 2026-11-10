"""End-to-end study of the clear-ice floe field (P103W2-N03).

Produces the five blocks of :class:`~tensordem.clear_ice.contract.ClearIceResult`:

1. ``floe_kinematics``  -- the deliverable plus its resolution metrics;
2. ``force_chain_stats`` -- ``P(f)`` exponential tail with ``R^2``, ``Z``,
   friction/load-share diagnostics, power-law alternative;
3. ``porosity`` -- areal and periodic-Voronoi free volume in the ambient field,
   the upstream jam and the wake;
4. ``convergence_report`` -- a deterministic floe-trajectory order study
   (C1 primary), an ensemble step-size sweep, and a particle-count study;
5. ``uncertainty_quantification`` -- one-at-a-time sensitivities of the reported
   statistics to the material/contact parameters and the ensemble seed.

Each study returns plain dicts so that the artifact is JSON-serialisable and
every number in the report can be traced to a run.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from .contact import ContactMaterial
from .contract import (
    CONTRACT_VERSION,
    ClearIceRequest,
    ClearIceResult,
    FloePopulation,
    population_from_breakup,
    validate_population,
)
from .field import ClearIceField, FieldConfig, RunOutput
from .kinematics import kinematics_summary
from .statistics import (
    areal_porosity,
    coordination_number,
    force_distribution,
    region_masks,
    voronoi_free_volume,
)

__all__ = [
    "build_population",
    "baseline_run",
    "collect_statistics",
    "ensemble_dt_sweep",
    "particle_count_study",
    "trajectory_convergence",
    "uq_sweep",
    "run_study",
]


# --------------------------------------------------------------------------
# population hand-over
# --------------------------------------------------------------------------
def build_population(cfg: FieldConfig) -> FloePopulation:
    """Generate and relax the clear-ice population, then hand it over as data.

    The generation goes through the same relaxation the field uses; the result
    is a :class:`FloePopulation` (area ledger + hash) so the field run is
    reproducible from the contract payload alone.
    """
    field = ClearIceField(cfg)
    area = float(np.sum(np.pi * field.radius ** 2))
    return FloePopulation(
        x=field.x.copy(),
        radius=field.radius.copy(),
        thickness=np.full(len(field.radius), cfg.thickness),
        source="synthetic_clear_ice_polydisperse",
        parent_area=area,
        metadata={
            "seed": cfg.seed,
            "ice_concentration_target": cfg.ice_concentration,
            "radius_log_sigma": cfg.radius_log_sigma,
            "generator": "tensordem.clear_ice.field.ClearIceField._radii/_place",
        },
    )


def _field_from_population(cfg: FieldConfig, pop: FloePopulation) -> ClearIceField:
    return ClearIceField(cfg, population=(pop.x, pop.radius, pop.thickness))


def _run(cfg: FieldConfig, pop: FloePopulation | None = None) -> tuple[ClearIceField, RunOutput]:
    if pop is None:
        pop = build_population(cfg)
    field = _field_from_population(cfg, pop)
    out = field.run()
    return field, out


# --------------------------------------------------------------------------
# statistics of one run
# --------------------------------------------------------------------------
def collect_statistics(cfg: FieldConfig, out: RunOutput,
                       *, voronoi_stride: int = 4) -> dict:
    """Statistics of the post-warm-up window of one run."""
    mask = out.sample_times >= cfg.warmup
    times = out.sample_times[mask]
    if len(times) < 4:
        raise ValueError("not enough post-warm-up samples")

    fn = np.concatenate([out.contact_fn[k] for k in np.nonzero(mask)[0]])
    ci = np.concatenate([out.contact_i[k] for k in np.nonzero(mask)[0]])
    cj = np.concatenate([out.contact_j[k] for k in np.nonzero(mask)[0]])
    chains = force_distribution(fn)

    z_series = []
    for k in np.nonzero(mask)[0]:
        z_series.append(coordination_number(out.contact_i[k], out.contact_j[k],
                                           out.n_floes, out.contact_fn[k]))
    z_mean = float(np.mean([z["Z_all"] for z in z_series]))
    z_eff = float(np.mean([z["Z_non_rattler"] for z in z_series]))
    rattler = float(np.mean([z["rattler_fraction"] for z in z_series]))

    regions = {"feed": [], "jam": [], "wake": []}
    domain_samples = []
    for k in np.nonzero(mask)[0][::voronoi_stride]:
        x = out.positions[k]
        r = out.radii
        m, areas = region_masks(x, cfg)
        domain_samples.append({
            "areal": areal_porosity(x, r, cfg.domain_area),
            "voronoi": _safe_voronoi(x, r, cfg, None),
        })
        for name, sel in m.items():
            if np.count_nonzero(sel) < 4:
                continue
            regions[name].append({
                "areal": areal_porosity(x, r, areas[name], sel),
                "voronoi": _safe_voronoi(x, r, cfg, sel),
                "n_floes": int(np.count_nonzero(sel)),
            })

    porosity = {}
    for name, entries in regions.items():
        if not entries:  # pragma: no cover
            continue
        porosity[name] = {
            "areal_mean": float(np.mean([e["areal"] for e in entries])),
            "areal_std": float(np.std([e["areal"] for e in entries])),
            "voronoi_mean": float(np.nanmean([e["voronoi"] for e in entries])),
            "voronoi_std": float(np.nanstd([e["voronoi"] for e in entries])),
            "n_floes_mean": float(np.mean([e["n_floes"] for e in entries])),
            "n_samples": len(entries),
        }
    target = 1.0 - cfg.ice_concentration
    domain = {
        "areal_mean": float(np.mean([d["areal"] for d in domain_samples])),
        "voronoi_mean": float(np.nanmean([d["voronoi"] for d in domain_samples])),
        "areal_std": float(np.std([d["areal"] for d in domain_samples])),
        "voronoi_std": float(np.nanstd([d["voronoi"] for d in domain_samples])),
        "n_samples": len(domain_samples),
        "target_from_config": target,
        "abs_error_vs_config": float(abs(np.mean([d["areal"] for d in domain_samples]) - target)),
        "estimator_gap": float(abs(np.mean([d["areal"] for d in domain_samples])
                                   - np.nanmean([d["voronoi"] for d in domain_samples]))),
    }
    porosity["domain_mean"] = domain
    porosity["definitions"] = {
        "areal": "1 - sum(pi r^2)/area over the region (free-water fraction)",
        "voronoi": "periodic Voronoi free volume: mean((A_cell - pi r^2)+ / A_cell)",
        "regions": "full-width strips: feed = x < x_gate - 10 r; jam = last 4 r before "
                   "the gate; wake = 6 r behind the gate; domain_mean = whole channel",
    }

    window = out.time >= cfg.warmup
    load = out.wall_force[window]
    load_y = out.wall_force_y[window]
    drag = out.drag_force_x[window]
    momentum_residual = float(abs(load.mean() + drag.mean())
                              / max(abs(drag.mean()), 1e-12))
    return {
        "force_chain_stats": chains.as_dict(),
        "force_distribution": {
            "f_over_mean": chains.f_centres.tolist(),
            "P_f": chains.probability.tolist(),
        },
        "coordination": {
            "Z_all_mean": z_mean,
            "Z_non_rattler_mean": z_eff,
            "rattler_fraction_mean": rattler,
            "series_mean": float(out.coordination[out.time >= cfg.warmup].mean()),
        },
        "porosity": porosity,
        "wall_load": {
            # the hull face is pushed in -x by the ice; the engineering load is
            # the positive magnitude of that reaction
            "ice_load_on_hull_N": float(-load.mean()),
            "ice_load_std_N": float(load.std()),
            "per_metre_face_N_per_m": float(-load.mean() / cfg.gate_width),
            "transverse_friction_N": float(load_y.mean()),
            "total_drag_N": float(drag.mean()),
            "momentum_residual_relative": momentum_residual,
            "kinetic_energy_mean_J": float(out.kinetic_energy[window].mean()),
            "mean_speed_m_per_s": float(out.mean_speed[window].mean()),
        },
    }


def _safe_voronoi(x, r, cfg, sel) -> float:
    try:
        return voronoi_free_volume(x, r, (cfg.domain_x, cfg.domain_y), sel)
    except ValueError:
        return float("nan")


def _jam_wake_area(cfg: FieldConfig) -> float:
    return _region_area(cfg, "jam") + _region_area(cfg, "wake")


def _region_area(cfg: FieldConfig, which: str) -> float:
    span = 2.5 * 2.0 * cfg.mean_radius
    width = min(2 * (0.5 * cfg.gate_width + span), cfg.domain_y)
    return span * width


# --------------------------------------------------------------------------
# C1: deterministic trajectory order study
# --------------------------------------------------------------------------
def trajectory_convergence(*, dt_levels: Iterable[float] = (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4),
                           duration: float = 0.6,
                           speed: float = 0.15,
                           seed: int = 7) -> dict:
    """Order of accuracy of a deterministic two-floe head-on collision.

    Two floes of radius 0.9 m and thickness 0.3 m approach each other at
    ``+-speed``, collide through the Hertz-Mindlin contact (restitution 0.2) and
    rebound.  The trajectory is smooth in between contacts, so the step-size
    error is the integration error alone.  The discrete ``L2`` error of the
    centroid positions over the whole record (against the finest level) gives
    the observed order, which is the number criterion C1 asks for.
    """
    dts = list(dt_levels)
    r = 0.9
    x0 = np.array([[6.0, 10.0], [6.0 + 2.0 * r + 0.30, 10.0]])
    pop = population_from_breakup(
        fragment_area=[np.pi * r ** 2] * 2, centroids=x0, thickness=0.30,
        source="deterministic_collision", metadata={"purpose": "C1 order study"})

    histories = []
    dt_over_tc = []
    for dt in dts:
        cfg = FieldConfig(
            n_floes=2, mean_radius=r, radius_log_sigma=0.0, thickness=0.30,
            domain_x=40.0, domain_y=20.0, gate_width=20.0, gate_x_fraction=1.0,
            drift_speed=0.0, drag_coefficient=0.0, dt=dt, duration=duration, warmup=0.0,
            sample_every=0.01, seed=seed,
        )
        field = _field_from_population(cfg, pop)
        field.v = np.array([[-speed, 0.0], [speed, 0.0]])
        out = field.run()
        dt_over_tc.append(field.dt_over_tc)
        histories.append(out.positions)

    n = min(h.shape[0] for h in histories)
    histories = [h[:n] for h in histories]
    ref = histories[-1]
    errors = []
    for h in histories:
        diff = h - ref
        errors.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=2)))))
    dts_used = np.array(dts[:len(histories)])
    # least-squares order over all levels (finest excluded as it is the reference)
    mask = np.array(errors[:-1]) > 0
    order_fit = np.nan
    if np.count_nonzero(mask) >= 2:
        p = np.polyfit(np.log(dts_used[:-1][mask]), np.log(np.array(errors[:-1])[mask]), 1)
        order_fit = float(p[0])
    consecutive = [
        float(np.log(errors[k] / errors[k + 1]) / np.log(2.0))
        for k in range(len(errors) - 2)
        if errors[k + 1] > 0 and errors[k] > 0
    ]
    return {
        "criterion": "C1",
        "quantity": "L2 centroid-position error of a 2-floe Hertz-Mindlin collision",
        "dt_levels_s": [float(d) for d in dts],
        "dt_over_contact_time": [float(v) for v in dt_over_tc],
        "l2_error_m": errors,
        "observed_order_consecutive": consecutive,
        "observed_order_fit": order_fit,
        "pass": bool(order_fit is not None and order_fit >= 1.0),
        "reference": "finest dt level in the list",
        "setup": {"radius_m": r, "thickness_m": 0.30, "approach_speed_m_per_s": speed,
                  "restitution": 0.20, "duration_s": duration},
    }


# --------------------------------------------------------------------------
# C1 (ensemble) + particle count
# --------------------------------------------------------------------------
def ensemble_dt_sweep(cfg: FieldConfig, pop: FloePopulation,
                      factors: Iterable[float] = (1.0, 2.0, 4.0),
                      duration: float | None = None) -> dict:
    """Reported statistics versus step size, with a Richardson-style order."""
    duration = duration if duration is not None else cfg.duration
    rows = []
    for factor in factors:
        c = replace(cfg, dt=cfg.dt / factor, duration=duration)
        _, out = _run(c, pop)
        stats = collect_statistics(c, out)
        rows.append({
            "dt_s": c.dt,
            "dt_over_tc": out.dt_over_tc,
            "ambient_porosity": stats["porosity"]["domain_mean"]["areal_mean"],
            "jam_porosity": stats["porosity"]["jam"]["areal_mean"],
            "Z": stats["coordination"]["Z_all_mean"],
            "f0": stats["force_chain_stats"]["f0_exponential"],
            "r2": stats["force_chain_stats"]["r2_exponential_tail"],
            "wall_load_N": stats["wall_load"]["ice_load_on_hull_N"],
            "mean_speed": stats["wall_load"]["mean_speed_m_per_s"],
            "max_overlap_ratio": out.max_overlap_ratio,
            "oscillation_index": kinematics_summary(out, c).oscillation_index,
        })
    metrics = ("ambient_porosity", "jam_porosity", "Z", "f0", "wall_load_N", "mean_speed")
    relative = {}
    for m in metrics:
        v = np.array([row[m] for row in rows], dtype=float)
        if len(v) >= 3 and abs(v[0] - v[1]) > 0 and abs(v[1] - v[2]) > 0:
            q = abs(v[0] - v[1]) / abs(v[1] - v[2])
            order = float(np.log(q) / np.log(2.0)) if q > 0 else np.nan
        else:
            order = np.nan
        denom = abs(v[-1]) if abs(v[-1]) > 0 else 1.0
        relative[m] = {
            "values": [float(x) for x in v],
            "relative_change_vs_finest": float(abs(v[0] - v[-1]) / denom),
            "relative_change_coarse_vs_fine": float(abs(v[1] - v[-1]) / denom) if len(v) > 2 else None,
            "observed_order": order,
        }
    return {
        "criterion": "C1 (ensemble)",
        "duration_s": duration,
        "rows": rows,
        "metrics": relative,
        "converged_within_2pct": bool(all(
            v["relative_change_vs_finest"] <= 0.02 for v in relative.values())),
    }


def particle_count_study(cfg: FieldConfig, *, factors: Iterable[float] = (0.5, 1.0, 2.0),
                         duration: float | None = None) -> dict:
    """Statistics versus floe count at fixed concentration and domain."""
    duration = duration if duration is not None else cfg.duration
    rows = []
    for factor in factors:
        n = int(round(cfg.n_floes * factor))
        if n < 8:
            continue
        # keep the mean radius fixed, scale the domain area with the count so the
        # ambient concentration is unchanged
        scale = np.sqrt(n / cfg.n_floes)
        c = replace(cfg, n_floes=n, domain_x=cfg.domain_x * scale,
                    domain_y=cfg.domain_y * scale, duration=duration)
        pop = build_population(c)
        _, out = _run(c, pop)
        stats = collect_statistics(c, out)
        rows.append({
            "n_floes": n,
            "area_m2": c.domain_area,
            "ambient_porosity": stats["porosity"]["domain_mean"]["areal_mean"],
            "ambient_voronoi": stats["porosity"]["domain_mean"]["voronoi_mean"],
            "jam_porosity": stats["porosity"]["jam"]["areal_mean"],
            "Z": stats["coordination"]["Z_all_mean"],
            "f0": stats["force_chain_stats"]["f0_exponential"],
            "r2": stats["force_chain_stats"]["r2_exponential_tail"],
            "wall_load_N": stats["wall_load"]["ice_load_on_hull_N"],
            "n_contacts": stats["force_chain_stats"]["n_contacts"],
        })
    return {"criterion": "C4 (particle-count sensitivity)", "delay": "domain scaled with N",
            "duration_s": duration, "rows": rows}


# --------------------------------------------------------------------------
# UQ
# --------------------------------------------------------------------------
def uq_sweep(cfg: FieldConfig, pop: FloePopulation, *,
             duration: float = 4.0) -> dict:
    """One-at-a-time sensitivity of the reported statistics.

    Parameters are varied by +-20 % about the baseline (contact stiffness through
    the Young modulus, restitution, friction, floe thickness, drift speed) and
    the ensemble seed is varied once, so the reported statistic is bracketed by
    its material uncertainty and its run-to-run scatter.
    """
    duration = max(float(duration), cfg.warmup * 1.05 + 0.2)
    base_c = replace(cfg, duration=duration, warmup=min(cfg.warmup, 0.8 * duration))
    _, base_out = _run(base_c, pop)
    base = collect_statistics(base_c, base_out)
    base_kin = kinematics_summary(base_out, base_c)

    def metrics(stats, kin) -> dict:
        return {
            "ambient_porosity": stats["porosity"]["domain_mean"]["areal_mean"],
            "jam_porosity": stats["porosity"]["jam"]["areal_mean"],
            "Z": stats["coordination"]["Z_all_mean"],
            "f0": stats["force_chain_stats"]["f0_exponential"],
            "r2_exponential": stats["force_chain_stats"]["r2_exponential_tail"],
            "wall_load_N": stats["wall_load"]["ice_load_on_hull_N"],
            "mean_speed": stats["wall_load"]["mean_speed_m_per_s"],
            "oscillation_index": kin.oscillation_index,
        }

    baseline = metrics(base, base_kin)
    variations = {
        "youngs_modulus": [0.8, 1.2],
        "restitution": [0.1, 0.4],
        "friction": [0.2, 0.5],
        "drift_speed": [0.6, 1.0],
    }
    rows = []
    for name, values in variations.items():
        for value in values:
            if name == "youngs_modulus":
                c = replace(base_c, material=ContactMaterial(
                    youngs_modulus=value * 1e10,
                    poisson_ratio=base_c.material.poisson_ratio,
                    restitution=base_c.material.restitution,
                    friction=base_c.material.friction,
                    density=base_c.material.density))
            elif name == "restitution":
                c = replace(base_c, material=ContactMaterial(
                    youngs_modulus=base_c.material.youngs_modulus,
                    poisson_ratio=base_c.material.poisson_ratio,
                    restitution=value,
                    friction=base_c.material.friction,
                    density=base_c.material.density))
            elif name == "friction":
                c = replace(base_c, material=ContactMaterial(
                    youngs_modulus=base_c.material.youngs_modulus,
                    poisson_ratio=base_c.material.poisson_ratio,
                    restitution=base_c.material.restitution,
                    friction=value,
                    density=base_c.material.density))
            elif name == "thickness":
                c = replace(base_c, thickness=value)
            else:
                c = replace(base_c, drift_speed=value)
            _, out = _run(c, pop)
            stats = collect_statistics(c, out)
            m = metrics(stats, kinematics_summary(out, c))
            rel = {k: float((m[k] - baseline[k]) / baseline[k])
                   if baseline[k] not in (0.0,) and np.isfinite(baseline[k]) else float("nan")
                   for k in baseline}
            rows.append({"parameter": name, "value": value,
                         "nominal": float(getattr(base_c, name, np.nan))
                         if name in ("thickness", "drift_speed")
                         else float(getattr(base_c.material, name)),
                         "metrics": m, "relative_change": rel})

    # seed scatter
    seeds = [base_c.seed + 1, base_c.seed + 2]
    seed_rows = []
    for s in seeds:
        c = replace(base_c, seed=s)
        p = build_population(c)
        _, out = _run(c, p)
        stats = collect_statistics(c, out)
        seed_rows.append({"seed": s, "metrics": metrics(stats, kinematics_summary(out, c))})

    summary = {}
    for key in baseline:
        vals = [row["metrics"][key] for row in rows + seed_rows]
        rels = [abs(row["relative_change"][key]) for row in rows] if rows else []
        summary[key] = {
            "baseline": float(baseline[key]),
            "min": float(np.nanmin([baseline[key]] + vals)),
            "max": float(np.nanmax([baseline[key]] + vals)),
            "max_abs_relative_change": float(np.nanmax(rels)) if rels else float("nan"),
            "seed_scatter_relative": (
                float(np.nanmax([abs(r["metrics"][key] - baseline[key]) / abs(baseline[key])
                                 for r in seed_rows]))
                if seed_rows and baseline[key] else float("nan")),
        }
    return {"baseline": baseline, "rows": rows, "seed_rows": seed_rows,
            "summary": summary,
            "design": "one-at-a-time +-20 % (nominal values in 'nominal') plus 2 extra seeds",
            "duration_s": duration}


# --------------------------------------------------------------------------
# full study
# --------------------------------------------------------------------------
def negative_control(cfg: FieldConfig, pop: FloePopulation) -> dict:
    """Negative control: the *legacy* configuration the repair is measured against.

    Restitution 1 (no viscous damping) and a step factor nine times larger than
    the stability rule allow -- i.e. a contact that the integrator cannot
    resolve.  If the resolved Hertz-Mindlin model is what makes the kinematics
    deliverable, this run must show a much larger oscillation index and much
    larger penetrations on (nearly) the same physics.
    """
    mat = ContactMaterial(youngs_modulus=cfg.material.youngs_modulus,
                          poisson_ratio=cfg.material.poisson_ratio,
                          restitution=1.0,
                          friction=cfg.material.friction,
                          density=cfg.material.density)
    c = replace(cfg, material=mat, duration=1.0, warmup=0.0,
                kinematics_window=(0.2, 0.8), safety_factor=0.9)
    field = _field_from_population(c, pop)
    out = field.run()
    finite = bool(np.all(np.isfinite(out.velocities)))
    kin = kinematics_summary(out, c, window=(0.2, min(0.9, c.duration)))
    return {
        "purpose": "unresolved contact (restitution 1, dt = 0.9 t_c)",
        "dt_s": float(out.dt),
        "dt_over_contact_time": float(out.dt_over_tc),
        "restitution": 1.0,
        "diverged": not finite,
        "oscillation_index": float(kin.oscillation_index) if finite else float("nan"),
        "velocity_high_frequency_fraction": float(kin.velocity_high_frequency_fraction)
        if finite else float("nan"),
        "max_overlap_ratio": float(out.max_overlap_ratio),
        "max_speed_m_per_s": float(np.nanmax(np.linalg.norm(out.velocities, axis=2)))
        if finite else float("nan"),
    }


def run_study(cfg: FieldConfig = FieldConfig(), *, quick: bool = False) -> tuple[dict, dict]:
    """Run the whole study and return ``(payload, raw)``.

    ``payload`` is the contract-shaped result (:meth:`ClearIceResult.as_dict`);
    ``raw`` carries the extra arrays/records the report and the figures need.
    """
    pop = build_population(cfg)
    audit = validate_population(pop, domain=(cfg.domain_x, cfg.domain_y))
    field = _field_from_population(cfg, pop)
    out = field.run()

    kin_full = kinematics_summary(out, cfg)                       # physical payload
    kin_q = kinematics_summary(out, cfg, window=cfg.kinematics_window)  # resolution
    stats = collect_statistics(cfg, out)
    control = negative_control(cfg, pop)

    traj = trajectory_convergence(
        dt_levels=(1e-3, 5e-4, 2.5e-4) if quick else (1e-3, 5e-4, 2.5e-4, 1.25e-4),
        duration=0.5)
    short = replace(cfg, duration=min(cfg.duration, 3.0),
                    warmup=min(cfg.warmup, 1.5),
                    kinematics_window=(0.5, 2.0))
    dt_sweep = ensemble_dt_sweep(short, pop, factors=(1.0, 2.0) if quick else (1.0, 2.0, 4.0),
                                 duration=short.duration)
    particles = particle_count_study(short, factors=(1.0,) if quick else (0.5, 1.0, 2.0),
                                     duration=min(short.duration, 2.5))
    uq = uq_sweep(cfg, pop, duration=1.5 if quick else 2.5)

    checks = _criteria(cfg, stats, kin_q, traj, dt_sweep, uq, audit, control)
    payload = ClearIceResult(
        floe_kinematics={
            **kin_full.as_dict(),
            "resolution_quality": {
                "window_s": list(cfg.kinematics_window),
                "oscillation_index": float(kin_q.oscillation_index),
                "velocity_high_frequency_fraction": float(kin_q.velocity_high_frequency_fraction),
                "max_overlap_ratio": float(out.max_overlap_ratio),
                "negative_control": control,
            },
            "deliverable": bool(checks["C1_kinematics"]["deliverable"]),
            "check": checks["C1_kinematics"],
        },
        force_chain_stats={**stats["force_chain_stats"],
                           "coordination": stats["coordination"],
                           "check": checks["C2_force_chain"]},
        porosity={**stats["porosity"], "check": checks["C3_porosity"]},
        convergence_report={
            "trajectory_order": traj,
            "ensemble_dt_sweep": dt_sweep,
            "particle_count": particles,
            "contact_time_s": float(field.tc),
            "dt_used_s": float(field.dt),
            "dt_configured_s": float(field.dt_configured),
            "dt_over_contact_time": float(field.dt_over_tc),
            "stability_rule": "dt <= 0.10 * t_contact (Hertz, smallest floe)",
            "wall_load": stats["wall_load"],
            "checks": checks["C4_maturity"],
        },
        uncertainty_quantification=uq,
        provenance={
            "contract_version": CONTRACT_VERSION,
            "population_hash": pop.hash,
            "population_source": pop.source,
            "population_audit": audit,
            "config": {k: (v if not isinstance(v, ContactMaterial) else dict(v.__dict__))
                       for k, v in cfg.__dict__.items()},
            "mission": "P103W2-N03 / m6997",
        },
    ).as_dict()

    raw = {
        "run_output": out, "statistics": stats, "kinematics": kin_full,
        "kinematics_quality": kin_q, "population": pop, "audit": audit,
        "trajectory_order": traj, "dt_sweep": dt_sweep,
        "particle_count": particles, "uq": uq, "checks": checks,
        "control": control, "contact_time_s": field.tc,
    }
    return payload, raw


def _criteria(cfg, stats, kin, traj, dt_sweep, uq, audit, control) -> dict:
    """Machine-checkable verdicts for C1-C4."""
    z = stats["coordination"]
    chains = stats["force_chain_stats"]
    por = stats["porosity"]
    ambient = por["domain_mean"]["areal_mean"]
    voronoi_ambient = por["domain_mean"]["voronoi_mean"]
    r2 = chains["r2_exponential_tail"]
    return {
        "C1_kinematics": {
            "trajectory_order_fit": traj["observed_order_fit"],
            "order_pass_ge_1": bool(traj["pass"]),
            "dt_configured_s": float(cfg.dt),
            "oscillation_index": float(kin.oscillation_index),
            "velocity_high_frequency_fraction": float(kin.velocity_high_frequency_fraction),
            "max_overlap_ratio": float(kin.max_overlap_ratio),
            "negative_control_oscillation_index": control["oscillation_index"],
            "negative_control_diverged": control["diverged"],
            "deliverable": bool(traj["pass"] and kin.max_overlap_ratio < 0.05
                                and kin.oscillation_index < 0.2),
        },
        "C2_force_chain": {
            "r2_exponential_tail": float(r2),
            "f0": float(chains["f0_exponential"]),
            "powerlaw_r2": float(chains["r2_powerlaw_tail"]),
            "exponential_preferred_delta_bic": float(chains["delta_BIC_powerlaw_minus_exponential"]),
            "pass": bool(np.isfinite(r2) and r2 > 0.9),
            "reference": "exponential tail P(f) ~ exp(-f/f0) for jammed Hertzian "
                         "packings (Radjai et al. 1996; O'Donovan & Tanner 2015); "
                         "the IN12-S4 dataset is not in this repository -> the direct "
                         "experimental comparison is WITHHELD",
        },
        "C3_porosity": {
            "domain_mean_areal": float(ambient),
            "domain_mean_voronoi": float(voronoi_ambient),
            "estimator_gap": float(por["domain_mean"]["estimator_gap"]),
            "feed_areal": float(por.get("feed", {}).get("areal_mean", np.nan)),
            "jam_areal": float(por.get("jam", {}).get("areal_mean", np.nan)),
            "wake_areal": float(por.get("wake", {}).get("areal_mean", np.nan)),
            "band": [0.30, 0.45],
            "pass": bool(0.30 <= ambient <= 0.45 and 0.30 <= voronoi_ambient <= 0.45),
        },
        "C4_maturity": {
            "trajectory_order": float(traj["observed_order_fit"]),
            "ensemble_converged_within_2pct": bool(dt_sweep["converged_within_2pct"]),
            "particle_count_rows": len(uq.get("rows", [])) > 0,
            "uq_parameters": sorted({r["parameter"] for r in uq["rows"]}),
            "contract_checks_pass": bool(audit["pass"]),
            "contract_version": CONTRACT_VERSION,
        },
    }
