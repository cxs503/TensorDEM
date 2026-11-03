#!/usr/bin/env python3
"""Supplementary experiment tables for ``paper-draft-tensordem-contact-friction``.

Five machine-readable tables, each with an independent closed-form reference and a
per-point relative error (see ``benchmarks/verified/dem3d_contact_friction/README.md``):

* ``restitution_kn_e_sweep.csv``   -- 2-D solver restitution sweep over ``k_n`` x ``e``
* ``rolling_friction_kn_e_sweep.csv`` -- 2-D realised ``mu_r`` / ``mu`` sweep (incline harness)
* ``friction_kernel_conformance.csv`` -- Cundall-Strack Coulomb clamp conformance in ``IceDEM3D``
* ``hertz_convergence.csv``        -- Hertz contact kernel: 3 contact-scale levels x 5 dt
* ``buoyancy_cap_convergence.csv`` -- spherical-cap buoyancy kernel: 3 mesh levels x 5 dt

Reproduce (repository root)::

    PYTHONPATH=src python scripts/benchmark_contact_friction_tables.py

Nothing is transcribed by hand: every number in the CSV/JSON output is written by this
script from the solver state or from a closed form evaluated here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import torch

from tensordem import buoyancy_wet3d, restitution, rolling_resistance
from tensordem.dem3d import DEM3DConfig, IceDEM3D

OUT_DEFAULT = Path("benchmarks/verified/dem3d_contact_friction")

K_N_GRID = (2.0e4, 5.0e4, 1.0e5, 5.0e5, 1.0e6)   # N/m
E_GRID = (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)        # target restitution
RADIUS = 0.05                                     # m
DENSITY = 900.0                                   # kg/m^3
V0 = 0.5                                          # m/s closing half-speed

INCLINE_PARAMS = ((0.40, 0.05), (0.05, 0.01))     # (mu, mu_r)
INCLINE_ANGLES = (2.0, 4.0, 10.0, 20.0)           # deg

HERTZ_LEVELS = (("soft_E100MPa", 0.10e9), ("ice_E1GPa", 1.00e9), ("stiff_E10GPa", 10.0e9))
HERTZ_STEPS_PER_TC = (10.0, 20.0, 40.0, 80.0, 160.0)
_HERTZ_TIME_COEFF = 2.0 * math.gamma(0.4) * math.gamma(0.5) / (2.5 * math.gamma(0.9))  # exact T_c coefficient, see README

BUOYANCY_RADIUS = 0.5
BUOYANCY_DENSITY = 900.0
WATER_DENSITY = 1025.0
GRAVITY = 9.81
BUOYANCY_MESH = (4, 6, 8)
BUOYANCY_STEPS_PER_PERIOD = (50.0, 100.0, 200.0, 400.0, 800.0)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _fmt(value):
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.12g}"
    return "" if value is None else str(value)


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in fieldnames})


def rel_err(measured: float, reference: float) -> float:
    if reference == 0.0:
        return math.nan if measured != 0.0 else 0.0
    return measured / reference - 1.0


def bilateral_zeta(e_target: float) -> float:
    """zeta of the bilateral mapping ``IceDEM3D.forces`` applies internally."""
    abs_ln = abs(math.log(e_target))
    return abs_ln / math.sqrt(math.pi ** 2 + abs_ln ** 2)


def engine_damping(k_n: float, e_target: float, m_eff: float) -> float:
    """``gamma_n = 2 zeta sqrt(k_n m_eff)`` exactly as the solver builds it."""
    return 2.0 * bilateral_zeta(e_target) * math.sqrt(k_n * m_eff)


# ---------------------------------------------------------------------------
# 1. Solver restitution sweep
# ---------------------------------------------------------------------------
def _solver_pair(k_n: float, e_target: float, *, radius: float = RADIUS,
                 density: float = DENSITY, dt_scale: float = 1.0):
    """Two free particles facing each other, driven through ``restitution_coefficient``."""
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=radius, density=density,
        contact_stiffness=k_n, contact_damping=0.0, drag=0.0,
        bond_stiffness=1.0, breaking_strain=1e-12, shear_breaking_strain=1e-12,
        fix_x_edges=False, fix_bottom=False, tool_gap=1.0e3, tool_speed=1.0e-9,
        restitution_coefficient=e_target,
    )
    dem = IceDEM3D(cfg)
    n = len(dem.positions)
    pos = torch.zeros_like(dem.positions)
    for i in range(n):
        pos[i] = torch.tensor([10.0 * i, 500.0, 500.0], dtype=dem.dtype)
    pos[0] = torch.tensor([0.0, 0.0, -(radius + 1.0e-6)], dtype=dem.dtype)
    pos[1] = torch.tensor([0.0, 0.0, radius + 1.0e-6], dtype=dem.dtype)
    dem.positions = pos.clone()
    dem.initial_positions = pos.clone()
    fixed = torch.ones(n, dtype=torch.bool)
    fixed[0] = False
    fixed[1] = False
    dem.fixed = fixed
    dem.forces()
    if int(dem.alive.sum().item()) != 0:
        raise RuntimeError("bonded lattice did not fully break")
    dem.dt = dem.dt * dt_scale
    return dem, 0, 1


def solver_restitution(k_n: float, e_target: float, *, v0: float = V0,
                       radius: float = RADIUS, density: float = DENSITY) -> dict:
    dem, i, j = _solver_pair(k_n, e_target, radius=radius, density=density)
    dem.velocities = torch.zeros_like(dem.positions)
    dem.velocities[i] = torch.tensor([0.0, 0.0, v0], dtype=dem.dtype)
    dem.velocities[j] = torch.tensor([0.0, 0.0, -v0], dtype=dem.dtype)
    dem.time = 0.0
    dem.step_count = 0
    momentum_before = float(dem.velocities[:, 2].sum())
    contact_steps = 0
    max_overlap = 0.0
    for _ in range(400_000):
        dem.step()
        overlap = 2.0 * radius - float(dem.positions[j, 2] - dem.positions[i, 2])
        if overlap > 0.0:
            contact_steps += 1
            max_overlap = max(max_overlap, overlap)
        elif contact_steps:
            break
    else:  # pragma: no cover
        raise RuntimeError("pair did not separate")
    return {
        "e_measured": -float(dem.velocities[i, 2]) / v0,
        "contact_steps": contact_steps,
        "max_overlap_m": max_overlap,
        "dt_s": dem.dt,
        "momentum_drift": float(dem.velocities[:, 2].sum()) - momentum_before,
    }


def restitution_sweep() -> list[dict]:
    mass = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    m_eff = 0.5 * mass
    rows = []
    for k_n in K_N_GRID:
        for e_target in E_GRID:
            meas = solver_restitution(k_n, e_target)
            c_engine = engine_damping(k_n, e_target, m_eff)
            e_ref_clipped = restitution.unilateral_restitution(c_engine, k_n, m_eff, V0)
            zeta = bilateral_zeta(e_target)
            e_ref_bilateral = math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))
            c_corrected = restitution.damping_for_restitution(e_target, k_n, m_eff, V0)
            period = 2.0 * math.pi / math.sqrt(k_n / m_eff)
            rows.append({
                "case_id": f"kn{k_n:.3g}_e{e_target:.3g}",
                "k_n_N_per_m": k_n,
                "e_target": e_target,
                "zeta_bilateral": zeta,
                "damping_c_engine_Ns_per_m": c_engine,
                "damping_c_corrected_Ns_per_m": c_corrected,
                "damping_correction_ratio": c_corrected / c_engine if c_engine else math.nan,
                "dt_s": meas["dt_s"],
                "elastic_period_s": period,
                "steps_per_elastic_period": period / meas["dt_s"],
                "contact_steps": meas["contact_steps"],
                "max_overlap_m": meas["max_overlap_m"],
                "e_measured": meas["e_measured"],
                "e_ref_clipped_law": e_ref_clipped,
                "e_ref_bilateral_formula": e_ref_bilateral,
                "rel_err_vs_clipped_law": rel_err(meas["e_measured"], e_ref_clipped),
                "rel_err_vs_target": rel_err(meas["e_measured"], e_target),
                "momentum_drift_kgm_per_s": meas["momentum_drift"],
            })
    return rows


# ---------------------------------------------------------------------------
# 2. Rolling-resistance / friction sweep on the incline harness
# ---------------------------------------------------------------------------
def incline_sweep() -> list[dict]:
    rows = []
    for k_n in K_N_GRID:
        base = rolling_resistance.SphereInclineConfig(radius=RADIUS, density=DENSITY, k_n=k_n)
        mass = base.mass
        gamma_n = engine_damping(k_n, e_target=E_GRID[0], m_eff=mass)
        dt = 2.0 * math.pi / math.sqrt(k_n / mass) / 60.0   # 60 steps per elastic period
        for e_target in E_GRID:
            gamma_n = engine_damping(k_n, e_target, mass)
            for mu, mu_r in INCLINE_PARAMS:
                cfg = rolling_resistance.SphereInclineConfig(
                    radius=RADIUS, density=DENSITY, k_n=k_n, gamma_n=gamma_n,
                    mu=mu, k_t=None, dt=dt, gravity=GRAVITY)
                rolling = rolling_resistance.RollingResistanceConfig(model="cdt", mu_r=mu_r)
                duration = 0.10
                result = rolling_resistance.simulate_sphere_on_incline(
                    INCLINE_ANGLES, rolling, config=cfg,
                    duration=duration, settle_time=0.5 * duration)
                acc, r_squared = rolling_resistance.fit_acceleration(result)
                inertia_factor = 1.0 / base.inertia_factor
                for idx, angle in enumerate(INCLINE_ANGLES):
                    a_meas = float(acc[idx])
                    rolling_ok = rolling_resistance.pure_rolling_feasible(angle, mu_r, mu)
                    if rolling_ok:
                        a_ref = rolling_resistance.incline_rolling_acceleration(angle, mu_r)
                        regime = "rolling"
                    else:
                        a_ref = rolling_resistance.incline_sliding_acceleration(angle, mu)
                        regime = "sliding"
                    theta = math.radians(angle)
                    mu_r_eff = math.tan(theta) - a_meas * (1.0 + inertia_factor) / (
                        GRAVITY * math.cos(theta))
                    mu_eff = math.tan(theta) - a_meas / (GRAVITY * math.cos(theta))
                    rows.append({
                        "case_id": (f"kn{k_n:.3g}_e{e_target:.3g}_mu{mu:.2g}"
                                    f"_mr{mu_r:.2g}_a{angle:g}"),
                        "k_n_N_per_m": k_n,
                        "e_target": e_target,
                        "damping_c_Ns_per_m": gamma_n,
                        "dt_s": dt,
                        "steps_per_elastic_period": 60.0,
                        "mu_declared": mu,
                        "mu_r_declared": mu_r,
                        "angle_deg": angle,
                        "critical_angle_deg": rolling_resistance.critical_incline_angle_deg(mu_r),
                        "regime": regime,
                        "a_measured_m_per_s2": a_meas,
                        "a_reference_m_per_s2": a_ref,
                        "rel_err_a": rel_err(a_meas, a_ref),
                        "mu_r_eff": mu_r_eff,
                        "rel_err_mu_r": rel_err(mu_r_eff, mu_r) if regime == "rolling" else math.nan,
                        "mu_eff": mu_eff,
                        "rel_err_mu": rel_err(mu_eff, mu) if regime == "sliding" else math.nan,
                        "fit_r_squared": float(r_squared[idx]),
                    })
    return rows


# ---------------------------------------------------------------------------
# 3. Coulomb clamp conformance inside IceDEM3D
# ---------------------------------------------------------------------------
def friction_kernel_rows() -> list[dict]:
    rows = []
    overlap = 2.0e-4
    v_tangential = 5.0
    for k_n in (5.0e4, 5.0e5, 1.0e6):
        for mu in (0.1, 0.3, 0.6):
            cfg = DEM3DConfig(
                nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
                contact_stiffness=k_n, contact_damping=0.0, drag=0.0,
                bond_stiffness=1.0, breaking_strain=1e-12, shear_breaking_strain=1e-12,
                fix_x_edges=False, fix_bottom=False, tool_gap=1.0e3, tool_speed=1.0e-9,
                friction_coefficient=mu)
            dem = IceDEM3D(cfg)
            n = len(dem.positions)
            pos = torch.zeros_like(dem.positions)
            for i in range(n):
                pos[i] = torch.tensor([10.0 * i, 500.0, 500.0], dtype=dem.dtype)
            half = 0.5 * (2.0 * RADIUS - overlap)
            pos[0] = torch.tensor([0.0, 0.0, -half], dtype=dem.dtype)
            pos[1] = torch.tensor([0.0, 0.0, half], dtype=dem.dtype)
            dem.positions = pos.clone()
            dem.initial_positions = pos.clone()
            fixed = torch.ones(n, dtype=torch.bool)
            fixed[0] = False
            fixed[1] = False
            dem.fixed = fixed
            dem.forces()
            dem.velocities = torch.zeros_like(dem.positions)
            dem.velocities[1, 0] = v_tangential
            force, _ = dem.forces(update_fracture=False)
            normal = torch.tensor([0.0, 0.0, 1.0], dtype=dem.dtype)
            f_i = force[0]
            f_n = float((f_i * normal).sum())
            f_t = float(torch.linalg.vector_norm(f_i - f_n * normal))
            ratio = f_t / abs(f_n) if f_n else math.nan
            rows.append({
                "k_n_N_per_m": k_n,
                "mu_declared": mu,
                "overlap_m": overlap,
                "tangential_speed_m_per_s": v_tangential,
                "F_n_N": abs(f_n),
                "F_t_N": f_t,
                "F_t_over_F_n": ratio,
                "rel_err_vs_mu": rel_err(ratio, mu),
            })
    return rows


# ---------------------------------------------------------------------------
# 4. Hertz contact-kernel convergence
# ---------------------------------------------------------------------------
def hertz_closed_form(e_star: float, r_star: float, m_eff: float, v0: float) -> dict:
    """Classical elastic Hertzian impact: ``K=4/3 E* sqrt(R*)``, energy balance."""
    k_hertz = (4.0 / 3.0) * e_star * math.sqrt(r_star)
    d_max = (5.0 * m_eff * v0 ** 2 / (4.0 * k_hertz)) ** 0.4
    return {
        "K": k_hertz,
        "d_max": d_max,
        "F_max": k_hertz * d_max ** 1.5,
        "t_c": _HERTZ_TIME_COEFF * (15.0 * m_eff / (16.0 * e_star * math.sqrt(r_star))) ** 0.4 * v0 ** -0.2,
    }


def hertz_integrate(k_hertz: float, m_eff: float, v0: float, dt: float) -> dict:
    """Semi-implicit Euler on ``m d'' = -K d^(3/2)`` (same update the DEM solver uses)."""
    d, v, t = 0.0, v0, 0.0
    peak_d = 0.0
    peak_f = 0.0
    contact_time = math.nan
    for _ in range(50_000_000):
        force = k_hertz * d ** 1.5 if d > 0.0 else 0.0
        if force > peak_f:
            peak_f, peak_d = force, d
        v_new = v - force / m_eff * dt
        d_new = d + v_new * dt
        if d_new <= 0.0 and v_new < 0.0:
            frac = d / (d - d_new) if d > d_new else 0.0
            contact_time = t + frac * dt
            break
        d, v, t = d_new, v_new, t + dt
    else:  # pragma: no cover
        raise RuntimeError("Hertz impact did not separate")
    return {"d_max": peak_d, "F_max": peak_f, "t_c": contact_time}


def hertz_rows() -> tuple[list[dict], dict]:
    rows = []
    meta = {}
    mass = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    m_eff = 0.5 * mass
    r_star = RADIUS / 2.0
    for label, e_star in HERTZ_LEVELS:
        ref = hertz_closed_form(e_star, r_star, m_eff, V0)
        level_rows = []
        for spc in HERTZ_STEPS_PER_TC:
            dt = ref["t_c"] / spc
            res = hertz_integrate(ref["K"], m_eff, V0, dt)
            level_rows.append({
                "level": label,
                "E_star_Pa": e_star,
                "R_star_m": r_star,
                "m_eff_kg": m_eff,
                "impact_speed_m_per_s": V0,
                "steps_per_contact_time": spc,
                "dt_s": dt,
                "d_max_m": res["d_max"],
                "d_max_ref_m": ref["d_max"],
                "rel_err_d_max": rel_err(res["d_max"], ref["d_max"]),
                "F_max_N": res["F_max"],
                "F_max_ref_N": ref["F_max"],
                "rel_err_F_max": rel_err(res["F_max"], ref["F_max"]),
                "t_c_s": res["t_c"],
                "t_c_ref_s": ref["t_c"],
                "rel_err_t_c": rel_err(res["t_c"], ref["t_c"]),
            })
        rows.extend(level_rows)
        coarse = [r for r in level_rows if r["steps_per_contact_time"] == 80.0][0]
        fine = level_rows[-1]
        meta[label] = {
            "observed_order_d_max": (
                math.log(abs(coarse["rel_err_d_max"]) / abs(fine["rel_err_d_max"])) / math.log(2.0)
                if fine["rel_err_d_max"] else math.nan),
            "contact_time_coefficient_measured": fine["t_c_s"] / (
                (m_eff ** 2 / (r_star * e_star ** 2 * V0)) ** 0.2),
            "t_c_ref_s": ref["t_c"],
        }
    return rows, meta


# ---------------------------------------------------------------------------
# 5. Spherical-cap buoyancy kernel convergence
# ---------------------------------------------------------------------------
def floe_cloud(radius: float, n_edge: int, density: float) -> dict:
    """Particle cloud discretising one sphere, mass-corrected particle radius."""
    spacing = 2.0 * radius / n_edge
    centres = []
    for k in range(n_edge):
        for j in range(n_edge):
            for i in range(n_edge):
                x = -radius + (i + 0.5) * spacing
                y = -radius + (j + 0.5) * spacing
                z = -radius + (k + 0.5) * spacing
                if x * x + y * y + z * z <= radius * radius:
                    centres.append((x, y, z))
    n_occ = len(centres)
    volume_sphere = 4.0 / 3.0 * math.pi * radius ** 3
    r_p = (volume_sphere / (n_occ * 4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
    volume_particle = 4.0 / 3.0 * math.pi * r_p ** 3
    return {
        "positions": torch.tensor(centres, dtype=torch.float64),
        "n_particles": n_occ,
        "r_p": r_p,
        "particle_volume": volume_particle,
        "total_volume": n_occ * volume_particle,
        "mass": density * volume_sphere,
        "volume_sphere": volume_sphere,
    }


def cloud_buoyancy(cloud: dict, centre_height: float,
                   water_density: float = WATER_DENSITY) -> float:
    h = cloud["positions"][:, 1] + centre_height
    fractions = buoyancy_wet3d.submerged_fraction_sphere_tensor(h, float(cloud["r_p"]))
    return water_density * GRAVITY * float(fractions.sum()) * cloud["particle_volume"]


def heave_test(cloud: dict, dt: float, h_eq: float, period_ref: float,
               periods: float = 6.0) -> dict:
    """Integrate ``m x'' = F_b(x) - m g - c x'``; measure draft and heave period."""
    omega_ref = 2.0 * math.pi / period_ref
    damping = 2.0 * 0.175 * cloud["mass"] * omega_ref
    x = h_eq + 0.02 * BUOYANCY_RADIUS
    v = 0.0
    t = 0.0
    crossings = []
    prev = x - h_eq
    for _ in range(int(round(periods * period_ref / dt))):
        force = cloud_buoyancy(cloud, x) - cloud["mass"] * GRAVITY - damping * v
        v += force / cloud["mass"] * dt
        x += v * dt
        t += dt
        excess = x - h_eq
        if prev <= 0.0 < excess:
            crossings.append(t - dt * excess / (excess - prev))
        prev = excess
    period = ((crossings[-1] - crossings[0]) / (len(crossings) - 1)
              if len(crossings) > 1 else math.nan)
    return {"period": period, "draft": BUOYANCY_RADIUS - x}


def buoyancy_rows() -> list[dict]:
    rows = []
    h_eq = buoyancy_wet3d.equilibrium_centre_height(
        BUOYANCY_RADIUS, BUOYANCY_DENSITY, WATER_DENSITY)
    volume_ref = buoyancy_wet3d.submerged_volume_sphere(h_eq, BUOYANCY_RADIUS)
    force_ref = WATER_DENSITY * GRAVITY * volume_ref
    stiffness_ref = WATER_DENSITY * GRAVITY * buoyancy_wet3d.waterplane_area_sphere(
        h_eq, BUOYANCY_RADIUS)
    for n_edge in BUOYANCY_MESH:
        cloud = floe_cloud(BUOYANCY_RADIUS, n_edge, BUOYANCY_DENSITY)
        force_meas = cloud_buoyancy(cloud, h_eq)
        stiffness_num = -(cloud_buoyancy(cloud, h_eq + 1e-4)
                          - cloud_buoyancy(cloud, h_eq - 1e-4)) / 2e-4
        period_ref = 2.0 * math.pi / math.sqrt(stiffness_ref / cloud["mass"])
        for spc in BUOYANCY_STEPS_PER_PERIOD:
            dt = period_ref / spc
            res = heave_test(cloud, dt, h_eq, period_ref)
            rows.append({
                "mesh_particles_per_edge": n_edge,
                "particles": cloud["n_particles"],
                "particle_radius_m": cloud["r_p"],
                "total_volume_over_sphere_volume": cloud["total_volume"] / cloud["volume_sphere"],
                "steps_per_heave_period": spc,
                "dt_s": dt,
                "buoyancy_at_equilibrium_N": force_meas,
                "buoyancy_ref_N": force_ref,
                "rel_err_buoyancy": rel_err(force_meas, force_ref),
                "stiffness_numeric_N_per_m": stiffness_num,
                "stiffness_ref_N_per_m": stiffness_ref,
                "rel_err_stiffness": rel_err(stiffness_num, stiffness_ref),
                "draft_measured_m": res["draft"],
                "draft_ref_m": BUOYANCY_RADIUS - h_eq,
                "rel_err_draft": rel_err(res["draft"], BUOYANCY_RADIUS - h_eq),
                "heave_period_measured_s": res["period"],
                "heave_period_ref_s": period_ref,
                "rel_err_heave_period": rel_err(res["period"], period_ref),
            })
    return rows


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def _worst(rows, key):
    finite = [r for r in rows if math.isfinite(r[key])]
    return max(finite, key=lambda r: abs(r[key])) if finite else None


def summarise(restitution_rows, incline_rows, friction_rows, hertz_table,
              hertz_meta, buoyancy_table) -> dict:
    worst_clipped = _worst(restitution_rows, "rel_err_vs_clipped_law")
    worst_target = _worst(restitution_rows, "rel_err_vs_target")
    worst_a = _worst(incline_rows, "rel_err_a")
    worst_mr = _worst(incline_rows, "rel_err_mu_r")
    worst_mu = _worst(incline_rows, "rel_err_mu")
    hertz_finest = [r for r in hertz_table if r["steps_per_contact_time"] == 160.0]
    worst_hertz = max(hertz_finest, key=lambda r: abs(r["rel_err_F_max"]))
    buoy_finest = [r for r in buoyancy_table if r["steps_per_heave_period"] == 800.0]
    worst_buoy = max(buoy_finest, key=lambda r: abs(r["rel_err_buoyancy"]))
    return {
        "commit": git_commit(),
        "command": "PYTHONPATH=src python scripts/benchmark_contact_friction_tables.py",
        "torch_version": torch.__version__,
        "restitution_max_abs_err_vs_clipped_law": abs(worst_clipped["rel_err_vs_clipped_law"]),
        "restitution_worst_case": worst_clipped["case_id"],
        "restitution_max_abs_err_vs_target": abs(worst_target["rel_err_vs_target"]),
        "restitution_target_worst_case": worst_target["case_id"],
        "restitution_within_3pct_vs_clipped_law": abs(worst_clipped["rel_err_vs_clipped_law"]) < 0.03,
        "restitution_within_3pct_vs_target": abs(worst_target["rel_err_vs_target"]) < 0.03,
        "restitution_damping_correction_ratio_range": [
            min(r["damping_correction_ratio"] for r in restitution_rows),
            max(r["damping_correction_ratio"] for r in restitution_rows)],
        "incline_max_abs_err_a": abs(worst_a["rel_err_a"]) if worst_a else None,
        "incline_worst_case": worst_a["case_id"] if worst_a else None,
        "incline_max_abs_err_mu_r": abs(worst_mr["rel_err_mu_r"]) if worst_mr else None,
        "incline_max_abs_err_mu": abs(worst_mu["rel_err_mu"]) if worst_mu else None,
        "incline_within_3pct_a": (abs(worst_a["rel_err_a"]) < 0.03) if worst_a else None,
        "friction_kernel_max_abs_err": max(abs(r["rel_err_vs_mu"]) for r in friction_rows),
        "hertz_observed_order_d_max": {k: v["observed_order_d_max"] for k, v in hertz_meta.items()},
        "hertz_contact_time_coefficient_measured": {
            k: v["contact_time_coefficient_measured"] for k, v in hertz_meta.items()},
        "hertz_max_abs_err_F_max_at_160_steps": abs(worst_hertz["rel_err_F_max"]),
        "hertz_within_3pct_F_max_at_160_steps": abs(worst_hertz["rel_err_F_max"]) < 0.03,
        "hertz_within_3pct_F_max_at_80_steps": all(
            abs(r["rel_err_F_max"]) < 0.03 for r in hertz_table
            if r["steps_per_contact_time"] == 80.0),
        "buoyancy_max_abs_err_at_800_steps": abs(worst_buoy["rel_err_buoyancy"]),
        "buoyancy_max_abs_err_over_all": max(abs(r["rel_err_buoyancy"]) for r in buoyancy_table),
        "buoyancy_period_max_abs_err": max(abs(r["rel_err_heave_period"]) for r in buoyancy_table),
        "buoyancy_within_3pct": abs(worst_buoy["rel_err_buoyancy"]) < 0.03,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--skip-incline", action="store_true")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    restitution_rows = restitution_sweep()
    write_csv(out / "restitution_kn_e_sweep.csv", list(restitution_rows[0]), restitution_rows)

    incline_rows = []
    if not args.skip_incline:
        incline_rows = incline_sweep()
        write_csv(out / "rolling_friction_kn_e_sweep.csv", list(incline_rows[0]), incline_rows)

    friction_rows = friction_kernel_rows()
    write_csv(out / "friction_kernel_conformance.csv", list(friction_rows[0]), friction_rows)

    hertz_table, hertz_meta = hertz_rows()
    write_csv(out / "hertz_convergence.csv", list(hertz_table[0]), hertz_table)

    buoyancy_table = buoyancy_rows()
    write_csv(out / "buoyancy_cap_convergence.csv", list(buoyancy_table[0]), buoyancy_table)

    summary = summarise(restitution_rows, incline_rows, friction_rows, hertz_table,
                        hertz_meta, buoyancy_table)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()