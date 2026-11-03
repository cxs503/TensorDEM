"""Buoyancy / wet-drag validation benchmark for the 2-D floe model.

Closes the validation debt of ``tensordem.buoyancy_wet``: those force kernels
shipped with unit tests on the submerged-fraction helper only, plus a single
damped settling run (``benchmarks/ice_floe_physics.py``).  No analytic
reference existed for the Archimedes draft, the heave natural period, the
wet-drag decay rate, or the discretisation bias of a floe built from discrete
particles.

Stages
  A  geometry   submerged fraction vs independent per-column quadrature
  B  statics    exact Archimedes draft vs the naive linearised formula
  C  dynamics   heave period vs analytic, over amplitude and dt
  D  damping    log-decrement damping ratio vs analytic zeta = C f*/(2 m omega)
  E  energy     undamped energy drift of the velocity-Verlet integrator
  F  lattice    circular floe of N tangent disks: draft, stiffness, period
  G  slab       rectangular floe: closed-form stiffness convergence with N

Everything is written to ``outputs/floe_buoyancy/`` (JSON + PNG + generated
markdown summary); no number in the docs is hand-typed.

Usage
    PYTHONPATH=src python3 scripts/benchmark_floe_buoyancy.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

from tensordem.buoyancy_wet import (
    WetCouplingConfig,
    submerged_fraction_tensor,
)
from tensordem.floe_buoyancy import (
    RigidHeaveFloe,
    analytic_damping_ratio,
    analytic_heave_period,
    body_envelope,
    discrete_buoyancy_profile,
    discrete_stiffness,
    equilibrium_center_height,
    hex_lattice_floe,
    linear_heave_stiffness,
    log_decrement_damping_ratio,
    mean_period,
    slab_lattice,
    upward_crossing_periods,
    waterplane_chord,
)

RHO_ICE = 917.0
RHO_WATER = 1025.0
GRAVITY = 9.81
RADIUS = 0.5
THICKNESS = 1.0
TRIANGULAR_PACKING = math.pi / (2.0 * math.sqrt(3.0))

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "floe_buoyancy"


def _cfg(wet_drag: float = 0.0) -> WetCouplingConfig:
    return WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=GRAVITY,
                             water_level=0.0, wet_drag_coeff=wet_drag)


# ---------------------------------------------------------------------------
# A. geometry: independent per-column quadrature reference
# ---------------------------------------------------------------------------

def reference_submerged_fraction(h: float, radius: float, n_columns: int = 200001) -> float:
    """Submerged area fraction by integrating analytic column chords over x.

    Independent of the ``acos`` segment formula used by the engine: for a disk
    of radius R whose centre is ``h`` above the surface, the column at abscissa
    x spans ``y in [h - s, h + s]`` with ``s = sqrt(R^2 - x^2)``, so its
    submerged length is ``max(0, min(h + s, 0) - (h - s))``.  Midpoint rule on
    ``n_columns`` strips, vectorised (the sqrt endpoint singularity is
    integrable and converges far below the gate used here).
    """
    n = n_columns
    dx = 2.0 * radius / n
    x = torch.linspace(-radius + 0.5 * dx, radius - 0.5 * dx, n, dtype=torch.float64)
    s = torch.sqrt(torch.clamp(radius * radius - x * x, min=0.0))
    submerged = torch.clamp(torch.minimum(h + s, torch.zeros_like(s)) - (h - s), min=0.0)
    return float(submerged.sum()) * dx / (math.pi * radius * radius)


def stage_a_geometry(n_offsets: int = 41) -> dict:
    offsets = [(-1.2 + 2.4 * k / (n_offsets - 1)) * RADIUS for k in range(n_offsets)]
    rows = []
    for h in offsets:
        engine = float(submerged_fraction_tensor(torch.tensor([h]), RADIUS, 0.0)[0])
        ref = reference_submerged_fraction(h, RADIUS)
        rows.append({"h_over_R": h / RADIUS, "engine": engine, "reference": ref,
                     "abs_error": abs(engine - ref)})
    max_err = max(r["abs_error"] for r in rows)
    return {"n_offsets": n_offsets, "max_abs_error": max_err, "rows": rows}


# ---------------------------------------------------------------------------
# B. statics: exact Archimedes draft vs naive linearisation
# ---------------------------------------------------------------------------

def stage_b_statics(ratios: tuple[float, ...] = (0.85, RHO_ICE / RHO_WATER, 0.95)) -> dict:
    rows = []
    for ratio in ratios:
        rho_ice = ratio * RHO_WATER
        h_eq = equilibrium_center_height(rho_ice, RHO_WATER, RADIUS)
        naive = -(1.0 - ratio) * RADIUS
        f_eq = float(submerged_fraction_tensor(torch.tensor([h_eq]), RADIUS, 0.0)[0])
        rows.append({
            "density_ratio": ratio,
            "f_sub_analytic": ratio,
            "f_sub_measured": f_eq,
            "f_sub_rel_error": abs(f_eq - ratio) / ratio,
            "h_eq_exact": h_eq,
            "h_naive_linear": naive,
            "h_naive_rel_error": abs(naive - h_eq) / abs(h_eq),
            "chord": waterplane_chord(RADIUS, h_eq),
        })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# single-disk heave driver
# ---------------------------------------------------------------------------

def single_disk_case(amplitude: float, dt: float, n_periods: float, period_ref: float,
                     wet_drag: float = 0.0, record_energy: bool = False) -> dict:
    h_eq = equilibrium_center_height(RHO_ICE, RHO_WATER, RADIUS)
    floe = RigidHeaveFloe(torch.zeros((1, 2), dtype=torch.float64), RADIUS,
                          THICKNESS, RHO_ICE, _cfg(wet_drag), dt, y0=h_eq + amplitude)
    n_steps = max(int(n_periods * period_ref / dt), 1)
    t0 = time.time()
    floe.run(n_steps, record_every=max(n_steps // 4000, 1), record_energy=record_energy)
    wall = time.time() - t0
    traj = floe.trajectory
    periods = upward_crossing_periods(traj.t, traj.y, h_eq)
    return {"floe": floe, "trajectory": traj, "h_eq": h_eq, "periods": periods,
            "mean_period": mean_period(periods, skip=1), "wall_s": wall,
            "n_steps": n_steps}


# ---------------------------------------------------------------------------
# C. dynamics: heave period vs analytic
# ---------------------------------------------------------------------------

def stage_c_period(quick: bool) -> dict:
    h_eq = equilibrium_center_height(RHO_ICE, RHO_WATER, RADIUS)
    stiffness = linear_heave_stiffness(RHO_WATER, GRAVITY, THICKNESS, RADIUS, h_eq)
    mass = RHO_ICE * math.pi * RADIUS ** 2 * THICKNESS
    period = analytic_heave_period(mass, stiffness)

    amplitudes = (0.005, 0.01, 0.02, 0.05, 0.10)
    amp_rows = []
    for frac in amplitudes:
        case = single_disk_case(frac * RADIUS, period / 500.0, 6.0, period)
        amp_rows.append({
            "amplitude_over_R": frac,
            "amplitude_m": frac * RADIUS,
            "period_measured": case["mean_period"],
            "rel_error": abs(case["mean_period"] - period) / period,
            "cycles": len(case["periods"]),
            "n_steps": case["n_steps"],
            "wall_s": case["wall_s"],
        })

    dt_factors = (250, 500, 1000, 2000) if not quick else (250, 1000)
    dt_rows = []
    for factor in dt_factors:
        case = single_disk_case(0.01 * RADIUS, period / factor, 6.0, period)
        dt_rows.append({
            "dt": period / factor,
            "steps_per_period": factor,
            "period_measured": case["mean_period"],
            "rel_error": abs(case["mean_period"] - period) / period,
            "n_steps": case["n_steps"],
            "wall_s": case["wall_s"],
        })
    return {
        "mass_kg": mass, "stiffness_N_per_m": stiffness, "period_analytic_s": period,
        "h_eq_m": h_eq, "omega_rad_per_s": 2.0 * math.pi / period,
        "amplitude_sweep": amp_rows, "dt_convergence": dt_rows,
    }


# ---------------------------------------------------------------------------
# D. damping: log decrement vs analytic zeta
# ---------------------------------------------------------------------------

def stage_d_damping(quick: bool) -> dict:
    h_eq = equilibrium_center_height(RHO_ICE, RHO_WATER, RADIUS)
    stiffness = linear_heave_stiffness(RHO_WATER, GRAVITY, THICKNESS, RADIUS, h_eq)
    mass = RHO_ICE * math.pi * RADIUS ** 2 * THICKNESS
    period = analytic_heave_period(mass, stiffness)
    omega = 2.0 * math.pi / period
    f_star = RHO_ICE / RHO_WATER

    zeta_targets = (0.02, 0.05, 0.10) if not quick else (0.05,)
    rows = []
    for zeta in zeta_targets:
        c_wet = 2.0 * mass * omega * zeta / f_star
        case = single_disk_case(0.02 * RADIUS, period / 250.0, 16.0, period, wet_drag=c_wet)
        traj = case["trajectory"]
        zeta_meas = log_decrement_damping_ratio(traj.t, traj.y, h_eq, max_cycles=10)
        period_damped = period / math.sqrt(1.0 - zeta ** 2)
        rows.append({
            "zeta_target": zeta,
            "zeta_from_formula": analytic_damping_ratio(c_wet, f_star, mass, omega),
            "wet_drag_coeff": c_wet,
            "zeta_measured": zeta_meas,
            "zeta_rel_error": abs(zeta_meas - zeta) / zeta,
            "period_measured": case["mean_period"],
            "period_damped_analytic": period_damped,
            "period_rel_error": abs(case["mean_period"] - period_damped) / period_damped,
            "n_steps": case["n_steps"], "wall_s": case["wall_s"],
        })
    return {"rows": rows, "period_undamped_s": period}


# ---------------------------------------------------------------------------
# E. energy drift (undamped)
# ---------------------------------------------------------------------------

def stage_e_energy() -> dict:
    h_eq = equilibrium_center_height(RHO_ICE, RHO_WATER, RADIUS)
    stiffness = linear_heave_stiffness(RHO_WATER, GRAVITY, THICKNESS, RADIUS, h_eq)
    mass = RHO_ICE * math.pi * RADIUS ** 2 * THICKNESS
    period = analytic_heave_period(mass, stiffness)
    amp = 0.02 * RADIUS
    case = single_disk_case(amp, period / 500.0, 10.0, period, record_energy=True)
    energy = case["trajectory"].energy
    scale = 0.5 * stiffness * amp ** 2
    drift = max(energy) - min(energy)
    return {"amplitude_m": amp, "energy_scale_J": scale, "drift_J": drift,
            "relative_drift": drift / scale, "n_samples": len(energy),
            "wall_s": case["wall_s"]}


# ---------------------------------------------------------------------------
# F. circular lattice floe
# ---------------------------------------------------------------------------

def lattice_equilibrium(offsets: torch.Tensor, a: float) -> float:
    """Bisection on centroid height where aggregate buoyancy balances weight."""
    cfg = _cfg()
    mass = RHO_ICE * math.pi * a ** 2 * THICKNESS * offsets.shape[0]
    reach = float(offsets.abs().max()) + a
    lo, hi = -reach, reach
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if discrete_buoyancy_profile(offsets, a, THICKNESS, RHO_ICE, cfg, mid) > mass * GRAVITY:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def stage_f_lattice(quick: bool) -> dict:
    cfg = _cfg()
    radius_eq = 0.5
    counts = (7, 19, 37, 61, 91) if not quick else (7, 19)
    ratio = RHO_ICE / RHO_WATER
    h_disk = equilibrium_center_height(RHO_ICE, RHO_WATER, radius_eq)
    k_disk = linear_heave_stiffness(RHO_WATER, GRAVITY, THICKNESS, radius_eq, h_disk)
    mass_disk = RHO_ICE * math.pi * radius_eq ** 2 * THICKNESS
    period_disk = analytic_heave_period(mass_disk, k_disk)

    rows = []
    for n in counts:
        a = radius_eq / math.sqrt(n)
        offsets = hex_lattice_floe(a, n)
        n_actual = int(offsets.shape[0])
        width, height = body_envelope(offsets, a)
        mass = RHO_ICE * math.pi * a ** 2 * THICKNESS * n_actual
        y_eq = lattice_equilibrium(offsets, a)
        f_bar = float(submerged_fraction_tensor(offsets[:, 1] + y_eq, a, 0.0).mean())
        k_num = discrete_stiffness(offsets, a, THICKNESS, RHO_ICE, cfg, y_eq)
        period_num = analytic_heave_period(mass, k_num)

        amp = 0.005 * radius_eq
        dt = period_num / 500.0
        floe = RigidHeaveFloe(offsets, a, THICKNESS, RHO_ICE, cfg, dt, y0=y_eq + amp)
        floe.run(int(6.0 * period_num / dt), record_every=max(int(0.01 * period_num / dt), 1))
        traj = floe.trajectory
        period_meas = mean_period(upward_crossing_periods(traj.t, traj.y, y_eq), skip=1)

        rows.append({
            "n_particles": n_actual,
            "particle_radius_m": a,
            "envelope_width_m": width,
            "envelope_height_m": height,
            "mass_kg": mass,
            "y_eq_m": y_eq,
            "f_sub_aggregate": f_bar,
            "f_sub_rel_error": abs(f_bar - ratio) / ratio,
            "stiffness_discrete": k_num,
            "stiffness_equivalent_disk": k_disk,
            "stiffness_rel_error": abs(k_num - k_disk) / k_disk,
            "period_analytic_discrete": period_num,
            "period_analytic_equivalent_disk": period_disk,
            "period_measured": period_meas,
            "period_meas_vs_discrete_error": abs(period_meas - period_num) / period_num,
            "period_meas_vs_disk_error": abs(period_meas - period_disk) / period_disk,
        })
    return {"radius_equivalent_m": radius_eq, "h_eq_disk_m": h_disk,
            "stiffness_equivalent_disk": k_disk,
            "period_equivalent_disk_s": period_disk, "rows": rows}


# ---------------------------------------------------------------------------
# G. slab convergence against the closed-form porous-body stiffness
# ---------------------------------------------------------------------------

def _band_bounds(offsets: torch.Tensor, a: float) -> tuple[float, float]:
    """Centroid heights where the floe is just fully dry / just fully submerged."""
    cfg = _cfg()
    mass = RHO_ICE * math.pi * a ** 2 * THICKNESS * offsets.shape[0]
    top = float(offsets[:, 1].max()) + a
    bottom = float(offsets[:, 1].min()) - a
    return -top, -bottom          # (fully dry, fully submerged) centroid heights


def _band_equilibrium(offsets: torch.Tensor, a: float, target_f: float) -> float:
    """Centroid height at which the aggregate submerged fraction equals ``target_f``.

    Submerged fraction *decreases* with height, so the bracket is
    ``[fully submerged (low), fully dry (high)]`` and the update rule keeps that
    ordering.
    """
    y_dry, y_wet = _band_bounds(offsets, a)
    lo, hi = y_wet, y_dry          # lo: fully submerged (f=1), hi: fully dry (f=0)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        f = float(submerged_fraction_tensor(offsets[:, 1] + mid, a, 0.0).mean())
        if f > target_f:
            lo = mid               # still too deep -> raise
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def stage_g_slab(quick: bool) -> dict:
    """Waterline-relative heave stiffness of a discrete floe (slab geometry).

    A slab has vertical sides, so the continuum heave stiffness
    ``K = rho_w g t * (submerged width)`` is height independent.  For a lattice
    of tangent disks the waterline only crosses one row at a time, so the
    instantaneous stiffness is a staircase in height.  Two exact statements
    anchor the measurement:

    * band integral: ``int K dy`` over the whole partial-submergence band equals
      ``rho_w g t * A_solid`` (every disk contributes exactly ``pi a^2``),
      hence the band mean is ``rho_w g t A_solid / H_band`` -- an identity the
      code must reproduce to quadrature accuracy;
    * local: ``K(y) = rho_w g t sum_i chord_i(y)`` -- closed form, tested by the
      simulated small-amplitude heave period.

    The physical output is the waterline-phase scatter of the heave period,
    which is what a real floe discretisation is limited by.
    """
    cfg = _cfg()
    width, height = 1.0, 2.0
    columns = (4, 8, 16, 32) if not quick else (8, 32)
    rows = []
    for m in columns:
        a = width / (2.0 * m)
        offsets = slab_lattice(width, height, a)
        env_w, env_h = body_envelope(offsets, a)
        n_particles = int(offsets.shape[0])
        area_solid = n_particles * math.pi * a ** 2
        mass = RHO_ICE * area_solid * THICKNESS
        y_dry, y_wet = _band_bounds(offsets, a)
        band_height = y_wet - y_dry
        y_eq = _band_equilibrium(offsets, a, RHO_ICE / RHO_WATER)

        n_samples = 121
        ys = [y_dry + band_height * k / (n_samples - 1) for k in range(n_samples)]
        ks = [discrete_stiffness(offsets, a, THICKNESS, RHO_ICE, cfg, y) for y in ys]
        k_mean = sum(ks) / len(ks)
        k_identity = RHO_WATER * GRAVITY * THICKNESS * area_solid / band_height
        k_porous = RHO_WATER * GRAVITY * THICKNESS * env_w * TRIANGULAR_PACKING
        # a lattice floe can sit in a *neutral* window where no disk straddles the
        # waterline at all: K = 0, heave has no restoring force.  Fraction of the
        # band that is unusable for a heave measurement:
        neutral = sum(1 for k in ks if k < 0.05 * k_identity) / len(ks)

        period_rows = []
        for frac in (0.25, 0.5, 0.75):
            y0 = _band_equilibrium(offsets, a, frac)
            k_local = discrete_stiffness(offsets, a, THICKNESS, RHO_ICE, cfg, y0)
            is_neutral = k_local < 0.05 * k_identity
            period_closed = (analytic_heave_period(mass, k_local)
                             if not is_neutral and k_local > 0.0 else float("inf"))
            period_rows.append({
                "f_sub_start": frac,
                "f_sub_actual": float(submerged_fraction_tensor(
                    offsets[:, 1] + y0, a, 0.0).mean()),
                "y0_m": y0,
                "k_local": k_local,
                "k_local_fraction_of_band_mean": k_local / k_mean,
                "neutral": is_neutral,
                "period_closed_form_s": period_closed,
            })
        finite_periods = [r["period_closed_form_s"] for r in period_rows
                          if math.isfinite(r["period_closed_form_s"])]
        scatter = ((max(finite_periods) - min(finite_periods))
                   / (sum(finite_periods) / len(finite_periods))) if len(finite_periods) > 1 else 0.0
        rows.append({
            "columns": m,
            "particle_radius_m": a,
            "n_particles": n_particles,
            "envelope_width_m": env_w,
            "envelope_height_m": env_h,
            "lattice_spacing_m": a * math.sqrt(3.0),
            "mass_kg": mass,
            "band_height_m": band_height,
            "y_eq_m": y_eq,
            "k_mean": k_mean,
            "k_min": min(ks),
            "k_max": max(ks),
            "k_oscillation": (max(ks) - min(ks)) / k_mean,
            "k_porous_closed_form": k_porous,
            "k_mean_vs_identity": k_mean / k_identity - 1.0,
            "k_mean_vs_porous": k_mean / k_porous - 1.0,
            "neutral_band_fraction": neutral,
            "period_rows": period_rows,
            "period_scatter_from_closed_form": scatter,
        })
    return {"triangular_packing_fraction": TRIANGULAR_PACKING,
            "geometry": {"width_m": width, "height_m": height}, "rows": rows}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(result: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax = axes[0]
    rows = result["stage_a_geometry"]["rows"]
    ax.semilogy([r["h_over_R"] for r in rows],
                [max(r["abs_error"], 1e-18) for r in rows], "o-", ms=3)
    ax.axhline(1e-3, color="r", ls="--", lw=1, label="1e-3 gate")
    ax.set_xlabel("disk centre height  h / R   (h > 0 above water)")
    ax.set_ylabel("|engine - quadrature|  (submerged fraction)")
    ax.set_title("A  submerged-fraction geometry")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)

    ax = axes[1]
    c = result["stage_c_period"]
    ax.plot([r["amplitude_over_R"] * 100 for r in c["amplitude_sweep"]],
            [r["rel_error"] * 100 for r in c["amplitude_sweep"]], "o-",
            label="amplitude sweep (dt = T/500)")
    ax.plot([1.0 / r["steps_per_period"] * 100 for r in c["dt_convergence"]],
            [r["rel_error"] * 100 for r in c["dt_convergence"]], "s--",
            label="dt refinement (amp = 1% R)")
    ax.axhline(3.0, color="r", ls="--", lw=1, label="3% gate")
    ax.set_xscale("log")
    ax.set_xlabel("amplitude [% R]   /   dt [% T]")
    ax.set_ylabel("period error vs analytic [%]")
    ax.set_title("B  heave period vs analytic  (T = %.4f s)" % c["period_analytic_s"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    d = result["stage_d_damping"]
    ax.plot([r["zeta_target"] for r in d["rows"]],
            [r["zeta_measured"] for r in d["rows"]], "o", ms=6,
            label="measured (log decrement)")
    lims = [0.0, max(r["zeta_target"] for r in d["rows"]) * 1.2]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlabel(r"target $\zeta = C f^*/(2 m \omega)$")
    ax.set_ylabel("measured damping ratio")
    ax.set_title("C  wet-drag decay rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "floe_buoyancy_anchors.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.name)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax = axes[0]
    f = result["stage_f_lattice"]
    ns = [r["n_particles"] for r in f["rows"]]
    ax.semilogx(ns, [r["f_sub_rel_error"] * 100 for r in f["rows"]], "o-",
                label=r"aggregate draft $\bar f$ vs $\rho_i/\rho_w$")
    ax.semilogx(ns, [r["stiffness_rel_error"] * 100 for r in f["rows"]], "s-",
                label="waterplane stiffness $K_N$ vs equal-area disk")
    ax.semilogx(ns, [r["period_meas_vs_disk_error"] * 100 for r in f["rows"]], "^-",
                label="heave period vs equal-area disk")
    ax.axhline(3.0, color="r", ls="--", lw=1, label="3% gate")
    ax.set_xlabel("particles in the circular floe  N")
    ax.set_ylabel("relative error [%]")
    ax.set_title("D  circular floe built from tangent disks")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    g = result["stage_g_slab"]
    col = [r["columns"] for r in g["rows"]]
    ax.plot(col, [r["k_mean_vs_identity"] * 100 for r in g["rows"]], "o-",
            label=r"band-mean $K$ vs $\rho_w g t A/H_{band}$ identity")
    ax.plot(col, [r["k_mean_vs_porous"] * 100 for r in g["rows"]], "s-",
            label=r"band-mean $K$ vs $\rho_w g t W\phi$")
    ax.plot(col, [r["k_oscillation"] * 100 for r in g["rows"]], "^--",
            label="staircase oscillation (max-min)/mean")
    ax.axhline(0.0, color="k", lw=1)
    ax.axhspan(-3.0, 3.0, color="r", alpha=0.08, label="3% gate")
    ax.set_xlabel("particles across the slab width")
    ax.set_ylabel("deviation [%]")
    ax.set_title("E  slab heave stiffness vs exact identities")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    e = result["stage_e_energy"]
    ax.bar(["undamped heave\n(10 periods)"], [e["relative_drift"] * 100], width=0.35)
    ax.axhline(3.0, color="r", ls="--", lw=1, label="3% gate")
    ax.set_ylabel(r"max $|\Delta E| / (0.5 K A^2)$  [%]")
    ax.set_title("F  energy drift, velocity Verlet  (%.3g J scale)" % e["energy_scale_J"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "floe_buoyancy_lattice.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.name)
    return paths


# ---------------------------------------------------------------------------
# markdown summary
# ---------------------------------------------------------------------------

def write_summary(result: dict, out_dir: Path) -> str:
    a, b = result["stage_a_geometry"], result["stage_b_statics"]
    c, d = result["stage_c_period"], result["stage_d_damping"]
    e, f, g = result["stage_e_energy"], result["stage_f_lattice"], result["stage_g_slab"]
    L = ["# Buoyancy / wet-drag validation — generated summary\n",
         "Generated by `scripts/benchmark_floe_buoyancy.py`; every number below is "
         "injected from the run.\n",
         "## A. Submerged-fraction geometry vs independent column quadrature\n",
         f"- offsets tested: {a['n_offsets']} over h/R in [-1.2, 1.2]",
         f"- max |engine - quadrature| = {a['max_abs_error']:.3e}\n",
         "## B. Statics: exact Archimedes draft vs naive linearisation\n",
         "| rho_i/rho_w | f_sub analytic | f_sub measured | rel err | h_eq exact [m] "
         "| h naive [m] | naive rel err |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in b["rows"]:
        L.append(f"| {r['density_ratio']:.5f} | {r['f_sub_analytic']:.5f} | "
                 f"{r['f_sub_measured']:.5f} | {r['f_sub_rel_error']:.2e} | "
                 f"{r['h_eq_exact']:.6f} | {r['h_naive_linear']:.6f} | "
                 f"{r['h_naive_rel_error']*100:.2f}% |")
    L += [f"\nAnalytic heave period (equivalent disk): K = {c['stiffness_N_per_m']:.3f} N/m, "
          f"m = {c['mass_kg']:.4f} kg, **T = {c['period_analytic_s']:.6f} s**\n",
          "## C. Heave period vs analytic\n",
          "| amplitude / R | T measured [s] | rel err | cycles | steps |",
          "|---:|---:|---:|---:|---:|"]
    for r in c["amplitude_sweep"]:
        L.append(f"| {r['amplitude_over_R']*100:.1f}% | {r['period_measured']:.6f} | "
                 f"{r['rel_error']*100:.3f}% | {r['cycles']} | {r['n_steps']} |")
    L += ["\n| dt [s] | steps/period | T measured [s] | rel err |",
          "|---:|---:|---:|---:|"]
    for r in c["dt_convergence"]:
        L.append(f"| {r['dt']:.3e} | {r['steps_per_period']} | {r['period_measured']:.6f} | "
                 f"{r['rel_error']*100:.3f}% |")
    L += ["\n## D. Wet-drag damping ratio\n",
          "| zeta target | C_wet [N s/m] | zeta measured | rel err | T measured [s] | "
          "T damped analytic [s] |",
          "|---:|---:|---:|---:|---:|---:|"]
    for r in d["rows"]:
        L.append(f"| {r['zeta_target']:.3f} | {r['wet_drag_coeff']:.3f} | "
                 f"{r['zeta_measured']:.5f} | {r['zeta_rel_error']*100:.2f}% | "
                 f"{r['period_measured']:.6f} | {r['period_damped_analytic']:.6f} |")
    L += ["\n## E. Energy drift (undamped)\n",
          f"- amplitude {e['amplitude_m']:.4f} m, reference energy {e['energy_scale_J']:.6e} J",
          f"- max drift {e['drift_J']:.3e} J = {e['relative_drift']*100:.4f}% of reference\n",
          "## F. Circular floe of tangent disks\n",
          f"Equal-area disk: R_eq = {f['radius_equivalent_m']} m, h_eq = {f['h_eq_disk_m']:.6f} m, "
          f"K = {f['stiffness_equivalent_disk']:.3f} N/m, T = {f['period_equivalent_disk_s']:.6f} s\n",
          "| N | a [m] | f_sub aggregate | vs rho_i/rho_w | K_N [N/m] | K_N vs disk | "
          "T measured [s] | vs discrete | vs equal-area disk |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in f["rows"]:
        L.append(f"| {r['n_particles']} | {r['particle_radius_m']:.4f} | "
                 f"{r['f_sub_aggregate']:.6f} | {r['f_sub_rel_error']*100:.3f}% | "
                 f"{r['stiffness_discrete']:.3f} | {r['stiffness_rel_error']*100:.2f}% | "
                 f"{r['period_measured']:.6f} | {r['period_meas_vs_discrete_error']*100:.3f}% | "
                 f"{r['period_meas_vs_disk_error']*100:.2f}% |")
    L += ["\n## G. Slab: band-integral identity and waterline-phase scatter\n",
          "Geometry: %.1f m x %.1f m slabs; band mean identity "
          "K = rho_w g t A_solid / H_band; phi = %.6f.\n"
          % (g["geometry"]["width_m"], g["geometry"]["height_m"],
             g["triangular_packing_fraction"]),
          "| cols | a [m] | N | band H [m] | K band mean [N/m] | vs identity | "
          "vs rho_w g t W phi | oscillation | neutral band | period scatter (closed form) |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in g["rows"]:
        L.append(f"| {r['columns']} | {r['particle_radius_m']:.4f} | {r['n_particles']} | "
                 f"{r['band_height_m']:.4f} | {r['k_mean']:.1f} | "
                 f"{r['k_mean_vs_identity']*100:+.3e}% | {r['k_mean_vs_porous']*100:+.2f}% | "
                 f"{r['k_oscillation']*100:.1f}% | {r['neutral_band_fraction']*100:.1f}% | "
                 f"{r['period_scatter_from_closed_form']*100:.2f}% |")
    L.append("\nPer-start breakdown (target 25 / 50 / 75 % submergence; `neutral` = no disk "
             "straddles the waterline, K < 5% of the band mean):\n")
    L.append("| cols | f_sub target | f_sub actual | K local [N/m] | K local / K band mean | neutral | T closed form [s] |")
    L.append("|---:|---:|---:|---:|---:|:---:|---:|")
    for r in g["rows"]:
        for pr in r["period_rows"]:
            t = ("inf" if not math.isfinite(pr["period_closed_form_s"])
                 else f"{pr['period_closed_form_s']:.4f}")
            L.append(f"| {r['columns']} | {pr['f_sub_start']:.2f} | {pr['f_sub_actual']:.4f} | "
                     f"{pr['k_local']:.1f} | {pr['k_local_fraction_of_band_mean']:.3f} | "
                     f"{'yes' if pr['neutral'] else 'no'} | {t} |")
    text = "\n".join(L) + "\n"
    (out_dir / "summary.md").write_text(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short run for smoke tests")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = {
        "meta": {"rho_ice": RHO_ICE, "rho_water": RHO_WATER, "gravity": GRAVITY,
                 "radius_m": RADIUS, "thickness_m": THICKNESS, "quick": args.quick},
        "stage_a_geometry": stage_a_geometry(),
        "stage_b_statics": stage_b_statics(),
        "stage_c_period": stage_c_period(args.quick),
        "stage_d_damping": stage_d_damping(args.quick),
        "stage_e_energy": stage_e_energy(),
        "stage_f_lattice": stage_f_lattice(args.quick),
        "stage_g_slab": stage_g_slab(args.quick),
    }
    result["meta"]["wall_s"] = time.time() - t0
    result["figures"] = make_figures(result, out_dir)
    summary = write_summary(result, out_dir)
    (out_dir / "results.json").write_text(json.dumps(result, indent=1))
    print(summary)
    print(f"[floe_buoyancy] wrote {out_dir} in {result['meta']['wall_s']:.1f} s; "
          f"figures={result['figures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())