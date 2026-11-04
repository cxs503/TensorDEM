"""Finite-amplitude (nonlinear) heave period of a floating ice floe.

Closes the verification debt recorded in ``docs/ice_floe_physics_validation.md``
§limitations ("*Small-amplitude only* ... the nonlinear (finite-amplitude)
heave period is not verified") and in ``docs/tensordem_floe_buoyancy_validation.md``.

For each configuration and each release amplitude the script measures/derives
three numbers from the *shipped* code:

``T_lin``   linearised anchor ``2 pi sqrt(m / K)`` with the closed-form discrete
            stiffness ``tensordem.floe_buoyancy.discrete_stiffness`` (this is the
            reference that was previously the only one available);
``T_quad``  exact energy-integral reference
            ``tensordem.floe_buoyancy.lattice_finite_amplitude_heave_period``,
            which drives the production buoyancy kernel;
``T_sim``   period measured from a ``RigidHeaveFloe`` trajectory (velocity
            Verlet on the same kernel) by interpolated upward zero crossings.

Nothing below is hand-copied: the table and every number in the JSON/PNG are
computed here.

Run:  PYTHONPATH=src python3 benchmarks/ice_floe_finite_amplitude_heave.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ellipk

from tensordem.buoyancy_wet import WetCouplingConfig
from tensordem.floe_buoyancy import (
    RigidHeaveFloe,
    discrete_stiffness,
    equilibrium_center_height,
    finite_amplitude_heave_period,
    hex_lattice_floe,
    lattice_finite_amplitude_heave_period,
    lattice_heave_acceleration,
    mean_period,
    upward_crossing_periods,
)

OUT = Path("artifacts/ice_floe_finite_amplitude_heave")

RHO_ICE = 917.0
RHO_WATER = 1025.0
G = 9.81

DISK = dict(name="single_disk", radius=0.5, thickness=1.0)
FLOE = dict(name="hex_floe_37", particle_radius=0.1, thickness=0.2, n_particles=37)

AMPLITUDE_RATIOS = (2e-3, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def wet_config(drag: float = 0.0) -> WetCouplingConfig:
    return WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=G,
                             water_level=0.0, wet_drag_coeff=drag)


def equilibrium_height(accel, lo: float, hi: float) -> float:
    """Bisection on ``a(y) = 0`` (a *float* equilibrium, no small-draft guess)."""
    a_lo = float(np.asarray(accel(np.array([lo])), dtype=float)[0])
    a_hi = float(np.asarray(accel(np.array([hi])), dtype=float)[0])
    if a_lo * a_hi > 0.0:
        raise RuntimeError(f"equilibrium not bracketed on [{lo}, {hi}]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        a_mid = float(np.asarray(accel(np.array([mid])), dtype=float)[0])
        if a_mid == 0.0:
            return mid
        if a_lo * a_mid < 0.0:
            hi, a_hi = mid, a_mid
        else:
            lo, a_lo = mid, a_mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def lattice_case(case: dict):
    """Return ``(offsets, radius, thickness, cfg, y_eq, mass, stiffness)``."""
    cfg = wet_config()
    if case["name"] == "single_disk":
        offsets = torch.zeros((1, 2), dtype=torch.float64)
        radius, thickness = case["radius"], case["thickness"]
    else:
        radius, thickness = case["particle_radius"], case["thickness"]
        offsets = hex_lattice_floe(radius, case["n_particles"])
    accel = lattice_heave_acceleration(offsets, radius, thickness, RHO_ICE, cfg)
    # bracket the waterline by the body envelope, not by one particle radius:
    # a 37-particle lattice spans several radii vertically.
    reach = float(offsets[:, 1].abs().max()) + radius
    y_eq = equilibrium_height(accel, -reach, reach)
    mass = RHO_ICE * math.pi * radius ** 2 * thickness * int(offsets.shape[0])
    stiffness = discrete_stiffness(offsets, radius, thickness, RHO_ICE, cfg, y_eq)
    return offsets, radius, thickness, cfg, y_eq, mass, stiffness


def measure_solver_period(offsets, radius, thickness, cfg, y_eq, amplitude,
                          period_guess: float, steps_per_period: int = 1500,
                          n_periods: float = 3.2) -> tuple[float, float]:
    """Measure the heave period from a drag-free ``RigidHeaveFloe`` trajectory."""
    dt = period_guess / steps_per_period
    floe = RigidHeaveFloe(offsets, radius, thickness, RHO_ICE, cfg, dt=dt,
                          y0=y_eq + amplitude)
    traj = floe.run(int(math.ceil(n_periods * steps_per_period)))
    periods = upward_crossing_periods(traj.t, traj.y, y_eq)
    return mean_period(periods, skip=1), dt


def rel(a: float, b: float) -> float:
    return (a - b) / b


def pct(a: float, b: float) -> float:
    return 100.0 * rel(a, b)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def quadrature_sanity() -> dict:
    """Validate the energy-integral machinery against a closed-form reference.

    Free Duffing oscillator ``x'' = -(w0^2 x + beta x^3)`` has the exact period
    ``T = 4 / sqrt(w0^2 + beta A^2) * K(k)``, ``k^2 = beta A^2 / (2 (w0^2 + beta A^2))``
    (Landau & Lifshitz, *Mechanics*, §11 problem).  This checks the quadrature,
    the turning-point search and the singularity handling -- none of which can
    be checked against itself.
    """
    w0, beta = 2.0, 50.0

    def accel(ys):
        ys = np.asarray(ys, dtype=np.float64)
        return -(w0 ** 2 * ys + beta * ys ** 3)

    rows = []
    for amplitude in (0.01, 0.05, 0.1, 0.2):
        est = finite_amplitude_heave_period(accel, 0.0, amplitude)
        k2 = beta * amplitude ** 2 / (2.0 * (w0 ** 2 + beta * amplitude ** 2))
        exact = 4.0 / math.sqrt(w0 ** 2 + beta * amplitude ** 2) * ellipk(k2)
        rows.append({"amplitude": amplitude, "quadrature": est.period,
                     "closed_form": exact, "rel_error": rel(est.period, exact),
                     "turning_point_quadrature": est.y_min,
                     "turning_point_closed_form": -amplitude})
    return {"duffing": rows,
            "max_rel_error": float(max(abs(r["rel_error"]) for r in rows))}


def grid_convergence(case: dict) -> dict:
    """Quadrature resolution study at a fixed amplitude (0.2 R)."""
    offsets, radius, thickness, cfg, y_eq, _mass, _k = lattice_case(case)
    amplitude = 0.2 * radius
    rows = []
    for n_grid in (1001, 2001, 4001, 8001):
        est = lattice_finite_amplitude_heave_period(offsets, radius, thickness,
                                                    RHO_ICE, cfg, y_eq, amplitude,
                                                    n_grid=n_grid)
        rows.append({"n_grid": n_grid, "period": est.period})
    ref = rows[-1]["period"]
    for row in rows:
        row["rel_error_vs_finest"] = rel(row["period"], ref)
    return {"case": case["name"], "rows": rows,
            "max_rel_error_vs_finest": float(max(abs(r["rel_error_vs_finest"]) for r in rows))}


def negative_controls(case: dict) -> dict:
    """Anti-tautology: the anchors must fail when the physics is stubbed."""
    offsets, radius, thickness, cfg, y_eq, mass, stiffness = lattice_case(case)
    amplitude = 0.3 * radius
    real = lattice_finite_amplitude_heave_period(offsets, radius, thickness,
                                                 RHO_ICE, cfg, y_eq, amplitude)

    # (a) the linearised reference is *not* a finite-amplitude reference
    linear_error = rel(real.period_linear, real.period)

    # (b) stub the kernel with a pure linear spring: the machinery then has to
    #     return the linear period at every amplitude (it may not manufacture
    #     an amplitude dependence out of its own numerics).
    accel = lattice_heave_acceleration(offsets, radius, thickness, RHO_ICE, cfg)
    omega2 = stiffness / mass

    def linear_stub(ys):
        return -omega2 * (np.asarray(ys, dtype=np.float64) - y_eq)

    stubbed = [finite_amplitude_heave_period(linear_stub, y_eq, f * radius).period
               for f in (0.05, 0.2, 0.4)]
    stub_spread = max(stubbed) - min(stubbed)
    stub_error = max(abs(rel(t, real.period_linear)) for t in stubbed)
    # tolerance = the machinery's own discretisation floor (~3e-6 at
    # n_grid=4001), measured independently by the Duffing closed-form check;
    # tighter than that the stub test would only re-measure the quadrature.

    # (c) over-stiff stub (2x): the anchor must move by ~1/sqrt(2) - 1
    def stiff_stub(ys):
        return -2.0 * omega2 * (np.asarray(ys, dtype=np.float64) - y_eq)

    stiff = finite_amplitude_heave_period(stiff_stub, y_eq, amplitude).period
    stiff_error = rel(stiff, real.period)

    return {
        "linear_reference_error_at_0.3R": linear_error,
        "linear_stub_periods": stubbed,
        "linear_stub_spread": stub_spread,
        "linear_stub_error_vs_linear_reference": stub_error,
        "double_stiffness_period": stiff,
        "double_stiffness_error_vs_anchor": stiff_error,
        "checks": {
            "linear_reference_is_wrong_beyond_3pct": bool(abs(linear_error) > 0.03),
            "linear_stub_has_no_amplitude_dependence": bool(abs(stub_spread) < 1e-9),
            "linear_stub_matches_linear_reference": bool(abs(stub_error) < 1e-5),
            "double_stiffness_moves_anchor_by_30pct": bool(abs(stiff_error) > 0.25),
        },
    }


# ---------------------------------------------------------------------------
# main campaign
# ---------------------------------------------------------------------------

def campaign(case: dict, ratios, steps_per_period: int, run_solver: bool) -> dict:
    offsets, radius, thickness, cfg, y_eq, mass, stiffness = lattice_case(case)
    rows = []
    for ratio in ratios:
        amplitude = ratio * radius
        est = lattice_finite_amplitude_heave_period(offsets, radius, thickness,
                                                    RHO_ICE, cfg, y_eq, amplitude)
        row = {
            "amplitude_ratio": ratio,
            "amplitude": amplitude,
            "y_max": est.y_max,
            "y_min": est.y_min,
            "energy_drop_per_unit_mass": est.energy_drop,
            "period_linear": est.period_linear,
            "period_quadrature": est.period,
            "quadrature_vs_linear": rel(est.period, est.period_linear),
        }
        if run_solver:
            sim, dt = measure_solver_period(offsets, radius, thickness, cfg, y_eq,
                                            amplitude, est.period,
                                            steps_per_period=steps_per_period)
            row["period_sim"] = sim
            row["dt"] = dt
            row["sim_vs_quadrature"] = rel(sim, est.period)
            row["sim_vs_linear"] = rel(sim, est.period_linear)
        rows.append(row)
    return {
        "case": case["name"],
        "radius": radius,
        "thickness": thickness,
        "n_particles": int(offsets.shape[0]),
        "mass": mass,
        "linear_stiffness": stiffness,
        "y_eq": y_eq,
        "period_linear_reference": 2.0 * math.pi * math.sqrt(mass / stiffness),
        "rows": rows,
    }


def force_profile(case: dict, n: int = 400) -> dict:
    """Restoring acceleration and its linear tangent, for the figure."""
    offsets, radius, thickness, cfg, y_eq, mass, stiffness = lattice_case(case)
    accel = lattice_heave_acceleration(offsets, radius, thickness, RHO_ICE, cfg)
    span = 0.6 * radius
    ys = np.linspace(y_eq - span, y_eq + span, n)
    a = np.asarray(accel(ys), dtype=float)
    tangent = -(stiffness / mass) * (ys - y_eq)
    return {"y": ys.tolist(), "accel": a.tolist(), "linear_tangent": tangent.tolist(),
            "y_eq": y_eq, "radius": radius, "kink_excess": float(np.max(np.abs(a - tangent)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the solver lane")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ratios = AMPLITUDE_RATIOS if not args.fast else (2e-3, 0.2, 0.4)

    report: dict = {
        "config": {"rho_ice": RHO_ICE, "rho_water": RHO_WATER, "gravity": G,
                   "wet_drag_coeff": 0.0,
                   "note": "drag-free: the energy integral is conservative"},
        "quadrature_validation": quadrature_sanity(),
        "grid_convergence": [grid_convergence(DISK), grid_convergence(FLOE)],
        "negative_controls": [dict(negative_controls(DISK), case="single_disk"),
                              dict(negative_controls(FLOE), case="hex_floe_37")],
        "campaigns": [],
    }

    for case in (DISK, FLOE):
        report["campaigns"].append(
            campaign(case, ratios, steps_per_period=1200 if args.fast else 1500,
                     run_solver=not args.fast))
    profiles = {"single_disk": force_profile(DISK), "hex_floe_37": force_profile(FLOE)}

    # --- error summary -----------------------------------------------------
    solver_rows = [r for c in report["campaigns"] for r in c["rows"] if "period_sim" in r]
    quad_rows = [r for c in report["campaigns"] for r in c["rows"]]
    criterion = 0.03
    summary = {
        "max_abs_sim_vs_quadrature": max((abs(r["sim_vs_quadrature"]) for r in solver_rows),
                                          default=float("nan")),
        "max_abs_quadrature_vs_linear": max(abs(r["quadrature_vs_linear"]) for r in quad_rows),
        "quadrature_validation_max_rel_error": report["quadrature_validation"]["max_rel_error"],
        "criterion": criterion,
    }
    summary["sim_vs_quadrature_within_3pct"] = (
        not solver_rows or max(abs(r["sim_vs_quadrature"]) for r in solver_rows) < criterion)
    summary["duffing_within_3pct"] = (
        report["quadrature_validation"]["max_rel_error"] < criterion)
    report["summary"] = summary
    report["runtime_s"] = time.time() - t0

    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (OUT / "force_profiles.json").write_text(json.dumps(profiles, indent=2))

    make_figure(report, profiles, OUT / "period_vs_amplitude.png")
    write_table(report, OUT / "table.md")

    print(f"artifacts -> {OUT}  ({report['runtime_s']:.1f} s)")
    for key in ("sim_vs_quadrature_within_3pct", "duffing_within_3pct"):
        print(f"  {key}: {summary[key]}")
    print(f"  max |sim-quad| = {summary['max_abs_sim_vs_quadrature']:.3e}"
          f"  max |quad-lin| = {summary['max_abs_quadrature_vs_linear']:.3f}")
    return 0


def make_figure(report: dict, profiles: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=140)

    ax = axes[0]
    colours = {"single_disk": "C0", "hex_floe_37": "C3"}
    for camp in report["campaigns"]:
        colour = colours.get(camp["case"], "C7")
        ratios = [r["amplitude_ratio"] for r in camp["rows"]]
        quad = [r["period_quadrature"] for r in camp["rows"]]
        ax.plot(ratios, quad, "o-", color=colour,
                label=f"{camp['case']}: energy integral")
        t_lin = camp["period_linear_reference"]
        ax.axhline(t_lin, color=colour, ls=":", lw=1.0)
        if "period_sim" in camp["rows"][0]:
            sims = [r["period_sim"] for r in camp["rows"]]
            ax.plot(ratios, sims, "x", color=colour, ms=6,
                    label=f"{camp['case']}: solver (Verlet)")
    ax.set_xlabel(r"release amplitude $A/R$")
    ax.set_ylabel("heave period $T$  [s]")
    ax.set_title("finite-amplitude heave period\n(dotted = linearised anchor)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    for name in ("single_disk", "hex_floe_37"):
        prof = profiles[name]
        colour = colours[name]
        y = np.asarray(prof["y"]) - prof["y_eq"]
        ax.plot(y / prof["radius"], np.asarray(prof["accel"]), color=colour,
                label=f"{name}: kernel")
        ax.plot(y / prof["radius"], np.asarray(prof["linear_tangent"]), color=colour,
                ls="--", lw=1.0, label=f"{name}: linear tangent")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"$(y - y_{eq}) / R$")
    ax.set_ylabel(r"restoring acceleration  [m/s$^2$]")
    ax.set_title("why the period moves:\nnonlinear restoring force")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_table(report: dict, path: Path) -> None:
    lines = ["| case | $A/R$ | $T_{lin}$ [s] | $T_{quad}$ [s] | $T_{sim}$ [s] | $T_{quad}/T_{lin}-1$ | $T_{sim}/T_{quad}-1$ |",
             "|---|---|---|---|---|---|---|"]
    for camp in report["campaigns"]:
        for row in camp["rows"]:
            sim = row.get("period_sim")
            sim_s = f"{sim:.5f}" if sim is not None else "—"
            sim_rel = f"{pct(row['period_sim'], row['period_quadrature']):+.4f} %" if sim is not None else "—"
            lines.append(
                f"| {camp['case']} | {row['amplitude_ratio']:.4g} | "
                f"{row['period_linear']:.5f} | {row['period_quadrature']:.5f} | {sim_s} | "
                f"{pct(row['period_quadrature'], row['period_linear']):+.3f} % | {sim_rel} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())