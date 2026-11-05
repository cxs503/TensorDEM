"""Benchmark: finite-amplitude (nonlinear) heave period of the 2-D floe model.

Closes the open item recorded in ``docs/ice_floe_physics_validation.md`` ("the
nonlinear (finite-amplitude) heave period is not verified and is expected to
differ"): the shipped ``RigidHeaveFloe`` + ``compute_buoyancy_force`` path is
timed over an amplitude sweep and compared with the exact energy quadrature of
``tensordem.floe_heave_nonlinear``.

Writes ``artifacts/floe_heave_nonlinear/report.json`` (all numbers),
``figure.png`` (period curve, error, trajectory) and ``tables.md`` (the tables
quoted in the verification document, so no number is transcribed by hand).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem.buoyancy_wet import WetCouplingConfig, submerged_fraction_tensor
from tensordem.floe_buoyancy import (
    RigidHeaveFloe,
    hex_lattice_floe,
    mean_period,
    upward_crossing_periods,
)
from tensordem.floe_heave_nonlinear import (
    HeaveOscillator,
    exact_heave_period,
)

RADIUS = 0.10           # m
THICKNESS = 1.0         # m (2-D extrusion)
DENSITY_ICE = 917.0     # kg/m^3
N_PARTICLES = 19
DT = 1e-3               # s
AMPLITUDES = (0.05, 0.1, 0.2, 0.3, 0.5)   # in units of particle radius
N_CYCLES = 4
SKIP = 1

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "floe_heave_nonlinear"


def build():
    cfg = WetCouplingConfig(enabled=True, water_density=1025.0, gravity=9.81,
                            wet_drag_coeff=0.0, model="2d")
    offsets = hex_lattice_floe(RADIUS, N_PARTICLES)
    osc = HeaveOscillator.from_config(offsets[:, 1].tolist(), RADIUS, THICKNESS,
                                      DENSITY_ICE, cfg)
    return cfg, offsets, osc


def run_case(cfg, offsets, osc, amplitude, dt, y_eq, n_cycles=N_CYCLES):
    floe = RigidHeaveFloe(offsets, RADIUS, THICKNESS, DENSITY_ICE, cfg, dt,
                          y0=y_eq + amplitude)
    period_guess = osc.linear_period(y_eq)
    n_steps = int(math.ceil((n_cycles + 1) * period_guess / dt))
    traj = floe.run(n_steps)
    periods = upward_crossing_periods(traj.t, traj.y, y_eq)
    measured = mean_period(periods, skip=min(SKIP, max(len(periods) - 1, 0)))
    return {
        "period": measured,
        "periods": periods,
        "cycle_spread": (max(periods) - min(periods)) if periods else float("nan"),
        "n_steps": n_steps,
        "dt": dt,
        "trajectory": {"t": traj.t, "y": traj.y},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, offsets, osc = build()
    y_eq = osc.equilibrium()
    t0 = osc.linear_period(y_eq)

    rows = []
    trajectory = None
    for a in AMPLITUDES:
        amplitude = a * RADIUS
        ref = exact_heave_period(osc, amplitude, y_eq)
        case = run_case(cfg, offsets, osc, amplitude, DT, y_eq)
        if a == AMPLITUDES[-1]:
            trajectory = case["trajectory"]
        rows.append({
            "amplitude_over_R": a,
            "amplitude_m": amplitude,
            "period_measured": case["period"],
            "period_exact": ref.period,
            "period_linear": ref.linear_period,
            "period_series": ref.series_period,
            "rel_error_vs_exact": case["period"] / ref.period - 1.0,
            "linear_error_vs_exact": ref.linear_period / ref.period - 1.0,
            "rel_error_vs_linear": case["period"] / ref.linear_period - 1.0,
            "cycle_spread": case["cycle_spread"],
            "x_lo": ref.x_lo,
            "x_hi": ref.x_hi,
            "energy": ref.energy,
            "n_steps": case["n_steps"],
        })
        print(f"A/R={a:4.2f} T_exact={ref.period:.9f} T_meas={case['period']:.9f} "
              f"rel={case['period'] / ref.period - 1.0:+.3e} lin_err={ref.linear_period / ref.period - 1.0:+.3e}")

    # time-step sensitivity at a mid amplitude (the integrator is 2nd order)
    a_dt = 0.2 * RADIUS
    dt_ref = exact_heave_period(osc, a_dt, y_eq).period
    dt_rows = []
    for dt in (4e-3, 2e-3, DT):
        case = run_case(cfg, offsets, osc, a_dt, dt, y_eq)
        dt_rows.append({"dt": dt, "period": case["period"],
                        "rel_error_vs_exact": case["period"] / dt_ref - 1.0})
        print(f"dt={dt:g} T={case['period']:.9f} rel={dt_rows[-1]['rel_error_vs_exact']:+.3e}")

    # per-particle submerged-fraction "field" at the three characteristic heights
    phases = {}
    for name, y in (("equilibrium", y_eq), ("upper_turning", y_eq + 0.5 * RADIUS),
                    ("lower_turning", y_eq + rows[-1]["x_lo"])):
        f_sub = submerged_fraction_tensor(
            torch.tensor([y + o - cfg.water_level for o in osc.offsets], dtype=torch.float64),
            RADIUS, 0.0).tolist()
        phases[name] = {"y": y, "offsets": list(osc.offsets), "f_sub": f_sub}

    report = {
        "case": {
            "radius_m": RADIUS, "thickness_m": THICKNESS,
            "density_ice_kg_m3": DENSITY_ICE, "n_particles": N_PARTICLES,
            "water_density_kg_m3": cfg.water_density, "gravity_m_s2": cfg.gravity,
            "mass_kg": osc.mass, "y_eq_m": y_eq,
            "T0_linear_s": t0, "k1_N_m": osc.curve_coefficients(y_eq)[0],
            "k2_N_m2": osc.curve_coefficients(y_eq)[1],
            "k3_N_m3": osc.curve_coefficients(y_eq)[2],
            "dt_s": DT, "n_cycles": N_CYCLES, "skip_cycles": SKIP,
        },
        "amplitude_sweep": rows,
        "timestep_sensitivity": dt_rows,
        "phases": phases,
        "trajectory": trajectory,
    }

    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    tables = []
    tables.append("| A/R | amplitude [m] | T exact [s] | T measured [s] | rel. error meas-vs-exact | linear T [s] | linear error |")
    tables.append("|---|---|---|---|---|---|---|")
    for r in rows:
        tables.append(f"| {r['amplitude_over_R']:.2f} | {r['amplitude_m']:.5f} | "
                      f"{r['period_exact']:.6f} | {r['period_measured']:.6f} | "
                      f"{r['rel_error_vs_exact']:+.2e} | {r['period_linear']:.6f} | "
                      f"{r['linear_error_vs_exact']:+.2e} |")
    tables.append("")
    tables.append("| dt [s] | T measured [s] | rel. error vs exact |")
    tables.append("|---|---|---|")
    for r in dt_rows:
        tables.append(f"| {r['dt']:g} | {r['period']:.6f} | {r['rel_error_vs_exact']:+.2e} |")
    (OUT / "tables.md").write_text("\n".join(tables) + "\n")

    try:
        make_figure(report, OUT / "figure.png")
        figure_note = "figure.png"
    except Exception as exc:  # pragma: no cover - environment dependent
        figure_note = f"figure not generated: {exc!r}"
    report["figure"] = figure_note
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT)
    print(figure_note)
    return report


def make_figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["amplitude_sweep"]
    traj = report["trajectory"]
    phases = report["phases"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    ax = axes[0][0]
    a = [r["amplitude_over_R"] for r in rows]
    ax.plot(a, [r["period_measured"] / r["period_linear"] for r in rows], "o-",
            label="measured (IceDEM floe integrator)")
    ax.plot(a, [r["period_exact"] / r["period_linear"] for r in rows], "s--",
            label="exact energy quadrature")
    ax.plot(a, [1.0 for _ in rows], "k:", label="linear theory (A = 2e-3 R anchor)")
    ax.plot(a, [r["period_series"] / r["period_linear"] for r in rows], "^:",
            label="Lindstedt-Poincare series")
    ax.set_xlabel("amplitude A / R")
    ax.set_ylabel("T(A) / T0")
    ax.set_title("Heave period vs amplitude")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.semilogy(a, [abs(r["rel_error_vs_exact"]) for r in rows], "o-",
                label="|measured / exact - 1|")
    ax.semilogy(a, [abs(r["linear_error_vs_exact"]) for r in rows], "s--",
                label="|linear / exact - 1| (rejected)")
    ax.set_xlabel("amplitude A / R")
    ax.set_ylabel("relative error")
    ax.set_title("Solver agreement and the linear-theory error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][0]
    t = traj["t"]
    y = traj["y"]
    y_eq = report["case"]["y_eq_m"]
    last = rows[-1]
    ax.plot(t, y, lw=0.8, label="simulated centroid height")
    ax.axhline(y_eq, color="k", ls=":", lw=0.8, label="equilibrium")
    ax.axhline(y_eq + last["x_hi"], color="r", ls="--", lw=0.8, label="exact upper turning point")
    ax.axhline(y_eq + last["x_lo"], color="b", ls="--", lw=0.8, label="exact lower turning point")
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Trajectory at A = {last['amplitude_over_R']:.2f} R")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    offs = phases["equilibrium"]["offsets"]
    for name, style in (("lower_turning", "b"), ("equilibrium", "k"), ("upper_turning", "r")):
        ax.plot(phases[name]["f_sub"], offs, style + "o-", ms=4, lw=1.0, label=name)
    ax.set_xlabel("submerged fraction $f_{sub}$ per particle")
    ax.set_ylabel("particle y-offset [m]")
    ax.set_title("Submerged-fraction profile at the three phases")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Finite-amplitude heave verification of the 2-D floe buoyancy model")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()