"""H1-BROKEN-ICE-DEM · analytic verification of the heave transient (verifier seat).

What this adds that the merged suite does not have
-------------------------------------------------
``tests/test_h1_broken_ice_dem.py`` checks the *endpoint* of the buoyancy case:
the settled submerged fraction must equal ``rho_ice/rho_water`` (Archimedes).
That is a static statement -- a wrong restoring stiffness, a wrong drag
coefficient or a broken energy ledger can all still relax to the right draft.

For a prismatic floe (vertical sides, constant waterplane area ``A``) released
from rest at the waterline the **transient** is analytic:

    .. math:: m \\ddot z = -\\rho_w g A (z - z_{eq}) - c \\dot z

i.e. a damped harmonic oscillator with

    .. math:: \\omega_0 = \\sqrt{\\rho_w g A / m}, \\quad
              \\gamma = c/(2m), \\quad
              \\omega_d = \\omega_0\\sqrt{1-\\zeta^2}, \\quad \\zeta = \\gamma/\\omega_0 .

The case builder chooses ``c = f_d \\cdot 2 m \\omega_0`` (``f_d`` = damping
fraction), so **every** coefficient of the reference solution is fixed by the
case set-up before the solver runs.  Fitting the measured ``z(t)`` therefore
measures the solver's effective stiffness, effective damping and energy ledger,
independently of the endpoint value:

1. **equilibrium** ``z_eq = h/2 - (rho_i/rho_w) h``  (static, Archimedes)
2. **damped frequency** ``omega_d`` (hence ``k_fit = m (omega_d^2 + gamma^2)``)
   vs ``rho_w g A`` -- catches a wrong waterplane force or a wrong mass.
3. **decay rate** ``gamma_fit = c_fit/(2m)`` vs the imposed ``c/(2m)`` --
   catches a drag closure applied with the wrong sign / wrong midpoint velocity.
4. **ledger** ``sum(c v^2 dt)`` (trapezoidal, from the recorded velocities) vs
   the simulator's own ``dissipated`` counter and vs the analytic energy loss
   ``E(0) - E(t)``.
5. **time-step refinement**: halving ``dt`` must reduce the frequency error
   (second-order leapfrog), reported as a convergence table, not asserted.

Everything printed is produced by the run; the JSON is the evidence artifact.

    PYTHONPATH=src python scripts/h1_heave_analytic_verification.py
    PYTHONPATH=src python scripts/h1_heave_analytic_verification.py --no-figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.polyhedron_floe3d.contact import ContactLaw          # noqa: E402
from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody  # noqa: E402
from tensordem.polyhedron_floe3d.geometry import equivalent_disc_prism  # noqa: E402
from tensordem.polyhedron_floe3d.h1_cases import (                  # noqa: E402
    SEA_ICE_DENSITY,
    SEA_WATER_DENSITY,
)

OUT_DIR = os.path.join(ROOT, "artifacts", "h1_broken_ice_dem")
OUT_JSON = os.path.join(OUT_DIR, "heave_analytic_verification.json")
OUT_PNG = os.path.join(OUT_DIR, "fig_heave_analytic.png")

PROTOCOL = "tensordem-h1-heave-analytic-verification-v1"


def run_transient(*, footprint_area=0.36, thickness=0.12, n_sides=16,
                  rho_ice=SEA_ICE_DENSITY, rho_water=SEA_WATER_DENSITY,
                  drag_fraction=0.7, dt=5.0e-4, n_steps=4000,
                  gravity=9.81, record_every=1):
    """Integrated heave transient of one freely floating prism floe."""
    volume = footprint_area * thickness
    shape = equivalent_disc_prism(volume, thickness, n_sides)
    body = FloeBody(shape, rho_ice, position=[0.0, 0.0, 0.0], label="floe")
    mass = float(body.mass)
    waterplane = float(shape.volume / thickness)          # prism: A = V / h
    omega0 = math.sqrt(rho_water * gravity * waterplane / mass)
    c = drag_fraction * 2.0 * mass * omega0
    system = DEMSystem3D([body], law=ContactLaw(), gravity=gravity,
                         water_level=0.0, water_density=rho_water, dt=dt,
                         linear_drag=c, angular_drag=0.3 * c, record=False)
    system.record_initial_energy()
    t = np.empty(int(n_steps) // record_every + 1)
    z = np.empty_like(t)
    v = np.empty_like(t)
    diss = np.empty_like(t)
    t[0], z[0] = 0.0, float(body.position[2])
    v[0], diss[0] = float(body.velocity[2]), float(system.dissipated)
    k = 0
    for step in range(1, int(n_steps) + 1):
        system.step()
        if step % record_every == 0:
            k += 1
            t[k] = float(system.time)
            z[k] = float(body.position[2])
            v[k] = float(body.velocity[2])
            diss[k] = float(system.dissipated)
    z_eq_analytic = 0.5 * thickness - (rho_ice / rho_water) * thickness
    return {
        "t": t[:k + 1], "z": z[:k + 1], "v": v[:k + 1], "diss": diss[:k + 1],
        "mass": mass, "waterplane_area": waterplane, "omega0": omega0,
        "c": c, "zeta": drag_fraction, "z_eq_analytic": z_eq_analytic,
        "k_analytic": rho_water * gravity * waterplane,
        "energy_residual": float(system.energy_residual()),
        "dt": dt, "thickness": thickness, "volume": volume,
        "rho_ice": rho_ice, "rho_water": rho_water,
    }


def fit_oscillator(t, z, z_eq):
    """Least-squares fit of ``z_eq + A exp(-gamma t) cos(omega t + phi)``."""
    from scipy.optimize import curve_fit

    y = z - z_eq
    amp = float(np.max(np.abs(y)))

    def model(tt, a, gam, om, phi, off):
        return a * np.exp(-gam * tt) * np.cos(om * tt + phi) + off

    # initial guesses: extrema give the first half period
    sign = np.sign(y)
    crossings = np.where(np.diff(sign) != 0)[0]
    period0 = 2.0 * (t[crossings[1]] - t[crossings[0]]) if len(crossings) > 1 else 0.9
    omega_guess = 2.0 * math.pi / max(period0, 1e-6)
    p0 = [amp, 0.3 * omega_guess, omega_guess, 0.0, 0.0]
    popt, _ = curve_fit(model, t, y, p0=p0, maxfev=40000)
    perr = None
    return {
        "amplitude_m": float(popt[0]),
        "gamma_per_s": float(popt[1]),
        "omega_d_rad_per_s": float(popt[2]),
        "phi_rad": float(popt[3]),
        "offset_m": float(popt[4]),
        "cov_diag": None if perr is None else perr.tolist(),
    }


def analyse(run):
    fit = fit_oscillator(run["t"], run["z"], run["z_eq_analytic"])
    m = run["mass"]
    gamma_fit, omega_d_fit = fit["gamma_per_s"], fit["omega_d_rad_per_s"]
    omega0_fit = math.hypot(omega_d_fit, gamma_fit)      # omega0^2 = omega_d^2 + gamma^2
    k_fit = m * omega0_fit ** 2
    c_fit = 2.0 * m * gamma_fit
    # ledger: analytically integrated drag work vs the simulator's counter
    t, v, diss = run["t"], run["v"], run["diss"]
    drag_work = float(np.trapezoid(run["c"] * v ** 2, t))
    ledger = float(diss[-1])
    z_settled = float(np.mean(run["z"][-50:]))
    return {
        "fit": fit,
        "derived": {
            "omega0_fit_rad_per_s": omega0_fit,
            "k_fit_N_per_m": k_fit,
            "c_fit_N_s_per_m": c_fit,
            "zeta_fit": gamma_fit / omega0_fit,
        },
        "reference": {
            "omega0_analytic_rad_per_s": run["omega0"],
            "k_analytic_N_per_m": run["k_analytic"],
            "c_analytic_N_s_per_m": run["c"],
            "zeta_analytic": run["zeta"],
            "z_eq_analytic_m": run["z_eq_analytic"],
        },
        "errors_pct": {
            "omega0": 100.0 * abs(omega0_fit - run["omega0"]) / run["omega0"],
            "gamma": 100.0 * abs(gamma_fit - 0.5 * run["c"] / m) / (0.5 * run["c"] / m),
            "stiffness": 100.0 * abs(k_fit - run["k_analytic"]) / run["k_analytic"],
            "damping": 100.0 * abs(c_fit - run["c"]) / run["c"],
            "equilibrium_z": 100.0 * abs(z_settled - run["z_eq_analytic"])
                             / abs(run["z_eq_analytic"]),
        },
        "ledger": {
            "drag_work_integrated_J": drag_work,
            "dissipated_counter_J": ledger,
            "ledger_vs_integral_pct": 100.0 * abs(ledger - drag_work) / max(abs(drag_work), 1e-12),
            "energy_residual_dimensionless": run["energy_residual"],
        },
        "measurement": {
            "z_settled_m": z_settled,
            "n_samples": int(run["t"].size),
            "dt_s": run["dt"],
        },
    }


def convergence_table(base_kwargs):
    rows = []
    for div in (1, 2, 4):
        run = run_transient(dt=base_kwargs["dt"] / div,
                            n_steps=int(base_kwargs["n_steps"] * div), **{
                                k: v for k, v in base_kwargs.items()
                                if k not in ("dt", "n_steps")})
        a = analyse(run)
        rows.append({
            "dt_s": run["dt"],
            "omega0_error_pct": a["errors_pct"]["omega0"],
            "gamma_error_pct": a["errors_pct"]["gamma"],
            "equilibrium_error_pct": a["errors_pct"]["equilibrium_z"],
        })
    return rows


def make_figure(run, analysis, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, z = run["t"], run["z"]
    m = run["mass"]
    gamma = 0.5 * run["c"] / m
    omega0 = run["omega0"]
    omega_d = omega0 * math.sqrt(1.0 - run["zeta"] ** 2)
    x0 = -run["z_eq_analytic"]
    analytic = run["z_eq_analytic"] + x0 * np.exp(-gamma * t) * (
        np.cos(omega_d * t) + (gamma / omega_d) * np.sin(omega_d * t))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(t, 1e3 * z, lw=1.4, label="DEM integration")
    ax.plot(t, 1e3 * analytic, "--", lw=1.2,
            label=r"analytic $m\ddot z=-\rho_w gA(z-z_{eq})-c\dot z$")
    ax.axhline(1e3 * run["z_eq_analytic"], color="0.5", lw=0.8, ls=":",
               label="analytic $z_{eq}$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("centroid heave z [mm]")
    ax.set_title("floe heave transient vs closed form")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    env = np.abs(z - run["z_eq_analytic"])
    keep = env > 1e-9
    ax.semilogy(t[keep], 1e3 * env[keep], ".", ms=2, label="|z - z_eq|")
    ax.semilogy(t[keep], 1e3 * np.abs(x0) * np.exp(-analysis["fit"]["gamma_per_s"] * t[keep]),
                "-", lw=1.2, label=r"fit $|A|e^{-\gamma_{fit}t}$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("envelope [mm]")
    title = (r"$\gamma$ err %.2f %%, $\omega_0$ err %.2f %%"
             % (analysis["errors_pct"]["gamma"], analysis["errors_pct"]["omega0"]))
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=5.0e-4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-convergence", action="store_true")
    args = ap.parse_args()

    base = dict(footprint_area=0.36, thickness=0.12, n_sides=16,
                drag_fraction=0.7, dt=args.dt, n_steps=args.steps)
    run = run_transient(**base)
    analysis = analyse(run)
    payload = {
        "protocol": PROTOCOL,
        "setup": {k: v for k, v in base.items()},
        "model": {
            "mass_kg": run["mass"], "waterplane_area_m2": run["waterplane_area"],
            "linear_drag_N_s_per_m": run["c"], "zeta": run["zeta"],
        },
        "analysis": analysis,
        "convergence": None if args.no_convergence else convergence_table(base),
        "gates": {
            "stiffness_err_pct_lt_3": analysis["errors_pct"]["stiffness"] < 3.0,
            "damping_err_pct_lt_3": analysis["errors_pct"]["damping"] < 3.0,
            "equilibrium_err_pct_lt_1": analysis["errors_pct"]["equilibrium_z"] < 1.0,
            "ledger_vs_integral_pct_lt_2": analysis["ledger"]["ledger_vs_integral_pct"] < 2.0,
            "energy_residual_lt_1e-2": analysis["ledger"]["energy_residual_dimensionless"] < 1e-2,
        },
    }
    payload["all_gates_pass"] = all(payload["gates"].values())
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.no_figures:
        make_figure(run, analysis, OUT_PNG)
        payload["figure"] = os.path.relpath(OUT_PNG, ROOT)
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print(json.dumps({
        "errors_pct": analysis["errors_pct"],
        "ledger": analysis["ledger"],
        "convergence": payload["convergence"],
        "gates": payload["gates"],
        "all_gates_pass": payload["all_gates_pass"],
        "written": os.path.relpath(OUT_JSON, ROOT),
    }, indent=1, default=float))
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())