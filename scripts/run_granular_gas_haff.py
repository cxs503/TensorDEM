"""Haff's-law (homogeneous cooling) benchmark for the TensorDEM contact law.

Runs a monodisperse, frictionless, gravity-free 2-D granular gas in a periodic
box with the engine's compressive-only linear spring-dashpot contact, then

* fits the granular temperature decay to Haff's law ``T = T0/(1+t/tau)**2`` and
  reports the free exponent ``alpha`` (Haff predicts ``alpha = 2``),
* checks the linearity of ``1/sqrt(T)`` in time,
* checks the scaling of the cooling coefficient with ``(1-e**2)`` and with the
  number density,
* reports the measured collision frequency and the prefactor deviation from the
  dilute (Boltzmann) kinetic-theory coefficient, and
* closes the energy ledger (kinetic-energy loss versus contact-damper work).

Outputs (committed as evidence)::

    docs/granular_gas_haff_metrics.json
    docs/granular_gas_haff.png

Usage::

    OMP_NUM_THREADS=1 python scripts/run_granular_gas_haff.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

# These tensors are small; thread oversubscription costs ~40x on 128-core hosts.
torch.set_num_threads(int(os.environ.get("GAS_THREADS", "1")))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.granular_gas import (  # noqa: E402
    GranularGas, GranularGasConfig, fit_haff_exponent,
    fit_inverse_sqrt_temperature, kinetic_theory_cooling_coefficient,
    maxwell_2d_diagnostics)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")


def run_case(*, packing_fraction: float, restitution: float, n_particles: int = 200,
             target_ratio: float = 0.45, seed: int = 3,
             target_overlap_fraction: float = 0.05,
             steps_per_contact: int = 25) -> dict:
    """Cool one gas and return its Haff-law fit plus diagnostics."""
    cfg = GranularGasConfig(n_particles=n_particles,
                            packing_fraction=packing_fraction,
                            restitution=restitution,
                            target_overlap_fraction=target_overlap_fraction,
                            steps_per_contact=steps_per_contact, seed=seed)
    gas = GranularGas(cfg)
    A_kt = kinetic_theory_cooling_coefficient(restitution, cfg.number_density,
                                             cfg.diameter, cfg.mass)
    tau_kt = 2.0 / (A_kt * math.sqrt(cfg.initial_temperature))
    T0, ke0 = gas.kinetic_temperature(), gas.kinetic_energy()
    times, temps = [], []
    wall = time.time()
    # Advance in ~1% of the kinetic-theory Haff time slabs so sampling is uniform
    # in t/tau_kt regardless of the (measured) cooling rate.
    slab = max(1, int(0.01 * tau_kt / cfg.dt))
    while gas.kinetic_temperature() > target_ratio * T0 and gas.time < 60.0 * tau_kt:
        gas.run(slab, sample_every=1)
        times.append(gas.time)
        temps.append(gas.kinetic_temperature())
    wall = time.time() - wall
    # Drop the initial transient (first 15% of the cooling) before fitting.
    lo = next(i for i, T in enumerate(temps) if T <= 0.85 * T0)
    inv = fit_inverse_sqrt_temperature(times[lo:], temps[lo:])
    exp = fit_haff_exponent(times[lo:], temps[lo:], tau=inv["tau_haff"])
    ke_end = gas.kinetic_energy()
    lost = ke0 - ke_end
    nu_measured = 2.0 * gas.collision_onsets / (cfg.n_particles * gas.time)
    nu_kt = cfg.number_density * cfg.diameter * math.sqrt(
        math.pi * 0.5 * (T0 + temps[-1]) / cfg.mass)
    return {
        "packing_fraction": packing_fraction,
        "restitution": restitution,
        "n_particles": n_particles,
        "number_density": cfg.number_density,
        "dt": cfg.dt,
        "stiffness": cfg.stiffness,
        "damping": cfg.damping,
        "contact_time": cfg.contact_time,
        "steps": gas.steps,
        "simulated_time": gas.time,
        "wall_seconds": wall,
        "temperature_ratio_end": temps[-1] / T0,
        "A_measured": inv["A"],
        "A_kinetic_theory": A_kt,
        "A_ratio": inv["A"] / A_kt,
        "haff_alpha": exp["alpha"],
        "haff_r2": exp["r_squared"],
        "inverse_sqrt_r2": inv["r_squared"],
        "tau_haff": inv["tau_haff"],
        "tau_kinetic_theory": tau_kt,
        "collision_frequency_ratio": nu_measured / nu_kt,
        "energy_ledger_gap": abs(lost - gas.damping_work) / max(lost, 1e-300),
        "maxwell": maxwell_2d_diagnostics(gas.velocities, cfg.mass),
        "history": {"time_over_tau_kt": [t / tau_kt for t in times],
                    "temperature_over_T0": [T / T0 for T in temps]},
    }


def elastic_control(n_particles: int = 150, steps: int = 3000) -> dict:
    """Negative control: ``e = 1`` must conserve kinetic energy."""
    cfg = GranularGasConfig(n_particles=n_particles, packing_fraction=0.25,
                            restitution=1.0, steps_per_contact=25, seed=5)
    gas = GranularGas(cfg)
    ke0 = gas.kinetic_energy()
    gas.run(steps)
    ke1 = gas.kinetic_energy()
    return {"restitution": 1.0, "steps": steps,
            "relative_energy_drift": abs(ke1 - ke0) / ke0,
            "damping_work": gas.damping_work,
            "collision_onsets": gas.collision_onsets}


def plot(cases: dict, phi_scan: list[dict], elastic: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # (a) cooling curves with the Haff prediction for the fitted tau
    ax = axes[0][0]
    for e, case in sorted(cases.items(), reverse=True):
        hist = case["history"]
        ax.plot(hist["time_over_tau_kt"], hist["temperature_over_T0"],
                lw=1.2, label="e=%.2f" % e)
    grid = [0.005 * i for i in range(400)]
    tau_ratio = cases[0.7]["tau_kinetic_theory"] / cases[0.7]["tau_haff"]
    ax.plot(grid, [1.0 / (1.0 + x * tau_ratio) ** 2 for x in grid],
            "k--", lw=1.0,
            label="Haff $(1+t/\\tau_{fit})^{-2}$, $\\tau_{fit}$ from e=0.7")
    ax.set_yscale("log")
    ax.set_xlim(0, max(cases[0.7]["history"]["time_over_tau_kt"]) * 1.05)
    ax.set_xlabel("$t/\\tau_{KT}$")
    ax.set_ylabel("$T/T_0$")
    ax.set_title("(a) homogeneous cooling, $(1+t/\\tau)^{-2}$ reference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Haff's law in its linear form, 1/sqrt(T) versus t
    ax = axes[0][1]
    for e, case in sorted(cases.items(), reverse=True):
        hist = case["history"]
        t = torch.tensor(hist["time_over_tau_kt"])
        T = torch.tensor(hist["temperature_over_T0"])
        y = 1.0 / torch.sqrt(T)
        ax.plot(t.numpy(), y.numpy(), lw=1.2, label="e=%.2f ($\\alpha$=%.2f)"
                % (e, case["haff_alpha"]))
        design = torch.stack((torch.ones_like(t), t), dim=1)
        sol = torch.linalg.lstsq(design, y).solution
        ax.plot(t.numpy(), (design @ sol).numpy(), "k:", lw=0.8)
    ax.set_xlabel("$t/\\tau_{KT}$")
    ax.set_ylabel("$1/\\sqrt{T/T_0}$")
    ax.set_title("(b) linearity of $1/\\sqrt{T}$ (Haff form)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) cooling coefficient versus (1-e^2)
    ax = axes[1][0]
    x = [1.0 - case["restitution"] ** 2 for case in cases.values()]
    y = [case["A_measured"] for case in cases.values()]
    kt = [case["A_kinetic_theory"] for case in cases.values()]
    ax.plot(x, y, "o", ms=5, label="measured $A$")
    ax.plot(x, kt, "s", ms=4, mfc="none", label="dilute Boltzmann $A_{kt}$")
    if len(x) > 1:
        slope = sum(xi * yi for xi, yi in zip(x, y)) / sum(xi * xi for xi in x)
        ax.plot([0, max(x)], [0, slope * max(x)], "k--", lw=0.8,
                label="fit $A\\propto(1-e^2)$, $A/(1-e^2)$=%.1f" % slope)
    ax.set_xlabel("$1-e^2$")
    ax.set_ylabel("$A$ [1/(s$\\cdot$J$^{1/2}$)]")
    ax.set_title("(c) rate scales with the dissipation factor")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (d) prefactor deviation versus packing fraction (honest diagnostic)
    ax = axes[1][1]
    phi = [c["packing_fraction"] for c in phi_scan]
    ax.plot(phi, [c["A_ratio"] for c in phi_scan], "o-", ms=4,
            label="$A/A_{kt}$")
    ax.plot(phi, [c["collision_frequency_ratio"] for c in phi_scan], "s--", ms=4,
            label="$\\nu/\\nu_{kt}$ (measured events)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("packing fraction $\\phi$")
    ax.set_ylabel("ratio to dilute kinetic theory")
    ax.set_title("(d) prefactor deviation (dilute limit NOT validated)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("TensorDEM granular gas: Haff's law verification "
                 "(2-D periodic, frictionless, e=0.7 unless noted)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(OUT_DIR, "granular_gas_haff.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)
    return path


def main() -> int:
    metrics: dict = {"cases_by_restitution": {}, "phi_scan": [], "elastic_control": {}}

    for e in (0.6, 0.7, 0.8):
        case = run_case(packing_fraction=0.18, restitution=e)
        metrics["cases_by_restitution"][str(e)] = case
        print("e=%.2f  A=%.2f A_kt=%.2f ratio=%.2f alpha=%.3f (R2=%.5f) "
              "1/sqrtT R2=%.5f  nu/nu_kt=%.2f ledger=%.2e wall=%.1fs" % (
                  e, case["A_measured"], case["A_kinetic_theory"], case["A_ratio"],
                  case["haff_alpha"], case["haff_r2"], case["inverse_sqrt_r2"],
                  case["collision_frequency_ratio"], case["energy_ledger_gap"],
                  case["wall_seconds"]))

    for phi in (0.05, 0.10, 0.18, 0.30):
        case = run_case(packing_fraction=phi, restitution=0.7)
        metrics["phi_scan"].append(case)
        print("phi=%.2f A=%.2f A/A_kt=%.2f alpha=%.3f 1/sqrtT R2=%.5f wall=%.1fs" % (
            phi, case["A_measured"], case["A_ratio"], case["haff_alpha"],
            case["inverse_sqrt_r2"], case["wall_seconds"]))

    metrics["elastic_control"] = elastic_control()
    print("elastic control: relative energy drift %.2e over %d steps" % (
        metrics["elastic_control"]["relative_energy_drift"],
        metrics["elastic_control"]["steps"]))

    # Density scaling of A at fixed e.
    phi_cases = metrics["phi_scan"]
    ratios = [c["A_measured"] / c["number_density"] for c in phi_cases]
    metrics["A_over_density"] = ratios
    metrics["A_over_density_spread"] = (max(ratios) - min(ratios)) / (
        sum(ratios) / len(ratios))

    plot({float(k): v for k, v in metrics["cases_by_restitution"].items()},
         phi_cases, metrics["elastic_control"])

    path = os.path.join(OUT_DIR, "granular_gas_haff_metrics.json")
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())