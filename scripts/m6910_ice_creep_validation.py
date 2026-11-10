"""Mission m6910 validation driver: Glen flow law + Norton creep in TensorDEM.

Produces ``artifacts/ice_creep/results.json`` (every number quoted in the
report is written here by the program, no hand-copied digits) and the figures
``fig_norton.png``, ``fig_three_stage.png``, ``fig_punch_relaxation.png``,
``fig_arrhenius.png``.

Run:  PYTHONPATH=src python scripts/m6910_ice_creep_validation.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.ice_creep import (
    GlenCreepLaw,
    LatticeCreepParameters,
    UniaxialCreepModel,
    glen_strain_rate,
    equivalent_strain_rate,
    identify_creep_stages,
    norton_uniaxial_rate,
    power_law_fit,
    punch_relaxation,
    single_bond_relaxation_analytic,
    uniaxial_constant_stress,
    BondedSphereCreepLattice,
)

OUT = os.path.join("artifacts", "ice_creep")
os.makedirs(OUT, exist_ok=True)
results: dict = {"mission": "m6910", "node": "IN13-W3-N02",
                 "repository": "TensorDEM"}
R_GAS = 8.314462618


def fluidity_for(n: float, *, tau_ref: float = 100.0,
                 sigma_ref: float = 1.0e7, young: float = 9.0e9) -> float:
    """Fluidity scaled per stress exponent.

    With ``tau = 1/(E A sigma^(n-1))`` the local bond relaxation time is what
    sets the *stability* of the explicit step and the visibility of creep inside
    a fixed time window.  Holding ``tau_ref`` and ``sigma_ref`` fixed keeps the
    three exponents (n = 1, 3, 5) on the same footing; the quantity under
    verification is the exponent itself.
    """
    return 1.0 / (tau_ref * young * sigma_ref ** (n - 1.0))


# --------------------------------------------------------------------------- #
# (a) Glen tensor form
# --------------------------------------------------------------------------- #
def part_a():
    law = GlenCreepLaw()
    a = law.fluidity(-10.0)
    sigma = np.diag([1.0e6, 0.0, 0.0])
    rate = glen_strain_rate(sigma, a, law.n)
    norton = float(norton_uniaxial_rate(1.0e6, a, law.n))
    rng = np.random.default_rng(20261109)
    m = rng.normal(size=(3, 3))
    sym = 0.5 * (m + m.T) * 1.0e6
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    r0 = glen_strain_rate(sym, a, law.n)
    r1 = glen_strain_rate(q @ sym @ q.T, a, law.n)
    dev = sym - np.trace(sym) / 3.0 * np.eye(3)
    ratio = r0[dev != 0.0] / dev[dev != 0.0]
    return {
        "trace_relative": abs(float(np.trace(rate))) / abs(rate[0, 0]),
        "uniaxial_vs_norton_relative": abs(rate[0, 0] - norton) / norton,
        "lateral_to_axial": float(rate[1, 1] / rate[0, 0]),
        "equivalent_rate_vs_norton_relative":
            abs(equivalent_strain_rate(rate) - norton) / norton,
        "rotation_invariance_max_abs": float(np.abs(r1 - q @ r0 @ q.T).max()),
        "coaxiality_spread_relative":
            float(np.abs(ratio - ratio[0]).max() / abs(ratio[0])),
        "hydrostatic_rate_max_abs": float(
            np.abs(glen_strain_rate(-5.0e6 * np.eye(3), a, law.n)).max()),
    }


# --------------------------------------------------------------------------- #
# (b) Norton power law, stress sweep
# --------------------------------------------------------------------------- #
def part_b():
    sweeps = {}
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sig = np.logspace(4.5, 7.5, 40)
    for n in (1.0, 3.0, 5.0):
        law = GlenCreepLaw(a_ref=fluidity_for(n), n=n)
        a = law.fluidity(-10.0)
        analytic = norton_uniaxial_rate(sig, a, n)
        fit = power_law_fit(sig, analytic)
        # lattice (DEM) exponent at the early-creep window
        params = LatticeCreepParameters(law=law, young_modulus=9.0e9)
        rates, sigmas = [], []
        for s0 in (5.0e5, 1.0e6, 2.0e6):
            out = uniaxial_constant_stress(sigma0=s0, n_cells=3, spacing=0.02,
                                           params=params, t_end=120.0, dt=10.0)
            t, strain = out["t"], out["strain"]
            mask = (t >= 10.0) & (t <= 40.0)
            rates.append(float(np.polyfit(t[mask], strain[mask], 1)[0]))
            sigmas.append(s0)
        lattice_fit = power_law_fit(np.array(sigmas), np.array(rates))
        sweeps[f"n={n:g}"] = {
            "fluidity_a_ref_Pa_n_s": law.a_ref,
            "analytic_exponent": fit["exponent"],
            "analytic_prefactor": fit["prefactor"],
            "lattice_exponent": lattice_fit["exponent"],
            "lattice_r2": lattice_fit["r2"],
            "lattice_rates": rates,
            "lattice_sigmas": sigmas,
            "lattice_creep_rate_over_A_sigma_n": rates[-1] / (
                law.fluidity(-10.0) * 1.0e6 ** n),
        }
        ax.loglog(sig, analytic, "-", label=f"analytic n={n:g}")
        ax.loglog(sigmas, rates, "o", mfc="none",
                  label=f"DEM lattice n={n:g}")
    ax.set_xlabel(r"$\sigma$ [Pa]")
    ax.set_ylabel(r"$\dot\varepsilon$ [s$^{-1}$]")
    ax.set_title("Norton power law: analytic and bonded-sphere DEM")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_norton.png"), dpi=150)
    plt.close(fig)
    return sweeps


# --------------------------------------------------------------------------- #
# (c) three-stage constant-stress creep
# --------------------------------------------------------------------------- #
def part_c():
    model = UniaxialCreepModel(kelvin_time_s=100.0, damage_exponent=4.0,
                               damage_time_s=5.0e5)
    sigma0 = 5.0e5
    t_r = model.time_to_rupture(sigma0)
    t = np.linspace(1.0, 0.95 * t_r, 3000)
    eps = model.strain(t, sigma0)
    rate = model.strain_rate(t, sigma0)
    stages = identify_creep_stages(t, rate, plateau_tolerance=0.02)
    num = model.integrate_rk4(sigma0, 0.9 * t_r, n_steps=4000)
    closed = model.strain(num["t"], sigma0)
    integrated = num["strain"] + sigma0 / model.young_modulus
    rk4_error = float(np.max(np.abs(integrated - closed) / np.abs(closed)))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].loglog(t, eps, "-")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel(r"$\varepsilon$ [-]")
    axes[0].set_title(f"constant stress $\\sigma_0$ = {sigma0/1e6:.2f} MPa")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].loglog(t, rate, "-")
    axes[1].axhline(stages["steady_rate"], ls="--", lw=0.8,
                    label="steady rate (measured)")
    axes[1].axvline(stages["steady_window"][0], ls=":", lw=0.8)
    axes[1].axvline(stages["steady_window"][1], ls=":", lw=0.8)
    if stages["tertiary_start_time"]:
        axes[1].axvline(stages["tertiary_start_time"], ls="-.", lw=0.8,
                        label="tertiary onset")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(r"$\dot\varepsilon$ [s$^{-1}$]")
    axes[1].set_title("primary | steady | tertiary")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_three_stage.png"), dpi=150)
    plt.close(fig)

    norton_rate = model.law.fluidity(-10.0) * sigma0 ** model.law.n
    return {
        "sigma0_Pa": sigma0,
        "time_to_rupture_s": t_r,
        "steady_rate_measured": stages["steady_rate"],
        "steady_rate_norton": norton_rate,
        "steady_rate_relative_error": abs(stages["steady_rate"] / norton_rate - 1.0),
        "steady_window_s": list(stages["steady_window"]),
        "steady_window_decades": stages["steady_window_decades"],
        "tertiary_onset_s": stages["tertiary_start_time"],
        "peak_to_steady_rate_ratio": stages["rate_ratio_peak_steady"],
        "rk4_vs_closed_form_max_relative": rk4_error,
        "strain_at_0.95tr": float(model.strain(0.95 * t_r, sigma0)),
    }


# --------------------------------------------------------------------------- #
# (d) contact force relaxation
# --------------------------------------------------------------------------- #
def part_d():
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    out_d = {}
    for n in (1.0, 3.0, 5.0):
        params = LatticeCreepParameters(law=GlenCreepLaw(a_ref=fluidity_for(n),
                                                         n=n))
        res = punch_relaxation(n_cells=4, spacing=0.02, params=params,
                               depth_fraction=1.0e-3, t_end=2.0e3, dt=2.0)
        t, f = res["t"], res["force"]
        fit = power_law_fit(t[1:], f[1:], x_min=t[-1] / 20.0)
        # window sensitivity: the fitted exponent of a relaxation that is not a
        # pure power law depends on the window, so report three of them
        windows = {}
        for frac in (50.0, 20.0, 10.0):
            w = power_law_fit(t[1:], f[1:], x_min=t[-1] / frac)
            windows[f"window_last_{int(frac)}th"] = {
                "exponent": w["exponent"], "r2": w["r2"]}
        target = -1.0 / (n + 1.0)
        assert res["creep_ratio"].max() < 0.5, "unstable explicit step"
        assert np.all(np.diff(f) <= 1e-6 * f[0]), "force not relaxing monotonically"
        out_d[f"n={n:g}"] = {
            "fluidity_a_ref_Pa_n_s": params.law.a_ref,
            "fitted_exponent": fit["exponent"],
            "fit_r2": fit["r2"],
            "window_sensitivity": windows,
            "target_minus_one_over_n_plus_one": target,
            "relative_deviation_from_target":
                abs(fit["exponent"] - target) / abs(target),
            "alternative_minus_one_over_n": -1.0 / n,
            "relative_deviation_from_minus_one_over_n":
                abs(fit["exponent"] + 1.0 / n) / (1.0 / n),
            "initial_force_N": res["initial_force"],
            "final_force_N": float(f[-1]),
            "force_ratio_final_initial": float(f[-1] / f[0]),
            "creep_step_ratio_max": float(res["creep_ratio"].max()),
            "n_nodes": res["n_nodes"],
            "n_bonds": res["n_bonds"],
            "penetration_m": res["depth"],
        }
        ax.loglog(t[1:], f[1:] / f[0], "-", lw=1.0, label=f"n={n:g} DEM")
        tt = t[1:]
        ax.loglog(tt, (tt / tt[0]) ** target, "k:", lw=0.7,
                  label=f"$t^{{-1/(n+1)}}$, n={n:g}" if n == 3.0 else None)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$F(t)/F_0$ [-]")
    ax.set_title("rigid-plate relaxation at constant penetration")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_punch_relaxation.png"), dpi=150)
    plt.close(fig)

    # single bond: DEM vs closed form (n = 1 and n = 3)
    single = {}
    for n, a_ref, sigma0 in ((1.0, 1.0e-14, 1.0e6), (3.0, 1.0e-24, 7.5e4),
                             (5.0, 1.0e-36, 3.0e4)):
        spacing = 0.02
        params = LatticeCreepParameters(law=GlenCreepLaw(a_ref=a_ref, n=n),
                                        young_modulus=9.0e9)
        lat = BondedSphereCreepLattice(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, spacing]]),
            np.array([[0, 1]]), radius=spacing / 2.0, params=params)
        area = math.pi * (spacing / 2.0) ** 2
        k = params.young_modulus * area / spacing
        for dof in range(6):
            lat.fixed[dof] = True
        lat.fixed_value[5] = -sigma0 * area / k
        lat.solve()
        f0 = abs(lat.bond_axial_force()[0])
        ts, fs = [0.0], [f0]
        for _ in range(1000):
            lat.step(20.0)
            lat.solve()
            ts.append(lat.time)
            fs.append(abs(lat.bond_axial_force()[0]))
        analytic = single_bond_relaxation_analytic(
            f0, k, area, lat.rest[0], lat.params.fluidity(), n, np.array(ts))
        single[f"n={n:g}"] = {
            "initial_force_N": f0,
            "analytic_initial_force_N": sigma0 * area,
            "max_relative_error_vs_closed_form":
                float(np.max(np.abs(np.array(fs) - analytic) / analytic)),
            "final_over_initial": float(fs[-1] / f0),
            "relaxation_exponent_analytic": -1.0 / (n - 1.0) if n > 1 else None,
        }
    return {"plate": out_d, "single_bond": single}


# --------------------------------------------------------------------------- #
# (e) Arrhenius temperature dependence
# --------------------------------------------------------------------------- #
def part_e():
    q = 61500.0
    law = GlenCreepLaw(q_over_r=q / R_GAS, t_ref_c=-10.0)
    temps = np.arange(-2.0, -20.1, -2.0)
    table = []
    for tc in temps:
        table.append({"T_C": float(tc), "A_Pa_n_s": law.fluidity(float(tc)),
                      "edot_at_1MPa": float(law.uniaxial_rate(1.0e6, float(tc)))})
    inv_t = 1.0 / (temps + 273.15)
    ln_a = np.log([row["A_Pa_n_s"] for row in table])
    slope = float(np.polyfit(inv_t, ln_a, 1)[0])
    q_recovered = -slope * R_GAS
    ratio = table[0]["A_Pa_n_s"] / table[-1]["A_Pa_n_s"]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.semilogy(temps, [row["A_Pa_n_s"] for row in table], "o-")
    ax.set_xlabel("T [degC]")
    ax.set_ylabel(r"$A(T)$ [Pa$^{-3}$ s$^{-1}$]")
    ax.set_title("Arrhenius fluidity, ref. $A(-10\\,^\\circ C)$ = 3.5e-24")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_arrhenius.png"), dpi=150)
    plt.close(fig)
    return {"table": table, "Q_J_per_mol_input": q,
            "Q_recovered_J_per_mol": q_recovered,
            "Q_relative_error": abs(q_recovered - q) / q,
            "A_ratio_minus2_over_minus20": float(ratio)}


# --------------------------------------------------------------------------- #
# (f) A-in-MPa sweep quoted by the mission (A in [1e-12, 1e-8] MPa^-n s^-1)
# --------------------------------------------------------------------------- #
def part_f():
    rows = []
    for n in (1.0, 3.0, 5.0):
        for a_mpa in (1e-12, 1e-10, 1e-8):
            a_pa = a_mpa * 10.0 ** (-6.0 * n)   # MPa^-n -> Pa^-n
            law = GlenCreepLaw(a_ref=a_pa, n=n)
            rows.append({"n": n, "A_MPa_n_s": a_mpa, "A_Pa_n_s": a_pa,
                         "edot_at_1MPa": float(law.uniaxial_rate(1.0e6)),
                         "effective_viscosity_Pa_s_at_1MPa":
                             float(1.0e6 / (2.0 * law.uniaxial_rate(1.0e6)))})
    return rows


def main() -> None:
    results["a_glen_tensor"] = part_a()
    results["b_norton"] = part_b()
    results["c_three_stage"] = part_c()
    results["d_relaxation"] = part_d()
    results["e_arrhenius"] = part_e()
    results["f_fluidity_sweep"] = part_f()
    results["artifacts"] = sorted(os.listdir(OUT))
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("b_norton", "e_arrhenius", "f_fluidity_sweep")},
                     indent=2))
    print("wrote", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()