"""Benchmark driver: freezing-front depth profile and freeze-bond strength (P316-W1-N03).

Runs every acceptance case of the node and writes, under
``benchmarks/freeze_bond_thermal_profile/``:

* ``results.json``   — all acceptance numbers, programmatically injected,
* ``figures/*.png``  — front kinetics, profiles, Biot/brine sweeps, energy audit,
* ``profiles.csv``   — the 24 h temperature / strength profile (audit trail),
* ``front_kinetics.csv`` — numerical vs analytical fronts,
* ``docs/P316_W1_N03_freeze_front_thermal_profile.md`` is rendered from
  ``results.json`` by ``build_report_md.py`` (no hand-typed number).

Run:  python benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py
"""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.freeze_bond import thermal_profile as tp  # noqa: E402

OUT = Path(__file__).resolve().parent
CACHE: dict = {}
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ACCEPT = {
    "a_front_vs_stefan_max_rel": 0.02,
    "a_front_vs_neumann_max_rel": 0.005,
    "b_energy_closure_max_rel": 1e-4,
    "b_latent_budget_max_rel": 1e-4,
    "c_uniform_limit_max_rel": 1e-3,
    "c_flat_caliber_max_rel": 1e-12,
    "d_spearman_min": 1.0,
    "d_min_levels": 5,
    "e_monotone": True,
    "e_zero_brine_max_rel": 1e-12,
    "f_front_max_m": 1e-9,
}


# ---------------------------------------------------------------------------
# (a) Stefan / Neumann anchor
# ---------------------------------------------------------------------------
def case_a_anchor() -> dict:
    """Pure-water front vs the quasi-steady Stefan and exact Neumann solutions."""
    rows = []
    for delta_T in (5.0, 15.0):
        thermal = tp.IceThermal(k=2.2)
        for n_cells, dt in ((200, 20.0), (400, 10.0), (800, 5.0)):
            solver = tp.EnthalpyFrontSolver(n_cells=n_cells, domain_length=0.3,
                                            thermal=thermal, band=0.05)
            t_end = 3600.0
            rec = [600.0, 1200.0, 2400.0, 3600.0]
            res = solver.run(T_env=thermal.liquidus_K - delta_T, t_end=t_end, dt=dt,
                             h_conv=None, record_times=rec)
            for i, t in enumerate(res["time"]):
                x_num = float(res["front"][i])
                x_qs = float(tp.stefan_quasi_steady_front(t, thermal, delta_T))
                x_nm = float(tp.neumann_front(t, thermal, delta_T))
                rows.append(dict(delta_T_K=delta_T, n_cells=n_cells, dt_s=dt, t_s=float(t),
                                 x_num_m=x_num, x_int_m=float(res["front_integral"][i]),
                                 x_quasi_m=x_qs, x_neumann_m=x_nm,
                                 rel_vs_quasi=(x_num - x_qs) / x_qs,
                                 rel_vs_neumann=(x_num - x_nm) / x_nm,
                                 rel_int_vs_neumann=(float(res["front_integral"][i]) - x_nm) / x_nm))
    fin = [r for r in rows if r["t_s"] == 3600.0]
    # best resolved set (finest grid) of each DeltaT carries the acceptance number
    best = max([r for r in fin if r["delta_T_K"] == 15.0], key=lambda r: r["n_cells"])
    best_small_ste = [r for r in fin if r["delta_T_K"] == 5.0]
    best_small_ste = max(best_small_ste, key=lambda r: r["n_cells"])
    max_ste = max(abs(best_small_ste["rel_vs_quasi"]),
                  abs(best["rel_vs_quasi"]))
    return dict(rows=rows,
                # headline anchor: small-Ste case is where the quasi-steady
                # formula is itself accurate; reported for both DeltaT
                headline=dict(delta_T_K=best_small_ste["delta_T_K"],
                              x_num_m=best_small_ste["x_num_m"],
                              x_quasi_m=best_small_ste["x_quasi_m"],
                              x_neumann_m=best_small_ste["x_neumann_m"],
                              rel_vs_quasi=best_small_ste["rel_vs_quasi"],
                              rel_vs_neumann=best_small_ste["rel_vs_neumann"]),
                delta_T15=dict(rel_vs_quasi=best["rel_vs_quasi"],
                               rel_vs_neumann=best["rel_vs_neumann"]),
                max_rel_vs_quasi=max_ste,
                max_rel_vs_neumann=max(abs(best_small_ste["rel_vs_neumann"]),
                                       abs(best["rel_vs_neumann"])),
                passed=bool(max_ste <= ACCEPT["a_front_vs_stefan_max_rel"]
                            and max(abs(best_small_ste["rel_vs_neumann"]),
                                    abs(best["rel_vs_neumann"])) <= ACCEPT["a_front_vs_neumann_max_rel"]))


# ---------------------------------------------------------------------------
# (b) Energy audit
# ---------------------------------------------------------------------------
def case_b_energy() -> dict:
    """Stored enthalpy == boundary flux; latent budget in the latent-only limit."""
    thermal = tp.IceThermal(k=2.2)
    solver = tp.EnthalpyFrontSolver(n_cells=400, domain_length=0.3, thermal=thermal)
    res = solver.run(T_env=thermal.liquidus_K - 15.0, t_end=3600.0, dt=10.0, h_conv=None,
                     record_times=[600.0, 1200.0, 2400.0, 3600.0])
    rel = np.abs(res["energy_stored"] - res["energy_flux"]) / np.abs(res["energy_flux"])
    closure = dict(times_s=res["time"].tolist(),
                   stored_J_m2=res["energy_stored"].tolist(),
                   flux_J_m2=res["energy_flux"].tolist(),
                   rel_residual=rel.tolist(), max_rel=float(np.max(rel)))

    # explicit latent / sensible split of the same balance: the released latent
    # heat is rho L * int s dz and the sensible part rho c_p * int (T - T_f) dz;
    # their sum must reproduce the flux integral (this is the statement that the
    # latent heat is released in full -- the predecessor module integrated only
    # L/2 of it).
    th = tp.IceThermal(k=2.2)
    L_dom, n = 0.3, 400
    sol = tp.EnthalpyFrontSolver(n_cells=n, domain_length=L_dom, thermal=th)
    r = sol.run(T_env=th.liquidus_K - 15.0, t_end=3600.0, dt=10.0, h_conv=None,
                record_times=[3600.0])
    T = r["profiles"][-1]
    s_field = tp.solid_fraction(T, th.liquidus_K, 0.05)
    latent = th.rho * th.L * float(np.sum(s_field) * sol.dz)
    sensible = th.rho * th.cp * float(np.sum((T - th.liquidus_K)) * sol.dz)
    flux = float(np.abs(r["energy_flux"][-1]))
    # released heat = latent released (positive) minus the (negative) sensible
    # storage of the cooled ice; the balance is compared against |flux|.
    released = latent - sensible
    latent_budget = dict(latent_J_m2=latent, sensible_J_m2=sensible,
                         released_J_m2=released,
                         flux_J_m2=flux,
                         rel_residual=(released - flux) / flux,
                         raw_latent_vs_flux_rel=(latent - flux) / flux,
                         sensible_fraction=abs(sensible) / released,
                         stefan_number=th.stefan_number(15.0))

    return dict(closure=closure, latent_limit=latent_budget,
                max_rel=float(max(np.max(rel), abs(latent_budget["rel_residual"]))),
                passed=bool(np.max(rel) <= ACCEPT["b_energy_closure_max_rel"]
                            and abs(latent_budget["rel_residual"]) <= ACCEPT["b_latent_budget_max_rel"]))


# ---------------------------------------------------------------------------
# (c) Uniform (m7439) limit
# ---------------------------------------------------------------------------
def case_c_uniform_limit() -> dict:
    ul = tp.uniform_limit_study()
    # numerical cross-check of the same profile family at resolvable k
    cross = []
    for k in (2.2, 22.0, 220.0):
        # Dirichlet surface so the numerical case and the Neumann reference
        # solve *the same* boundary-value problem
        out = tp.run_case(k=k, h_conv=None, delta_T_K=15.0, H=1.0, t_end=86400.0,
                          n_cells=400, dt=60.0, domain_length=4.0)
        th = tp.IceThermal(k=k)
        law = tp.BondStrengthLaw()
        z = np.linspace(0.0, 1.0, 2001)
        T_an = tp.neumann_temperature_profile(z, 86400.0, th, 15.0)
        tau_an = tp.effective_strength(z, T_an, 1.0, law, th)
        ratio_an = tau_an / out["strength"]["tau_uniform_Pa"]
        cross.append(dict(k_W_mK=k, ratio_numeric=out["strength"]["ratio"],
                          ratio_neumann=ratio_an,
                          rel_diff=(out["strength"]["ratio"] - ratio_an) / ratio_an))
    return dict(levels=ul["rows"], flat_profile=ul["flat_profile"],
                rel_diff_at_flat=abs(ul["flat_profile"]["rel_diff"]),
                rel_diff_final=ul["rel_diff_final"],
                k_for_0p1pct=ul["k_for_0p1pct"], biot_for_0p1pct=ul["biot_for_0p1pct"],
                numerical_crosscheck=cross,
                passed=bool(abs(ul["flat_profile"]["rel_diff"]) <= ACCEPT["c_flat_caliber_max_rel"]
                            and abs(1.0 - ul["rows"][-1]["ratio"]) <= ACCEPT["c_uniform_limit_max_rel"]))


# ---------------------------------------------------------------------------
# (d) Baseline deviation and Biot sweep
# ---------------------------------------------------------------------------
def case_d_deviation() -> dict:
    base = {}
    for n_cells in (200, 400, 800):
        c = tp.run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                        n_cells=n_cells, dt=60.0)
        base[n_cells] = c
    c = base[800]
    CACHE["c24"] = c
    grid = [dict(n_cells=n, ratio=base[n]["strength"]["ratio"],
                 tau_eff_Pa=base[n]["strength"]["tau_eff_Pa"],
                 front_m=base[n]["front_m"]["numerical"]) for n in (200, 400, 800)]

    # time-step refinement of the same case (dt = 60 / 20 / 5 s)
    dt_study = []
    for dt in (60.0, 20.0, 5.0):
        cc = tp.run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                         n_cells=400, dt=dt)
        dt_study.append(dict(dt_s=dt, ratio=cc["strength"]["ratio"],
                             tau_eff_Pa=cc["strength"]["tau_eff_Pa"],
                             front_m=cc["front_m"]["numerical"]))
    sweep = tp.run_biot_sweep(
        biot_levels=(0.0044, 0.0088, 0.0176, 0.0352, 0.0704, 0.1408, 0.2816),
        h_conv=500.0, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
        n_cells=400, dt=60.0)
    from scipy.stats import spearmanr

    bis = [r["biot"] for r in sweep["rows"]]
    ratios = [r["ratio"] for r in sweep["rows"]]
    rho, _ = spearmanr(bis, ratios)
    return dict(baseline=dict(
        parameters=c["parameters"],
        front=c["front_m"],
        energy=c["energy"],
        strength=c["strength"],
        T_min_K=c["profile_T_min_K"], T_max_K=c["profile_T_max_K"],
        ratio=c["strength"]["ratio"],
        tau_eff_Pa=c["strength"]["tau_eff_Pa"],
        tau_uniform_Pa=c["strength"]["tau_uniform_Pa"],
        tau_frozen_caliber_Pa=c["strength"]["tau_frozen_caliber_Pa"]),
        grid=[grid][0],
        dt_study=dt_study,
        biot_sweep=dict(rows=sweep["rows"], ratios=ratios,
                        monotone=sweep["monotone_increasing"],
                        monotone_neumann=sweep["monotone_neumann"],
                        spearman=float(rho), n_levels=len(bis)),
        passed=bool(rho >= ACCEPT["d_spearman_min"] and len(bis) >= ACCEPT["d_min_levels"]
                    and sweep["monotone_increasing"]))


# ---------------------------------------------------------------------------
# (e) Brine reduction
# ---------------------------------------------------------------------------
def case_e_brine() -> dict:
    pure = tp.run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                       n_cells=800, dt=60.0)
    # same discretisation as the sweep, so the nu_b = 0 return is bit-for-bit
    pure_same_grid = tp.run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0,
                                 t_end=86400.0, n_cells=400, dt=60.0)
    sweep = tp.run_brine_sweep(brine_levels=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
                               k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0,
                               t_end=86400.0, n_cells=400, dt=60.0)
    zero_rel = ((sweep["rows"][0]["tau_eff_Pa"] - pure_same_grid["strength"]["tau_eff_Pa"])
                / pure_same_grid["strength"]["tau_eff_Pa"])
    # coupled case: salinity enters the thermal problem (liquidus) *and* the
    # Cox-Weeks brine fraction
    saline = tp.run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0, t_end=86400.0,
                         n_cells=800, dt=60.0, salinity_ppt=5.0)
    # Cox-Weeks diagnostic curve
    th = tp.IceThermal(salinity_ppt=5.0)
    temps_C = np.linspace(-20.0, -0.05, 200)
    nu = [tp.local_brine_fraction(th.liquidus_K + t, th) for t in temps_C]
    return dict(
        pure=dict(parameters=pure["parameters"], front=pure["front_m"],
                  strength=pure["strength"], energy=pure["energy"]),
        pure_same_grid_tau_eff_Pa=pure_same_grid["strength"]["tau_eff_Pa"],
        saline=dict(parameters=saline["parameters"], front=saline["front_m"],
                    strength=saline["strength"], energy=saline["energy"]),
        brine_sweep=dict(rows=sweep["rows"], tau_eff_Pa=sweep["tau_eff_Pa"],
                         monotone=sweep["monotone_decreasing"]),
        zero_brine_rel_diff=float(zero_rel),
        cox_weeks=dict(temp_C=temps_C.tolist(), nu_b=nu),
        passed=bool(sweep["monotone_decreasing"] and abs(zero_rel) <= ACCEPT["e_zero_brine_max_rel"]))


# ---------------------------------------------------------------------------
# (f) Negative control
# ---------------------------------------------------------------------------
def case_f_negative_control() -> dict:
    thermal = tp.IceThermal(k=2.2)
    solver = tp.EnthalpyFrontSolver(n_cells=200, domain_length=0.3, thermal=thermal)
    res = solver.run(T_env=thermal.liquidus_K, t_end=86400.0, dt=60.0, h_conv=None,
                     record_times=[86400.0])
    xf = float(res["front"][-1])
    law = tp.BondStrengthLaw()
    T = res["profiles"][-1]
    z = res["z"]
    tau_eff_pure = tp.effective_strength(z, T, 1.0, law, thermal)
    return dict(front_m=xf, tau_eff_Pa=tau_eff_pure,
                T_drift_K=float(np.max(np.abs(T - thermal.liquidus_K))),
                passed=bool(abs(xf) <= ACCEPT["f_front_max_m"]))


# ---------------------------------------------------------------------------
# Baseline-before: the predecessor module (defect record)
# ---------------------------------------------------------------------------
def case_baseline_before() -> dict:
    """Re-measure the predecessor implementation on the same anchor."""
    import math

    out = dict(available=False)
    try:
        from tensordem.stefan_freezing_front import (EnthalpyStefanSolver as OldSolver,
                                                   IceThermalProperties)
    except Exception as exc:  # pragma: no cover
        out["error"] = str(exc)
        return out
    props = IceThermalProperties()
    delta_T = 15.0
    t = 3600.0
    x_qs = math.sqrt(2 * props.k_ice * delta_T * t / (props.rho_ice * props.L_fusion))
    st = OldSolver(n_cells=400, domain_length=0.5, props=props)
    r = st.run(T_env=props.T_melt - delta_T, h_conv=1e9, T_init=props.T_melt,
               t_end=t, dt=20.0, record_times=[t])
    x_old = float(r["freezing_front"][-1])
    out.update(available=True, x_old_m=x_old, x_stefan_m=x_qs,
               rel_err=(x_old - x_qs) / x_qs,
               note="predecessor latent heat was integrated to L/2 (cosine ramp "
                    "amplitude L/(2b) over [-b,b]) and the front was located on the "
                    "cell-below-melting mask, so x_f is ~2x the analytical value")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(a, b, c, d, e, f) -> list[dict]:
    figs = []

    # fig1: front kinetics + grid/time convergence
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = [r for r in a["rows"] if r["delta_T_K"] == 15.0 and r["n_cells"] == 800]
    tt = np.linspace(60.0, 3600.0, 200)
    th = tp.IceThermal(k=2.2)
    ax[0].plot(tt, tp.stefan_quasi_steady_front(tt, th, 15.0), "k--", label="quasi-steady Stefan")
    ax[0].plot(tt, tp.neumann_front(tt, th, 15.0), "r-", lw=1.0, label="exact Neumann")
    ax[0].plot([r["t_s"] for r in rows], [r["x_num_m"] for r in rows], "o",
               label="enthalpy solver (n=800)")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("$x_f$ [m]")
    ax[0].set_title("(a) freezing front: numerical vs analytical anchors")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    fin = [r for r in a["rows"] if r["t_s"] == 3600.0]
    for dT, marker, style_q, style_n in ((5.0, "s", "-", "--"), (15.0, "o", "-", "--")):
        sel = [r for r in fin if r["delta_T_K"] == dT]
        ax[1].plot([r["n_cells"] for r in sel],
                   [100 * r["rel_vs_quasi"] for r in sel], marker=marker,
                   ls=style_q, label=f"$\\Delta T$={dT:g} K vs quasi-steady")
        ax[1].plot([r["n_cells"] for r in sel],
                   [100 * r["rel_vs_neumann"] for r in sel], marker=marker,
                   ls=style_n, label=f"$\\Delta T$={dT:g} K vs Neumann")
    ax[1].axhline(2.0, color="k", ls=":", label="2 % acceptance band")
    ax[1].axhline(0.0, color="gray", lw=0.6)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("n_cells")
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title("(d) mesh convergence of the front at t = 1 h")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig1_front_kinetics.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(dict(name="fig1_front_kinetics.png",
                     caption="Freezing-front kinetics versus the quasi-steady Stefan and exact "
                             "Neumann solutions, and the mesh-convergence of the front error."))

    # fig2: profiles (temperature + strength)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    c24 = CACHE.get("c24") or tp.run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0,
                                          t_end=86400.0, n_cells=800, dt=60.0)
    z, T = c24["z"], c24["T_profile"]
    m = z <= 0.6
    ax[0].plot(z[m], T[m] - 273.15, "b-", label="T(z,t)  numerically")
    zz = np.linspace(0, 0.6, 400)
    ax[0].plot(zz, tp.neumann_temperature_profile(zz, 86400.0, tp.IceThermal(k=2.2), 15.0) - 273.15,
               "r--", lw=1.0, label="Neumann (exact)")
    ax[0].axhline(0.0, color="gray", lw=0.6)
    ax[0].set_xlabel("z [m]")
    ax[0].set_ylabel("T [°C]")
    ax[0].set_title("(b) temperature profile at t = 24 h ($\\Delta T$ = 15 K, H = 1 m)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    sig = c24["sigma_profile"][z <= 1.0] / 1e6
    ax2 = ax[1]
    ax2.plot(z[z <= 1.0], sig, "b-", label="$\\sigma_b(z)$ [MPa]")
    ax2.axhline(c24["strength"]["tau_uniform_Pa"] / 1e6, color="orange", ls="--",
                label="m7439 uniform caliper")
    ax2.axhline(c24["strength"]["tau_eff_Pa"] / 1e6, color="green", ls="-.",
                label="depth-averaged $\\tau_\\mathrm{eff}$")
    ax2.set_xlabel("z [m]")
    ax2.set_ylabel("bond strength [MPa]")
    ax2.set_title("(c) strength profile and the two calipers")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig2_profiles.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(dict(name="fig2_profiles.png",
                     caption="24 h temperature and freeze-bond strength profiles with the uniform "
                             "(m7439) and depth-averaged calipers."))

    # fig3: Biot sweep
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rows = d["biot_sweep"]["rows"]
    ax.plot([r["biot"] for r in rows], [r["ratio"] for r in rows], "o-", color="b",
            label="numerical (Robin, h = 500 W/m²K)")
    ax.plot([r["biot"] for r in rows], [r["ratio_neumann"] for r in rows], "s--",
            color="r", label="exact Neumann profile")
    lev = c["levels"]
    ax.plot([r["biot"] for r in lev], [r["ratio"] for r in lev], "^-", color="green",
            alpha=0.6, label="Neumann family, k → ∞")
    ax.set_xscale("log")
    ax.set_xlabel("Bi = k / (h H)")
    ax.set_ylabel("$\\tau_\\mathrm{eff}/\\tau_\\mathrm{uniform}$")
    ax.set_title("(d) deviation ratio grows monotonically with Bi")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig3_biot_sweep.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(dict(name="fig3_biot_sweep.png",
                     caption="Deviation ratio tau_eff/tau_uniform versus Biot number for the "
                             "numerical solver, the exact Neumann family, and the k → ∞ limit."))

    # fig4: brine
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = e["brine_sweep"]["rows"]
    ax[0].plot([r["brine_fraction"] for r in rows], [r["tau_eff_Pa"] / 1e3 for r in rows],
               "o-", color="purple")
    ax[0].set_xlabel("prescribed brine fraction $\\nu_b$ [-]")
    ax[0].set_ylabel("$\\tau_\\mathrm{eff}$ [kPa]")
    ax[0].set_title("(e) brine reduction of the bond strength")
    ax[0].grid(alpha=0.3)
    ax[1].plot(e["cox_weeks"]["temp_C"], e["cox_weeks"]["nu_b"], "b-")
    ax[1].set_xlabel("T [°C]")
    ax[1].set_ylabel("Cox-Weeks $\\nu_b$ [-]")
    ax[1].set_title("(c') liquid fraction at S = 5 ppt")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig4_brine.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(dict(name="fig4_brine.png",
                     caption="Brine-fraction reduction of the effective strength and the "
                             "Cox-Weeks liquid-fraction curve at S = 5 ppt."))

    # fig5: energy audit
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cl = b["closure"]
    ax.plot(cl["times_s"], np.abs(cl["flux_J_m2"]) / 1e6, "o-", label="|boundary flux integral|")
    ax.plot(cl["times_s"], np.abs(cl["stored_J_m2"]) / 1e6, "x--", label="|stored enthalpy|")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("energy [MJ/m²]")
    ax.set_title("(b) energy audit (max rel. residual %.1e)" % b["max_rel"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig5_energy.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(dict(name="fig5_energy.png",
                     caption="Energy audit: stored enthalpy versus the boundary flux integral "
                             "(conservative discrete scheme)."))
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    out = {"node": "P316-W1-N03",
           "module": "src/tensordem/freeze_bond/thermal_profile.py",
           "generated_by": "benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py",
           "environment": {"python": platform.python_version(),
                           "numpy": np.__version__,
                           "platform": platform.platform()},
           "acceptance_criteria": ACCEPT,
           "artifacts": {
               "module": "src/tensordem/freeze_bond/thermal_profile.py",
               "driver": "benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py",
               "report_md": "docs/P316_W1_N03_freeze_front_thermal_profile.md",
               "report_builder": "benchmarks/freeze_bond_thermal_profile/build_report_md.py",
               "tests": "tests/test_thermal_profile.py",
               "results": "benchmarks/freeze_bond_thermal_profile/results.json",
               "figures": "benchmarks/freeze_bond_thermal_profile/figures/*.png",
               "reproduce": ["python benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py",
                             "python benchmarks/freeze_bond_thermal_profile/build_report_md.py",
                             "python -m pytest tests/test_thermal_profile.py -q"],
           }}
    out["a_anchor"] = case_a_anchor()
    out["b_energy"] = case_b_energy()
    out["c_uniform_limit"] = case_c_uniform_limit()
    out["d_deviation"] = case_d_deviation()
    out["e_brine"] = case_e_brine()
    out["f_negative_control"] = case_f_negative_control()
    out["baseline_before"] = case_baseline_before()
    out["figures"] = make_figures(out["a_anchor"], out["b_energy"], out["c_uniform_limit"],
                                  out["d_deviation"], out["e_brine"],
                                  out["f_negative_control"])
    out["verdict"] = {
        "a": out["a_anchor"]["passed"], "b": out["b_energy"]["passed"],
        "c": out["c_uniform_limit"]["passed"], "d": out["d_deviation"]["passed"],
        "e": out["e_brine"]["passed"], "f": out["f_negative_control"]["passed"],
    }
    out["all_passed"] = all(out["verdict"].values())
    out["runtime_s"] = round(time.time() - t0, 2)

    # audit-trail CSVs
    with (OUT / "front_kinetics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["delta_T_K", "n_cells", "dt_s", "t_s", "x_num_m", "x_int_m",
                    "x_quasi_m", "x_neumann_m", "rel_vs_quasi", "rel_vs_neumann"])
        for r in out["a_anchor"]["rows"]:
            w.writerow([r["delta_T_K"], r["n_cells"], r["dt_s"], r["t_s"], f"{r['x_num_m']:.8f}",
                        f"{r['x_int_m']:.8f}", f"{r['x_quasi_m']:.8f}", f"{r['x_neumann_m']:.8f}",
                        f"{r['rel_vs_quasi']:.3e}", f"{r['rel_vs_neumann']:.3e}"])

    c24 = CACHE.get("c24") or tp.run_case(k=2.2, exponent_n=0.3, delta_T_K=15.0, H=1.0,
                                          t_end=86400.0, n_cells=800, dt=60.0)
    with (OUT / "profiles.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["z_m", "T_K", "T_degC", "sigma_b_Pa"])
        for zi, Ti, si in zip(c24["z"], c24["T_profile"], c24["sigma_profile"]):
            w.writerow([f"{zi:.6e}", f"{Ti:.6f}", f"{Ti - 273.15:.6f}", f"{si:.6e}"])

    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "all_passed": out["all_passed"],
                      "ratio_baseline": out["d_deviation"]["baseline"]["ratio"],
                      "runtime_s": out["runtime_s"]}, indent=1))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())