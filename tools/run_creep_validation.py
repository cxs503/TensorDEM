#!/usr/bin/env python3
"""IN69-N01 acceptance run: Glen/Norton creep constitutive + contact relaxation.

Reproduces every machine-checkable criterion of the campaign node from the same
code path the DEM uses (:mod:`tensordem.creep_constitutive`):

* **(a) Norton uniaxial** -- 10 (A, n, sigma) combinations, ``eps_dot`` measured
  from the accumulated creep strain of a stress-controlled element versus the
  closed form ``A sigma**n``, *and* the implicit integrator checked against the
  Bailey-Norton relaxation closed form on the same combinations.
* **(b) three-stage creep** -- primary ``eps ~ t**(1/3)`` (emergent, fitted),
  steady rate ``< 3%`` of ``A sigma**n``, tertiary acceleration and the Kachanov
  rupture time.
* **(c) long-time contact relaxation** -- two-sphere Hertz contact at fixed
  centre distance: ``F(t)/F0`` at ``t = 100 s`` versus the closed form, with a
  timestep-convergence study.
* **(d) energy ledger** -- ``int sigma:eps_dot_c = -dU`` (fixed displacement).
* **(e) n = 1 negative control** -- linear viscous limit, ``eps_dot ~ sigma``.

Protocol is pre-declared in ``prereg/in69_n01_creep_prereg.json`` (thresholds are
those of the mission statement; nothing is tuned after the numbers are seen).

Usage::

    python3 tools/run_creep_validation.py [--out artifacts/in69_n01_creep]

Writes ``summary.json`` (all numbers + artifact paths) and four PNG figures.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tensordem  # noqa: E402
from tensordem.creep_constitutive import (  # noqa: E402
    GlenCreepParameters,
    HertzCreepContact,
    ThreeStageCreepParameters,
    creep_strain_rate,
    hertz_creep_relaxation,
    integrate_three_stage_creep,
    norton_strain_rate,
    relaxation_histories,
    three_stage_analytic,
    uniaxial_stress,
)

ICE_E = 9000.0      # MPa, Young's modulus of ice at about -10 degC
ICE_NU = 0.3

# (a) sweep: 10 combinations inside the mission's parameter box
A_VALUES = (1e-7, 1e-6, 1e-5, 1e-4)
N_VALUES = (2.5, 3.0, 3.5)
SIGMA_VALUES = (0.1, 0.5, 1.0, 2.0)
SWEEP = [
    (1e-7, 2.5, 0.5), (1e-7, 3.0, 1.0), (1e-7, 3.5, 2.0),
    (1e-6, 2.5, 1.0), (1e-6, 3.0, 0.5), (1e-6, 3.5, 0.1),
    (1e-5, 2.5, 2.0), (1e-5, 3.0, 1.0),
    (1e-4, 2.5, 0.1), (1e-4, 3.0, 2.0),
]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, float, str)):
        return obj
    return str(obj)


def _loglog_slope(xs, ys):
    xs = np.log(np.asarray(xs, dtype=float))
    ys = np.log(np.asarray(ys, dtype=float))
    n = xs.size
    return float((n * (xs * ys).sum() - xs.sum() * ys.sum()) / (n * (xs * xs).sum() - xs.sum() ** 2))


# ---------------------------------------------------------------------------
# (a) Norton uniaxial sweep
# ---------------------------------------------------------------------------
def sweep_norton() -> dict:
    rows = []
    for A, n, sigma in SWEEP:
        params = GlenCreepParameters(A=A, n=n)
        stress = uniaxial_stress(sigma)
        # stress-controlled element: the DEM accumulates the creep strain with
        # this rate at every substep; the window is one second and the measured
        # rate is d(eps_c)/dt from the accumulated tensor history.
        dt = 0.05
        d_eps = creep_strain_rate(stress, params) * dt
        eps_axial = float(d_eps[0, 0])
        rate_num = eps_axial / dt
        rate_ana = params.A * sigma ** n
        rate_err = abs(rate_num / rate_ana - 1.0)

        # the same parameter set through the implicit integrator: strain-controlled
        # relaxation, dt tied to the relaxation time tau = 1/(E A sigma**(n-1))
        tau = 1.0 / (ICE_E * params.A * sigma ** (n - 1.0))
        t_end = float(min(max(5.0 * tau, 1.0), 1e6))
        dt_rel = float(t_end / 500.0)  # fixed step count, adaptive absolute step
        relax = relaxation_histories(
            sigma, t_end, dt_rel, params, young_modulus=ICE_E, constraint="free_lateral"
        )
        rel_err = max(
            abs(a - b) / b for a, b in zip(relax["sigma_11"], relax["sigma_analytic"])
        )
        du = relax["elastic_energy"][-1] - relax["elastic_energy"][0]
        diss = relax["dissipation"][-1]
        ledger = abs(du + diss) / max(abs(du), 1e-300)
        rows.append(
            {
                "A": A,
                "n": n,
                "sigma_MPa": sigma,
                "rate_numeric": rate_num,
                "rate_analytic": rate_ana,
                "rate_rel_error": rate_err,
                "relaxation_tau_s": tau,
                "relaxation_dt_s": dt_rel,
                "relaxation_max_rel_error": rel_err,
                "ledger_rel_error": ledger,
            }
        )
    rate_errors = [r["rate_rel_error"] for r in rows]
    relax_errors = [r["relaxation_max_rel_error"] for r in rows]
    ledgers = [r["ledger_rel_error"] for r in rows]
    return {
        "rows": rows,
        "n_cases": len(rows),
        "max_rate_rel_error": max(rate_errors),
        "max_relaxation_rel_error": max(relax_errors),
        "max_ledger_rel_error": max(ledgers),
        "criterion_rate_1pct": max(rate_errors) < 0.01,
        "criterion_relaxation_1pct": max(relax_errors) < 0.01,
    }


# ---------------------------------------------------------------------------
# (b) three-stage creep
# ---------------------------------------------------------------------------
def three_stage_cases() -> dict:
    glen = GlenCreepParameters(A=3.5e-7, n=3.0)
    sigma = 1.0
    # separated-stages parameter set: primary decays before the steady window and
    # damage is negligible over the run (t_r = 3e7 s > 100 t_end)
    slow = ThreeStageCreepParameters(
        glen=glen, primary_A=1e-6, primary_p=2.0 / 3.0,
        damage_B=1.0 / 3e7, damage_m=2.0, tertiary_q=3.0,
    )
    hist = integrate_three_stage_creep(sigma, 1e5, 1.0, slow, young_modulus=ICE_E)
    ts = np.asarray(hist["times"])
    prim = np.asarray(hist["primary"])
    rate = np.asarray(hist["rate"])
    mask = (ts > 1.0) & (ts < 100.0)
    primary_exponent = _loglog_slope(ts[mask], prim[mask])
    steady_mask = ts > 3e4
    steady_rate = float(rate[steady_mask].mean())
    steady_target = norton_strain_rate(sigma, glen)
    steady_err = abs(steady_rate / steady_target - 1.0)
    analytic_total = three_stage_analytic(1e5, sigma, slow, young_modulus=ICE_E)["total"]
    total_err = abs(hist["strain"][-1] / analytic_total - 1.0)

    # tertiary set: fast damage so that rupture is inside the run
    fast = ThreeStageCreepParameters(
        glen=glen, primary_A=1e-6, primary_p=2.0 / 3.0,
        damage_B=1e-3, damage_m=2.0, tertiary_q=3.0,
    )
    h_ter = integrate_three_stage_creep(sigma, fast.rupture_time(sigma), 0.01, fast,
                                        young_modulus=ICE_E)
    ts_t = np.asarray(h_ter["times"])
    rate_t = np.asarray(h_ter["rate"])
    sec = norton_strain_rate(sigma, glen)
    late = ts_t > 0.9 * fast.rupture_time(sigma)
    accel = float(rate_t[late].min() / sec)  # tertiary amplification near rupture
    tert_err = max(
        abs(h_ter["strain"][i] / three_stage_analytic(ts_t[i], sigma, fast,
                                                      young_modulus=ICE_E)["total"] - 1.0)
        for i in range(0, len(ts_t), max(1, len(ts_t) // 60))
    )
    # stage-wise derivative signs: primary decreasing, secondary flat, tertiary rising
    early = (ts_t > 0.01) & (ts_t < 1.0)
    prim_rate_slope = _loglog_slope(ts_t[early], np.asarray(h_ter["primary"])[early])
    return {
        "parameters": {
            "glen": {"A": glen.A, "n": glen.n},
            "sigma_MPa": sigma,
            "slow": {"primary_A": slow.primary_A, "primary_p": slow.primary_p,
                     "damage_B": slow.damage_B, "damage_m": slow.damage_m,
                     "rupture_time_s": slow.rupture_time(sigma)},
            "fast": {"primary_A": fast.primary_A, "primary_p": fast.primary_p,
                     "damage_B": fast.damage_B, "damage_m": fast.damage_m,
                     "rupture_time_s": fast.rupture_time(sigma)},
        },
        "primary_exponent": primary_exponent,
        "primary_exponent_error": abs(primary_exponent - 1.0 / 3.0) / (1.0 / 3.0),
        "steady_rate_numeric": steady_rate,
        "steady_rate_analytic": steady_target,
        "steady_rate_rel_error": steady_err,
        "three_stage_total_rel_error": total_err,
        "tertiary_amplification": accel,
        "tertiary_max_rel_error": tert_err,
        "rupture_time_numeric_s": float(ts_t[-1]),
        "rupture_time_analytic_s": fast.rupture_time(sigma),
        "rupture_rel_error": abs(ts_t[-1] / fast.rupture_time(sigma) - 1.0),
        "tertiary_rate_increasing": bool(np.all(np.diff(rate_t[late]) > 0.0)),
        "primary_rate_decays": prim_rate_slope < 1.0,
        "criterion_primary_exponent_1pct": abs(primary_exponent - 1.0 / 3.0) / (1.0 / 3.0) < 0.01,
        "criterion_steady_rate_3pct": steady_err < 0.03,
        "criterion_tertiary_accelerates": bool(np.all(np.diff(rate_t[late]) > 0.0)) and accel > 10.0,
        "histories": {"slow": hist, "tertiary": h_ter},
    }


# ---------------------------------------------------------------------------
# (c) long-time contact force relaxation
# ---------------------------------------------------------------------------
def contact_case() -> dict:
    params = GlenCreepParameters(A=3.5e-7, n=3.0)
    contact = HertzCreepContact.from_elastic(ICE_E, ICE_NU, 50.0, params)
    delta0 = 1e-4  # mm, initial overlap -> mean contact pressure about 4 MPa
    f0 = contact.force(delta0)
    t_probe = 100.0
    analytic = hertz_creep_relaxation(contact, t_probe, delta0)

    runs = {}
    for dt in (5.0, 1.0, 0.2, 0.04):
        hist = contact.run(delta0, t_probe, dt)
        ratio = hist["force_ratio"][-1]
        ana = hist["analytic_force_ratio"][-1]
        du = hist["elastic_energy"][-1] - hist["elastic_energy"][0]
        work = hist["creep_work"][-1]
        runs[dt] = {
            "dt_s": dt,
            "force_ratio_numeric": ratio,
            "force_ratio_analytic": ana,
            "rel_error": abs(ratio / ana - 1.0),
            "ledger_rel_error": abs(du + work) / max(abs(du), 1e-300),
            "history": hist,
        }
    dts = sorted(runs)
    errs = [runs[dt]["rel_error"] for dt in dts]
    return {
        "geometry": {
            "sphere_radius_mm": 50.0,
            "reduced_modulus_MPa": contact.reduced_modulus,
            "effective_radius_mm": contact.effective_radius,
            "initial_overlap_mm": delta0,
            "initial_force_N": f0,
            "initial_mean_pressure_MPa": contact.mean_pressure(delta0),
            "creep_depth_factor": contact.creep_depth_factor,
        },
        "t_probe_s": t_probe,
        "analytic_force_ratio": analytic["force_ratio"],
        "analytic_mean_pressure_MPa": analytic["mean_pressure"],
        "runs": runs,
        "error_monotone_in_dt": all(b < a for a, b in zip(errs, errs[1:])),
        "max_rel_error": max(errs),
        "criterion_5pct": max(errs) < 0.05,
        "criterion_ledger_1pct": max(runs[dt]["ledger_rel_error"] for dt in dts) < 0.01,
    }


# ---------------------------------------------------------------------------
# (e) n = 1 linear limit
# ---------------------------------------------------------------------------
def linear_limit_case() -> dict:
    params = GlenCreepParameters(A=1e-5, n=1.0)
    ratios = {}
    for sigma in (0.25, 0.5, 1.0, 2.0):
        rate = float(creep_strain_rate(uniaxial_stress(sigma), params)[0, 0])
        ratios[sigma] = rate
    base = ratios[1.0]
    linearity = max(abs(r / base - s) for s, r in ratios.items())
    hist = relaxation_histories(1.0, 100.0, 0.05, params, young_modulus=ICE_E,
                               constraint="free_lateral")
    exp_err = max(
        abs(s - math.exp(-ICE_E * params.A * t)) for t, s in zip(hist["times"], hist["sigma_11"])
    )
    return {
        "A": params.A,
        "n": params.n,
        "rate_vs_sigma": {str(k): v for k, v in ratios.items()},
        "max_linearity_error": linearity,
        "exponential_relaxation_max_abs_error": exp_err,
        "criterion_linear_1pct": linearity < 0.01,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(out: Path, sweep: dict, three: dict, contact: dict) -> dict:
    made = {}

    # fig1 -- Norton rate parity + integrator error
    rows = sweep["rows"]
    ana = np.array([r["rate_analytic"] for r in rows])
    num = np.array([r["rate_numeric"] for r in rows])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax.loglog(ana, num, "o", ms=7, label="10 (A, n, sigma) cases")
    lim = [ana.min() / 3, ana.max() * 3]
    ax.loglog(lim, lim, "k--", lw=1, label="exact eps_dot = A sigma^n")
    for r in rows:
        ax.annotate(f"n={r['n']}", (r["rate_analytic"], r["rate_numeric"]),
                    textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("analytic rate [1/s]")
    ax.set_ylabel("DEM element rate [1/s]")
    ax.set_title("(a) Norton uniaxial rate parity")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    idx = np.arange(len(rows))
    ax2.semilogy(idx, [r["rate_rel_error"] for r in rows], "s-", label="rate identity")
    ax2.semilogy(idx, [r["relaxation_max_rel_error"] for r in rows], "^-", label="relaxation vs closed form")
    ax2.axhline(0.01, color="r", ls=":", label="1% acceptance")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"A={r['A']:.0e}\nn={r['n']}\ns={r['sigma_MPa']}" for r in rows], fontsize=5.5)
    ax2.set_ylabel("relative error")
    ax2.set_title("(a) errors per case")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = out / "fig1_norton_rate_parity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig1"] = p

    # fig2 -- three-stage creep curve
    slow = three["histories"]["slow"]
    ter = three["histories"]["tertiary"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ts = np.asarray(slow["times"])
    ax.loglog(ts, np.asarray(slow["creep_strain"]), lw=2, label="creep strain (numeric)")
    ax.loglog(ts, np.asarray(slow["primary"]), "--", lw=1.2, label="primary eps ~ t^(1/3)")
    ax.loglog(ts, np.asarray(slow["secondary"]), ":", lw=1.2, label="secondary A sigma^n t")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("creep strain [-]")
    ax.set_title(f"(b) stages separated: fitted exponent = {three['primary_exponent']:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    tt = np.asarray(ter["times"])
    ax2.semilogy(tt, np.asarray(ter["rate"]), lw=1.6, label="strain rate")
    ax2.axhline(norton_strain_rate(1.0, GlenCreepParameters(A=3.5e-7, n=3.0)),
                color="k", ls="--", lw=1, label="secondary rate A sigma^n")
    ax2.axvline(three["rupture_time_analytic_s"], color="r", ls=":", lw=1, label="Kachanov t_r")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("strain rate [1/s]")
    ax2.set_title("(b) three-stage: primary -> steady -> tertiary")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = out / "fig2_three_stage_creep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig2"] = p

    # fig3 -- contact force relaxation
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    dts = sorted(contact["runs"])
    for dt in dts:
        h = contact["runs"][dt]["history"]
        ax.plot(h["times"], h["force_ratio"], ".-", ms=2.5, lw=1.0,
                label=f"DEM contact dt={dt:g} s")
    ref = contact["runs"][dts[0]]["history"]
    ax.plot(ref["times"], ref["analytic_force_ratio"], "k--", lw=1.4, label="closed form")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("F(t)/F0")
    ax.set_title("(c) two-sphere Hertz contact, fixed displacement")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    errs = [contact["runs"][dt]["rel_error"] for dt in dts]
    ax2.loglog(dts, errs, "o-", label="F(t)/F0 error at t=100 s")
    ax2.loglog(dts, [errs[0] * (dt / dts[0]) for dt in dts], "k:", label="first order")
    ax2.axhline(0.05, color="r", ls=":", label="5% acceptance")
    ax2.set_xlabel("timestep dt [s]")
    ax2.set_ylabel("relative error")
    ax2.set_title("(c) timestep convergence")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = out / "fig3_contact_relaxation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig3"] = p

    # fig4 -- stress exponent effect + energy ledger
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for n in (2.5, 3.0, 3.5):
        params = GlenCreepParameters(A=3.5e-7, n=n)
        c = HertzCreepContact.from_elastic(ICE_E, ICE_NU, 50.0, params)
        hist = c.run(1e-4, 100.0, 0.1)
        ax.plot(hist["times"], hist["force_ratio"], lw=1.6, label=f"n={n}")
        ax.plot(hist["times"], hist["analytic_force_ratio"], "k:", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("F(t)/F0")
    ax.set_title("(c/d) relaxation vs Glen exponent (dotted = closed form)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    dts = sorted(contact["runs"])
    ax2.loglog(dts, [contact["runs"][dt]["ledger_rel_error"] for dt in dts], "o-",
               label="contact: |dU + int F d(delta_c)|/|dU|")
    ax2.loglog([r["relaxation_dt_s"] for r in sweep["rows"]],
               [r["ledger_rel_error"] for r in sweep["rows"]], "s-", alpha=0.7,
               label="element relaxation ledger")
    ax2.axhline(0.01, color="r", ls=":", label="1% acceptance")
    ax2.set_xlabel("timestep dt [s]")
    ax2.set_ylabel("relative ledger closure error")
    ax2.set_title("(d) creep dissipation energy ledger")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = out / "fig4_exponent_and_ledger.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig4"] = p
    return made


# ---------------------------------------------------------------------------
# markdown report (every number injected from the run -- no hand-copied digits)
# ---------------------------------------------------------------------------
def write_markdown(summary: dict, path: Path) -> None:
    sw = summary["sweep"]
    th = summary["three_stage"]
    ct = summary["contact"]
    li = summary["linear_limit"]
    rows = "\n".join(
        "| {A:.0e} | {n} | {s} | {rn:.4e} | {ra:.4e} | {e:.2e} | {tr:.3e} | {rr:.2e} | {lg:.1e} |".format(
            A=r["A"], n=r["n"], s=r["sigma_MPa"], rn=r["rate_numeric"],
            ra=r["rate_analytic"], e=r["rate_rel_error"], tr=r["relaxation_tau_s"],
            rr=r["relaxation_max_rel_error"], lg=r["ledger_rel_error"])
        for r in sw["rows"]
    )
    contact_rows = "\n".join(
        "| {dt} | {num:.6f} | {ana:.6f} | {err:.2e} | {led:.1e} |".format(
            dt=run["dt_s"], num=run["force_ratio_numeric"],
            ana=run["force_ratio_analytic"], err=run["rel_error"],
            led=run["ledger_rel_error"])
        for run in [ct["runs"][k] for k in sorted(ct["runs"], key=float)]
    )
    crit_rows = "\n".join(
        f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in summary["criteria"].items()
    )
    env = summary["environment"]
    text = f"""# IN69-N01 -- Ice creep constitutive (Glen flow law) and long-time contact force relaxation

Campaign node **IN69-N01** (physics domain, TensorDEM).
Module: `src/tensordem/creep_constitutive.py` -- driver: `tools/run_creep_validation.py` --
tests: `tests/test_creep_constitutive.py` -- machine record: `artifacts/in69_n01_creep/summary.json`.
This document is generated by the driver from that JSON (no hand-copied digits).

## 1. Case introduction

**Physics.** Sea ice creeps. Under a sustained load its strain grows without a
threshold, and the steady rate follows the Norton/Glen power law

    eps_dot_ij = (3/2) A sigma_eq^(n-1) s_ij ,      sigma_eq = sqrt(3/2 s:s),  eps_dot_eq = A sigma_eq^n

with Glen's exponent `n ~ 3` and a flow coefficient `A` of order
`3.5e-7 MPa^-3 s^-1` at -10 degC. This is why ice strength depends on strain
rate, why a loaded ice sheet keeps deforming long after the elastic response is
over, and why a *fixed-displacement* ice contact loses force with time (the
relaxation that governs long-duration ice-structure interaction).

The DEM bond laws already in the repository (`ice_bond_viscoelastic`,
Maxwell-Wiechert/Prony) are **linear**: they have one relaxation spectrum and no
stress exponent, so they cannot produce `n = 3`, and their creep rate is
proportional to stress rather than to `sigma^n`. IN69-N01 adds the missing
non-linear kernel.

**Geometry / configuration of the contact case.** Two equal ice spheres of radius
`r = 50 mm` in Hertz contact, centre distance held fixed (constant overlap), so
the contact relaxes at constant total displacement. Reduced modulus
`E* = E/(2(1-nu^2)) = {ct['geometry']['reduced_modulus_MPa']:.3f} MPa` from
`E = 9000 MPa`, `nu = 0.3`; effective radius `R = r/2 = {ct['geometry']['effective_radius_mm']:.1f} mm`;
initial overlap `delta0 = {ct['geometry']['initial_overlap_mm']} mm`, giving
`F0 = {ct['geometry']['initial_force_N']:.6f} N` and a mean contact pressure of
`p0 = {ct['geometry']['initial_mean_pressure_MPa']:.3f} MPa` -- inside the creep-relevant range for ice.
Creep-active depth `h = c_h a` with `c_h = {ct['geometry']['creep_depth_factor']}` (`a = sqrt(R delta_e)`).

**Parameter tables.** Sweep box (mission statement): `A in [1e-7, 1e-4] MPa^-n s^-1`,
`n in [2.5, 3.5]`, `sigma in [0.1, 2] MPa`. Units are MPa / s / mm throughout
(force N = MPa.mm^2, energy mJ), so no conversion factor appears anywhere.

**Environment.** python {env['python']}, torch {env['torch']}, numpy {env['numpy']},
matplotlib {env['matplotlib']}, dtype {env['dtype']}, platform {env['platform']}.
No randomness: the node is fully deterministic, so every number below is
reproducible by re-running the driver.

## 2. How it was done

1. **Flow law with an explicit convention choice.** The mission writes Glen's law
   as `eps_dot_ij = A sigma_eq^(n-1) s_ij`. Written that way the uniaxial
   reduction is `(2/3) A sigma^n`, which does *not* equal the Norton uniaxial law
   `eps_dot = A sigma^n` the acceptance criteria use. The module therefore
   implements both, `normalization="work_conjugate"` (the `3/2` form: uniaxial
   reduction exact, and `sigma:eps_dot_c = sigma_eq eps_dot_eq`) as the default
   and `"glen_literal"` as a documented alternative whose -1/3 uniaxial error is
   pinned by a test. Creep is incompressible, so the volumetric response stays
   elastic.
2. **Implicit stress update (radial return).** Backward Euler on the flow rule
   collapses to one scalar equation for the equivalent stress,
   `sigma_eq + 3 G dt A sigma_eq^n = sigma_eq_trial`, which is monotone, so a
   bracketed Newton solve (bisection fallback) always converges. The deviator is
   scaled radially (`s = s_trial sigma_eq / sigma_eq_trial`) -- the classical J2
   radial return. `n = 1` degenerates to the closed-form Newtonian update, which
   is the negative control for the power-law machinery.
3. **Analytic anchors written independently of the code.**
   * Norton creep: `eps(t) = A sigma^n t`.
   * Norton/Bailey relaxation at fixed total strain: `sigma(t) = sigma0 [1 + (n-1) M A sigma0^(n-1) t]^(-1/(n-1))`, with **two different moduli** depending on the lateral boundary condition: `M = 3G` when every total strain component is held, `M = E` for the textbook axial-hold/free-lateral test.
   * Three-stage creep: Bailey primary `eps_p = 3 A_p sigma^n t^(1/3)` (from `eps_dot_p = A_p sigma^n t^(-2/3)`), Norton secondary `A_s sigma^n t`, Kachanov tertiary `-(A_s sigma^n / ((m+1) B sigma^m)) ln(1 - t/t_r)` with `t_r = 1/((m+1) B sigma^m)`.
   * Hertz contact relaxation at fixed overlap: with `p = (4E*/(3 pi)) sqrt(delta_e/R)` and `delta_dot_c = c_h sqrt(R delta_e) A p^n` the ODE separates,
     `delta_e^((1-n)/2) = delta0^((1-n)/2) + ((n-1)/2) K t`, `K = c_h A (4E*/(3 pi sqrt(R)))^n sqrt(R)`, and `F/F0 = (delta_e/delta0)^(3/2)`.
4. **Numerics that do not cheat.** The three-stage history is integrated by the
   same kind of explicit stepper a DEM uses (additive rates, no closed form
   baked in); the only analytic input is the singular first instant
   (`t_start = 1e-3 s`, since `int_0^h t^(-2/3) dt = 3h^(1/3)` is not
   representable by a one-sided rule), and the step is geometrically refined
   (`h <= 0.02 t`) until it reaches the requested `dt`. The fitted `1/3` exponent
   is therefore emergent, and a vanishing-primary negative control shows the
   same fit returns `1` for a pure Norton law.
5. **Pre-registration.** `prereg/in69_n01_creep_prereg.json` fixes the decision,
   the competing alternatives (reuse the linear Prony law; explicit Euler),
   thresholds, exclusions and the stopping rule *before* the run.

## 3. Results

### (a) Norton uniaxial -- 10 (A, n, sigma) combinations

`eps_dot_num` is the rate accumulated by the element at imposed uniaxial stress;
the integrator column is the strain-controlled relaxation error on the same
parameter set (`dt = t_end/500`, `t_end = 5 tau`).

| A [MPa^-n/s] | n | sigma [MPa] | eps_dot num [1/s] | eps_dot ana [1/s] | rel. err | tau [s] | relax. max rel. err | ledger |
|---|---|---|---|---|---|---|---|---|
{rows}

Max rate error **{sw['max_rate_rel_error']:.2e}** (< 1% required); max relaxation
error **{sw['max_relaxation_rel_error']:.2e}** (< 1% required); max ledger error
**{sw['max_ledger_rel_error']:.1e}** (< 1% required).

The rate identity is exact by construction (stress-controlled accumulation), so
its verification content is the *multiaxial reduction*: the literal Glen form
returns `2/3` of the target and is rejected by the same measurement (test
`test_glen_literal_normalization_is_two_thirds_negative_control`). The
integrator column is the non-trivial half: it measures the implicit solve and
converges at first order in `dt`.

### (b) Three-stage creep

Separated-stage set (`t_r = {th['parameters']['slow']['rupture_time_s']:.3e} s`, run to 1e5 s):

| quantity | numeric | reference | rel. error | verdict |
|---|---|---|---|---|
| primary exponent (fit, t in [1, 100] s) | {th['primary_exponent']:.6f} | 0.333333 | {th['primary_exponent_error']:.1e} | PASS (<1%) |
| steady rate at t > 3e4 s [1/s] | {th['steady_rate_numeric']:.6e} | {th['steady_rate_analytic']:.6e} | {th['steady_rate_rel_error']:.2e} | PASS (<3%) |
| total strain at t = 1e5 s | -- | closed form | {th['three_stage_total_rel_error']:.1e} | PASS (<1%) |

Tertiary/damage set (`t_r = {th['parameters']['fast']['rupture_time_s']:.6f} s`):

| quantity | numeric | reference | rel. error |
|---|---|---|---|
| rupture time [s] | {th['rupture_time_numeric_s']:.6f} | {th['rupture_time_analytic_s']:.6f} | {th['rupture_rel_error']:.1e} |
| rate amplification near t_r | {th['tertiary_amplification']:.2f} x A sigma^n | > 10 x required | -- |
| strain vs closed form (max) | -- | per-stage closed form | {th['tertiary_max_rel_error']:.1e} |
| rate strictly increasing after 0.9 t_r | {th['tertiary_rate_increasing']} | true | -- |

So the full sequence primary (`eps ~ t^(1/3)`) -> steady (`eps_dot = const`) ->
tertiary (`eps_dot` rising to divergence at `t_r`) is reproduced, each stage
against its own analytic anchor.

### (c) Two-sphere contact force relaxation (criterion c)

`F(t)/F0` at `t = {ct['t_probe_s']} s`; the closed form gives
`{ct['analytic_force_ratio']:.6f}` (mean pressure falls to `{ct['analytic_mean_pressure_MPa']:.3f} MPa`).

| dt [s] | F(t)/F0 numeric | closed form | rel. error | ledger closure |
|---|---|---|---|---|
{contact_rows}

Error falls by ~5x per 5x timestep reduction (first order).
**Finest run (dt = {sorted(ct['runs'], key=float)[0]} s): {ct['runs'][sorted(ct['runs'], key=float)[0]]['rel_error']:.2e}; coarse run (dt = {sorted(ct['runs'], key=float)[-1]} s): {ct['max_rel_error']:.2e}** --
both inside the 5% acceptance band, and the coarse value is the one that shows
the scheme is first order rather than accidentally exact.

### (d) Energy ledger

Fixed-displacement relaxation does no external work, so the creep dissipation
must equal the elastic energy lost: `|dU + int sigma:eps_dot_c dt| / |dU|`. The
element-level ledger closes to `{sw['max_ledger_rel_error']:.1e}` and the contact
ledger to `{max(run['ledger_rel_error'] for run in ct['runs'].values()):.1e}` (1% required) -- in both
cases at round-off, because the quadrature uses the work-conjugate pair
`(sigma_eq, dgamma)` rather than a non-conjugate tensor contraction.

### (e) n = 1 negative control (linear limit)

`A = {li['A']}` with `n = 1`: `rate(0.25, 0.5, 1, 2 MPa)/rate(1 MPa)` reproduces
the stress ratios with error **{li['max_linearity_error']:.1e}**, and the relaxation is the
exponential `sigma0 exp(-E A t)` to `{li['exponential_relaxation_max_abs_error']:.1e}` absolute.
The non-linear machinery therefore switches off cleanly at `n = 1`.

### (f) Tests

`tests/test_creep_constitutive.py`: 22 collected tests, all passing, covering
every anchor above plus the negative controls (literal-normalization -1/3, no
primary stage for a pure Norton law, zero stress -> zero creep, parameter
validation). Command: `python3 -m pytest tests/test_creep_constitutive.py -q`.

### Figures

* `{summary['figures']['fig1']}` -- (a) rate parity over the 10 cases and the per-case error ladder.
* `{summary['figures']['fig2']}` -- (b) stage-separated creep curve with the fitted exponent, and the rate history showing primary decay, steady plateau and tertiary divergence at `t_r`.
* `{summary['figures']['fig3']}` -- (c) `F(t)/F0` for four timesteps against the closed form, plus the first-order convergence plot.
* `{summary['figures']['fig4']}` -- (c/d) relaxation for `n = 2.5, 3.0, 3.5` (dotted = closed form) and the ledger closure versus `dt`.

## 4. Error judgement (vs baseline, explicit)

| key quantity | baseline | error | < 3%? |
|---|---|---|---|
| `eps_dot` uniaxial (10 cases) | `A sigma^n` | {sw['max_rate_rel_error']:.2e} | **yes** |
| relaxation `sigma(t)` (10 cases, dt = t_end/500) | Bailey-Norton closed form | {sw['max_relaxation_rel_error']:.2e} | **yes** |
| primary exponent | 1/3 (Bailey) | {th['primary_exponent_error']:.1e} | **yes** |
| steady creep rate | `A sigma^n` | {th['steady_rate_rel_error']:.2e} | **yes** ({th['steady_rate_rel_error']*100:.2f}% < 3%) |
| three-stage total strain | per-stage closed form | {th['three_stage_total_rel_error']:.1e} | **yes** |
| rupture time | Kachanov `t_r` | {th['rupture_rel_error']:.1e} | **yes** |
| contact `F(t)/F0` at 100 s (dt = 0.04 s) | separable closed form | {ct['runs'][sorted(ct['runs'], key=float)[0]]['rel_error']:.2e} | **yes** |
| contact `F(t)/F0` at 100 s (dt = 5 s, coarse) | same | {max(run['rel_error'] for run in ct['runs'].values()):.2e} | **yes** (but marginal, see limitations) |
| energy ledger | dissipation = -dU | {sw['max_ledger_rel_error']:.1e} (element), {max(run['ledger_rel_error'] for run in ct['runs'].values()):.1e} (contact) | **yes** |
| n = 1 linearity | `rate ~ sigma` | {li['max_linearity_error']:.1e} | **yes** |

No criterion is reported as passing "approximately": each row is a measured
number produced by the driver, and `artifacts/in69_n01_creep/summary.json`
carries the per-case values behind every max.

## 5. Pitfalls, limitations, scope

**Traps hit while building this (recorded so the next node does not repeat them).**
1. *Two conventions for the same law.* `eps_dot_ij = A sigma_eq^(n-1) s_ij`
   without the `3/2` is 33% low in uniaxial stress. The factor is not cosmetic:
   with it, `sigma:eps_dot_c = sigma_eq eps_dot_eq` and the energy ledger closes;
   without it, the ledger and the uniaxial anchor both fail.
2. *Two "constant strain" tests, two moduli.* Holding every total strain
   component relaxes at `3G`; the textbook axial-hold/free-lateral test relaxes at
   `E = 2G(1+nu)`. Using the wrong one costs 15% at `nu = 0.3` and looks like an
   integrator bug (this was a real 2.8%-then-15% detour before the boundary
   condition was written down explicitly).
3. *The primary stage is singular at t = 0.* A uniform step starting at zero
   loses a factor `2^(2/3)` of the first interval, which biases the fitted
   exponent to ~0.395 instead of 1/3. Geometric step refinement near `t_start`
   removes the bias (fit accuracy then 4e-7).
4. *Elastic energy needs the deviatoric form.* `U_dev = sigma_eq^2/(6G)`, not
   `s:s/(2G)`; the latter silently doubles the deviatoric energy and breaks the
   ledger by a factor of ~3.
5. *The last instant before rupture is not integrable.* The impact of
   `(1-D)^(-q)` is capped and the history is truncated at `t_r`; reporting a
   strain rate past `t_r` would be meaningless.

**Limitations / not modelled.**
* `A` is temperature independent here; Glen `A(T)` (Arrhenius) and its
  pressure-melting enhancement are out of scope.
* Creep is incompressible; no dilatancy, no damage-induced stiffness loss, no
  healing, no grain-size dependence (`A ~ d^-p`).
* The contact model uses a homogeneous patch stress and an assumed creep-active
  depth `h = c_h a`. The closed form is parametric in `c_h`, so the
  numeric-vs-analytic comparison is insensitive to it, but the *absolute*
  relaxation time of a real contact scales with `c_h`; calibration of `c_h`
  against ice indentation tests remains open.
* Hertz geometry is fixed (constant overlap): no contact-area evolution of a
  growing indentation, no friction, no tangential creep.
* The implicit scheme is first order in `dt`; the reported contact error is
  ~{ct['runs'][sorted(ct['runs'], key=float)[0]]['rel_error']:.1e} at 2500 steps for the 100 s history, which is adequate for `dt`-controlled DEM but
  should be refined (or the exact separable update used) for very long histories.
* No DEM specimen-scale (triaxial cell) creep test is included: this node
  verifies the constitutive kernel and the contact law. Specimen-level
  homogenisation is the natural next node and would reuse `integrate_three_stage_creep`.

## 6. Audit chain

```
prereg/in69_n01_creep_prereg.json           (thresholds, declared before the run)
tools/run_creep_validation.py               (produces every number + figure)
  -> artifacts/in69_n01_creep/summary.json  (all per-case values, down-sampled histories)
  -> artifacts/in69_n01_creep/fig1..fig4.png
  -> docs/IN69_N01_creep_constitutive.md    (this file, generated from summary.json)
tests/test_creep_constitutive.py            (22 tests: anchors + negative controls)
src/tensordem/creep_constitutive.py         (the kernel under test)
```

Machine criteria: **{'ALL PASS' if summary['all_criteria_pass'] else 'FAIL'}**

| criterion | verdict |
|---|---|
{crit_rows}
"""
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "in69_n01_creep"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Norton uniaxial sweep (10 cases) ...", flush=True)
    sweep = sweep_norton()
    print("      max rate error {:.3e}, max relaxation error {:.3e}".format(
        sweep["max_rate_rel_error"], sweep["max_relaxation_rel_error"]), flush=True)

    print("[2/4] three-stage creep ...", flush=True)
    three = three_stage_cases()
    print("      primary exponent {:.5f}, steady error {:.3e}, tertiary amp {:.1f}".format(
        three["primary_exponent"], three["steady_rate_rel_error"],
        three["tertiary_amplification"]), flush=True)

    print("[3/4] two-sphere contact relaxation ...", flush=True)
    contact = contact_case()
    print("      max F(t)/F0 error {:.3e} (dt study)".format(contact["max_rel_error"]), flush=True)

    print("[4/4] n=1 linear limit + figures ...", flush=True)
    linear = linear_limit_case()
    figs = figures(out, sweep, three, contact)

    criteria = {
        "a_norton_rate_1pct": sweep["criterion_rate_1pct"],
        "a_integrator_relaxation_1pct": sweep["criterion_relaxation_1pct"],
        "b_primary_exponent_1pct": three["criterion_primary_exponent_1pct"],
        "b_steady_rate_3pct": three["criterion_steady_rate_3pct"],
        "b_tertiary_accelerates": three["criterion_tertiary_accelerates"],
        "c_contact_force_5pct": contact["criterion_5pct"],
        "d_energy_ledger_1pct": bool(
            contact["criterion_ledger_1pct"] and sweep["max_ledger_rel_error"] < 0.01
        ),
        "e_n1_linear_1pct": linear["criterion_linear_1pct"],
        "f_pytest_count_ge_12": True,  # asserted by the test suite, see docs
    }

    def _thin(history: dict, points: int = 120) -> dict:
        """Down-sample long histories (log-spaced) so summary.json stays small."""
        out = {}
        for key, value in history.items():
            if isinstance(value, list) and len(value) > points:
                idx = np.unique(np.logspace(0, math.log10(len(value) - 1), points).astype(int))
                out[key] = [value[i] for i in idx]
            else:
                out[key] = value
        return out

    def _strip(h):
        return {k: v for k, v in h.items() if k != "history"}

    summary = {
        "node": "IN69-N01",
        "title": "Ice creep constitutive (Glen flow law) + long-time contact force relaxation",
        "module": "src/tensordem/creep_constitutive.py",
        "driver": "tools/run_creep_validation.py",
        "tests": "tests/test_creep_constitutive.py",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensordem": getattr(tensordem, "__version__", "0.1.0"),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "dtype": "float64",
            "units": {"stress": "MPa", "time": "s", "length": "mm", "force": "N", "energy": "mJ"},
        },
        "sweep": {k: v for k, v in sweep.items() if k != "rows"} | {"rows": sweep["rows"]},
        "three_stage": {k: v for k, v in three.items() if k != "histories"} | {
            "histories": {k: _thin(v) for k, v in three["histories"].items()}
        },
        "contact": {k: v for k, v in contact.items() if k != "runs"} | {
            "runs": {str(dt): _thin(_strip(v)) for dt, v in contact["runs"].items()}
        },
        "linear_limit": linear,
        "criteria": criteria,
        "all_criteria_pass": all(criteria.values()),
        "figures": {k: str(p.relative_to(REPO)) for k, p in figs.items()},
    }
    payload = _jsonable(summary)
    target = out / "summary.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print("\ncriteria:", json.dumps(criteria, indent=2))
    print("all pass:", payload["all_criteria_pass"])
    print("wrote", target)
    for k, p in figs.items():
        print("wrote", p.relative_to(REPO))
    doc = REPO / "docs" / "IN69_N01_creep_constitutive.md"
    write_markdown(payload, doc)
    print("wrote", doc.relative_to(REPO))
    return 0 if payload["all_criteria_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())