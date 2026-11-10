#!/usr/bin/env python
"""Analysis of the p302w2-n04 keel-porosity DEM campaign.

Reads the per-case JSONs written by ``p302_n04_dem_run.py``, fits the
mechanical consolidation of each case, adds the pore-ice closure
(:mod:`tensordem.ice_ridge_keel_porosity.pore_ice`), fits the effective
single-exponential law, and writes

* ``results.json`` -- every acceptance number plus the evidence chain,
* figures (PNG) under ``figures/``.

All numbers are computed from the DEM records; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

from tensordem.ice_ridge_keel_porosity import (
    KeelGeometry,
    PoreIceConfig,
    composite_porosity,
    exponential_decay,
    fit_exponential_decay,
    hydraulic_pore_radius,
    polydisperse_radii,
    pore_filling_fraction_for,
    pore_ice_time_scale,
)


def load_cases(outdir: str) -> dict:
    cases = {}
    for path in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        if "case" not in data:
            continue
        # key by file name so a variant run (same case name, different friction)
        # does not overwrite the base case
        cases[os.path.splitext(os.path.basename(path))[0]] = data
    return cases


def analyse_case(data: dict, pore_config: PoreIceConfig | None = None) -> dict:
    """Mechanical fit + pore-ice closure + effective exponential fit."""
    cfg = pore_config or PoreIceConfig()
    case = data["case"]
    t = np.asarray(data["time"], dtype=float)
    # primary void ratio = Monte-Carlo point fraction (unbiased free volume);
    # the oedometer/void-ratio record (whole sphere volumes, hence ~0.01 low by
    # the contact overlap and the caps above the mean free surface) is kept as
    # the independent cross-check.
    n = np.asarray(data["porosity_point"], dtype=float)
    n_slab = np.asarray(data["porosity"], dtype=float)
    n0 = float(n[0])
    # ---- mechanical stage ---------------------------------------------------
    from tensordem.ice_ridge_keel_porosity import fit_exponential_fixed_origin
    n_m, tau_m, r2_m, rms_m = _fit_fixed_origin(t, n, n0)
    # linear trend of the void ratio over the shaking window, with its standard
    # error: the honest measure of whether the DEM reorganises the bed at all.
    if len(t) >= 4:
        slope, intercept = np.polyfit(t, n, 1)
        resid = n - (slope * t + intercept)
        dof = max(len(t) - 2, 1)
        sxx = float(np.sum((t - t.mean()) ** 2))
        slope_err = float(np.sqrt(np.sum(resid ** 2) / dof / sxx)) if sxx > 0 else float("nan")
        trend_per_100s = float(-slope * 100.0)
        trend_significant = bool(abs(slope) > 2.0 * slope_err)
        noise_rms = float(np.std(resid))
    else:
        trend_per_100s, slope, slope_err, trend_significant = (float("nan"),) * 4
        noise_rms = float("nan")
    n_drop_window = float(n[0] - n[-1])
    if not np.isfinite(n_m) or n_m >= n0 - 2.0e-3:
        # no resolvable mechanical densification inside the simulated window
        n_m, tau_m = n0, float("inf")
        r2_m = float("nan")
    # ---- pore geometry from the same block set (deterministic regeneration) --
    radii = polydisperse_radii(case["n_blocks"], case["d_min"], case["d_max"],
                               case["distribution"], -2.0,
                               np.random.default_rng(case["seed"]))
    n_plateau = float(np.mean(n[-max(3, len(n) // 10):]))
    r_h = hydraulic_pore_radius(radii, n_plateau)
    tau_p = pore_ice_time_scale(radii, n_plateau, cfg)
    f_eq = pore_filling_fraction_for(n_plateau, cfg.n_consolidated)
    # ---- composite + effective fit -----------------------------------------
    t_model = np.linspace(0.0, 5.0 * tau_p, 61)
    n_model = composite_porosity(t_model, n0, n_m, tau_m, f_eq, tau_p)
    n0_fit, n_inf_fit, tau_fit, r2, rms = fit_exponential_decay(
        t_model, n_model, p0=(n0, float(n_model[-1]), tau_p))
    n_fit = exponential_decay(t_model, n0_fit, n_inf_fit, tau_fit)
    t_50 = _crossing(t_model, n_model, n0_fit, n_inf_fit, 0.5)
    t_90 = _crossing(t_model, n_model, n0_fit, n_inf_fit, 0.9)
    return dict(
        name=case["name"],
        case=case,
        n0=n0, n0_slab=float(n_slab[0]) if len(n_slab) else float("nan"),
        n_end_dem=float(n[-1]),
        n_mech_inf=n_m, tau_mech=tau_m, r_squared_mech=r2_m, rms_mech=rms_m,
        n_plateau=n_plateau,
        n_drop_window=n_drop_window,
        noise_rms=noise_rms,
        densification_rate_per_100s=trend_per_100s,
        densification_rate_err=slope_err,
        densification_significant=trend_significant,
        pore_radius=r_h, pore_diameter=2.0 * r_h, tau_pore=tau_p, f_eq=f_eq,
        n_inf=n_inf_fit, n_0_fit=n0_fit, tau=tau_fit, r_squared=r2,
        residual_rms=rms,
        max_rel_deviation=float(np.max(np.abs(n_model - n_fit))
                                / max(n0_fit - n_inf_fit, 1e-12)),
        t_50=t_50, t_90=t_90,
        model=dict(t=t_model.tolist(), n=n_model.tolist(), n_fit=n_fit.tolist()),
        series=dict(t=t.tolist(), n=n.tolist(), n_point=n_slab.tolist(),
                    coordination=data["coordination"], kinetic=data["kinetic"]),
        run=dict(wall_seconds=data["wall_seconds"], steps=data["steps"],
                 ms_per_step=data["ms_per_step"], dt=data["dt_max_stable"],
                 gravity=data["gravity"], parcel=data["parcel"],
                 energy_ledger=data["energy_ledger"],
                 ledger_closure_shake=_closure(data["energy_ledger"]),
                 placement_porosity=data["placement_porosity"],
                 wall_penetration_max=data["wall_penetration_max"]),
        resistance=dict(
            t=t_model.tolist(),
            ratio=((1.0 - n_model) / (1.0 - n0)).tolist(),
            note="keel resistance factor R(t)/R(0) = (1-n(t))/(1-n(0)): "
                 "solid fraction grows as the voids close"),
    )


def _closure(ledger: dict) -> float:
    """Relative closure residual of the shaking-phase energy ledger."""
    shake = ledger.get("shake_phase") or {}
    if not shake:
        return float("nan")
    scale = (abs(shake.get("work_body", 0.0)) + abs(shake.get("work_wall", 0.0))
             + abs(shake.get("work_dissipated", 0.0)))
    if scale <= 0:
        return float("nan")
    return abs(shake.get("closure_residual", 0.0)) / scale


def _fit_fixed_origin(t, n, n0):
    from tensordem.ice_ridge_keel_porosity import fit_exponential_fixed_origin
    n_inf, tau, r2, rms = fit_exponential_fixed_origin(t, n, n_origin=n0,
                                                       tau_guess=max(float(np.max(t)) / 3.0, 1.0))
    return n_inf, tau, r2, rms


def _fit_fixed_origin_unused(t, n, n0):
    from scipy.optimize import curve_fit

    def model(tt, n_inf, tau):
        return n_inf + (n0 - n_inf) * np.exp(-tt / tau)

    best = None
    for guess in (max(float(t.max()) / 3.0, 1.0), 30.0, 300.0, 3000.0):
        try:
            popt, _ = curve_fit(model, t, n, p0=(float(n[-1]), guess),
                                bounds=([0.01, 1.0], [n0 - 1e-4, 1e6]),
                                maxfev=40000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((n - model(t, *popt)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(popt[0]), float(popt[1]))
    if best is None:
        return float(n[-1]), float("nan"), 0.0, float("nan")
    sse, n_inf, tau = best
    ss_tot = float(np.sum((n - n.mean()) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0
    return n_inf, tau, r2, math.sqrt(sse / len(t))


def _crossing(t, n, n_start, n_end, fraction):
    target = n_start - fraction * (n_start - n_end)
    below = np.nonzero(n <= target)[0]
    if len(below) == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(t[0])
    x0, x1, y0, y1 = t[i - 1], t[i], n[i - 1], n[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def _power_law_fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return dict(c=float("nan"), alpha=float("nan"), r_squared=float("nan"), n=0)
    lx, ly = np.log(x[mask]), np.log(y[mask])
    alpha, log_c = np.polyfit(lx, ly, 1)
    y_fit = np.exp(log_c) * x[mask] ** alpha
    ss_res = float(np.sum((y[mask] - y_fit) ** 2))
    ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
    return dict(c=float(np.exp(log_c)), alpha=float(alpha),
                r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                n=int(mask.sum()))


def summarise(cases: dict[str, dict], pore_config: PoreIceConfig) -> dict:
    out = {}
    for name, data in cases.items():
        out[name] = analyse_case(data, pore_config)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="artifacts/keel_porosity/p302_n04")
    ap.add_argument("--out", default="artifacts/keel_porosity/p302_n04/results.json")
    ap.add_argument("--figdir", default="artifacts/keel_porosity/p302_n04/figures")
    args = ap.parse_args()

    pore_config = PoreIceConfig()
    cases = load_cases(args.indir)
    if not cases:
        raise SystemExit(f"no case JSONs in {args.indir}")
    res = summarise(cases, pore_config)

    # ---- criterion numbers --------------------------------------------------
    main_case = res.get("main") or list(res.values())[0]
    amp_names = [n for n in ("amp_1.0", "amp_3.0", "amp_6.0", "amp_12.0") if n in res]
    amp = [res[n]["case"]["wave_amplitude"] for n in amp_names]
    tau_eff = [res[n]["tau"] for n in amp_names]
    tau_mech = [res[n]["tau_mech"] for n in amp_names]
    grad_names = [n for n in ("grad_mono", "grad_ref", "grad_wide") if n in res]
    grad = [res[n]["case"]["d_max"] / res[n]["case"]["d_min"] for n in grad_names]

    summary = dict(
        main=dict(name=main_case["name"],
                  n_0=main_case["n0"], n_0_slab=main_case["n0_slab"],
                  n_inf=main_case["n_inf"], tau=main_case["tau"],
                  r_squared=main_case["r_squared"],
                  residual_rms=main_case["residual_rms"],
                  max_rel_deviation=main_case["max_rel_deviation"],
                  n_mech_inf=main_case["n_mech_inf"],
                  tau_mech=main_case["tau_mech"],
                  r_squared_mech=main_case["r_squared_mech"],
                  tau_pore=main_case["tau_pore"],
                  f_eq=main_case["f_eq"],
                  pore_diameter=main_case["pore_diameter"],
                  t_50=main_case["t_50"], t_90=main_case["t_90"]),
        wave_sweep=dict(amplitude=amp, tau_eff=tau_eff, tau_mech=tau_mech,
                        n0=[res[n]["n0"] for n in amp_names],
                        n_end=[res[n]["n_end_dem"] for n in amp_names],
                        n_drop=[res[n]["n_drop_window"] for n in amp_names],
                        dens_rate=[res[n]["densification_rate_per_100s"] for n in amp_names],
                        dens_rate_err=[res[n]["densification_rate_err"] for n in amp_names],
                        noise_rms=[res[n]["noise_rms"] for n in amp_names],
                        dens_significant=[bool(res[n]["densification_significant"]) for n in amp_names],
                        fit_eff=_power_law_fit(amp, tau_eff),
                        fit_mech=_power_law_fit(amp, tau_mech),
                        monotone_eff=bool(np.all(np.diff(tau_eff) < 0)),
                        monotone_mech=bool(np.all(np.diff([x for x in tau_mech
                                                           if np.isfinite(x)]) < 0))),
        gradation_sweep=dict(
            ratio=grad,
            n0=[res[n]["n0"] for n in grad_names],
            n_mech_inf=[res[n]["n_mech_inf"] for n in grad_names],
            tau_mech=[res[n]["tau_mech"] for n in grad_names],
            pore_diameter=[res[n]["pore_diameter"] for n in grad_names],
            names=grad_names),
        friction_sweep=dict(
            base={k: res[k]["n0"] for k in res if not k.startswith("mu09_")},
            mu09={k[5:]: res[k]["n0"] for k in res if k.startswith("mu09_")},
            note="mu 0.70 -> 0.90 (effective friction of the rotation-locked blocks)"),
        checks=dict(
            dt_fine=res.get("dt_fine", {}).get("n0"),
            dt_ref=res.get("dt_ref", {}).get("n0"),
            damp_low=res.get("damp_low", {}).get("n0"),
            damp_high=res.get("damp_high", {}).get("n0"),
            seed_1=res.get("seed_1", {}).get("n0"),
            seed_2=res.get("seed_2", {}).get("n0"),
            buoyant=res.get("buoyant", {}).get("n0"),
            keel_walls=res.get("keel_walls", {}).get("n0"),
        ),
    )
    payload = dict(criteria=summary, cases={k: {kk: vv for kk, vv in v.items()
                                               if kk not in ("model", "series", "resistance")}
                                           for k, v in res.items()},
                   pore_ice=dict(alpha_eff=pore_config.alpha_eff,
                                 n_consolidated=pore_config.n_consolidated,
                                 composite_form="n(t)=[n_m+(n_0-n_m)exp(-t/tau_m)]*[1-f_eq(1-exp(-t/tau_p))]"))
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {args.out}")

    try:
        make_figures(res, summary, args.figdir)
    except Exception as exc:                        # pragma: no cover
        print(f"[warn] figures skipped: {exc}")


def make_figures(res, summary, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    # --- figure 1: main case evolution --------------------------------------
    main_case = res.get("main") or list(res.values())[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    s = main_case["series"]
    ax.plot(s["t"], s["n"], "o", ms=3, color="tab:blue", label="DEM void ratio (mechanical)")
    ax.plot(s["t"], s["n_point"], ".", ms=2, color="tab:cyan", alpha=0.7,
            label="DEM void ratio (point estimator)")
    m = main_case["model"]
    ax.plot(m["t"], m["n"], "-", color="tab:red", lw=2,
            label="composite model (DEM mechanics + pore ice)")
    ax.plot(m["t"], m["n_fit"], "--", color="k", lw=1.2,
            label=f"single-exponential fit ($R^2$={main_case['r_squared']:.3f}, "
                  f"$\\tau$={main_case['tau']:.0f} s)")
    ax.axhline(0.20, color="grey", ls=":", lw=1, label="consolidated keel $n=0.20$")
    ax.set_xscale("log")
    ax.set_xlabel("time since ridge formation t [s]")
    ax.set_ylabel("keel void ratio n [-]")
    ax.set_title("Keel porosity evolution: DEM calibration + pore-ice closure")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig1_porosity_evolution.png"), dpi=150)
    plt.close(fig)

    # --- figure 2: amplitude sweep ------------------------------------------
    ws = summary["wave_sweep"]
    if len(ws["amplitude"]) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
        A = np.array(ws["amplitude"])
        ax1.loglog(A, ws["tau_eff"], "o-", label="effective $\\tau$ (composite)")
        ax1.loglog(A, ws["tau_mech"], "s--", label="mechanical $\\tau$ (DEM)")
        fit = ws["fit_eff"]
        if np.isfinite(fit.get("c", np.nan)):
            aa = np.linspace(A.min(), A.max(), 50)
            ax1.loglog(aa, fit["c"] * aa ** fit["alpha"], ":", color="grey",
                       label=f"$\\tau={fit['c']:.0f}A^{{{fit['alpha']:.2f}}}$")
        ax1.set_xlabel("wave acceleration amplitude A [m/s$^2$]")
        ax1.set_ylabel("consolidation time $\\tau$ [s]")
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.3, which="both")
        ax1.set_title("(a) $\\tau$ vs wave amplitude")
        gs = summary["gradation_sweep"]
        ax2.semilogx(gs["ratio"], gs["n_mech_inf"], "o-", color="tab:green")
        ax2.set_xlabel("gradation ratio $d_{max}/d_{min}$")
        ax2.set_ylabel("mechanical plateau $n_m$ [-]")
        ax2.grid(alpha=0.3, which="both")
        ax2.set_title("(b) RCP plateau vs gradation")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig2_sensitivity.png"), dpi=150)
        plt.close(fig)

    # --- figure 3: diagnostics of the DEM ------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax1.semilogx(s["t"], s["coordination"], color="tab:purple")
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("mean coordination number Z")
    ax1.set_title("(a) contact network during shaking")
    ax1.grid(alpha=0.3)
    ax2.semilogx(s["t"], np.maximum(s["kinetic"], 1e-9), color="tab:orange")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("kinetic energy [J]")
    ax2.set_title("(b) agitation of the parcel")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig3_dem_diagnostics.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
