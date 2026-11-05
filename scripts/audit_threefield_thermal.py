#!/usr/bin/env python3
"""Verification audit of the three-field melting core (``tensordem.threefield.thermal``).

Five independent checks, each graded against a reference that does **not** reuse
the code path under test:

1. **Closed form vs quadrature.** ``analytic_ransom_marshall_time`` claims
   ``t = ∫_D^{D0} dD'/(A/D' + B/√D')``.  Reference: quadrature of exactly that
   integrand (``scipy.integrate.quad``; 40-digit ``mpmath`` when available),
   evaluated from the same ``A``/``B`` but never touching the primitive.
2. **Cancellation floor.**  The textbook primitive is re-implemented here (as it
   stood before this audit) and compared with the shipped implementation, so the
   before/after error of the numerical fix is a number, not a claim.
3. **ODE consistency.**  Numerically differentiate the closed-form ``t(D)`` and
   compare with ``1/(dD/dt)`` from ``diameter_rate``: grades the sign and scaling
   of the primitive against the right-hand side the solver integrates.
4. **Integrator order.**  ``integrate_melting`` (explicit RK2 midpoint) against
   the closed form over a geometric ``dt`` sweep: error must fall as ``dt²``.
5. **Inversion, degenerate limit, disappearance.**  ``analytic_diameter_ransom_
   marshall`` back-substitution residual, the ``B → 0`` branch against the
   conduction law, and the melt-completion timing error (bounded by one ``dt``).

Artifacts go to ``docs/threefield-thermal/``.  Runtime a few seconds; no DEM time
stepping is involved (load-safe under a saturated machine).

Usage: ``python scripts/audit_threefield_thermal.py [--out docs/threefield-thermal]``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.threefield.thermal import (  # noqa: E402
    ThermalProperties,
    _analytic_case_constants,  # private but this is a white-box audit
    analytic_conduction_melt_time,
    analytic_diameter_conduction,
    analytic_diameter_ransom_marshall,
    analytic_ransom_marshall_time,
    diameter_rate,
    integrate_melting,
)

try:  # optional high-precision reference
    import mpmath as _mp

    _mp.mp.dps = 40
except Exception:  # pragma: no cover - mpmath is optional
    _mp = None

SCHEMA = "tensordem.threefield-thermal-verification/1"
VELOCITIES = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 0.02, 0.05, 0.15, 0.25, 0.5)


def _rel(a: float, b: float) -> float:
    scale = abs(b) if abs(b) > 0.0 else 1.0
    return abs(a - b) / scale


def reference_time(d0: float, d: float, props: ThermalProperties, u: float,
                   include_sensible: bool) -> dict:
    """Independent reference for ∫_D^{D0} dD/(A/D + B/√D)."""
    a, b = _analytic_case_constants(props, u, include_sensible=include_sensible)
    if b <= 0.0:
        return {"value": (d0 * d0 - d * d) / (2.0 * a), "method": "conduction-law", "err": 0.0}
    integrand = lambda dd: 1.0 / (a / dd + b / math.sqrt(dd))  # noqa: E731
    quad_value, quad_err = quad(integrand, d, d0, epsabs=1e-18, epsrel=1e-14, limit=400)
    if _mp is not None:
        mp_af, mp_bf = _mp.mpf(a), _mp.mpf(b)
        mp_value = _mp.quad(
            lambda x: 1.0 / (mp_af / x + mp_bf / _mp.sqrt(x)), [_mp.mpf(d), _mp.mpf(d0)]
        )
        mp_value = float(mp_value)
        return {
            "value": mp_value,
            "method": "mpmath-40dps",
            "err": abs(quad_value - mp_value),
            "quad_value": quad_value,
            "quad_abs_err_estimate": quad_err,
        }
    return {"value": quad_value, "method": "scipy-quad", "err": quad_err}


def textbook_primitive_time(props: ThermalProperties, d0: float, d: float, u: float,
                            include_sensible: bool) -> float:
    """The pre-audit implementation: duplicated here to measure its failure."""
    a, b = _analytic_case_constants(props, u, include_sensible=include_sensible)
    if b <= 1e-300:
        return (d0 * d0 - d * d) / (2.0 * a)

    def primitive(diameter: float) -> float:
        s = math.sqrt(diameter)
        return (
            2.0 * s**3 / (3.0 * b)
            - a * s**2 / b**2
            + 2.0 * a**2 * s / b**3
            - 2.0 * a**3 * math.log(a + b * s) / b**4
        )

    return primitive(d0) - primitive(d)


def check_closed_form(props: ThermalProperties) -> dict:
    rows = []
    for sensible in (False, True):
        for u in VELOCITIES:
            for d0, frac in ((0.30, 0.9), (0.05, 0.5), (0.02, 0.1)):
                d = d0 * frac
                ref = reference_time(d0, d, props, u, sensible)
                closed = analytic_ransom_marshall_time(
                    d0, d, u, props, include_sensible=sensible)
                rows.append(
                    {
                        "include_sensible": sensible,
                        "velocity_m_s": u,
                        "b_over_a_sqrt_m": _analytic_case_constants(
                            props, u, include_sensible=sensible)[1] / _analytic_case_constants(
                                props, u, include_sensible=sensible)[0],
                        "d0_m": d0,
                        "d_m": d,
                        "closed_form_s": closed,
                        "reference_s": ref["value"],
                        "reference_method": ref["method"],
                        "rel_error": _rel(closed, ref["value"]),
                    }
                )
    worst = max(rows, key=lambda r: r["rel_error"])
    return {"rows": rows, "max_rel_error": worst["rel_error"], "worst": worst}


def check_cancellation(props: ThermalProperties) -> dict:
    """Before/after the fix: textbook primitive vs the shipped implementation."""
    rows = []
    for sensible in (False, True):
        for u in (1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.25, 0.5):
            d0, d = 0.02, 0.01
            ref = reference_time(d0, d, props, u, sensible)
            old = textbook_primitive_time(props, d0, d, u, sensible)
            new = analytic_ransom_marshall_time(d0, d, u, props, include_sensible=sensible)
            eps = _analytic_case_constants(props, u, include_sensible=sensible)[1] * math.sqrt(
                d0) / _analytic_case_constants(props, u, include_sensible=sensible)[0]
            rows.append(
                {
                    "include_sensible": sensible,
                    "velocity_m_s": u,
                    "eps_b_sqrt_d0_over_a": eps,
                    "reference_s": ref["value"],
                    "reference_method": ref["method"],
                    "textbook_primitive_s": old,
                    "shipped_s": new,
                    "rel_error_textbook": _rel(old, ref["value"]),
                    "rel_error_shipped": _rel(new, ref["value"]),
                }
            )
    worst_old = max(rows, key=lambda r: r["rel_error_textbook"])
    worst_new = max(rows, key=lambda r: r["rel_error_shipped"])
    return {
        "rows": rows,
        "max_rel_error_textbook": worst_old["rel_error_textbook"],
        "max_rel_error_shipped": worst_new["rel_error_shipped"],
        "worst_textbook": worst_old,
        "worst_shipped": worst_new,
    }


def check_ode_consistency(props: ThermalProperties) -> dict:
    """d t/d D from the primitive vs 1/(dD/dt) from the rate function."""
    rows = []
    h = 1e-9
    for sensible in (False, True):
        for u in (0.05, 0.25, 0.5):
            for d0 in (0.05, 0.02):
                for frac in (0.9, 0.6, 0.3):
                    d = d0 * frac
                    tp = analytic_ransom_marshall_time(
                        d0, d + h, u, props, include_sensible=sensible)
                    tm = analytic_ransom_marshall_time(
                        d0, d - h, u, props, include_sensible=sensible)
                    dtdd = (tp - tm) / (2.0 * h)
                    rate = diameter_rate(d, u, props, include_sensible=sensible)
                    rows.append(
                        {
                            "include_sensible": sensible,
                            "velocity_m_s": u,
                            "d0_m": d0,
                            "d_m": d,
                            "dtdD_numeric": dtdd,
                            "one_over_rate": 1.0 / rate,
                            "rel_error": _rel(dtdd, 1.0 / rate),
                        }
                    )
    worst = max(rows, key=lambda r: r["rel_error"])
    return {"rows": rows, "max_rel_error": worst["rel_error"], "worst": worst}


def _rk2_error(props: ThermalProperties, u: float, sensible: bool, d0: float,
               frac: float, steps: int) -> dict:
    t_end = analytic_ransom_marshall_time(
        d0, d0 * frac, u, props, include_sensible=sensible)
    dt = t_end / steps
    hist = integrate_melting(
        d0, props, relative_velocity_m_s=u, duration_s=t_end, dt_s=dt,
        include_sensible=sensible)
    numeric = hist[-1]["diameter_m"]
    exact = analytic_diameter_ransom_marshall(
        d0, t_end, u, props, include_sensible=sensible)
    return {"steps": steps, "dt_s": dt, "diameter_numeric_m": numeric,
            "diameter_closed_form_m": exact, "abs_error_m": abs(numeric - exact),
            "rel_error": _rel(numeric, exact), "t_end_s": t_end}


def _observed_order(rows: list[dict]) -> tuple[float, float]:
    x = np.log(np.array([r["dt_s"] for r in rows]))
    y = np.log(np.array([r["abs_error_m"] for r in rows]))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), r2


def check_integrator_order(props: ThermalProperties) -> dict:
    steps = (20, 40, 80, 160, 320, 640, 1280)
    cases = {}
    for label, u, sensible, d0, frac in (
        ("conduction_u0", 0.0, False, 0.02, 0.5),
        ("conduction_u0_sensible", 0.0, True, 0.02, 0.5),
        ("ransom_marshall_u025", 0.25, False, 0.03, 0.4),
        ("ransom_marshall_u025_sensible", 0.25, True, 0.03, 0.4),
        ("ransom_marshall_u05", 0.5, False, 0.03, 0.4),
        ("ransom_marshall_u0001", 1e-4, False, 0.03, 0.4),
    ):
        rows = [_rk2_error(props, u, sensible, d0, frac, n) for n in steps]
        slope, r2 = _observed_order(rows)
        cases[label] = {
            "velocity_m_s": u,
            "include_sensible": sensible,
            "d0_m": d0,
            "final_diameter_target_m": d0 * frac,
            "rows": rows,
            "observed_order": slope,
            "order_fit_r2": r2,
        }
    return cases


def check_operational_dt(props: ThermalProperties, operational_dt_s: float = 0.25) -> dict:
    """Error of the coupled-pass operating point (CouplingConfig.melt_dt_s = 0.25)."""
    rows = []
    for u in (0.0, 0.1, 0.25, 0.5):
        for d0 in (0.01, 0.03, 0.10):
            frac = 0.9
            t_end = analytic_ransom_marshall_time(
                d0, d0 * frac, u, props, include_sensible=False)
            n = max(1, int(round(t_end / operational_dt_s)))
            dt = t_end / n
            hist = integrate_melting(
                d0, props, relative_velocity_m_s=u, duration_s=t_end, dt_s=dt,
                include_sensible=False)
            numeric = hist[-1]["diameter_m"]
            exact = analytic_diameter_ransom_marshall(
                d0, t_end, u, props, include_sensible=False)
            rows.append(
                {
                    "velocity_m_s": u,
                    "d0_m": d0,
                    "requested_dt_s": operational_dt_s,
                    "used_dt_s": dt,
                    "steps": n,
                    "diameter_numeric_m": numeric,
                    "diameter_closed_form_m": exact,
                    "rel_error": _rel(numeric, exact),
                    "melt_fraction": hist[-1]["melt_fraction"],
                }
            )
    worst = max(rows, key=lambda r: r["rel_error"])
    return {"operational_dt_s": operational_dt_s, "rows": rows,
            "max_rel_error": worst["rel_error"], "worst": worst}


def check_inversion(props: ThermalProperties) -> dict:
    rows = []
    for u in (0.0, 0.05, 0.25, 0.5):
        d0 = 0.03
        t_full = analytic_ransom_marshall_time(
            d0, d0 * 1e-9, u, props, include_sensible=False)
        for frac in (0.1, 0.25, 0.5, 0.75, 0.95):
            t = t_full * frac
            d = analytic_diameter_ransom_marshall(
                d0, t, u, props, include_sensible=False)
            t_back = analytic_ransom_marshall_time(
                d0, d, u, props, include_sensible=False)
            rows.append(
                {
                    "velocity_m_s": u,
                    "diameter_m": d,
                    "time_s": t,
                    "back_substituted_time_s": t_back,
                    "rel_error": _rel(t_back, t),
                }
            )
    worst = max(rows, key=lambda r: r["rel_error"])
    return {"rows": rows, "max_rel_error": worst["rel_error"], "worst": worst}


def check_degenerate_limit(props: ThermalProperties) -> dict:
    rows = []
    for d0, frac in ((0.05, 0.5), (0.02, 0.25), (0.01, 0.9)):
        d = d0 * frac
        t_zero_velocity = analytic_ransom_marshall_time(
            d0, d, 0.0, props, include_sensible=False)
        ref = reference_time(d0, d, props, 0.0, False)["value"]
        rows.append(
            {
                "d0_m": d0,
                "d_m": d,
                "closed_form_u0_s": t_zero_velocity,
                "conduction_law_s": ref,
                "rel_error": _rel(t_zero_velocity, ref),
            }
        )
    d0 = 0.02
    t = analytic_conduction_melt_time(d0, props, include_sensible=False) * 0.5
    via_rm = analytic_diameter_ransom_marshall(
        d0, t, 0.0, props, include_sensible=False)
    via_cond = analytic_diameter_conduction(d0, t, props, include_sensible=False)
    history = integrate_melting(d0, props, relative_velocity_m_s=0.0,
                                duration_s=t, dt_s=t / 800.0,
                                include_sensible=False)
    worst = max(rows, key=lambda r: r["rel_error"])
    return {
        "rows": rows,
        "max_rel_error": worst["rel_error"],
        "diameter_u0": {
            "time_s": t,
            "ransom_marshall_inverse_m": via_rm,
            "conduction_law_m": via_cond,
            "rk2_history_m": history[-1]["diameter_m"],
            "inverse_vs_conduction_rel_error": _rel(via_rm, via_cond),
            "rk2_vs_conduction_rel_error": _rel(history[-1]["diameter_m"], via_cond),
        },
    }


def check_vanish(props: ThermalProperties) -> dict:
    """A fragment that disappears inside the horizon: ledger closure and timing."""
    rows = []
    d0 = 1.0e-4
    u = 0.5
    for dt in (0.02, 0.01, 0.005):
        t_melt = analytic_ransom_marshall_time(
            d0, d0 * 1e-9, u, props, include_sensible=False)
        hist = integrate_melting(d0, props, relative_velocity_m_s=u,
                                 duration_s=t_melt * 2.0, dt_s=dt,
                                 include_sensible=False)
        vanish = next(r for r in hist if r["diameter_m"] <= 0.0)
        rows.append(
            {
                "dt_s": dt,
                "analytic_melt_time_s": t_melt,
                "numeric_vanish_time_s": vanish["time_s"],
                "timing_error_s": abs(vanish["time_s"] - t_melt),
                "timing_error_over_dt": abs(vanish["time_s"] - t_melt) / dt,
                "final_melt_fraction": hist[-1]["melt_fraction"],
                "melted_mass_kg": hist[-1]["melted_mass_kg"],
                "initial_mass_kg": hist[0]["mass_kg"],
                "mass_ledger_rel_error": _rel(hist[-1]["melted_mass_kg"], hist[0]["mass_kg"]),
            }
        )
    return {"rows": rows,
            "max_timing_error_over_dt": max(r["timing_error_over_dt"] for r in rows),
            "max_mass_ledger_rel_error": max(r["mass_ledger_rel_error"] for r in rows)}


def _figures(out: Path, cancellation: dict, order: dict, vanish: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []

    # 1) cancellation: textbook primitive vs shipped vs quadrature reference
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for sensible, marker in ((False, "o"), (True, "s")):
        sub = [r for r in cancellation["rows"] if r["include_sensible"] is sensible]
        x = [max(r["velocity_m_s"], 1e-17) for r in sub]
        ax.loglog(x, [max(r["rel_error_textbook"], 1e-18) for r in sub], marker + "-",
                  ms=4, lw=1.0, label=f"textbook primitive (sensible={sensible})")
        ax.loglog(x, [max(r["rel_error_shipped"], 1e-18) for r in sub], marker + "--",
                  ms=3, lw=0.9, mfc="none", label=f"shipped (sensible={sensible})")
    ax.axhline(1e-12, ls=":", c="k", lw=0.8, label="round-off floor 1e-12")
    ax.axhline(1e-2, ls="--", c="r", lw=0.8, label="1 % accuracy line")
    ax.set_xlabel("relative velocity u [m/s]")
    ax.set_ylabel("|closed form − quadrature| / quadrature")
    ax.set_title("Ranz–Marshall closed form: cancellation before/after fix")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / "closed_form_error.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    # 2) RK2 convergence
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, case in order.items():
        ax.loglog([r["dt_s"] for r in case["rows"]],
                  [r["abs_error_m"] for r in case["rows"]],
                  "o-", ms=3, label=f"{label} (slope {case['observed_order']:.2f})")
    ref = order["ransom_marshall_u025"]["rows"]
    dt_arr = np.array([r["dt_s"] for r in ref])
    e0 = np.array([r["abs_error_m"] for r in ref])[0]
    ax.loglog(dt_arr, e0 * (dt_arr / dt_arr[0]) ** 2, "k--", lw=1.0,
              label="slope 2 reference")
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("|D(t_end) RK2 − closed form| [m]")
    ax.set_title("integrate_melting (RK2 midpoint) convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / "rk2_convergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    # 3) diameter history + disappearance timing
    props_fig = ThermalProperties()
    d0, u = 0.03, 0.25
    t_end = analytic_ransom_marshall_time(d0, d0 * 0.4, u, props_fig,
                                          include_sensible=False)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ts = np.linspace(0.0, t_end, 200)
    ax0.plot(ts, [analytic_diameter_ransom_marshall(d0, t, u, props_fig,
                                                    include_sensible=False) for t in ts],
             "k-", lw=1.6, label="closed form")
    for n, style in ((20, "o"), (160, "s")):
        hist = integrate_melting(d0, props_fig, relative_velocity_m_s=u,
                                 duration_s=t_end, dt_s=t_end / n,
                                 include_sensible=False)
        ax0.plot([r["time_s"] for r in hist], [r["diameter_m"] for r in hist],
                 style, ms=3, ls="-", lw=0.9, label=f"RK2, {n} steps")
    ax0.set_xlabel("t [s]")
    ax0.set_ylabel("D [m]")
    ax0.set_title(f"melting history, D0={d0} m, u={u} m/s")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.plot([r["dt_s"] for r in vanish["rows"]],
             [r["timing_error_over_dt"] for r in vanish["rows"]], "o-")
    ax1.axhline(1.0, ls="--", c="r", lw=0.9, label="1 step = hard bound")
    ax1.set_xlabel("dt [s]")
    ax1.set_ylabel("|vanish time − closed form| / dt")
    ax1.set_title("disappearance timing error (in units of dt)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    p = figdir / "diameter_history.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/threefield-thermal")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    props = ThermalProperties()
    closed = check_closed_form(props)
    cancellation = check_cancellation(props)
    consistency = check_ode_consistency(props)
    order = check_integrator_order(props)
    operational = check_operational_dt(props)
    inversion = check_inversion(props)
    degenerate = check_degenerate_limit(props)
    vanish = check_vanish(props)
    figures = _figures(out, cancellation, order, vanish)

    ledger = {
        "schema": SCHEMA,
        "properties": {
            "water_temperature_c": props.water_temperature_c,
            "melt_temperature_c": props.melt_temperature_c,
            "water_conductivity_w_m_k": props.water_conductivity_w_m_k,
            "water_kinematic_viscosity_m2_s": props.water_kinematic_viscosity_m2_s,
            "water_prandtl": props.water_prandtl,
            "latent_heat_j_kg": props.latent_heat_j_kg,
            "ice_density_kg_m3": props.ice_density_kg_m3,
            "ice_initial_temperature_c": props.ice_initial_temperature_c,
            "sensible_heat_j_kg": props.sensible_heat_j_kg,
        },
        "closed_form": closed,
        "cancellation": cancellation,
        "ode_consistency": consistency,
        "integrator_order": order,
        "operational_dt": operational,
        "inversion": inversion,
        "degenerate_limit": degenerate,
        "vanish": vanish,
        "figures": figures,
        "verdict": {
            "closed_form_max_rel_error": closed["max_rel_error"],
            "cancellation_max_rel_error_textbook": cancellation["max_rel_error_textbook"],
            "cancellation_max_rel_error_shipped": cancellation["max_rel_error_shipped"],
            "ode_consistency_max_rel_error": consistency["max_rel_error"],
            "min_observed_order": min(c["observed_order"] for c in order.values()),
            "max_deviation_from_order_2": max(abs(c["observed_order"] - 2.0)
                                              for c in order.values()),
            "operational_dt_max_rel_error": operational["max_rel_error"],
            "inversion_max_rel_error": inversion["max_rel_error"],
            "degenerate_limit_max_rel_error": degenerate["max_rel_error"],
            "vanish_max_timing_error_over_dt": vanish["max_timing_error_over_dt"],
            "vanish_max_mass_ledger_rel_error": vanish["max_mass_ledger_rel_error"],
        },
    }
    path = out / "thermal_verification.json"
    path.write_text(json.dumps(ledger, indent=2) + "\n")

    def _csv(name: str, rows: list[dict]) -> None:
        keys = list(rows[0])
        lines = [",".join(keys)]
        for r in rows:
            lines.append(",".join(str(r[k]) for k in keys))
        (out / name).write_text("\n".join(lines) + "\n")

    _csv("closed_form_error.csv", closed["rows"])
    _csv("cancellation_before_after.csv", cancellation["rows"])
    _csv("ode_consistency.csv", consistency["rows"])
    _csv("rk2_convergence.csv",
         [{**r, "case": label} for label, c in order.items() for r in c["rows"]])
    _csv("operational_dt_error.csv", operational["rows"])
    _csv("inversion_residual.csv", inversion["rows"])
    _csv("vanish_timing.csv", vanish["rows"])

    print(json.dumps(ledger["verdict"], indent=2))
    print(f"wrote {path}")
    for label, case in order.items():
        print(f"  order[{label}] = {case['observed_order']:.3f} (R2={case['order_fit_r2']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())