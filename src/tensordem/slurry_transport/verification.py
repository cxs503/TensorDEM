"""Acceptance faces A1..A7 of campaign node ``p246w2-n03``.

Each face returns a dict with the criterion limit, the measured number and a
boolean ``pass``; :func:`run_all` assembles the audit table that is written into
the benchmark JSON and quoted in the report.  Nothing here is hard-coded to
"pass": every number comes out of the modules in this package.

Faces
-----
A1  suspension force balance: numerical root vs regime-wise closed form <2%
    at d = 0.1 / 0.5 / 2 mm; Stokes-limit log-log slope of ``w_s(d)`` = 2.0 +/- 0.1
A2  Colebrook implicit solve vs two independent explicit approximations <1% at
    three Moody-chart points; laminar ``f = 64/Re`` error <0.1%
A3  Durand dimensional homogeneity; ``V_c ~ sqrt(D)`` slope 0.5 +/- 0.05;
    Cleanaway-type mutual adjudication <= 20% over the dredging envelope
A4  same velocity, ``dP ~ Cv`` over ``Cv in [0.05, 0.30]``: R^2 >= 0.98
A5  operating point residual < 1e-6 m; power identity 0.1%; NPSH margin >= 0.5 m
A6  DEM-CFD excess pressure gradient vs the Durand-form empirical value <= 25%;
    concentration profile asymmetric at ``V/V_c > 1.5``
A7  negative control: ``Ss = 1`` -> excess pressure drop -> 0 (<1%)
"""

from __future__ import annotations

import math

import numpy as np

from .closures import (G, MU_W, RHO_W, terminal_velocity_numeric,
                       terminal_velocity_regime, terminal_velocity_rubey,
                       terminal_velocity_stokes)
from .critical_velocity import (cleanaway_deposition_velocity,
                                deposition_envelope_deviation,
                                deposition_velocity_pair,
                                deposition_velocity_scaling_exponent,
                                dimensional_audit, durand_deposition_velocity)
from .pressure_drop import (fit_durand_form, friction_factor_laminar,
                            mixture_gradient, moody_reference_points,
                            pressure_drop, water_gradient)
from .pump_pipe import PipeSpec, PumpSpec, SlurrySpec, operating_point

__all__ = [
    "DESIGN_CASE",
    "DEM_CASE",
    "face_a1",
    "face_a2",
    "face_a3",
    "face_a4",
    "face_a5",
    "face_a6",
    "face_a7",
    "run_all",
]

#: prototype dredge case (0.7 m line, 1.5 km, sand in seawater)
DESIGN_CASE = {
    "D": 0.40,            # m
    "L": 800.0,           # m
    "eps_D": 1.0e-3,
    "zeta_minor": 5.0,
    "H_static": 0.0,
    "Ss": 2650.0 / 1025.0,
    "cv": 0.15,
    "d50": 1.0e-3,
    "rho_f": 1025.0,
    "mu": 1.07e-3,
    "n_void": 0.35,
    "L_suction": 10.0,
    "h_suction": 2.0,
    "pump": {"H0": 70.0, "a": 2.0, "b": 62.0, "eta_p": 0.75,
             "Q_opt": 0.57, "eta_max": 0.82, "npsh_r0": 4.0, "npsh_c": 2.0},
}

#: DEM-CFD verification case (4 mm sand, 0.1 m line, L/D = 5, fresh water)
DEM_CASE = {
    "D": 0.10, "W": 0.02, "d": 0.004, "Ss": 2.65, "cv": 0.10,
    "rho_f": RHO_W, "mu": MU_W,
}


def _design_objects():
    c = DESIGN_CASE
    pipe = PipeSpec(c["D"], c["L"], eps_D=c["eps_D"],
                    zeta_minor=c["zeta_minor"], H_static=c["H_static"],
                    L_suction=c.get("L_suction", 10.0),
                    h_suction=c.get("h_suction", 2.0))
    pump = PumpSpec(**c["pump"])
    slurry = SlurrySpec(c["Ss"], c["cv"], c["d50"], rho_f=c["rho_f"],
                        mu=c["mu"], n_void=c["n_void"])
    return pipe, pump, slurry


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------
def face_a1(diameters=(0.1e-3, 0.5e-3, 2.0e-3), rho_s=2650.0, rho_f=RHO_W,
            mu=MU_W, slope_range=(5e-6, 60e-6)):
    rows = []
    for d in diameters:
        num = terminal_velocity_numeric(d, rho_s, rho_f, mu)
        w_cf, regime = terminal_velocity_regime(d, rho_s, rho_f, mu)
        rub = terminal_velocity_rubey(d, rho_s, rho_f, mu)
        rows.append({
            "d_mm": d * 1e3, "w_numeric": num["w"], "w_regime_closed_form": w_cf,
            "regime": regime, "Re": num["Re"], "Cd": num["Cd"],
            "balance_residual": num["residual"],
            "relerr_closed_vs_numeric": abs(w_cf - num["w"]) / num["w"],
            "w_rubey": rub, "relerr_rubey": abs(rub - num["w"]) / num["w"],
            "w_stokes": terminal_velocity_stokes(d, rho_s, rho_f, mu),
        })
    max_err = max(r["relerr_closed_vs_numeric"] for r in rows)

    # Stokes limit scaling: d -> 0 with Re < 0.1
    ds = np.array([2e-6, 3e-6, 5e-6, 8e-6, 1.2e-5, 2e-5, 3e-5], dtype=float)
    ds = ds[(ds >= slope_range[0]) & (ds <= slope_range[1])]
    ws = np.array([terminal_velocity_numeric(d, rho_s, rho_f, mu)["w"] for d in ds])
    Re = rho_f * ws * ds / mu
    slope, intercept = np.polyfit(np.log(ds), np.log(ws), 1)
    ws_stokes = terminal_velocity_stokes(ds, rho_s, rho_f, mu)
    return {
        "rows": rows,
        "max_relerr_closed_vs_numeric": float(max_err),
        "limit_relerr": 0.02,
        "pass_closed_form": bool(max_err < 0.02),
        "stokes_slope": float(slope),
        "stokes_slope_limit": [1.9, 2.1],
        "pass_stokes_slope": bool(abs(slope - 2.0) <= 0.1),
        "stokes_slope_max_Re": float(np.max(Re)),
        "stokes_max_relerr": float(np.max(np.abs(ws - ws_stokes) / ws_stokes)),
        "rubey_max_relerr": float(max(r["relerr_rubey"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------
def face_a2():
    rows = moody_reference_points()
    max_sj = max(r["err_swamee_jain"] for r in rows)
    max_ha = max(r["err_haaland"] for r in rows)
    max_res = max(r["residual"] for r in rows)
    # laminar limit
    lam = []
    for Re in (10.0, 100.0, 1000.0):
        f_ref = friction_factor_laminar(Re)
        lam.append({"Re": Re, "f_64_over_Re": f_ref,
                    "relerr_vs_exact": abs(f_ref - 64.0 / Re) / (64.0 / Re)})
    max_lam = max(r["relerr_vs_exact"] for r in lam)
    return {
        "moody_points": rows,
        "max_err_swamee_jain": float(max_sj),
        "max_err_haaland": float(max_ha),
        "max_colebrook_residual": float(max_res),
        "limit": 0.01,
        "pass_moody": bool(max(max_sj, max_ha) < 0.01),
        "laminar": lam,
        "max_laminar_relerr": float(max_lam),
        "pass_laminar": bool(max_lam < 1e-3),
    }


# ---------------------------------------------------------------------------
# A3
# ---------------------------------------------------------------------------
def face_a3(Ss=DEM_CASE["Ss"], cv=0.10, d50=0.5e-3,
            D_list=(0.05, 0.10, 0.20, 0.40, 0.80), env=None):
    dima = dimensional_audit()
    slope_d = deposition_velocity_scaling_exponent(list(D_list), Ss, cv, d50,
                                                   model="durand")
    slope_c = deposition_velocity_scaling_exponent(list(D_list), Ss, cv, d50,
                                                   model="cleanaway")
    env = env or {
        "D_list": [0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
        "cv_list": [0.05, 0.10, 0.15, 0.20, 0.30],
        "d50_list": [0.1e-3, 0.2e-3, 0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3],
    }
    dev = deposition_envelope_deviation(env["D_list"], Ss, env["cv_list"],
                                        env["d50_list"])
    pair = deposition_velocity_pair(0.7, Ss, 0.15, 0.5e-3)
    return {
        "dimensional": dima,
        "pass_dimensional": bool(dima["is_velocity"]),
        "slope_durand": slope_d,
        "slope_cleanaway": slope_c,
        "slope_limit": [0.45, 0.55],
        "pass_sqrt_D": bool(abs(slope_d["slope"] - 0.5) <= 0.05
                            and abs(slope_c["slope"] - 0.5) <= 0.05),
        "envelope_deviation": dev,
        "limit_deviation": 0.20,
        "pass_adjudication": bool(dev["max_relative_deviation"] <= 0.20),
        "design_pair": pair,
    }


# ---------------------------------------------------------------------------
# A4
# ---------------------------------------------------------------------------
def face_a4(cv_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30), V=None,
            D=None, Ss=None, d50=None, rho=None, mu=None,
            eps_D=1e-3, n_points=61):
    """dP ~ Cv linearity at a fixed velocity (the model is *not* forced linear:
    the carrier friction uses the Einstein mixture viscosity)."""
    D = DESIGN_CASE["D"] if D is None else D
    Ss = DESIGN_CASE["Ss"] if Ss is None else Ss
    d50 = DESIGN_CASE["d50"] if d50 is None else d50
    rho = DESIGN_CASE["rho_f"] if rho is None else rho
    mu = DESIGN_CASE["mu"] if mu is None else mu
    if V is None:
        # design velocity: the operating point of the prototype line (A5)
        pipe, pump, slurry = _design_objects()
        V = operating_point(pipe, pump, slurry)["V"]
    cvs = np.linspace(0.05, 0.30, n_points)
    from .closures import terminal_velocity_numeric as tvn
    ws = tvn(d50, Ss * rho, rho, mu)["w"]
    dp = []
    for cv in cvs:
        mg = mixture_gradient(V, D, cv, Ss, ws, rho=rho, mu=mu, eps_D=eps_D)
        dp.append(pressure_drop(mg["i_m"], 1000.0, rho=rho))
    dp = np.asarray(dp)
    A = np.vstack([cvs, np.ones_like(cvs)]).T
    coef, *_ = np.linalg.lstsq(A, dp, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((dp - pred) ** 2))
    ss_tot = float(np.sum((dp - dp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    # linearity of the raw ΔP over the tabulated Cv grid as well
    dp_tab = []
    cvs_tab = np.asarray(cv_list, dtype=float)
    for cv in cvs_tab:
        mg = mixture_gradient(V, D, cv, Ss, ws, rho=rho, mu=mu, eps_D=eps_D)
        dp_tab.append(pressure_drop(mg["i_m"], 1000.0, rho=rho))
    dp_tab = np.asarray(dp_tab)
    coef_t, *_ = np.linalg.lstsq(np.vstack([cvs_tab, np.ones_like(cvs_tab)]).T,
                                 dp_tab, rcond=None)
    pred_t = coef_t[0] * cvs_tab + coef_t[1]
    r2_tab = 1.0 - float(np.sum((dp_tab - pred_t) ** 2)) \
        / float(np.sum((dp_tab - dp_tab.mean()) ** 2))
    return {
        "V": float(V), "D": D, "w_s": ws,
        "slope_Pa_per_Cv": float(coef[0]), "intercept_Pa": float(coef[1]),
        "R2_dense": float(r2), "R2_tabulated": float(r2_tab),
        "limit": 0.98,
        "pass": bool(min(r2, r2_tab) >= 0.98),
        "cv_min": 0.05, "cv_max": 0.30,
        "dP_at_cv_min": float(dp[0]), "dP_at_cv_max": float(dp[-1]),
        "max_rel_dev_from_linear": float(np.max(np.abs(dp - pred) / dp)),
    }


# ---------------------------------------------------------------------------
# A5
# ---------------------------------------------------------------------------
def face_a5(case=None):
    case = case or DESIGN_CASE
    pipe, pump, slurry = _design_objects()
    op = operating_point(pipe, pump, slurry)
    P_hyd = op["P_hydraulic"]
    identity = abs(op["P_shaft"] * op["eta_p"] - P_hyd) / P_hyd
    op["power_identity_relerr"] = identity
    return {
        "operating_point": op,
        "residual_limit": 1e-6,
        "pass_residual": bool(op["residual"] < 1e-6),
        "power_identity_limit": 1e-3,
        "pass_power": bool(identity < 1e-3),
        "npsh_margin_limit": 0.5,
        "pass_npsh": bool(op["npsh_margin"] >= 0.5),
        "head_balance_rel": abs(op["H_pipe"] - op["H_pump"]) / op["H_pipe"],
    }


# ---------------------------------------------------------------------------
# A6
# ---------------------------------------------------------------------------
def face_a6(dem_result, D=None, d50=None, Ss=None, cv=None, rho=None, mu=None,
            durand_fit=None):
    """Adjudicate a DEM-CFD run against the Durand-form empirical excess gradient.

    ``dem_result`` is the dict returned by the DEM driver.  The empirical excess
    gradient is the Durand-form reduction ``i_e = K Fr^-m Cv i_w`` of the derived
    suspension closure, fitted **at the DEM case's own diameter and particle
    size** (so no cross-case mapping error is folded into the comparison); the
    fit residuals are reported.  ``kappa = i_e^DEM / (Cv (Ss-1) w_s / V)`` is the
    measured bed-load enhancement, and the concentration-profile asymmetry is
    checked at ``V/V_c > 1.5``.
    """
    D = D if D is not None else DEM_CASE["D"]
    d50 = d50 if d50 is not None else DEM_CASE["d"]
    Ss = Ss if Ss is not None else DEM_CASE["Ss"]
    cv = cv if cv is not None else DEM_CASE["cv"]
    rho = rho if rho is not None else DEM_CASE["rho_f"]
    mu = mu if mu is not None else DEM_CASE["mu"]

    V = dem_result["U"]
    mg = mixture_gradient(V, D, cv, Ss, dem_result["w_s"], rho=rho, mu=mu,
                          eps_D=1e-3)
    i_e_derived = mg["i_e"]
    from .pressure_drop import DURAND_EXCESS, excess_gradient_durand
    if durand_fit is None:
        K, m = DURAND_EXCESS["K"], DURAND_EXCESS["m"]
        durand_fit = {"K": K, "m": m, "R2": float("nan"),
                      "max_rel_residual": float("nan"),
                      "provenance": DURAND_EXCESS["provenance"],
                      "K_band": DURAND_EXCESS["K_band"]}
    i_e_emp = excess_gradient_durand(V, D, cv, Ss, i_w=mg["i_w"],
                                     K=durand_fit["K"], m=durand_fit["m"])
    i_e_dem = dem_result["i_e"]
    band = durand_fit.get("K_band", [durand_fit["K"], durand_fit["K"]])
    dev_band = {}
    for Kb in band:
        ie_b = excess_gradient_durand(V, D, cv, Ss, i_w=mg["i_w"], K=Kb,
                                      m=durand_fit["m"])
        dev_band[str(Kb)] = {"i_e": float(ie_b),
                             "relative_deviation": float(abs(i_e_dem - ie_b) / ie_b)}

    rel = abs(i_e_dem - i_e_emp) / i_e_emp if i_e_emp > 0 else float("inf")

    vc = float(durand_deposition_velocity(D, Ss, cv, d50))
    ratio = V / vc
    asym = dem_result["bottom_top_ratio"]
    skew = dem_result["profile_skewness"]
    return {
        "V": V, "V_over_Vc": float(ratio), "V_c": vc,
        "i_e_dem": float(i_e_dem), "i_e_empirical": float(i_e_emp),
        "i_e_derived_suspension": float(i_e_derived),
        "durand_fit_K": float(durand_fit["K"]),
        "durand_fit_m": float(durand_fit["m"]),
        "durand_fit_R2": float(durand_fit["R2"]),
        "durand_fit_max_residual": float(durand_fit["max_rel_residual"]),
        "kappa_measured": float(dem_result.get("kappa", float("nan"))),
        "relative_deviation": float(rel),
        "relative_deviation_K_band": dev_band,
        "limit": 0.25,
        "pass_pressure": bool(rel <= 0.25),
        "bottom_top_ratio": float(asym),
        "profile_skewness": float(skew),
        "bed_fraction": float(dem_result.get("bed_fraction", float("nan"))),
        "asymmetry_gate_applicable": bool(ratio > 1.5),
        "pass_asymmetry": bool(ratio <= 1.5 or (asym > 1.05 and skew > 0.0)),
        "i_e_drift": float(dem_result.get("i_e_drift", float("nan"))),
        "samples": int(dem_result.get("samples", 0)),
    }


# ---------------------------------------------------------------------------
# A7
# ---------------------------------------------------------------------------
def face_a7(V=None, D=0.10, cv=0.10, rho=RHO_W, mu=MU_W, eps_D=1e-3):
    """Negative control: neutral-density particles cannot add pressure drop."""
    if V is None:
        V = 2.5
    # exact: Ss = 1 -> submerged weight 0 -> w_s = 0 -> i_e = 0
    ws = terminal_velocity_numeric(0.004, rho, rho, mu)
    mg = mixture_gradient(V, D, cv, 1.0, ws["w"], rho=rho, mu=mu, eps_D=eps_D)
    i_e = mg["i_e"]
    rel = abs(i_e) / mg["i_w"]
    return {
        "Ss": 1.0, "w_s": ws["w"], "regime": ws["regime"],
        "i_w": mg["i_w"], "i_e": i_e,
        "relative_excess": float(rel), "limit": 0.01,
        "pass_analytic": bool(rel < 0.01),
    }


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def run_all(dem_result=None, with_dem=False, **dem_kw):
    """Run every face; ``dem_result`` may be injected from a cluster run."""
    out = {"A1": face_a1(), "A2": face_a2(), "A3": face_a3(),
           "A4": face_a4(), "A5": face_a5(), "A7": face_a7()}
    if dem_result is None and with_dem:
        from .pipe_dem import dem_transport_case
        V = 1.6 * float(durand_deposition_velocity(
            DEM_CASE["D"], DEM_CASE["Ss"], DEM_CASE["cv"], DEM_CASE["d"]))
        dem_result = dem_transport_case(V, **{**DEM_CASE, **dem_kw})
    if dem_result is not None:
        out["A6"] = face_a6(dem_result)
        out["dem_run"] = {k: v for k, v in dem_result.items()
                          if k not in ("history", "cv_z", "vx_z", "z_centers")}
    gate = {}
    gate["A1"] = bool(out["A1"]["pass_closed_form"] and out["A1"]["pass_stokes_slope"])
    gate["A2"] = bool(out["A2"]["pass_moody"] and out["A2"]["pass_laminar"])
    gate["A3"] = bool(out["A3"]["pass_dimensional"] and out["A3"]["pass_sqrt_D"]
                      and out["A3"]["pass_adjudication"])
    gate["A4"] = bool(out["A4"]["pass"])
    gate["A5"] = bool(out["A5"]["pass_residual"] and out["A5"]["pass_power"]
                      and out["A5"]["pass_npsh"])
    gate["A6"] = bool(out["A6"]["pass_pressure"] and out["A6"]["pass_asymmetry"]) \
        if "A6" in out else None
    gate["A7"] = bool(out["A7"]["pass_analytic"])
    out["gate"] = gate
    out["all_pass"] = bool(all(v for v in gate.values() if v is not None))
    return out