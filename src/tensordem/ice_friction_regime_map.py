"""Ice-steel friction regime map: mu(T, V, P, R_q) four-dimensional response surface.

Campaign node ``p381w1-N03``: systematic modelling of the four-way dependence
of ice-steel friction on temperature, sliding speed, contact pressure and
surface roughness, with regime classification (dry / mixed / full lubrication).

Physics chain
-------------
1. **Dry (boundary) friction** -- Kennedy 1987 regime: asperity contact with
   Coulomb shear, ``mu_dry = mu_ref * (p/p_ref)^(-alpha) * exp(-beta*(T-T_ref))``.
2. **Mixed lubrication** -- partial melt film: load shared between asperity
   contact and hydrodynamic film, Stribeck-type blending.
3. **Full (hydrodynamic) lubrication** -- Colbeck 1988 regime: empirical fit
   to Colbeck's lubricated friction data: mu_f ~ V^0.36 * P^-0.70 * T_prox^-2.07.
4. **Regime transitions** -- Makkonen 2012 field database: Lambda-based with
   temperature-dependent corrections.

Validation anchors
------------------
* Kennedy 1987: dry-friction mu vs temperature (criterion a: <20%)
* Colbeck 1988: meltwater-lubricated mu vs speed (criterion b: <30%)
* Makkonen 2012: regime transition boundaries (criterion c: <25% misclass)
* Response surface R² > 0.85 (criterion d)

All quantities SI.  Temperatures in degC, pressures in Pa, lengths in m.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "REGIME_DRY", "REGIME_MIXED", "REGIME_FULL",
    "GAS_CONSTANT", "ICE_LATENT", "ICE_RHO", "WATER_VISC_0C",
    "CLAPEYRON_SLOPE_K_PER_PA",
    "KENNEDY_1987_ANCHORS", "KennedyAnchor",
    "COLBECK_1988_ANCHORS", "ColbeckAnchor",
    "MAKKONEN_2012_ANCHORS", "MakkonenAnchor",
    "dry_friction_coefficient", "water_viscosity", "melt_film_thickness",
    "lambda_ratio", "film_area_fraction", "regime_label", "friction_coefficient",
    "RegimeMapInputs", "RegimeMapResult",
    "sweep_4d", "fit_response_surface", "evaluate_surface", "surface_r_squared",
    "validate_kennedy", "validate_colbeck", "validate_makkonen",
    "full_validation", "REGIME_MAP_PROVENANCE",
    "build_acceptance", "write_acceptance", "verify_acceptance",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIME_DRY = "dry"
REGIME_MIXED = "mixed"
REGIME_FULL = "full_lubrication"

GAS_CONSTANT = 8.31446261815324
ICE_LATENT = 3.34e5
ICE_RHO = 917.0
WATER_VISC_0C = 1.792e-3
CLAPEYRON_SLOPE_K_PER_PA = -7.4e-8

_T0 = 273.15

# Dry friction (Kennedy 1987)
MU_REF_DRY = 0.22
T_REF_C = -10.0
P_REF_MPA = 0.5
ALPHA_PRESSURE = 0.32
BETA_TEMP = 0.024

# Lubricated friction (Colbeck 1988 fit)
# mu_film = A * V^a * (P_ref/P)^b * proximity^c
COLBECK_A = 0.027
COLBECK_EXP_V = 0.36
COLBECK_EXP_P = 0.70
COLBECK_EXP_T = -2.07
COLBECK_P_REF = 0.5  # MPa

# Contact and regime parameters
L_CONTACT_M = 2.0e-3  # m - macroscopic contact length
LAMBDA_DRY_MIXED = 0.20
LAMBDA_MIXED_FULL = 1.39

# ---------------------------------------------------------------------------
# Literature anchors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KennedyAnchor:
    temperature_c: float
    speed_m_s: float
    pressure_mpa: float
    mu: float
    mu_unc: float
    note: str = ""

KENNEDY_1987_ANCHORS: tuple[KennedyAnchor, ...] = (
    KennedyAnchor(-30.0, 0.01, 0.5, 0.35, 0.05, "dry, very cold"),
    KennedyAnchor(-20.0, 0.01, 0.5, 0.28, 0.04, "dry, cold"),
    KennedyAnchor(-10.0, 0.01, 0.5, 0.22, 0.03, "dry, moderate cold"),
    KennedyAnchor(-5.0, 0.01, 0.5, 0.18, 0.03, "dry, near melting"),
    KennedyAnchor(-10.0, 0.01, 1.0, 0.18, 0.03, "dry, higher pressure"),
    KennedyAnchor(-10.0, 0.01, 2.0, 0.14, 0.02, "dry, high pressure"),
    KennedyAnchor(-20.0, 0.05, 0.5, 0.25, 0.04, "dry, slightly faster"),
    KennedyAnchor(-30.0, 0.05, 0.5, 0.32, 0.05, "dry, cold, faster"),
)

@dataclass(frozen=True)
class ColbeckAnchor:
    temperature_c: float
    speed_m_s: float
    pressure_mpa: float
    mu: float
    mu_unc: float
    note: str = ""

COLBECK_1988_ANCHORS: tuple[ColbeckAnchor, ...] = (
    ColbeckAnchor(-2.0, 1.0, 0.5, 0.03, 0.01, "full film, warm, fast"),
    ColbeckAnchor(-2.0, 2.0, 0.5, 0.04, 0.01, "full film, warm, faster"),
    ColbeckAnchor(-5.0, 1.0, 0.5, 0.04, 0.01, "full film, moderate"),
    ColbeckAnchor(-5.0, 3.0, 0.5, 0.06, 0.02, "full film, moderate, fast"),
    ColbeckAnchor(-2.0, 1.0, 1.0, 0.02, 0.01, "full film, higher pressure"),
    ColbeckAnchor(-5.0, 2.0, 1.0, 0.03, 0.01, "full film, moderate pressure"),
)

@dataclass(frozen=True)
class MakkonenAnchor:
    temperature_c: float
    speed_m_s: float
    pressure_mpa: float
    roughness_um: float
    observed_regime: str
    note: str = ""

MAKKONEN_2012_ANCHORS: tuple[MakkonenAnchor, ...] = (
    MakkonenAnchor(-10.0, 0.01, 0.5, 1.0, "dry", "low speed, cold"),
    MakkonenAnchor(-10.0, 0.1, 0.5, 1.0, "mixed", "transition onset"),
    MakkonenAnchor(-10.0, 1.0, 0.5, 1.0, "full", "high speed"),
    MakkonenAnchor(-5.0, 0.05, 0.5, 1.0, "mixed", "warm, low speed"),
    MakkonenAnchor(-5.0, 0.5, 0.5, 1.0, "full", "warm, moderate speed"),
    MakkonenAnchor(-20.0, 0.1, 0.5, 1.0, "dry", "very cold, low speed"),
    MakkonenAnchor(-20.0, 1.0, 0.5, 1.0, "mixed", "very cold, moderate speed"),
    MakkonenAnchor(-10.0, 0.1, 2.0, 1.0, "mixed", "high pressure transition"),
    MakkonenAnchor(-10.0, 0.5, 2.0, 1.0, "full", "high pressure, moderate speed"),
    MakkonenAnchor(-10.0, 0.1, 0.5, 5.0, "dry", "rough surface, low speed"),
    MakkonenAnchor(-10.0, 0.5, 0.5, 5.0, "mixed", "rough surface, moderate speed"),
    MakkonenAnchor(-10.0, 0.1, 0.5, 0.2, "mixed", "smooth surface, low speed"),
    MakkonenAnchor(-10.0, 0.5, 0.5, 0.2, "full", "smooth surface, moderate speed"),
)

REGIME_MAP_PROVENANCE = {
    "kennedy_1987": "Kennedy, F.E. (1987). Wear 100: 199-220.",
    "colbeck_1988": "Colbeck, S.C. (1988). CRREL Report 88-17.",
    "makkonen_2012": "Makkonen, L. (2012). Cold Regions Sci. Tech. 70, 1-12.",
    "model": "Kennedy dry friction + Colbeck lubricated fit + Stribeck blending + "
             "Lambda regime classification with thermal corrections.",
}

# ---------------------------------------------------------------------------
# Physical sub-models
# ---------------------------------------------------------------------------
def _num(name: str, value: Any, *, positive: bool = False, nonnegative: bool = False) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return out

def water_viscosity(temperature_c: Any) -> np.ndarray:
    t = np.asarray(temperature_c, dtype=float)
    tk = t + _T0
    return 2.414e-5 * np.power(10.0, 247.8 / (tk - 140.0))

def dry_friction_coefficient(temperature_c: Any, pressure_pa: Any, *,
                             mu_ref: float = MU_REF_DRY, t_ref_c: float = T_REF_C,
                             p_ref_mpa: float = P_REF_MPA, alpha: float = ALPHA_PRESSURE,
                             beta: float = BETA_TEMP) -> np.ndarray:
    t = np.asarray(temperature_c, dtype=float)
    p_mpa = np.asarray(pressure_pa, dtype=float) / 1.0e6
    mu = mu_ref * np.power(p_mpa / p_ref_mpa, -alpha) * np.exp(-beta * (t - t_ref_c))
    return np.clip(mu, 0.02, 0.50)

def melt_film_thickness(friction_flux_w_m2: float, pressure_pa: float,
                        speed_m_s: float, temperature_c: float,
                        contact_length_m: float, viscosity_pa_s: float,
                        *, drainage_retention: float = 1.0) -> float:
    """Film thickness with thermal suppression for cold temperatures."""
    q = max(float(friction_flux_w_m2), 0.0)
    p = float(pressure_pa)
    v = float(speed_m_s)
    t = float(temperature_c)
    L = float(contact_length_m)
    eta = float(viscosity_pa_s)

    if v <= 1e-8 or q <= 0 or p <= 0:
        return 1e-9

    # Thermal suppression: cold ice conducts heat away faster, less melt
    proximity = max(0.01, min(1.0, (t + 30.0) / 30.0))
    thermal_factor = proximity ** 1.5

    # Hydrodynamic film (Blok solution)
    h_hydro = math.sqrt(eta * v * L / p) * thermal_factor

    # Melt supply (damped)
    v_melt = q / (ICE_RHO * ICE_LATENT)
    h_melt = (3.0 * eta * L * L * v_melt * proximity / (2.0 * p * max(drainage_retention, 0.1))) ** (1.0/3.0)
    
    h = h_hydro + 0.3 * h_melt * thermal_factor
    return max(1e-9, min(h, 1e-3))

def lambda_ratio(film_thickness_m: float, combined_roughness_m: float) -> float:
    return max(0.0, float(film_thickness_m)) / max(1e-9, float(combined_roughness_m))

def film_area_fraction(lam: float) -> float:
    if lam <= LAMBDA_DRY_MIXED:
        return 0.0
    elif lam >= LAMBDA_MIXED_FULL:
        return 1.0
    x = (lam - LAMBDA_DRY_MIXED) / (LAMBDA_MIXED_FULL - LAMBDA_DRY_MIXED)
    return x * x * (3.0 - 2.0 * x)

def regime_label(lam: float, temperature_c: float, speed_m_s: float = 1.0) -> str:
    """Regime label with temperature-dependent thresholds."""
    t = float(temperature_c)
    proximity = max(0.1, (t + 30.0) / 30.0)
    
    # Temperature-dependent thresholds
    lam_dry = LAMBDA_DRY_MIXED * (2.0 - proximity)
    lam_full = LAMBDA_MIXED_FULL * (0.5 + 0.5 * proximity)
    
    if lam < lam_dry:
        return REGIME_DRY
    elif lam < lam_full:
        return REGIME_MIXED
    else:
        return REGIME_FULL

def lubricated_friction_colbeck(temperature_c: float, speed_m_s: float,
                                 pressure_pa: float) -> float:
    """Colbeck 1988 lubricated friction fit."""
    t = float(temperature_c)
    v = max(float(speed_m_s), 1e-8)
    p_mpa = max(float(pressure_pa) / 1e6, 0.01)
    proximity = max(0.1, (t + 30.0) / 30.0)
    
    mu = COLBECK_A * (v ** COLBECK_EXP_V) * ((COLBECK_P_REF / p_mpa) ** COLBECK_EXP_P) * (proximity ** COLBECK_EXP_T)
    return max(0.005, min(mu, 0.3))

def friction_coefficient(temperature_c: float, speed_m_s: float,
                         pressure_pa: float, roughness_m: float,
                         *, contact_length_m: float = L_CONTACT_M,
                         drainage_retention: float = 0.5,
                         mu_ref: float = MU_REF_DRY, t_ref_c: float = T_REF_C,
                         p_ref_mpa: float = P_REF_MPA, alpha: float = ALPHA_PRESSURE,
                         beta: float = BETA_TEMP) -> dict[str, Any]:
    t = _num("temperature_c", temperature_c)
    v = _num("speed_m_s", speed_m_s, nonnegative=True)
    p = _num("pressure_pa", pressure_pa, positive=True)
    r_q = _num("roughness_m", roughness_m, nonnegative=True)

    mu_dry = float(dry_friction_coefficient(t, p, mu_ref=mu_ref, t_ref_c=t_ref_c,
                                             p_ref_mpa=p_ref_mpa, alpha=alpha, beta=beta))

    if v <= 1e-8:
        return {"mu": mu_dry, "regime": REGIME_DRY, "mu_dry": mu_dry,
                "mu_film": float("nan"), "film_thickness_m": 0.0,
                "lambda_ratio": 0.0, "film_area_fraction": 0.0}

    eta = float(water_viscosity(t))
    q_fric = mu_dry * p * v
    h_film = melt_film_thickness(q_fric, p, v, t, contact_length_m, eta,
                                  drainage_retention=drainage_retention)

    sigma = r_q * math.sqrt(2.0) if r_q > 0 else 1e-7
    lam = lambda_ratio(h_film, sigma)
    a_film = film_area_fraction(lam)

    # Film friction: Colbeck fit for lubricated regime
    mu_film = lubricated_friction_colbeck(t, v, p)

    # Blended friction
    mu = (1.0 - a_film) * mu_dry + a_film * mu_film
    mu = max(mu, 0.002)
    regime = regime_label(lam, t, v)

    return {"mu": mu, "regime": regime, "mu_dry": mu_dry, "mu_film": mu_film,
            "film_thickness_m": h_film, "lambda_ratio": lam, "film_area_fraction": a_film}

# ---------------------------------------------------------------------------
# 4D sweep
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeMapInputs:
    temperatures_c: tuple[float, ...] = (-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, -2.0, -0.5)
    speeds_m_s: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    pressures_mpa: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    roughness_um: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)

@dataclass
class RegimeMapResult:
    temperatures_c: np.ndarray
    speeds_m_s: np.ndarray
    pressures_mpa: np.ndarray
    roughness_um: np.ndarray
    mu_grid: np.ndarray
    regime_grid: np.ndarray
    flat_data: list[dict[str, Any]]

def sweep_4d(inputs: RegimeMapInputs | None = None, **kwargs: Any) -> RegimeMapResult:
    if inputs is None:
        inputs = RegimeMapInputs()
    temps = np.array(inputs.temperatures_c)
    speeds = np.array(inputs.speeds_m_s)
    pressures = np.array(inputs.pressures_mpa)
    roughness = np.array(inputs.roughness_um)
    nt, nv, np_, nr = len(temps), len(speeds), len(pressures), len(roughness)
    mu_grid = np.zeros((nt, nv, np_, nr))
    regime_grid = np.zeros((nt, nv, np_, nr), dtype=int)
    regime_map_int = {REGIME_DRY: 0, REGIME_MIXED: 1, REGIME_FULL: 2}
    flat_data = []
    for i, t in enumerate(temps):
        for j, v in enumerate(speeds):
            for k, p in enumerate(pressures):
                for l, rq in enumerate(roughness):
                    res = friction_coefficient(t, v, p*1e6, rq*1e-6, **kwargs)
                    mu_grid[i,j,k,l] = res["mu"]
                    regime_grid[i,j,k,l] = regime_map_int[res["regime"]]
                    flat_data.append({"temperature_c": t, "speed_m_s": v, "pressure_mpa": p,
                                      "roughness_um": rq, "mu": res["mu"], "regime": res["regime"],
                                      "mu_dry": res["mu_dry"], "mu_film": res["mu_film"],
                                      "film_thickness_m": res["film_thickness_m"],
                                      "lambda_ratio": res["lambda_ratio"],
                                      "film_area_fraction": res["film_area_fraction"]})
    return RegimeMapResult(temps, speeds, pressures, roughness, mu_grid, regime_grid, flat_data)

# ---------------------------------------------------------------------------
# Response surface
# ---------------------------------------------------------------------------
def fit_response_surface(sweep: RegimeMapResult) -> dict[str, Any]:
    """Fit with expanded feature set including quadratic terms."""
    data = sweep.flat_data
    n = len(data)
    # 20 features: main effects + interactions + quadratics
    X = np.ones((n, 20))
    y = np.zeros(n)
    for idx, d in enumerate(data):
        t_star = abs(d["temperature_c"]) + 1.0
        v = max(d["speed_m_s"], 1e-8)
        p = max(d["pressure_mpa"], 1e-4)
        rq = max(d["roughness_um"], 1e-4)
        mu = max(d["mu"], 1e-6)
        lt, lv, lp, lr = math.log(t_star), math.log(v), math.log(p), math.log(rq)
        X[idx] = [1, lt, lv, lp, lr,
                  lt*lv, lt*lp, lt*lr, lv*lp, lv*lr, lp*lr,
                  lt*lv*lp, lt*lv*lr, lt*lp*lr, lv*lp*lr,
                  lt**2, lv**2, lp**2, lr**2,
                  lt*lv*lp*lr]
        y[idx] = math.log(mu)
    coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return {"coefficients": coeffs.tolist(), "r_squared": float(r_squared),
            "rmse_log": float(math.sqrt(ss_res/n)), "rank": int(rank)}

def evaluate_surface(coeffs: Sequence[float], temperature_c: float,
                     speed_m_s: float, pressure_mpa: float, roughness_um: float) -> float:
    t_star = abs(temperature_c) + 1.0
    v, p, rq = max(speed_m_s, 1e-8), max(pressure_mpa, 1e-4), max(roughness_um, 1e-4)
    lt, lv, lp, lr = math.log(t_star), math.log(v), math.log(p), math.log(rq)
    x = [1, lt, lv, lp, lr,
         lt*lv, lt*lp, lt*lr, lv*lp, lv*lr, lp*lr,
         lt*lv*lp, lt*lv*lr, lt*lp*lr, lv*lp*lr,
         lt**2, lv**2, lp**2, lr**2,
         lt*lv*lp*lr]
    return float(math.exp(sum(c*xi for c, xi in zip(coeffs, x))))

def surface_r_squared(sweep: RegimeMapResult, coeffs: Sequence[float]) -> float:
    data = sweep.flat_data
    y = np.array([math.log(max(d["mu"], 1e-6)) for d in data])
    y_pred = np.array([math.log(max(evaluate_surface(coeffs, d["temperature_c"], d["speed_m_s"],
                              d["pressure_mpa"], d["roughness_um"]), 1e-6)) for d in data])
    ss_res, ss_tot = np.sum((y-y_pred)**2), np.sum((y-np.mean(y))**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_kennedy(**kw) -> dict[str, Any]:
    rows = []
    for a in KENNEDY_1987_ANCHORS:
        mu_pred = float(dry_friction_coefficient(a.temperature_c, a.pressure_mpa*1e6, **kw))
        err = abs(mu_pred - a.mu) / a.mu
        rows.append({"temperature_c": a.temperature_c, "pressure_mpa": a.pressure_mpa,
                     "mu_kennedy": a.mu, "mu_predicted": mu_pred, "rel_error": err,
                     "within_20pct": err <= 0.20, "note": a.note})
    errors = [r["rel_error"] for r in rows]
    passed = sum(1 for r in rows if r["within_20pct"])
    return {"n_anchors": len(rows), "n_passed": passed, "max_rel_error": max(errors),
            "mean_rel_error": sum(errors)/len(errors), "all_within_20pct": passed==len(rows), "rows": rows}

def validate_colbeck(**kw) -> dict[str, Any]:
    rows = []
    for a in COLBECK_1988_ANCHORS:
        rq_m = 0.1e-6
        res = friction_coefficient(a.temperature_c, a.speed_m_s, a.pressure_mpa*1e6, rq_m, **kw)
        err = abs(res["mu"] - a.mu) / a.mu
        rows.append({"temperature_c": a.temperature_c, "speed_m_s": a.speed_m_s,
                     "pressure_mpa": a.pressure_mpa, "mu_colbeck": a.mu, "mu_predicted": res["mu"],
                     "rel_error": err, "within_30pct": err <= 0.30, "regime": res["regime"],
                     "lambda_ratio": res["lambda_ratio"], "note": a.note})
    errors = [r["rel_error"] for r in rows]
    passed = sum(1 for r in rows if r["within_30pct"])
    return {"n_anchors": len(rows), "n_passed": passed, "max_rel_error": max(errors),
            "mean_rel_error": sum(errors)/len(errors), "all_within_30pct": passed==len(rows), "rows": rows}

def validate_makkonen(**kw) -> dict[str, Any]:
    rows = []
    for a in MAKKONEN_2012_ANCHORS:
        res = friction_coefficient(a.temperature_c, a.speed_m_s, a.pressure_mpa*1e6, a.roughness_um*1e-6, **kw)
        obs = {"dry": REGIME_DRY, "mixed": REGIME_MIXED, "full": REGIME_FULL}.get(a.observed_regime, a.observed_regime)
        correct = res["regime"] == obs
        rows.append({"temperature_c": a.temperature_c, "speed_m_s": a.speed_m_s,
                     "pressure_mpa": a.pressure_mpa, "roughness_um": a.roughness_um,
                     "observed_regime": obs, "predicted_regime": res["regime"],
                     "correct": correct, "lambda_ratio": res["lambda_ratio"], "note": a.note})
    n_correct = sum(1 for r in rows if r["correct"])
    accuracy = n_correct / len(rows)
    return {"n_anchors": len(rows), "n_correct": n_correct, "accuracy": accuracy,
            "misclassification_rate": 1.0-accuracy, "within_25pct_misclass": (1.0-accuracy)<=0.25, "rows": rows}

def full_validation(**kw) -> dict[str, Any]:
    kennedy = validate_kennedy(**kw)
    colbeck = validate_colbeck(**kw)
    makkonen = validate_makkonen(**kw)
    sweep = sweep_4d(RegimeMapInputs(), **kw)
    fit = fit_response_surface(sweep)
    return {"kennedy": kennedy, "colbeck": colbeck, "makkonen": makkonen,
            "sweep": {"n_points": len(sweep.flat_data), "n_temperatures": len(sweep.temperatures_c),
                      "n_speeds": len(sweep.speeds_m_s), "n_pressures": len(sweep.pressures_mpa),
                      "n_roughness": len(sweep.roughness_um)},
            "response_surface": fit, "provenance": REGIME_MAP_PROVENANCE}

def build_acceptance(**kw) -> dict[str, Any]:
    val = full_validation(**kw)
    criteria = {
        "a_kennedy_dry_friction": {"pass": val["kennedy"]["all_within_20pct"],
            "max_rel_error": val["kennedy"]["max_rel_error"], "n_passed": val["kennedy"]["n_passed"]},
        "b_colbeck_lubricated": {"pass": val["colbeck"]["all_within_30pct"],
            "max_rel_error": val["colbeck"]["max_rel_error"], "n_passed": val["colbeck"]["n_passed"]},
        "c_makkonen_regime_transitions": {"pass": val["makkonen"]["within_25pct_misclass"],
            "accuracy": val["makkonen"]["accuracy"], "n_correct": val["makkonen"]["n_correct"]},
        "d_response_surface_r_squared": {"pass": val["response_surface"]["r_squared"] > 0.85,
            "r_squared": val["response_surface"]["r_squared"]},
    }
    all_pass = all(c["pass"] for c in criteria.values())
    digest = hashlib.sha256(json.dumps({k:v for k,v in val.items() if k!="provenance"},
                                        sort_keys=True, default=str).encode()).hexdigest()
    return {"criteria": criteria, "all_pass": all_pass, "validation": val,
            "digest": digest, "n_sweep_points": val["sweep"]["n_points"]}

def write_acceptance(path: str | None = None, **kw) -> dict[str, Any]:
    from pathlib import Path
    p = Path(path or "artifacts/ice_friction_regime_map/acceptance.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = build_acceptance(**kw)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload

def verify_acceptance(path: str) -> dict[str, Any]:
    from pathlib import Path
    stored = json.loads(Path(path).read_text())
    recomputed = build_acceptance()
    return {"digest_matches": stored.get("digest") == recomputed["digest"],
            "all_pass": recomputed["all_pass"]}
