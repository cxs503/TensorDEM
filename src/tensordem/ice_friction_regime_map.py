"""Ice-steel friction regime map: mu(T, V, P, R_q) four-dimensional response surface.

Campaign node ``p381w1-N03``: systematic modelling of the four-way dependence
of ice-steel friction on temperature, sliding speed, contact pressure and
surface roughness, with regime classification (dry / mixed / full lubrication).

Physics chain
-------------
1. **Dry (boundary) friction** -- Kennedy 1987 regime: asperity contact with
   Coulomb shear, ``mu_dry = mu_ref * (p/p_ref)^(-alpha) * exp(-beta*(T-T_ref))``.
   Calibrated on Kennedy's dry-friction experiments (alpha=0.32, beta=0.024).
2. **Mixed lubrication** -- partial melt film: load shared between asperity
   contact and hydrodynamic film, ``mu = (1-A) mu_b + A mu_f`` with
   ``A = A(Lambda)`` the film area fraction.
3. **Full (hydrodynamic) lubrication** -- Colbeck 1988 regime: the interface
   is fully separated by a melt-water film, ``mu_f = eta_w(T) V / (h p)``.
4. **Regime transitions** -- Makkonen 2012 field database: the boundaries
   between regimes in (T, V, P, R_q) space are predicted by the Lambda-based
   criterion and validated against Makkonen's compiled transition points.

The response surface is built by a structured 4D parameter sweep and fitted
with a multi-linear model in log-space to achieve R^2 > 0.85.

Validation anchors
------------------
* Kennedy 1987: dry-friction mu vs temperature at low speed (criterion a: <20%)
* Colbeck 1988: meltwater-lubricated mu vs speed (criterion b: <30%)
* Makkonen 2012: regime transition boundaries in (T, V) space (criterion c: <25%)

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
    # regime labels
    "REGIME_DRY", "REGIME_MIXED", "REGIME_FULL",
    # physical constants
    "GAS_CONSTANT", "ICE_LATENT", "ICE_RHO", "WATER_VISC_0C",
    "CLAPEYRON_SLOPE_K_PER_PA",
    # Kennedy 1987 dry-friction anchors
    "KENNEDY_1987_ANCHORS", "KennedyAnchor",
    # Colbeck 1988 meltwater anchors
    "COLBECK_1988_ANCHORS", "ColbeckAnchor",
    # Makkonen 2012 regime transition anchors
    "MAKKONEN_2012_ANCHORS", "MakkonenAnchor",
    # core model
    "dry_friction_coefficient",
    "water_viscosity", "melt_film_thickness",
    "lambda_ratio", "film_area_fraction", "regime_label",
    "friction_coefficient",
    # 4D sweep and response surface
    "RegimeMapInputs", "RegimeMapResult",
    "sweep_4d", "fit_response_surface", "evaluate_surface", "surface_r_squared",
    # validation
    "validate_kennedy", "validate_colbeck", "validate_makkonen",
    "full_validation", "REGIME_MAP_PROVENANCE",
    # artifacts
    "build_acceptance", "write_acceptance", "verify_acceptance",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIME_DRY = "dry"
REGIME_MIXED = "mixed"
REGIME_FULL = "full_lubrication"

GAS_CONSTANT = 8.31446261815324
ICE_LATENT = 3.34e5          # J/kg
ICE_RHO = 917.0              # kg/m^3
WATER_VISC_0C = 1.792e-3     # Pa s at 0 degC
CLAPEYRON_SLOPE_K_PER_PA = -7.4e-8  # K/Pa (dT_m/dp for ice Ih)

_T0 = 273.15

# Calibrated dry-friction parameters (from Kennedy 1987 data)
MU_REF_DRY = 0.22       # at T_ref=-10C, p_ref=0.5 MPa
T_REF_C = -10.0
P_REF_MPA = 0.5
ALPHA_PRESSURE = 0.32   # mu ~ p^(-alpha)
BETA_TEMP = 0.024       # mu ~ exp(-beta * (T - T_ref))

# ---------------------------------------------------------------------------
# Literature anchors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KennedyAnchor:
    """Kennedy 1987 dry-friction measurement."""
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
    """Colbeck 1988 meltwater-lubricated friction measurement."""
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
    """Makkonen 2012 regime transition boundary point."""
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
    "kennedy_1987": "Kennedy, F.E. (1987). Thermal and thermomechanical effects "
                    "in dry sliding. Wear 100: 1-3, 199-220.",
    "colbeck_1988": "Colbeck, S.C. (1988). The kinetic friction of ice on smooth "
                    "surfaces. CRREL Report 88-17.",
    "makkonen_2012": "Makkonen, L. (2012). Ice friction: A review of experimental "
                     "data and theoretical models. Cold Regions Science and "
                     "Technology, 70, 1-12.",
    "model": "Power-law dry friction (mu ~ p^-alpha * exp(-beta*T)) + Stribeck "
             "load-sharing with Lambda-based regime classification.",
}


# ---------------------------------------------------------------------------
# Physical sub-models
# ---------------------------------------------------------------------------
def _num(name: str, value: Any, *, positive: bool = False,
         nonnegative: bool = False) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out}")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out}")
    return out


def water_viscosity(temperature_c: Any) -> np.ndarray:
    """Dynamic viscosity of water (VFT form), Pa s."""
    t = np.asarray(temperature_c, dtype=float)
    tk = t + _T0
    if np.any(tk <= 140.0):
        raise ValueError("temperature must be above -133 degC")
    return 2.414e-5 * np.power(10.0, 247.8 / (tk - 140.0))


def dry_friction_coefficient(temperature_c: Any, pressure_pa: Any, *,
                             mu_ref: float = MU_REF_DRY,
                             t_ref_c: float = T_REF_C,
                             p_ref_mpa: float = P_REF_MPA,
                             alpha: float = ALPHA_PRESSURE,
                             beta: float = BETA_TEMP) -> np.ndarray:
    """Dry (boundary) friction coefficient.

    ``mu = mu_ref * (p/p_ref)^(-alpha) * exp(-beta*(T - T_ref))``

    Calibrated on Kennedy 1987 data: alpha=0.32, beta=0.024.
    """
    t = np.asarray(temperature_c, dtype=float)
    p_mpa = np.asarray(pressure_pa, dtype=float) / 1.0e6
    if np.any(p_mpa <= 0.0):
        raise ValueError("pressure_pa must be > 0")

    mu = (mu_ref
          * np.power(p_mpa / p_ref_mpa, -alpha)
          * np.exp(-beta * (t - t_ref_c)))
    return np.clip(mu, 0.02, 0.50)


def melt_film_thickness(friction_flux_w_m2: float, pressure_pa: float,
                        speed_m_s: float, temperature_c: float,
                        contact_radius_m: float, viscosity_pa_s: float,
                        *, drainage_retention: float = 1.0) -> float:
    """Sustained melt-film thickness from friction-heating balance."""
    q = _num("friction_flux_w_m2", friction_flux_w_m2, nonnegative=True)
    p = _num("pressure_pa", pressure_pa, positive=True)
    v = _num("speed_m_s", speed_m_s, nonnegative=True)
    t = _num("temperature_c", temperature_c)
    r = _num("contact_radius_m", contact_radius_m, positive=True)
    eta = _num("viscosity_pa_s", viscosity_pa_s, positive=True)

    if v <= 1.0e-8 or q <= 0:
        return 1.0e-9

    # Melt generation rate
    v_melt = q / (ICE_RHO * ICE_LATENT)

    # Temperature factor
    temp_factor = math.exp((t + 30.0) / 12.0)

    # Hydrodynamic film thickness
    h_hydro = math.sqrt(eta * v * r / p)

    # Melt-supply film thickness
    h_melt = (3.0 * eta * r * r * v_melt * temp_factor / (2.0 * p * drainage_retention)) ** (1.0 / 3.0)

    h = max(h_hydro, h_melt, 1.0e-9)
    return min(h, 1.0e-4)


def lambda_ratio(film_thickness_m: float, combined_roughness_m: float) -> float:
    """Dimensionless film ratio ``Lambda = h / sigma``."""
    sigma = _num("combined_roughness_m", combined_roughness_m, positive=True)
    h = _num("film_thickness_m", film_thickness_m, nonnegative=True)
    return h / sigma


def film_area_fraction(lam: float) -> float:
    """Load fraction carried by the film: 0 below Lambda=0.3, 1 above Lambda=3."""
    return float(np.clip((lam - 0.3) / 2.7, 0.0, 1.0))


def regime_label(lam: float, temperature_c: float) -> str:
    """Regime label from Lambda."""
    if lam < 0.3:
        return REGIME_DRY
    elif lam < 3.0:
        return REGIME_MIXED
    else:
        return REGIME_FULL


# ---------------------------------------------------------------------------
# Core friction model: mu(T, V, P, R_q)
# ---------------------------------------------------------------------------
def friction_coefficient(temperature_c: float, speed_m_s: float,
                         pressure_pa: float, roughness_m: float,
                         *, contact_radius_m: float = 1.0e-3,
                         drainage_retention: float = 0.3,
                         mu_ref: float = MU_REF_DRY,
                         t_ref_c: float = T_REF_C,
                         p_ref_mpa: float = P_REF_MPA,
                         alpha: float = ALPHA_PRESSURE,
                         beta: float = BETA_TEMP) -> dict[str, Any]:
    """Compute friction coefficient and regime for one (T, V, P, R_q) point."""
    t = _num("temperature_c", temperature_c)
    v = _num("speed_m_s", speed_m_s, nonnegative=True)
    p = _num("pressure_pa", pressure_pa, positive=True)
    r_q = _num("roughness_m", roughness_m, nonnegative=True)

    mu_dry = float(dry_friction_coefficient(t, p, mu_ref=mu_ref, t_ref_c=t_ref_c,
                                             p_ref_mpa=p_ref_mpa, alpha=alpha, beta=beta))

    if v <= 1.0e-8:
        return {
            "mu": mu_dry, "regime": REGIME_DRY,
            "mu_dry": mu_dry, "mu_film": float("nan"),
            "film_thickness_m": 0.0, "lambda_ratio": 0.0,
            "film_area_fraction": 0.0,
        }

    eta = float(water_viscosity(t))
    q_fric = mu_dry * p * v
    h_film = melt_film_thickness(q_fric, p, v, t, contact_radius_m, eta,
                                 drainage_retention=drainage_retention)

    sigma = r_q * math.sqrt(2.0) if r_q > 0 else 1.0e-7
    lam = lambda_ratio(h_film, sigma)
    a_film = film_area_fraction(lam)

    mu_film = eta * v / (max(h_film, 1.0e-9) * p)
    mu_film = min(mu_film, 0.5)

    mu = (1.0 - a_film) * mu_dry + a_film * mu_film
    mu = max(mu, 0.002)

    regime = regime_label(lam, t)

    return {
        "mu": mu, "regime": regime,
        "mu_dry": mu_dry, "mu_film": mu_film,
        "film_thickness_m": h_film, "lambda_ratio": lam,
        "film_area_fraction": a_film,
    }


# ---------------------------------------------------------------------------
# 4D parameter sweep
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeMapInputs:
    """Parameter ranges for the 4D sweep."""
    temperatures_c: tuple[float, ...] = field(
        default=(-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, -2.0, -0.5))
    speeds_m_s: tuple[float, ...] = field(
        default=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0))
    pressures_mpa: tuple[float, ...] = field(
        default=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))
    roughness_um: tuple[float, ...] = field(
        default=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))


@dataclass
class RegimeMapResult:
    """Result of the 4D sweep."""
    temperatures_c: np.ndarray
    speeds_m_s: np.ndarray
    pressures_mpa: np.ndarray
    roughness_um: np.ndarray
    mu_grid: np.ndarray
    regime_grid: np.ndarray
    flat_data: list[dict[str, Any]]


def sweep_4d(inputs: RegimeMapInputs | None = None,
             **kwargs: Any) -> RegimeMapResult:
    """Execute the 4D parameter sweep T x V x P x R_q."""
    if inputs is None:
        inputs = RegimeMapInputs()

    temps = np.array(inputs.temperatures_c, dtype=float)
    speeds = np.array(inputs.speeds_m_s, dtype=float)
    pressures = np.array(inputs.pressures_mpa, dtype=float)
    roughness = np.array(inputs.roughness_um, dtype=float)

    nt, nv, np_, nr = len(temps), len(speeds), len(pressures), len(roughness)
    mu_grid = np.zeros((nt, nv, np_, nr), dtype=float)
    regime_grid = np.zeros((nt, nv, np_, nr), dtype=int)
    regime_map_int = {REGIME_DRY: 0, REGIME_MIXED: 1, REGIME_FULL: 2}

    flat_data: list[dict[str, Any]] = []

    for i, t in enumerate(temps):
        for j, v in enumerate(speeds):
            for k, p in enumerate(pressures):
                for l, rq in enumerate(roughness):
                    p_pa = p * 1.0e6
                    rq_m = rq * 1.0e-6
                    res = friction_coefficient(t, v, p_pa, rq_m, **kwargs)
                    mu_grid[i, j, k, l] = res["mu"]
                    regime_grid[i, j, k, l] = regime_map_int[res["regime"]]
                    flat_data.append({
                        "temperature_c": t, "speed_m_s": v,
                        "pressure_mpa": p, "roughness_um": rq,
                        "mu": res["mu"], "regime": res["regime"],
                        "mu_dry": res["mu_dry"], "mu_film": res["mu_film"],
                        "film_thickness_m": res["film_thickness_m"],
                        "lambda_ratio": res["lambda_ratio"],
                        "film_area_fraction": res["film_area_fraction"],
                    })

    return RegimeMapResult(
        temperatures_c=temps, speeds_m_s=speeds,
        pressures_mpa=pressures, roughness_um=roughness,
        mu_grid=mu_grid, regime_grid=regime_grid,
        flat_data=flat_data,
    )


# ---------------------------------------------------------------------------
# Response surface fitting
# ---------------------------------------------------------------------------
def fit_response_surface(sweep: RegimeMapResult) -> dict[str, Any]:
    """Fit multi-linear model in log-space."""
    data = sweep.flat_data
    n = len(data)

    X = np.ones((n, 8), dtype=float)
    y = np.zeros(n, dtype=float)

    for idx, d in enumerate(data):
        t_star = abs(d["temperature_c"]) + 1.0
        v = max(d["speed_m_s"], 1.0e-8)
        p = max(d["pressure_mpa"], 1.0e-4)
        rq = max(d["roughness_um"], 1.0e-4)
        mu = max(d["mu"], 1.0e-6)

        lt = math.log(t_star)
        lv = math.log(v)
        lp = math.log(p)
        lr = math.log(rq)

        X[idx] = [1, lt, lv, lp, lr, lt*lv, lt*lp, lv*lp]
        y[idx] = math.log(mu)

    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "coefficients": coeffs.tolist(),
        "feature_names": ["1", "log(T*)", "log(V)", "log(P)", "log(R_q)",
                          "log(T*)*log(V)", "log(T*)*log(P)", "log(V)*log(P)"],
        "r_squared": float(r_squared),
        "rank": int(rank),
        "residual_sum_sq": float(ss_res),
        "total_sum_sq": float(ss_tot),
        "rmse_log": float(math.sqrt(ss_res / n)),
    }


def evaluate_surface(coeffs: Sequence[float], temperature_c: float,
                     speed_m_s: float, pressure_mpa: float,
                     roughness_um: float) -> float:
    """Evaluate the fitted response surface at one point."""
    t_star = abs(temperature_c) + 1.0
    v = max(speed_m_s, 1.0e-8)
    p = max(pressure_mpa, 1.0e-4)
    rq = max(roughness_um, 1.0e-4)

    lt = math.log(t_star)
    lv = math.log(v)
    lp = math.log(p)
    lr = math.log(rq)

    x = [1, lt, lv, lp, lr, lt*lv, lt*lp, lv*lp]
    log_mu = sum(c * xi for c, xi in zip(coeffs, x))
    return float(math.exp(log_mu))


def surface_r_squared(sweep: RegimeMapResult, coeffs: Sequence[float]) -> float:
    """Compute R^2 of the fitted surface against the sweep data."""
    data = sweep.flat_data
    n = len(data)
    y = np.array([math.log(max(d["mu"], 1.0e-6)) for d in data])
    y_pred = np.array([
        math.log(max(evaluate_surface(coeffs, d["temperature_c"], d["speed_m_s"],
                                       d["pressure_mpa"], d["roughness_um"]), 1.0e-6))
        for d in data
    ])
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_kennedy(mu_ref: float = MU_REF_DRY, t_ref_c: float = T_REF_C,
                     p_ref_mpa: float = P_REF_MPA,
                     alpha: float = ALPHA_PRESSURE,
                     beta: float = BETA_TEMP) -> dict[str, Any]:
    """Validate dry-friction predictions against Kennedy 1987 anchors."""
    rows = []
    for a in KENNEDY_1987_ANCHORS:
        p_pa = a.pressure_mpa * 1.0e6
        mu_pred = float(dry_friction_coefficient(a.temperature_c, p_pa,
                                                  mu_ref=mu_ref, t_ref_c=t_ref_c,
                                                  p_ref_mpa=p_ref_mpa,
                                                  alpha=alpha, beta=beta))
        err = abs(mu_pred - a.mu) / a.mu
        rows.append({
            "temperature_c": a.temperature_c, "speed_m_s": a.speed_m_s,
            "pressure_mpa": a.pressure_mpa,
            "mu_kennedy": a.mu, "mu_unc": a.mu_unc,
            "mu_predicted": mu_pred,
            "rel_error": err, "within_20pct": err <= 0.20,
            "note": a.note,
        })

    errors = [r["rel_error"] for r in rows]
    passed = sum(1 for r in rows if r["within_20pct"])
    return {
        "n_anchors": len(rows), "n_passed": passed,
        "max_rel_error": max(errors), "mean_rel_error": sum(errors) / len(errors),
        "all_within_20pct": passed == len(rows),
        "rows": rows,
    }


def validate_colbeck(drainage_retention: float = 0.3,
                     contact_radius_m: float = 1.0e-3) -> dict[str, Any]:
    """Validate full-lubrication predictions against Colbeck 1988 anchors."""
    rows = []
    for a in COLBECK_1988_ANCHORS:
        p_pa = a.pressure_mpa * 1.0e6
        rq_m = 1.0e-6  # smooth surface
        res = friction_coefficient(a.temperature_c, a.speed_m_s, p_pa, rq_m,
                                    contact_radius_m=contact_radius_m,
                                    drainage_retention=drainage_retention)
        err = abs(res["mu"] - a.mu) / a.mu
        rows.append({
            "temperature_c": a.temperature_c, "speed_m_s": a.speed_m_s,
            "pressure_mpa": a.pressure_mpa,
            "mu_colbeck": a.mu, "mu_unc": a.mu_unc,
            "mu_predicted": res["mu"],
            "rel_error": err, "within_30pct": err <= 0.30,
            "regime": res["regime"], "lambda_ratio": res["lambda_ratio"],
            "note": a.note,
        })

    errors = [r["rel_error"] for r in rows]
    passed = sum(1 for r in rows if r["within_30pct"])
    return {
        "n_anchors": len(rows), "n_passed": passed,
        "max_rel_error": max(errors), "mean_rel_error": sum(errors) / len(errors),
        "all_within_30pct": passed == len(rows),
        "rows": rows,
    }


def validate_makkonen(drainage_retention: float = 0.3,
                      contact_radius_m: float = 1.0e-3) -> dict[str, Any]:
    """Validate regime transition boundaries against Makkonen 2012 anchors."""
    rows = []
    for a in MAKKONEN_2012_ANCHORS:
        p_pa = a.pressure_mpa * 1.0e6
        rq_m = a.roughness_um * 1.0e-6
        res = friction_coefficient(a.temperature_c, a.speed_m_s, p_pa, rq_m,
                                    contact_radius_m=contact_radius_m,
                                    drainage_retention=drainage_retention)
        correct = res["regime"] == a.observed_regime
        rows.append({
            "temperature_c": a.temperature_c, "speed_m_s": a.speed_m_s,
            "pressure_mpa": a.pressure_mpa, "roughness_um": a.roughness_um,
            "observed_regime": a.observed_regime,
            "predicted_regime": res["regime"],
            "correct": correct,
            "lambda_ratio": res["lambda_ratio"],
            "note": a.note,
        })

    n_correct = sum(1 for r in rows if r["correct"])
    n_total = len(rows)
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return {
        "n_anchors": n_total, "n_correct": n_correct,
        "accuracy": accuracy, "misclassification_rate": 1.0 - accuracy,
        "within_25pct_misclass": (1.0 - accuracy) <= 0.25,
        "rows": rows,
    }


def full_validation(mu_ref: float = MU_REF_DRY, t_ref_c: float = T_REF_C,
                    p_ref_mpa: float = P_REF_MPA,
                    alpha: float = ALPHA_PRESSURE,
                    beta: float = BETA_TEMP,
                    drainage_retention: float = 0.3,
                    contact_radius_m: float = 1.0e-3) -> dict[str, Any]:
    """Run all three validation suites."""
    kennedy = validate_kennedy(mu_ref=mu_ref, t_ref_c=t_ref_c, p_ref_mpa=p_ref_mpa,
                                alpha=alpha, beta=beta)
    colbeck = validate_colbeck(drainage_retention=drainage_retention,
                                contact_radius_m=contact_radius_m)
    makkonen = validate_makkonen(drainage_retention=drainage_retention,
                                  contact_radius_m=contact_radius_m)

    sweep = sweep_4d(RegimeMapInputs(),
                      contact_radius_m=contact_radius_m,
                      drainage_retention=drainage_retention,
                      mu_ref=mu_ref, t_ref_c=t_ref_c, p_ref_mpa=p_ref_mpa,
                      alpha=alpha, beta=beta)
    fit = fit_response_surface(sweep)

    return {
        "kennedy": kennedy,
        "colbeck": colbeck,
        "makkonen": makkonen,
        "sweep": {
            "n_points": len(sweep.flat_data),
            "n_temperatures": len(sweep.temperatures_c),
            "n_speeds": len(sweep.speeds_m_s),
            "n_pressures": len(sweep.pressures_mpa),
            "n_roughness": len(sweep.roughness_um),
        },
        "response_surface": fit,
        "provenance": REGIME_MAP_PROVENANCE,
    }


# ---------------------------------------------------------------------------
# Acceptance artifact
# ---------------------------------------------------------------------------
def build_acceptance(**kwargs: Any) -> dict[str, Any]:
    """Build the full acceptance payload."""
    val = full_validation(**kwargs)

    criteria = {
        "a_kennedy_dry_friction": {
            "pass": val["kennedy"]["all_within_20pct"],
            "max_rel_error": val["kennedy"]["max_rel_error"],
            "mean_rel_error": val["kennedy"]["mean_rel_error"],
            "n_anchors": val["kennedy"]["n_anchors"],
            "n_passed": val["kennedy"]["n_passed"],
        },
        "b_colbeck_lubricated": {
            "pass": val["colbeck"]["all_within_30pct"],
            "max_rel_error": val["colbeck"]["max_rel_error"],
            "mean_rel_error": val["colbeck"]["mean_rel_error"],
            "n_anchors": val["colbeck"]["n_anchors"],
            "n_passed": val["colbeck"]["n_passed"],
        },
        "c_makkonen_regime_transitions": {
            "pass": val["makkonen"]["within_25pct_misclass"],
            "accuracy": val["makkonen"]["accuracy"],
            "misclassification_rate": val["makkonen"]["misclassification_rate"],
            "n_anchors": val["makkonen"]["n_anchors"],
            "n_correct": val["makkonen"]["n_correct"],
        },
        "d_response_surface_r_squared": {
            "pass": val["response_surface"]["r_squared"] > 0.85,
            "r_squared": val["response_surface"]["r_squared"],
            "rmse_log": val["response_surface"]["rmse_log"],
        },
    }

    all_pass = all(c["pass"] for c in criteria.values())

    payload_for_digest = {k: v for k, v in val.items() if k != "provenance"}
    digest = hashlib.sha256(
        json.dumps(payload_for_digest, sort_keys=True, default=str).encode()
    ).hexdigest()

    return {
        "criteria": criteria,
        "all_pass": all_pass,
        "validation": val,
        "digest": digest,
        "n_sweep_points": val["sweep"]["n_points"],
    }


def write_acceptance(path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Write acceptance JSON."""
    from pathlib import Path
    if path is None:
        path = "artifacts/ice_friction_regime_map/acceptance.json"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = build_acceptance(**kwargs)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def verify_acceptance(path: str) -> dict[str, Any]:
    """Verify an acceptance artifact."""
    from pathlib import Path
    p = Path(path)
    stored = json.loads(p.read_text())
    recomputed = build_acceptance()
    digest_match = stored.get("digest") == recomputed["digest"]
    return {
        "digest_matches": digest_match,
        "stored_digest": stored.get("digest"),
        "recomputed_digest": recomputed["digest"],
        "all_pass": recomputed["all_pass"],
    }
