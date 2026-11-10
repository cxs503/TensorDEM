"""Solid-particle erosion models: Finnie, Oka, DNV-RP-O501, brittle scaling.

Physical picture
----------------
A particle stream of mass rate ``m_dot`` strikes a wall of density ``rho_t`` over
an exposed area ``A`` at mean impact angle ``alpha`` and speed ``v``.  All models
below answer the same question in two coordinates:

* the **erosion ratio** ``E`` -- mass of wall removed per mass of erodent
  (kg/kg, dimensionless).  This is the quantity a mass ledger can close.
* the **wall-loss rate** -- thickness per time (mm/year), which is what an
  inspection interval is written against.

Model structure
---------------
``Finnie (1960)`` -- ductile, derived from a single-particle mechanics argument.
The particle is arrested in the wall and the volume it cuts out is proportional
to the impact kinetic energy divided by the target's flow stress ``p``, with an
angular factor that peaks in the grazing regime:

    V/m_p = (v^2 / (4 p)) * f(alpha),
    f(alpha) = sin(2 alpha) - 3 sin^2(alpha)      for alpha <= atan(1/3)
    f(alpha) = cos^2(alpha) / 3                   for alpha >  atan(1/3)

Both branches meet at ``tan(alpha) = 1/3`` with the common value ``3/10``, and
the maximum ``f = 0.3027`` sits at ``alpha = atan(2/3)/2 = 16.845 deg`` -- the
analytic root of ``d/dalpha [sin 2a - 3 sin^2 a] = 0``.

``Oka et al. (2005)`` -- ductile, empirical:

    E = K Hv^k1 (v/v_ref)^k2 (d/d_ref)^k3 g(alpha),
    g(alpha) = sin(alpha)^n1 * (1 + Hv (1 - sin(alpha)))^n2,
    n1 = 0.71 Hv^0.14,  n2 = 2.4 Hv^-0.94            (Hv in GPa)

The bracketed factor is the hardness screening of the tangential crater: a hard
target deflects the erodent into a more normal impact, so the peak of ``g``
slides from ~29 deg (soft metal) towards ~64 deg (hard ceramic) as Hv rises.

``DNV-RP-O501`` -- engineering standard, ductile phase of the simplified model:

    E = K (v/v_ref)^n F(alpha),  F(alpha) = sin(alpha)^n

with ``K``, ``n`` tabulated per material/sand pair (``n = 2.6`` for steel/sand).
The standard's ``F`` peaks at normal incidence; this is a deliberate
conservative simplification of the ductile angular response and is *not* the
Finnie/Oka peak -- see the module report for the practical consequence.

``brittle scaling`` -- ``E ~ v^2.4 sin(alpha)^2.4`` (Evans et al. 1978), where
only the normal velocity component grows lateral cracks, so the maximum is at
``alpha = 90 deg``.

Every function accepts scalars or NumPy arrays, returns floats for scalar input,
and never mutates its arguments.  Units are SI unless a name says otherwise.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .materials import (
    BRITTLE_CONSTANTS,
    OKA_CONSTANTS,
    Material,
    dnv_coefficients,
    get_material,
)

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
"""Julian year in seconds (3.15576e7)."""

RAD_TO_DEG = 180.0 / math.pi

FINNIE_BRANCH_ANGLE_DEG = math.degrees(math.atan(1.0 / 3.0))
"""``atan(1/3)`` = 18.4349 deg: switch-over of the two Finnie branches."""

FINNIE_PEAK_ANGLE_DEG = 0.5 * math.degrees(math.atan(2.0 / 3.0))
"""``atan(2/3)/2`` = 16.8452 deg: analytic maximum of the cutting branch."""


def _scalar_or_array(value):
    """Return ``(array, was_scalar)`` for float/array input."""
    arr = np.asarray(value, dtype=float)
    return arr, arr.ndim == 0


def _out(arr: np.ndarray, was_scalar: bool):
    return float(arr) if was_scalar else arr


def _check_angle(alpha_deg) -> None:
    arr = np.asarray(alpha_deg, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 90.0):
        raise ValueError("impact angle must be finite and within [0, 90] deg")


# ---------------------------------------------------------------------------
# Finnie (1960) -- ductile cutting
# ---------------------------------------------------------------------------
def finnie_angle_function(alpha_deg):
    """Finnie's angular factor ``f(alpha)`` (dimensionless, max 0.3027).

    ``f(0) = 0`` (pure tangential impact removes nothing in this model) and
    ``f(90) = 0`` (a normal impact cannot cut); the maximum is at
    ``alpha = 16.845 deg``, the classic ductile erosion peak.
    """
    _check_angle(alpha_deg)
    a = np.radians(np.asarray(alpha_deg, dtype=float))
    grazing = np.sin(2.0 * a) - 3.0 * np.sin(a) ** 2
    normal = np.cos(a) ** 2 / 3.0
    out = np.where(a <= math.atan(1.0 / 3.0), grazing, normal)
    return _out(np.maximum(out, 0.0), np.asarray(alpha_deg, dtype=float).ndim == 0)


def finnie_peak_angle_deg() -> float:
    """Analytic location of the ``f(alpha)`` maximum (16.8452 deg)."""
    return FINNIE_PEAK_ANGLE_DEG


def finnie_peak_angle_deg_numeric(n: int = 90001) -> float:
    """Grid-refined maximum of :func:`finnie_angle_function` (cross-check)."""
    grid = np.linspace(0.0, 90.0, n)
    values = finnie_angle_function(grid)
    return float(grid[int(np.argmax(values))])


def finnie_peak_value() -> float:
    """Analytic maximum of ``f``: ``3 sqrt(3) / (2 sqrt(...))`` evaluated exactly."""
    a = math.atan(2.0 / 3.0) / 2.0
    return math.sin(2.0 * a) - 3.0 * math.sin(a) ** 2


def finnie_erosion_ratio(
    v_m_s: float,
    alpha_deg,
    material: Material | str,
    restitution: float = 0.0,
) -> float:
    """Mass of wall removed per mass of erodent (kg/kg) -- Finnie (1960).

    ``V/m_p = (v^2 / (4 p)) f(alpha)`` with ``p`` the target flow stress, so the
    erosion ratio is ``rho_t V / m_p``.  ``restitution`` discounts the energy
    that leaves with the rebounding particle (``1 - e^2``), which matters for
    hard erodent on a hard wall; the classical formula corresponds to ``e = 0``.
    """
    if v_m_s < 0.0 or not math.isfinite(v_m_s):
        raise ValueError("impact speed must be finite and non-negative")
    if not 0.0 <= restitution <= 1.0:
        raise ValueError("restitution must be within [0, 1]")
    mat = get_material(material) if isinstance(material, str) else material
    f = float(finnie_angle_function(alpha_deg))
    energy_factor = 1.0 - restitution**2
    return mat.density_kg_m3 * v_m_s**2 * f * energy_factor / (4.0 * mat.flow_stress_pa)


# ---------------------------------------------------------------------------
# Oka et al. (2005) -- empirical ductile law with hardness screening
# ---------------------------------------------------------------------------
def oka_exponents(hv_gpa: float) -> tuple[float, float]:
    """``(n1, n2)`` of Oka's angle function for a target hardness in GPa."""
    if not hv_gpa > 0.0:
        raise ValueError("hardness must be positive")
    return 0.71 * hv_gpa**0.14, 2.4 * hv_gpa ** (-0.94)


def oka_angle_function(alpha_deg, hv_gpa: float):
    """Oka's angular factor ``g(alpha)`` normalised to ``g(90) = 1``."""
    _check_angle(alpha_deg)
    n1, n2 = oka_exponents(hv_gpa)
    s = np.sin(np.radians(np.asarray(alpha_deg, dtype=float)))
    value = s**n1 * (1.0 + hv_gpa * (1.0 - s)) ** n2
    was_scalar = np.asarray(alpha_deg, dtype=float).ndim == 0
    return _out(np.asarray(value, dtype=float), was_scalar)


def oka_peak_angle_deg(hv_gpa: float) -> float:
    """Analytic argmax of ``g``: ``sin(alpha*) = n1(1+Hv) / (Hv (n1+n2))``.

    Returns 90 deg when the root exceeds unity (hard targets, where the
    screening factor dominates and the peak saturates at normal incidence).
    """
    n1, n2 = oka_exponents(hv_gpa)
    s = n1 * (1.0 + hv_gpa) / (hv_gpa * (n1 + n2))
    if s >= 1.0:
        return 90.0
    return math.degrees(math.asin(s))


def oka_peak_angle_deg_numeric(hv_gpa: float, n: int = 90001) -> float:
    grid = np.linspace(0.0, 90.0, n)
    return float(grid[int(np.argmax(oka_angle_function(grid, hv_gpa)))])


def oka_relative_erosion(
    v_m_s: float,
    alpha_deg,
    d_um: float,
    material: Material | str,
    k: float | None = None,
) -> float:
    """Oka's erosion ``E`` in the model's published units (mm^3 of wall / g erodent).

    .. warning::
       The leading constant ``K`` of Oka et al. is quoted as 65 in the open
       literature with a unit convention that cannot be re-derived offline (see
       :data:`tensordem.erosion.materials.OKA_CONSTANTS`).  Consequently this
       function is used for the **scaling law** -- the exponents ``k1, k2, k3``
       and the angle function -- and not for absolute rate prediction.  Pass an
       explicit ``k`` when an application has its own calibration.
    """
    mat = get_material(material) if isinstance(material, str) else material
    kk = OKA_CONSTANTS["K"] if k is None else float(k)
    if v_m_s < 0.0 or not math.isfinite(v_m_s):
        raise ValueError("impact speed must be finite and non-negative")
    if not d_um > 0.0:
        raise ValueError("particle diameter must be positive")
    hv = mat.hardness_hv_gpa
    return float(
        kk
        * hv ** OKA_CONSTANTS["k1"]
        * (v_m_s / OKA_CONSTANTS["v_ref_m_s"]) ** OKA_CONSTANTS["k2"]
        * (d_um / OKA_CONSTANTS["d_ref_um"]) ** OKA_CONSTANTS["k3"]
        * float(oka_angle_function(alpha_deg, hv))
    )


# ---------------------------------------------------------------------------
# DNV-RP-O501 -- ductile simplified model
# ---------------------------------------------------------------------------
def dnv_angle_function(alpha_deg, n: float):
    """``F(alpha) = sin(alpha)^n`` of DNV-RP-O501 (ductile phase)."""
    _check_angle(alpha_deg)
    s = np.sin(np.radians(np.asarray(alpha_deg, dtype=float)))
    return _out(np.asarray(s**n, dtype=float), np.asarray(alpha_deg, dtype=float).ndim == 0)


def dnv_ductile_erosion_ratio(
    v_m_s: float,
    alpha_deg,
    material: Material | str,
    k: float | None = None,
    n: float | None = None,
) -> float:
    """DNV-RP-O501 ductile erosion ratio ``E = K (v/v_ref)^n F(alpha)`` in kg/kg."""
    mat = get_material(material) if isinstance(material, str) else material
    coeff = dnv_coefficients(mat.name)
    kk = coeff.k if k is None else float(k)
    nn = coeff.n if n is None else float(n)
    if v_m_s < 0.0 or not math.isfinite(v_m_s):
        raise ValueError("impact speed must be finite and non-negative")
    return float(kk * v_m_s**nn * float(dnv_angle_function(alpha_deg, nn)))


def dnv_ductile_wall_loss_rate(
    v_m_s: float,
    alpha_deg,
    material: Material | str,
    sand_rate_kg_s: float,
    area_m2: float,
    k: float | None = None,
    n: float | None = None,
    wall_thickness_mm: float | None = None,
) -> dict:
    """Wall loss rate from the DNV-RP-O501 ductile model.

    Returns a dict with the erosion ratio (kg/kg), the mass-loss rate (kg/s),
    the thickness-loss rate (mm/year) and, when a wall thickness is given, the
    time to reach it (years).  ``loss_rate = E * sand_rate / (rho_t A)``.
    """
    if not sand_rate_kg_s >= 0.0:
        raise ValueError("sand rate must be non-negative")
    if not area_m2 > 0.0:
        raise ValueError("area must be positive")
    mat = get_material(material) if isinstance(material, str) else material
    ratio = dnv_ductile_erosion_ratio(v_m_s, alpha_deg, mat, k=k, n=n)
    mass_rate = ratio * sand_rate_kg_s
    thickness_rate_m_s = mass_rate / (mat.density_kg_m3 * area_m2)
    mm_per_year = thickness_rate_m_s * SECONDS_PER_YEAR * 1000.0
    out = {
        "erosion_ratio_kg_per_kg": ratio,
        "mass_loss_rate_kg_s": mass_rate,
        "wall_loss_mm_per_year": mm_per_year,
        "material": mat.name,
        "impact_speed_m_s": float(v_m_s),
        "impact_angle_deg": float(np.asarray(alpha_deg, dtype=float)),
    }
    if wall_thickness_mm is not None:
        if not wall_thickness_mm > 0.0:
            raise ValueError("wall_thickness_mm must be positive")
        out["life_years_to_thickness"] = (
            math.inf if mm_per_year <= 0.0 else wall_thickness_mm / mm_per_year
        )
    return out


# ---------------------------------------------------------------------------
# Brittle scaling
# ---------------------------------------------------------------------------
def brittle_angle_function(alpha_deg, n_angle: float | None = None):
    """``sin(alpha)^n`` brittleness factor (maximum at normal incidence)."""
    _check_angle(alpha_deg)
    n = BRITTLE_CONSTANTS["n_angle"] if n_angle is None else float(n_angle)
    s = np.sin(np.radians(np.asarray(alpha_deg, dtype=float)))
    return _out(np.asarray(s**n, dtype=float), np.asarray(alpha_deg, dtype=float).ndim == 0)


def brittle_relative_erosion(
    v_m_s: float,
    alpha_deg: float,
    v_ref_m_s: float = 1.0,
    k: float = 1.0,
    n_velocity: float | None = None,
    n_angle: float | None = None,
) -> float:
    """Brittle-regime erosion ``E = k (v/v_ref)^n_v sin(alpha)^n_a`` (relative units)."""
    if v_m_s < 0.0 or not math.isfinite(v_m_s):
        raise ValueError("impact speed must be finite and non-negative")
    nv = BRITTLE_CONSTANTS["n_velocity"] if n_velocity is None else float(n_velocity)
    na = BRITTLE_CONSTANTS["n_angle"] if n_angle is None else float(n_angle)
    return float(k * (v_m_s / v_ref_m_s) ** nv * float(brittle_angle_function(alpha_deg, na)))


def brittle_peak_angle_deg_numeric(n: int = 90001, n_angle: float | None = None) -> float:
    grid = np.linspace(0.0, 90.0, n)
    return float(grid[int(np.argmax(brittle_angle_function(grid, n_angle)))])


# ---------------------------------------------------------------------------
# Velocity-exponent extraction (criterion C2)
# ---------------------------------------------------------------------------
def fit_velocity_exponent(
    speeds_m_s: Iterable[float],
    erosion_rates: Iterable[float],
) -> tuple[float, float]:
    """Least-squares slope of ``ln E`` against ``ln v`` -> ``(n, rms_residual)``.

    All samples must be strictly positive and at least two distinct speeds must
    be present, otherwise the fit is meaningless and ``ValueError`` is raised.
    """
    v = np.asarray(list(speeds_m_s), dtype=float)
    e = np.asarray(list(erosion_rates), dtype=float)
    if v.shape != e.shape:
        raise ValueError("speeds and erosion rates must have the same length")
    if v.size < 3:
        raise ValueError("at least three samples are needed for a stable slope")
    if np.any(v <= 0.0) or np.any(e <= 0.0):
        raise ValueError("all speeds and erosion rates must be strictly positive")
    if np.unique(v).size < 2:
        raise ValueError("at least two distinct speeds are needed")
    slope, intercept = np.polyfit(np.log(v), np.log(e), 1)
    residual = np.log(e) - (slope * np.log(v) + intercept)
    return float(slope), float(np.sqrt(np.mean(residual**2)))


def emergent_velocity_exponent(
    model,
    speeds_m_s: Iterable[float],
    **kwargs,
) -> float:
    """Fit the velocity exponent of any model callable ``model(v, **kwargs)``."""
    speeds = list(speeds_m_s)
    values = [model(v, **kwargs) for v in speeds]
    slope, _ = fit_velocity_exponent(speeds, values)
    return slope


__all__ = [
    "FINNIE_BRANCH_ANGLE_DEG",
    "FINNIE_PEAK_ANGLE_DEG",
    "SECONDS_PER_YEAR",
    "brittle_angle_function",
    "brittle_peak_angle_deg_numeric",
    "brittle_relative_erosion",
    "dnv_angle_function",
    "dnv_ductile_erosion_ratio",
    "dnv_ductile_wall_loss_rate",
    "emergent_velocity_exponent",
    "finnie_angle_function",
    "finnie_erosion_ratio",
    "finnie_peak_angle_deg",
    "finnie_peak_angle_deg_numeric",
    "finnie_peak_value",
    "fit_velocity_exponent",
    "oka_angle_function",
    "oka_exponents",
    "oka_peak_angle_deg",
    "oka_peak_angle_deg_numeric",
    "oka_relative_erosion",
]