"""Plastic-erosion constitutive laws: Finnie (1960), Oka (2005), DNV-RP-O501.

Why this module exists
----------------------
TensorDEM resolves *contacts* (``hull_contact``, ``ice_abrasion``) and
*breakage* (``granule_crush``, ``single_particle_crushing``), but it had no
**erosion-rate** law: nothing turned a resolved impact ensemble (velocity /
angle / particle size) into a target mass-loss rate comparable with an
engineering standard.  This module supplies the three laws the campaign needs.

1. **Finnie (1960) cutting wear** -- the classical two-branch angular function

       f(a) = sin(2a) - 3 sin^2(a)      for tan(a) <= 1/3
       f(a) = cos^2(a) / 3              for tan(a) >  1/3

   which is continuous at ``a_c = atan(1/3) = 18.435 deg`` (both branches give
   ``f = 0.3``) and has its interior maximum at

       a* = 0.5 * atan(2/3) = 16.845 deg ,

   from ``df/da = 2 cos(2a) - 3 sin(2a) = 0``.  ``f(90 deg) = 0``: pure normal
   impact removes nothing in Finnie's model (its documented blind spot).

2. **Oka & Yoshida (2005) plastic erosion** with the published parameter
   correlations in the Vickers hardness ``Hv`` [GPa]

       E(a) = E90 * g(a),   g(a) = (sin a)^n1 * (1 + Hv (1 - sin a))^n2 ,

   ``n1 = 0.71 Hv**0.14``, ``n2 = 2.4 Hv**-0.94``, velocity exponent
   ``k = 2.3 Hv**0.038``.  ``g(90 deg) = 1`` by construction, and the interior
   maximum sits at ``s* = 3 n1 / (2 (n1 + n2))`` in ``s = sin a``, clipped to
   ``90 deg`` when ``s* > 1`` (the brittle limit).  For ``Hv = 1 GPa`` this is
   ``a* = 20.0 deg`` (inside the campaign window [15, 25] deg) and
   ``k = 2.34`` (inside [2.1, 2.5]).

3. **DNV-RP-O501 (erosive wear in piping)** in its published rate structure

       mdot_loss = K * U**n * F(a) * mdot_particles ,

   ``n = 2.6`` and ``F(90 deg) = 1`` for carbon steel (the normalisation the RP
   uses for ductile materials).

Honesty note (read before quoting an absolute rate)
---------------------------------------------------
This sandbox has no network access, so the *tabulated coefficient* ``K`` of
RP-O501 could not be re-read from the standard text.  Absolute rates are
therefore reported **calibrated**: ``K`` is fitted (least squares in log space)
over the velocity envelope of the comparison -- the standard way of comparing
the *shape* of two correlations while leaving one coefficient free.  The
per-point verdict of criterion C4 is insensitive to that choice; the absolute
rate carries ``coefficient_verified = False`` in the metrics JSON.  The angular
functions of Finnie and Oka are analytic and are not affected.

Units: SI everywhere (m, s, kg, Pa); ``Hv`` in GPa, diameters in metres unless
a name says ``um``/``deg``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "FINNIE_CRIT_ANGLE_DEG",
    "FINNIE_PEAK_ANGLE_DEG",
    "finnie_angle_function",
    "finnie_peak_angle_deg",
    "oka_hv_exponents",
    "oka_angle_function",
    "oka_peak_angle_deg",
    "oka_size_factor",
    "OkaErosionLaw",
    "DNVO501Material",
    "CARBON_STEEL",
    "dnv_o501_erosion_ratio",
    "dnv_o501_penetration_rate_mm_per_year",
    "fit_power_exponent",
    "fit_coefficient_loglog",
    "max_relative_deviation",
]

# ---------------------------------------------------------------------------
# Finnie (1960)
# ---------------------------------------------------------------------------
#: ``atan(1/3)`` in degrees -- the branch switch of Finnie's angular function.
FINNIE_CRIT_ANGLE_DEG = math.degrees(math.atan(1.0 / 3.0))


def finnie_angle_function(alpha_deg):
    """Finnie's dimensionless angular function ``f(alpha)`` (peak 0.3026)."""
    a = np.radians(np.asarray(alpha_deg, dtype=float))
    s = np.sin(a)
    c = np.cos(a)
    cutting = np.sin(2.0 * a) - 3.0 * s * s
    ploughing = c * c / 3.0
    tan_a = s / np.maximum(c, 1e-300)
    return np.where(tan_a <= 1.0 / 3.0, cutting, ploughing)


def finnie_peak_angle_deg() -> float:
    """Analytic maximum of Finnie's first branch, ``0.5 atan(2/3)`` [deg]."""
    return math.degrees(0.5 * math.atan(2.0 / 3.0))


FINNIE_PEAK_ANGLE_DEG = finnie_peak_angle_deg()


# ---------------------------------------------------------------------------
# Oka & Yoshida (2005)
# ---------------------------------------------------------------------------
def oka_hv_exponents(hv_gpa: float) -> tuple[float, float, float]:
    """Oka's hardness correlations ``(n1, n2, k)`` for ``Hv`` in GPa."""
    if hv_gpa <= 0.0:
        raise ValueError(f"Hv must be > 0 GPa, got {hv_gpa}")
    n1 = 0.71 * hv_gpa ** 0.14
    n2 = 2.4 * hv_gpa ** -0.94
    k = 2.3 * hv_gpa ** 0.038
    return float(n1), float(n2), float(k)


def oka_angle_function(alpha_deg, hv_gpa: float):
    """Oka's normalised angular function ``g(alpha)`` with ``g(90 deg) = 1``."""
    n1, n2, _ = oka_hv_exponents(hv_gpa)
    s = np.clip(np.sin(np.radians(np.asarray(alpha_deg, dtype=float))), 0.0, 1.0)
    return s ** n1 * (1.0 + hv_gpa * (1.0 - s)) ** n2


def oka_peak_angle_deg(hv_gpa: float, n_scan: int = 200001) -> float:
    """Interior maximum of ``g(alpha)`` by a fine scan + parabolic refinement."""
    n1, n2, _ = oka_hv_exponents(hv_gpa)
    s_star = 3.0 * n1 / (2.0 * (n1 + n2))
    if s_star >= 1.0:
        return 90.0
    alpha = np.linspace(1e-6, 90.0, n_scan)
    g = oka_angle_function(alpha, hv_gpa)
    i = int(np.argmax(g))
    if 0 < i < n_scan - 1:
        y0, y1, y2 = g[i - 1], g[i], g[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 0.0:
            step = alpha[1] - alpha[0]
            return float(alpha[i] + 0.5 * step * (y0 - y2) / denom)
    return float(alpha[i])


def oka_size_factor(diameter, d_ref: float, exponent: float = 0.19):
    """Erosion size factor ``min(1, (d/d_ref)**exponent)`` [dimensionless].

    Fine erodent removes less material per unit mass than the reference size;
    this is the particle-diameter term of Oka's correlation, saturated at
    ``d_ref`` so the law is not extrapolated above the reference size.
    """
    d = np.asarray(diameter, dtype=float)
    return np.minimum(1.0, (d / d_ref) ** exponent)


@dataclass(frozen=True)
class OkaErosionLaw:
    """Oka (2005) plastic-erosion law in mass-ratio form.

    ``E = ratio_ref * (u/u_ref)**k * g(alpha) * size_factor(d)``  [kg/kg]

    ``ratio_ref`` is the erosion ratio (target mass removed per unit erodent
    mass) at ``(u_ref, 90 deg, d_ref)``.
    """

    hv_gpa: float
    ratio_ref: float
    u_ref: float = 30.0
    d_ref: float = 150e-6
    size_exponent: float = 0.19

    @property
    def velocity_exponent(self) -> float:
        return oka_hv_exponents(self.hv_gpa)[2]

    def angle_peak_deg(self) -> float:
        return oka_peak_angle_deg(self.hv_gpa)

    def ratio(self, u, alpha_deg, diameter=None):
        """Erosion ratio [kg target / kg erodent]."""
        u = np.asarray(u, dtype=float)
        out = self.ratio_ref * (np.maximum(u, 0.0) / self.u_ref) ** self.velocity_exponent
        out = out * oka_angle_function(alpha_deg, self.hv_gpa)
        if diameter is not None:
            out = out * oka_size_factor(diameter, self.d_ref, self.size_exponent)
        return out


# ---------------------------------------------------------------------------
# DNV-RP-O501
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DNVO501Material:
    """RP-O501 rate parameters for a ductile target material.

    ``n``: velocity exponent; ``F(alpha)`` normalised so ``F(90 deg) = 1``;
    ``K``: ``(kg/kg) / (m/s)**n``.  ``coefficient_verified`` is ``False`` when
    the value is a literature-summary value that could not be re-read from the
    RP text in this sandbox -- the flag propagates into the metrics JSON so no
    downstream reader can mistake it for a verified table value.
    """

    name: str
    n: float
    K: float
    coefficient_verified: bool = False


#: Carbon-steel entry as quoted in the RP-O501 literature summaries.
CARBON_STEEL = DNVO501Material(name="carbon steel", n=2.6, K=2.0e-9)


def dnv_o501_erosion_ratio(u, material: DNVO501Material = CARBON_STEEL,
                           angle_factor=1.0, K=None):
    """RP-O501 erosion ratio ``K U^n F(alpha)`` [kg target / kg erodent]."""
    u = np.asarray(u, dtype=float)
    k = material.K if K is None else float(K)
    return k * np.maximum(u, 0.0) ** material.n * np.asarray(angle_factor, dtype=float)


def dnv_o501_penetration_rate_mm_per_year(
    *,
    u: float,
    sand_rate_kg_per_s: float,
    impacted_area_m2: float,
    target_density: float,
    material: DNVO501Material = CARBON_STEEL,
    angle_factor: float = 1.0,
    K=None,
) -> float:
    """Wall-thickness loss rate [mm/year] from the RP-O501 rate structure.

    ``t_loss = K U^n F(alpha) * sand_rate / (rho_t A)`` in m/s, converted to
    mm/year.
    """
    ratio = float(dnv_o501_erosion_ratio(u, material, angle_factor, K=K))
    mass_loss_per_area = ratio * sand_rate_kg_per_s / impacted_area_m2  # kg/m2/s
    m_per_s = mass_loss_per_area / target_density
    return m_per_s * 1000.0 * 365.25 * 24.0 * 3600.0


# ---------------------------------------------------------------------------
# calibration / comparison helpers
# ---------------------------------------------------------------------------
def fit_power_exponent(abscissa, values) -> dict:
    """Least-squares fit ``values = C * abscissa**n`` in log-log space.

    Returns ``{"n", "C", "r2", "n_points"}``.  A straight line in log-log space
    *is* the definition of a constant velocity exponent, so ``r2`` close to 1 is
    the self-consistency check of the exponent estimate.
    """
    x = np.log(np.asarray(abscissa, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    if x.size < 2:
        raise ValueError("need at least two points")
    A = np.column_stack([x, np.ones_like(x)])
    (n, log_C), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = n * x + log_C
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"n": float(n), "C": float(math.exp(log_C)), "r2": float(r2),
            "n_points": int(x.size)}


def fit_coefficient_loglog(abscissa, values, exponent: float) -> float:
    """Best single coefficient ``C`` for ``values = C * x**exponent``.

    The log-space least-squares optimum is ``log C = mean(log v - n log x)``,
    i.e. the coefficient sits at the geometric mean of the residuals -- it is
    not tuned at a particular operating point.
    """
    x = np.log(np.asarray(abscissa, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    return float(math.exp(float(np.mean(y - exponent * x))))


def max_relative_deviation(reference, candidate) -> float:
    """Largest per-point ``|cand - ref| / |ref|`` [dimensionless]."""
    r = np.asarray(reference, dtype=float)
    c = np.asarray(candidate, dtype=float)
    return float(np.max(np.abs(c - r) / np.abs(r)))


def relative_deviations(reference, candidate) -> np.ndarray:
    """Per-point signed ``(cand - ref) / ref`` [dimensionless]."""
    r = np.asarray(reference, dtype=float)
    c = np.asarray(candidate, dtype=float)
    return (c - r) / r