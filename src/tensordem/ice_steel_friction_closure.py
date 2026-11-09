"""Coverage operator + regime-resolved physical closure for ice-steel friction.

Why this module exists
----------------------
:mod:`tensordem.ice_steel_friction` delivers a *fitted* coefficient
``mu(v, T, p)`` from a compiled anchor set.  A fit is only as good as the
region it is fitted in: the compiled set has a structural gap (no anchor
combines ``v > 0.1 m/s`` with ``p > 1 MPa``), and away from the anchors the
log-linear form simply continues, which produced a 37.6 % error against the
heating-dominated extrapolation probe.

This module adds the two things a coefficient table cannot provide:

1. :func:`coverage` -- a *coverage operator* that answers, for any query,
   how much of the surrounding operating cell is actually backed by calibration
   data (2-D area ratio in the ``(log v, log p)`` plane at the query
   temperature), the distance to the nearest calibration row, and an explicit
   ``interior`` / ``boundary`` / ``extrapolation`` label.  Nothing is silently
   extrapolated any more.
2. :func:`ice_steel_friction` -- a *physical* closure with three regimes built
   from the friction-power chain

   ``q = mu p v``  (frictional heat flux)
   -> Blok partition ``gamma q`` into the ice
   -> 1-D Stefan melting (with the transient conductive onset: the interface
      reaches the melting point only after ``t_m``)
   -> Reynolds squeeze-out of the melt film (steady thickness ``h_ss`` and the
      thinning time), i.e. a film *lifetime*
   -> load-bearing film area fraction ``A``
   -> regime:

      * **boundary lubrication** (``A < 0.05``): Coulomb plateau, ``mu = mu_c``
        -- the classical dry/boundary coefficient; ``pv = 0`` degenerates here
        *by construction*, which is the negative control of the campaign;
      * **mixed lubrication** (``0.05 <= A < 0.5``): load shared between
        asperities and film, ``mu = (1-A) mu_c + A mu_film``;
      * **melt lubrication limit** (``A >= 0.5``): the film carries most of the
        load and ``mu -> mu_film = eta v / (h p)``.

   The delivered coefficient is a *coverage-weighted blend* of the calibrated
   interpolant ``mu_c`` and the physical closure ``mu_phys``:
   ``mu = W mu_c + (1 - W) mu_phys`` with ``W = 1`` inside the calibration
   convex hull and ``W = exp(-(d/sigma_blend)^2)`` outside.  Inside the data the
   delivered model is therefore *exactly* the calibrated interpolant (A2 is
   satisfied by construction and the closure cannot degrade the fit); outside it
   is the physics, which is the only defensible way to leave the table.

Physical parameters (documented, *not* tuned on the extrapolation probes)
------------------------------------------------------------------------
====================================  ==========  ==================================
symbol                                value       meaning / provenance
====================================  ==========  ==================================
``ICE_K``, ``ICE_RHO``, ``ICE_CP``    2.2, 917,   ice thermal properties near the
                                      2050        melting point
``STEEL_K/RHO/CP``                    50, 7800,   hull structural steel
                                      470
``LATENT_HEAT``                       3.34e5      ice latent heat of fusion [J/kg]
``WATER_VISCOSITY``                   1.79e-3     liquid water at 0 degC [Pa s]
``SURFACE_ROUGHNESS_M``               1.0e-6      combined rms asperity height of
                                                  an ice-rubbed steel plate
                                                  (plausible band 0.5-2 um)
``SLIDE_DURATION_S``                  120.0       sustained hull-ice engagement of
                                                  one channel transit (band
                                                  60-300 s)
``CONTACT_RADIUS_M``                  2.0e-3      nominal contact-patch radius
                                                  (band 1-5 mm)
====================================  ==========  ==================================

The two entries that matter for the extrapolation corner are the roughness
scale and the engagement time; :func:`parameter_sensitivity` sweeps their
plausible bands so the gate is reported as a band, not as a single lucky point.
No extrapolation-probe value enters the module: the anchors are imported, the
probes are only *scored* by the caller.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from .ice_steel_friction import (
    CALIBRATION_ANCHORS,
    calibrated_envelope,
    FrictionAnchor,
    IceSteelFrictionModel,
    P_MAX_MPA,
    P_MIN_MPA,
    P_REF_MPA,
    T_MIN_C,
    V_MAX,
    V_MIN,
    fit_ice_steel_friction,
)
from .interface_thermal.friction_heat import blok_partition_coefficient
from .interface_thermal.lubrication import stefan_melt_rate

__all__ = [
    "ICE_K", "ICE_RHO", "ICE_CP", "LATENT_HEAT", "WATER_VISCOSITY",
    "STEEL_K", "STEEL_RHO", "STEEL_CP",
    "SURFACE_ROUGHNESS_M", "SLIDE_DURATION_S", "CONTACT_RADIUS_M",
    "BLEND_LENGTH", "COVERAGE_CELL_HALF_WIDTH", "COVERAGE_TEMPERATURE_WINDOW_C",
    "REGIME_BOUNDARY_MAX_A", "REGIME_MELT_MIN_A",
    "INTERIOR_GATE", "EXTRAPOLATION_GATE",
    "mu_coulomb_plateau",
    "CoverageResult",
    "ClosureResult",
    "calibration_law",
    "blok_partition_ice",
    "coverage",
    "coverage_grid",
    "hull_gap",
    "validity_gap",
    "friction_heat_flux",
    "melting_onset_time",
    "melt_film_thickness",
    "squeeze_out_time",
    "film_area_fraction",
    "film_shear_mu",
    "ice_steel_friction",
    "friction_grid",
    "regime_map",
    "parameter_sensitivity",
]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
ICE_K = 2.2                 # W/(m K)
ICE_RHO = 917.0             # kg/m3
ICE_CP = 2050.0             # J/(kg K)
STEEL_K = 50.0              # W/(m K)
STEEL_RHO = 7800.0          # kg/m3
STEEL_CP = 470.0            # J/(kg K)
LATENT_HEAT = 3.34e5        # J/kg
WATER_VISCOSITY = 1.79e-3   # Pa s at 0 degC

ICE_ALPHA = ICE_K / (ICE_RHO * ICE_CP)     # thermal diffusivity [m2/s]
ICE_EFFUSIVITY = math.sqrt(ICE_K * ICE_RHO * ICE_CP)

SURFACE_ROUGHNESS_M = 1.0e-6
SLIDE_DURATION_S = 120.0
CONTACT_RADIUS_M = 2.0e-3

#: Normalised length over which the coverage blend fades from the interpolant
#: to the physical closure (in units of the hull-normalised coordinates).
BLEND_LENGTH = 0.15
#: Half-width of the query cell used by the 2-D coverage area ratio, in units of
#: the normalised ``(log10 v, log10 p)`` spans of the calibration set.
COVERAGE_CELL_HALF_WIDTH = 0.10
#: Temperature slab used to project the calibration set onto the ``(v, p)``
#: plane for a query temperature.
COVERAGE_TEMPERATURE_WINDOW_C = 10.0

#: Regime thresholds on the load-bearing film area fraction ``A``.
REGIME_BOUNDARY_MAX_A = 0.05
REGIME_MELT_MIN_A = 0.50

#: Campaign accuracy gates (calibrated domain / extrapolation region).
INTERIOR_GATE = 0.05
EXTRAPOLATION_GATE = 0.15

#: Speed below which the coefficient is the static Coulomb plateau: the
#: frictional heat flux ``mu p v -> 0`` and melting cannot start.
V_PLATEAU = max(V_MIN, 1.0e-4)

REGIME_BOUNDARY = "boundary"
REGIME_MIXED = "mixed"
REGIME_MELT = "melt"
REGIMES = (REGIME_BOUNDARY, REGIME_MIXED, REGIME_MELT)


def _num(name: str, value: Any, *, positive: bool = False,
         nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


# ---------------------------------------------------------------------------
# 1. The calibrated interpolant (single source of truth)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def calibration_law() -> IceSteelFrictionModel:
    """The `loglinear+thermal` interpolant fitted on the calibration anchors."""
    return fit_ice_steel_friction(list(CALIBRATION_ANCHORS),
                                 form="loglinear+thermal")


def mu_coulomb_plateau(v: Any, temperature_c: Any, pressure_mpa: Any) -> np.ndarray:
    """Coulomb / boundary-lubrication value: the interpolant at the plateau speed.

    The speed of the query is clamped at :data:`V_PLATEAU`, so ``mu_c`` **does not
    depend on ``v``** below the plateau: that is the ``pv -> 0`` negative control
    (A4).  At ``v = 0`` there is no frictional heat, no melt film, and the
    tangential stress is the Coulomb value ``mu_c N``.
    """
    law = calibration_law()
    speed = np.maximum(np.asarray(v, dtype=float), V_PLATEAU)
    return np.asarray(law.mu(speed, temperature_c, pressure_mpa), dtype=float)


# ---------------------------------------------------------------------------
# 2. Coverage operator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoverageResult:
    """Auditable answer to "is this query backed by calibration data?".

    Attributes
    ----------
    area_ratio
        Fraction of the query cell (a square of half width
        :data:`COVERAGE_CELL_HALF_WIDTH` in normalised ``(log10 v, log10 p)``)
        that lies inside the convex hull of the calibration rows *at the query
        temperature*.  ``1`` = the whole cell is backed by data, ``0`` = none.
    mode
        ``"interior"`` (cell fully covered), ``"boundary"`` (partially covered)
        or ``"extrapolation"`` (cell not covered at all).
    nearest_distance
        Euclidean distance to the nearest calibration row in normalised
        ``(log10 v, log10 p, T)`` coordinates; the normalisation spans are
        reported in :attr:`normalisation` so the number is reproducible.
    nearest_anchor
        Label (``v, T, p``) of that row plus its source, for the audit trail.
    """

    area_ratio: float
    mode: str
    nearest_distance: float
    nearest_anchor: str
    nearest_source: str
    nearest_deltas: dict[str, float]
    inside_hull: bool
    hull_gap: float
    temperature_support: int
    slab_temperature_c: tuple[float, float]
    slab_half_width_c: float
    power_density_w_per_m2: float
    normalisation: dict[str, tuple[float, float]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _anchor_geometry() -> dict[str, Any]:
    """Normalised calibration geometry (spans, hulls, Delaunay)."""
    anchors = tuple(CALIBRATION_ANCHORS)
    log_v = np.array([math.log10(a.v_m_per_s) for a in anchors])
    log_p = np.array([math.log10(a.pressure_mpa) for a in anchors])
    temp = np.array([a.temperature_c for a in anchors])
    span_v = (float(log_v.min()), float(log_v.max()))
    span_p = (float(log_p.min()), float(log_p.max()))
    span_t = (float(temp.min()), float(temp.max()))

    def normalise(log_v_q, log_p_q, temp_q):
        x = (np.asarray(log_v_q, dtype=float) - span_v[0]) / (span_v[1] - span_v[0])
        y = (np.asarray(log_p_q, dtype=float) - span_p[0]) / (span_p[1] - span_p[0])
        z = (np.asarray(temp_q, dtype=float) - span_t[0]) / (span_t[1] - span_t[0])
        return x, y, z

    x, y, z = normalise(log_v, log_p, temp)
    points = np.column_stack([x, y, z])
    return {
        "points": points,
        "anchors": anchors,
        "span_v": span_v,
        "span_p": span_p,
        "span_t": span_t,
        "hull3d": ConvexHull(points),
        "delaunay3d": Delaunay(points),
        "temperatures": temp,
    }


def _slab_selection(axis_temperature_c: float,
                    window_c: float = COVERAGE_TEMPERATURE_WINDOW_C):
    """Temperature slab used to project the calibration set on the ``(v, p)`` plane.

    The nominal slab is ``+-window_c`` around the query; it is widened (doubling)
    until the rows inside it actually span a 2-D ``(log v, log p)`` region, so a
    query whose own temperature band only contains the temperature sweep (a
    degenerate line in that plane) is still judged against the nearest slab that
    can carry a 2-D area.  Returns ``(mask, width, hull_2d_or_None)``.
    """
    geometry = _anchor_geometry()
    temps = geometry["temperatures"]
    points = geometry["points"]
    width = float(window_c)
    mask = np.abs(temps - axis_temperature_c) <= width
    for _ in range(8):
        mask = np.abs(temps - axis_temperature_c) <= width
        if int(mask.sum()) >= 3:
            slab = points[mask][:, :2]
            if np.ptp(slab[:, 0]) > 0.0 and np.ptp(slab[:, 1]) > 0.0:
                try:
                    hull = ConvexHull(slab)
                except QhullError:
                    hull = None
                if hull is not None and hull.volume > 1.0e-12:
                    return mask, width, np.column_stack([slab[hull.vertices, 0],
                                                         slab[hull.vertices, 1]])
        width *= 2.0
    return mask, width, None


def _cell_hull(axis_temperature_c: float,
               window_c: float = COVERAGE_TEMPERATURE_WINDOW_C) -> np.ndarray | None:
    """Convex hull (normalised ``(x, y)``) of the rows in the temperature slab."""
    return _slab_selection(axis_temperature_c, window_c)[2]


def _clip_polygon(subject: np.ndarray, square: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of ``subject`` by a convex ``square``."""
    output = [tuple(point) for point in subject]

    def inside(point, a, b):
        # square is counter-clockwise: inside == left of edge a->b
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0.0

    def intersect(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-18:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    for index in range(len(square)):
        a = tuple(square[index])
        b = tuple(square[(index + 1) % len(square)])
        if not output:
            return np.empty((0, 2))
        current = output
        output = []
        for position in range(len(current)):
            p1 = current[position]
            p2 = current[(position + 1) % len(current)]
            p1_in = inside(p1, a, b)
            p2_in = inside(p2, a, b)
            if p1_in:
                output.append(p1)
                if not p2_in:
                    output.append(intersect(p1, p2, a, b))
            elif p2_in:
                output.append(intersect(p1, p2, a, b))
    return np.array(output) if output else np.empty((0, 2))


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def hull_gap(v: Any, temperature_c: Any, pressure_mpa: Any) -> float:
    """Normalised gap between the query and the calibration convex hull.

    ``0`` inside the hull; outside it the smallest positive distance to a hull
    facet (the standard distance-to-polytope lower bound, exact whenever the
    closest point lies on the facet rather than on an edge), measured in the
    normalised ``(log10 v, log10 p, T)`` coordinates.
    """
    geometry = _anchor_geometry()
    log_v = math.log10(max(_num("v", v, nonnegative=True), V_PLATEAU))
    log_p = math.log10(max(_num("pressure_mpa", pressure_mpa, positive=True), P_MIN_MPA))
    temp = _num("temperature_c", temperature_c)
    x = (log_v - geometry["span_v"][0]) / (geometry["span_v"][1] - geometry["span_v"][0])
    y = (log_p - geometry["span_p"][0]) / (geometry["span_p"][1] - geometry["span_p"][0])
    z = (temp - geometry["span_t"][0]) / (geometry["span_t"][1] - geometry["span_t"][0])
    point = np.array([x, y, z])
    delaunay = geometry["delaunay3d"]
    if delaunay.find_simplex(point) >= 0:
        return 0.0
    hull = geometry["hull3d"]
    equations = hull.equations
    normals = np.linalg.norm(equations[:, :3], axis=1)
    gaps = (equations[:, :3] @ point + equations[:, 3]) / normals
    positive = gaps[gaps > 0.0]
    return float(positive.min()) if positive.size else 0.0


def validity_gap(v: Any, temperature_c: Any, pressure_mpa: Any) -> dict[str, Any]:
    """Normalised gap between the query and the *validity envelope* of the data.

    The fitted law is additive in ``log v``, ``T`` and ``log p``, so the anchor
    set supports any combination of individually measured values (the sweeps are
    separable): a convex-hull test would reject the perfectly supported corner
    ``(0.5 m/s, -15 degC, 1 MPa)`` merely because no single row measured it.  What
    the data do **not** support is the *coupling* -- the friction power density
    ``p v`` beyond the largest anchor value (the campaign's structural gap) and
    the per-variable boxes.  The gap is the largest *relative* excess over those
    bounds (``0`` inside, ``1`` = one full envelope width outside).

    The delivered coefficient switches from the calibrated interpolant to the
    physical closure with this gap: ``W = exp(-(gap / BLEND_LENGTH)^2)``.
    """
    speed = _num("v", v, nonnegative=True)
    temperature = _num("temperature_c", temperature_c)
    pressure = _num("pressure_mpa", pressure_mpa, positive=True)
    envelope = calibrated_envelope()
    excess: dict[str, float] = {}
    for name, value, key in (("v", speed, "v_m_per_s"),
                             ("temperature", temperature, "temperature_c"),
                             ("pressure", pressure, "pressure_mpa")):
        low, high = envelope[key]
        width = max(high - low, 1e-12)
        excess[name] = max(0.0, (value - high) / width, (low - value) / width)
    power = pressure * 1.0e6 * speed
    pv_low, pv_high = envelope["power_density_w_per_m2"]
    excess["power_density"] = max(0.0,
                                  (power - pv_high) / max(pv_high, 1e-12),
                                  (pv_low - power) / max(pv_high, 1e-12))
    return {
        "gap": float(max(excess.values())),
        "excess": {name: float(value) for name, value in excess.items()},
        "power_density_w_per_m2": float(power),
        "power_density_envelope": [float(pv_low), float(pv_high)],
        "mode": "interior" if max(excess.values()) <= 0.0 else "extrapolation",
    }


def coverage(v: Any, temperature_c: Any, pressure_mpa: Any,
             half_width: float = COVERAGE_CELL_HALF_WIDTH,
             window_c: float = COVERAGE_TEMPERATURE_WINDOW_C) -> CoverageResult:
    """Coverage operator: 2-D area ratio + nearest-row distance + regime label.

    Parameters
    ----------
    v, temperature_c, pressure_mpa
        Query point (m/s, degC, MPa).
    half_width
        Half width of the query cell in normalised ``(log10 v, log10 p)`` units.
    window_c
        Temperature slab used for the 2-D projection.
    """
    speed = max(_num("v", v, nonnegative=True), V_PLATEAU)
    temperature = _num("temperature_c", temperature_c)
    pressure = _num("pressure_mpa", pressure_mpa, positive=True)
    half_width = _num("half_width", half_width, positive=True)
    if not (T_MIN_C <= temperature < 0.0):
        raise ValueError("temperature_c must be in [-60, 0) degC")
    if speed > V_MAX or speed < V_MIN:
        raise ValueError("v outside the model domain")
    if pressure > P_MAX_MPA or pressure < P_MIN_MPA:
        raise ValueError("pressure_mpa outside the model domain")

    geometry = _anchor_geometry()
    log_v = math.log10(speed)
    log_p = math.log10(pressure)
    x = (log_v - geometry["span_v"][0]) / (geometry["span_v"][1] - geometry["span_v"][0])
    y = (log_p - geometry["span_p"][0]) / (geometry["span_p"][1] - geometry["span_p"][0])

    # -- 2-D area ratio in the temperature slab -----------------------------
    slab_mask, slab_width, hull_2d = _slab_selection(temperature, window_c)
    square = np.array([[x - half_width, y - half_width],
                       [x + half_width, y - half_width],
                       [x + half_width, y + half_width],
                       [x - half_width, y + half_width]])
    cell_area = float((2.0 * half_width) ** 2)
    if hull_2d is None:
        area_ratio = 0.0
    else:
        area_ratio = _polygon_area(_clip_polygon(hull_2d, square)) / cell_area
    area_ratio = float(min(1.0, max(0.0, area_ratio)))

    # -- nearest calibration row (brute force: auditable, 17 rows) ----------
    points = geometry["points"]
    query = np.array([x, y,
                      (temperature - geometry["span_t"][0])
                      / (geometry["span_t"][1] - geometry["span_t"][0])])
    distances = np.linalg.norm(points - query, axis=1)
    nearest = int(np.argmin(distances))
    anchor: FrictionAnchor = geometry["anchors"][nearest]

    gap = hull_gap(speed, temperature, pressure)
    inside_hull = gap == 0.0
    if area_ratio >= 0.999:
        mode = "interior"
    elif area_ratio > 0.0:
        mode = "boundary"
    else:
        mode = "extrapolation"

    slab_lo, slab_hi = (float(geometry["temperatures"][slab_mask].min()),
                        float(geometry["temperatures"][slab_mask].max())) \
        if slab_mask.any() else (temperature, temperature)

    return CoverageResult(
        area_ratio=area_ratio,
        mode=mode,
        nearest_distance=float(distances[nearest]),
        nearest_anchor=anchor.label,
        nearest_source=anchor.source,
        nearest_deltas={
            "dv_m_per_s": speed - anchor.v_m_per_s,
            "dT_c": temperature - anchor.temperature_c,
            "dp_mpa": pressure - anchor.pressure_mpa,
        },
        inside_hull=bool(inside_hull),
        hull_gap=float(gap),
        temperature_support=int(slab_mask.sum()),
        slab_temperature_c=(slab_lo, slab_hi),
        slab_half_width_c=float(slab_width),
        power_density_w_per_m2=float(pressure * 1.0e6 * speed),
        normalisation={
            "log10_v": geometry["span_v"],
            "log10_p": geometry["span_p"],
            "temperature_c": geometry["span_t"],
        },
    )


def coverage_grid(v: Sequence[float], temperature_c: float,
                  pressure_mpa: Sequence[float],
                  **kwargs: Any) -> dict[str, Any]:
    """Vectorised coverage over a ``(p, v)`` grid at fixed temperature (figures)."""
    speed = np.atleast_1d(np.asarray(v, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure_mpa, dtype=float))
    area = np.zeros((pressure.size, speed.size))
    mode = np.empty((pressure.size, speed.size), dtype=object)
    gap = np.zeros_like(area)
    for i, p in enumerate(pressure):
        for j, speed_j in enumerate(speed):
            result = coverage(float(speed_j), temperature_c, float(p), **kwargs)
            area[i, j] = result.area_ratio
            mode[i, j] = result.mode
            gap[i, j] = result.hull_gap
    return {
        "v_m_per_s": speed,
        "pressure_mpa": pressure,
        "temperature_c": float(temperature_c),
        "area_ratio": area,
        "hull_gap": gap,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# 3. Physical closure: friction power -> Stefan melt -> Reynolds squeeze
# ---------------------------------------------------------------------------
def blok_partition_ice() -> dict[str, float]:
    """Blok heat partition for an ice/steel sliding pair (effusivity ratio)."""
    return blok_partition_coefficient(ICE_K, ICE_RHO, ICE_CP,
                                     STEEL_K, STEEL_RHO, STEEL_CP)


def friction_heat_flux(mu: Any, v: Any, pressure_mpa: Any) -> np.ndarray:
    """Frictional heat flux ``q = mu p v`` [W/m2] (nominal contact area)."""
    coefficient = np.asarray(mu, dtype=float)
    speed = np.asarray(v, dtype=float)
    pressure = np.asarray(pressure_mpa, dtype=float) * 1.0e6
    return coefficient * pressure * speed


def melting_onset_time(flux_ice_w_m2: Any, temperature_c: Any) -> np.ndarray:
    """Time for a constant flux to bring a semi-infinite ice surface to 0 degC.

    From the constant-flux surface temperature ``dT = 2 q sqrt(alpha t / pi)/k``
    (Carslaw & Jaeger, implemented in
    :func:`tensordem.interface_thermal.conduction.semi_infinite_surface_temperature`)
    solved for ``dT = -T``::

        t_m = pi k^2 T^2 / (4 alpha q^2)

    ``t_m -> inf`` as ``q -> 0``: no frictional heat, no melting, which is what
    makes ``pv = 0`` degenerate to the Coulomb plateau.
    """
    flux = np.asarray(flux_ice_w_m2, dtype=float)
    temperature = np.abs(np.asarray(temperature_c, dtype=float))
    safe_flux = np.where(flux > 0.0, flux, np.nan)
    time = (math.pi * ICE_K ** 2 * temperature ** 2) / (4.0 * ICE_ALPHA * safe_flux ** 2)
    return np.where(np.isfinite(time), time, np.inf)


def melt_film_thickness(flux_ice_w_m2: Any, temperature_c: Any, speed: Any,
                        engagement_s: float = SLIDE_DURATION_S,
                        pressure_mpa: Any | None = None,
                        radius_m: float = CONTACT_RADIUS_M) -> dict[str, Any]:
    """Melt-film thickness from the Stefan condition capped by Reynolds drainage.

    ``h_growth = u (t_e - t_m)`` with the Stefan melt velocity
    ``u = q_ice / (rho (L + c |T|))`` (the sensible term warms the melted ice to
    the melting point) and ``t_m`` the conductive onset time.  The film cannot
    grow beyond the quasi-steady Reynolds state of the squeeze flow

    ``dh/dt = -2 p h^3 / (3 eta R^2)``   ->   ``h_ss = (3 eta R^2 u / (2 p))^(1/3)``,

    reached when melt supply balances squeeze drainage.
    """
    flux = np.asarray(flux_ice_w_m2, dtype=float)
    temperature = np.abs(np.asarray(temperature_c, dtype=float))
    speed = np.maximum(np.asarray(speed, dtype=float), V_PLATEAU)
    engagement = _num("engagement_s", engagement_s, positive=True)
    radius = _num("radius_m", radius_m, positive=True)

    onset = melting_onset_time(flux, temperature)
    with np.errstate(invalid="ignore", divide="ignore"):
        active = flux > 0.0
        melt_time = np.where(active, np.maximum(engagement - onset, 0.0), 0.0)
        melt_time = np.where(np.isfinite(melt_time), melt_time, 0.0)
        energy_density = ICE_RHO * (LATENT_HEAT + ICE_CP * temperature)
        melt_rate = np.where(active, flux / energy_density, 0.0)
        growth = melt_rate * melt_time
        if pressure_mpa is None:
            steady = np.full_like(growth, np.inf)
        else:
            pressure = np.asarray(pressure_mpa, dtype=float) * 1.0e6
            steady = np.where(pressure > 0.0,
                              (3.0 * WATER_VISCOSITY * radius ** 2 * melt_rate
                               / (2.0 * pressure)) ** (1.0 / 3.0), np.inf)
        thickness = np.minimum(growth, np.where(np.isfinite(steady), steady, growth))
    return {
        "thickness_m": thickness,
        "growth_m": growth,
        "steady_m": steady,
        "melt_rate_m_per_s": melt_rate,
        "onset_time_s": onset,
        "melting_time_s": melt_time,
        "transit_time_s": 2.0 * radius / speed,
    }


def squeeze_out_time(thickness_m: Any, pressure_mpa: Any,
                     radius_m: float = CONTACT_RADIUS_M) -> np.ndarray:
    """Reynolds squeeze-out time of a film of thickness ``h`` down to zero.

    Integrating ``-dh/dt = 2 p h^3/(3 eta R^2)`` from ``h`` gives
    ``t = 3 eta R^2 / (4 p h^2)``; half the thickness needs one quarter of it.
    """
    thickness = np.asarray(thickness_m, dtype=float)
    pressure = np.asarray(pressure_mpa, dtype=float) * 1.0e6
    radius = _num("radius_m", radius_m, positive=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        time = 3.0 * WATER_VISCOSITY * radius ** 2 / (4.0 * pressure * thickness ** 2)
    return np.where((thickness > 0.0) & (pressure > 0.0), time, 0.0)


def film_area_fraction(thickness_m: Any, roughness_m: float = SURFACE_ROUGHNESS_M,
                       squeeze_time_s: Any | None = None,
                       transit_time_s: Any | None = None) -> np.ndarray:
    """Load-bearing area fraction of the melt film, in ``[0, 1]``.

    Geometric coverage of the asperity field: ``A = min(1, h / sigma)`` (a film
    thinner than the rms asperity height only wets the valleys).  If the squeeze
    lifetime is shorter than the contact transit, the film is drained before the
    contact moves on and the coverage is scaled by the survival fraction
    ``Lambda/(1+Lambda)`` with ``Lambda = t_squeeze / t_transit``.
    """
    thickness = np.asarray(thickness_m, dtype=float)
    roughness = _num("roughness_m", roughness_m, positive=True)
    coverage_area = np.clip(thickness / roughness, 0.0, 1.0)
    if squeeze_time_s is not None and transit_time_s is not None:
        squeeze = np.asarray(squeeze_time_s, dtype=float)
        transit = np.maximum(np.asarray(transit_time_s, dtype=float), 1e-12)
        lam = squeeze / transit
        coverage_area = coverage_area * (lam / (1.0 + lam))
    return np.clip(coverage_area, 0.0, 1.0)


def film_shear_mu(v: Any, pressure_mpa: Any, thickness_m: Any) -> np.ndarray:
    """Couette shear of the film expressed as a friction coefficient.

    ``mu_film = tau / p = (eta v / h) / p`` -- the lubricating limit: the shear
    stress the water layer can transmit divided by the normal pressure.
    """
    speed = np.maximum(np.asarray(v, dtype=float), V_PLATEAU)
    pressure = np.asarray(pressure_mpa, dtype=float) * 1.0e6
    thickness = np.asarray(thickness_m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = WATER_VISCOSITY * speed / (thickness * pressure)
    return np.where((thickness > 0.0) & (pressure > 0.0), mu, np.inf)


def _regime_of(area_fraction: float, mu_film: float, mu_c: float) -> str:
    if area_fraction < REGIME_BOUNDARY_MAX_A or not mu_film < mu_c:
        return REGIME_BOUNDARY
    if area_fraction >= REGIME_MELT_MIN_A:
        return REGIME_MELT
    return REGIME_MIXED


@dataclass(frozen=True)
class ClosureResult:
    """One evaluation of the regime closure, with every intermediate reported."""

    v_m_per_s: float
    temperature_c: float
    pressure_mpa: float
    mu: float
    mu_coulomb: float
    mu_physical: float
    mu_film: float
    regime: str
    film_area_fraction: float
    film_thickness_m: float
    film_steady_thickness_m: float
    melting_onset_s: float
    melting_time_s: float
    transit_time_s: float
    squeeze_time_s: float
    heat_flux_w_per_m2: float
    ice_heat_flux_w_per_m2: float
    blend_weight_interior: float
    validity: dict[str, Any]
    coverage: CoverageResult | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["coverage"] = (self.coverage.as_dict()
                               if self.coverage is not None else None)
        return payload


def ice_steel_friction(v: Any, temperature_c: Any, pressure_mpa: Any, *,
                       engagement_s: float = SLIDE_DURATION_S,
                       roughness_m: float = SURFACE_ROUGHNESS_M,
                       radius_m: float = CONTACT_RADIUS_M,
                       with_coverage: bool = True) -> ClosureResult | np.ndarray:
    """Regime-resolved ice-steel friction coefficient ``mu(p, v, T)``.

    The delivered coefficient is the coverage-weighted blend

    ``mu = W mu_c + (1 - W) mu_phys``,   ``W = exp(-(gap / BLEND_LENGTH)^2)``

    with ``gap`` the normalised distance to the calibration hull from
    :func:`hull_gap` (``W = 1`` inside: the delivered model *is* the calibrated
    interpolant there) and ``mu_phys`` the three-regime closure described in the
    module docstring.  Scalars in, :class:`ClosureResult` out; arrays in, array of
    coefficients out.
    """
    scalar = np.isscalar(v) and np.isscalar(temperature_c) and np.isscalar(pressure_mpa)
    speed = np.atleast_1d(np.asarray(v, dtype=float))
    temperature = np.atleast_1d(np.asarray(temperature_c, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure_mpa, dtype=float))
    speed, temperature, pressure = np.broadcast_arrays(speed, temperature, pressure)
    if np.any(~np.isfinite(speed)) or np.any(~np.isfinite(temperature)) \
            or np.any(~np.isfinite(pressure)):
        raise ValueError("inputs must be finite")
    if np.any(speed < 0.0) or np.any(speed > V_MAX):
        raise ValueError("v must be in [0, V_MAX]")
    if np.any(temperature < T_MIN_C) or np.any(temperature >= 0.0):
        raise ValueError("temperature_c must be in [-60, 0) degC")
    if np.any(pressure <= 0.0) or np.any(pressure > P_MAX_MPA):
        raise ValueError("pressure_mpa must be in (0, P_MAX_MPA]")

    plateau_speed = np.maximum(speed, V_PLATEAU)
    mu_c = np.asarray(calibration_law().mu(plateau_speed, temperature, pressure),
                      dtype=float)
    gamma_ice = blok_partition_ice()["partition_1"]
    flux = friction_heat_flux(mu_c, speed, pressure)
    flux_ice = gamma_ice * flux
    film = melt_film_thickness(flux_ice, temperature, plateau_speed,
                               engagement_s=engagement_s, pressure_mpa=pressure,
                               radius_m=radius_m)
    thickness = film["thickness_m"]
    squeeze = squeeze_out_time(thickness, pressure, radius_m=radius_m)
    area = film_area_fraction(thickness, roughness_m=roughness_m,
                              squeeze_time_s=squeeze,
                              transit_time_s=film["transit_time_s"])
    mu_film = film_shear_mu(plateau_speed, pressure, thickness)

    mu_physical = np.empty_like(mu_c)
    regimes: list[str] = []
    for index in np.ndindex(mu_c.shape):
        a = float(area[index])
        film_mu = float(min(mu_film[index], mu_c[index]))
        # a film cannot raise the friction above the dry/boundary value: if the
        # viscous shear exceeds the Coulomb capacity the film is sheared out and
        # the interface stays on the asperities.
        mu_physical[index] = mu_c[index] * (1.0 - a) + a * film_mu
        if a <= 0.0 and speed[index] <= V_PLATEAU:
            regimes.append(REGIME_BOUNDARY)
        else:
            regimes.append(_regime_of(a, float(mu_film[index]), float(mu_c[index])))

    validity_rows = [validity_gap(max(float(s_), 0.0), float(t_), float(p_))
                     for s_, t_, p_ in zip(speed.ravel(), temperature.ravel(),
                                           pressure.ravel())]
    gap = np.array([row["gap"] for row in validity_rows],
                   dtype=float).reshape(speed.shape)
    weight = np.exp(-np.square(gap / BLEND_LENGTH))
    weight = np.clip(weight, 0.0, 1.0)
    mu = weight * mu_c + (1.0 - weight) * mu_physical

    x = (0,)
    if scalar:
        cov = (coverage(float(speed[x]), float(temperature[x]), float(pressure[x]))
               if with_coverage else None)
        return ClosureResult(
            v_m_per_s=float(speed[x]), temperature_c=float(temperature[x]),
            pressure_mpa=float(pressure[x]), mu=float(mu[x]),
            mu_coulomb=float(mu_c[x]), mu_physical=float(mu_physical[x]),
            mu_film=float(mu_film[x]), regime=regimes[0],
            film_area_fraction=float(area[x]),
            film_thickness_m=float(thickness[x]),
            film_steady_thickness_m=float(np.atleast_1d(film["steady_m"])[x]),
            melting_onset_s=float(np.atleast_1d(film["onset_time_s"])[x]),
            melting_time_s=float(np.atleast_1d(film["melting_time_s"])[x]),
            transit_time_s=float(np.atleast_1d(film["transit_time_s"])[x]),
            squeeze_time_s=float(np.atleast_1d(squeeze)[x]),
            heat_flux_w_per_m2=float(flux[x]),
            ice_heat_flux_w_per_m2=float(flux_ice[x]),
            blend_weight_interior=float(np.atleast_1d(weight)[0]),
            validity=validity_rows[0],
            coverage=cov,
        )
    return mu


def friction_grid(v: Sequence[float], temperature_c: float,
                  pressure_mpa: Sequence[float],
                  **kwargs: Any) -> dict[str, Any]:
    """Evaluate the closure on a ``(p, v)`` grid (returns arrays of diagnostics)."""
    speed = np.atleast_1d(np.asarray(v, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure_mpa, dtype=float))
    shape = (pressure.size, speed.size)
    output = {key: np.zeros(shape) for key in
              ("mu", "mu_coulomb", "mu_physical", "mu_film", "film_area_fraction",
               "film_thickness_m", "heat_flux_w_per_m2", "blend_weight_interior")}
    regime = np.empty(shape, dtype=object)
    for i, p in enumerate(pressure):
        for j, s in enumerate(speed):
            result = ice_steel_friction(float(s), temperature_c, float(p),
                                        with_coverage=False, **kwargs)
            assert isinstance(result, ClosureResult)
            output["mu"][i, j] = result.mu
            output["mu_coulomb"][i, j] = result.mu_coulomb
            output["mu_physical"][i, j] = result.mu_physical
            output["mu_film"][i, j] = min(result.mu_film, 1.0e6)
            output["film_area_fraction"][i, j] = result.film_area_fraction
            output["film_thickness_m"][i, j] = result.film_thickness_m
            output["heat_flux_w_per_m2"][i, j] = result.heat_flux_w_per_m2
            output["blend_weight_interior"][i, j] = result.blend_weight_interior
            regime[i, j] = result.regime
    output.update({"v_m_per_s": speed, "pressure_mpa": pressure,
                   "temperature_c": float(temperature_c), "regime": regime})
    return output


def regime_map(v: Sequence[float], pressure_mpa: Sequence[float],
               temperatures_c: Sequence[float], **kwargs: Any) -> dict[str, Any]:
    """Regime classification over a ``(T, p, v)`` grid (one string per cell)."""
    speed = np.atleast_1d(np.asarray(v, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure_mpa, dtype=float))
    temperatures = np.atleast_1d(np.asarray(temperatures_c, dtype=float))
    result = np.empty((temperatures.size, pressure.size, speed.size), dtype=object)
    for i, t in enumerate(temperatures):
        for j, p in enumerate(pressure):
            for k, s in enumerate(speed):
                value = ice_steel_friction(float(s), float(t), float(p),
                                          with_coverage=False, **kwargs)
                assert isinstance(value, ClosureResult)
                result[i, j, k] = value.regime
    return {"v_m_per_s": speed, "pressure_mpa": pressure,
            "temperature_c": temperatures, "regime": result}


def parameter_sensitivity(anchors: Iterable[FrictionAnchor],
                          roughness_values: Sequence[float] = (0.5e-6, 1.0e-6, 2.0e-6),
                          engagement_values: Sequence[float] = (60.0, 120.0, 300.0),
                          ) -> list[dict[str, Any]]:
    """Relative error over a parameter band; no entry is chosen from the data.

    ``anchors`` are scored, never fitted: the band shows how much of the
    extrapolation gate is carried by the two physical parameters.
    """
    rows: list[dict[str, Any]] = []
    for roughness in roughness_values:
        for engagement in engagement_values:
            errors = []
            details = []
            for anchor in anchors:
                value = ice_steel_friction(anchor.v_m_per_s, anchor.temperature_c,
                                           anchor.pressure_mpa,
                                           roughness_m=roughness,
                                           engagement_s=engagement)
                assert isinstance(value, ClosureResult)
                rel = abs(value.mu - anchor.mu) / anchor.mu
                errors.append(rel)
                details.append({"label": anchor.label, "mu": anchor.mu,
                                "mu_model": value.mu, "rel_error": rel,
                                "regime": value.regime})
            rows.append({
                "roughness_m": float(roughness),
                "engagement_s": float(engagement),
                "max_rel_error": float(max(errors)),
                "rms_rel_error": float(np.sqrt(np.mean(np.square(errors)))),
                "points": details,
            })
    return rows