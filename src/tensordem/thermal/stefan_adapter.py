"""Adapter between static-bed effective conductivity and the Stefan melting front.

The discrete melting front lives in :mod:`tensordem.threefield.thermal`
(Ranz-Marshall convection + isothermal Stefan condition).  That model assumes
every ice fragment is bathed in water at the free-stream temperature.  Inside a
*static* granular ice bed the heat has to travel through the bed first, so the
interface advance is conduction limited and the effective bed conductivity
``k_eff`` (see :mod:`tensordem.thermal.packed_bed`) enters the Stefan condition

    X(t)^2 = 2 k_eff (T_w - T_m) t / (rho_bed L_bed)

with ``rho_bed L_bed = (1 - phi) rho_ice L_f + (1 - phi) rho_ice c_p (T_m - T_i)``
the latent + sensible heat per unit bed volume.  The full-depth melting time is

    t_melt = rho_bed L_bed H^2 / (2 k_eff (T_w - T_m)).

This module is deliberately the *only* place that touches
``tensordem.threefield.thermal``; if the upstream thermal API drifts, only this
adapter changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..threefield.thermal import ThermalProperties, analytic_ransom_marshall_time

__all__ = ["BedStefanState", "melt_time_from_conductivity", "melt_time_sensitivity"]


@dataclass(frozen=True)
class BedStefanState:
    """Geometry and thermal state of a static granular ice bed."""

    thickness_m: float = 0.10
    porosity: float = 0.40
    ice_initial_temperature_c: float = -2.0
    props: ThermalProperties = ThermalProperties()

    def __post_init__(self) -> None:
        if not (self.thickness_m > 0.0 and math.isfinite(self.thickness_m)):
            raise ValueError("thickness_m must be positive and finite")
        if not 0.0 < self.porosity < 1.0:
            raise ValueError("porosity must lie in (0, 1)")

    @property
    def solid_fraction(self) -> float:
        return 1.0 - self.porosity

    @property
    def bed_heat_density(self) -> float:
        """Latent + sensible heat per unit bed volume (J/m^3)."""
        sensible = 2100.0 * max(
            self.props.melt_temperature_c - self.ice_initial_temperature_c, 0.0
        )
        return self.solid_fraction * self.props.ice_density_kg_m3 * (
            self.props.latent_heat_j_kg + sensible
        )

    @property
    def driving_temperature_k(self) -> float:
        return self.props.water_temperature_c - self.props.melt_temperature_c


def melt_time_from_conductivity(k_eff: float, state: BedStefanState) -> float:
    """Conduction-limited Stefan melt time for the full bed depth (s)."""
    if not (isinstance(k_eff, (int, float)) and not isinstance(k_eff, bool)):
        raise TypeError("k_eff must be a real number")
    if not math.isfinite(k_eff) or k_eff <= 0.0:
        raise ValueError("k_eff must be positive and finite")
    if state.driving_temperature_k <= 0.0:
        raise ValueError("no melting without a driving temperature difference")
    return (
        state.bed_heat_density * state.thickness_m**2
        / (2.0 * k_eff * state.driving_temperature_k)
    )


def melt_time_sensitivity(
    k_eff_values: list[float] | tuple[float, ...], state: BedStefanState
) -> dict[str, float | list[float]]:
    """``t_melt(k_eff)`` plus the numerical derivative ``dt_melt/dk_eff``.

    The derivative is evaluated with a central difference on a log-spaced grid
    so that the monotonicity statement ``dt_melt/dk_eff < 0`` is a computed
    quantity rather than an assertion.
    """
    values = [float(k) for k in k_eff_values]
    if len(values) < 3:
        raise ValueError("need at least three k_eff samples")
    times = [melt_time_from_conductivity(k, state) for k in values]
    log_k = [math.log(k) for k in values]
    derivative: list[float] = []
    for i in range(len(values)):
        if i == 0:
            d = (times[1] - times[0]) / (log_k[1] - log_k[0])
        elif i == len(values) - 1:
            d = (times[-1] - times[-2]) / (log_k[-1] - log_k[-2])
        else:
            d = (times[i + 1] - times[i - 1]) / (log_k[i + 1] - log_k[i - 1])
        derivative.append(d / values[i])  # d t / d k
    return {
        "k_eff": values,
        "t_melt_s": times,
        "dt_melt_dk": derivative,
        "monotonic_decreasing": all(
            times[i + 1] < times[i] for i in range(len(times) - 1)
        ),
        "max_dt_melt_dk": max(derivative),
    }


def convective_reference_melt_time(
    fragment_diameter_m: float, state: BedStefanState, *, relative_velocity_m_s: float = 0.5
) -> float:
    """Melt time of an equal-size fragment under free-stream convection.

    Provided as the "original" conclusion that has to be re-checked once
    conduction through the static bed is accounted for.
    """
    return analytic_ransom_marshall_time(
        fragment_diameter_m,
        fragment_diameter_m * 1e-9,
        relative_velocity_m_s,
        state.props,
    )