"""Blade-ice contact mechanics: normal crushing, tangential friction, rolling.

Three closed-form contact laws are used by the milling simulator, each with an
analytical limiting case that the test suite checks:

*Normal (crushing) law* -- the crush pressure saturates at the ice compressive
strength ``sigma_c`` after a short elastic/ductile transition, and the contact
area is the rounded-edge contact patch of half-width ``sqrt(2 R_e delta)``::

    p(delta) = sigma_c * (1 - exp(-delta / delta_sat))
    A_c(delta, L) = 2 sqrt(2 R_e delta) * L
    F_n      = p(delta) * A_c

``L`` is the edge length in contact (the radial strip width) and ``R_e`` the edge
rounding radius -- a geometric property of an ice-class blade edge, not a fitted
constant.  The ``sqrt(delta)`` growth is the documented Hertzian blade-edge
contact width; because the pressure plateau is reached almost immediately the
force behaves as ``sigma_c * 2 sqrt(2 R_e delta) * L``, i.e. it grows with the
square root of the penetration, which is what makes the entry ramp of the cut
milder than the cut-area growth.  This reproduces the two documented limits: Hertzian/elastic for
``delta << delta_sat`` (``p ~ sigma_c delta / delta_sat``, i.e. linear in
penetration with an effective stiffness ``sigma_c A / delta_sat``) and the
pressure-plateau crushing regime ``p -> sigma_c`` for ``delta >> delta_sat``.

*Tangential law* -- Coulomb friction with the ice-steel coefficient of friction
supplied by :class:`tensordem.ice_steel_friction.IceSteelFrictionModel`, i.e.
``mu`` is a function of sliding speed, temperature and contact pressure::

    F_t = mu(v_slip, T, p) * F_n,   F_t <= mu_s F_n

*Rolling law* -- a debris chip rolling on the blade face carries the viscous/EPSD
rolling-resistance torque ``M_r = min(mu_r F_n r_eff, k_r omega_rel)``; the
critical-incline-angle limit ``tan(theta_c) = mu_r`` of
:func:`tensordem.rolling_resistance.critical_incline_angle_deg` is used as the
verification anchor.

All laws are pure functions of their arguments; contact energy is bookkept by the
caller so that the simulator can audit crush/friction work against the drive work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from functools import lru_cache

from ..ice_steel_friction import IceSteelFrictionModel, fit_ice_steel_friction

__all__ = [
    "BladeIceContact",
    "calibrated_friction_model",
    "edge_contact_area",
    "normal_pressure",
    "normal_force",
    "tangential_force",
    "rolling_moment",
    "critical_rolling_angle_deg",
]

# Documented validity box of the ice-steel friction correlation (the fitted
# anchors span these ranges; inputs are clipped, never extrapolated).
_V_MIN, _V_MAX = 0.01, 5.0
_T_MIN, _T_MAX = -20.0, 0.0
_P_MIN_MPA, _P_MAX_MPA = 0.01, 10.0


@lru_cache(maxsize=1)
def calibrated_friction_model() -> IceSteelFrictionModel:
    """The repository's calibrated ``mu(v, T, p)`` ice-steel correlation (cached)."""
    return fit_ice_steel_friction()


def normal_pressure(penetration: np.ndarray | float, sigma_c: float,
                    delta_sat: float) -> np.ndarray | float:
    """Crush pressure ``p(delta)`` [Pa]; asymptote ``sigma_c``, linear for small ``delta``."""
    delta = np.asarray(penetration, dtype=float)
    if delta_sat <= 0.0:
        raise ValueError("delta_sat must be positive")
    p = sigma_c * (1.0 - np.exp(-np.maximum(delta, 0.0) / delta_sat))
    return float(p) if np.ndim(p) == 0 else p


def edge_contact_area(penetration: np.ndarray | float, edge_radius: float,
                      span: np.ndarray | float) -> np.ndarray | float:
    """Rounded-edge contact area ``2 sqrt(2 R_e delta) L`` [m^2]."""
    delta = np.maximum(np.asarray(penetration, dtype=float), 0.0)
    area = 2.0 * np.sqrt(2.0 * float(edge_radius) * delta) * np.asarray(span, dtype=float)
    return float(area) if np.ndim(area) == 0 else area


def normal_force(penetration: np.ndarray | float, span: np.ndarray | float,
                 sigma_c: float, delta_sat: float,
                 edge_radius: float) -> np.ndarray | float:
    """Normal (crushing) force ``F_n = p(delta) * A_c(delta, L)`` [N]."""
    return (normal_pressure(penetration, sigma_c, delta_sat)
            * edge_contact_area(penetration, edge_radius, span))


def tangential_force(normal_force_mag: np.ndarray | float, mu: np.ndarray | float,
                     mu_static: float | None = None) -> np.ndarray | float:
    """Coulomb tangential force ``F_t = min(mu, mu_s) F_n`` [N]."""
    mu_eff = np.asarray(mu, dtype=float)
    if mu_static is not None:
        mu_eff = np.minimum(mu_eff, float(mu_static))
    return mu_eff * np.asarray(normal_force_mag, dtype=float)


def rolling_moment(normal_force_mag: np.ndarray | float,
                   omega_rel: np.ndarray | float, r_eff: np.ndarray | float,
                   mu_r: float, viscous_gain: float | None = None) -> np.ndarray | float:
    """Rolling-resistance moment magnitude [N m].

    ``M_r = mu_r F_n r_eff`` saturated by a viscous cap ``viscous_gain * omega_rel``
    when ``viscous_gain`` is given (the cap keeps the moment bounded for the tiny
    chip inertia at the millisecond time step).
    """
    elastic = float(mu_r) * np.asarray(normal_force_mag, dtype=float) * np.asarray(r_eff, dtype=float)
    if viscous_gain is None:
        out = elastic
    else:
        cap = float(viscous_gain) * np.abs(np.asarray(omega_rel, dtype=float))
        out = np.minimum(elastic, cap)
    return out


def critical_rolling_angle_deg(mu_r: float) -> float:
    """Incline angle [deg] at which a chip starts to roll: ``tan(theta) = mu_r``."""
    return math.degrees(math.atan(float(mu_r)))


@dataclass(frozen=True)
class BladeIceContact:
    """Blade-ice contact parameter set.

    Parameters
    ----------
    crush_strength
        Ice compressive (crushing) strength ``sigma_c`` [Pa].
    delta_sat
        Penetration scale over which the crush pressure saturates [m].
    temperature_c
        Ice temperature used by the friction correlation [degC].
    mu_static
        Upper bound on the friction coefficient (static ceiling).
    rolling_mu
        Rolling-resistance coefficient of a chip on the blade face [-].
    rolling_viscous_gain
        Viscous cap gain for the rolling moment [N m s].
    friction_model
        ``mu(v, T, p)`` correlation; defaults to the fitted ice-steel law.
    """

    crush_strength: float = 1.0e6
    delta_sat: float = 2.0e-4
    edge_radius: float = 2.0e-3
    temperature_c: float = -2.0
    mu_static: float = 0.30
    rolling_mu: float = 0.02
    rolling_viscous_gain: float = 5.0e-4
    friction_model: IceSteelFrictionModel = field(default_factory=calibrated_friction_model)

    def __post_init__(self) -> None:
        if self.crush_strength <= 0.0:
            raise ValueError("crush_strength must be positive")
        if self.delta_sat <= 0.0:
            raise ValueError("delta_sat must be positive")
        if self.edge_radius <= 0.0:
            raise ValueError("edge_radius must be positive")
        if not 0.0 < self.rolling_mu < 0.5:
            raise ValueError("rolling_mu must lie in (0, 0.5)")
        if not _T_MIN <= self.temperature_c <= _T_MAX:
            raise ValueError(f"temperature_c must lie in [{_T_MIN}, {_T_MAX}]")

    # -- normal ------------------------------------------------------------
    def pressure(self, penetration: Any) -> Any:
        """Crush pressure [Pa] at ``penetration`` [m]."""
        return normal_pressure(penetration, self.crush_strength, self.delta_sat)

    def contact_area(self, penetration: Any, span: Any) -> Any:
        """Rounded-edge contact area [m^2] at ``penetration`` over edge length ``span``."""
        return edge_contact_area(penetration, self.edge_radius, span)

    def elastic_stiffness(self, span: Any, penetration: Any = None) -> Any:
        """Tangent stiffness ``dF_n/ddelta`` at ``penetration`` [N/m].

        Differentiates both factors of ``F_n = p(delta) 2 sqrt(2 R_e delta) L``:
        the pressure-ramp term and the edge-area growth term.  In the pressure
        plateau (``delta >> delta_sat``) it reduces to ``sigma_c L sqrt(2 R_e/delta)``.
        """
        delta = self.delta_sat if penetration is None else float(penetration)
        delta = max(delta, 1e-12)
        span_v = np.asarray(span, dtype=float)
        area_growth = span_v * math.sqrt(2.0 * self.edge_radius / delta)
        pressure_ramp = (self.crush_strength / self.delta_sat
                         * math.exp(-delta / self.delta_sat))
        area = edge_contact_area(delta, self.edge_radius, span_v)
        pressure = normal_pressure(delta, self.crush_strength, self.delta_sat)
        return pressure_ramp * area + pressure * area_growth

    def normal(self, span: Any, penetration: Any) -> Any:
        """Normal crushing force [N]: pressure plateau times rounded-edge area."""
        return normal_force(penetration, span, self.crush_strength, self.delta_sat,
                            self.edge_radius)

    # -- tangential --------------------------------------------------------
    def friction_coefficient(self, slip_speed: Any, pressure_pa: Any | None = None) -> Any:
        """``mu(v_slip, T, p)`` from the ice-steel correlation (inputs clipped to its box)."""
        v = np.clip(np.asarray(slip_speed, dtype=float), _V_MIN, _V_MAX)
        p_mpa = np.clip(np.asarray(self.crush_strength if pressure_pa is None else pressure_pa,
                                   dtype=float) / 1.0e6, _P_MIN_MPA, _P_MAX_MPA)
        temp = np.full_like(v, self.temperature_c, dtype=float)
        mu = np.asarray(self.friction_model.mu(v, temp, p_mpa), dtype=float)
        return float(mu) if mu.ndim == 0 else mu

    def tangential(self, normal_force_mag: Any, slip_speed: Any,
                   pressure_pa: Any | None = None) -> tuple[Any, Any]:
        """Tangential force [N] and the friction coefficient used."""
        mu = self.friction_coefficient(slip_speed, pressure_pa)
        return tangential_force(normal_force_mag, mu, self.mu_static), mu

    # -- rolling -----------------------------------------------------------
    def rolling(self, normal_force_mag: Any, omega_rel: Any, r_eff: Any) -> Any:
        """Rolling-resistance moment [N m] for a chip rolling on the blade face."""
        return rolling_moment(normal_force_mag, omega_rel, r_eff, self.rolling_mu,
                              self.rolling_viscous_gain)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready contact record."""
        mu_ref = float(np.asarray(self.friction_coefficient(1.0, self.crush_strength)))
        return {
            "crush_strength_MPa": self.crush_strength / 1.0e6,
            "edge_radius_mm": self.edge_radius * 1.0e3,
            "delta_sat_m": self.delta_sat,
            "temperature_c": self.temperature_c,
            "mu_at_1ms_and_sigma_c": mu_ref,
            "mu_static": self.mu_static,
            "rolling_mu": self.rolling_mu,
            "friction_form": self.friction_model.form,
        }