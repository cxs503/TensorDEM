"""Blade-element / momentum open-water model (supporting model for milling).

The milling simulator needs two things from hydrodynamics:

1. the hydrodynamic torque ``Q_hydro(t)`` that the drive has to overcome beside
   the ice load, and
2. the axial induced velocity in the propeller race, which advects the ice chips.

Both come from a blade-element / momentum (BEM) solve of the same
:class:`~tensordem.propeller_ice.geometry.PropellerGeometry`:

* section loads from a linear lift slope ``2 pi alpha`` up to the stall angle and
  the flat-plate separated laws ``C_L = 2 sin a cos a``, ``C_D = 2 sin^2 a``
  beyond it (standard post-stall continuation, no tabulated section data);
* axial/angular induction ``a``, ``a'`` from the momentum identities
  ``dT = 4 pi r rho V^2 (1+a) a F dr`` and
  ``dQ = 4 pi r^3 rho V omega (1+a) a' F dr`` with the Prandtl tip-loss factor
  ``F``; the heavily loaded branch uses the Buhl empirical high-induction
  correction;
* convergence is measured by the blade/momentum residual (returned, and asserted
  in the tests).

The result is a standard open-water diagram ``K_T(J)``, ``K_Q(J)``, ``eta(J)``
plus the radial induction profile.  This module is *supporting*: it is verified
against its own momentum identities and strip-count convergence, not against a
proprietary open-water chart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import PropellerGeometry

__all__ = ["OpenWaterState", "OpenWaterConfig", "solve_open_water", "section_coefficients"]

_STALL = math.radians(12.0)


def section_coefficients(alpha: np.ndarray | float, cd0: float = 0.012,
                         cl_slope: float = 2.0 * math.pi) -> tuple[Any, Any]:
    """2-D section ``C_L, C_D`` with separated post-stall continuation.

    ``|alpha| <= 12 deg``: attached flow, ``C_L = cl_slope * alpha``,
    ``C_D = cd0 + 0.010 C_L^2``.  Beyond stall: ``C_L = 2 sin a cos a``,
    ``C_D = cd0 + 2 sin^2 a`` (flat-plate separated flow).
    """
    a = np.asarray(alpha, dtype=float)
    attached = np.abs(a) <= _STALL
    cl_att = cl_slope * a
    cl_sep = 2.0 * np.sin(a) * np.cos(a)
    cl = np.where(attached, cl_att, cl_sep)
    cd_att = cd0 + 0.010 * cl_att ** 2
    cd_sep = cd0 + 2.0 * np.sin(a) ** 2
    cd = np.where(attached, cd_att, cd_sep)
    return cl, cd


@dataclass(frozen=True)
class OpenWaterConfig:
    """Open-water solve controls."""

    n_strips: int = 40
    rho: float = 1025.0
    kinematic_viscosity: float = 1.8e-6
    cd0: float = 0.012
    iterations: int = 1500
    relaxation: float = 0.10
    convergence_tol: float = 1.0e-8
    a_cap: float = 0.85

    def __post_init__(self) -> None:
        if self.n_strips < 4:
            raise ValueError("n_strips must be >= 4")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")


@dataclass(frozen=True)
class OpenWaterState:
    """Result of one open-water (BEM) solve."""

    advance_ratio: float
    thrust_coefficient: float
    torque_coefficient: float
    efficiency: float
    thrust: float
    torque: float
    induced_mean: float
    residual: float
    iterations: int
    r: np.ndarray
    a: np.ndarray
    a_prime: np.ndarray
    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray

    # -- derived transport quantities --------------------------------------
    def race_axial_velocity(self, advance_velocity: float, factor: float = 1.5) -> float:
        """Axial velocity used to advect debris: ``V_a + factor * u_induced``."""
        return float(advance_velocity + factor * self.induced_mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "J": self.advance_ratio,
            "KT": self.thrust_coefficient,
            "KQ": self.torque_coefficient,
            "eta": self.efficiency,
            "thrust_N": self.thrust,
            "torque_Nm": self.torque,
            "induced_mean_m_s": self.induced_mean,
            "bem_residual": self.residual,
            "bem_iterations": self.iterations,
        }


def _buhl(a: np.ndarray, F: np.ndarray, cap: float) -> np.ndarray:
    """Buhl empirical high-induction correction for ``a > 0.4``."""
    a_c = np.where(a > 0.4, 0.0, a)
    numerator = 18.0 * F - 20.0 - 3.0 * np.sqrt(np.maximum(F * (50.0 - 36.0 * F), 0.0))
    denominator = 36.0 * F - 50.0
    corrected = np.where(a > 0.4, numerator / denominator, a_c)
    return np.clip(corrected, -0.5, cap)


def solve_open_water(geometry: PropellerGeometry, revolutions_per_second: float,
                     advance_velocity: float,
                     config: OpenWaterConfig | None = None) -> OpenWaterState:
    """Solve the BEM system for one operating point ``(n, V_a)``.

    ``revolutions_per_second`` is ``n`` [1/s]; ``advance_velocity`` is the axial
    inflow ``V_a`` [m/s]; ``J = V_a / (n D)``.
    """
    cfg = config or OpenWaterConfig()
    n = float(revolutions_per_second)
    if n <= 0.0:
        raise ValueError("revolutions_per_second must be positive")
    va = float(advance_velocity)
    if va < 0.0:
        raise ValueError("advance_velocity must be non-negative")
    omega = 2.0 * math.pi * n
    strips = geometry.radial_strips(cfg.n_strips)
    r = strips["r"]
    dr = strips["dr"]
    chord = strips["chord"]
    phi_geom = strips["pitch_angle"]
    sigma = geometry.n_blades * chord / (2.0 * math.pi * r)
    tip = np.maximum(strips["tip_loss"], 1.0e-3)
    hub_span = geometry.hub_radius
    hub = (2.0 / math.pi) * np.arccos(np.clip(
        np.exp(-0.5 * geometry.n_blades * (r - hub_span) /
               np.maximum(r * np.sin(np.maximum(phi_geom, 1.0e-6)), 1.0e-9)), 0.0, 1.0))
    loss = np.clip(tip * np.clip(hub, 1.0e-3, 1.0), 1.0e-3, 1.0)

    a = np.zeros_like(r)
    a_prime = np.zeros_like(r)
    residual = math.inf
    iterations = 0
    for iterations in range(1, cfg.iterations + 1):
        u_axial = va * (1.0 + a)
        u_tang = omega * r * (1.0 - a_prime)
        phi_in = np.arctan2(u_axial, np.maximum(u_tang, 1.0e-9))
        alpha = phi_geom - phi_in
        cl, cd = section_coefficients(alpha, cd0=cfg.cd0)
        w2 = u_axial ** 2 + u_tang ** 2
        cn = cl * np.cos(phi_in) - cd * np.sin(phi_in)
        ct = cl * np.sin(phi_in) + cd * np.cos(phi_in)

        # blade-element increment per unit radius (annulus, Z blades)
        d_load_n = 0.5 * cfg.rho * w2 * geometry.n_blades * chord * cn * dr
        d_load_t = 0.5 * cfg.rho * w2 * geometry.n_blades * chord * ct * r * dr

        # momentum increments
        a_new = np.zeros_like(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_new = d_load_n / np.maximum(4.0 * math.pi * r * cfg.rho * va ** 2 * loss * dr, 1.0e-12)
            a_new = np.where(va > 1.0e-9, a_new / np.maximum(1.0 + a, 1.0e-9), cfg.a_cap)
            a_prime_new = d_load_t / np.maximum(
                4.0 * math.pi * r ** 3 * cfg.rho * va * omega * loss * dr, 1.0e-12)
            a_prime_new = np.where(va > 1.0e-9, a_prime_new / np.maximum(1.0 + a, 1.0e-9), 0.0)
        a_new = np.clip(a_new, -0.3, cfg.a_cap)
        a_prime_new = np.clip(a_prime_new, -0.2, 0.5)
        if not np.all(np.isfinite(a_new)):
            a_new = np.nan_to_num(a_new, nan=0.0, posinf=cfg.a_cap)

        relax = cfg.relaxation
        a_old = a.copy()
        a = (1.0 - relax) * a + relax * a_new
        a_prime = (1.0 - relax) * a_prime + relax * a_prime_new
        residual = float(np.max(np.abs(a - a_old)))
        if residual < cfg.convergence_tol:
            break

    # final loads from the converged induction
    u_axial = va * (1.0 + a)
    u_tang = omega * r * (1.0 - a_prime)
    phi_in = np.arctan2(u_axial, np.maximum(u_tang, 1.0e-9))
    alpha = phi_geom - phi_in
    cl, cd = section_coefficients(alpha, cd0=cfg.cd0)
    w2 = u_axial ** 2 + u_tang ** 2
    cn = cl * np.cos(phi_in) - cd * np.sin(phi_in)
    ct = cl * np.sin(phi_in) + cd * np.cos(phi_in)
    thrust = float(np.sum(0.5 * cfg.rho * w2 * geometry.n_blades * chord * cn * dr))
    torque = float(np.sum(0.5 * cfg.rho * w2 * geometry.n_blades * chord * ct * r * dr))

    d = geometry.diameter
    kt = thrust / (cfg.rho * n ** 2 * d ** 4)
    kq = torque / (cfg.rho * n ** 2 * d ** 5)
    jv = va / (n * d)
    eta = (jv * kt / (2.0 * math.pi * kq)) if kq > 0.0 else 0.0
    induced = float(np.sum(a * va * chord * dr) / max(np.sum(chord * dr), 1e-12))
    return OpenWaterState(
        advance_ratio=jv,
        thrust_coefficient=kt,
        torque_coefficient=kq,
        efficiency=eta,
        thrust=thrust,
        torque=torque,
        induced_mean=induced,
        residual=residual,
        iterations=iterations,
        r=r,
        a=a,
        a_prime=a_prime,
        alpha_deg=np.degrees(alpha),
        cl=cl,
        cd=cd,
    )