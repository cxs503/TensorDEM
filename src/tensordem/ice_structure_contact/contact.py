"""Ice-structure contact laws: Hertz normal elasticity + Coulomb friction + crushing cap.

The surface here is the *contact law* face of the ice-structure verification
surface used by :mod:`tensordem.ice_structure_contact`:

``CrushingContact``
    Hertzian elastic normal law ``F = k_H delta^(3/2)`` with
    ``k_H = (4/3) E* sqrt(R) t`` (``E* = E/(1-nu^2)``, ``t`` the out-of-plane
    slice thickness), **capped** by the ice indentation crushing pressure
    ``p_c`` acting on the projected contact width of the particle:

        ``F_cap(delta) = p_c * t * w(delta)``,  ``w = 2 sqrt(2 R delta - delta^2)``

    The cap is the classical ice-crushing limit state (crushing pressure
    ``sigma_c``, ISO 19906 style).  Work done *while the cap is active* is
    plastic work and is returned as ``plastic_work_increment`` so the energy
    ledger can book it as fracture/crushing energy instead of silently losing
    it.

``CoulombFriction``
    Regularised Coulomb law ``F_t = -mu F_n tanh(v_slip / v_eps)``.  The
    regularisation is only a numerical device near zero slip; for
    ``|v_slip| >> v_eps`` the law is exactly ``|F_t| = mu F_n`` and the
    dissipation rate is ``|F_t v_slip|``.

Both laws are written twice: a scalar reference implementation (easy to check
by hand, used by the unit tests) and vectorised numpy kernels (used by the
simulation).  They are required to agree to machine precision.

All quantities SI (m, Pa, N, N/m, m/s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CrushingContact",
    "CoulombFriction",
    "projected_contact_width",
]


def projected_contact_width(delta: float, radius: float) -> float:
    """Chord width ``w`` of a circle of radius ``R`` cut at penetration ``delta``.

    ``w = 2 sqrt(2 R delta - delta^2)`` for ``0 <= delta <= 2R``; clipped to
    ``[0, 2R]`` so over-driven penetrations cannot inflate the crushing area.
    """
    if delta <= 0.0:
        return 0.0
    d = min(delta, 2.0 * radius)
    value = 2.0 * radius * d - d * d
    if value <= 0.0:
        return 0.0
    return 2.0 * math.sqrt(value)


@dataclass(frozen=True)
class CrushingContact:
    """Hertz elastic normal contact with an ice crushing pressure cap."""

    ice_modulus_pa: float = 9.0e9
    poisson_ratio: float = 0.33
    crushing_pressure_pa: float = 1.5e6
    out_of_plane_thickness_m: float = 1.0

    @property
    def effective_modulus_pa(self) -> float:
        return self.ice_modulus_pa / (1.0 - self.poisson_ratio**2)

    def hertz_stiffness(self, radius: float) -> float:
        """``k_H`` in ``F = k_H delta^(3/2)`` for a rigid plane on a sphere."""
        return (
            4.0 / 3.0
            * self.effective_modulus_pa
            * math.sqrt(radius)
            * self.out_of_plane_thickness_m
        )

    def elastic_force(self, delta: float, radius: float) -> float:
        if delta <= 0.0:
            return 0.0
        return self.hertz_stiffness(radius) * delta**1.5

    def capacity(self, delta: float, radius: float) -> float:
        return self.crushing_pressure_pa * self.out_of_plane_thickness_m * projected_contact_width(
            delta, radius
        )

    def force(self, delta: float, radius: float) -> float:
        """Normal force: Hertz until the crushing cap, plastic above it."""
        return min(self.elastic_force(delta, radius), self.capacity(delta, radius))

    def is_yielding(self, delta: float, radius: float) -> bool:
        return self.elastic_force(delta, radius) >= self.capacity(delta, radius)


@dataclass(frozen=True)
class CoulombFriction:
    """Regularised Coulomb tangential law with explicit dissipation bookkeeping."""

    coefficient: float = 0.15
    regularization_speed_m_s: float = 1.0e-3

    def tangential_force(self, normal_force: float, slip_speed: float) -> float:
        """Tangential force on the ice particle along the contact tangent.

        Positive ``slip_speed`` means the particle slides forward relative to
        the hull, so the friction force on the particle points backward.
        """
        if normal_force <= 0.0:
            return 0.0
        return -self.coefficient * normal_force * math.tanh(
            slip_speed / self.regularization_speed_m_s
        )

    def dissipation_rate(self, normal_force: float, slip_speed: float) -> float:
        """``|F_t| * |v_slip|`` [W] — the quantity booked as friction energy."""
        return abs(self.tangential_force(normal_force, slip_speed)) * abs(slip_speed)


def hertz_force_vector(
    delta: np.ndarray, stiffness: np.ndarray
) -> np.ndarray:
    """Vectorised Hertz ``k_H delta^(3/2)`` (delta already clipped at >= 0)."""
    return stiffness * np.power(delta, 1.5)


def capacity_vector(
    delta: np.ndarray, radius: np.ndarray, pressure: float, thickness: float
) -> np.ndarray:
    """Vectorised crushing capacity ``p_c t w(delta)``."""
    d = np.clip(delta, 0.0, 2.0 * radius)
    w2 = 2.0 * radius * d - d * d
    w = 2.0 * np.sqrt(np.maximum(w2, 0.0))
    return pressure * thickness * w


def coulomb_force_vector(
    normal: np.ndarray, slip: np.ndarray, mu: float, v_eps: float
) -> np.ndarray:
    """Vectorised regularised Coulomb force (negative = opposing the slip)."""
    return -mu * normal * np.tanh(slip / v_eps)