"""Long-wave multiple scattering through a spaced array of ice floes.

Why this module exists
----------------------
Nothing in the tree propagates a wave *through* a floe field, so the standard
ice-zone quantity -- the transmission coefficient ``T`` of an array, and its
dependence on floe concentration, spacing and wavelength -- could not be
computed at all.  The coupled floe solver in
:mod:`tensordem.wave_floe.response` is driven by a *prescribed* incident wave, so
it cannot produce a scattered field either; a genuine ``T`` needs a free-surface
unknown.

This module supplies the first scattering model in the tree: the classical
**long-wave (slab) formulation** of ice-coupled waves, which is exact within its
assumption ``k d << 1``.

Model
-----
In shallow water with a continuous floe cover of mass per unit area
``mu = rho_i h_i`` the long-wave momentum equation carries the extra inertia of
the cover, giving the well-known reduced phase speed

    c_i = sqrt(g d / (1 + mu/(rho_w d))).

Open water has ``c_0 = sqrt(g d)``.  Both regions satisfy the same non-dispersive
long-wave equation, so the surface elevation is a sum of two counter-propagating
plane waves in every segment and the array is solved **exactly** by a transfer
matrix: propagate through each segment, match ``eta`` and the volume flux
``d u`` at every interface.  For an array of ``N`` floes of length ``L_f``
separated by gaps ``L_g`` the result is a Bragg grating: ``T`` shows transmission
bands and a gap near ``k_0 (L_f + L_g) = pi``.

Checks that are independent of the transfer-matrix algebra
----------------------------------------------------------
* a single interface reproduces the closed-form Fresnel result
  ``T_eta = 2 Z_1/(Z_1 + Z_2)`` with the impedance ``Z = d k/omega``;
* for a non-dissipative array ``|T|^2 + |R|^2 = 1`` (energy closure, exact);
* ``T = 1`` for an empty channel (no floes) at every frequency;
* reciprocity: illuminating from the right gives the same ``|T|``.

Honest scope: this is a *long-wave* model without dispersion, so it cannot be
compared against finite-depth transmission experiments (Kohout et al. 2014 style
data are in the dispersive regime ``k d ~ 1``); that comparison is reported as
WITHHELD, with the reason, in the validation report.  What it does provide is a
self-consistent, energy-closing transmission capability with exact limits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .response import RHO_ICE, RHO_WATER
from .waves import GRAVITY


def long_wave_speed(depth: float, rho_ice: float = RHO_ICE,
                    thickness: float = 0.0, rho_water: float = RHO_WATER,
                    g: float = GRAVITY) -> float:
    """Phase speed ``sqrt(g d/(1 + mu/(rho_w d)))`` (``thickness=0`` -> open water)."""
    mu = rho_ice * thickness
    return math.sqrt(g * depth / (1.0 + mu / (rho_water * depth)))


def wavenumber(omega: float, depth: float, rho_ice: float = RHO_ICE,
               thickness: float = 0.0, rho_water: float = RHO_WATER,
               g: float = GRAVITY) -> float:
    return omega / long_wave_speed(depth, rho_ice, thickness, rho_water, g)


def _propagate(k: float, length: float) -> np.ndarray:
    """Coefficient propagator over a homogeneous segment (A,B) -> (A e^{ikl}, B e^{-ikl})."""
    return np.array([[complex(np.exp(1j * k * length)), 0.0],
                     [0.0, complex(np.exp(-1j * k * length))]])


def _interface(k1: float, k2: float) -> np.ndarray:
    """Matching matrix for ``eta`` and ``d u`` continuity at a local interface.

    ``eta = A + B`` and, from the long-wave momentum equation,
    ``u = (omega/(k d))(A - B)``; ``d`` and ``omega`` are common so continuity
    of ``u`` reduces to ``(A1 - B1)/k1 = (A2 - B2)/k2``, i.e. the impedance is
    ``Z = 1/k`` (equivalently ``Z = c``).  Using ``k1(A1-B1) = k2(A2-B2)``
    instead -- the reciprocal ratio -- violates the single-interface energy
    balance; :func:`single_interface_transmission` is the check that catches it.
    """
    a = 0.5 * (1.0 + k2 / k1)
    b = 0.5 * (1.0 - k2 / k1)
    return np.array([[a, b], [b, a]], dtype=complex)


@dataclass
class FloeArrayTransmission:
    """Transmission/reflection of a spaced floe array in the long-wave limit."""

    period: float = 8.0
    depth: float = 1.0
    floe_length: float = 5.0
    gap: float = 5.0
    n_floes: int = 12
    thickness: float = 0.2
    rho_ice: float = RHO_ICE
    rho_water: float = RHO_WATER
    gravity: float = GRAVITY

    def __post_init__(self) -> None:
        if self.n_floes < 0:
            raise ValueError("n_floes must be >= 0")
        if not math.isfinite(self.period) or self.period <= 0.0:
            raise ValueError("period must be positive")

    # -- derived --------------------------------------------------------
    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.period

    @property
    def k0(self) -> float:
        return wavenumber(self.omega, self.depth, g=self.gravity)

    @property
    def k_floe(self) -> float:
        return wavenumber(self.omega, self.depth, self.rho_ice, self.thickness,
                          self.rho_water, self.gravity)

    @property
    def impedance_ratio(self) -> float:
        return self.k_floe / self.k0

    @property
    def total_length(self) -> float:
        """Along-array extent of the covered+gapped part."""
        if self.n_floes == 0:
            return 0.0
        return self.n_floes * self.floe_length + (self.n_floes - 1) * self.gap

    @property
    def concentration(self) -> float:
        if self.n_floes == 0:
            return 0.0
        span = self.floe_length + self.gap
        return self.n_floes * self.floe_length / (self.n_floes * self.floe_length
                                                  + (self.n_floes - 1) * self.gap) \
            if span > 0 else 0.0

    # -- transfer matrix ------------------------------------------------
    def _total_matrix(self) -> np.ndarray:
        M = np.eye(2, dtype=complex)
        k0, ki = self.k0, self.k_floe
        for j in range(int(self.n_floes)):
            M = _interface(k0, ki) @ M
            M = _propagate(ki, self.floe_length) @ M
            M = _interface(ki, k0) @ M
            if j < int(self.n_floes) - 1 and self.gap > 0.0:
                M = _propagate(k0, self.gap) @ M
        return M

    def coefficients(self, from_right: bool = False) -> tuple[complex, complex]:
        """``(T, R)`` complex amplitude ratios (eta transmitted / incident)."""
        M = self._total_matrix()
        if from_right:
            # Illumination from the right = transpose (reciprocity of the closed
            # system); swapping roles of A and B is the exact transpose map.
            M = M[::-1, ::-1]
        m11, m12 = M[0, 0], M[0, 1]
        m21, m22 = M[1, 0], M[1, 1]
        r = -m21 / m22
        t = m11 - m12 * m21 / m22
        return complex(t), complex(r)

    @property
    def transmission(self) -> float:
        return abs(self.coefficients()[0])

    @property
    def reflection(self) -> float:
        return abs(self.coefficients()[1])

    @property
    def energy_closure(self) -> float:
        """``|T|^2 + |R|^2`` -- 1 for a non-dissipative array (exact check)."""
        t, r = self.coefficients()
        return abs(t) ** 2 + abs(r) ** 2

    def to_dict(self) -> dict:
        t, r = self.coefficients()
        t_r, r_r = self.coefficients(from_right=True)
        return {
            "period": self.period,
            "omega": self.omega,
            "depth": self.depth,
            "floe_length": self.floe_length,
            "gap": self.gap,
            "n_floes": int(self.n_floes),
            "thickness": self.thickness,
            "k0": self.k0,
            "k_floe": self.k_floe,
            "k0_depth": self.k0 * self.depth,
            "k0_floe_length": self.k0 * self.floe_length,
            "impedance_ratio": self.impedance_ratio,
            "concentration": self.concentration,
            "transmission": abs(t),
            "reflection": abs(r),
            "energy_closure": abs(t) ** 2 + abs(r) ** 2,
            "reciprocity_error": abs(abs(t) - abs(t_r)),
            "transmission_phase": math.atan2(t.imag, t.real),
        }


def single_interface_transmission(period: float, depth: float,
                                  thickness: float, rho_ice: float = RHO_ICE,
                                  rho_water: float = RHO_WATER,
                                  gravity: float = GRAVITY) -> dict:
    """Closed-form Fresnel result for one open-water / floe-covered interface.

    ``Z = d k / omega``; ``T_eta = 2 Z_1/(Z_1 + Z_2)``,
    ``R_eta = (Z_1 - Z_2)/(Z_1 + Z_2)``; energy closure in the flux sense is
    ``|T|^2 + |R|^2 = 1`` because both sides share the same depth.
    """
    omega = 2.0 * math.pi / period
    k0 = wavenumber(omega, depth, g=gravity)
    ki = wavenumber(omega, depth, rho_ice, thickness, rho_water, gravity)
    # Flux impedance Z = c (phase speed), so Z0/Zi = ki/k0.
    z0 = omega / k0
    zi = omega / ki
    t = 2.0 * z0 / (z0 + zi)
    r = (z0 - zi) / (z0 + zi)
    return {
        "transmission": abs(t), "reflection": abs(r),
        "energy_closure": t ** 2 + r ** 2,
        "energy_closure_flux": abs(t) ** 2 * zi / z0 + abs(r) ** 2,
        "impedance_open": z0, "impedance_floe": zi,
    }


def transmission_vs_kh(periods: Sequence[float], **kwargs) -> list[dict]:
    """``T`` versus the dimensionless abscissae ``k0 d`` and ``k0 L_f``."""
    out = []
    for T in periods:
        arr = FloeArrayTransmission(period=T, **kwargs)
        rec = arr.to_dict()
        rec["kh_depth"] = rec["k0_depth"]
        rec["k_floe_length"] = rec["k0_floe_length"]
        out.append(rec)
    return out


def concentration_sweep(period: float, n_floes: int = 12,
                        depths: Sequence[float] = (1.0,),
                        thicknesses: Sequence[float] = (0.2,),
                        gaps: Sequence[float] = (2.5, 5.0, 10.0)) -> list[dict]:
    """``T`` versus gap (i.e. concentration) for a fixed number of floes."""
    out = []
    for d in depths:
        for h in thicknesses:
            for gap in gaps:
                arr = FloeArrayTransmission(period=period, depth=d, gap=gap,
                                            n_floes=n_floes, thickness=h,
                                            floe_length=5.0)
                rec = arr.to_dict()
                rec["gap"] = gap
                rec["depth"] = d
                rec["thickness"] = h
                out.append(rec)
    return out

def slab_transmission_analytic(period: float, depth: float, floe_length: float,
                               thickness: float, rho_ice: float = RHO_ICE,
                               rho_water: float = RHO_WATER,
                               gravity: float = GRAVITY) -> dict:
    """Closed form for a single homogeneous floe-covered slab in open water.

    ``T = 1/(cos(k_i L) - i (Z_i^2 + Z_0^2)/(2 Z_0 Z_i) sin(k_i L))`` with the flux
    impedance ``Z = c``.  Independent of the transfer-matrix algebra, so it is the
    reference the transfer matrix is checked against (two interfaces plus
    propagation, no recursion).
    """
    omega = 2.0 * math.pi / period
    k0 = wavenumber(omega, depth, g=gravity)
    ki = wavenumber(omega, depth, rho_ice, thickness, rho_water, gravity)
    z0, zi = omega / k0, omega / ki
    denom = complex(math.cos(ki * floe_length),
                    -(zi ** 2 + z0 ** 2) / (2.0 * z0 * zi) * math.sin(ki * floe_length))
    t = 1.0 / denom
    return {
        "transmission": abs(t),
        "energy": abs(t) ** 2,
        "k0": k0, "k_floe": ki,
        "impedance_ratio": zi / z0,
    }
