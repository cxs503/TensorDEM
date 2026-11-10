"""Linear (Airy) wave kinematics with a *solved* finite-depth dispersion relation.

The shipped :func:`tensordem.buoyancy_wet.wave_kinematics` builds ``k = 2 pi/lambda``
from a wavelength the caller supplies and selects the depth form by
``d > 0.5 lambda`` (a switch, not a solve); it also returns only ``u_x, u_y,
du_x/dt`` -- there is no vertical acceleration, which is why the shipped Morison
load has no heave inertia term (see the defect fix in
:mod:`tensordem.buoyancy_wet`).

This module adds the missing pieces without duplicating the physics:

* :func:`solve_wavenumber` -- Newton iteration on ``w^2 = g k tanh(k d)`` with the
  deep-water closed form as the initial guess and the exact deep-water limit
  ``k = w^2/g`` as the ``d -> inf`` branch;
* :class:`RegularWave` / :class:`IrregularWave` -- surface elevation *and* full
  kinematics (both velocities and both accelerations) at an arbitrary point
  ``(x, z)``, with group velocity and energy flux;
* :func:`wave_energy_flux` -- the exact linear result ``P = E c_g`` together with
  an independent numerical check ``P = <p u>`` from the pressure/velocity
  quadrature, used as the C5 energy benchmark.

Kinematics (``z`` up, mean water level ``z = 0``, water depth ``d``)::

    eta   = A cos(theta),                     theta = k x - w t + phi
    u     = A w  cosh(k(z+d))/sinh(kd) cos(theta)
    w     = A w  sinh(k(z+d))/sinh(kd) sin(theta)
    du/dt = A w^2 cosh(k(z+d))/sinh(kd) sin(theta)
    dw/dt = -A w^2 sinh(k(z+d))/sinh(kd) cos(theta)

and the deep-water limit replaces ``cosh(k(z+d))/sinh(kd) -> exp(kz)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x renamed trapz

GRAVITY = 9.81
INFINITE_DEPTH = float("inf")


def deep_water_wavenumber(omega: float, g: float = GRAVITY) -> float:
    """Exact deep-water branch ``k = w^2 / g``."""
    return float(omega) ** 2 / g


def solve_wavenumber(omega: float, depth: float | None = None,
                     g: float = GRAVITY, tol: float = 1e-14,
                     max_iter: int = 100) -> float:
    """Solve ``w^2 = g k tanh(k d)`` for ``k`` (Newton, machine precision).

    ``depth`` None or infinite -> deep-water branch.  The Newton step is
    ``k <- k - f/f'`` with ``f' = g tanh(kd) + g k d sech^2(kd)``, seeded by the
    deep-water value scaled down by the shallow-water ratio; it converges in a
    handful of iterations for every ``kd`` away from 0, where ``tanh`` is still
    perfectly behaved because ``f' -> g k d`` stays positive.
    """
    omega = float(omega)
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if depth is None or not np.isfinite(depth):
        return deep_water_wavenumber(omega, g)
    d = float(depth)
    if d <= 0.0:
        raise ValueError("depth must be positive")
    k = deep_water_wavenumber(omega, g)
    if k * d < 1.0:  # shallow-water seed keeps Newton out of the flat region
        k = omega / math.sqrt(g * d)
    for _ in range(max_iter):
        kd = k * d
        th = math.tanh(kd)
        f = g * k * th - omega * omega
        if abs(f) <= tol * omega * omega:
            break
        sech2 = 1.0 - th * th
        df = g * th + g * k * d * sech2
        step = f / df
        k -= step
        if abs(step) <= tol * max(k, 1e-30):
            break
    return k


@dataclass(frozen=True)
class RegularWave:
    """A single Airy wave in water of depth ``depth`` (``inf`` = deep)."""

    height: float = 0.1
    period: float = 6.0
    depth: float = INFINITE_DEPTH
    phase: float = 0.0
    gravity: float = GRAVITY
    water_density: float = 1025.0

    def __post_init__(self) -> None:
        # Cache the dispersion solve: the force kernels call ``wavenumber`` and
        # ``omega`` several times per RK4 stage, and a Newton solve there would
        # dominate the time-domain cost.
        omega = 2.0 * math.pi / self.period
        object.__setattr__(self, "_omega", omega)
        object.__setattr__(self, "_wavenumber", solve_wavenumber(omega, self.depth,
                                                                 self.gravity))

    # -- derived properties ---------------------------------------------
    @property
    def amplitude(self) -> float:
        return 0.5 * self.height

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def wavenumber(self) -> float:
        return self._wavenumber

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.wavenumber

    @property
    def kh(self) -> float:
        return self.wavenumber * self.amplitude  # amplitude-normalised abscissa

    @property
    def kd(self) -> float:
        return self.wavenumber * self.depth

    def _depth_factor(self, z: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """``(cosh(k(z+d))/sinh(kd), sinh(k(z+d))/sinh(kd))`` or deep limits."""
        z = np.asarray(z, dtype=float)
        k = self.wavenumber
        if not np.isfinite(self.depth):
            e = np.exp(k * z)
            return e, e
        kd = self.kd
        sh = math.sinh(kd)
        return np.cosh(k * (z + self.depth)) / sh, np.sinh(k * (z + self.depth)) / sh

    def surface_elevation(self, x, t: float) -> np.ndarray:
        th = self.wavenumber * np.asarray(x, dtype=float) - self.omega * t + self.phase
        return self.amplitude * np.cos(th)

    def kinematics(self, x, z, t: float) -> dict:
        """Surface-referenced kinematics: velocities and accelerations."""
        x = np.asarray(x, dtype=float)
        th = self.wavenumber * x - self.omega * t + self.phase
        ch, sh = self._depth_factor(z)
        A, w = self.amplitude, self.omega
        return {
            "theta": th,
            "u": A * w * ch * np.cos(th),
            "w": A * w * sh * np.sin(th),
            "du_dt": A * w ** 2 * ch * np.sin(th),
            "dw_dt": -A * w ** 2 * sh * np.cos(th),
        }

    # -- energy ---------------------------------------------------------
    @property
    def phase_speed(self) -> float:
        return self.omega / self.wavenumber

    @property
    def group_velocity(self) -> float:
        """``c_g = c/2 (1 + 2kd/sinh(2kd))`` (deep: ``c/2``)."""
        c = self.phase_speed
        if not np.isfinite(self.depth):
            return 0.5 * c
        kd = self.kd
        return 0.5 * c * (1.0 + 2.0 * kd / math.sinh(2.0 * kd))

    @property
    def energy_per_area(self) -> float:
        """Wave energy density ``E = 1/2 rho g A^2`` [J/m^2]."""
        return 0.5 * self.water_density * self.gravity * self.amplitude ** 2

    @property
    def energy_flux(self) -> float:
        """Exact linear result ``P = E c_g`` [W/m]."""
        return self.energy_per_area * self.group_velocity

    def energy_flux_quadrature(self, n_z: int = 4000,
                               n_cells: int = 64) -> float:
        """Independent numerical check of ``P``: ``P = <p u>`` depth-integrated.

        ``p = rho g A cosh(k(z+d))/cosh(kd) cos(theta)`` (hydrostatic part of the
        linear pressure), ``u`` as above; the time average of ``cos^2`` is 1/2.
        Used as the C5 cross-check of :attr:`energy_flux` -- an unrelated route
        (depth quadrature of the pressure work) to the same number.
        """
        k = self.wavenumber
        A = self.amplitude
        if not np.isfinite(self.depth):
            z = np.linspace(-20.0 / k, 0.0, n_z)
            ch_p = np.exp(k * z)
            ch_u = np.exp(k * z)
        else:
            z = np.linspace(-self.depth, 0.0, n_z)
            ch_p = np.cosh(k * (z + self.depth)) / math.cosh(k * self.depth)
            ch_u = np.cosh(k * (z + self.depth)) / math.sinh(k * self.depth)
        integrand = self.water_density * self.gravity * A * ch_p * A * self.omega * ch_u
        # <cos^2> = 1/2 for the time average of p*u.
        return float(_trapz(integrand, z)) * 0.5


@dataclass(frozen=True)
class IrregularWave:
    """A linear superposition of Airy components (a realised sea state).

    Components are stored as ``(omega, amplitude, phase)`` triples; each carries
    its own wavenumber from the finite-depth dispersion relation, so the field is
    dispersive.  Elevation and kinematics are the componentwise sums.
    """

    omega: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    depth: float = INFINITE_DEPTH
    gravity: float = GRAVITY
    water_density: float = 1025.0
    x0: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "omega", np.asarray(self.omega, dtype=float))
        object.__setattr__(self, "amplitude", np.asarray(self.amplitude, dtype=float))
        object.__setattr__(self, "phase", np.asarray(self.phase, dtype=float))
        if not (self.omega.shape == self.amplitude.shape == self.phase.shape):
            raise ValueError("omega/amplitude/phase must share a shape")

    @classmethod
    def from_spectrum(cls, spec, depth: float = INFINITE_DEPTH,
                      phase_seed: int | None = 0, **kwargs) -> "IrregularWave":
        from .spectra import discretize

        w, a, p = discretize(spec, phase_seed=phase_seed)
        return cls(w, a, p, depth=depth, **kwargs)

    @property
    def wavenumber(self) -> np.ndarray:
        return np.array([solve_wavenumber(w, self.depth, self.gravity)
                         for w in self.omega])

    def surface_elevation(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        k = self.wavenumber
        out = np.zeros_like(x, dtype=float)
        for ki, wi, ai, pi in zip(k, self.omega, self.amplitude, self.phase):
            out += ai * np.cos(ki * x - wi * t + pi)
        return out

    def kinematics(self, x, z, t: float) -> dict:
        """Componentwise sum of the Airy kinematics (same keys as RegularWave)."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        k = self.wavenumber
        zeros = np.zeros(np.broadcast(x, z).shape, dtype=float)
        acc = {key: zeros.copy() for key in ("u", "w", "du_dt", "dw_dt")}
        for ki, wi, ai, pi in zip(k, self.omega, self.amplitude, self.phase):
            th = ki * x - wi * t + pi
            if not np.isfinite(self.depth):
                ch = sh = np.exp(ki * z)
            else:
                sh_d = math.sinh(ki * self.depth)
                ch = np.cosh(ki * (z + self.depth)) / sh_d
                sh = np.sinh(ki * (z + self.depth)) / sh_d
            acc["u"] += ai * wi * ch * np.cos(th)
            acc["w"] += ai * wi * sh * np.sin(th)
            acc["du_dt"] += ai * wi ** 2 * ch * np.sin(th)
            acc["dw_dt"] += -ai * wi ** 2 * sh * np.cos(th)
        return acc

    @property
    def variance(self) -> float:
        return float(np.sum(0.5 * self.amplitude ** 2))

    @property
    def significant_height(self) -> float:
        return 4.0 * math.sqrt(self.variance)


def wave_energy_flux(wave, n_z: int = 4000) -> dict:
    """Compare ``E c_g`` with the quadrature ``<p u>`` for a regular wave."""
    exact = wave.energy_flux
    quad = wave.energy_flux_quadrature(n_z=n_z)
    return {
        "flux_ecg": exact,
        "flux_quadrature": quad,
        "rel_error": abs(quad - exact) / abs(exact) if exact else float("nan"),
        "energy_density": wave.energy_per_area,
        "group_velocity": wave.group_velocity,
        "phase_speed": wave.phase_speed,
    }