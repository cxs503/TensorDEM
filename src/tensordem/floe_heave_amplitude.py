"""Finite-amplitude heave of a floating ice floe: nonlinear closed-form reference.

Why this module exists
----------------------
``tensordem.floe_buoyancy`` provides the production heave model (per-particle
Archimedes buoyancy on a lattice of disks) plus a *linearised* period anchor
``T0 = 2 pi sqrt(m / K)`` with ``K = rho_w g t chord(h_eq)``.  The validation
document ``docs/ice_floe_physics_validation.md`` therefore recorded an explicit
open item: the anchor only holds at infinitesimal amplitude, and the
finite-amplitude (nonlinear) heave period was **not verified**.

The waterplane chord of a *disk* shrinks away from its centre line, so the
restoring force of a floe is genuinely nonlinear and the heave period is a
function of amplitude: quoting ``T0`` at large amplitude is a modelling error.
This module supplies an **independent** nonlinear reference so the production
model can be checked at finite amplitude:

* the submerged area of a disk comes from exact circular-segment geometry,
  available twice over (analytic closed form and a Gauss-Legendre column
  quadrature of the chord), so the reference does not simply reuse the
  engine's ``acos`` kernel;
* the buoyancy potential has a closed-form antiderivative, ``U(h)``, so the
  energy integral ``T = 2 int dy sqrt(m / (2 (E - U)))`` is evaluated with
  machine-precision accuracy and the standard sine substitution that removes
  the turning-point square-root singularity;
* a second, algorithmically unrelated path -- RK4 time integration of the same
  continuum ODE at a very small step -- cross-checks the quadrature.

Scope (must be quoted with any number taken from here): the reference is the
*model's own* single-degree-of-freedom heave oscillator -- no added mass, no
free-surface deformation, no wave damping, no rotation, no horizontal drift.  A
match therefore verifies the force-law/integrator/period-estimator bookkeeping
of the solver, **not** the physical heave period of a real floe (the missing
added mass alone would lengthen the physical period by ``sqrt(1 + Ca)``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .buoyancy_wet import WetCouplingConfig
from .floe_buoyancy import (
    RigidHeaveFloe,
    mean_period,
    upward_crossing_periods,
    waterplane_chord,
)

__all__ = [
    "DiskHeaveReference",
    "HeaveAmplitudeResult",
    "submerged_disk_area",
    "submerged_disk_area_quadrature",
    "submerged_disk_area_antiderivative",
    "simulate_heave_at_amplitude",
    "amplitude_sweep",
]


# ---------------------------------------------------------------------------
# Disk geometry: submerged area and its antiderivative
# ---------------------------------------------------------------------------

def submerged_disk_area(h: float, radius: float) -> float:
    """Submerged area [m^2] of a disk of ``radius`` whose centre is ``h`` above water.

    Circular-segment closed form.  With the waterline a signed distance
    ``d = -h`` above the centre, the area of the circle below the chord is
    ``pi R^2 - [R^2 acos(d/R) - d sqrt(R^2 - d^2)]``, i.e.

        A(h) = pi R^2 - R^2 acos(-h/R) - h sqrt(R^2 - h^2),   |h| < R,

    ``A = 0`` for ``h >= R`` (dry) and ``A = pi R^2`` for ``h <= -R``.
    ``dA/dh = 2 sqrt(R^2 - h^2)`` reproduces the waterplane chord.
    """
    r = float(radius)
    h = float(h)
    if h >= r:
        return 0.0
    if h <= -r:
        return math.pi * r * r
    return math.pi * r * r - r * r * math.acos(-h / r) - h * math.sqrt(r * r - h * h)


def submerged_disk_area_quadrature(h: float, radius: float, order: int = 64) -> float:
    """Same area by Gauss-Legendre column quadrature (independent derivation).

    Columns are parametrised as ``x = R sin(phi)`` so the square-root circle
    boundary is resolved exactly; the ``phi`` range is split at ``s(phi) = |h|``
    where the submerged column length has a C0 kink, so each piece is smooth and
    ``order`` nodes give machine-precision accuracy.
    """
    r = float(radius)
    h = float(h)
    if abs(h) >= r:
        return 0.0 if h >= r else math.pi * r * r
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    phi_c = math.acos(abs(h) / r)

    def _integrand(phi: np.ndarray) -> np.ndarray:
        chord = r * np.cos(phi)
        if h >= 0.0:
            sub = np.maximum(chord - h, 0.0)        # partly dry column
        else:
            # centre below the waterline: either the whole column is submerged
            # (chord <= -h) or only its top is dry, submerged length s - h.
            sub = np.where(chord >= -h, chord - h, 2.0 * chord)
        return sub * r * np.cos(phi)                # column width r cos(phi) dphi

    total = 0.0
    for lo, hi in ((-math.pi / 2.0, -phi_c), (-phi_c, phi_c), (phi_c, math.pi / 2.0)):
        if hi <= lo:
            continue
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        total += float(np.dot(weights, _integrand(mid + half * nodes)) * half)
    return total


def submerged_disk_area_antiderivative(h: float, radius: float) -> float:
    """``I(h) = int_{0}^{h} A(xi) dxi`` [m^3], closed form (clamped outside ``[-R, R]``).

    Needed for the exact buoyancy potential.  With ``u = h/R``

        I(h) = R^3 [ pi u - (u acos(-u) + sqrt(1-u^2) - 1)
                     + ((1-u^2)^{3/2} - 1)/3 ],

    and ``dI/dh = A(h)``.  Outside the disk range the area is constant
    (0 or ``pi R^2``), so the antiderivative is continued linearly.
    """
    r = float(radius)
    h = float(h)
    if h >= r:
        return submerged_disk_area_antiderivative(r - 1e-15 * r, r) + math.pi * r * r * (h - r)
    if h <= -r:
        return submerged_disk_area_antiderivative(-r + 1e-15 * r, r)
    u = h / r
    s = math.sqrt(max(1.0 - u * u, 0.0))
    return r ** 3 * (math.pi * u - (u * math.acos(-u) + s - 1.0)
                     + ((s ** 3) - 1.0) / 3.0)


# ---------------------------------------------------------------------------
# Nonlinear heave reference (single-degree-of-freedom, exact energy integral)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiskHeaveReference:
    """Exact nonlinear heave oscillator of one floating disk (the model's own physics).

    ``mass`` is the disk mass, the restoring force is the Archimedes buoyancy of
    the circular segment minus the weight, and the potential is referred to the
    model's own equilibrium centre height ``h_eq`` (``U(h_eq) = 0``, also
    ``F_net(h_eq) = 0`` to machine precision because the equilibrium is solved
    here rather than taken from the engine).
    """

    radius: float
    thickness: float = 1.0
    rho_water: float = 1025.0
    rho_ice: float = 917.0
    gravity: float = 9.81
    water_level: float = 0.0
    order: int = 400

    # -- bulk properties -------------------------------------------------
    @property
    def mass(self) -> float:
        return self.rho_ice * math.pi * self.radius ** 2 * self.thickness

    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2

    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    @property
    def buoyancy_coeff(self) -> float:
        return self.rho_water * self.gravity * self.thickness

    def buoyancy(self, h: float) -> float:
        return self.buoyancy_coeff * submerged_disk_area(h, self.radius)

    def net_force(self, h: float) -> float:
        """Restoring force [N] in +y (buoyancy minus weight), zero at ``h_eq``."""
        return self.buoyancy(h) - self.weight

    def equilibrium_height(self) -> float:
        """Centre height relative to the water surface at ``F_net = 0``."""
        lo, hi = -self.radius, self.radius
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.net_force(mid) > 0.0:
                lo = mid          # too much buoyancy -> raise the centre
            else:
                hi = mid
            if hi - lo < 1e-16 * self.radius:
                break
        return 0.5 * (lo + hi)

    @property
    def h_eq(self) -> float:
        return self.equilibrium_height()

    def linear_stiffness(self) -> float:
        """``K = rho_w g t chord(h_eq)`` -- the linearised anchor."""
        return self.buoyancy_coeff * waterplane_chord(self.radius, self.h_eq + self.water_level)

    def linear_period(self) -> float:
        """``T0 = 2 pi sqrt(m / K)`` at infinitesimal amplitude."""
        return 2.0 * math.pi * math.sqrt(self.mass / self.linear_stiffness())

    # -- potential -------------------------------------------------------
    def potential(self, h: float) -> float:
        """``U(h) = int_{h_eq}^{h} (m g - F_b(xi)) dxi`` [J], ``U(h_eq) = 0``.

        Uses the closed-form antiderivative, so the value is exact rather than
        quadrature-approximated; ``U`` is positive on both sides of ``h_eq``.
        """
        h_eq = self.h_eq
        d_i = submerged_disk_area_antiderivative(h, self.radius) \
            - submerged_disk_area_antiderivative(h_eq, self.radius)
        return self.weight * (h - h_eq) - self.buoyancy_coeff * d_i

    def escape_amplitude(self) -> float:
        """Largest amplitude whose orbit still stays inside the buoyancy well.

        The well is bounded: below ``h = -R`` the disk is fully submerged and the
        net force is constant (no restoring branch), above ``h = +R`` it is dry
        and in free fall.  The bound state therefore ends where
        ``U(h_eq + A)`` reaches ``min(U(-R), U(+R))``.
        """
        h_eq = self.h_eq
        bound = min(self.potential(-self.radius), self.potential(self.radius))
        lo, hi = 1e-12 * self.radius, self.radius
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.potential(h_eq + mid) < bound:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-15 * self.radius:
                break
        return 0.5 * (lo + hi)

    def turning_point(self, amplitude: float, tol: float = 1e-15) -> float:
        """Lower turning height for a release from rest at ``h_eq + amplitude``.

        Bisection on ``U(h) = U(h_eq + A)`` in ``[h_eq - 2A, h_eq]``; ``U`` is
        increasing downward there, so the bracket is ``(U - target) > 0`` at the
        low end and ``< 0`` at ``h_eq``.
        """
        h_eq = self.h_eq
        target = self.potential(h_eq + amplitude)
        lo, hi = max(h_eq - 2.0 * amplitude, -self.radius), h_eq
        if self.potential(lo) < target:
            raise ValueError("amplitude leaves the single-well heave regime "
                             "(the floe would leave the water)")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.potential(mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * self.radius:
                break
        return 0.5 * (lo + hi)

    # -- period ----------------------------------------------------------
    def period(self, amplitude: float) -> float:
        """Nonlinear full period [s] for a release from rest at ``h_eq + A``.

        ``T = 2 int_{y_lo}^{y_hi} dy sqrt(m / (2 (E - U)))``; the ``1/sqrt``
        singularity at both turning points is removed by ``y = y_m + a sin``.
        """
        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive")
        h_eq = self.h_eq
        y_hi = h_eq + amplitude
        y_lo = self.turning_point(amplitude)
        energy = self.potential(y_hi)
        nodes, weights = np.polynomial.legendre.leggauss(self.order)
        theta = 0.5 * math.pi * nodes
        y_m, a = 0.5 * (y_lo + y_hi), 0.5 * (y_hi - y_lo)
        ys = y_m + a * np.sin(theta)
        deficit = np.maximum(energy - np.array([self.potential(float(y)) for y in ys]), 1e-300)
        integrand = a * np.cos(theta) * np.sqrt(self.mass / (2.0 * deficit))
        half = 0.5 * math.pi * float(np.dot(weights, integrand))
        return 2.0 * half

    def growth_factor(self, amplitude: float) -> float:
        """``T(A) / T(0)`` -- the amplitude correction the linear anchor cannot see."""
        return self.period(amplitude) / self.linear_period()

    # -- independent numeric cross-check ---------------------------------
    def rk4_period(self, amplitude: float, steps_per_period: int = 20000,
                   periods: int = 5) -> float:
        """Period by RK4 time integration of the same ODE (unrelated to the quadrature)."""
        h_eq = self.h_eq
        dt = self.linear_period() / steps_per_period
        y, v = h_eq + amplitude, 0.0

        def _acc(h: float, vel: float = 0.0) -> float:
            return self.net_force(h) / self.mass

        ts: list[float] = []
        ys: list[float] = []
        n = int(periods * steps_per_period) + 1
        for k in range(n):
            ts.append(k * dt)
            ys.append(y)
            if k == n - 1:
                break
            k1y, k1v = v, _acc(y)
            k2y, k2v = v + 0.5 * dt * k1v, _acc(y + 0.5 * dt * k1y)
            k3y, k3v = v + 0.5 * dt * k2v, _acc(y + 0.5 * dt * k2y)
            k4y, k4v = v + dt * k3v, _acc(y + dt * k3y)
            y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return mean_period(upward_crossing_periods(ts, ys, h_eq), skip=1)


# ---------------------------------------------------------------------------
# Production model driver (velocity-Verlet on the engine's buoyancy kernel)
# ---------------------------------------------------------------------------

@dataclass
class HeaveAmplitudeResult:
    """One simulated finite-amplitude heave run of the production model."""

    amplitude: float
    period: float
    t_linear: float
    dt: float
    n_particles: int
    mass: float
    y_eq: float
    peak_up: float
    peak_down: float
    energy_drift: float
    periods_measured: int

    def as_dict(self) -> dict:
        return {
            "amplitude": self.amplitude,
            "T_s": self.period,
            "T_linear_s": self.t_linear,
            "growth_vs_linear": self.period / self.t_linear,
            "dt_s": self.dt,
            "n_particles": self.n_particles,
            "mass_kg": self.mass,
            "y_eq": self.y_eq,
            "peak_up": self.peak_up,
            "peak_down": self.peak_down,
            "energy_drift_relative": self.energy_drift,
            "periods_measured": self.periods_measured,
        }


def _discrete_equilibrium(floe: RigidHeaveFloe, offsets: torch.Tensor,
                          particle_radius: float, cfg: WetCouplingConfig) -> float:
    """Bisection equilibrium height of the *lattice* floe (engine force kernel)."""
    weight = floe.mass * cfg.gravity
    lo = -float(offsets[:, 1].abs().max()) - 3.0 * particle_radius
    hi = float(offsets[:, 1].abs().max()) + 3.0 * particle_radius
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if floe.buoyancy_force(mid) > weight:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * particle_radius:
            break
    return 0.5 * (lo + hi)


def simulate_heave_at_amplitude(
    offsets: torch.Tensor,
    particle_radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
    dt: float,
    amplitude: float,
    linear_period_hint: float,
    periods: float = 6.0,
    settle_periods: float = 1.0,
) -> HeaveAmplitudeResult:
    """Release the lattice floe from rest at ``y_eq + amplitude``; measure the period.

    The period definition matches the reference: consecutive upward crossings of
    the equilibrium height (one full orbit), averaged after the transient.
    """
    n = int(offsets.shape[0])
    floe = RigidHeaveFloe(offsets, particle_radius, thickness, density_ice, cfg, dt)
    y_eq = _discrete_equilibrium(floe, offsets, particle_radius, cfg)
    floe.y, floe.v = y_eq + amplitude, 0.0
    n_steps = int(round((periods + settle_periods) * linear_period_hint / dt))
    traj = floe.run(n_steps, record_every=1, record_energy=False)

    crossings = upward_crossing_periods(traj.t, traj.y, y_eq)
    skip = max(int(round(settle_periods)) - 1, 0)
    period = mean_period(crossings, skip=skip)

    t_settle = settle_periods * linear_period_hint
    mask = [k for k, t in enumerate(traj.t) if t >= t_settle]
    ys = [traj.y[k] for k in mask]
    # Energy ledger on a thinned sample: the analytic per-particle potential is
    # comparatively expensive, so only ~200 points are audited.
    thin = max(len(mask) // 200, 1)
    energies = [floe.total_energy(traj.y[k], traj.v[k]) for k in mask[::thin]]
    e0 = energies[0] if energies else 0.0
    drift = max((abs(e - e0) for e in energies), default=0.0) / max(abs(e0), 1e-300)
    return HeaveAmplitudeResult(
        amplitude=amplitude,
        period=period,
        t_linear=linear_period_hint,
        dt=dt,
        n_particles=n,
        mass=floe.mass,
        y_eq=y_eq,
        peak_up=max(ys) - y_eq,
        peak_down=y_eq - min(ys),
        energy_drift=drift,
        periods_measured=len(crossings) - skip,
    )


def amplitude_sweep(
    offsets: torch.Tensor,
    particle_radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
    amplitudes: Sequence[float],
    dt: float,
    linear_period_hint: float,
    periods: float = 6.0,
) -> list[HeaveAmplitudeResult]:
    """Run :func:`simulate_heave_at_amplitude` over an amplitude ladder."""
    return [
        simulate_heave_at_amplitude(
            offsets, particle_radius, thickness, density_ice, cfg, dt, float(a),
            linear_period_hint=linear_period_hint, periods=periods,
        )
        for a in amplitudes
    ]