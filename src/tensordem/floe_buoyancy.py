"""Rigid heave dynamics and analytic anchors for the 2-D floe buoyancy model.

The ``buoyancy_wet`` module provides *force kernels* (submerged-fraction
buoyancy and linear wet drag) that were never validated above the level of a
single damped settling run.  This module adds the missing dynamics layer as a
small, dependency-free tool:

1. Exact Archimedes anchors.  For a floating body in the model the submerged
   area fraction at rest satisfies ``f_sub = rho_ice / rho_water`` exactly
   (per particle: ``sum_i f_i = N * rho_ice / rho_water``).  The equivalent
   centre height is obtained by bisection on the circular-segment relation --
   *not* by the small-draft linearisation ``h = -(1 - rho_i/rho_w) R``.

2. Linearised heave anchors.  With waterplane chord ``c(h) = 2 sqrt(R^2 - h^2)``
   the restoring stiffness is ``K = rho_w g t c(h_eq)``, giving
   ``T = 2 pi sqrt(m / K)``.  For linear wet drag ``F = -C f_sub v`` the
   damping ratio is ``zeta = C f* / (2 m omega)``.

3. A rigid vertical (1-DOF) integrator that drives the *production* force
   kernels (``compute_buoyancy_force`` / ``compute_wet_drag_force``) so the
   anchors are checked against the shipped code, not against a reimplementation.

All quantities SI, per unit depth where applicable (``thickness`` is the
out-of-plane extrusion length, exactly as in ``buoyancy_wet``).

Model scope (deliberate): no free-surface deformation, no added mass, no
rotation, no horizontal drift, no tangential/contact forces inside the floe.
Added mass is the dominant missing term for a real floe (it lengthens the
physical heave period by ``sqrt(1 + Ca)``); every period comparison below is
therefore a check of the *model's own* stiffness/mass bookkeeping, which must
be stated as such whenever the number is quoted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

from .buoyancy_wet import (
    WetCouplingConfig,
    compute_buoyancy_force,
    compute_wet_drag_force,
    submerged_fraction_tensor,
)


# ---------------------------------------------------------------------------
# Geometric / hydrostatic anchors
# ---------------------------------------------------------------------------

def submerged_fraction(h: float, radius: float) -> float:
    """Submerged area fraction of a disk of ``radius`` whose centre sits
    ``h`` above the free surface (h > 0 = centre above water).

    Delegates to the production kernel so anchors and dynamics cannot drift.
    """
    return float(submerged_fraction_tensor(torch.tensor([float(h)]), radius, 0.0)[0])


def _simpson(points: torch.Tensor, values: torch.Tensor, dx: float) -> float:
    """Composite Simpson rule on an evenly spaced sample set (even count)."""
    weights = torch.ones_like(values)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float((weights * values).sum()) * dx / 3.0


def buoyancy_potential(radius: float, h: float, water_density: float, gravity: float,
                       thickness: float, n_slices: int = 2048) -> float:
    """Potential energy of the buoyancy force for one disk, ``U_b(h) = -int F_b dh``.

    Reference: ``U_b = 0`` with the disk fully above water.  The integral
    ``-rho_w g t * int_{-R}^{h} A_sub(xi) dxi`` is evaluated with Simpson
    quadrature on a vectorised sample of the production submerged-fraction
    kernel (no scalar python loop, so it is cheap enough for energy audits).
    """
    lo = -radius
    hi = min(max(h, lo), radius)
    if hi <= lo:
        return 0.0
    n = n_slices if n_slices % 2 == 0 else n_slices + 1
    dx = (hi - lo) / n
    xs = torch.linspace(lo, hi, n + 1, dtype=torch.float64)
    areas = math.pi * radius ** 2 * submerged_fraction_tensor(xs, radius, 0.0)
    return -water_density * gravity * thickness * _simpson(xs, areas, dx)


def equilibrium_center_height(rho_ice: float, rho_water: float, radius: float,
                              tol: float = 1e-15) -> float:
    """Exact centre height (relative to the water surface) of a floating disk.

    Solves ``f_sub(h) = rho_ice / rho_water`` by bisection on ``h in (-R, R)``.
    The naive small-draft estimate ``h = -(1 - rho_i/rho_w) R`` is *not* used;
    it is quantitatively off for floe-like density ratios (see the benchmark
    script, which reports the difference).
    """
    target = rho_ice / rho_water
    if not (0.0 < target < 1.0):
        raise ValueError("need 0 < rho_ice/rho_water < 1 for a floating body")
    lo, hi = -radius, radius
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if submerged_fraction(mid, radius) > target:
            lo = mid          # still too low in the water -> raise centre
        else:
            hi = mid
        if hi - lo < tol * radius:
            break
    return 0.5 * (lo + hi)


def waterplane_chord(radius: float, h: float) -> float:
    """Free-surface chord length of a disk whose centre is ``h`` above water."""
    if abs(h) >= radius:
        return 0.0
    return 2.0 * math.sqrt(max(radius * radius - h * h, 0.0))


def linear_heave_stiffness(rho_w: float, g: float, thickness: float,
                           radius: float, h_eq: float) -> float:
    """``K = -dF_b/dh = rho_w g t * chord(h_eq)``  [N/m]."""
    return rho_w * g * thickness * waterplane_chord(radius, h_eq)


def analytic_heave_period(mass: float, stiffness: float) -> float:
    """``T = 2 pi sqrt(m / K)`` [s] for the linearised heave mode."""
    if stiffness <= 0.0:
        raise ValueError("heave stiffness must be positive for a restoring mode")
    return 2.0 * math.pi * math.sqrt(mass / stiffness)


def analytic_damping_ratio(wet_drag_coeff: float, f_sub: float, mass: float,
                           omega: float) -> float:
    """``zeta = C f* / (2 m omega)`` for linear wet drag at the equilibrium draft."""
    return wet_drag_coeff * f_sub / (2.0 * mass * omega)


# ---------------------------------------------------------------------------
# Discrete floes (hex lattice) -- resolution-dependent anchors
# ---------------------------------------------------------------------------

def hex_lattice_floe(particle_radius: float, n_particles: int) -> torch.Tensor:
    """Offsets of the ``n_particles`` hex-lattice disks nearest the origin.

    Disks of radius ``a = particle_radius`` sit on a triangular lattice with
    spacing ``2a`` in x and ``sqrt(3) a`` in y, so they are tangent and never
    overlap -- the layout a DEM floe would actually use.  Selecting the ``N``
    sites nearest the origin (rather than clipping a circle) keeps the area
    identity ``N pi a^2 = pi R_eq^2`` exact for a prescribed equivalent floe
    radius ``R_eq = a sqrt(N)``, which is what makes the stiffness comparison
    below a pure *shape/lattice* discretisation error.

    Returns an ``(N, 2)`` float64 tensor of centre offsets.
    """
    a = float(particle_radius)
    n = int(n_particles)
    if a <= 0.0 or n <= 0:
        raise ValueError("particle_radius and n_particles must be positive")
    reach = 2.0 * a * (math.ceil(math.sqrt(n)) + 2)
    dy = a * math.sqrt(3.0)
    j_max = int(math.ceil(reach / dy)) + 1
    i_max = int(math.ceil(reach / (2.0 * a))) + 2
    sites: list[tuple[float, float]] = []
    for j in range(-j_max, j_max + 1):
        y = j * dy
        shift = a if j % 2 else 0.0
        for i in range(-i_max, i_max + 1):
            sites.append((i * 2.0 * a + shift, y))
    sites.sort(key=lambda p: math.hypot(p[0], p[1]))
    return torch.tensor(sites[:n], dtype=torch.float64)


def slab_lattice(width: float, height: float, particle_radius: float) -> torch.Tensor:
    """Rectangular (staggered-row) cluster of disks filling ``width`` x ``height``.

    Used for the *slab* convergence study: a body with vertical sides has an
    exactly height-independent continuum heave stiffness
    ``K = rho_w g t * W``, so the discrete model can be checked against a
    closed form that does not itself depend on the discretisation.
    """
    a = float(particle_radius)
    if a <= 0.0:
        raise ValueError("particle_radius must be positive")
    n_rows = max(int(round(height / (a * math.sqrt(3.0)))), 1)
    offsets = []
    for j in range(n_rows):
        y = j * a * math.sqrt(3.0)
        shift = a if j % 2 else 0.0
        n_cols = max(int((width - 2.0 * a) // (2.0 * a)) + 1, 1)
        x0 = -0.5 * (n_cols - 1) * 2.0 * a
        for i in range(n_cols):
            offsets.append((x0 + i * 2.0 * a + shift, y))
    off = torch.tensor(offsets, dtype=torch.float64)
    off[:, 1] -= off[:, 1].mean()
    off[:, 0] -= off[:, 0].mean()
    return off


def body_envelope(offsets: torch.Tensor, particle_radius: float) -> tuple[float, float]:
    """(width, height) of the axis-aligned envelope of a disk cluster."""
    a = float(particle_radius)
    width = float(offsets[:, 0].max() - offsets[:, 0].min()) + 2.0 * a
    height = float(offsets[:, 1].max() - offsets[:, 1].min()) + 2.0 * a
    return width, height


def discrete_buoyancy_profile(offsets: torch.Tensor, particle_radius: float,
                              thickness: float, density_ice: float,
                              cfg: WetCouplingConfig, y: float) -> float:
    """Total buoyancy force [N] on a rigid floe whose centroid is at height ``y``."""
    positions = offsets.clone()
    positions[:, 1] += y
    return float(compute_buoyancy_force(positions, particle_radius, thickness,
                                        density_ice, cfg)[:, 1].sum())


def discrete_stiffness(offsets: torch.Tensor, particle_radius: float,
                       thickness: float, density_ice: float,
                       cfg: WetCouplingConfig, y_eq: float) -> float:
    """Exact discrete restoring stiffness ``-dF_b/dy`` of the lattice at ``y_eq``.

    Per-particle ``dF_i/dy = -rho_w g t chord_i(y_i)``, so the aggregate is
    ``K = rho_w g t sum_i chord_i(y_i)`` -- a closed form, no finite
    differencing (the aggregate force is only piecewise smooth because
    particles enter/leave the water).
    """
    positions = offsets.clone()
    positions[:, 1] += y_eq
    fac = cfg.water_density * cfg.gravity * thickness
    total = 0.0
    for yi in positions[:, 1].tolist():
        total += waterplane_chord(particle_radius, yi - cfg.water_level)
    return fac * total


def discrete_buoyancy_potential(offsets: torch.Tensor, particle_radius: float,
                                thickness: float, density_ice: float,
                                cfg: WetCouplingConfig, y: float,
                                n_slices: int = 512) -> float:
    """Buoyancy potential energy of the lattice (Simpson, reference = fully dry).

    Vectorised over both particles and the height sample, so it stays usable
    inside energy audits of long trajectories.
    """
    reach = float(offsets[:, 1].abs().max()) + particle_radius
    lo = cfg.water_level - reach
    hi = min(max(y, lo), cfg.water_level + reach)
    if hi <= lo:
        return 0.0
    n = n_slices if n_slices % 2 == 0 else n_slices + 1
    dx = (hi - lo) / n
    ys = torch.linspace(lo, hi, n + 1, dtype=torch.float64)
    # centre heights of every particle for every sampled centroid height
    heights = ys[:, None] + offsets[:, 1][None, :] - cfg.water_level
    frac = submerged_fraction_tensor(heights.reshape(-1), particle_radius, 0.0)
    frac = frac.reshape(heights.shape)
    force = cfg.water_density * cfg.gravity * math.pi * particle_radius ** 2 * thickness \
        * frac.sum(dim=1)
    return -_simpson(ys, force, dx)


# ---------------------------------------------------------------------------
# Rigid heave integrator driving the production force kernels
# ---------------------------------------------------------------------------

@dataclass
class HeaveTrajectory:
    """Recorded rigid heave time series."""
    t: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"t": self.t, "y": self.y, "v": self.v, "energy": self.energy}


class RigidHeaveFloe:
    """Single-degree-of-freedom (vertical) rigid floe on the wet buoyancy model.

    The floe translates as a rigid body; every particle carries its own
    submerged fraction, so aggregate buoyancy is resolved per particle -- that
    is what makes the lattice resolution study meaningful.

    Integration is velocity Verlet on the *assembled* force, matching the
    scheme used by the existing floe benchmark (``benchmarks/ice_floe_physics.py``):

        v_{1/2} = v_n + a(y_n, v_n) dt/2
        y_{n+1} = y_n + v_{1/2} dt
        v_{n+1} = v_{1/2} + a(y_{n+1}, v_{1/2}) dt/2
    """

    def __init__(self, offsets: torch.Tensor, particle_radius: float,
                 thickness: float, density_ice: float,
                 cfg: WetCouplingConfig, dt: float,
                 y0: float | None = None) -> None:
        self.offsets = offsets.double()
        self.radius = float(particle_radius)
        self.thickness = float(thickness)
        self.density_ice = float(density_ice)
        self.cfg = cfg
        self.dt = float(dt)
        self.n_particles = int(offsets.shape[0])
        self.particle_area = math.pi * self.radius ** 2
        self.mass = self.density_ice * self.particle_area * self.thickness * self.n_particles
        self.water_level = float(cfg.water_level)
        self.y = float(cfg.water_level if y0 is None else y0)
        self.v = 0.0
        self.trajectory = HeaveTrajectory()

    # -- forces ---------------------------------------------------------
    def _positions(self, y: float) -> torch.Tensor:
        pos = self.offsets.clone()
        pos[:, 1] += y
        return pos

    def buoyancy_force(self, y: float) -> float:
        return float(compute_buoyancy_force(self._positions(y), self.radius,
                                            self.thickness, self.density_ice,
                                            self.cfg)[:, 1].sum())

    def wet_drag_force(self, y: float, v: float) -> float:
        vel = torch.zeros_like(self._positions(y))
        vel[:, 1] = v
        return float(compute_wet_drag_force(self._positions(y), vel, self.radius,
                                            self.cfg)[:, 1].sum())

    def acceleration(self, y: float, v: float) -> float:
        return (self.buoyancy_force(y) + self.wet_drag_force(y, v)) / self.mass - self.cfg.gravity

    # -- bookkeeping ----------------------------------------------------
    def total_energy(self, y: float, v: float) -> float:
        pot = 0.0
        for off in self.offsets[:, 1].tolist():
            pot += buoyancy_potential(self.radius, y + off - self.water_level,
                                      self.cfg.water_density, self.cfg.gravity,
                                      self.thickness)
        return 0.5 * self.mass * v * v + self.mass * self.cfg.gravity * y + pot

    # -- time stepping --------------------------------------------------
    def step(self) -> None:
        dt = self.dt
        y, v = self.y, self.v
        a1 = self.acceleration(y, v)
        v_half = v + 0.5 * a1 * dt
        y_new = y + v_half * dt
        a2 = self.acceleration(y_new, v_half)
        v_new = v_half + 0.5 * a2 * dt
        self.y, self.v = y_new, v_new

    def run(self, n_steps: int, record_every: int = 1,
            record_energy: bool = False) -> HeaveTrajectory:
        traj = self.trajectory
        for k in range(n_steps):
            self.step()
            if record_every and k % record_every == 0:
                traj.t.append((k + 1) * self.dt)
                traj.y.append(self.y)
                traj.v.append(self.v)
                if record_energy:
                    traj.energy.append(self.total_energy(self.y, self.v))
        return traj


# ---------------------------------------------------------------------------
# Signal analysis helpers
# ---------------------------------------------------------------------------

def upward_crossing_periods(t: Sequence[float], y: Sequence[float],
                            y_eq: float) -> list[float]:
    """Periods from linear interpolated upward crossings of ``y = y_eq``."""
    crossings: list[float] = []
    for k in range(len(y) - 1):
        a = y[k] - y_eq
        b = y[k + 1] - y_eq
        if a < 0.0 <= b and b != a:
            frac = -a / (b - a)
            crossings.append(t[k] + frac * (t[k + 1] - t[k]))
    return [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]


def mean_period(periods: Sequence[float], skip: int = 1) -> float:
    """Mean period after discarding ``skip`` transient cycles."""
    window = list(periods[skip:]) or list(periods)
    if not window:
        raise ValueError("no periods measured")
    return sum(window) / len(window)


def log_decrement_damping_ratio(t: Sequence[float], y: Sequence[float],
                                y_eq: float, max_cycles: int = 12) -> float:
    """Damping ratio from the log decrement of successive upward peaks.

    ``delta = ln(A_k / A_{k+1})`` per cycle and ``zeta = delta / sqrt(4 pi^2 + delta^2)``.
    A least-squares line through ``ln A_k`` vs cycle index is used so that all
    recorded cycles contribute.
    """
    peaks: list[tuple[float, float]] = []
    for k in range(1, len(y) - 1):
        a = y[k] - y_eq
        if a > 0.0 and a >= (y[k - 1] - y_eq) and a >= (y[k + 1] - y_eq):
            peaks.append((t[k], a))
    peaks = peaks[:max_cycles]
    if len(peaks) < 2:
        raise ValueError("fewer than two peaks recorded")
    xs = list(range(len(peaks)))
    ys = [math.log(p[1]) for p in peaks]
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    slope = sum((xs[i] - xm) * (ys[i] - ym) for i in range(n)) / sum((x - xm) ** 2 for x in xs)
    delta = -slope
    return delta / math.sqrt(4.0 * math.pi ** 2 + delta * delta)


# ---------------------------------------------------------------------------
# Finite-amplitude (nonlinear) heave period: exact energy integral
# ---------------------------------------------------------------------------
#
# The small-amplitude anchor in ``benchmarks/ice_floe_physics.py`` (anchor T)
# compares a measured period with ``2 pi sqrt(m/k)`` at ``2e-3 R``.  That
# estimate is linearised and the validation document recorded the
# finite-amplitude period as unverified.  The functions below close that debt
# without introducing a second physics model: the reference period comes from
# the *exact* energy integral of the same production force kernel, so the
# comparison isolates the amplitude effect rather than a reimplementation.

@dataclass
class HeavePeriodEstimate:
    """Finite-amplitude heave period from the exact energy integral.

    ``T = 2 (I_up + I_dn)`` with ``I = ∫ dy / sqrt(2 ΔU(y))`` taken over the
    two half-branches ``[y_eq, y_max]`` and ``[y_min, y_eq]``, where
    ``ΔU(y) = ∫_y^{y_max} (-a(s)) ds`` is the per-unit-mass energy gained by
    falling from the release height to ``y`` (so ``ΔU(y_max) = 0`` exactly and
    no large potentials are ever subtracted from each other).

    At a turning point ``ΔU`` vanishes *linearly* (the release height is a
    simple root of the energy difference), so the integrand carries an
    integrable ``1/sqrt(y_end - y)`` singularity.  The quadrature removes it by
    the substitution ``y = y_end ∓ A u²``: the resulting integrand
    ``2 A u / sqrt(2 ΔU)`` is finite at ``u -> 0``, so no quadrature node lands
    on a singular value.
    """
    period: float
    amplitude: float
    y_eq: float
    y_min: float
    y_max: float
    energy_drop: float
    n_grid: int
    n_quad: int
    period_linear: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "period": self.period,
            "amplitude": self.amplitude,
            "y_eq": self.y_eq,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "energy_drop": self.energy_drop,
            "n_grid": self.n_grid,
            "n_quad": self.n_quad,
            "period_linear": self.period_linear,
        }


def _simpson_scalar(y_lo: float, y_hi: float, fn, n: int = 256) -> float:
    """Composite Simpson of ``fn`` on ``[y_lo, y_hi]`` (``n`` even, ``n >= 2``)."""
    n = n if n % 2 == 0 else n + 1
    ys = np.linspace(y_lo, y_hi, n + 1)
    vals = np.asarray(fn(ys), dtype=np.float64)
    h = (y_hi - y_lo) / n
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float((weights * vals).sum()) * h / 3.0


def energy_drop(acceleration, y: float, y_end: float, n_slices: int = 256) -> float:
    """Per-unit-mass energy gained falling from ``y_end`` to ``y``.

    ``ΔU(y) = ∫_y^{y_end} (-a(s)) ds``: positive between the two turning
    points, exactly zero at ``y = y_end`` and negative past ``y_min``.
    """
    return _simpson_scalar(float(y), float(y_end),
                           lambda s: -np.asarray(acceleration(s), dtype=np.float64),
                           n_slices)


def finite_amplitude_heave_period(acceleration, y_eq: float, amplitude: float,
                                  n_grid: int = 4001, n_quad: int = 400,
                                  linear_period: float = float("nan"),
                                  span_growth: float = 1.35,
                                  max_bracket: int = 200) -> HeavePeriodEstimate:
    """Period of a conservative 1-DOF heave oscillation released at
    ``y_eq + amplitude`` from rest, from the exact energy integral.

    ``acceleration`` must be a **vectorised** callable mapping an array of
    heights to an array of accelerations (positive up).  It is the *only*
    physics input, so the reference period uses the same force law as the
    solver under test: the comparison verifies the amplitude/integrator
    bookkeeping, not a second model.

    Damping is excluded by construction; callers must pass a drag-free
    configuration.
    """
    if not amplitude > 0.0:
        raise ValueError("amplitude must be positive (release above equilibrium)")
    y_eq = float(y_eq)
    y_max = y_eq + float(amplitude)
    a_release = float(np.asarray(acceleration(np.array([y_max])), dtype=np.float64)[0])
    if not a_release < 0.0:
        raise ValueError("acceleration at the release point is not restoring "
                         f"(a(y_max) = {a_release:g} >= 0)")

    # --- lower turning point: bisection on ΔU(y) = 0 -----------------------
    span = float(amplitude)
    for _ in range(max_bracket):
        lo = y_max - span
        if energy_drop(acceleration, lo, y_max) < 0.0:
            break
        span *= span_growth
    else:  # pragma: no cover - reachable only for a non-confining potential
        raise RuntimeError("no lower turning point found; potential not confining")
    hi = y_max - span / span_growth
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if energy_drop(acceleration, mid, y_max) < 0.0:
            lo = mid
        else:
            hi = mid
    y_min = 0.5 * (lo + hi)

    # --- dense force grid and cumulative energy drop -----------------------
    ys = np.linspace(y_min, y_max, int(n_grid))
    acc = np.asarray(acceleration(ys), dtype=np.float64)
    seg = 0.5 * ((-acc[1:]) + (-acc[:-1])) * (ys[1:] - ys[:-1])
    drop = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    drop = np.maximum(drop, 0.0)

    def drop_at(y):
        return np.interp(y, ys, drop)

    # --- Gauss-Legendre quadrature with the turning-point substitution -----
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    u = 0.5 * (nodes + 1.0)
    wq = 0.5 * weights
    half = 0.0
    for width, anchor, direction in ((y_max - y_eq, y_max, -1.0),
                                     (y_eq - y_min, y_min, +1.0)):
        if width <= 0.0:
            continue
        y = anchor + direction * width * u ** 2
        integrand = 2.0 * width * u / np.sqrt(2.0 * drop_at(y))
        half += float((wq * integrand).sum())

    return HeavePeriodEstimate(period=2.0 * half,
                               amplitude=float(amplitude),
                               y_eq=y_eq, y_min=float(y_min), y_max=float(y_max),
                               energy_drop=energy_drop(acceleration, y_eq, y_max),
                               n_grid=int(n_grid), n_quad=int(n_quad),
                               period_linear=float(linear_period))


def lattice_heave_acceleration(offsets: torch.Tensor, particle_radius: float,
                               thickness: float, density_ice: float,
                               cfg: WetCouplingConfig, chunk: int = 8192):
    """Vectorised vertical acceleration of a rigid lattice floe.

    Drives the production kernel ``compute_buoyancy_force`` (drag excluded), so
    the reference period and :class:`RigidHeaveFloe` see identical per-particle
    submerged fractions.  ``cfg.enabled`` must be true.
    """
    off = offsets.double()
    n_part = int(off.shape[0])
    mass = density_ice * math.pi * particle_radius ** 2 * thickness * n_part
    x_off = off[:, 0].contiguous()
    y_off = off[:, 1].contiguous().numpy()

    def accel(ys):
        ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
        out = np.empty_like(ys)
        step = int(chunk)
        for start in range(0, ys.shape[0], step):
            block = ys[start:start + step]
            heights = block[:, None] + y_off[None, :]
            pos = torch.empty((heights.size, 2), dtype=torch.float64)
            pos[:, 0] = x_off.repeat(block.shape[0])
            pos[:, 1] = torch.from_numpy(heights.reshape(-1))
            force = compute_buoyancy_force(pos, particle_radius, thickness,
                                           density_ice, cfg)
            fy = force[:, 1].reshape(block.shape[0], n_part).sum(dim=1).numpy()
            out[start:start + block.shape[0]] = fy / mass - cfg.gravity
        return out

    return accel


def lattice_finite_amplitude_heave_period(offsets: torch.Tensor,
                                          particle_radius: float,
                                          thickness: float,
                                          density_ice: float,
                                          cfg: WetCouplingConfig,
                                          y_eq: float, amplitude: float,
                                          n_grid: int = 4001,
                                          n_quad: int = 400,
                                          chunk: int = 8192) -> HeavePeriodEstimate:
    """Finite-amplitude heave period of a rigid lattice floe.

    The linear reference ``2 pi sqrt(m / K)`` uses the closed-form discrete
    stiffness :func:`discrete_stiffness` at ``y_eq``, so the returned estimate
    carries both the nonlinear reference and the linearised one; their
    difference *is* the amplitude effect.
    """
    accel = lattice_heave_acceleration(offsets, particle_radius, thickness,
                                       density_ice, cfg, chunk=chunk)
    stiffness = discrete_stiffness(offsets, particle_radius, thickness,
                                   density_ice, cfg, y_eq)
    n_part = int(offsets.shape[0])
    mass = density_ice * math.pi * particle_radius ** 2 * thickness * n_part
    return finite_amplitude_heave_period(
        accel, y_eq, amplitude, n_grid=n_grid, n_quad=n_quad,
        linear_period=analytic_heave_period(mass, stiffness),
    )


__all__ = [
    "HeaveTrajectory",
    "RigidHeaveFloe",
    "analytic_damping_ratio",
    "analytic_heave_period",
    "buoyancy_potential",
    "discrete_buoyancy_potential",
    "discrete_buoyancy_profile",
    "discrete_stiffness",
    "energy_drop",
    "equilibrium_center_height",
    "body_envelope",
    "finite_amplitude_heave_period",
    "HeavePeriodEstimate",
    "hex_lattice_floe",
    "slab_lattice",
    "lattice_finite_amplitude_heave_period",
    "lattice_heave_acceleration",
    "linear_heave_stiffness",
    "log_decrement_damping_ratio",
    "mean_period",
    "submerged_fraction",
    "upward_crossing_periods",
    "waterplane_chord",
]