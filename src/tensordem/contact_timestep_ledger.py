"""Time-integration verification surface for the non-spherical DEM layer.

Why this module exists
----------------------
``tensordem.hertz_impact_time`` already pins the *Hertzian* contact duration of a
single sphere against a rigid platen (``t_c = 2.9427 (5m/4k)^{2/5} v^{-1/5}``) and
``tensordem.contact_restitution`` pins the normal dashpot convention.  What is
still missing is the layer *below* the contact law: the explicit integrator's
own verification surface, i.e. the statements that a reviewer reproduces without
trusting the solver:

* **contact duration of the Hookean (linear-spring) contact** --
  ``t_c = pi sqrt(m_eff / k)`` -- measured with the shipped velocity-Verlet loop
  for three stiffness levels;
* **restitution law** ``e(v_imp)`` of a Hertzian contact with Hunt–Crossley
  damping, scored against the semi-analytic first-order dissipation integral
  (Hunt & Crossley 1975; the same quasi-static energy balance Weber & Stillinger
  1985 use for the ``e = 1 - O(v^{1/5})`` law) -- and against the requirement
  that ``e`` decreases monotonically with impact speed;
* **Cundall & Strack (1979) critical time step** ``dt_cr = 2 sqrt(m_min / k_max)``
  measured by *bisection on the integrator itself* (stable / divergent), not
  quoted;
* **energy ledger**: a 64-sphere gas with no dissipation for ``1e5`` steps
  (drift < 0.1 %) and the same gas with a normal dashpot, where the residual
  ``|dE_mech + W_dissipated| / E0`` must stay below ``1e-6``;
* **negative controls**: ``dt = 1.5 dt_cr`` must diverge, a negative damping
  coefficient must *add* energy, and a torn ledger (dissipation dropped from the
  accounting) must be detected.  These are what make the positive evidence
  non-vacuous.

Contact law conventions (stated once, used everywhere)
-----------------------------------------------------
Normal force for a contact with overlap ``d >= 0`` and closing speed
``v_n = -d(delta)/dt``::

    linear  : F_n = k * d
    hertz   : F_n = k * d^{3/2}                        (k = (4/3) E* sqrt(R*))
    dashpot : F_d = c * v_n                            (linear, clipped to F >= 0)
    H-C     : F_d = (3/2) alpha k d^{1/2} v_n          (Hunt–Crossley)

For a single sphere of mass ``m`` against a rigid wall the *effective* mass is
``m_eff = m``.  ``t_c = pi sqrt(m_eff/k)`` is the half period of the equivalent
harmonic oscillator ``m_eff d'' = -k d``; the mission's
``t_c = pi sqrt(m_eff/(2 k_n))`` is the same statement in the per-body stiffness
convention ``k_n = k/2`` (two equal springs in series), which is spelled out
explicitly by :func:`hookean_contact_time` so no reader has to guess.

All quantities are SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "hookean_contact_time",
    "hertz_contact_time_weber_stillinger",
    "cundall_critical_timestep",
    "hunt_crossley_semi_analytic_restitution",
    "hunt_crossley_fit_exponent",
    "measure_linear_contact_time",
    "measure_restitution_sweep",
    "measure_critical_timestep",
    "SpringOscillator",
    "SphereGas",
    "gas_energy_ledger",
]

# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------
def hookean_contact_time(m_eff: float, stiffness: float, *, per_body_stiffness: bool = False) -> float:
    """Contact duration of a Hookean (linear-spring) impact: ``pi sqrt(m_eff/k)``.

    ``per_body_stiffness=True`` interprets the argument as the *per-body*
    stiffness ``k_n`` of the series pair ``k = 2 k_n``, which turns the formula
    into the ``t_c = pi sqrt(m_eff / (2 k_n))`` form used in the DEM literature.
    """
    k = 2.0 * stiffness if per_body_stiffness else stiffness
    return math.pi * math.sqrt(m_eff / k)


def hertz_contact_time_weber_stillinger(v_in: float, m_eff: float, k_hertz: float) -> float:
    """Hertzian contact duration ``C (5 m_eff / (4 k))^{2/5} v^{-1/5}``, ``C = 2.9427``.

    ``C = 2 int_0^1 du / sqrt(1 - u^{5/2}) = (4/5) B(2/5, 1/2)`` (Johnson 1985,
    Sec. 4.2).  Kept independent of :mod:`tensordem.hertz_impact_time` so the two
    implementations cross-check each other.
    """
    if v_in <= 0.0:
        raise ValueError("v_in must be positive")
    integral = 2.0 * (2.0 / 5.0) * _beta(2.0 / 5.0, 0.5)
    return integral * (5.0 * m_eff / (4.0 * k_hertz)) ** 0.4 * v_in ** -0.2


def _beta(a: float, b: float) -> float:
    from math import gamma

    return gamma(a) * gamma(b) / gamma(a + b)


def cundall_critical_timestep(m_min: float, k_max: float) -> float:
    """Cundall & Strack (1979) critical step ``2 sqrt(m_min / k_max)``.

    Equal to the marginal velocity-Verlet step ``2 / omega_max`` of the stiffest
    oscillator present in the system.
    """
    return 2.0 * math.sqrt(m_min / k_max)


def _hunt_crossley_integral() -> float:
    """``J = int_0^1 u^{1/2} (1 - u^{5/2})^{1/2} du`` (Gauss–Legendre, 96 nodes)."""
    nodes, weights = np.polynomial.legendre.leggauss(96)
    u = 0.5 * (nodes + 1.0)
    return float(0.5 * np.sum(weights * np.sqrt(u) * np.sqrt(1.0 - u ** 2.5)))


def hunt_crossley_semi_analytic_restitution(v_in: float, m_eff: float, k_hertz: float,
                                            alpha: float) -> float:
    """Semi-analytic restitution of a Hunt–Crossley contact.

    First-order (quasi-static) dissipation balance: the dashpot work over one
    impact is ``W = 2 * (3/2) alpha k int_0^{d_max} d^{1/2} |d'| dd`` and the
    coefficient of restitution follows from ``e = sqrt(1 - W/E0)``.  With the
    elastic trajectory ``d(t)`` this integral is analytic up to the pure number
    ``J``:

        ``W/E0 = 3 alpha d_max^{3/4} sqrt(0.8 k / m_eff) J``,
        ``d_max = (5 m_eff v^2 / (4 k))^{2/5}``.

    The formula is first order in ``alpha``; the numerically measured ``e``
    includes the ``O(alpha^2)`` correction, which is why the acceptance band is
    5 % rather than machine precision.
    """
    if alpha <= 0.0:
        return 1.0
    d_max = (5.0 * m_eff * v_in ** 2 / (4.0 * k_hertz)) ** 0.4
    energy = 0.5 * m_eff * v_in ** 2
    work = (3.0 * alpha * k_hertz * d_max ** 1.5
            * math.sqrt(2.0 * energy / m_eff) * _hunt_crossley_integral())
    return math.sqrt(max(0.0, 1.0 - work / energy))


def hunt_crossley_fit_exponent(v_imp: np.ndarray, restitution: np.ndarray) -> tuple[float, float]:
    """Least-squares fit ``e = 1 - A v^p``; returns ``(A, p)`` and the R^2 in ``p``.

    The exponent is the ``v^{1/5}`` signature that distinguishes a Hertzian
    contact from a linear one (which would give ``e = const`` for a fixed
    dashpot ratio).  ``R^2`` is computed on the fitted curve.
    """
    v = np.asarray(v_imp, dtype=float)
    e = np.asarray(restitution, dtype=float)
    loss = 1.0 - e
    # log-log fit on the loss, avoiding the e == 1 point
    mask = loss > 1e-12
    slope, intercept = np.polyfit(np.log(v[mask]), np.log(loss[mask]), 1)
    a = math.exp(intercept)
    model = 1.0 - a * v ** slope
    residual = float(np.sum((e - model) ** 2))
    total = float(np.sum((e - e.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else 1.0
    return a, float(slope), r2


# ---------------------------------------------------------------------------
# 1-DOF spring oscillator (the Cundall-Strack probe)
# ---------------------------------------------------------------------------
@dataclass
class SpringOscillator:
    """Bilateral linear spring between a mass and a fixed wall.

    Bilateral (the force acts for both signs of the overlap) so that the
    configuration is the textbook harmonic oscillator whose velocity-Verlet
    stability limit is *exactly* ``dt = 2/omega`` -- that is the case in which the
    Cundall-Strack bound is sharp, and therefore the case that can be measured
    without a systematic bias.
    """

    mass: float = 1.0
    stiffness: float = 100.0
    displacement: float = 0.05
    velocity: float = 0.0

    def omega(self) -> float:
        return math.sqrt(self.stiffness / self.mass)

    def energy(self) -> float:
        return 0.5 * self.mass * self.velocity ** 2 + 0.5 * self.stiffness * self.displacement ** 2

    def step(self, dt: float) -> None:
        """Velocity Verlet (kick-drift-kick), the integrator used by the gas."""
        accel = -self.stiffness / self.mass * self.displacement
        self.velocity += 0.5 * dt * accel
        self.displacement += dt * self.velocity
        accel_new = -self.stiffness / self.mass * self.displacement
        self.velocity += 0.5 * dt * accel_new

    def run(self, dt: float, steps: int) -> float:
        """Return the bounded growth factor of the amplitude over ``steps`` steps.

        ``inf`` is returned as soon as the trajectory leaves the representable
        range (the divergent branch), which keeps the bisection decisive.
        """
        amplitude = abs(self.displacement)
        largest = amplitude
        for _ in range(steps):
            self.step(dt)
            if not math.isfinite(self.displacement):
                return math.inf
            largest = max(largest, abs(self.displacement))
            if largest > 1.0e12 * amplitude:
                return math.inf
        return largest / amplitude


def measure_critical_timestep(mass: float = 1.0, stiffness: float = 100.0,
                              *, steps: int = 400, growth_threshold: float = 4.0,
                              rtol: float = 1.0e-3, max_iter: int = 60) -> dict:
    """Bisect the velocity-Verlet step where the oscillator stops being bounded.

    Returns a dict with the measured threshold, the theory value, their ratio and
    the bracket that bounds the measurement.
    """
    theory = cundall_critical_timestep(mass, stiffness)

    def stable(dt: float) -> bool:
        probe = SpringOscillator(mass=mass, stiffness=stiffness,
                                 displacement=0.05, velocity=0.0)
        growth = probe.run(dt, steps)
        return growth < growth_threshold

    low = 0.25 * theory           # safely stable
    high = 4.0 * theory           # safely divergent
    assert stable(low), "lower bracket must be stable"
    assert not stable(high), "upper bracket must diverge"
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if stable(mid):
            low = mid
        else:
            high = mid
        if high - low <= rtol * theory:
            break
    measured = 0.5 * (low + high)
    return {"measured_dt_cr": measured, "theory_dt_cr": theory,
            "ratio": measured / theory, "stable_bracket": low,
            "divergent_bracket": high, "steps": steps,
            "growth_threshold": growth_threshold}


# ---------------------------------------------------------------------------
# Single-sphere impact against a wall (contact time + restitution)
# ---------------------------------------------------------------------------
class WallImpact:
    """Normal impact of one sphere on a rigid wall (scalar overlap dynamics).

    The state is the penetration ``delta`` (positive inside the wall) and its rate
    ``delta_dot``; the equation of motion is ``m delta'' = -F(delta, delta')`` with
    the contact force resisting *approach*, and the integration is the same
    velocity-Verlet step used by :class:`SphereGas`, so the timing evidence
    transfers to the many-body integrator.
    """

    def __init__(self, mass: float, stiffness: float, law: str = "linear",
                 damping: float = 0.0) -> None:
        if law not in ("linear", "hertz"):
            raise ValueError("law must be 'linear' or 'hertz'")
        self.mass = float(mass)
        self.stiffness = float(stiffness)
        self.law = law
        self.damping = float(damping)
        self.overlap = 0.0
        self.rate = 0.0

    def elastic(self, overlap: float) -> float:
        if overlap <= 0.0:
            return 0.0
        if self.law == "linear":
            return self.stiffness * overlap
        return self.stiffness * overlap ** 1.5

    def dashpot(self, overlap: float, rate: float) -> float:
        """Hunt–Crossley dashpot ``(3/2) alpha k delta^{1/2} delta'`` (or linear)."""
        if overlap <= 0.0 or self.damping == 0.0:
            return 0.0
        if self.law == "linear":
            return self.damping * rate
        return 1.5 * self.damping * self.stiffness * math.sqrt(overlap) * rate

    def potential(self, overlap: float) -> float:
        if overlap <= 0.0:
            return 0.0
        if self.law == "linear":
            return 0.5 * self.stiffness * overlap ** 2
        return 0.4 * self.stiffness * overlap ** 2.5

    def force(self, overlap: float, rate: float) -> float:
        """Contact force along ``delta`` (<= 0: it always pushes the bodies apart)."""
        return min(0.0, -(self.elastic(overlap) + self.dashpot(overlap, rate)))

    def energy(self) -> float:
        return 0.5 * self.mass * self.rate ** 2 + self.potential(self.overlap)

    def step(self, dt: float) -> None:
        accel = self.force(self.overlap, self.rate) / self.mass
        rate_half = self.rate + 0.5 * dt * accel
        overlap_new = self.overlap + dt * rate_half
        accel_new = self.force(overlap_new, rate_half) / self.mass
        self.rate = rate_half + 0.5 * dt * accel_new
        self.overlap = overlap_new


def measure_linear_contact_time(v_in: float, mass: float, stiffness: float,
                                *, law: str = "linear", samples_per_contact: int = 2000,
                                max_steps: int = 4_000_000) -> dict:
    """Measure the contact duration of one impact with a fixed ``dt`` scan.

    The trajectory is integrated with ``dt = t_c_analytic / samples_per_contact``
    and the duration is taken between the two linear-interpolated zero crossings of
    the overlap, so the measurement is unbiased to ``O(dt^3)``.
    """
    if law == "linear":
        theory = hookean_contact_time(mass, stiffness)
    else:
        theory = hertz_contact_time_weber_stillinger(v_in, mass, stiffness)
    dt = theory / samples_per_contact
    impact = WallImpact(mass=mass, stiffness=stiffness, law=law)
    impact.rate = abs(float(v_in))
    t = 0.0
    t_entry = None
    t_exit = None
    steps = 0
    while steps < max_steps:
        overlap_old, rate_old = impact.overlap, impact.rate
        impact.step(dt)
        t += dt
        steps += 1
        if overlap_old <= 0.0 < impact.overlap and t_entry is None:
            fraction = -overlap_old / (impact.overlap - overlap_old)
            t_entry = t - dt + fraction * dt
        if impact.overlap <= 0.0 < overlap_old and t_entry is not None:
            fraction = (overlap_old - 0.0) / (overlap_old - impact.overlap)
            t_exit = t - dt + fraction * dt
            break
    if t_entry is None or t_exit is None:
        raise RuntimeError("impact was not bracketed; increase max_steps")
    measured = t_exit - t_entry
    return {"measured_tc": measured, "analytic_tc": theory,
            "relative_error": (measured - theory) / theory,
            "dt": dt, "steps": steps, "v_in": v_in, "law": law,
            "stiffness": stiffness, "mass": mass}


def measure_restitution_sweep(v_imp: np.ndarray, mass: float, k_hertz: float,
                              alpha: float, *, samples_per_contact: int = 1500,
                              max_steps: int = 4_000_000) -> dict:
    """Impact-restitution sweep of a Hunt–Crossley Hertzian contact."""
    results = []
    for v in np.asarray(v_imp, dtype=float):
        tc = hertz_contact_time_weber_stillinger(v, mass, k_hertz)
        dt = tc / samples_per_contact
        impact = WallImpact(mass=mass, stiffness=k_hertz, law="hertz", damping=alpha)
        impact.rate = float(v)
        steps = 0
        while steps < max_steps:
            overlap_old = impact.overlap
            impact.step(dt)
            steps += 1
            if impact.overlap <= 0.0 < overlap_old and overlap_old > 0.0:
                break
        else:
            raise RuntimeError("impact did not rebound; check the damping range")
        e_measured = abs(impact.rate) / float(v)
        results.append({"v_imp": float(v), "v_out": abs(impact.rate), "e": e_measured,
                        "e_semi_analytic": hunt_crossley_semi_analytic_restitution(
                            float(v), mass, k_hertz, alpha)})
    v = np.array([r["v_imp"] for r in results])
    e = np.array([r["e"] for r in results])
    semi = np.array([r["e_semi_analytic"] for r in results])
    a, exponent, r2 = hunt_crossley_fit_exponent(v, e)
    return {"points": results, "fit_amplitude": a, "fit_exponent": exponent,
            "fit_r2": r2, "monotone": bool(np.all(np.diff(e) < 0.0)),
            "max_relative_deviation": float(np.max(np.abs(e - semi) / semi))}


# ---------------------------------------------------------------------------
# N-sphere gas with an exact mechanical-energy ledger
# ---------------------------------------------------------------------------
@dataclass
class SphereGas:
    """Dilute N-sphere gas in an axis-aligned box, velocity-Verlet.

    The ledger is exact by construction: the elastic energy of every contact and
    the body-force potential are evaluated from the *same* state that produces the
    forces, and the dashpot work is accumulated as the trapezoidal integral of
    ``F_d`` over the overlap ``delta`` of each contact, so
    ``dE_mech + W_diss`` must vanish to the integrator's truncation error.
    """

    positions: np.ndarray
    velocities: np.ndarray
    radius: float
    mass: float
    stiffness: float
    box: np.ndarray
    law: str = "linear"
    damping: float = 0.0
    gravity: float = 0.0
    batch_overlap: np.ndarray = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        self.box = np.asarray(self.box, dtype=float)
        n = self.positions.shape[0]
        self._iu, self._ju = np.triu_indices(n, k=1)
        normals = np.zeros((1, 6, 3))
        for axis in range(3):
            normals[0, axis, axis] = 1.0
            normals[0, axis + 3, axis] = -1.0
        self._wall_normals = np.broadcast_to(normals, (n, 6, 3))

    # -- forces ---------------------------------------------------------
    def _wall_geometry(self, positions):
        """Overlaps and outward normals for the six box faces, shape ``(N, 6)``/``(N, 6, 3)``."""
        lo = self.box[0] - positions + self.radius
        hi = positions - (self.box[1] - self.radius)
        overlaps = np.concatenate([lo, hi], axis=1)
        return self._wall_normals, overlaps

    def forces(self, positions, velocities):
        """Return (force, elastic_energy, dashpot_force_array, overlap_array)."""
        delta = positions[self._ju] - positions[self._iu]
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        inv = np.divide(1.0, dist, out=np.zeros_like(dist), where=dist > 0.0)
        normal = delta * inv[:, None]
        overlap = 2.0 * self.radius - dist
        positive = np.clip(overlap, 0.0, None)
        if self.law == "linear":
            elastic = self.stiffness * positive
        else:
            elastic = self.stiffness * positive ** 1.5
        closing = np.zeros_like(dist)
        if self.damping:
            closing = -np.einsum("ij,ij->i", velocities[self._ju] - velocities[self._iu], normal)
            closing = np.clip(closing, 0.0, None) * (overlap > 0.0)
        magnitude = elastic + self.damping * closing
        pair_force = magnitude[:, None] * normal
        force = np.empty_like(positions)
        for axis in range(3):
            force[:, axis] = (np.bincount(self._iu, weights=-pair_force[:, axis],
                                          minlength=positions.shape[0])
                              + np.bincount(self._ju, weights=pair_force[:, axis],
                                            minlength=positions.shape[0]))

        wall_overlap = np.concatenate([self.box[0] - positions + self.radius,
                                       positions - (self.box[1] - self.radius)], axis=1)
        wall_positive = np.clip(wall_overlap, 0.0, None)
        if self.law == "linear":
            wall_elastic = self.stiffness * wall_positive
        else:
            wall_elastic = self.stiffness * wall_positive ** 1.5
        wall_closing = np.zeros_like(wall_overlap)
        if self.damping:
            wall_closing = -np.sum(self._wall_normals * velocities[:, None, :], axis=2)
            wall_closing = np.clip(wall_closing, 0.0, None) * (wall_overlap > 0.0)
        wall_magnitude = wall_elastic + self.damping * wall_closing
        force += np.sum(self._wall_normals * wall_magnitude[:, :, None], axis=1)

        if self.law == "linear":
            elastic_energy = 0.5 * self.stiffness * (float(np.sum(positive ** 2))
                                                     + float(np.sum(wall_positive ** 2)))
        else:
            elastic_energy = 0.4 * self.stiffness * (float(np.sum(positive ** 2.5))
                                                     + float(np.sum(wall_positive ** 2.5)))
        if self.gravity:
            force[:, 2] -= self.mass * self.gravity
        return (force, elastic_energy,
                np.concatenate([self.damping * closing, (self.damping * wall_closing).ravel()]),
                np.concatenate([positive, wall_positive.ravel()]))

    # -- energy ---------------------------------------------------------
    def energy(self, positions=None, velocities=None):
        positions = self.positions if positions is None else positions
        velocities = self.velocities if velocities is None else velocities
        kinetic = 0.5 * self.mass * float(np.sum(velocities ** 2))
        _, elastic, _, _ = self.forces(positions, velocities)
        potential = float(elastic)
        if self.gravity:
            potential += self.mass * self.gravity * float(np.sum(positions[:, 2]))
        return kinetic + potential

    # -- integrator -----------------------------------------------------
    def step(self, dt: float):
        """One velocity-Verlet step; returns the dissipated work of the step."""
        force, _, dashpot, overlap = self.forces(self.positions, self.velocities)
        velocity_half = self.velocities + 0.5 * dt * force / self.mass
        positions_new = self.positions + dt * velocity_half
        force_new, _, dashpot_new, overlap_new = self.forces(positions_new, velocity_half)
        velocities_new = velocity_half + 0.5 * dt * force_new / self.mass
        # Dashpot work: trapezoidal integral of F_d over the overlap change.
        dissipated = float(np.sum(0.5 * (dashpot + dashpot_new) * (overlap_new - overlap)))
        self.positions = positions_new
        self.velocities = velocities_new
        return dissipated


def _initial_gas(count: int, radius: float, mass: float, box: np.ndarray,
                 stiffness: float, *, law: str, damping: float, gravity: float,
                 speed: float, seed: int) -> SphereGas:
    rng = np.random.default_rng(seed)
    lo = box[0] + radius
    hi = box[1] - radius
    positions = lo + (hi - lo) * rng.random((count, 3))
    velocities = rng.normal(size=(count, 3))
    velocities *= speed / np.sqrt(np.mean(np.sum(velocities ** 2, axis=1)))
    return SphereGas(positions=positions, velocities=velocities, radius=radius,
                     mass=mass, stiffness=stiffness, box=box, law=law,
                     damping=damping, gravity=gravity)


def gas_energy_ledger(count: int = 64, steps: int = 100_000, *, radius: float = 0.05,
                      mass: float = 1.0, stiffness: float = 2.0e4,
                      box_lo=(-0.6, -0.6, -0.6), box_hi=(0.6, 0.6, 0.6),
                      law: str = "linear", damping: float = 0.0,
                      gravity: float = 0.0, speed: float = 0.4,
                      safety: float = 0.05, seed: int = 20261106,
                      check_every: int = 1000) -> dict:
    """Integrate the gas and return the energy ledger summary.

    ``safety`` is ``dt / dt_cr`` with ``dt_cr = 2 sqrt(m_min/k_max)``.
    """
    box = np.array([box_lo, box_hi], dtype=float)
    gas = _initial_gas(count, radius, mass, box, stiffness, law=law,
                       damping=damping, gravity=gravity, speed=speed, seed=seed)
    dt = safety * cundall_critical_timestep(mass, stiffness)
    e0 = gas.energy()
    dissipated = 0.0
    series = []
    max_drift = 0.0
    for k in range(steps):
        dissipated += gas.step(dt)
        if (k + 1) % check_every == 0 or k == steps - 1:
            e = gas.energy()
            drift = (e - e0 + dissipated) / e0
            series.append({"step": k + 1, "time": (k + 1) * dt, "energy": e,
                           "dissipated": dissipated, "residual": drift})
            max_drift = max(max_drift, abs(drift))
    final = series[-1]
    return {"count": count, "steps": steps, "dt": dt, "dt_over_cr": safety,
            "energy_initial": e0, "energy_final": final["energy"],
            "dissipated": dissipated,
            "energy_drift_relative": (final["energy"] - e0) / e0,
            "ledger_residual_relative": final["residual"],
            "max_abs_residual_relative": max_drift, "series": series}