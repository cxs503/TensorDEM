"""Tilted-plane Coulomb friction benchmark surface (closed-form anchor).

Physics and why this module exists
----------------------------------
A body resting on a plane inclined at ``theta`` is held by static friction iff

    tan(theta) <= mu_s          (Coulomb sliding limit, theta_c = atan(mu_s))

and, once sliding, its down-slope acceleration obeys

    a = g (sin(theta) - mu_k cos(theta)).

Both are closed forms, so the two coefficients can be measured *independently*
from the same DEM configuration:

* **static branch** -- bisect the tilt angle for first motion and invert
  ``mu_s = tan(theta_c)``;
* **dynamic branch** -- integrate a steady slide and invert
  ``mu_k = tan(theta) - a / (g cos(theta))`` from the measured acceleration.

Reporting both is what makes the pair *distinguishable*: with a single
coefficient the two routes must agree (``theta_s == atan(mu_k)``), while a
static/dynamic pair (``mu_s != mu_k``) must separate by
``atan(mu_s) - atan(mu_k)``.  The static/dynamic pair is carried by the contact
law itself (``FrictionPolyhedronConfig.mu_dynamic``, see
:func:`tensordem.friction_polyhedron.friction_cone_limit`), not by switching
coefficients between runs.

Configuration and the rolling trap
----------------------------------
Two configurations are used and they are *not* interchangeable:

* ``rotation_locked=True`` -- the rotational degree of freedom is frozen, so the
  contact carries a pure translational Coulomb cap.  This is a numerical
  isolation experiment (documented in
  :mod:`tensordem.friction_polyhedron_real_verification`); it is the
  configuration whose measurement the closed form predicts.
Initial state (why it matters)
------------------------------
The block is not dropped: it is placed in *static equilibrium* and its tangential
spring is pre-loaded to ``m g sin(theta) / k_t``, so from step one the contact
carries ``m g sin(theta)`` up-slope and ``m g cos(theta)`` through the wall.  A
free-fall gap or a zero spring would make the onset angle a measurement of the
start-up transient instead of the friction limit: the Cundall-Strack spring is
elastic, so the first oscillation overshoots the static cone and the block trips
into the residual cone at any angle above ``atan(mu_k)``.  The micro-disturbance
required by the benchmark is a fixed down-slope velocity ``kick_speed = 1e-6
m/s``; its kinetic energy is far below the static barrier outside a ~0.02 deg
band around the limit, and the test suite checks that the bracket does not move
when the kick is varied by an order of magnitude.

* free rotation -- a *sphere* rolls without slipping below
  ``tan(theta) = 3.5 mu`` (static friction needed for rolling: f = (2/7) m g
  sin(theta)), so the naive inversion returns the apparent coefficient
  ``mu_app = (2/7) tan(theta)`` instead of ``mu``.  The ``sphere`` tasks measure
  that trap explicitly rather than hiding it; a freely rotating flat-faced cube
  additionally creeps below ``atan(mu)`` because a collapsed face contact cannot
  shift its centre of pressure (measured defect, same reference module).

Predeclared thresholds (written verbatim into ``results.json``)
--------------------------------------------------------------
* ``REL_ERR_LIMIT_PCT = 5.0`` -- static inversion ``mu_measured`` vs ``mu_set``
  for ``mu in {0.2, 0.4, 0.6}``; the acceptance gate of this benchmark.
* ``REPORT_STRICT_PCT = 3.0`` -- the stricter level the report must answer
  explicitly (no assertion is relaxed if it is missed; the miss is reported).
* ``DUAL_MIN_SEPARATION_DEG = 2.0`` -- required static-vs-dynamic friction-angle
  separation in the dual-coefficient case ``(mu_s, mu_k) = (0.6, 0.4)``, whose
  analytic separation is ``atan(0.6) - atan(0.4) = 9.16 deg``.
* ``DT_LEVELS = (0, 1, 2)`` -- time step refinement ``dt, dt/2, dt/4`` with
  ``dt = dt_safety * 2 pi sqrt(m / k_max)`` (``dt_safety = 0.01``, i.e. 100
  steps per contact spring period); the convergence gate is that the absolute
  angle error does not grow as the step is refined.
* ``slide_tolerance = 0.01 * particle extent`` (1 % of the cube edge / sphere
  diameter) -- the "first motion" criterion.  Rationale: the elastic stick
  displacement at the onset of sliding is ``mu_s m g cos(theta) / k_t`` ~ 1e-6 m,
  two orders of magnitude below the tolerance, so the criterion cannot be met by
  elastic deformation alone.

Determinism: no random numbers are drawn anywhere (the "micro disturbance" is a
fixed down-slope kick ``kick_speed``); results are assembled in sorted task order
and carry a sha256 over the canonical JSON payload, so the artifact is
reproducible from a clean checkout with one command (see ``repro_cmd``).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .friction_polyhedron import (
    ContactState,
    FrictionPolyhedronConfig,
    PolyhedronDEMSystem,
    PolyhedronParticle,
    compute_wall_contact_force,
    regular_polyhedron,
    wall_penetration,
)

__all__ = [
    "MU_TARGETS",
    "ROLLING_SPRING_DEFAULT",
    "SPHERE_INERTIA_RATIO",
    "REL_ERR_LIMIT_PCT",
    "REPORT_STRICT_PCT",
    "DUAL_MIN_SEPARATION_DEG",
    "DT_LEVELS",
    "TiltPlaneCase",
    "rolling_acceleration_analytic",
    "plane_basis",
    "build_case",
    "simulate",
    "critical_angle",
    "dynamic_friction",
    "campaign_tasks",
    "run_task",
    "run_campaign",
    "assemble_results",
    "results_sha256",
    "write_results",
    "render_figures",
    "main",
]

# --- predeclared acceptance thresholds -------------------------------------
MU_TARGETS: Tuple[float, ...] = (0.2, 0.4, 0.6)
REL_ERR_LIMIT_PCT = 5.0
REPORT_STRICT_PCT = 3.0
DUAL_MIN_SEPARATION_DEG = 2.0
DUAL_CASE: Tuple[float, float] = (0.6, 0.4)          # (mu_s, mu_k)
DT_LEVELS: Tuple[int, ...] = (0, 1, 2)
SLIDE_TOLERANCE_FRACTION = 0.01                      # of the particle extent
KICK_SPEED = 1.0e-6                                  # m/s, deterministic
DT_SAFETY = 0.01                                     # 100 steps per spring period
SPHERE_ROLLING_FACTOR = 3.5                          # tan(theta) = 3.5 mu rolling limit
GRAVITY = 9.81
#: ``I / (m R^2)`` of a solid sphere -- the inertia ratio that turns the
#: translational balance into ``a = g(sin - mu_r cos) / (1 + 2/5)``.
SPHERE_INERTIA_RATIO = 2.0 / 5.0
#: Elastic rolling-spring stiffness (N m/rad) used when ``mu_rolling > 0`` and
#: the case does not set ``k_r``.  The EPSD moment is capped at
#: ``mu_r R |F_n|``, so ``k_r`` only shapes the transient (it never changes the
#: steady rolling-resistance torque); 1e2 N m/rad is the
#: :class:`FrictionPolyhedronConfig` default.
ROLLING_SPRING_DEFAULT = 1.0e2


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TiltPlaneCase:
    """One tilted-plane configuration (contact law + body + time step).

    Attributes:
        mu_s: static Coulomb coefficient (``FrictionPolyhedronConfig.mu``).
        mu_k: dynamic coefficient; ``None`` -> single-coefficient cone.
        size: cube half-edge / sphere radius (m).
        density: particle density (kg/m^3).
        k_n: normal stiffness (N/m); ``k_t`` tangential stiffness (N/m).
        gamma_n: normal dashpot (N s/m).
        gravity: gravitational acceleration (m/s^2).
        shape: ``"cube"`` (face-flat polyhedron) or ``"sphere"``.
        raft: ``n`` -> ``n x n`` layer of bodies sharing the plane (1 = one body).
        raft_gap: relative gap between layer neighbours (0 -> touching).  A
            non-zero gap keeps the bodies out of particle-particle contact: the
            current GJK/EPA pair kernel spends ~17 ms per *penetrating* pair
            (measured), which makes a touching layer unaffordable, so the layer
            task verifies load sharing across ``n^2`` simultaneous wall contacts
            without inter-body coupling (stated as a limitation).
        rotation_locked: freeze the rotational degree of freedom (isolation run).
        t_end: simulated time per run (s).
        dt_level: ``0,1,2`` -> ``dt, dt/2, dt/4``.
        dt_safety: ``dt = dt_safety * 2 pi sqrt(m / k_max)``.
        kick_speed: deterministic down-slope initial speed (m/s).
        samples: number of trajectory samples recorded per run.
    """

    mu_s: float
    mu_k: Optional[float] = None
    size: float = 0.006
    density: float = 1500.0
    k_n: float = 1.0e5
    k_t: float = 1.0e5
    gamma_n: float = 20.0
    gravity: float = GRAVITY
    shape: str = "cube"
    raft: int = 1
    raft_gap: float = 0.0
    rotation_locked: bool = True
    mu_rolling: float = 0.0
    """Rolling-resistance coefficient ``mu_r`` (dimensionless, EPSD/Ai et al.).

    ``0.0`` (default) keeps the historic run bit-for-bit: the wall law is
    configured with ``mu_r = k_r = 0`` and the rolling moment is identically
    zero.  A positive value switches on the capped rolling moment
    ``M_r = min(k_r * integral(omega_roll dt), mu_r * R * F_n)`` in the
    production wall contact law (:func:`tensordem.friction_polyhedron.compute_wall_contact_force`).
    Only meaningful for ``shape="sphere"``: the closed-form anchor
    :func:`rolling_acceleration_analytic` is a *rolling* solution.
    """
    k_r: Optional[float] = None
    """Elastic rolling-spring stiffness (N m/rad); ``None`` -> 1e2 when
    ``mu_rolling > 0``, ``0`` (moment disabled) otherwise."""
    t_end: float = 0.30
    dt_level: int = 0
    dt_safety: float = DT_SAFETY
    kick_speed: float = KICK_SPEED
    samples: int = 200

    # -- derived quantities -------------------------------------------------
    @property
    def mu_dynamic(self) -> float:
        """Coefficient actually used once sliding."""
        return self.mu_s if self.mu_k is None else float(self.mu_k)

    @property
    def effective_k_r(self) -> float:
        """Rolling-spring stiffness actually handed to the contact law."""
        if self.k_r is not None:
            return float(self.k_r)
        return ROLLING_SPRING_DEFAULT if self.mu_rolling > 0.0 else 0.0

    @property
    def rolling_active(self) -> bool:
        return self.mu_rolling > 0.0 and self.effective_k_r > 0.0

    @property
    def particle_extent(self) -> float:
        """Cube edge / sphere diameter (m)."""
        return 2.0 * self.size

    @property
    def particle_volume(self) -> float:
        if self.shape == "sphere":
            return 4.0 / 3.0 * math.pi * self.size ** 3
        return (2.0 * self.size) ** 3

    @property
    def particle_mass(self) -> float:
        return self.density * self.particle_volume

    @property
    def spring_period(self) -> float:
        """Period of the stiffest contact spring (s)."""
        k_max = max(self.k_n, self.k_t)
        return 2.0 * math.pi * math.sqrt(self.particle_mass / k_max)

    @property
    def dt(self) -> float:
        return self.dt_safety * self.spring_period / (2 ** int(self.dt_level))

    @property
    def slide_tolerance(self) -> float:
        return SLIDE_TOLERANCE_FRACTION * self.particle_extent

    def resolved(self) -> Dict[str, Any]:
        """JSON-ready case record (adds the derived numbers used by the run)."""
        rec = asdict(self)
        rec.update({
            "mu_dynamic": self.mu_dynamic,
            "effective_k_r": self.effective_k_r,
            "rolling_active": self.rolling_active,
            "particle_mass": self.particle_mass,
            "particle_extent": self.particle_extent,
            "spring_period": self.spring_period,
            "dt": self.dt,
            "slide_tolerance": self.slide_tolerance,
        })
        return rec


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def plane_basis(theta_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Plane normal (into the domain) and down-slope direction for an incline.

    The plane is ``n . x = 0`` with ``n = (-sin t, 0, cos t)``: it rises with
    ``+x``, so downhill is ``(-cos t, 0, -sin t)``.
    """
    th = math.radians(float(theta_deg))
    normal = np.array([-math.sin(th), 0.0, math.cos(th)])
    downslope = np.array([-math.cos(th), 0.0, -math.sin(th)])
    return normal, downslope


def _rotation_y(theta_deg: float) -> np.ndarray:
    """Rotation about +y that maps the local +z axis onto ``plane_basis`` n."""
    a = math.radians(float(theta_deg))
    return np.array([
        [math.cos(a), 0.0, math.sin(a)],
        [0.0, 1.0, 0.0],
        [-math.sin(a), 0.0, math.cos(a)],
    ])


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------
def build_case(case: TiltPlaneCase, theta_deg: float
               ) -> Tuple[PolyhedronDEMSystem, List[PolyhedronParticle],
                          np.ndarray, np.ndarray]:
    """Instantiate the DEM system for one tilt angle.

    A cube is oriented face-parallel to the plane (a face, not a vertex, touches
    the wall) and placed ``1e-7 m`` above it, so the first step is a resting
    contact rather than an impact.  An ``n x n`` layer is tiled in the plane and
    shares the wall, so ``n^2`` wall contacts carry the weight.
    """
    normal, downslope = plane_basis(theta_deg)
    config = FrictionPolyhedronConfig(
        k_n=case.k_n, gamma_n=case.gamma_n, k_t=case.k_t,
        mu=case.mu_s, mu_dynamic=case.mu_k,
        mu_r=case.mu_rolling, k_r=case.effective_k_r,
        density=case.density,
    )
    eps = 1.0e-9
    th = math.radians(float(theta_deg))
    # Static equilibrium: the wall carries the normal load through the penalty
    # overlap m g cos(theta) / k_n, and the tangential spring is pre-loaded to
    # m g sin(theta) / k_t (see the module docstring: a free-fall start or an
    # empty tangential spring turns the anchor into a start-up transient test).
    overlap_eq = case.particle_mass * case.gravity * math.cos(th) / case.k_n
    delta_t_eq = (case.particle_mass * case.gravity * math.sin(th) / case.k_t)
    rotation = _rotation_y(-float(theta_deg))
    transverse = np.array([0.0, 1.0, 0.0])
    n_grid = int(case.raft)
    pitch = 2.0 * case.size * (1.0 + float(case.raft_gap))
    particles: List[PolyhedronParticle] = []
    for i in range(n_grid):
        for j in range(n_grid):
            offset = ((i - 0.5 * (n_grid - 1)) * pitch * downslope
                      + (j - 0.5 * (n_grid - 1)) * pitch * transverse)
            center = normal * (case.size - overlap_eq + eps) + offset
            if case.shape == "cube":
                local = regular_polyhedron("cube", np.zeros(3), case.size)
                verts = local @ rotation.T + center
                particle = PolyhedronParticle.from_polytope(
                    verts, density=case.density, position=center,
                    label=f"tilt_cube_{i}{j}")
            elif case.shape == "sphere":
                particle = PolyhedronParticle.from_sphere(
                    case.size, density=case.density, position=center,
                    label=f"tilt_sphere_{i}{j}")
            else:  # pragma: no cover - guarded by the caller
                raise ValueError(f"unknown shape {case.shape!r}")
            if case.rotation_locked:
                # Frozen rotation: a numerical isolation experiment (see the
                # module docstring).  Same device as
                # friction_polyhedron_real_verification.build_incline_case.
                particle.inertia = np.eye(3) * 1.0e12
                particle.omega = np.zeros(3)
            particle.v = case.kick_speed * downslope
            particles.append(particle)
    system = PolyhedronDEMSystem(
        particles, config, gravity=np.array([0.0, 0.0, -case.gravity]))
    system.add_wall(normal, 0.0)
    for idx in range(len(particles)):
        system.wall_states[(idx, 0)] = ContactState(
            delta_t=delta_t_eq * downslope.copy())
    return system, particles, normal, downslope

# ---------------------------------------------------------------------------
# Rolling-resistance anchor (closed form)
# ---------------------------------------------------------------------------
def rolling_acceleration_analytic(case: TiltPlaneCase,
                                  theta_deg: float) -> Dict[str, float]:
    """Down-slope acceleration of a *rolling* sphere with rolling resistance.

    Moment balance about the contact with the no-slip constraint ``a = alpha R``
    and the capped EPSD moment ``M_r = mu_r R F_n`` (``F_n = m g cos(theta)``)::

        m g sin - F_s = m a
        F_s R - mu_r R m g cos = I a / R          ->  a = g (sin - mu_r cos) / (1 + I/mR^2)

    For a solid sphere (``I/mR^2 = 2/5``) this is ``a = (5/7) g (sin - mu_r cos)``,
    which reduces at ``mu_r = 0`` to the pure-rolling value ``(5/7) g sin`` (i.e.
    the ``tan(theta) = 3.5 mu`` rolling limit of the module docstring) and is
    non-positive when ``tan(theta) <= mu_r`` (the sphere is held).

    Returns ``a``, the rolling limit ``mu_r``, the pure-rolling (``mu_r = 0``)
    value and the hold verdict, so the report can quote the bracket rather than
    a single number.
    """
    th = math.radians(float(theta_deg))
    inertia_ratio = SPHERE_INERTIA_RATIO
    a = (case.gravity * (math.sin(th) - case.mu_rolling * math.cos(th))
         / (1.0 + inertia_ratio))
    a_no_resistance = (case.gravity * math.sin(th) / (1.0 + inertia_ratio))
    return {
        "a_m_s2": a,
        "a_no_rolling_resistance_m_s2": a_no_resistance,
        "inertia_ratio_I_over_mR2": inertia_ratio,
        "mu_rolling": float(case.mu_rolling),
        "rolling_hold": bool(math.tan(th) <= case.mu_rolling),
        "rolling_resistance_ratio": (float(case.mu_rolling / math.tan(th))
                                     if math.tan(th) > 0.0 else float("inf")),
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def _com(particles: Sequence[PolyhedronParticle]) -> np.ndarray:
    return np.mean([p.x for p in particles], axis=0)


def _fit_acceleration(t: np.ndarray, x: np.ndarray, start_fraction: float
                      ) -> Tuple[float, float, float]:
    """Least-squares ``x = x0 + v0 tau + a tau^2 / 2`` on the late window.

    Returns ``(a, r_squared, window_start)``.  The early part of the run holds
    the stick-to-slip transient and is excluded by ``start_fraction``.
    """
    t0 = float(t[0] + start_fraction * (t[-1] - t[0]))
    mask = t >= t0
    tau = t[mask] - t0
    values = x[mask]
    coef = np.polyfit(tau, values, 2)
    a = 2.0 * float(coef[0])
    pred = np.polyval(coef, tau)
    ss_res = float(np.sum((values - pred) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return a, r2, t0


def _probe_normal_load(system: PolyhedronDEMSystem,
                       particles: Sequence[PolyhedronParticle],
                       normal: np.ndarray) -> Tuple[float, float]:
    """Static normal load and wall-contact count, probed on a *copy* of state.

    The probe calls the production wall law with a deep-copied contact state, so
    it cannot advance the Cundall-Strack / stick-slip history of the running
    contact (probing inside the loop would double-integrate the tangential
    spring).
    """
    total = 0.0
    contacts = 0
    for idx, particle in enumerate(particles):
        state = system.wall_states.get((idx, 0))
        state = ContactState() if state is None else copy.deepcopy(state)
        result, contacting = compute_wall_contact_force(
            particle.shape, particle.v, particle.omega, particle.mass,
            particle.inertia, normal, 0.0, state, system.config, 1.0e-5)
        if contacting:
            total += float(result["f_n"])
            contacts += 1
    return total, float(contacts)


def _slip_speed(particles: Sequence[PolyhedronParticle],
                normal: np.ndarray) -> float:
    """Maximum tangential (slip) speed at the wall contacts, m/s."""
    worst = 0.0
    for particle in particles:
        _, contact_point = wall_penetration(particle.shape, normal)
        lever = contact_point - particle.x
        v_contact = particle.v + np.cross(particle.omega, lever)
        tangential = v_contact - np.dot(v_contact, normal) * normal
        worst = max(worst, float(np.linalg.norm(tangential)))
    return worst


def simulate(case: TiltPlaneCase, theta_deg: float, *,
             t_end: Optional[float] = None,
             start_fraction: float = 0.4) -> Dict[str, Any]:
    """Integrate one tilted-plane run and return the measured quantities.

    Returns a JSON-ready dict with the down-slope centroid displacement, the
    first-motion verdict against ``case.slide_tolerance``, the steady
    acceleration and its fit quality, the slip speed at the wall and the
    normal-load check ``F_n / W`` against ``cos(theta)``.
    """
    system, particles, normal, downslope = build_case(case, theta_deg)
    dt = case.dt
    duration = float(case.t_end if t_end is None else t_end)
    steps = max(int(round(duration / dt)), 2)
    every = max(1, steps // int(case.samples))
    total_mass = float(sum(p.mass for p in particles))
    x0 = _com(particles)
    times: List[float] = []
    positions: List[np.ndarray] = []
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            times.append(step * dt)
            positions.append(_com(particles))
        if step < steps:
            system.step(dt)
    t = np.asarray(times, dtype=np.float64)
    com = np.asarray(positions, dtype=np.float64)
    down = (com - x0) @ downslope
    slide = float(down[-1])
    acceleration, r2, fit_start = _fit_acceleration(t, down, start_fraction)
    normal_load, contacts = _probe_normal_load(system, particles, normal)
    weight = total_mass * case.gravity
    th = math.radians(float(theta_deg))
    return {
        "theta_deg": float(theta_deg),
        "t_end": duration,
        "dt": dt,
        "steps": steps,
        "slide_m": slide,
        "slide_tolerance_m": case.slide_tolerance,
        "moved": bool(slide > case.slide_tolerance),
        "com_speed_m_s": float(np.linalg.norm(
            np.mean([p.v for p in particles], axis=0))),
        "slip_speed_m_s": _slip_speed(particles, normal),
        "acceleration_m_s2": acceleration,
        "acceleration_r2": r2,
        "acceleration_window_start_s": fit_start,
        "acceleration_analytic_m_s2": case.gravity * (
            math.sin(th) - case.mu_dynamic * math.cos(th)),
        "rolling_anchor": (rolling_acceleration_analytic(case, theta_deg)
                           if case.shape == "sphere" else None),
        "rolling_moment_on": bool(case.rolling_active),
        "normal_load_n": normal_load,
        "wall_contacts": contacts,
        "weight_n": weight,
        "normal_load_over_weight": normal_load / weight if weight > 0 else 0.0,
        "cos_theta": math.cos(th),
    }


# ---------------------------------------------------------------------------
# Static branch: first-motion angle
# ---------------------------------------------------------------------------
def critical_angle(case: TiltPlaneCase, *, low: float = 0.0, high: float = 60.0,
                   iterations: int = 11, tolerance_deg: float = 0.05,
                   t_end: Optional[float] = None) -> Dict[str, Any]:
    """Bisect the tilt angle for first motion and invert ``mu_s = tan(theta_c)``.

    The bracket ``[low, high]`` is generic (never seeded with ``atan(mu)``), so
    the measurement can disagree with the closed form.  Motion at ``high`` and
    rest at ``low`` are both required; otherwise the case is reported as
    ``inconclusive`` instead of returning a number.
    """
    probes: List[Dict[str, Any]] = []
    hi_probe = simulate(case, high, t_end=t_end)
    lo_probe = simulate(case, low, t_end=t_end)
    probes.extend([lo_probe, hi_probe])
    if not hi_probe["moved"] or lo_probe["moved"]:
        return {
            "theta_c_deg": None,
            "theta_c_bracket": [low, high],
            "mu_measured": None,
            "runs": len(probes),
            "status": "inconclusive",
            "reason": ("high probe did not move" if not hi_probe["moved"]
                       else "low probe moved"),
            "probes": probes,
        }
    lo, hi = low, high
    while (hi - lo) > tolerance_deg and len(probes) < iterations + 2:
        mid = 0.5 * (lo + hi)
        probe = simulate(case, mid, t_end=t_end)
        probes.append(probe)
        if probe["moved"]:
            hi = mid
        else:
            lo = mid
    theta_c = 0.5 * (lo + hi)
    return {
        "theta_c_deg": theta_c,
        "theta_c_bracket": [lo, hi],
        "mu_measured": math.tan(math.radians(theta_c)),
        "runs": len(probes),
        "status": "ok",
        "probes": probes,
    }


# ---------------------------------------------------------------------------
# Dynamic branch: acceleration inversion
# ---------------------------------------------------------------------------
def dynamic_friction(case: TiltPlaneCase, theta_deg: float, *,
                     t_end: Optional[float] = None) -> Dict[str, Any]:
    """Invert the steady sliding acceleration for the dynamic coefficient.

    ``mu_k = tan(theta) - a / (g cos(theta))``; the reported dynamic friction
    angle is ``atan(mu_k)``.
    """
    run = simulate(case, theta_deg, t_end=t_end)
    th = math.radians(float(theta_deg))
    a = run["acceleration_m_s2"]
    mu_app = math.tan(th) - a / (case.gravity * math.cos(th))
    return {
        "run": run,
        "theta_deg": float(theta_deg),
        "acceleration_m_s2": a,
        "acceleration_r2": run["acceleration_r2"],
        "mu_measured": mu_app,
        "friction_angle_deg": math.degrees(math.atan(mu_app)),
        "mu_reference": case.mu_dynamic,
        "slip_speed_m_s": run["slip_speed_m_s"],
    }


# ---------------------------------------------------------------------------
# Load-sharing check on an n x n layer
# ---------------------------------------------------------------------------
def layer_check(case: TiltPlaneCase, *, margin_deg: float = 1.5,
                offset_deg: float = 12.0) -> Dict[str, Any]:
    """Aggregate check on an ``n x n`` layer sharing the plane.

    Two runs bracket the closed-form limit: at ``atan(mu_s) - margin`` the layer
    must stay put, and at ``atan(mu_s) + offset`` the steady acceleration of the
    centroid must invert to the coefficient.  This is a *verification* probe
    placed at the analytic angle (the primary coefficient measurement is the
    generic-bracket bisection of :func:`critical_angle`); what it adds is that
    ``n^2`` wall contacts carrying the weight reproduce the same law.
    """
    theta_c = math.degrees(math.atan(case.mu_s))
    rest = simulate(case, theta_c - margin_deg)
    slide = dynamic_friction(case, theta_c + offset_deg)
    mu_measured = slide["mu_measured"]
    return {
        "theta_c_analytic_deg": theta_c,
        "rest_theta_deg": theta_c - margin_deg,
        "rest_run": rest,
        "rested": bool(not rest["moved"]),
        "slide_theta_deg": theta_c + offset_deg,
        "slide": slide,
        "theta_deg": theta_c + offset_deg,
        "acceleration_m_s2": slide["acceleration_m_s2"],
        "acceleration_r2": slide["acceleration_r2"],
        "mu_measured": mu_measured,
        "rel_err_pct": 100.0 * abs(mu_measured - case.mu_s) / case.mu_s,
        "friction_angle_deg": math.degrees(math.atan(mu_measured)),
        "wall_contacts": slide["run"]["wall_contacts"],
        "normal_load_over_weight": slide["run"]["normal_load_over_weight"],
        "cos_theta": slide["run"]["cos_theta"],
    }


# ---------------------------------------------------------------------------
# Campaign definition
# ---------------------------------------------------------------------------
def _dynamic_angles(mu: float) -> Tuple[float, ...]:
    base = math.degrees(math.atan(mu))
    return tuple(round(base + offset, 6) for offset in (8.0, 12.0, 16.0))


def campaign_tasks() -> List[Dict[str, Any]]:
    """Deterministic task list of the benchmark campaign.

    Task families:

    ``static``        single rotation-locked cube, dt levels 0/1/2, mu in
                      ``MU_TARGETS`` -- the acceptance measurement.
    ``dynamic``       same cube, three tilts above ``atan(mu)`` -- dynamic
                      inversion of the same coefficient.
    ``dual_static`` / ``dual_dynamic``
                      static/dynamic pair ``(0.6, 0.4)`` -- the
                      distinguishability measurement.
    ``raft_check``    3x3 layer of rotation-locked cubes -- load sharing across
                      nine wall contacts must reproduce the same law (bracket +
                      steady-acceleration inversion).
    ``sphere``        free-rotation regime probes (rolling vs gross slip).
    ``cube_free``     free-rotation flat-faced cube (documented creep defect).
    """
    tasks: List[Dict[str, Any]] = []

    def add(kind: str, case_kwargs: Dict[str, Any], *,
            theta: Optional[float] = None, extra: Optional[Dict[str, Any]] = None,
            label: Optional[str] = None) -> None:
        task_id = kind
        for key in sorted(case_kwargs):
            task_id += f":{key}={case_kwargs[key]}"
        if theta is not None:
            task_id += f":theta={round(float(theta), 6)}"
        tasks.append({
            "task_id": task_id,
            "kind": kind,
            "label": label or task_id,
            "case": case_kwargs,
            "theta_deg": None if theta is None else float(theta),
            "extra": extra or {},
        })

    # --- acceptance: static inversion, three mu, three dt levels -----------
    for mu in MU_TARGETS:
        for level in DT_LEVELS:
            add("static", {"mu_s": mu, "dt_level": level},
                extra={"mu_target": mu, "dt_level": level})

    # --- dynamic inversion of the same coefficient -------------------------
    for mu in MU_TARGETS:
        for theta in _dynamic_angles(mu):
            add("dynamic", {"mu_s": mu}, theta=theta, extra={"mu_target": mu})

    # --- static/dynamic pair: distinguishability ---------------------------
    mu_s, mu_k = DUAL_CASE
    add("dual_static", {"mu_s": mu_s, "mu_k": mu_k},
        extra={"mu_target": mu_s, "mu_k": mu_k})
    for theta in (35.0, 45.0, 55.0):
        add("dual_dynamic", {"mu_s": mu_s, "mu_k": mu_k}, theta=theta,
            extra={"mu_target": mu_s, "mu_k": mu_k})

    # --- load sharing across a 3x3 layer (non-contacting, see raft_gap) ----
    # Spheres are used because PolyhedronDEMSystem rebuilds the convex hull of
    # every polytope at every step (measured ~130 ms/step for nine cubes vs
    # ~18 ms/step for nine spheres); the wall contact of a sphere is also a
    # genuine point contact, so rotation locking isolates the Coulomb cap
    # without the flat-face centre-of-pressure defect.
    for mu in MU_TARGETS:
        add("raft_check", {"mu_s": mu, "raft": 3, "raft_gap": 0.02,
                           "shape": "sphere", "t_end": 0.10},
            extra={"mu_target": mu})

    # --- free-rotation regime probes --------------------------------------
    for theta in (20.0, 40.0, 60.0):
        add("sphere", {"mu_s": 0.4, "shape": "sphere", "rotation_locked": False},
            theta=theta, extra={"mu_target": 0.4})
    add("cube_free", {"mu_s": 0.6, "rotation_locked": False},
        theta=round(math.degrees(math.atan(0.6)) - 2.0, 6),
        extra={"mu_target": 0.6})
    return tasks


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one campaign task (module-level so it is picklable)."""
    case = TiltPlaneCase(**task["case"])
    kind = task["kind"]
    if kind in ("static", "dual_static"):
        result = critical_angle(case)
        result["mu_reference"] = case.mu_s
        result["theta_analytic_deg"] = math.degrees(math.atan(case.mu_s))
        if result["mu_measured"] is None:
            result["rel_err_pct"] = None
            result["angle_err_deg"] = None
        else:
            result["rel_err_pct"] = (100.0 * abs(result["mu_measured"] - case.mu_s)
                                     / case.mu_s)
            result["angle_err_deg"] = (result["theta_c_deg"]
                                       - result["theta_analytic_deg"])
    elif kind == "raft_check":
        result = layer_check(case)
    elif kind in ("dynamic", "dual_dynamic"):
        result = dynamic_friction(case, task["theta_deg"])
        result["rel_err_pct"] = (100.0 * abs(result["mu_measured"] - case.mu_dynamic)
                                 / case.mu_dynamic)
    elif kind in ("sphere", "cube_free"):
        result = dynamic_friction(case, task["theta_deg"])
        th = math.radians(task["theta_deg"])
        result["tan_theta"] = math.tan(th)
        result["rolling_limit_tan_theta"] = SPHERE_ROLLING_FACTOR * case.mu_dynamic
        result["mu_app_rolling_prediction"] = math.tan(th) / SPHERE_ROLLING_FACTOR
        result["gross_slip_acceleration_prediction"] = case.gravity * (
            math.sin(th) - case.mu_dynamic * math.cos(th))
        result["rel_err_pct"] = None
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown task kind {kind!r}")
    return {
        "task_id": task["task_id"],
        "kind": kind,
        "label": task["label"],
        "case": case.resolved(),
        "theta_deg": task["theta_deg"],
        "extra": task["extra"],
        "result": result,
    }


def run_campaign(workers: int = 1,
                 tasks: Optional[Sequence[Dict[str, Any]]] = None
                 ) -> List[Dict[str, Any]]:
    """Run every campaign task, returning results sorted by ``task_id``.

    The sort makes the assembled payload independent of ``workers`` and of
    completion order.
    """
    task_list = list(campaign_tasks() if tasks is None else tasks)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_task, task_list))
    else:
        results = [run_task(task) for task in task_list]
    results.sort(key=lambda item: item["task_id"])
    return results


# ---------------------------------------------------------------------------
# Assembly into the reported tables
# ---------------------------------------------------------------------------
def _task_id(kind: str, **params: Any) -> str:
    task_id = kind
    for key in sorted(params):
        task_id += f":{key}={params[key]}"
    return task_id


def assemble_results(task_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compose the reported tables and verdicts from the raw task results.

    Every number in ``results.json`` originates here from a task payload; no
    value is typed by hand, and every pass/fail flag is computed from the
    predeclared thresholds above.
    """
    index = {item["task_id"]: item for item in task_results}

    # --- static inversion table -------------------------------------------
    static_rows: List[Dict[str, Any]] = []
    for mu in MU_TARGETS:
        for level in DT_LEVELS:
            task = index[_task_id("static", mu_s=mu, dt_level=level)]
            result, case = task["result"], task["case"]
            static_rows.append({
                "mu_set": mu,
                "dt_level": level,
                "dt_s": case["dt"],
                "t_end_s": case["t_end"],
                "theta_c_deg": result["theta_c_deg"],
                "theta_analytic_deg": result["theta_analytic_deg"],
                "angle_err_deg": result["angle_err_deg"],
                "mu_measured": result["mu_measured"],
                "rel_err_pct": result["rel_err_pct"],
                "runs": result["runs"],
                "status": result["status"],
                "within_limit": bool(result["rel_err_pct"] is not None
                                     and result["rel_err_pct"] <= REL_ERR_LIMIT_PCT),
                "within_strict": bool(result["rel_err_pct"] is not None
                                      and result["rel_err_pct"] <= REPORT_STRICT_PCT),
            })

    # --- dynamic inversion table ------------------------------------------
    dynamic_rows: List[Dict[str, Any]] = []
    for mu in MU_TARGETS:
        for theta in _dynamic_angles(mu):
            task = index[_task_id("dynamic", mu_s=mu, theta=theta)]
            result = task["result"]
            dynamic_rows.append({
                "mu_set": mu,
                "theta_deg": result["theta_deg"],
                "acceleration_m_s2": result["acceleration_m_s2"],
                "acceleration_analytic_m_s2":
                    result["run"]["acceleration_analytic_m_s2"],
                "acceleration_r2": result["acceleration_r2"],
                "slip_speed_m_s": result["slip_speed_m_s"],
                "mu_measured": result["mu_measured"],
                "friction_angle_deg": result["friction_angle_deg"],
                "rel_err_pct": result["rel_err_pct"],
                "within_limit": bool(result["rel_err_pct"] <= REL_ERR_LIMIT_PCT),
            })

    # --- convergence of the critical angle --------------------------------
    convergence: List[Dict[str, Any]] = []
    for mu in MU_TARGETS:
        rows = sorted((row for row in static_rows if row["mu_set"] == mu),
                      key=lambda row: row["dt_level"])
        errors = [abs(row["angle_err_deg"]) for row in rows]
        convergence.append({
            "mu_set": mu,
            "dt_s": [row["dt_s"] for row in rows],
            "theta_c_deg": [row["theta_c_deg"] for row in rows],
            "abs_angle_err_deg": errors,
            "coarsest_abs_angle_err_deg": errors[0],
            "finest_abs_angle_err_deg": errors[-1],
            "error_shrinks_with_refinement": bool(errors[-1] <= errors[0] + 1.0e-12),
            "non_increasing": bool(all(errors[i + 1] <= errors[i] + 1.0e-12
                                       for i in range(len(errors) - 1))),
        })

    # --- static vs dynamic (distinguishability) ---------------------------
    mu_s, mu_k = DUAL_CASE
    dual_static = index[_task_id("dual_static", mu_k=mu_k, mu_s=mu_s)]["result"]
    dual_dynamic_rows: List[Dict[str, Any]] = []
    for theta in (35.0, 45.0, 55.0):
        result = index[_task_id("dual_dynamic", mu_k=mu_k, mu_s=mu_s,
                                theta=theta)]["result"]
        dual_dynamic_rows.append({
            "theta_deg": result["theta_deg"],
            "acceleration_m_s2": result["acceleration_m_s2"],
            "acceleration_analytic_m_s2":
                result["run"]["acceleration_analytic_m_s2"],
            "acceleration_r2": result["acceleration_r2"],
            "mu_measured": result["mu_measured"],
            "friction_angle_deg": result["friction_angle_deg"],
            "rel_err_pct": result["rel_err_pct"],
        })
    mu_k_measured = float(np.mean([row["mu_measured"] for row in dual_dynamic_rows]))
    theta_dynamic = math.degrees(math.atan(mu_k_measured))
    theta_static = dual_static["theta_c_deg"]
    separation = (theta_static - theta_dynamic) if theta_static is not None else None
    static_dynamic = {
        "case": {"mu_s": mu_s, "mu_k": mu_k},
        "theta_static_deg": theta_static,
        "theta_static_analytic_deg": math.degrees(math.atan(mu_s)),
        "mu_s_measured": dual_static["mu_measured"],
        "mu_s_rel_err_pct": dual_static["rel_err_pct"],
        "mu_k_measured_mean": mu_k_measured,
        "mu_k_rel_err_pct": 100.0 * abs(mu_k_measured - mu_k) / mu_k,
        "theta_dynamic_deg": theta_dynamic,
        "theta_dynamic_analytic_deg": math.degrees(math.atan(mu_k)),
        "separation_deg": separation,
        "separation_analytic_deg": (math.degrees(math.atan(mu_s))
                                    - math.degrees(math.atan(mu_k))),
        "min_required_separation_deg": DUAL_MIN_SEPARATION_DEG,
        "distinguishable": bool(separation is not None
                                and separation >= DUAL_MIN_SEPARATION_DEG),
        "dynamic_rows": dual_dynamic_rows,
    }

    # --- secondary evidence ----------------------------------------------
    raft_rows: List[Dict[str, Any]] = []
    for mu in MU_TARGETS:
        task = index[_task_id("raft_check", mu_s=mu, raft=3, raft_gap=0.02,
                              shape="sphere", t_end=0.1)]
        result = task["result"]
        raft_rows.append({
            "mu_set": mu,
            "theta_c_analytic_deg": result["theta_c_analytic_deg"],
            "rest_theta_deg": result["rest_theta_deg"],
            "rested_below_analytic": result["rested"],
            "slide_theta_deg": result["slide_theta_deg"],
            "acceleration_m_s2": result["acceleration_m_s2"],
            "acceleration_r2": result["acceleration_r2"],
            "mu_measured": result["mu_measured"],
            "rel_err_pct": result["rel_err_pct"],
            "wall_contacts": result["wall_contacts"],
            "normal_load_over_weight": result["normal_load_over_weight"],
            "cos_theta": result["cos_theta"],
            "within_limit": bool(result["rel_err_pct"] <= REL_ERR_LIMIT_PCT),
        })
    sphere_rows: List[Dict[str, Any]] = []
    for theta in (20.0, 40.0, 60.0):
        result = index[_task_id("sphere", mu_s=0.4, rotation_locked=False,
                                shape="sphere", theta=theta)]["result"]
        sphere_rows.append({
            "theta_deg": result["theta_deg"],
            "mu_set": 0.4,
            "tan_theta": result["tan_theta"],
            "rolling_limit_tan_theta": result["rolling_limit_tan_theta"],
            "acceleration_m_s2": result["acceleration_m_s2"],
            "acceleration_gross_slip_prediction":
                result["gross_slip_acceleration_prediction"],
            "mu_apparent_measured": result["mu_measured"],
            "mu_apparent_rolling_prediction": result["mu_app_rolling_prediction"],
            "slip_speed_m_s": result["slip_speed_m_s"],
            "regime": ("rolling_without_slip"
                       if result["tan_theta"] <= result["rolling_limit_tan_theta"]
                       else "gross_slip"),
        })
    free_task = index[_task_id(
        "cube_free", mu_s=0.6, rotation_locked=False,
        theta=round(math.degrees(math.atan(0.6)) - 2.0, 6))]["result"]
    free_cube_row = {
        "mu_set": 0.6,
        "theta_deg": free_task["theta_deg"],
        "theta_analytic_deg": math.degrees(math.atan(0.6)),
        "slide_m": free_task["run"]["slide_m"],
        "slide_tolerance_m": free_task["run"]["slide_tolerance_m"],
        "moved": free_task["run"]["moved"],
        "note": ("free-rotation flat faces creep below atan(mu): the collapsed "
                 "face contact cannot shift its centre of pressure (measured "
                 "defect; not used for the anchor)"),
    }

    # --- verdicts ---------------------------------------------------------
    reference_rows = [row for row in static_rows if row["dt_level"] == 0]
    reference_probes = [index[_task_id("static", mu_s=mu, dt_level=0)]["result"]
                        ["probes"][-1] for mu in MU_TARGETS]
    verdicts = {
        "A_static_inversion_within_5pct_all_dt_levels": bool(
            all(row["within_limit"] for row in static_rows)),
        "B_static_inversion_within_5pct_reference_dt": bool(
            all(row["within_limit"] for row in reference_rows)),
        "B_max_rel_err_pct_reference_dt": max(row["rel_err_pct"]
                                              for row in reference_rows),
        "B_all_within_strict_3pct_reference_dt": bool(
            all(row["within_strict"] for row in reference_rows)),
        "C_static_dynamic_distinguishable": bool(static_dynamic["distinguishable"]),
        "C_separation_deg": separation,
        "D_error_non_increasing_with_dt_refinement": bool(
            all(item["non_increasing"] for item in convergence)),
        "E_dynamic_inversion_within_5pct": bool(
            all(row["within_limit"] for row in dynamic_rows)),
        "F_raft_load_sharing_within_5pct": bool(
            all(row["within_limit"] and row["rested_below_analytic"]
                for row in raft_rows)),
        "G_normal_load_matches_cos_theta": bool(all(
            abs(probe["normal_load_over_weight"] - probe["cos_theta"]) <= 0.01
            for probe in reference_probes)),
    }

    payload: Dict[str, Any] = {
        "schema": "tensordem.tilt_plane_friction/v1",
        "case_family": ("tilted plane, Coulomb sliding limit tan(theta_c) = mu_s, "
                        "a = g(sin theta - mu_k cos theta)"),
        "thresholds": {
            "mu_targets": list(MU_TARGETS),
            "rel_err_limit_pct": REL_ERR_LIMIT_PCT,
            "report_strict_pct": REPORT_STRICT_PCT,
            "dual_min_separation_deg": DUAL_MIN_SEPARATION_DEG,
            "dual_case": list(DUAL_CASE),
            "dt_levels": list(DT_LEVELS),
            "slide_tolerance_fraction_of_extent": SLIDE_TOLERANCE_FRACTION,
            "kick_speed_m_s": KICK_SPEED,
            "dt_safety": DT_SAFETY,
            "sphere_rolling_factor": SPHERE_ROLLING_FACTOR,
            "gravity_m_s2": GRAVITY,
        },
        "static_inversion": static_rows,
        "dynamic_inversion": dynamic_rows,
        "convergence": convergence,
        "static_dynamic": static_dynamic,
        "raft_static": raft_rows,
        "sphere_regime_probe": sphere_rows,
        "free_cube_probe": free_cube_row,
        "verdicts": verdicts,
        "provenance": {
            "generator": "tensordem.tilt_plane_friction",
            "repro_cmd": ("python -m tensordem.tilt_plane_friction campaign "
                          "--out artifacts/tilted_plane --workers 8"),
            "figure_cmd": ("python -m tensordem.tilt_plane_friction figures "
                           "--results artifacts/tilted_plane/results.json"),
            "tasks": [item["task_id"] for item in task_results],
        },
    }
    return payload


def results_sha256(payload: Dict[str, Any]) -> str:
    """sha256 over the canonical (sorted, compact) JSON of ``payload``."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_results(payload: Dict[str, Any], out_dir) -> Dict[str, Any]:
    """Write ``results.json`` (carrying its own sha256) plus a provenance record."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.pop("sha256", None)
    digest = results_sha256(body)
    body["sha256"] = digest
    text = json.dumps(body, sort_keys=True, indent=2, allow_nan=False)
    (out / "results.json").write_text(text + "\n")
    prod = {
        "results_sha256": digest,
        "results_file": str(out / "results.json"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": sys.platform,
        "repro_cmd": body["provenance"]["repro_cmd"],
        "sha_note": ("sha256 covers the canonical compact JSON of results.json "
                     "with the 'sha256' key removed; regenerate and compare"),
    }
    (out / "provenance.json").write_text(
        json.dumps(prod, sort_keys=True, indent=2) + "\n")
    return {"sha256": digest, "path": str(out / "results.json")}


# ---------------------------------------------------------------------------
# Figures (generated from results.json only -- zero hand-typed numbers)
# ---------------------------------------------------------------------------
def render_figures(results_path, out_dir=None, dpi: int = 150) -> List[Path]:
    """Render the benchmark figures straight from ``results.json``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = Path(results_path)
    payload = json.loads(src.read_text())
    out = Path(out_dir) if out_dir is not None else src.parent
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    g = payload["thresholds"]["gravity_m_s2"]

    # 1) static inversion: measured vs set coefficient + angle identity
    rows = payload["static_inversion"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for level in payload["thresholds"]["dt_levels"]:
        subset = [row for row in rows if row["dt_level"] == level]
        axes[0].plot([row["mu_set"] for row in subset],
                     [row["mu_measured"] for row in subset],
                     marker="o", label="dt level %d" % level)
        axes[1].plot([row["theta_analytic_deg"] for row in subset],
                     [row["theta_c_deg"] for row in subset],
                     marker="s", label="dt level %d" % level)
    axes[0].plot([0.0, 0.7], [0.0, 0.7], "k--", lw=1, label="identity")
    axes[0].set_xlabel(r"$\mu$ set (-)")
    axes[0].set_ylabel(r"$\mu$ measured = tan$\theta_c$ (-)")
    axes[0].set_title("static inversion")
    axes[0].legend(fontsize=8)
    angles = [row["theta_analytic_deg"] for row in rows]
    axes[1].plot([min(angles), max(angles)], [min(angles), max(angles)],
                 "k--", lw=1, label="identity")
    axes[1].set_xlabel(r"atan($\mu_s$) (deg)")
    axes[1].set_ylabel(r"measured $\theta_c$ (deg)")
    axes[1].set_title("critical angle vs closed form")
    axes[1].legend(fontsize=8)
    fig.suptitle("TensorDEM tilted-plane Coulomb anchor", fontsize=10)
    fig.tight_layout()
    path = out / "static_inversion.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    # 2) time-step convergence of the critical-angle error
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for item in payload["convergence"]:
        ax.loglog(item["dt_s"],
                  [max(value, 1e-4) for value in item["abs_angle_err_deg"]],
                  marker="o", label=r"$\mu$ = %.1f" % item["mu_set"])
    ax.set_xlabel("time step (s)")
    ax.set_ylabel(r"$|\theta_c - \mathrm{atan}(\mu)|$ (deg)")
    ax.set_title("critical-angle error vs time step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "convergence.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    # 3) dynamic inversion + static/dynamic separation
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for mu in payload["thresholds"]["mu_targets"]:
        subset = [row for row in payload["dynamic_inversion"] if row["mu_set"] == mu]
        thetas = np.linspace(min(row["theta_deg"] for row in subset) - 2.0,
                             max(row["theta_deg"] for row in subset) + 2.0, 60)
        th = np.radians(thetas)
        axes[0].plot(thetas, g * (np.sin(th) - mu * np.cos(th)), "-",
                     label=r"analytic $\mu_k$ = %.1f" % mu)
        axes[0].plot([row["theta_deg"] for row in subset],
                     [row["acceleration_m_s2"] for row in subset],
                     "o", mfc="none")
    axes[0].set_xlabel(r"tilt $\theta$ (deg)")
    axes[0].set_ylabel(r"$a$ (m/s$^2$)")
    axes[0].set_title("steady sliding acceleration (open markers = measured)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    dual = payload["static_dynamic"]
    axes[1].bar([0], [dual["theta_static_deg"]], width=0.5,
                label=r"static $\theta_s$ = atan$\mu_s$")
    axes[1].bar([1], [dual["theta_dynamic_deg"]], width=0.5,
                label=r"dynamic atan$\mu_k$")
    axes[1].axhline(dual["theta_static_analytic_deg"], ls="--", c="k", lw=1)
    axes[1].axhline(dual["theta_dynamic_analytic_deg"], ls=":", c="k", lw=1)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["static", "dynamic"])
    axes[1].set_ylabel("friction angle (deg)")
    axes[1].set_title("separation = %.2f deg (min %.1f)"
                      % (dual["separation_deg"], dual["min_required_separation_deg"]))
    axes[1].legend(fontsize=8)
    fig.suptitle("dynamic branch + static/dynamic distinguishability", fontsize=10)
    fig.tight_layout()
    path = out / "dynamic_and_dual.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    # 4) free-rotation regime probe: naive inversion vs the rolling trap
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    probe = payload["sphere_regime_probe"]
    ax.plot([row["theta_deg"] for row in probe],
            [row["mu_apparent_measured"] for row in probe],
            "o-", label="measured apparent " + r"$\mu$")
    ax.plot([row["theta_deg"] for row in probe],
            [row["mu_apparent_rolling_prediction"] for row in probe],
            "s--", label=r"rolling $(2/7)\tan\theta$")
    ax.axhline(probe[0]["mu_set"], c="k", ls=":", lw=1, label=r"true $\mu$")
    ax.set_xlabel(r"tilt $\theta$ (deg)")
    ax.set_ylabel("apparent friction coefficient (-)")
    ax.set_title("free-rotation sphere: the rolling trap")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "sphere_regime.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensordem.tilt_plane_friction",
        description="Tilted-plane Coulomb friction benchmark (analytic anchor).")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("campaign", help="run the full campaign and write results.json")
    run_p.add_argument("--out", default="artifacts/tilted_plane")
    run_p.add_argument("--workers", type=int, default=1)
    run_p.add_argument("--figures", action="store_true",
                       help="also render the figures from the written results.json")
    run_p.add_argument("--shard", default=None,
                       help="i/n: run only the i-th of n contiguous task slices")
    run_p.add_argument("--dump-tasks", action="store_true",
                       help="print the task list and exit (planning)")
    run_p.add_argument("--smoke", action="store_true",
                       help="end-to-end pipeline smoke test: every task runs with "
                            "t_end = 0.02 s (numbers are NOT the benchmark)")

    probe_p = sub.add_parser("probe", help="single measurement (debug/inspection)")
    probe_p.add_argument("--mu", type=float, default=0.4)
    probe_p.add_argument("--mu-k", type=float, default=None)
    probe_p.add_argument("--theta", type=float, default=None)
    probe_p.add_argument("--shape", default="cube")
    probe_p.add_argument("--raft", type=int, default=1)
    probe_p.add_argument("--free-rotation", action="store_true")
    probe_p.add_argument("--t-end", type=float, default=0.30)
    probe_p.add_argument("--dt-level", type=int, default=0)

    fig_p = sub.add_parser("figures", help="render figures from results.json")
    fig_p.add_argument("--results", default="artifacts/tilted_plane/results.json")
    fig_p.add_argument("--out", default=None)

    args = parser.parse_args(argv)

    if args.command == "probe":
        case = TiltPlaneCase(mu_s=args.mu, mu_k=args.mu_k, shape=args.shape,
                             raft=args.raft, rotation_locked=not args.free_rotation,
                             t_end=args.t_end, dt_level=args.dt_level)
        theta = args.theta
        if theta is None:
            theta = math.degrees(math.atan(case.mu_dynamic)) + 12.0
        print(json.dumps(dynamic_friction(case, theta), indent=2, sort_keys=True))
        return 0

    if args.command == "figures":
        paths = render_figures(args.results, args.out)
        for path in paths:
            print(path)
        return 0

    tasks = campaign_tasks()
    if args.dump_tasks:
        print(json.dumps(tasks, indent=2))
        return 0
    if args.smoke:
        for task in tasks:
            task["case"]["t_end"] = 0.02
    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        tasks = [task for k, task in enumerate(tasks) if k % total == index]
    results = run_campaign(workers=args.workers, tasks=tasks)
    payload = assemble_results(results)
    written = write_results(payload, args.out)
    print(f"results -> {written['path']}  sha256={written['sha256']}")
    verdicts = payload["verdicts"]
    for key in sorted(verdicts):
        print(f"  {key}: {verdicts[key]}")
    if args.figures:
        for path in render_figures(written["path"]):
            print(f"figure -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
