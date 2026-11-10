"""Lagrangian point particles and their force budget.

The particle carries mass, position and velocity only: this is the *point
particle* (unresolved) assumption of CFD-DEM.  All hydrodynamic interaction is
carried by :mod:`tensordem.point_cfd_dem.closures` and is evaluated with the
fluid state interpolated at the particle centre.

Momentum equation actually integrated
-------------------------------------
::

    m_p du_p/dt = F_drag + F_pg + F_am + (rho_p - rho_f) V_p g + F_ext

    F_drag = 3 pi mu d (u_f - u_p) f(Re) H(eps)      (Stokes * Schiller-Naumann * RZ)
    F_pg   = -V_p grad p                              (pressure gradient)
    F_am   = -C_A rho_f V_p (du_p/dt - Du_f/Dt)       (added mass)

The added-mass term depends on the particle acceleration, so it is moved to the
left-hand side and integrated **implicitly**::

    (m_p + C_A rho_f V_p) du_p/dt = F_drag + F_pg + (rho_p - rho_f) V_p g
                                    + C_A rho_f V_p Du_f/Dt + F_ext

which is what makes the added-mass oscillator reproduce
``omega = sqrt(k / (m_p + C_A m_a))`` to within the time-integration error
instead of being limited by an explicit-coupling stability condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import closures

__all__ = ["ParticleSet", "PointParticleModel", "evaluate_forces"]


@dataclass
class ParticleSet:
    """Vectorised set of identical-or-polydisperse point particles."""

    position: np.ndarray          # (N, 3)
    velocity: np.ndarray          # (N, 3)
    radius: np.ndarray            # (N,)
    density: np.ndarray           # (N,)

    @classmethod
    def create(cls, positions, velocities, radius, density):
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        vel = np.atleast_2d(np.asarray(velocities, dtype=float))
        if vel.shape != pos.shape:
            if vel.shape[0] == 1:
                vel = np.repeat(vel, pos.shape[0], axis=0)
            else:
                raise ValueError("position/velocity shape mismatch")
        n = pos.shape[0]
        rad = np.broadcast_to(np.asarray(radius, dtype=float), (n,)).copy()
        rho = np.broadcast_to(np.asarray(density, dtype=float), (n,)).copy()
        return cls(position=pos, velocity=vel, radius=rad, density=rho)

    # ------------------------------------------------------------- geometry
    @property
    def count(self):
        return self.position.shape[0]

    @property
    def diameter(self):
        return 2.0 * self.radius

    @property
    def volume(self):
        return 4.0 / 3.0 * np.pi * self.radius ** 3

    @property
    def mass(self):
        return self.density * self.volume

    # -------------------------------------------------------------- measures
    def momentum(self):
        return np.sum(self.mass[:, None] * self.velocity, axis=0)

    def kinetic_energy(self):
        return float(np.sum(0.5 * self.mass[:, None] * self.velocity ** 2))

    def mean_velocity(self):
        return np.mean(self.velocity, axis=0)

    def copy(self):
        return ParticleSet(self.position.copy(), self.velocity.copy(),
                           self.radius.copy(), self.density.copy())


@dataclass
class PointParticleModel:
    """Physical parameters and closure switches of the point-particle model."""

    rho_f: float = 1.0
    mu: float = 0.2
    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )
    added_mass: float = 0.5          # C_A, 0.5 for a sphere
    drag_scheme: str = "schiller_naumann"
    include_pressure_gradient: bool = True
    include_added_mass: bool = True
    #: optional callable ``(particles, voidage, model) -> (N,)`` drag multiplier
    hindrance: object = None

    def hindrance_factor(self, particles, voidage):
        if self.hindrance is None:
            return np.ones(particles.count)
        return np.asarray(self.hindrance(particles, voidage, self), dtype=float)


def evaluate_forces(particles, model, u_fluid, du_dt_fluid=None,
                    grad_p=None, voidage=None, external_force=None,
                    self_response=None):
    """Assemble the force budget of every particle.

    Parameters
    ----------
    u_fluid : (3, N) fluid velocity at the particle centres
    du_dt_fluid : (3, N) or None, fluid material acceleration at the centres
    grad_p : (3, N) or None, pressure gradient at the centres
    voidage : (N,) or None, local voidage ``eps`` used by the hindrance closure
    external_force : (N, 3) or None, additional (e.g. spring) force
    self_response : (N,) or None
        Discrete self-response ``R`` (velocity per unit force handed to the
        fluid).  The Eulerian velocity at the particle contains the particle's
        own induced flow ``R * (-F_drag)``; solving the drag relation
        ``F = C (u_und - u_p)`` with ``u_und = u + R F`` implicitly gives the
        effective coefficient ``C / (1 - C R)`` used here.

    Returns
    -------
    dict with ``drag``, ``pressure_gradient``, ``weight``, ``added_mass_explicit``,
    ``external``, ``effective_mass`` and ``total`` (all ``(N, 3)`` except
    ``effective_mass`` which is ``(N,)``).
    """
    n = particles.count
    m_p = particles.mass
    v_p = particles.volume
    d = particles.diameter
    u_rel_pf = u_fluid.T - particles.velocity   # u_f - u_p  (N,3)

    if voidage is None:
        voidage = np.ones(n)

    hind = model.hindrance_factor(particles, voidage)
    # per-particle drag coefficient * correction * hindrance
    speed = np.linalg.norm(u_rel_pf, axis=1)
    if model.mu > 0.0:
        Re = model.rho_f * speed * d / model.mu
    else:
        Re = np.zeros(n)
    corr = closures.drag_correction(Re, model.drag_scheme) * hind
    coeff = closures.stokes_drag_coefficient(model.mu, d) * corr
    stiffness = None
    if self_response is not None:
        R = np.asarray(self_response, dtype=float)
        stiffness = coeff * R
        if np.any(stiffness >= 0.95):
            raise ValueError(
                "explicit point-particle coupling is ill-conditioned: "
                f"max(C*R)={float(np.max(stiffness)):.3f} >= 0.95; "
                "use a coarser grid (h >= d) or a smaller drag coefficient")
        coeff = coeff / (1.0 - stiffness)
    drag = coeff[:, None] * u_rel_pf

    if model.include_pressure_gradient and grad_p is not None:
        pressure_gradient = -v_p[:, None] * grad_p.T
    else:
        pressure_gradient = np.zeros((n, 3))

    weight = (particles.density - model.rho_f)[:, None] * v_p[:, None] * model.gravity

    m_added = model.added_mass * model.rho_f * v_p if model.include_added_mass \
        else np.zeros(n)
    if model.include_added_mass and du_dt_fluid is not None:
        added_mass_explicit = (m_added[:, None]) * du_dt_fluid.T
    else:
        added_mass_explicit = np.zeros((n, 3))

    ext = np.zeros((n, 3)) if external_force is None else np.asarray(
        external_force, dtype=float)

    total = drag + pressure_gradient + weight + added_mass_explicit + ext
    return {
        "drag": drag,
        "pressure_gradient": pressure_gradient,
        "weight": weight,
        "added_mass_explicit": added_mass_explicit,
        "external": ext,
        "effective_mass": m_p + m_added,
        "reynolds": Re,
        "total": total,
        "coefficient": coeff,
        "stiffness": stiffness,
    }