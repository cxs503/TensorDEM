"""Rigid-body dynamics of a multi-sphere clump with member-summed contacts.

The clump geometry (:mod:`tensordem.clump`) carries mass, centre of mass and the
union inertia tensor; this module gives it motion.  The clump is one rigid body
with six degrees of freedom (translation + ``clump_dynamics`` quaternion
rotation), and **every contact is a sum over member-sphere contacts**:

.. code-block:: text

   for each member i inside the contact:
       delta_i = overlap of sphere i with the wall / other sphere
       F_i     = k_n delta_i n_i + damping + tangential friction   (at contact point)
       F      += F_i
       T      += (x_contact,i - x_com) x F_i

This is the standard "multi-sphere" (a.k.a. clump / glued-sphere) contact model:
it reproduces convex-ish non-spherical shapes with sphere contact detection only.

Time integration is velocity Verlet for translation and a body-frame
quaternion/Euler integration for rotation.  Tangential (Cundall-Strack) history
is advanced exactly once per step (on the second force evaluation), which keeps
the friction state consistent with the trajectory instead of double-counting a
``v_t dt`` slip increment -- the defect class documented for ``dem3d.forces``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np

from .clump import Clump
from .clump_rolling import resistance_moment

__all__ = [
    "ClumpBody",
    "ClumpContact",
    "ContactConfig",
    "PlaneContact",
    "SphereObstacle",
    "effective_mass_along",
    "quaternion_from_axis_angle",
    "quaternion_multiply",
    "quaternion_to_matrix",
    "restitution_to_damping_ratio",
]

_IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# Contact parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactConfig:
    """Linear spring-dashpot + Cundall-Strack friction parameters (SI).

    ``normal_stiffness`` [N/m], ``tangential_stiffness`` [N/m] (default: equal to
    the normal one), ``restitution`` [-] sets the normal damping ratio through
    :func:`restitution_to_damping_ratio`, ``friction`` [-] is the Coulomb limit.

    ``rolling_friction`` (``mu_r``) and ``drilling_friction`` (``mu_t``) enable
    the per-member rolling / torsion resistance of
    :mod:`tensordem.clump_rolling`; both are zero by default, so a clump with the
    default configuration behaves exactly as before.  ``rolling_model`` selects
    the branch (``"epsd"`` elastic-plastic, ``"cdt"`` memoryless, ``"viscous"``)
    and ``rolling_stiffness_factor`` sets ``k_r = factor * mu_r * k_n * r**2``.
    """

    normal_stiffness: float = 1.0e5
    tangential_stiffness: float | None = None
    restitution: float = 0.5
    friction: float = 0.5
    rolling_friction: float = 0.0
    drilling_friction: float = 0.0
    rolling_model: str = "epsd"
    rolling_stiffness_factor: float = 2.25
    rolling_omega_ref: float = 1.0e-3

    def __post_init__(self) -> None:
        for name in ("normal_stiffness", "restitution", "friction",
                     "rolling_friction", "drilling_friction", "rolling_stiffness_factor",
                     "rolling_omega_ref"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.normal_stiffness <= 0.0:
            raise ValueError("normal_stiffness must be positive")
        if self.tangential_stiffness is not None:
            if not math.isfinite(self.tangential_stiffness) or self.tangential_stiffness <= 0.0:
                raise ValueError("tangential_stiffness must be positive")
        if not (0.0 < self.restitution <= 1.0):
            raise ValueError("restitution must be in (0, 1]")
        if self.friction < 0.0:
            raise ValueError("friction must be non-negative")
        if self.rolling_friction < 0.0 or self.drilling_friction < 0.0:
            raise ValueError("rolling_friction and drilling_friction must be non-negative")
        if self.rolling_stiffness_factor <= 0.0:
            raise ValueError("rolling_stiffness_factor must be positive")
        if self.rolling_omega_ref <= 0.0:
            raise ValueError("rolling_omega_ref must be positive")
        if self.rolling_model not in ("cdt", "viscous", "epsd"):
            raise ValueError("rolling_model must be 'cdt', 'viscous' or 'epsd'")

    @property
    def k_t(self) -> float:
        return self.normal_stiffness if self.tangential_stiffness is None else float(
            self.tangential_stiffness)

    @property
    def damping_ratio(self) -> float:
        return restitution_to_damping_ratio(self.restitution)


def restitution_to_damping_ratio(restitution: float) -> float:
    """Viscous damping ratio giving the target restitution (log-decrement law)."""
    if not (0.0 < restitution <= 1.0):
        raise ValueError("restitution must be in (0, 1]")
    if restitution == 1.0:
        return 0.0
    ln_e = math.log(restitution)
    return float(-ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2))


# --------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z)
# --------------------------------------------------------------------------
def quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 0.0:
        raise ValueError("rotation axis must be non-zero")
    unit = unit / norm
    half = 0.5 * float(angle)
    return np.concatenate(([math.cos(half)], math.sin(half) * unit))


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / float(
        np.linalg.norm(quaternion))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# --------------------------------------------------------------------------
# Contact mechanics helpers
# --------------------------------------------------------------------------
def effective_mass_along(clump: Clump, arm: Sequence[float],
                         normal: Sequence[float], inverse_inertia: np.ndarray
                         ) -> float:
    """Mass that responds to a unit impulse along ``normal`` at ``arm``.

    ``1/m_eff = 1/m + n . [(I^-1 (a x n)) x a]``; this is the mass the normal
    damping term ``c_n = 2 zeta sqrt(k_n m_eff)`` must use so that the pair of
    members reproduces the requested coefficient of restitution.
    """
    a = np.asarray(arm, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    rotational = np.cross(inverse_inertia @ np.cross(a, n), a)
    inverse = 1.0 / clump.mass + float(n @ rotational)
    if inverse <= 0.0:
        return math.inf
    return 1.0 / inverse


class ClumpContact:
    """Interface for a contact generator acting on a :class:`ClumpBody`.

    Implements the bookkeeping shared by every contact: the per-member rolling /
    drilling resistance moment of :mod:`tensordem.clump_rolling` (switched on by
    ``ContactConfig.rolling_friction`` / ``drilling_friction``) and its energy
    audit.  Subclasses call :meth:`_resistance` once per active member.
    """

    name = "contact"

    def forces(self, body: "ClumpBody", *, dt: float,
               commit: bool = True) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def summary(self) -> dict[str, object]:
        return {"name": self.name, "active_members": 0}

    def reset(self) -> None:
        self._rolling_moment = None
        self._rolling_moment_work = 0.0
        self._rolling_power_max = -math.inf

    # -- rolling / drilling resistance -----------------------------------
    def _init_resistance(self) -> None:
        self._rolling_moment = None          # (N, 3) epsd spring state, world frame
        self._rolling_moment_work = 0.0      # dissipated moment work [J]
        self._rolling_power_max = -math.inf  # max of M . omega (must stay <= 0)

    def _ensure_resistance_state(self, count: int) -> None:
        if self._rolling_moment is None or len(self._rolling_moment) != count:
            self._rolling_moment = np.zeros((count, 3))

    def _resistance(self, cfg: "ContactConfig", normal: np.ndarray, body: "ClumpBody",
                    normal_force: float, radius: float, index: int, dt: float,
                    commit: bool) -> np.ndarray:
        """Moment on the clump from rolling/drilling resistance at one member."""
        if cfg.rolling_friction <= 0.0 and cfg.drilling_friction <= 0.0:
            return np.zeros(3)
        self._ensure_resistance_state(len(body.clump))
        omega_world = body.angular_velocity_world()
        moment, state = resistance_moment(
            normal, omega_world, normal_force, radius,
            mu_r=cfg.rolling_friction, mu_t=cfg.drilling_friction,
            model=cfg.rolling_model, moment_state=self._rolling_moment[index], dt=dt,
            normal_stiffness=cfg.normal_stiffness,
            stiffness_factor=cfg.rolling_stiffness_factor,
            omega_ref=cfg.rolling_omega_ref)
        if commit:
            self._rolling_moment[index] = state
            power = float(moment @ omega_world)
            self._rolling_moment_work += -power * dt
            self._rolling_power_max = max(self._rolling_power_max, power)
        return moment

    def rolling_moment_state(self) -> np.ndarray:
        """Per-member rolling-spring moment state (world frame, N m)."""
        if self._rolling_moment is None:
            return np.zeros((0, 3))
        return np.array(self._rolling_moment, dtype=np.float64, copy=True)

    def moment_work(self) -> float:
        """Energy dissipated by the rolling/drilling moments [J] (>= 0)."""
        return float(self._rolling_moment_work)

    def moment_power_max(self) -> float:
        """Largest instantaneous ``moment . omega`` seen; must not be positive."""
        return float(self._rolling_power_max)


class PlaneContact(ClumpContact):
    """Static rigid wall: the clump lives in the half-space ``(x - point) . n >= 0``."""

    name = "plane"

    def __init__(self, normal: Sequence[float], point: Sequence[float] = (0.0, 0.0, 0.0),
                 config: ContactConfig | None = None, name: str = "plane") -> None:
        n = np.asarray(normal, dtype=np.float64)
        norm = float(np.linalg.norm(n))
        if norm <= 0.0:
            raise ValueError("plane normal must be non-zero")
        self.normal = n / norm
        self.point = np.asarray(point, dtype=np.float64)
        self.config = config or ContactConfig()
        self.name = name
        self._tangential = None           # body-frame (N, 3) history
        self._tangential_world = None
        self._init_resistance()
        self._last = {"member_forces": np.zeros((0, 3)), "penetrations": np.zeros(0)}

    def _ensure_state(self, count: int) -> None:
        if self._tangential is None or len(self._tangential) != count:
            self._tangential = np.zeros((count, 3))
            self._tangential_world = np.zeros((count, 3))

    def forces(self, body: "ClumpBody", *, dt: float,
               commit: bool = True) -> tuple[np.ndarray, np.ndarray]:
        clump = body.clump
        self._ensure_state(len(clump))
        rotation = body.rotation_matrix()
        centers = body.member_centers()
        inverse_inertia = body.inverse_inertia_world()
        cfg = self.config
        k_n, k_t, mu = cfg.normal_stiffness, cfg.k_t, cfg.friction

        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        member_forces = np.zeros((len(clump), 3))
        penetrations = np.zeros(len(clump))
        tangential_world = np.zeros((len(clump), 3))
        for i in range(len(clump)):
            signed = float((centers[i] - self.point) @ self.normal) - clump.radii[i]
            if signed >= 0.0:
                if commit:
                    self._tangential[i] = 0.0
                    if self._rolling_moment is not None:
                        self._rolling_moment[i] = 0.0
                continue
            penetration = -signed
            penetrations[i] = penetration
            contact_point = centers[i] - clump.radii[i] * self.normal
            arm = contact_point - body.position
            velocity = body.point_velocity(contact_point)
            v_normal = float(velocity @ self.normal)
            v_tangent = velocity - v_normal * self.normal
            m_eff = effective_mass_along(clump, arm, self.normal, inverse_inertia)
            c_n = 2.0 * cfg.damping_ratio * math.sqrt(k_n * m_eff)
            f_n = k_n * penetration - c_n * v_normal
            f_n = max(f_n, 0.0)
            xi_world = self._tangential_world[i] + v_tangent * dt
            f_t = -k_t * xi_world
            magnitude = float(np.linalg.norm(f_t))
            limit = mu * f_n
            if magnitude > limit:
                f_t = f_t * (limit / magnitude) if magnitude > 0.0 else f_t
                if commit:
                    xi_world = -f_t / k_t
                    self._tangential_world[i] = xi_world
                    self._tangential[i] = rotation.T @ xi_world
            elif commit:
                self._tangential_world[i] = xi_world
                self._tangential[i] = rotation.T @ xi_world
            force = f_n * self.normal + f_t
            # rolling / drilling resistance about the member centre
            member_moment = self._resistance(cfg, self.normal, body, f_n, clump.radii[i],
                                             i, dt, commit)
            member_forces[i] = force
            total_force += force
            total_torque += np.cross(arm, force) + member_moment
            tangential_world[i] = f_t
        self._last = {"member_forces": member_forces, "penetrations": penetrations,
                      "tangential": tangential_world}
        return total_force, total_torque

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "normal": self.normal.tolist(),
            "point": self.point.tolist(),
            "active_members": int(np.count_nonzero(self._last["penetrations"])),
            "max_penetration": float(self._last["penetrations"].max()) if len(
                self._last["penetrations"]) else 0.0,
            "net_force": self._last["member_forces"].sum(axis=0).tolist()
            if self._last["member_forces"].size else [0.0, 0.0, 0.0],
        }

    def reset(self) -> None:
        self._tangential = None
        self._tangential_world = None
        ClumpContact.reset(self)

    # diagnostics ---------------------------------------------------------
    def member_forces(self) -> np.ndarray:
        return np.asarray(self._last["member_forces"], dtype=np.float64).copy()

    def penetrations(self) -> np.ndarray:
        return np.asarray(self._last["penetrations"], dtype=np.float64).copy()


class SphereObstacle(ClumpContact):
    """Fixed (or kinematically driven) sphere that the clump can hit.

    The obstacle has infinite mass; :meth:`obstacle_force` returns

    .. code-block:: text

       F_obstacle = - sum_i F_on_clump,i

    which is Newton's third law applied to the summed member contacts.
    """

    name = "sphere"

    def __init__(self, center: Sequence[float], radius: float,
                 velocity: Sequence[float] = (0.0, 0.0, 0.0),
                 config: ContactConfig | None = None, name: str = "obstacle",
                 mass: float = math.inf) -> None:
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("obstacle radius must be finite and positive")
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.config = config or ContactConfig()
        self.name = name
        self.mass = mass
        self._tangential = None
        self._tangential_world = None
        self._init_resistance()
        self._last = {"member_forces": np.zeros((0, 3)), "penetrations": np.zeros(0)}

    def _ensure_state(self, count: int) -> None:
        if self._tangential is None or len(self._tangential) != count:
            self._tangential = np.zeros((count, 3))
            self._tangential_world = np.zeros((count, 3))

    def forces(self, body: "ClumpBody", *, dt: float,
               commit: bool = True) -> tuple[np.ndarray, np.ndarray]:
        clump = body.clump
        self._ensure_state(len(clump))
        rotation = body.rotation_matrix()
        centers = body.member_centers()
        inverse_inertia = body.inverse_inertia_world()
        cfg = self.config
        k_n, k_t, mu = cfg.normal_stiffness, cfg.k_t, cfg.friction

        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        member_forces = np.zeros((len(clump), 3))
        penetrations = np.zeros(len(clump))
        for i in range(len(clump)):
            delta = centers[i] - self.center
            distance = float(np.linalg.norm(delta))
            penetration = clump.radii[i] + self.radius - distance
            if penetration <= 0.0:
                if commit:
                    self._tangential[i] = 0.0
                    self._tangential_world[i][:] = 0.0
                    if self._rolling_moment is not None:
                        self._rolling_moment[i] = 0.0
                continue
            penetrations[i] = penetration
            normal = delta / distance if distance > 0.0 else np.array([0.0, 0.0, 1.0])
            contact_point = centers[i] - clump.radii[i] * normal
            arm = contact_point - body.position
            relative = body.point_velocity(contact_point) - self.velocity
            v_normal = float(relative @ normal)
            v_tangent = relative - v_normal * normal
            m_eff = effective_mass_along(clump, arm, normal, inverse_inertia)
            c_n = 2.0 * cfg.damping_ratio * math.sqrt(k_n * m_eff)
            f_n = max(k_n * penetration - c_n * v_normal, 0.0)
            xi_world = self._tangential_world[i] + v_tangent * dt
            f_t = -k_t * xi_world
            magnitude = float(np.linalg.norm(f_t))
            limit = mu * f_n
            if magnitude > limit:
                f_t = f_t * (limit / magnitude) if magnitude > 0.0 else f_t
                if commit:
                    xi_world = -f_t / k_t
                    self._tangential_world[i] = xi_world
                    self._tangential[i] = rotation.T @ xi_world
            elif commit:
                self._tangential_world[i] = xi_world
                self._tangential[i] = rotation.T @ xi_world
            force = f_n * normal + f_t
            # rolling / drilling resistance (obstacle spin assumed zero)
            member_moment = self._resistance(cfg, normal, body, f_n, clump.radii[i],
                                             i, dt, commit)
            member_forces[i] = force
            total_force += force
            total_torque += np.cross(arm, force) + member_moment
        self._last = {"member_forces": member_forces, "penetrations": penetrations}
        return total_force, total_torque

    def member_forces(self) -> np.ndarray:
        return np.asarray(self._last["member_forces"], dtype=np.float64).copy()

    def penetrations(self) -> np.ndarray:
        return np.asarray(self._last["penetrations"], dtype=np.float64).copy()

    def obstacle_force(self) -> np.ndarray:
        """Net force the clump exerts on the obstacle (third-law partner)."""
        forces = self.member_forces()
        return -forces.sum(axis=0) if forces.size else np.zeros(3)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "center": self.center.tolist(),
            "radius": self.radius,
            "active_members": int(np.count_nonzero(self._last["penetrations"])),
            "max_penetration": float(self._last["penetrations"].max()) if len(
                self._last["penetrations"]) else 0.0,
            "obstacle_force": self.obstacle_force().tolist(),
        }

    def reset(self) -> None:
        self._tangential = None
        self._tangential_world = None
        ClumpContact.reset(self)


# --------------------------------------------------------------------------
# Rigid body
# --------------------------------------------------------------------------
class ClumpBody:
    """Six-DOF rigid clump: translation of the COM plus body-frame rotation."""

    def __init__(self, clump: Clump, *, position: Sequence[float] = (0.0, 0.0, 0.0),
                 velocity: Sequence[float] = (0.0, 0.0, 0.0),
                 orientation: Sequence[float] = _IDENTITY_QUATERNION,
                 angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
                 gravity: Sequence[float] = (0.0, 0.0, -9.81)) -> None:
        self.clump = clump
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.asarray(velocity, dtype=np.float64).copy()
        quaternion = np.asarray(orientation, dtype=np.float64)
        self.orientation = quaternion / float(np.linalg.norm(quaternion))
        # body-frame angular velocity; I_b is constant so Euler's equations apply
        self.angular_velocity = np.asarray(angular_velocity, dtype=np.float64).copy()
        self.gravity = np.asarray(gravity, dtype=np.float64).copy()
        self.inertia_body = clump.inertia_about_com
        self.inverse_inertia_body = np.linalg.inv(self.inertia_body)
        self.potential_reference = 0.0

    # -- kinematics -------------------------------------------------------
    def rotation_matrix(self) -> np.ndarray:
        return quaternion_to_matrix(self.orientation)

    def inverse_inertia_world(self) -> np.ndarray:
        rotation = self.rotation_matrix()
        return rotation @ self.inverse_inertia_body @ rotation.T

    def member_centers(self) -> np.ndarray:
        return self.position + self.clump.local_centers @ self.rotation_matrix().T

    def angular_velocity_world(self) -> np.ndarray:
        return self.rotation_matrix() @ self.angular_velocity

    def point_velocity(self, point: Sequence[float]) -> np.ndarray:
        """Velocity of the material point of the rigid clump at ``point``.

        Notes
        -----
        Contacts must use the *contact point* arm ``x_contact - x_com``.  Using
        the member-centre arm instead silently drops the ``omega x (-r n)``
        contribution; for a single-sphere clump that removes the rolling
        kinematics entirely (the body then spins in place with a frozen contact
        point), which is the defect this helper exists to prevent.
        """
        arm = np.asarray(point, dtype=np.float64) - self.position
        return self.velocity + np.cross(self.angular_velocity_world(), arm)

    def member_contact_velocities(self) -> np.ndarray:
        """Velocity of each member's centre (not of its contact point)."""
        omega = self.angular_velocity_world()
        arms = self.clump.local_centers @ self.rotation_matrix().T
        return self.velocity + np.cross(np.broadcast_to(omega, arms.shape), arms)

    # -- energetics -------------------------------------------------------
    def kinetic_energy(self) -> float:
        translational = 0.5 * self.clump.mass * float(self.velocity @ self.velocity)
        rotational = 0.5 * float(
            self.angular_velocity @ self.inertia_body @ self.angular_velocity)
        return translational + rotational

    def potential_energy(self) -> float:
        return -self.clump.mass * float(self.gravity @ self.position)

    def energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def angular_momentum_world(self) -> np.ndarray:
        return self.rotation_matrix() @ (self.inertia_body @ self.angular_velocity)

    # -- dynamics ---------------------------------------------------------
    def net_force(self, contacts: Iterable[ClumpContact], *, dt: float,
                  commit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        force = self.clump.mass * self.gravity
        torque = np.zeros(3)
        for contact in contacts:
            f, t = contact.forces(self, dt=dt, commit=commit)
            force = force + f
            torque = torque + t
        return force, torque

    def angular_acceleration_body(self, torque_world: np.ndarray) -> np.ndarray:
        torque_body = self.rotation_matrix().T @ torque_world
        omega = self.angular_velocity
        gyroscopic = np.cross(omega, self.inertia_body @ omega)
        return self.inverse_inertia_body @ (torque_body - gyroscopic)

    def step(self, dt: float, contacts: Sequence[ClumpContact] = ()) -> None:
        """One velocity-Verlet step; friction history is committed once."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if not isinstance(contacts, (list, tuple)):
            contacts = tuple(contacts)
        force0, torque0 = self.net_force(contacts, dt=dt, commit=False)
        accel0 = force0 / self.clump.mass
        alpha0 = self.angular_acceleration_body(torque0)

        velocity_half = self.velocity + 0.5 * dt * accel0
        omega_half = self.angular_velocity + 0.5 * dt * alpha0
        self.position = self.position + dt * velocity_half

        spin = float(np.linalg.norm(omega_half))
        if spin > 0.0:
            increment = quaternion_from_axis_angle(omega_half, spin * dt)
            self.orientation = quaternion_multiply(self.orientation, increment)
            self.orientation /= float(np.linalg.norm(self.orientation))

        self.velocity = velocity_half
        self.angular_velocity = omega_half

        force1, torque1 = self.net_force(contacts, dt=dt, commit=True)
        accel1 = force1 / self.clump.mass
        alpha1 = self.angular_acceleration_body(torque1)
        self.velocity = velocity_half + 0.5 * dt * accel1
        self.angular_velocity = omega_half + 0.5 * dt * alpha1

    def run(self, n_steps: int, dt: float, contacts: Sequence[ClumpContact] = (),
            record_every: int = 1) -> dict[str, np.ndarray]:
        """Integrate ``n_steps`` steps and return a monitor history."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        record_every = max(int(record_every), 1)
        history: dict[str, list] = {key: [] for key in (
            "time", "position", "velocity", "omega_body", "quaternion",
            "kinetic", "potential", "total_energy", "angular_momentum")}
        for step in range(n_steps + 1):
            if step % record_every == 0:
                history["time"].append(step * dt)
                history["position"].append(self.position.copy())
                history["velocity"].append(self.velocity.copy())
                history["omega_body"].append(self.angular_velocity.copy())
                history["quaternion"].append(self.orientation.copy())
                history["kinetic"].append(self.kinetic_energy())
                history["potential"].append(self.potential_energy())
                history["total_energy"].append(self.energy())
                history["angular_momentum"].append(self.angular_momentum_world())
            if step == n_steps:
                break
            self.step(dt, contacts)
        return {key: np.asarray(value) for key, value in history.items()}

    def summary(self) -> dict[str, object]:
        return {
            "clump": self.clump.summary(),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "orientation": self.orientation.tolist(),
            "omega_body": self.angular_velocity.tolist(),
            "kinetic_energy": self.kinetic_energy(),
            "potential_energy": self.potential_energy(),
            "total_energy": self.energy(),
        }