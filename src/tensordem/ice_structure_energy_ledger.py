"""Ice-structure contact-collision energy-conservation audit (four-ledger closure).

Why this module exists
----------------------
Ice-going structures (ship bows, offshore lighthouses, jacket legs) are sized by
the *energy* delivered in a floe impact, not only by the peak force.  A DEM model
of that impact is only trustworthy if it can show, step by step, where the initial
kinetic energy went.  This module embeds a four-ledger energy audit inside the DEM
time loop and reports the closure residual every step:

* **Ledger 1 -- kinetic energy**  ``E_k = sum(1/2 m_i |v_i|^2) + sum(1/2 I_i |w_i|^2)``
  (translation + rotation; ``I = 2/5 m r^2`` for a sphere).
* **Ledger 2 -- contact strain energy**
  ``E_s = sum(1/2 k_n dn^2 + 1/2 k_t |dt|^2) + sum(1/2 (1-D) k_b e^2)``
  (normal + tangential contact springs plus the recoverable part of the
  damaged cohesive bonds).
* **Ledger 3 -- fracture energy**  ``E_f = sum_broken G_f * A_crack``
  accumulated as the dissipated work of the cohesive traction-separation law.
  The law is built so that the area under the traction curve is exactly
  ``G_f`` per unit area, hence a fully separated bond contributes exactly
  ``G_f A`` to ``E_f``.
* **Ledger 4 -- viscous dissipation**
  ``E_v = sum_contacts int (c_n v_n^2 + c_t |v_t|^2) dt``
  accumulated every step with exactly the integrand of the task definition.

Closure: ``E_total(t) = E_k + E_s + E_f + E_v``; the reported residual is
``eps_E = |E_total(t) - E_total(0)| / E_total(0)``.

Model notes / scope
-------------------
* Spheres, translational + rotational degrees of freedom, **no gravity**: a
  gravity-free impact is the cleanest energy audit because gravity would add an
  external potential-well term that is not part of the four ledgers.
* Unbonded contacts: linear normal spring + dashpot, Cundall-Strack tangential
  spring + dashpot.  Viscous coefficients follow from a target restitution
  ``e``: ``zeta = |ln e| / sqrt(pi^2 + ln^2 e)``, ``c = 2 zeta sqrt(k_n m_eff)``.
* Bonds are **axial only** (normal spring + cohesive damage law).  Shear
  stiffness of the ice lattice comes from the bond *topology* (axis + face
  diagonal bonds); keeping the bond axial makes the fracture bookkeeping
  unambiguous because the stored/released energy is a scalar function of the
  bond extension.
* A bond becomes unbonded (and can then carry an ordinary contact spring) once
  its damage reaches 1.  Damage only grows in tension, so a fully broken bond is
  always in the separated configuration at that instant and no spurious contact
  strain energy is created at the break event (fragment re-contact is out of
  scope for this audit node).
* Channels that are *not* part of the four ledgers are recorded separately and
  deliberately left inside the residual: normal-force clamping (``F_n >= 0``),
  Coulomb friction sliding work, and tangential-spring energy discarded when a
  contact opens.  Keeping them out of ``E_v`` is what makes the residual an
  honest check rather than a tautology.

All quantities are SI and float64.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

__all__ = [
    "ContactLaw",
    "CohesiveBondLaw",
    "IceStructureEnergyAudit",
    "build_bonded_lattice",
    "damping_from_restitution",
    "kinetic_energy_reference",
    "contact_strain_energy_reference",
    "bond_dissipation_reference",
    "viscous_dissipation_reference",
    "run_energy_audit_case",
    "LEDGER_KEYS",
]

LEDGER_KEYS = ("E_k", "E_s", "E_f", "E_v")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactLaw:
    """Unbonded contact law (linear spring-dashpot, Cundall-Strack tangential)."""

    k_normal: float = 1.0e7
    kt_ratio: float = 0.5
    restitution: float = 1.0
    friction: float = 5.0

    def __post_init__(self) -> None:
        if self.k_normal <= 0.0:
            raise ValueError("k_normal must be positive")
        if not 0.0 < self.restitution <= 1.0:
            raise ValueError("restitution must lie in (0, 1]")
        if self.friction <= 0.0:
            raise ValueError("friction must be positive")

    @property
    def k_tangential(self) -> float:
        return self.kt_ratio * self.k_normal


@dataclass(frozen=True)
class CohesiveBondLaw:
    """Axial cohesive bond: elastic -> linear softening -> broken.

    ``e0 = sigma_t A / k`` is the elastic limit and ``e_f = 2 G_f / sigma_t`` the
    separation at which the traction vanishes, so the total area under the
    traction-separation curve is ``sigma_t e_f / 2 = G_f`` per unit area and a
    fully separated bond dissipates exactly ``G_f * A``.
    """

    stiffness: float = 1.0e7
    area: float = 1.0e-3
    tensile_strength: float = 1.0e5
    fracture_energy: float = 10.0

    def __post_init__(self) -> None:
        for name in ("stiffness", "area", "tensile_strength", "fracture_energy"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        # Griffith consistency: the fracture energy must exceed the elastic
        # energy stored at peak traction, otherwise the softening branch has a
        # negative length and the traction-separation law is not admissible.
        if self.fracture_energy <= self.minimum_fracture_energy:
            raise ValueError(
                "fracture_energy must exceed sigma_t^2 A / (2 k) = "
                f"{self.minimum_fracture_energy:.6g} J/m^2 for an admissible "
                "traction-separation law"
            )

    @property
    def minimum_fracture_energy(self) -> float:
        """Griffith-consistent lower bound ``sigma_t^2 A / (2 k)`` (J/m^2)."""
        return (self.tensile_strength ** 2 * self.area
                / (2.0 * self.stiffness))

    @property
    def elastic_limit(self) -> float:
        """Tensile extension at peak traction."""
        return self.tensile_strength * self.area / self.stiffness

    @property
    def separation(self) -> float:
        """Extension at which the traction has fallen to zero."""
        return 2.0 * self.fracture_energy / self.tensile_strength

    @property
    def peak_force(self) -> float:
        return self.tensile_strength * self.area

    @property
    def surface_energy(self) -> float:
        """``G_f * A``: dissipation of one fully separated bond (J)."""
        return self.fracture_energy * self.area

    def secant_stiffness(self, max_extension: np.ndarray) -> np.ndarray:
        """Secant stiffness of the damage model, ``F(e) = k_eff(max) * e``.

        The traction envelope is linear in the *force*::

            F(e) = k e                       for e <= e0
            F(e) = peak (ef - e)/(ef - e0)   for e0 < e < ef
            F(e) = 0                         for e >= ef

        with ``peak = sigma_t A = k e0``.  Its total area is
        ``peak * ef / 2 = G_f A``, which is why a fully separated bond
        dissipates exactly ``G_f A``.  Unloading/reloading follows the secant,
        i.e. the stored energy is ``1/2 k_eff e^2``.
        """
        e0, ef = self.elastic_limit, self.separation
        em = np.clip(np.asarray(max_extension, dtype=float), 0.0, None)
        f_env = np.where(em <= e0, self.stiffness * em,
                         np.where(em < ef,
                                  self.peak_force * (ef - em) / (ef - e0), 0.0))
        k_eff = np.where(em > 1.0e-30, f_env / np.maximum(em, 1.0e-30),
                         self.stiffness)
        return np.minimum(k_eff, self.stiffness)

    def damage(self, max_extension: np.ndarray) -> np.ndarray:
        return 1.0 - self.secant_stiffness(max_extension) / self.stiffness


def damping_from_restitution(restitution: float, stiffness: float,
                             effective_mass: float) -> float:
    """Dashpot coefficient matching a target restitution (linear contact)."""
    if restitution >= 1.0:
        return 0.0
    ln_e = math.log(max(restitution, 1.0e-6))
    zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)
    return 2.0 * zeta * math.sqrt(stiffness * effective_mass)


# ---------------------------------------------------------------------------
# geometry helper
# ---------------------------------------------------------------------------
def build_bonded_lattice(nx: int, ny: int, nz: int, radius: float,
                         density: float = 900.0) -> dict[str, Any]:
    """Cubic array of touching spheres bonded to axis + face-diagonal neighbours.

    The face diagonals remove the shear mechanism of a pure cubic axial lattice,
    so the floe behaves as a solid body up to fracture.
    """
    xs = (np.arange(nx) - (nx - 1) / 2.0) * 2.0 * radius
    ys = (np.arange(ny) - (ny - 1) / 2.0) * 2.0 * radius
    zs = (np.arange(nz) - (nz - 1) / 2.0) * 2.0 * radius
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    n = grid.shape[0]
    i_idx: list[int] = []
    j_idx: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            step = np.abs(grid[j] - grid[i]) / (2.0 * radius)
            rounded = np.round(step)
            if not np.all(np.abs(step - rounded) < 1e-9):
                continue
            nonzero = rounded[rounded > 0.0]
            # axis neighbours (one unit step) and face diagonals (two unit steps)
            if nonzero.size in (1, 2) and np.all(nonzero == 1.0):
                i_idx.append(i)
                j_idx.append(j)
    mass = density * 4.0 / 3.0 * math.pi * radius ** 3
    return {
        "positions": grid,
        "bond_i": np.asarray(i_idx, dtype=int),
        "bond_j": np.asarray(j_idx, dtype=int),
        "radius": radius,
        "density": density,
        "mass": mass,
        "inertia": 0.4 * mass * radius ** 2,
    }


# ---------------------------------------------------------------------------
# main simulator + auditor
# ---------------------------------------------------------------------------
@dataclass
class _Ledger:
    """Cumulative four-ledger state."""

    E_k: float = 0.0
    E_s: float = 0.0
    E_f: float = 0.0
    E_v: float = 0.0
    E_k_trans: float = 0.0
    E_k_rot: float = 0.0
    E_s_contact: float = 0.0
    E_s_contact_normal: float = 0.0
    E_s_contact_tangential: float = 0.0
    E_s_bond: float = 0.0
    E_v_normal: float = 0.0
    E_v_tangential: float = 0.0

    @property
    def total(self) -> float:
        return self.E_k + self.E_s + self.E_f + self.E_v


class IceStructureEnergyAudit:
    """DEM impact of a bonded ice floe against a rigid structure + energy audit."""

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        radius: float,
        density: float = 900.0,
        contact: ContactLaw = ContactLaw(),
        bond_law: CohesiveBondLaw = CohesiveBondLaw(),
        bond_i: np.ndarray | None = None,
        bond_j: np.ndarray | None = None,
        dt: float = 5.0e-6,
        wall_plane_x: float | None = None,
        wall_velocity: float = 0.0,
        omegas: np.ndarray | None = None,
    ) -> None:
        self.positions = np.array(positions, dtype=np.float64, copy=True)
        self.velocities = np.array(velocities, dtype=np.float64, copy=True)
        n = self.positions.shape[0]
        self.omegas = (np.zeros((n, 3)) if omegas is None
                       else np.array(omegas, dtype=np.float64, copy=True))
        self.radius = float(radius)
        self.density = float(density)
        self.mass = self.density * 4.0 / 3.0 * math.pi * self.radius ** 3
        self.inertia = 0.4 * self.mass * self.radius ** 2
        self.contact = contact
        self.bond_law = bond_law
        self.dt = float(dt)
        if bond_i is None:
            bond_i = np.zeros(0, dtype=int)
        if bond_j is None:
            bond_j = np.zeros(0, dtype=int)
        self.bond_i = np.asarray(bond_i, dtype=int).copy()
        self.bond_j = np.asarray(bond_j, dtype=int).copy()
        nb = self.bond_i.size
        self._rest_length = (np.linalg.norm(self.positions[self.bond_j]
                                            - self.positions[self.bond_i], axis=1)
                             if nb else np.zeros(0))
        self.bond_damage = np.zeros(nb)
        self.bond_max_extension = np.zeros(nb)
        self.bond_alive = np.ones(nb, dtype=bool)
        self._bond_lookup: dict[tuple[int, int], int] = {
            (int(a), int(b)): k for k, (a, b) in enumerate(zip(self.bond_i, self.bond_j))
        }

        # rigid structure modelled as a plane wall at x = wall_x (normal +x)
        if wall_plane_x is None:
            wall_plane_x = float(self.positions[:, 0].min() - 0.05)
        self.wall_x = float(wall_plane_x)
        self.wall_velocity = float(wall_velocity)

        # contact memory (Cundall-Strack tangential displacement)
        self._tangential: dict[tuple, np.ndarray] = {}
        # dissipative channels deliberately kept OUTSIDE the four ledgers
        self.clamp_loss_J = 0.0
        self.friction_sliding_J = 0.0
        self.spring_release_J = 0.0
        self.wall_work_J = 0.0

        self.ledger = _Ledger()
        self.n_broken_total = 0
        self.max_tangential_utilization = 0.0
        self.peak_wall_force_N = 0.0
        self.max_penetration_m = 0.0
        self._bond_force_prev: np.ndarray | None = None
        self._bond_ext_prev: np.ndarray | None = None
        self._bond_stored_prev: np.ndarray | None = None
        self.time = 0.0
        self.history: list[dict[str, float]] = []

        self._force, self._torque, self._diag = self._evaluate(
            self.positions, self.velocities, self.omegas, update_tangential=False
        )
        self._refresh_state_ledger()
        self.reference_total_J = self.ledger.total
        self._record()

    # -- state ledgers ------------------------------------------------------
    def _refresh_state_ledger(self) -> None:
        v2 = np.einsum("ij,ij->i", self.velocities, self.velocities)
        w2 = np.einsum("ij,ij->i", self.omegas, self.omegas)
        self.ledger.E_k_trans = 0.5 * self.mass * float(v2.sum())
        self.ledger.E_k_rot = 0.5 * self.inertia * float(w2.sum())
        self.ledger.E_k = self.ledger.E_k_trans + self.ledger.E_k_rot
        self.ledger.E_s = self.ledger.E_s_contact + self.ledger.E_s_bond

    # -- force evaluation ---------------------------------------------------
    def _evaluate(self, positions, velocities, omegas, update_tangential):
        n = positions.shape[0]
        force = np.zeros((n, 3))
        torque = np.zeros((n, 3))
        rate_n = 0.0
        rate_t = 0.0
        clamp_loss = 0.0
        fric_loss = 0.0
        s_normal = 0.0
        s_tangential = 0.0
        live_keys: set[tuple] = set()

        # ---- cohesive bonds ------------------------------------------------
        if self.bond_i.size:
            delta = positions[self.bond_j] - positions[self.bond_i]
            dist = np.linalg.norm(delta, axis=1)
            ext = dist - self._rest_length
            law = self.bond_law
            alive = self.bond_alive
            grow = alive & (ext > 0.0) & (ext > self.bond_max_extension)
            self.bond_max_extension = np.where(grow, ext, self.bond_max_extension)
            secant = law.secant_stiffness(self.bond_max_extension)
            stiffness = np.where(alive, secant, 0.0)
            damage = 1.0 - stiffness / law.stiffness
            magnitude = np.where(alive, stiffness * ext, 0.0)
            unit = delta / np.maximum(dist, 1e-30)[:, None]
            # unit points from i to j; a tensile bond (magnitude>0) pulls the
            # pair together, so j feels -magnitude*unit and i feels +magnitude*unit
            pair_force = -magnitude[:, None] * unit        # force on j
            np.add.at(force, self.bond_i, -pair_force)
            np.add.at(force, self.bond_j, pair_force)

            stored = np.where(alive, 0.5 * stiffness * ext ** 2, 0.0)
            if self._bond_force_prev is not None:
                work = 0.5 * (self._bond_force_prev + magnitude) * (ext - self._bond_ext_prev)
                energy_change = stored - self._bond_stored_prev
                inc = np.where(alive, work - energy_change, 0.0)
                self.ledger.E_f += float(np.maximum(inc, 0.0).sum())
            self._bond_force_prev = np.where(alive, magnitude, 0.0)
            self._bond_ext_prev = ext.copy()
            self._bond_stored_prev = stored
            self.bond_damage = np.where(alive, damage, 1.0)
            self.ledger.E_s_bond = float(stored.sum())
            newly_dead = alive & (damage >= 1.0)
            if newly_dead.any():
                self.bond_alive = alive & ~newly_dead
                self.n_broken_total += int(newly_dead.sum())
                self._bond_force_prev = np.where(self.bond_alive, self._bond_force_prev, 0.0)
                self._bond_stored_prev = np.where(self.bond_alive, self._bond_stored_prev, 0.0)
        else:
            self.ledger.E_s_bond = 0.0

        # ---- particle-particle contacts ------------------------------------
        iu, ju = np.triu_indices(n, k=1)
        if iu.size:
            delta = positions[ju] - positions[iu]
            dist = np.linalg.norm(delta, axis=1)
            overlap = 2.0 * self.radius - dist
            bonded_alive = np.zeros(iu.size, dtype=bool)
            for k, (a, b) in enumerate(zip(iu, ju)):
                idx = self._bond_lookup.get((int(a), int(b)))
                if idx is not None and self.bond_alive[idx]:
                    bonded_alive[k] = True
            active = (overlap > 0.0) & ~bonded_alive
            if active.any():
                idx = np.nonzero(active)[0]
                p_i, p_j = iu[idx], ju[idx]
                ov = overlap[idx]
                d = dist[idx]
                unit = delta[idx] / d[:, None]
                r_i = self.radius * unit
                r_j = -self.radius * unit
                v_i = velocities[p_i] + np.cross(omegas[p_i], r_i)
                v_j = velocities[p_j] + np.cross(omegas[p_j], r_j)
                v_rel = v_j - v_i
                v_n = np.einsum("ij,ij->i", v_rel, unit)
                v_t = v_rel - v_n[:, None] * unit
                cn = damping_from_restitution(self.contact.restitution,
                                              self.contact.k_normal, self.mass / 2.0)
                ct = damping_from_restitution(self.contact.restitution,
                                              self.contact.k_tangential, self.mass / 2.0)
                normal_mag = self.contact.k_normal * ov - cn * v_n
                clamped = np.maximum(normal_mag, 0.0)
                clamp_loss += float((0.5 * (normal_mag - clamped) * ov).sum())
                normal_mag = clamped

                keys = [(int(a), int(b)) for a, b in zip(p_i, p_j)]
                for k, key in enumerate(keys):
                    tang = self._tangential.get(key)
                    if tang is None:
                        tang = np.zeros(3)
                    if update_tangential:
                        tang = tang + v_t[k] * self.dt
                        tang = tang - np.dot(tang, unit[k]) * unit[k]
                    self._tangential[key] = tang
                tang_stack = np.array([self._tangential[key] for key in keys])
                f_t = -self.contact.k_tangential * tang_stack - ct * v_t
                cap = self.contact.friction * normal_mag
                f_t_mag = np.linalg.norm(f_t, axis=1)
                loaded = cap > 0.0
                if loaded.any():
                    util = f_t_mag[loaded] / cap[loaded]
                    self.max_tangential_utilization = max(
                        self.max_tangential_utilization, float(util.max()))
                over = f_t_mag > cap
                if over.any():
                    fric_loss += float(np.abs(
                        (f_t_mag[over] - cap[over])
                        * np.linalg.norm(v_t[over], axis=1) * self.dt).sum())
                    f_t = np.where(over[:, None],
                                   f_t * (cap / np.maximum(f_t_mag, 1e-30))[:, None], f_t)
                f_on_i = -(normal_mag[:, None] * unit) + f_t
                f_on_j = -f_on_i
                np.add.at(force, p_i, f_on_i)
                np.add.at(force, p_j, f_on_j)
                np.add.at(torque, p_i, np.cross(r_i, f_on_i))
                np.add.at(torque, p_j, np.cross(r_j, f_on_j))

                s_normal += float((0.5 * self.contact.k_normal * ov ** 2).sum())
                s_tangential += float((0.5 * self.contact.k_tangential
                                       * np.einsum("ij,ij->i", tang_stack, tang_stack)).sum())
                rate_n += float((cn * v_n ** 2).sum())
                rate_t += float((ct * np.einsum("ij,ij->i", v_t, v_t)).sum())
                live_keys.update(keys)

        # ---- contacts that just opened: release tangential spring ----------
        for key in list(self._tangential.keys()):
            if isinstance(key[0], str):
                continue
            if key not in live_keys:
                tang = self._tangential.pop(key)
                self.spring_release_J += float(
                    0.5 * self.contact.k_tangential * np.dot(tang, tang))

        # ---- rigid structure (plane wall) ----------------------------------
        gap = positions[:, 0] - self.wall_x
        ov_w = self.radius - gap
        act = ov_w > 0.0
        wall_keys: list[tuple] = []
        if act.any():
            pid = np.nonzero(act)[0]
            ovv = ov_w[pid]
            normal = np.array([1.0, 0.0, 0.0])
            r_i = np.tile(-self.radius * normal, (pid.size, 1))
            v_wall = np.array([self.wall_velocity, 0.0, 0.0])
            v_i = velocities[pid] + np.cross(omegas[pid], r_i)
            v_rel = v_i - v_wall
            v_n = v_rel[:, 0]
            v_t = v_rel - v_n[:, None] * normal
            cn = damping_from_restitution(self.contact.restitution,
                                          self.contact.k_normal, self.mass)
            ct = damping_from_restitution(self.contact.restitution,
                                          self.contact.k_tangential, self.mass)
            normal_mag = self.contact.k_normal * ovv - cn * v_n
            clamped = np.maximum(normal_mag, 0.0)
            clamp_loss += float((0.5 * (normal_mag - clamped) * ovv).sum())
            normal_mag = clamped
            wall_keys = [("wall", int(p)) for p in pid]
            for k, key in enumerate(wall_keys):
                tang = self._tangential.get(key)
                if tang is None:
                    tang = np.zeros(3)
                if update_tangential:
                    tang = tang + v_t[k] * self.dt
                    tang = tang - np.dot(tang, normal) * normal
                self._tangential[key] = tang
            tang_stack = np.array([self._tangential[key] for key in wall_keys])
            f_t = -self.contact.k_tangential * tang_stack - ct * v_t
            cap = self.contact.friction * normal_mag
            f_t_mag = np.linalg.norm(f_t, axis=1)
            loaded = cap > 0.0
            if loaded.any():
                util = f_t_mag[loaded] / cap[loaded]
                self.max_tangential_utilization = max(
                    self.max_tangential_utilization, float(util.max()))
            over = f_t_mag > cap
            if over.any():
                fric_loss += float(np.abs(
                    (f_t_mag[over] - cap[over])
                    * np.linalg.norm(v_t[over], axis=1) * self.dt).sum())
                f_t = np.where(over[:, None],
                               f_t * (cap / np.maximum(f_t_mag, 1e-30))[:, None], f_t)
            self.peak_wall_force_N = max(self.peak_wall_force_N,
                                         float(normal_mag.sum()))
            self.max_penetration_m = max(self.max_penetration_m,
                                         float(ovv.max()))
            f_on_i = normal_mag[:, None] * normal + f_t
            np.add.at(force, pid, f_on_i)
            np.add.at(torque, pid, np.cross(r_i, f_on_i))
            s_normal += float((0.5 * self.contact.k_normal * ovv ** 2).sum())
            s_tangential += float((0.5 * self.contact.k_tangential
                                   * np.einsum("ij,ij->i", tang_stack, tang_stack)).sum())
            rate_n += float((cn * v_n ** 2).sum())
            rate_t += float((ct * np.einsum("ij,ij->i", v_t, v_t)).sum())
            if self.wall_velocity != 0.0:
                self.wall_work_J += float((-normal_mag * self.wall_velocity).sum() * self.dt)

        for key in list(self._tangential.keys()):
            if isinstance(key[0], str) and key not in wall_keys:
                tang = self._tangential.pop(key)
                self.spring_release_J += float(
                    0.5 * self.contact.k_tangential * np.dot(tang, tang))

        self.clamp_loss_J += clamp_loss
        self.friction_sliding_J += fric_loss
        self.ledger.E_s_contact_normal = s_normal
        self.ledger.E_s_contact_tangential = s_tangential
        self.ledger.E_s_contact = s_normal + s_tangential
        diag = {"rate_normal": rate_n, "rate_tangential": rate_t}
        return force, torque, diag

    # -- time integration ---------------------------------------------------
    def step(self) -> dict[str, float]:
        dt = self.dt
        m, inertia = self.mass, self.inertia
        v_half = self.velocities + 0.5 * dt * self._force / m
        w_half = self.omegas + 0.5 * dt * self._torque / inertia
        new_positions = self.positions + dt * v_half
        force, torque, diag = self._evaluate(new_positions, v_half, w_half,
                                             update_tangential=True)
        self.velocities = v_half + 0.5 * dt * force / m
        self.omegas = w_half + 0.5 * dt * torque / inertia
        self.positions = new_positions
        self._force, self._torque, self._diag = force, torque, diag

        # viscous dissipation: exact integrand of the task definition, left
        # Riemann over the step (the rate is evaluated at the new state)
        self.ledger.E_v_normal += diag["rate_normal"] * dt
        self.ledger.E_v_tangential += diag["rate_tangential"] * dt
        self.ledger.E_v += (diag["rate_normal"] + diag["rate_tangential"]) * dt

        self.time += dt
        return self._record()

    def _record(self) -> dict[str, float]:
        self._refresh_state_ledger()
        led = self.ledger
        total = led.total
        ref = self.reference_total_J
        rate = (0.0 if not self.history
                else self._diag["rate_normal"] + self._diag["rate_tangential"])
        row = {
            "t": self.time,
            "E_k": led.E_k,
            "E_k_trans": led.E_k_trans,
            "E_k_rot": led.E_k_rot,
            "E_s": led.E_s,
            "E_s_contact": led.E_s_contact,
            "E_s_contact_normal": led.E_s_contact_normal,
            "E_s_contact_tangential": led.E_s_contact_tangential,
            "E_s_bond": led.E_s_bond,
            "E_f": led.E_f,
            "E_v": led.E_v,
            "E_v_normal": led.E_v_normal,
            "E_v_tangential": led.E_v_tangential,
            "E_total": total,
            "residual_J": total - ref,
            "residual_relative": abs(total - ref) / max(abs(ref), 1e-30),
            "dissipation_rate": rate,
            "n_broken": float(self.n_broken_total),
            "clamp_loss_J": self.clamp_loss_J,
            "friction_sliding_J": self.friction_sliding_J,
            "spring_release_J": self.spring_release_J,
        }
        self.history.append(row)
        return row

    def run(self, n_steps: int) -> list[dict[str, float]]:
        for _ in range(n_steps):
            self.step()
        return self.history

    def run_until(self, t_end: float) -> list[dict[str, float]]:
        n = int(round((t_end - self.time) / self.dt))
        return self.run(max(n, 0))

    # -- summary ------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        led = self.ledger
        ref = self.reference_total_J
        rows = self.history
        return {
            "n_steps": len(rows),
            "dt": self.dt,
            "t_end": self.time,
            "E_total_0": ref,
            "E_k": led.E_k,
            "E_s": led.E_s,
            "E_f": led.E_f,
            "E_v": led.E_v,
            "E_total": led.total,
            "residual_relative_final": abs(led.total - ref) / max(abs(ref), 1e-30),
            "residual_relative_max": max((r["residual_relative"] for r in rows),
                                         default=0.0),
            "n_broken": self.n_broken_total,
            "max_tangential_utilization": self.max_tangential_utilization,
            "clamp_loss_J": self.clamp_loss_J,
            "friction_sliding_J": self.friction_sliding_J,
            "spring_release_J": self.spring_release_J,
            "wall_work_J": self.wall_work_J,
            "peak_wall_force_N": self.peak_wall_force_N,
            "max_penetration_m": self.max_penetration_m,
        }


# ---------------------------------------------------------------------------
# independent reference implementations (deliberately separate code paths)
# ---------------------------------------------------------------------------
def kinetic_energy_reference(velocities: np.ndarray, omegas: np.ndarray,
                             mass: float, inertia: float) -> dict[str, float]:
    """Literal transcription of the textbook definition (no shared code)."""
    trans = 0.0
    for i in range(velocities.shape[0]):
        trans += 0.5 * mass * (velocities[i, 0] ** 2 + velocities[i, 1] ** 2
                               + velocities[i, 2] ** 2)
    rot = 0.0
    for i in range(omegas.shape[0]):
        rot += 0.5 * inertia * (omegas[i, 0] ** 2 + omegas[i, 1] ** 2
                                + omegas[i, 2] ** 2)
    return {"E_k_trans": trans, "E_k_rot": rot, "E_k": trans + rot}


def contact_strain_energy_reference(sim: "IceStructureEnergyAudit") -> dict[str, float]:
    """Spring-potential sum re-derived from raw positions (independent path)."""
    k_n = sim.contact.k_normal
    k_t = sim.contact.k_tangential
    normal = 0.0
    tangential = 0.0
    pos = sim.positions
    rad = sim.radius

    bond = 0.0
    for k in range(sim.bond_i.size):
        if not sim.bond_alive[k]:
            continue
        i, j = int(sim.bond_i[k]), int(sim.bond_j[k])
        d = math.dist(pos[i], pos[j])
        ext = d - float(sim._rest_length[k])
        stiffness = float(sim.bond_law.secant_stiffness(
            np.array([sim.bond_max_extension[k]]))[0])
        bond += 0.5 * stiffness * ext ** 2

    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            idx = sim._bond_lookup.get((i, j))
            if idx is not None and sim.bond_alive[idx]:
                continue
            ov = 2.0 * rad - math.dist(pos[i], pos[j])
            if ov > 0.0:
                normal += 0.5 * k_n * ov ** 2
                tang = sim._tangential.get((i, j))
                if tang is not None:
                    tangential += 0.5 * k_t * float(np.dot(tang, tang))
        ov_w = rad - (pos[i, 0] - sim.wall_x)
        if ov_w > 0.0:
            normal += 0.5 * k_n * ov_w ** 2
            tang = sim._tangential.get(("wall", i))
            if tang is not None:
                tangential += 0.5 * k_t * float(np.dot(tang, tang))

    return {"E_s_contact_normal": normal, "E_s_contact_tangential": tangential,
            "E_s_bond": bond, "E_s": normal + tangential + bond}


def bond_dissipation_reference(law: CohesiveBondLaw,
                               max_extension: np.ndarray) -> dict[str, float]:
    """Fine-grid quadrature of the cohesive law (independent of the step loop)."""
    ef = law.separation
    e0 = law.elastic_limit
    peak = law.peak_force
    fully = max_extension >= ef
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    per_bond = np.zeros(len(max_extension))
    for k, e_max in enumerate(max_extension):
        e_max = float(e_max)
        if e_max <= 1.0e-14:
            continue
        grid = np.linspace(0.0, e_max, 40001)
        traction = np.where(grid <= e0, law.stiffness * grid,
                            np.where(grid < ef, peak * (ef - grid) / (ef - e0), 0.0))
        work = float(trapezoid(traction, grid))
        k_eff = float(law.secant_stiffness(np.array([e_max]))[0])
        stored = 0.5 * k_eff * e_max ** 2
        per_bond[k] = max(work - stored, 0.0)
    return {
        "E_f_reference": float(per_bond.sum()),
        "E_f_fully_separated": float(fully.sum()) * law.surface_energy,
        "n_fully_separated": int(fully.sum()),
        "per_bond": per_bond,
    }


def viscous_dissipation_reference(rows: Iterable[dict[str, float]]) -> float:
    """Trapezoidal numerical integration of the recorded dissipation rate."""
    rows = list(rows)
    if len(rows) < 2:
        return 0.0
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    t = np.array([r["t"] for r in rows])
    rate = np.array([r.get("dissipation_rate", 0.0) for r in rows])
    return float(trapezoid(rate, t))


# ---------------------------------------------------------------------------
# convenience case runner
# ---------------------------------------------------------------------------
def run_energy_audit_case(
    *,
    n_cells: tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.05,
    density: float = 900.0,
    impact_velocity: tuple[float, float, float] = (-2.0, 0.0, 0.0),
    gap: float = 0.01,
    dt: float = 5.0e-6,
    duration: float = 2.0e-3,
    contact: ContactLaw | None = None,
    bond_law: CohesiveBondLaw | None = None,
    rigid: bool = False,
) -> "IceStructureEnergyAudit":
    """Build the standard ice-floe-vs-structure impact audit case.

    ``rigid=True`` freezes all bonds (damage disabled) so the elastic
    reference case cannot fracture - used for the negative control and for the
    closure criterion without fracture.
    """
    lattice = build_bonded_lattice(*n_cells, radius, density)
    pos = lattice["positions"].copy()
    pos[:, 0] += gap + radius - pos[:, 0].min()
    vel = np.tile(np.array(impact_velocity, dtype=float), (pos.shape[0], 1))
    law = bond_law or CohesiveBondLaw()
    if rigid:
        # strength and toughness both -> infinity but with separation >> elastic
        # limit so the damage variable stays identically zero
        law = CohesiveBondLaw(stiffness=law.stiffness, area=law.area,
                              tensile_strength=1.0e30, fracture_energy=1.0e60)
    sim = IceStructureEnergyAudit(
        positions=pos,
        velocities=vel,
        radius=radius,
        density=density,
        contact=contact or ContactLaw(),
        bond_law=law,
        bond_i=lattice["bond_i"],
        bond_j=lattice["bond_j"],
        dt=dt,
        wall_plane_x=0.0,
    )
    sim.run_until(duration)
    return sim