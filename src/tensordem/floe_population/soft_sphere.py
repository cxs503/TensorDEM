"""Three-dimensional soft-sphere DEM used as the *measuring instrument* of the
floe-population model.

Why a second DEM engine
-----------------------
The repository already owns bonded-lattice engines (:mod:`tensordem.dem`,
:mod:`tensordem.dem3d`) that solve ice indentation and specimen mechanics.  A
population-balance closure needs something those engines do not provide: a
*dilute, statistically stationary, translationally invariant* ensemble in which
the collision rate, the contact value of the radial distribution function and
the aggregate breakage rate can be counted without walls, a tool or a strain
path.  This module is that instrument: spheres of arbitrary diameter, periodic
in three directions, thermostatted, optionally cemented into aggregates with
breakable bonds so that a "floe" is a set of spheres and its size is measured
from the surviving bond network.

Model
-----
Contacts are linear spring-dashpots,

.. math:: F_n = k_n \\delta - c_n \\dot\\delta, \\qquad
          c_n = 2\\zeta\\sqrt{k_n m_{ij}}, \\quad
          \\zeta = \\frac{-\\ln e}{\\sqrt{\\pi^2+\\ln^2 e}} ,

with :math:`\\delta = r_i+r_j-|x_j-x_i|` the overlap and :math:`e` the
coefficient of restitution.  A bond is the same element with the *rest length*
of the parent lattice, so it carries tension as well as compression and fails
when its axial force leaves ``[-bond_compression, +bond_tension]``.  A bonded
pair does not also carry a contact force (the bond is its load path), so the
stiffness is not double counted; a *failed* bond reverts to a plain contact.

Integration is velocity Verlet on cached accelerations (one force evaluation per
step).  An optional Berendsen thermostat holds a granular temperature
``T = sum_i m_i |v_i - u|^2 / (3 N)`` (``u`` the mass-averaged velocity), which
is what makes stationary collision statistics possible.

Units
-----
The module is unit agnostic and all campaign results are reported
non-dimensionalised: lengths in the smallest resolved floe diameter ``D_min``,
times in ``D_min/v_rms``, masses with unit density (``m_i = r_i**3``; the common
factor ``4*pi/3`` cancels everywhere).  Nothing in the closure depends on the
absolute scale, and claiming physical rates would need a stiffness ratio no
explicit DEM can afford (see the limitations section of
``docs/floe_population_P80W2N05.md``).

What is measured (and therefore what the closure may use)
--------------------------------------------------------
* :attr:`SphereAssembly.contact_events` -- every pair that becomes a contact
  (birth events) with the relative normal speed at birth.  Its count per unit
  time per unit volume *is* the measured collision kernel.
* :attr:`SphereAssembly.bond_break_events` -- time, axial separation and elastic
  energy of every failed bond.
* :meth:`SphereAssembly.cluster_census` -- connected components of the surviving
  bond network, i.e. the floe size distribution of the DEM state.
* :func:`pair_distribution` -- shell-resolved radial distribution function, from
  which the contact value ``g(sigma)`` is read.

Only these outputs feed the closure (:mod:`tensordem.floe_population.kernels`,
:mod:`tensordem.floe_population.population_balance`); no distribution is
prescribed anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = [
    "ContactModel",
    "PairHistogramAccumulator",
    "Wall",
    "SphereAssembly",
    "build_floe",
    "maxwellian_velocities",
    "place_floes",
    "rotate_about",
    "pair_distribution",
    "ClusterCensus",
]


# --------------------------------------------------------------------------- model
@dataclass(frozen=True)
class ContactModel:
    """Contact and bond law of :class:`SphereAssembly` (non-dimensional)."""

    k_n: float = 2000.0
    restitution: float = 0.60
    bond_stiffness: float = 4000.0
    bond_damping_ratio: float = 0.30
    bond_tension: float = 3.0
    bond_compression: float = 30.0
    skin: float = 0.20
    #: Gap (length) below which a pair counts as being "in collision contact" for
    #: the event ledger.  A pair has to leave this shell before a new event can be
    #: registered, which suppresses the spurious re-births of a soft contact that
    #: oscillates (they inflate the measured rate of the slow, heavy classes).
    event_gap: float = 0.05

    def __post_init__(self) -> None:
        if self.k_n <= 0.0 or self.bond_stiffness <= 0.0:
            raise ValueError("stiffnesses must be positive")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must lie in [0, 1]")
        if self.bond_tension <= 0.0 or self.bond_compression <= 0.0:
            raise ValueError("bond strength limits must be positive")
        if self.skin < 0.0:
            raise ValueError("skin must be non-negative")

    @property
    def damping_ratio(self) -> float:
        """Critical damping ratio that reproduces the restitution ``e``."""
        if self.restitution >= 1.0:
            return 0.0
        ln_e = math.log(self.restitution)
        return -ln_e / math.sqrt(math.pi**2 + ln_e**2)


@dataclass(frozen=True)
class Wall:
    """Rigid plane.  ``sign`` is the outward normal that points *into the fluid*:
    the solid occupies ``sign * (x[axis] - coordinate) < 0``, and the fluid gap is
    ``sign * (x[axis] - coordinate)``.  ``Wall(2, 0.0, +1.0)`` is therefore a
    floor at ``z = 0`` with the ice above it.

    Used by the breakage calibration, where one floe is driven into an anvil at a
    prescribed speed so that the impact kinetic energy is exactly known.
    """

    axis: int
    coordinate: float
    sign: float = 1.0

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if abs(self.sign) != 1.0:
            raise ValueError("sign must be +1 or -1")


def _wrap(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image displacement (result in ``[-L/2, L/2]``)."""
    return delta - box * np.round(delta / box)


def _fold(position: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Fold coordinates into the primary cell ``[0, L)`` (required by cKDTree)."""
    folded = np.mod(position, box)
    folded[folded >= box] = 0.0          # np.mod can return exactly L
    return folded


def maxwellian_velocities(mass, temperature: float, *, rng: np.random.Generator,
                          drift: np.ndarray | None = None) -> np.ndarray:
    """Gaussian velocities with zero net momentum at granular temperature ``T``."""
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    sigma = np.sqrt(np.maximum(temperature, 0.0) / mass)[:, None]
    velocity = rng.normal(size=(mass.size, 3)) * sigma
    velocity -= (mass[:, None] * velocity).sum(axis=0) / mass.sum()
    if drift is not None:
        velocity = velocity + np.asarray(drift, dtype=np.float64)
    return velocity


@dataclass
class ClusterCensus:
    """Floe census of one DEM state (one row per cluster)."""

    label: np.ndarray
    n_spheres: np.ndarray
    diameter: np.ndarray
    mass: np.ndarray
    diameter_per_sphere: float

    def __len__(self) -> int:
        return int(self.n_spheres.size)

    @property
    def sizes(self) -> np.ndarray:
        return self.diameter


class SphereAssembly:
    """Periodic soft-sphere assembly, optionally cemented into floes by bonds.

    Parameters
    ----------
    radius, position
        ``(N,)`` and ``(N, 3)``; unit density implies ``mass = radius**3``.
    box
        ``(3,)`` periodic cell edge lengths.
    bonds
        ``(M, 2)`` integer array of bonded sphere pairs stored with ``i < j``.
    tag
        ``(N,)`` integer label grouping the spheres of one parent body; the floe
        census starts from this labelling and ``break`` events are reported
        against it.
    """

    def __init__(
        self,
        radius,
        position,
        *,
        box,
        model: ContactModel | None = None,
        velocity=None,
        bonds=None,
        tag=None,
        walls=None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model or ContactModel()
        self.radius = np.array(radius, dtype=np.float64).reshape(-1)
        self.position = np.array(position, dtype=np.float64).reshape(-1, 3)
        self.box = np.array(box, dtype=np.float64).reshape(3)
        n = self.radius.size
        if self.position.shape != (n, 3):
            raise ValueError("position must have shape (N, 3)")
        if np.any(self.radius <= 0.0):
            raise ValueError("radii must be positive")
        if np.any(self.box <= 0.0):
            raise ValueError("box edges must be positive")
        self.position = _fold(self.position, self.box)
        self.mass = self.radius**3
        self.walls = tuple(walls) if walls else ()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.velocity = (
            np.zeros((n, 3)) if velocity is None
            else np.array(velocity, dtype=np.float64).reshape(n, 3)
        )
        self.tag = (
            np.arange(n) if tag is None else np.array(tag, dtype=np.int64).reshape(-1)
        )
        if self.tag.size != n:
            raise ValueError("tag must have length N")
        if bonds is None:
            bonds = np.zeros((0, 2), dtype=np.int64)
        bonds = np.array(bonds, dtype=np.int64).reshape(-1, 2)
        if bonds.size:
            if bonds.min() < 0 or bonds.max() >= n:
                raise ValueError("bond indices out of range")
            if np.any(bonds[:, 0] >= bonds[:, 1]):
                raise ValueError("bonds must be stored with i < j")
        self.bonds = bonds
        if bonds.size:
            delta = _wrap(self.position[bonds[:, 1]] - self.position[bonds[:, 0]], self.box)
            self.bond_rest = np.linalg.norm(delta, axis=1)
            self.bond_alive = np.ones(bonds.shape[0], dtype=bool)
        else:
            self.bond_rest = np.zeros(0)
            self.bond_alive = np.zeros(0, dtype=bool)
        self.time = 0.0
        self.steps = 0
        self.contact_events: list[tuple[float, int, int, float]] = []
        self.bond_break_events: list[tuple[float, int, int, float, float]] = []
        self._contact_keys = np.zeros(0, dtype=np.int64)
        self._bond_key_cache: np.ndarray | None = None
        self._acceleration = np.zeros((n, 3))
        self.force_evaluations = 0

    # -------------------------------------------------------------- geometry
    def max_coordination(self) -> int:
        """Largest number of force elements attached to one sphere."""
        if not self.bonds.size:
            return 0
        return int(np.bincount(self.bonds.reshape(-1), minlength=self.radius.size).max())

    def recommended_dt(self, safety: float = 0.20) -> float:
        """Elastic-limited step of the *assembly*, not of a single contact.

        A sphere inside a cemented aggregate is attached to up to 18 bonds, so the
        relevant frequency is ``sqrt((2 z_max + 1) k / m)`` with ``z_max`` the
        maximum coordination.  Ignoring the coordination overestimates the stable
        step by a factor ``sqrt(2 z_max + 1)`` (a factor of six for the floe
        lattice) and the aggregate then explodes within a hundred steps -- a
        failure mode that is easy to mistake for physical fragmentation, so the
        step is derived here and the stability is asserted in the tests.

        .. warning::
            This is a **stability** bound only.  It does not resolve the
            *response* of an unbonded contact: at this step a collision is
            advanced by roughly 2.5 steps and the produced restitution can be
            off by tens of per cent (and an elastic contact can gain energy).
            Use :meth:`restitution_dt` whenever the coefficient of restitution
            is a physical input, as it is for the collision statistics.
        """
        m_min = float(self.mass.min())
        k_max = max(self.model.k_n, self.model.bond_stiffness)
        elements = max(1, 2 * self.max_coordination() + 1)
        return safety * 2.0 * math.pi * math.sqrt(m_min / (elements * k_max))

    def restitution_dt(self, *, steps_per_contact: float = 80.0) -> float:
        r"""Step that *resolves the contact response*, not merely stability.

        :meth:`recommended_dt` is an elastic-stability bound: it keeps the
        stiffest bonded mode from blowing up, but it says nothing about whether
        a collision is integrated accurately.  For a bond-free pair it returns
        ``0.2 * 2 pi sqrt(m / k_max)`` with ``k_max = bond_stiffness``, i.e.
        roughly **2.5 steps per contact**, and at that step the *restitution
        the engine actually produces* is not the restitution that was asked
        for.  Measured on a head-on two-sphere collision (three sub-step
        phases each, ``benchmarks/floe_population_restitution/probe.py``)::

            e prescribed   dt = recommended_dt      dt = t_c/40    dt = t_c/80
            1.0             +11 ... +21 %            0.00 %         0.00 %
            0.6             -18 ... +34 %            2.09 %         1.03 %
            0.4             -48 ... +24 %            1.62 %         1.54 %
            0.2             +45 ... +278 %           4.55 %         2.22 %
            0.1             +524 ... +669 %          8.29 %         4.12 %

        So an "elastic" contact can *gain* a fifth of the kinetic energy and an
        strongly dissipative one (``e = 0.1``) bounces with ``e ~ 0.6`` when the
        recommended stability step is used.  The bound below is the exact
        contact half-period of the lightest pair,

        .. math:: t_c = \frac{\pi}{\sqrt{k_n/m_{eff}}\sqrt{1-\zeta^2}},
                  \qquad m_{eff} = m_{min}/2,

        divided by ``steps_per_contact``.  ``zeta = 0`` is used inside the
        square root on purpose: damping only *lengthens* the contact, so the
        undamped period is the conservative (smaller) bound.  ``80`` steps per
        contact holds every ``e >= 0.2`` inside 2.4 % (ice-like ``e = 0.3 ...
        0.6`` inside 2.1 %); ``e = 0.1`` needs ``steps_per_contact = 160``.
        """
        if steps_per_contact <= 0.0:
            raise ValueError("steps_per_contact must be positive")
        m_light = float(self.mass.min())
        t_c = math.pi * math.sqrt(m_light / (2.0 * self.model.k_n))
        return t_c / float(steps_per_contact)

    def volume(self) -> float:
        return float(np.prod(self.box))

    def _neighbour_pairs(self):
        rmax = 2.0 * float(self.radius.max()) * (1.0 + self.model.skin)
        tree = cKDTree(self.position, boxsize=self.box)
        pairs = tree.query_pairs(rmax, output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3)), np.zeros(0)
        pairs = np.clip(np.sort(np.asarray(pairs, dtype=np.int64), axis=1), 0, None)
        delta = _wrap(self.position[pairs[:, 1]] - self.position[pairs[:, 0]], self.box)
        return pairs, delta, np.linalg.norm(delta, axis=1)

    def _alive_bond_keys(self) -> np.ndarray:
        if self._bond_key_cache is not None:
            return self._bond_key_cache
        if not self.bonds.size:
            self._bond_key_cache = np.zeros(0, dtype=np.int64)
            return self._bond_key_cache
        alive = self.bonds[self.bond_alive]
        n = self.radius.size
        self._bond_key_cache = (
            np.sort(alive[:, 0] * n + alive[:, 1]) if alive.size else np.zeros(0, dtype=np.int64)
        )
        return self._bond_key_cache

    def _bonded_pair_mask(self, pairs: np.ndarray) -> np.ndarray:
        n = self.radius.size
        keys = self._alive_bond_keys()
        if keys.size == 0 or pairs.size == 0:
            return np.zeros(pairs.shape[0], dtype=bool)
        query = pairs[:, 0] * n + pairs[:, 1]
        slot = np.clip(np.searchsorted(keys, query), 0, keys.size - 1)
        return keys[slot] == query

    # ---------------------------------------------------------------- forces
    def forces(self, *, record_contacts: bool = True) -> np.ndarray:
        """Force on every sphere; updates the contact ledger when asked."""
        model = self.model
        n = self.radius.size
        force = np.zeros((n, 3))
        pairs, delta, dist = self._neighbour_pairs()
        self.force_evaluations += 1
        if pairs.size:
            i, j = pairs[:, 0], pairs[:, 1]
            nhat = delta / np.maximum(dist, 1e-12)[:, None]
            v_n = np.einsum("ij,ij->i", self.velocity[j] - self.velocity[i], nhat)
            overlap = self.radius[i] + self.radius[j] - dist
            bonded_pair = self._bonded_pair_mask(pairs)
            contact = (overlap > 0.0) & (~bonded_pair)
            if np.any(contact):
                m_eff = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
                c_n = 2.0 * model.damping_ratio * np.sqrt(model.k_n * m_eff)
                mag = model.k_n * overlap - c_n * v_n
                ci, cj = i[contact], j[contact]
                np.add.at(force, ci, -mag[contact, None] * nhat[contact])
                np.add.at(force, cj, mag[contact, None] * nhat[contact])
                del ci, cj, mag
            if record_contacts:
                near = (dist < self.radius[i] + self.radius[j] + model.event_gap) & (~bonded_pair)
                if np.any(near):
                    self._record_contacts(i[near], j[near], v_n[near])
        force = self._bond_forces(force)
        return self._wall_forces(force)

    def _record_contacts(self, i: np.ndarray, j: np.ndarray, v_n: np.ndarray) -> None:
        """Append the pairs that were not in contact at the previous evaluation.

        The ledger therefore counts *births* of contacts, which is the quantity a
        collision kernel is defined on; an enduring contact is counted once.
        """
        n = self.radius.size
        keys = i.astype(np.int64) * n + j.astype(np.int64)
        previous = self._contact_keys
        if previous.size:
            fresh = ~np.isin(keys, previous, assume_unique=True)
        else:
            fresh = np.ones(keys.size, dtype=bool)
        self._contact_keys = keys
        if np.any(fresh):
            stamp = self.time
            for a, b, v in zip(i[fresh], j[fresh], v_n[fresh]):
                self.contact_events.append((stamp, int(a), int(b), float(v)))

    def _bond_forces(self, force: np.ndarray) -> np.ndarray:
        if not self.bonds.size or not np.any(self.bond_alive):
            return force
        alive = self.bond_alive
        b = self.bonds[alive]
        i, j = b[:, 0], b[:, 1]
        delta = _wrap(self.position[j] - self.position[i], self.box)
        dist = np.linalg.norm(delta, axis=1)
        nhat = delta / np.maximum(dist, 1e-12)[:, None]
        v_n = np.einsum("ij,ij->i", self.velocity[j] - self.velocity[i], nhat)
        rest = self.bond_rest[alive]
        m_eff = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        c_b = 2.0 * self.model.bond_damping_ratio * np.sqrt(self.model.bond_stiffness * m_eff)
        elastic = self.model.bond_stiffness * (dist - rest)
        mag = -(elastic + c_b * v_n)
        np.add.at(force, i, -mag[:, None] * nhat)
        np.add.at(force, j, mag[:, None] * nhat)
        broken = (elastic > self.model.bond_tension) | (elastic < -self.model.bond_compression)
        if np.any(broken):
            local = np.flatnonzero(alive)[broken]
            self.bond_alive[local] = False
            self._bond_key_cache = None
            for slot, dist_value, energy in zip(
                local, dist[broken], 0.5 * elastic[broken] ** 2 / self.model.bond_stiffness
            ):
                a, c = self.bonds[slot]
                self.bond_break_events.append(
                    (self.time, int(a), int(c), float(dist_value), float(energy))
                )
        return force

    def _wall_forces(self, force: np.ndarray) -> np.ndarray:
        """Rigid-plane contacts (wall mass is infinite: no m_eff factor)."""
        model = self.model
        for wall in self.walls:
            axis = wall.axis
            gap = wall.sign * (self.position[:, axis] - wall.coordinate)
            penetration = self.radius - gap
            hit = penetration > 0.0
            if not np.any(hit):
                continue
            v_n = wall.sign * self.velocity[:, axis]
            c_n = 2.0 * model.damping_ratio * np.sqrt(model.k_n * self.mass)
            mag = model.k_n * penetration - c_n * v_n
            force[hit, axis] += wall.sign * mag[hit, None].reshape(-1)
        return force

    # ------------------------------------------------------------ integration
    def granular_temperature(self) -> float:
        velocity = self.velocity
        momentum = (self.mass[:, None] * velocity).sum(axis=0)
        drift = momentum / self.mass.sum()
        residual = velocity - drift
        kinetic = float((self.mass[:, None] * residual**2).sum())
        return kinetic / (3.0 * self.mass.size)

    def floe_temperature(self, *, min_cluster_spheres: int = 8) -> float:
        """Granular temperature of the floe ensemble (bodies above the debris floor).

        This is the temperature the field is actually driven at; the per-sphere
        granular temperature divides the floe centre-of-mass energy by the number
        of *spheres* and is therefore smaller by the mean number of spheres per
        floe.  Reporting the wrong one would mislabel the driving energy.
        """
        census = self.cluster_census()
        labels = census.label
        n_clusters = int(labels.max()) + 1
        eligible = census.n_spheres >= int(min_cluster_spheres)
        if int(eligible.sum()) < 2:
            return float("nan")
        mass_cluster = np.bincount(labels, weights=self.mass, minlength=n_clusters)
        momentum = np.stack(
            [np.bincount(labels, weights=self.mass * self.velocity[:, k], minlength=n_clusters)
             for k in range(3)],
            axis=1,
        )
        com_velocity = momentum / mass_cluster[:, None]
        active = mass_cluster[eligible]
        drift = (active[:, None] * com_velocity[eligible]).sum(axis=0) / active.sum()
        peculiar = com_velocity[eligible] - drift
        return float((active[:, None] * peculiar**2).sum()) / (3.0 * int(eligible.sum()))

    def thermostat(self, temperature: float, *, strength: float = 0.5) -> None:
        """Berendsen rescaling of the peculiar velocity field to ``T``."""
        momentum = (self.mass[:, None] * self.velocity).sum(axis=0)
        drift = momentum / self.mass.sum()
        self.velocity -= drift
        current = self.granular_temperature()
        if current <= 0.0:
            self.velocity = maxwellian_velocities(self.mass, temperature, rng=self.rng)
            return
        scale = math.sqrt(1.0 + strength * (temperature / current - 1.0))
        self.velocity *= scale
        self.velocity += drift

    def cluster_thermostat(
        self, temperature: float, *, strength: float = 0.5, min_cluster_spheres: int = 8
    ) -> None:
        """Berendsen rescaling of the *floe* (cluster) velocities only.

        The granular temperature of a broken-ice field is the agitation of its
        floes, not of the ice crystal.  Rescaling the centre-of-mass velocity of
        every cluster therefore leaves the internal (elastic) motion of each floe
        untouched and cold, which is both physical and necessary: an isokinetic
        per-sphere thermostat would shake the bond network apart on its own and
        the breakage statistics would measure the thermostat instead of the
        mechanics.
        """
        census = self.cluster_census()
        labels = census.label
        n_clusters = int(labels.max()) + 1
        # Only bodies at or above the floe scale are thermostatted.  Re-energising
        # the debris would set the granular temperature of the smallest possible
        # cluster, i.e. it would launch every freed sphere at ``sqrt(3T/m)`` and
        # the cascade would then be driven by the thermostat instead of by the
        # mechanics (a runaway that reaches single spheres from any initial state).
        eligible = census.n_spheres >= int(min_cluster_spheres)
        if int(eligible.sum()) < 2:
            return  # the floe ensemble has no granular temperature
        mass_cluster = np.bincount(labels, weights=self.mass, minlength=n_clusters)
        momentum = np.stack(
            [np.bincount(labels, weights=self.mass * self.velocity[:, k], minlength=n_clusters)
             for k in range(3)],
            axis=1,
        )
        com_velocity = momentum / mass_cluster[:, None]
        active_mass = mass_cluster[eligible]
        total_mass = float(active_mass.sum())
        drift = (active_mass[:, None] * com_velocity[eligible]).sum(axis=0) / total_mass
        peculiar = com_velocity - drift
        current = float((active_mass[:, None] * peculiar[eligible] ** 2).sum()) / (
            3.0 * int(eligible.sum())
        )
        if current <= 0.0:
            fresh = maxwellian_velocities(active_mass, temperature, rng=self.rng) + drift
            delta = np.zeros_like(com_velocity)
            delta[eligible] = fresh - com_velocity[eligible]
        else:
            factor = math.sqrt(1.0 + strength * (temperature / current - 1.0))
            delta = np.zeros_like(com_velocity)
            delta[eligible] = (factor - 1.0) * peculiar[eligible]
        self.velocity = self.velocity + delta[labels]

    def resolve_contacts(self, *, passes: int = 1) -> int:
        """Remove residual overlap of un-bonded pairs before a run starts.

        The placed configuration is only approximately non-overlapping; a few
        relaxation passes keep the first force evaluation from launching a
        spurious shock.  Returns the number of overlapping pairs still present.
        """
        count = 0
        for _ in range(passes):
            pairs, delta, dist = self._neighbour_pairs()
            if not pairs.size:
                return 0
            overlap = self.radius[pairs[:, 0]] + self.radius[pairs[:, 1]] - dist
            mask = (overlap > 0.0) & (~self._bonded_pair_mask(pairs))
            count = int(mask.sum())
            if count == 0:
                return 0
            nhat = delta[mask] / np.maximum(dist[mask], 1e-12)[:, None]
            push = 0.5 * overlap[mask]
            np.add.at(self.position, pairs[mask, 0], -push[:, None] * nhat)
            np.add.at(self.position, pairs[mask, 1], push[:, None] * nhat)
            self.position = _fold(self.position, self.box)
        return count

    def advance(
        self,
        dt: float,
        steps: int,
        *,
        temperature: float | None = None,
        cluster_temperature: float | None = None,
        thermostat_interval: int = 10,
        record_contacts: bool = True,
        callback=None,
        callback_interval: int = 1,
    ) -> None:
        """Velocity Verlet marching with an optional thermostatted drive.

        ``temperature`` rescales the per-sphere velocities (free-sphere gas),
        ``cluster_temperature`` rescales the floe centre-of-mass velocities only
        (broken-ice field).  Both are Berendsen-weak and applied every
        ``thermostat_interval`` steps.
        """
        if dt <= 0.0 or steps < 0:
            raise ValueError("dt must be positive and steps non-negative")
        self._acceleration = self.forces(record_contacts=record_contacts) / self.mass[:, None]
        for step in range(steps):
            half = 0.5 * dt
            self.velocity += half * self._acceleration
            self.position += dt * self.velocity
            self.position = _fold(self.position, self.box)
            self._acceleration = self.forces(record_contacts=record_contacts) / self.mass[:, None]
            self.velocity += half * self._acceleration
            self.time += dt
            self.steps += 1
            if thermostat_interval > 0 and (step + 1) % thermostat_interval == 0:
                if temperature is not None:
                    self.thermostat(temperature)
                if cluster_temperature is not None:
                    self.cluster_thermostat(cluster_temperature)
            if callback is not None and callback_interval > 0:
                if (step + 1) % callback_interval == 0:
                    callback(self)

    # ---------------------------------------------------------------- census
    def cluster_census(self) -> ClusterCensus:
        """Connected components of the surviving bond network."""
        n = self.radius.size
        alive = self.bonds[self.bond_alive] if self.bonds.size else np.zeros((0, 2), dtype=np.int64)
        if alive.size:
            graph = coo_matrix(
                (np.ones(alive.shape[0]), (alive[:, 0], alive[:, 1])), shape=(n, n)
            )
            _, labels = connected_components(graph, directed=False)
        else:
            labels = np.arange(n)
        unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        diameter_per_sphere = float(2.0 * np.median(self.radius))
        diameter = diameter_per_sphere * counts ** (1.0 / 3.0)
        mass = np.bincount(inverse, weights=self.mass)
        return ClusterCensus(
            label=inverse,
            n_spheres=counts,
            diameter=diameter,
            mass=mass,
            diameter_per_sphere=diameter_per_sphere,
        )

    def floe_sizes(self) -> np.ndarray:
        return self.cluster_census().diameter


# ------------------------------------------------------------------- generation
def build_floe(
    n_side: int,
    *,
    radius: float = 0.5,
    spacing: float | None = None,
    diagonal_bonds: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple-cubic aggregate of ``n_side**3`` touching spheres.

    Face diagonals are bonded as well: axis-aligned central-force bonds alone
    leave the shear modes floppy, and a floe has to resist shear to be a body.

    Returns ``(position, bonds, radius_array)`` centred on the origin.
    """
    if n_side < 2:
        raise ValueError("n_side must be at least 2")
    step = 2.0 * radius if spacing is None else float(spacing)
    axis = (np.arange(n_side) - (n_side - 1) / 2.0) * step
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    if diagonal_bonds:
        tree = cKDTree(grid)
        pairs = tree.query_pairs(step * math.sqrt(2.0) * 1.02, output_type="ndarray")
    else:
        tree = cKDTree(grid)
        pairs = tree.query_pairs(step * 1.02, output_type="ndarray")
    bonds = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
    return grid, bonds, np.full(grid.shape[0], float(radius))


def rotate_about(position: np.ndarray, centre: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random rotation (uniform on SO(3)) of point cloud ``position``."""
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return (position - centre) @ rot.T + centre


def place_floes(
    specs,
    *,
    box,
    rng: np.random.Generator,
    radius: float = 0.5,
    clearance: float = 0.02,
    max_attempts: int = 4000,
):
    """Random sequential placement of ``specs`` aggregates in a periodic box.

    ``specs`` is a sequence of ``n_side`` values.  Returns
    ``(radius, position, bonds, tag)`` ready for :class:`SphereAssembly`.
    """
    box = np.asarray(box, dtype=np.float64).reshape(3)
    radii: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    bonds: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    placed: list[np.ndarray] = []
    offset = 0
    for floe_id, n_side in enumerate(specs):
        base, bond, rad = build_floe(n_side, radius=radius)
        radius_min = radius * (1.0 + clearance)
        for _ in range(max_attempts):
            centre = rng.uniform(0.0, 1.0, size=3) * box
            cloud = rotate_about(base, np.zeros(3), rng) + centre
            wrapped = _fold(cloud, box)
            ok = True
            for previous in placed:
                if previous.size == 0:
                    continue
                delta = _wrap(wrapped[:, None, :] - previous[None, :, :], box)
                dist = np.linalg.norm(delta, axis=2)
                if np.any(dist < 2.0 * radius_min):
                    ok = False
                    break
            if ok:
                break
        else:
            raise RuntimeError(f"could not place floe {floe_id} ({n_side}^3 spheres)")
        placed.append(wrapped)
        positions.append(wrapped)
        radii.append(rad)
        bonds.append(bond + offset)
        tags.append(np.full(rad.size, floe_id, dtype=np.int64))
        offset += rad.size
    return (
        np.concatenate(radii),
        np.concatenate(positions),
        np.concatenate(bonds) if bonds else np.zeros((0, 2), dtype=np.int64),
        np.concatenate(tags),
    )


# ------------------------------------------------------------------ structure
def pair_distribution(
    position: np.ndarray,
    box: np.ndarray,
    *,
    class_id: np.ndarray | None = None,
    r_max: float | None = None,
    n_bins: int = 60,
):
    """Shell-resolved pair distribution ``g_ab(r)`` for diameter classes.

    ``g_ab(r) = V/(N_a N_b) * dn_ab/(4 pi r^2 dr)`` (``a == b`` counted once, so
    the same factor applies).  Returns ``(centres, g, counts)`` with ``g`` an
    ``(n_classes, n_classes, n_bins)`` array.
    """
    position = np.asarray(position, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64).reshape(3)
    volume = float(np.prod(box))
    labels = (
        np.zeros(position.shape[0], dtype=np.int64)
        if class_id is None else np.asarray(class_id, dtype=np.int64)
    )
    n_classes = int(labels.max()) + 1 if labels.size else 0
    if r_max is None:
        r_max = 0.5 * float(box.min())
    edges = np.linspace(0.0, r_max, n_bins + 1)
    tree = cKDTree(position, boxsize=box)
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    counts = np.zeros((n_classes, n_classes, n_bins))
    if len(pairs):
        pairs = np.asarray(pairs, dtype=np.int64)
        delta = _wrap(position[pairs[:, 1]] - position[pairs[:, 0]], box)
        dist = np.linalg.norm(delta, axis=1)
        bin_index = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
        a = labels[pairs[:, 0]]
        b = labels[pairs[:, 1]]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        np.add.at(counts, (lo, hi, bin_index), 1.0)
        # symmetric partner so that g[a][b] is readable for both orders
        off = lo != hi
        np.add.at(counts, (hi[off], lo[off], bin_index[off]), 1.0)
    shell = 4.0 / 3.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    centres = 0.5 * (edges[1:] + edges[:-1])
    g = np.zeros_like(counts)
    for a in range(n_classes):
        n_a = float((labels == a).sum())
        for b in range(n_classes):
            n_b = float((labels == b).sum())
            if n_a == 0.0 or n_b == 0.0:
                continue
            prefactor = volume / (n_a * n_b)
            if a == b:
                # each unordered pair was counted once; use N(N-1)/2 normalisation
                prefactor = volume / (0.5 * n_a * (n_a - 1.0)) if n_a > 1.0 else 0.0
            g[a, b] = prefactor * counts[a, b] / shell
    return centres, g, counts


class PairHistogramAccumulator:
    r"""Time-averaged shell histogram of pair separations, per diameter class.

    The contact value of the radial distribution function is a *fluctuating*
    quantity: in a dilute gas only a fraction of a pair sits inside the contact
    shell at any instant, so a single snapshot cannot measure ``g(sigma)``.  This
    accumulator averages the shell counts over many snapshots and evaluates

    .. math:: g_{ab}(\sigma) = \frac{V}{N_a N_b}\,
              \frac{\overline{\Delta n_{ab}}}{4\pi\sigma^2\Delta r},

    with the shell straddling ``sigma_{ab}`` (``[0.9 sigma, 1.1 sigma]`` by
    default), which is the standard soft-sphere estimate of the contact value.
    """

    def __init__(
        self,
        *,
        box,
        diameters,
        n_classes: int,
        r_max: float | None = None,
        n_bins: int = 120,
    ) -> None:
        self.box = np.asarray(box, dtype=np.float64).reshape(3)
        self.volume = float(np.prod(self.box))
        self.diameters = np.asarray(diameters, dtype=np.float64).reshape(-1)
        self.n_classes = int(n_classes)
        if r_max is None:
            r_max = 1.15 * float(self.diameters.max())
        self.r_max = float(r_max)
        self.edges = np.linspace(0.0, self.r_max, n_bins + 1)
        self.counts = np.zeros((self.n_classes, self.n_classes, n_bins))
        self.class_counts = np.zeros(self.n_classes)
        self.samples = 0

    def observe(self, position, class_id) -> None:
        labels = np.asarray(class_id, dtype=np.int64).reshape(-1)
        pairs = cKDTree(position, boxsize=self.box).query_pairs(
            self.r_max, output_type="ndarray"
        )
        if len(pairs):
            pairs = np.asarray(pairs, dtype=np.int64)
            delta = _wrap(position[pairs[:, 1]] - position[pairs[:, 0]], self.box)
            dist = np.linalg.norm(delta, axis=1)
            index = np.clip(np.digitize(dist, self.edges) - 1, 0, self.edges.size - 2)
            a = labels[pairs[:, 0]]
            b = labels[pairs[:, 1]]
            low = np.minimum(a, b)
            high = np.maximum(a, b)
            np.add.at(self.counts, (low, high, index), 1.0)
            # mirror so that g_ab is readable for both orders (counts are then
            # symmetric, like the pair histogram they come from)
            cross = low != high
            if np.any(cross):
                np.add.at(self.counts, (high[cross], low[cross], index[cross]), 1.0)
        self.class_counts = np.bincount(labels, minlength=self.n_classes).astype(np.float64)
        self.samples += 1

    def contact_value(self, *, shell_width: float | None = None) -> np.ndarray:
        """Shell-averaged contact value ``g_ab(sigma_ab)``.

        The shell has an **absolute** half-width (``0.1 D_min`` by default), not
        a width proportional to ``sigma``: the structure of the pair distribution
        is set by the smallest diameter, so a shell proportional to ``sigma``
        smears the contact peak of the large classes and biases their ``g``
        downwards (a bias that showed up directly in the C1 residual before the
        width was fixed).
        """
        if self.samples == 0:
            raise ValueError("no snapshot has been accumulated")
        mean_counts = self.counts / float(self.samples)
        g = np.zeros((self.n_classes, self.n_classes))
        if shell_width is None:
            shell_width = 0.1 * float(self.diameters.min())
        sigma = 0.5 * (self.diameters[:, None] + self.diameters[None, :])
        for a in range(self.n_classes):
            n_a = self.class_counts[a]
            for b in range(self.n_classes):
                n_b = self.class_counts[b]
                normaliser = n_a * n_b if a != b else 0.5 * n_a * (n_a - 1.0)
                if normaliser <= 0.0:
                    continue
                lo = sigma[a, b] - shell_width
                hi = sigma[a, b] + shell_width
                mask = (self.edges[:-1] >= lo) & (self.edges[1:] <= hi)
                if not np.any(mask):
                    index = int(
                        np.argmin(np.abs(0.5 * (self.edges[1:] + self.edges[:-1]) - sigma[a, b]))
                    )
                    mask = np.zeros(self.edges.size - 1, dtype=bool)
                    mask[index] = True
                shell_volume = (
                    4.0 / 3.0 * math.pi * (self.edges[1:][mask] ** 3 - self.edges[:-1][mask] ** 3)
                ).sum()
                g[a, b] = self.volume / normaliser * mean_counts[a, b, mask].sum() / shell_volume
        return g
