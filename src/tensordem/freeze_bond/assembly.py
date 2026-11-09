"""Bonded disk assembly: geometry, internal forces and tangent stiffness.

The specimen is a two-dimensional central-force bonded packing of equal disks on
a triangular lattice (spacing ``a = 2 * radius``).  Bonds are *isotropic*
cohesive springs in the small-rotation linearisation::

    delta_vec = (x_j - x_i) - r0      separation vector of the bond
    delta_n   = delta_vec . n0        opening (positive = separated)
    delta_t   = delta_vec . t0        slip
    t_e       = c * A * |delta_vec|   traction magnitude [Pa]

with ``c = (1 - d) * k0`` the secant stiffness per unit area [Pa/m] from
:class:`~tensordem.freeze_bond.cohesive.CohesiveBondSet`, ``A`` the nominal bond
cross-section (``2 * radius * thickness``) and ``(n0, t0)`` the bond frame frozen
at the reference configuration.  The lattice is central-force, so it carries
bending only through the network (no rotational springs); its Poisson ratio is
that of a central-force triangular lattice (1/3), not that of ice.  That is a
declared idealisation, not a calibration.

Contacts are unilateral linear springs with Cundall-Strack tangential history and
a Coulomb cap ``|F_t| <= mu * F_n``; walls are the same model against a rigid
plane.  Everything is in SI with an explicit ``thickness_m`` so the energy ledger
can be closed in joules.

Nothing here is presented as a validated ice-rubble material model: the declared
computational parameters are collected in :class:`SpecimenConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .cohesive import BondEnvironment, CohesiveBondSet
from .constitutive import FreezeBondLaw

__all__ = [
    "Wall",
    "BondTopology",
    "SpecimenConfig",
    "BondedSpecimen",
    "FrictionState",
    "build_channel_specimen",
    "build_box_specimen",
    "internal_forces",
]


# --------------------------------------------------------------------- walls
@dataclass
class Wall:
    """Rigid plane; material lies on the ``+normal`` side of the plane.

    A particle of radius ``r`` at ``x`` penetrates by ``r - (x - point).normal``;
    the wall then pushes it along ``+normal`` with ``stiffness_n * penetration``.
    Moving ``point`` along ``+normal`` drives the wall into the material, which is
    the loading direction used by the campaign.
    """

    point: np.ndarray
    normal: np.ndarray
    stiffness_n: float
    friction: float = 0.0
    stiffness_t: float | None = None
    name: str = "wall"
    movable: bool = False

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=np.float64).reshape(2)
        n = np.asarray(self.normal, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(n))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("wall normal must be a finite non-zero vector")
        self.normal = n / norm
        self.tangent = np.array([-self.normal[1], self.normal[0]])
        if self.stiffness_n <= 0.0:
            raise ValueError("wall stiffness_n must be positive")
        if self.stiffness_t is None:
            self.stiffness_t = self.stiffness_n
        if self.friction < 0.0:
            raise ValueError("wall friction must be >= 0")
        self.base_point = self.point.copy()

    def offset(self) -> float:
        """Signed plane offset ``point . normal`` (the loading degree of freedom)."""
        return float(self.point @ self.normal)


# -------------------------------------------------------------------- bonds
@dataclass
class BondTopology:
    i: np.ndarray
    j: np.ndarray
    rest_vector: np.ndarray
    rest_length: np.ndarray
    normal: np.ndarray      # (m, 2) reference unit normal n0
    tangent: np.ndarray     # (m, 2) reference unit tangent t0
    area: np.ndarray        # (m,) nominal cross-section [m^2]

    @property
    def n_bonds(self) -> int:
        return int(self.i.size)


def _triangular_positions(n_x: int, n_y: int, spacing: float, *, jitter: float,
                          seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dy = spacing * np.sqrt(3.0) / 2.0
    rows = []
    for row in range(n_y):
        xs = np.arange(n_x, dtype=np.float64) * spacing + (row % 2) * spacing / 2.0
        rows.append(np.stack([xs, np.full(n_x, row * dy)], axis=1))
    pos = np.concatenate(rows, axis=0)
    if jitter > 0.0:
        pos = pos + rng.normal(scale=jitter * spacing, size=pos.shape)
    return pos


def _lattice(cfg: SpecimenConfig) -> tuple[np.ndarray, np.ndarray]:
    """Jittered triangular lattice with radii fitted to the local spacing.

    Equal radii on a jittered lattice would overlap by up to ``4 * jitter`` of the
    spacing, which would pre-load the contacts far beyond the freeze-bond
    strength.  The radii are therefore fitted per particle,
    ``r_i = 0.5 * min_j d(i, j)`` (``j`` the six lattice neighbours), so that
    every bonded pair is *exactly* touching: the as-built assembly has zero bond
    force and zero contact force, and contacts only become active once the bonds
    break or the material compacts.  That is the bonded-particle idealisation
    ("bridges across a touching packing"), declared here and not calibrated.
    """
    pos = _triangular_positions(cfg.n_x, cfg.n_y, cfg.spacing,
                                jitter=cfg.jitter, seed=cfg.seed)
    # Uniform radii = half the lattice spacing, inflated so that neighbouring
    # disks overlap by ``-contact_gap_fraction`` of the spacing.  Uniform radii
    # keep *every* boundary particle in contact with the walls (a radius fitted to
    # a jittered local spacing leaves gaps of the order of the jitter, i.e. 1e-3 m,
    # which is 100x the elastic overlap and destroys the lateral load path).
    radii = np.full(pos.shape[0], 0.5 * cfg.spacing * (1.0 - cfg.contact_gap_fraction))
    return pos, radii


def _build_topology(positions: np.ndarray, radii: np.ndarray,
                    cfg: SpecimenConfig) -> BondTopology:
    spacing = cfg.spacing
    tree = cKDTree(positions)
    pairs = sorted(tree.query_pairs(r=spacing * cfg.bond_cutoff_factor))
    if not pairs:
        raise ValueError("no bond pairs found; check the lattice spacing")
    idx = np.array(pairs, dtype=np.int64)
    if 0.0 < cfg.bond_fraction < 1.0:
        # Refrozen *rubble*: freeze bonds occupy only a fraction of the contacts,
        # the rest stay frictional ice-ice contacts.  This is the physically
        # relevant state for an ice arch (broken, refrozen rubble), declared here
        # and seeded so the topology is reproducible.
        rng = np.random.default_rng(cfg.seed + 977)
        keep = rng.random(idx.shape[0]) < cfg.bond_fraction
        idx = idx[keep]
        if idx.shape[0] < 3:
            raise ValueError("bond_fraction left fewer than 3 bonds")
    i, j = idx[:, 0], idx[:, 1]
    rest = positions[j] - positions[i]
    length = np.linalg.norm(rest, axis=1)
    normal = rest / length[:, None]
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    width = cfg.bond_width_factor * (radii[i] + radii[j])
    area = width * cfg.thickness_m
    return BondTopology(i=i, j=j, rest_vector=rest, rest_length=length,
                        normal=normal, tangent=tangent, area=area)


@dataclass(frozen=True)
class SpecimenConfig:
    """Declared geometry and computational material parameters.

    ``bond_modulus_pa`` is a **computational** bond modulus: it is chosen so that
    the bond reaches its strength at ``bond_peak_strain = sigma / (k0 * L0)``,
    which sets the load-step size.  The physical inputs of the campaign are the
    freeze-bond strength law, the fracture energy, the contact friction and the
    bulk compressive strength; the bond modulus only controls elastic stiffness
    and the arch threshold is reported as a strength ratio ``kappa`` so that it
    is independent of the absolute modulus.
    """

    n_x: int = 30
    n_y: int = 10
    radius_m: float = 0.01
    thickness_m: float = 1.0
    jitter: float = 0.0
    seed: int = 0
    bond_modulus_pa: float = 3.0e7
    bond_width_factor: float = 1.0
    contact_stiffness_N_m: float = 3.0e7
    contact_stiffness_t_N_m: float = 1.0e7
    contact_friction: float = 0.3
    wall_stiffness_N_m: float = 3.0e7
    wall_stiffness_t_N_m: float = 1.0e7
    wall_friction: float = 0.3
    bond_cutoff_factor: float = 1.15
    contact_cutoff_factor: float = 1.02
    wall_preload_m: float = 4.0e-7
    contact_gap_fraction: float = -2.0e-5
    bond_fraction: float = 1.0
    bond_strength_cv: float = 0.0
    bond_strength_seed: int = 0

    @property
    def spacing(self) -> float:
        return 2.0 * self.radius_m

    def __post_init__(self) -> None:
        for name in ("n_x", "n_y"):
            if int(getattr(self, name)) < 2:
                raise ValueError(f"{name} must be >= 2")
        for name in ("radius_m", "thickness_m", "bond_modulus_pa",
                     "contact_stiffness_N_m", "contact_stiffness_t_N_m",
                     "wall_stiffness_N_m", "wall_stiffness_t_N_m"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("contact_friction", "wall_friction", "jitter",
                     "bond_strength_cv"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 < float(self.bond_fraction) <= 1.0:
            raise ValueError("bond_fraction must be in (0, 1]")


@dataclass
class BondedSpecimen:
    """Geometry, bond topology, cohesive state and boundary walls."""

    positions: np.ndarray                 # committed positions (n, 2) [m]
    radii: np.ndarray
    config: SpecimenConfig
    topology: BondTopology
    cohesive: CohesiveBondSet
    walls: list[Wall] = field(default_factory=list)
    bond_stiffness_per_area: float = 1.0   # k0 [Pa/m]
    width_m: float = 0.0
    height_m: float = 0.0

    # -------------------------------------------------------------- derived
    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    def movable_walls(self) -> list[Wall]:
        return [w for w in self.walls if w.movable]

    def wall_by_name(self, name: str) -> Wall:
        for wall in self.walls:
            if wall.name == name:
                return wall
        raise KeyError(f"no wall named {name!r}")

    # --------------------------------------------------------------- energy
    def bond_stored_energy(self, state: dict[str, np.ndarray]) -> float:
        """Recoverable energy in the bonds for a trial state [J]."""
        return float(0.5 * np.sum(state["traction"] * state["delta_e"]
                                  * self.topology.area))


# ---------------------------------------------------------------- builders
def _make_wall(point, normal, cfg: SpecimenConfig, *, name: str,
               friction: float | None = None, movable: bool = False) -> Wall:
    """Wall with a declared preload: shifted into the material by ``wall_preload_m``.

    The preload only removes the rigid-body null space of the as-built specimen
    (an unconfined bonded network has three rigid-body modes); the reference
    state is subsequently equilibrated and the ledger is re-zeroed there, so the
    preload does no work in the reported ledger.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return Wall(point=np.asarray(point, dtype=np.float64) + cfg.wall_preload_m * n,
                normal=n,
                stiffness_n=cfg.wall_stiffness_N_m,
                friction=cfg.wall_friction if friction is None else friction,
                stiffness_t=cfg.wall_stiffness_t_N_m,
                name=name, movable=movable)


def build_channel_specimen(
    law: FreezeBondLaw,
    *,
    config: SpecimenConfig | None = None,
    environment: BondEnvironment | None = None,
) -> BondedSpecimen:
    """Channel of bonded ice rubble between two frictional walls.

    Geometry: ``x`` along the channel, ``y`` across it.  Two rigid frictional
    walls confine the field to the channel width ``W``, a movable platen at the
    upstream end pushes along ``+x``, and a fixed frictional *load cell* wall
    closes the downstream end so the transmitted force can be measured.  This is
    the classical ice-arch (channel-jam) set-up: an arch is a load path from the
    platen to the side walls, so a well-formed arch shows up as a drop of the
    load-cell force together with spanning strong-force loops.
    """
    cfg = config or SpecimenConfig()
    pos, radii = _lattice(cfg)
    topo = _build_topology(pos, radii, cfg)
    env = environment or BondEnvironment.uniform(topo.n_bonds)
    k0 = cfg.bond_modulus_pa / cfg.spacing
    cohesive = CohesiveBondSet(law, env, stiffness_pa_per_m=k0,
                               areas_m2=topo.area)
    xmin, xmax = float(np.min(pos[:, 0] - radii)), float(np.max(pos[:, 0] + radii))
    ymin, ymax = float(np.min(pos[:, 1] - radii)), float(np.max(pos[:, 1] + radii))
    walls = [
        _make_wall([0.0, ymin], [0.0, 1.0], cfg, name="wall_bottom"),
        _make_wall([0.0, ymax], [0.0, -1.0], cfg, name="wall_top"),
        _make_wall([xmin, 0.0], [1.0, 0.0], cfg, name="platen", movable=True),
        _make_wall([xmax, 0.0], [-1.0, 0.0], cfg, name="load_cell"),
    ]
    return BondedSpecimen(positions=pos, radii=radii, config=cfg,
                          topology=topo, cohesive=cohesive, walls=walls,
                          bond_stiffness_per_area=k0,
                          width_m=float(xmax - xmin), height_m=float(ymax - ymin))


def build_box_specimen(
    law: FreezeBondLaw,
    *,
    config: SpecimenConfig | None = None,
    environment: BondEnvironment | None = None,
) -> BondedSpecimen:
    """Confined biaxial compression cell: four frictional walls, top platen.

    The top wall is the displacement-controlled platen and the lateral walls are
    fixed, so this is a plane-strain *confined* biaxial test: the lateral wall
    reaction gives the confining stress and ``sigma_lateral / sigma_axial`` the
    apparent stress ratio.  The platen is frictionless so that the external work
    has a single well-defined source.
    """
    cfg = config or SpecimenConfig()
    pos, radii = _lattice(cfg)
    topo = _build_topology(pos, radii, cfg)
    env = environment or BondEnvironment.uniform(topo.n_bonds)
    k0 = cfg.bond_modulus_pa / cfg.spacing
    cohesive = CohesiveBondSet(law, env, stiffness_pa_per_m=k0,
                               areas_m2=topo.area)
    xmin, xmax = float(np.min(pos[:, 0] - radii)), float(np.max(pos[:, 0] + radii))
    ymin, ymax = float(np.min(pos[:, 1] - radii)), float(np.max(pos[:, 1] + radii))
    walls = [
        _make_wall([xmin, 0.0], [1.0, 0.0], cfg, name="wall_left"),
        _make_wall([xmax, 0.0], [-1.0, 0.0], cfg, name="wall_right"),
        _make_wall([0.0, ymin], [0.0, 1.0], cfg, name="wall_bottom"),
        _make_wall([0.0, ymax], [0.0, -1.0], cfg, name="platen", friction=0.0,
                   movable=True),
    ]
    return BondedSpecimen(positions=pos, radii=radii, config=cfg,
                          topology=topo, cohesive=cohesive, walls=walls,
                          bond_stiffness_per_area=k0,
                          width_m=float(xmax - xmin), height_m=float(ymax - ymin))


# ------------------------------------------------------- force evaluation
def _gather(keys: np.ndarray, values: np.ndarray,
            query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``dict.get`` on a sorted key array; also returns the hit mask.

    The hit mask matters: a contact that has just formed has *no* tangential
    anchor, so its slip must be measured from its birth configuration (``xi = 0``)
    rather than from a zero anchor, which would apply a spurious tangential force
    of ``k_t * |relative displacement|`` and manufacture friction work.
    """
    if keys.size == 0 or query.size == 0:
        return np.zeros((query.size,) + values.shape[1:]), np.zeros(query.size, dtype=bool)
    pos = np.clip(np.searchsorted(keys, query), 0, keys.size - 1)
    hit = keys[pos] == query
    out = np.zeros((query.size,) + values.shape[1:])
    out[hit] = values[pos[hit]]
    return out, hit


def _sorted_unique(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    return keys[order], values[order]


@dataclass
class FrictionState:
    """Cundall-Strack tangential memory, stored as sorted sparse key arrays.

    The anchor is a **vector** (the relative displacement, or for a wall the
    particle position, at the last re-anchoring), not a scalar in a rotating
    frame: the tangential spring stretch is ``xi = (I - n n) . (x_rel - anchor)``
    with ``n`` the *current* contact normal.  A scalar-in-a-rotating-frame anchor
    would manufacture slip (and therefore friction work) whenever the contact
    normal rotates, which is a well-known bookkeeping trap; the vector form is
    the standard Cundall-Strack update and is what makes the energy ledger of
    this package close (``tests/test_freeze_bond_energy.py``).

    Keys are ``i * n_nodes + j`` for a disk contact and ``i * n_walls + w`` for a
    wall contact.
    """

    n_nodes: int = 1
    n_walls: int = 1
    contact_keys: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    contact_anchor: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    wall_keys: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    wall_anchor: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def copy(self) -> "FrictionState":
        return FrictionState(
            n_nodes=self.n_nodes, n_walls=self.n_walls,
            contact_keys=self.contact_keys.copy(),
            contact_anchor=self.contact_anchor.copy(),
            wall_keys=self.wall_keys.copy(),
            wall_anchor=self.wall_anchor.copy(),
        )

    def contact_anchor_for(self, i: np.ndarray, j: np.ndarray):
        return _gather(self.contact_keys, self.contact_anchor,
                       i.astype(np.int64) * self.n_nodes + j)

    def wall_anchor_for(self, i: np.ndarray, w: int):
        return _gather(self.wall_keys, self.wall_anchor,
                       i.astype(np.int64) * self.n_walls + w)


def contact_pairs(positions: np.ndarray, radii: np.ndarray,
                  cfg: SpecimenConfig) -> np.ndarray:
    """Candidate disk-disk pairs within the contact cutoff (n, 2) index array."""
    tree = cKDTree(positions)
    r_max = 2.0 * float(np.max(radii)) * cfg.contact_cutoff_factor
    pairs = tree.query_pairs(r=r_max)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64)


def bond_separations(spec: BondedSpecimen, positions: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """``(delta_n, delta_t)`` for every bond, in the frozen bond frame."""
    topo = spec.topology
    delta = positions[topo.j] - positions[topo.i] - topo.rest_vector
    dn = np.einsum("ij,ij->i", delta, topo.normal)
    dt_ = np.einsum("ij,ij->i", delta, topo.tangent)
    return dn, dt_


def internal_forces(
    spec: BondedSpecimen,
    positions: np.ndarray,
    friction: FrictionState,
    *,
    bond_state: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Internal forces on every particle [N] plus diagnostics.

    Force sign: the returned array is the force *on* the particle from bonds,
    disk contacts and walls.  Equilibrium of a free particle means ``forces == 0``.
    """
    n = spec.n_particles
    forces = np.zeros((n, 2))
    topo = spec.topology
    cfg = spec.config

    # ---------------------------------------------------------------- bonds
    state = bond_state if bond_state is not None else None
    dn, dt_ = bond_separations(spec, positions)
    if state is None:
        trial = spec.cohesive.trial(dn, dt_)
    else:  # caller supplies a pre-evaluated trial (same separations)
        trial = state
    delta_vec = positions[topo.j] - positions[topo.i]
    delta_vec = delta_vec - topo.rest_vector
    # One-sided cohesive force: normal traction only in tension, plus shear.
    dn_now = np.einsum("ij,ij->i", delta_vec, topo.normal)
    dt_now = np.einsum("ij,ij->i", delta_vec, topo.tangent)
    bond_force = (trial["t_n"][:, None] * topo.normal
                  + trial["t_t"][:, None] * topo.tangent) * topo.area[:, None]
    # A stretched bond pulls the two particles together: force on ``j`` is
    # ``-t_e A n`` (``n`` from i to j), i.e. a restoring spring.
    np.add.at(forces, topo.i, bond_force)
    np.add.at(forces, topo.j, -bond_force)

    # ------------------------------------------------------------- contacts
    pairs = contact_pairs(positions, spec.radii, cfg)
    k_n = cfg.contact_stiffness_N_m
    k_t = cfg.contact_stiffness_t_N_m
    mu = cfg.contact_friction
    contact_energy = 0.0
    n_active = 0
    contact_edges = np.zeros((0, 2), dtype=np.int64)
    contact_magnitudes = np.zeros(0)
    if pairs.size:
        pi, pj = pairs[:, 0], pairs[:, 1]
        d_vec = positions[pj] - positions[pi]
        dist = np.linalg.norm(d_vec, axis=1)
        overlap = spec.radii[pi] + spec.radii[pj] - dist
        active = overlap > 0.0
        pi, pj, d_vec, dist, overlap = (pi[active], pj[active], d_vec[active],
                                        dist[active], overlap[active])
        n_active = int(pi.size)
        if n_active:
            normal = d_vec / dist[:, None]
            tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
            f_n = k_n * overlap
            anchor, known = friction.contact_anchor_for(pi, pj)
            rel = d_vec - anchor
            xi = np.where(known, np.einsum("ij,ij->i", rel, tangent), 0.0)
            f_t = -k_t * xi
            cap = mu * f_n
            f_t = np.clip(f_t, -cap, cap)
            force_vec = normal * f_n[:, None] + tangent * f_t[:, None]
            np.add.at(forces, pi, -force_vec)
            np.add.at(forces, pj, force_vec)
            contact_energy = float(0.5 * np.sum(f_n * overlap)
                                   + 0.5 * np.sum(f_t * f_t) / k_t)
            contact_edges = np.stack([pi, pj], axis=1)
            contact_magnitudes = np.linalg.norm(force_vec, axis=1)

    # ---------------------------------------------------------------- walls
    if friction.n_walls != len(spec.walls):
        friction.n_walls = len(spec.walls)
    wall_forces: dict[str, float] = {}
    wall_energy = 0.0
    wall_tangential_energy = 0.0
    for widx, wall in enumerate(spec.walls):
        pen = spec.radii - (positions - wall.point) @ wall.normal
        idx = np.nonzero(pen > 0.0)[0]
        wall_forces[wall.name] = 0.0
        if idx.size == 0:
            continue
        f_n = wall.stiffness_n * pen[idx]
        anchor, known = friction.wall_anchor_for(idx, widx)
        rel = positions[idx] - anchor
        xi = np.where(known, rel @ wall.tangent, 0.0)
        f_t = -wall.stiffness_t * xi
        cap = wall.friction * f_n
        f_t = np.clip(f_t, -cap, cap)
        force_vec = (wall.normal[None, :] * f_n[:, None]
                     + wall.tangent[None, :] * f_t[:, None])
        np.add.at(forces, idx, force_vec)
        wall_energy += float(0.5 * np.sum(f_n * pen[idx]))
        wall_tangential_energy += float(0.5 * np.sum(f_t * f_t)) / wall.stiffness_t
        wall_forces[wall.name] = float(np.sum(f_n))

    return {
        "forces": forces,
        "bond_trial": trial,
        "contact_pairs": pairs,
        "contact_edges": contact_edges,
        "contact_magnitudes": contact_magnitudes,
        "n_contacts": n_active,
        "contact_energy": contact_energy,
        "contact_friction_energy": 0.0,
        "wall_energy": wall_energy,
        "wall_tangential_energy": wall_tangential_energy,
        "wall_normal_forces": wall_forces,
    }


def commit_friction(
    spec: BondedSpecimen,
    positions: np.ndarray,
    friction: FrictionState,
) -> dict[str, float]:
    """Re-anchor the Cundall-Strack memory on the converged configuration.

    Rules (standard Cundall-Strack bookkeeping, stated because the energy ledger
    depends on them):

    * a **new** contact starts with zero tangential stretch, so it dissipates no
      slip work at birth;
    * an **inactive** contact keeps its anchor, so a contact that closes again
      resumes from its stored tangential state instead of from zero;
    * slip work is ``mu * F_n * |delta xi_slip|`` with
      ``delta xi_slip = (|F_t| - mu F_n) / k_t`` -- the amount by which the
      tangential spring has to be cut back.
    """
    cfg = spec.config
    slip_work = 0.0
    friction.n_nodes = spec.n_particles
    if friction.n_walls != len(spec.walls):
        friction.n_walls = len(spec.walls)

    pairs = contact_pairs(positions, spec.radii, cfg)
    k_t = cfg.contact_stiffness_t_N_m
    k_n = cfg.contact_stiffness_N_m
    mu = cfg.contact_friction
    new_keys = np.zeros(0, np.int64)
    new_anchor = np.zeros((0, 2))
    if pairs.size:
        pi, pj = pairs[:, 0], pairs[:, 1]
        d_vec = positions[pj] - positions[pi]
        dist = np.linalg.norm(d_vec, axis=1)
        overlap = spec.radii[pi] + spec.radii[pj] - dist
        active = overlap > 0.0
        pi, pj, d_vec, dist, overlap = (pi[active], pj[active], d_vec[active],
                                        dist[active], overlap[active])
        if pi.size:
            normal = d_vec / dist[:, None]
            tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
            anchor, known = friction.contact_anchor_for(pi, pj)
            xi = np.where(known, np.einsum("ij,ij->i", d_vec - anchor, tangent), 0.0)
            f_t = -k_t * xi
            cap = mu * k_n * overlap
            over = np.abs(f_t) > cap
            xi_post = np.where(over, -np.sign(f_t) * cap / k_t, xi)
            slip_work += float(np.sum(cap[over] * np.abs(xi[over] - xi_post[over])))
            new_keys = pi.astype(np.int64) * friction.n_nodes + pj
            new_anchor = d_vec - xi_post[:, None] * tangent
    # merge: anchors of pairs that are inactive now survive
    keep = ~np.isin(friction.contact_keys, new_keys)
    friction.contact_keys, friction.contact_anchor = _sorted_unique(
        np.concatenate([friction.contact_keys[keep], new_keys]),
        np.concatenate([friction.contact_anchor[keep], new_anchor], axis=0))

    keys_all: list[np.ndarray] = []
    anchors_all: list[np.ndarray] = []
    live_keys = np.zeros(0, np.int64)
    for widx, wall in enumerate(spec.walls):
        pen = spec.radii - (positions - wall.point) @ wall.normal
        idx = np.nonzero(pen > 0.0)[0]
        if idx.size == 0:
            continue
        f_n = wall.stiffness_n * pen[idx]
        pos_now = positions[idx]
        anchor, known = friction.wall_anchor_for(idx, widx)
        rel_t = np.where(known, (pos_now - anchor) @ wall.tangent, 0.0)
        f_t = -wall.stiffness_t * rel_t
        cap = wall.friction * f_n
        over = np.abs(f_t) > cap
        rel_post = np.where(over, -np.sign(f_t) * cap / wall.stiffness_t, rel_t)
        slip_work += float(np.sum(cap[over] * np.abs(rel_t[over] - rel_post[over])))
        keys = idx.astype(np.int64) * friction.n_walls + widx
        live_keys = np.concatenate([live_keys, keys])
        keys_all.append(keys)
        anchors_all.append(pos_now - rel_post[:, None] * wall.tangent[None, :])
    if keys_all:
        keys_new = np.concatenate(keys_all)
        anchor_new = np.concatenate(anchors_all)
        keepw = ~np.isin(friction.wall_keys, live_keys)
        friction.wall_keys, friction.wall_anchor = _sorted_unique(
            np.concatenate([friction.wall_keys[keepw], keys_new]),
            np.concatenate([friction.wall_anchor[keepw], anchor_new], axis=0))
    return {"slip_work_J": float(slip_work)}
