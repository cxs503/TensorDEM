"""Gravity-deposition random packing of monodisperse spheres (3-D DEM).

What this module owns
---------------------
A packings measurement that is **not imposed by the analyst**: spheres are
released into a box with a floor, they fall, collide and settle, and the bulk
solid fraction is read off the *free-surface geometry* of the settled bed.
There is no target porosity anywhere in the pipeline (contrast
:func:`tensordem.bedcoupling.packing.make_random_close_packing`, whose returned
solid fraction equals ``1 - target_porosity`` to machine precision, so a gate
on it cannot fail -- see ``tests/test_rcp_deposition.py`` and the m16308 audit
in ``benchmarks/rcp_jamming_audit``).

Physics
-------
* normal contact: linear spring-dashpot, ``F_n = max(k_n delta - gamma v_n, 0)``
  with ``gamma`` matched to a coefficient of restitution (identical convention
  to :mod:`tensordem.granular3d_core`, whose :func:`damping_ratio` is reused
  here; the pair/state machinery below is new because it must be periodic in
  the two horizontal directions, have a frictional floor and keep the
  per-pair history in numpy arrays rather than Python dicts);
* tangential contact: Cundall-Strack incremental spring, saturated at
  ``mu |F_n|``, per-pair history stored on the persistent candidate list;
* rolling: optional isotropic rolling spring saturated at ``mu_r R |F_n|``;
* gravity ``-m g z^`` and a floor plane at ``z = 0`` with the same
  spring-dashpot/Coulomb law, wall friction switchable so the frictionless
  tier is genuinely frictionless (no hidden tangential force anywhere).

Integration
-----------
Velocity Verlet on translation and rotation with ``dt = t_c / 20`` where
``t_c = pi sqrt(m_eff / k_n)`` is the linear-spring contact time.  The
tangential history is advanced with the current contact-point velocity before
the force summation (first-order Cundall-Strack update, as in
:func:`tensordem.granular3d_core.contact_forces`).

Measurement
-----------
:func:`measure_packing` bins the settled configuration into horizontal slabs
with a cloud-in-cell (linear) assignment of each sphere centre and reports

* ``phi_bulk`` -- volume-weighted mean profile between ``2.5 d`` above the
  floor and ``2.5 d`` below the free surface (both wall layers dropped);
* ``phi_bulk_overlap_corrected`` -- the same with each sphere deflated by its
  own average overlap (soft-sphere packings are compressible, so this is a
  documented systematic; with ``delta/d ~ 3e-3`` the correction is ~1 % and the
  two numbers must agree to 1 % for the claim not to ride on it);
* ``coordination_bulk`` / ``z12_fraction`` -- mean contacts per bulk particle
  and the fraction with exactly 12 contacts (crystalline-order proxy);
* residual overlaps, free-surface height, the profile itself.

Cost
----
Pairs come from a cKDTree on a periodic-in-x,y box refreshed under a 20 % skin,
so a step is O(N_contacts) numpy work; the per-pair state lives on sorted
integer keys (``searchsorted``), not in dicts.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from .granular3d_core import damping_ratio

__all__ = [
    "DepositionSpec",
    "DepositionResult",
    "run_deposition",
    "measure_packing",
    "packing_profile",
    "initial_configuration",
]


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepositionSpec:
    """Case definition (all quantities in one self-consistent unit system).

    Defaults are dimensionless: diameter ``d = 1``, density ``rho = 1``,
    gravity ``g = 1``, so the gravity timescale ``sqrt(d/g) = 1``.
    """

    n_particles: int = 2000
    mu: float = 0.0                 # inter-particle friction coefficient
    mu_wall: float | None = None    # floor friction; ``None`` -> same as mu
    mu_rolling: float = 0.0
    kn: float = 4000.0              # normal stiffness
    kt_ratio: float = 0.5           # k_t = kt_ratio * k_n
    restitution: float = 0.1
    radius: float = 0.5
    density: float = 1.0
    gravity: float = 1.0
    seed: int = 1
    initial_phi: float = 0.18       # placement solid fraction (dilute)
    target_phi_guess: float = 0.64  # only sets the box aspect ratio
    target_aspect: float = 1.0      # settled height / box width
    dt_frac: float = 1.0 / 20.0     # dt = dt_frac * t_c
    t_max: float = 150.0
    settle_tol: float = 5.0e-4      # KE/(N m g d) threshold for "settled"
    settle_window: float = 4.0      # gravity times of sustained low KE
    sample_every: float = 0.5
    # -- analytic-anchor hooks (tests use them; the deposition keeps them zero)
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_omega: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_positions: Any = None   # (n, 3) array overriding the placement

    # -- derived -----------------------------------------------------------
    @property
    def particle_volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def mass(self) -> float:
        return self.density * self.particle_volume

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def mu_floor(self) -> float:
        return self.mu if self.mu_wall is None else float(self.mu_wall)

    @property
    def contact_time(self) -> float:
        """Linear-spring contact time ``pi sqrt(m_eff / k_n)``."""
        return math.pi * math.sqrt(0.5 * self.mass / self.kn)

    @property
    def dt(self) -> float:
        return self.dt_frac * self.contact_time

    @property
    def box_width(self) -> float:
        """Horizontal box length; the settled height is ``target_aspect`` x it."""
        return (self.n_particles * self.particle_volume
                / (self.target_phi_guess * self.target_aspect)) ** (1.0 / 3.0)

    @property
    def settled_height_guess(self) -> float:
        return self.target_aspect * self.box_width

    @property
    def initial_height(self) -> float:
        return (self.n_particles * self.particle_volume
                / (self.initial_phi * self.box_width ** 2))

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            diameter=self.diameter,
            mass=self.mass,
            box_width=self.box_width,
            dt=self.dt,
            contact_time=self.contact_time,
            initial_height=self.initial_height,
            settled_height_guess=self.settled_height_guess,
            mu_floor=self.mu_floor,
            zeta=damping_ratio(self.restitution),
        )
        return d


# ---------------------------------------------------------------------------
# initial configuration: random sequential addition (no lattice imprint)
# ---------------------------------------------------------------------------
def initial_configuration(n: int, box_width: float, height: float, radius: float,
                          seed: int, min_sep_factor: float = 1.02,
                          max_tries_per_particle: int = 4000
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    """Place ``n`` spheres uniformly at random with a hard core, periodic in x
    and y, inside ``[0, box_width)^2 x [0, height)``.

    A random placement (not a lattice) matters: a lattice seed imprints
    crystalline order that survives the deposition and inflates the measured
    solid fraction.
    """
    rng = np.random.default_rng(seed)
    min_sep = min_sep_factor * 2.0 * radius
    cell = min_sep
    n_cells = max(1, int(math.floor(box_width / cell)))
    grid: dict[tuple[int, int, int], list[int]] = {}
    pos = np.zeros((n, 3))
    rejected = 0
    for i in range(n):
        for _ in range(max_tries_per_particle):
            p = rng.random(3) * np.array([box_width, box_width, height - radius])
            p[2] += radius
            # wrap the cell *index* (the box width is not an integer multiple
            # of the cell size, so ``int(p[0] // cell)`` can equal n_cells)
            cx = int(p[0] // cell) % n_cells
            cy = int(p[1] // cell) % n_cells
            cz = int(p[2] // cell)
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = ((cx + dx) % n_cells, (cy + dy) % n_cells, cz + dz)
                        for j in grid.get(key, ()):  # pragma: no branch
                            dv = p - pos[j]
                            dv[0] -= box_width * round(dv[0] / box_width)
                            dv[1] -= box_width * round(dv[1] / box_width)
                            if dv @ dv < min_sep * min_sep:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                pos[i] = p
                grid.setdefault((cx, cy, cz), []).append(i)
                break
            rejected += 1
        else:  # pragma: no cover - only reachable for an over-tight request
            raise RuntimeError(
                f"random sequential addition failed at particle {i}: "
                f"phi={n * 4.0 / 3.0 * math.pi * radius ** 3 / (box_width ** 2 * height):.3f} "
                f"is too high for min separation {min_sep:.3f}")
    achieved_phi = (n * (4.0 / 3.0 * math.pi * radius ** 3)
                    / (box_width ** 2 * height))
    info = {
        "placement": "random_sequential_addition",
        "min_separation": min_sep,
        "placement_phi": achieved_phi,
        "rejected_candidates": rejected,
    }
    return pos, info


# ---------------------------------------------------------------------------
# persistent candidate list with vectorised per-pair state
# ---------------------------------------------------------------------------
class _PairList:
    """Candidate pair list with a skin; per-pair state on sorted integer keys.

    State (tangential / rolling history) is stored in arrays aligned with the
    sorted key array, so a rebuild remaps it with one ``searchsorted`` instead
    of a Python loop over pairs.
    """

    def __init__(self, skin: float) -> None:
        self.skin = skin
        self.iu = np.zeros(0, dtype=np.int64)
        self.ju = np.zeros(0, dtype=np.int64)
        self.keys = np.zeros(0, dtype=np.int64)
        self.tang = np.zeros((0, 3))
        self.roll = np.zeros((0, 3))
        self.origin: np.ndarray | None = None
        self.rebuilds = 0

    @staticmethod
    def min_image(dvec: np.ndarray, width: float) -> np.ndarray:
        """Minimum image in x and y only (z has a floor, not a ceiling)."""
        dvec[:, 0] -= width * np.round(dvec[:, 0] / width)
        dvec[:, 1] -= width * np.round(dvec[:, 1] / width)
        return dvec

    def needs_rebuild(self, pos: np.ndarray, width: float) -> bool:
        """True when some particle moved more than ``skin / 2``.

        The displacement is taken as a *minimum image* in x, y, otherwise the
        periodic wrap of a particle's stored coordinate (a jump of one box
        width) would force a rebuild every few steps.
        """
        if self.origin is None:
            return True
        disp = self.min_image(pos - self.origin, width)
        return bool(np.max(np.einsum("ij,ij->i", disp, disp))
                    > (0.5 * self.skin) ** 2)

    def rebuild(self, pos: np.ndarray, radius: float, width: float) -> None:
        n = len(pos)
        cutoff = 2.0 * radius + self.skin
        # z is not periodic: hand the tree a periodic length far above the bed
        tree = cKDTree(pos, boxsize=(width, width, 1.0e6))
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        if len(pairs) == 0:
            iu = np.zeros(0, dtype=np.int64)
            ju = np.zeros(0, dtype=np.int64)
        else:
            iu = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
            ju = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
        new_keys = iu * n + ju
        order = np.argsort(new_keys, kind="stable")
        new_keys = new_keys[order]
        self.iu, self.ju = iu[order], ju[order]
        self.tang = self._remap(self.keys, self.tang, new_keys)
        self.roll = self._remap(self.keys, self.roll, new_keys)
        self.keys = new_keys
        self.origin = pos.copy()
        self.rebuilds += 1

    @staticmethod
    def _remap(old_keys: np.ndarray, state: np.ndarray,
               new_keys: np.ndarray) -> np.ndarray:
        out = np.zeros((len(new_keys), 3))
        if len(old_keys) == 0 or len(new_keys) == 0:
            return out
        idx = np.searchsorted(old_keys, new_keys)
        idx_c = np.clip(idx, 0, len(old_keys) - 1)
        hit = old_keys[idx_c] == new_keys
        out[hit] = state[idx_c[hit]]
        return out

    def rows_of(self, keys: np.ndarray) -> np.ndarray:
        """Row index of every key (the keys must be a subset of the list)."""
        if len(self.keys) == 0:
            return np.zeros(0, dtype=np.int64)
        idx = np.searchsorted(self.keys, keys)
        return np.clip(idx, 0, len(self.keys) - 1)

    def branch_vectors(self, pos: np.ndarray, width: float
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Branch vector i -> j (minimum image in x, y) and the distance."""
        dvec = self.min_image(pos[self.ju] - pos[self.iu], width)
        return dvec, np.linalg.norm(dvec, axis=1)


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class DepositionResult:
    spec: DepositionSpec
    positions: np.ndarray
    velocities: np.ndarray
    omegas: np.ndarray
    positions_initial: np.ndarray
    metrics: dict[str, Any]
    history: dict[str, list[float]]
    settled: bool
    final_time: float
    steps: int
    wall_clock: float
    impl_sha256: str
    placement: dict[str, Any] = field(default_factory=dict)
    rebuilds: int = 0

    def summary(self) -> dict[str, Any]:
        metrics = {k: v for k, v in self.metrics.items()
                   if not isinstance(v, dict)}
        profiles = {k: v for k, v in self.metrics.items() if isinstance(v, dict)}
        return {
            "spec": self.spec.as_dict(),
            "metrics": metrics,
            "profiles": {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
                         for k, v in profiles.items()},
            "settled": self.settled,
            "final_time": self.final_time,
            "steps": self.steps,
            "wall_clock": self.wall_clock,
            "impl_sha256": self.impl_sha256,
            "placement": self.placement,
            "pair_list_rebuilds": self.rebuilds,
            "history": {k: list(v) for k, v in self.history.items()},
        }


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def packing_profile(positions: np.ndarray, volumes: np.ndarray, area: float,
                    bin_width: float, height: float) -> dict[str, np.ndarray]:
    """Solid-fraction profile ``phi(z)`` with cloud-in-cell (linear) weights.

    Each sphere contributes its volume to the two slabs bracketing its centre,
    which removes the staircase bias of nearest-bin assignment at
    ``bin_width`` resolution.
    """
    n_bins = max(1, int(math.ceil(height / bin_width)))
    edges = np.arange(n_bins + 1) * bin_width
    z = np.clip(positions[:, 2], edges[0], edges[-1] - 1e-12)
    frac = (z - edges[0]) / bin_width
    lo = np.clip(np.floor(frac).astype(np.int64), 0, n_bins - 1)
    w = frac - lo
    phi = np.zeros(n_bins)
    np.add.at(phi, lo, volumes * (1.0 - w))
    np.add.at(phi, np.minimum(lo + 1, n_bins - 1), volumes * w)
    phi /= area * bin_width
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {"z": centres, "phi": phi, "edges": edges}


def _band_mean(prof: dict[str, np.ndarray], z_lo: float, z_hi: float) -> float:
    z, phi = prof["z"], prof["phi"]
    mask = (z >= z_lo) & (z <= z_hi)
    if not mask.any():
        return float("nan")
    return float(np.mean(phi[mask]))


def measure_packing(spec: DepositionSpec, positions: np.ndarray,
                    pair_i: np.ndarray, pair_j: np.ndarray,
                    pair_overlap: np.ndarray, wall_overlap: np.ndarray,
                    n_contacts: np.ndarray,
                    bin_width: float | None = None) -> dict[str, Any]:
    """Bulk solid fraction and diagnostics for one settled configuration."""
    r = spec.radius
    d = spec.diameter
    area = spec.box_width ** 2
    bin_width = bin_width or 0.5 * d
    n = len(positions)

    # per-particle mean overlap over its active pair contacts
    overlap_sum = np.zeros(n)
    overlap_cnt = np.zeros(n)
    np.add.at(overlap_sum, pair_i, pair_overlap)
    np.add.at(overlap_cnt, pair_i, 1.0)
    np.add.at(overlap_sum, pair_j, pair_overlap)
    np.add.at(overlap_cnt, pair_j, 1.0)
    mean_overlap = np.where(overlap_cnt > 0,
                            overlap_sum / np.maximum(overlap_cnt, 1.0),
                            2.0 * np.maximum(wall_overlap, 0.0))

    surface_height = float(np.max(positions[:, 2]) + r)
    vol = np.full(n, spec.particle_volume)
    prof_raw = packing_profile(positions, vol, area, bin_width, surface_height)

    # overlap-corrected geometry: each sphere deflated by its own mean overlap
    d_eff = np.maximum(d - mean_overlap, 0.5 * d)
    vol_eff = np.pi / 6.0 * d_eff ** 3
    surf_eff = float(np.max(positions[:, 2] + 0.5 * d_eff))
    prof_eff = packing_profile(positions, vol_eff, area, bin_width, surf_eff)

    z_lo = 2.5 * d
    z_hi = surface_height - 2.5 * d
    phi_bulk = _band_mean(prof_raw, z_lo, z_hi)
    phi_bulk_eff = _band_mean(prof_eff, z_lo, z_hi)
    phi_vol_corr = float(np.sum(vol_eff) / (area * surf_eff))
    phi_surface = float(n * spec.particle_volume / (area * surface_height))

    bulk_mask = (positions[:, 2] >= z_lo) & (positions[:, 2] <= max(z_hi, z_lo))
    z_bulk = float(np.mean(n_contacts[bulk_mask])) if bulk_mask.any() else float("nan")
    z12 = float(np.mean(n_contacts[bulk_mask] == 12)) if bulk_mask.any() else float("nan")

    zb, pb = prof_raw["z"], prof_raw["phi"]
    m = (zb >= z_lo) & (zb <= max(z_hi, z_lo)) & (pb > 0)
    slope = (float(np.polyfit(np.log(zb[m]), np.log(pb[m]), 1)[0])
             if int(m.sum()) >= 4 else float("nan"))

    return {
        "phi_bulk": phi_bulk,
        "phi_bulk_overlap_corrected": phi_bulk_eff,
        "phi_volume_balance_overlap_corrected": phi_vol_corr,
        "phi_surface_to_floor": phi_surface,
        "surface_height": surface_height,
        "surface_height_overlap_corrected": surf_eff,
        "band": [z_lo, z_hi],
        "bin_width": bin_width,
        "mean_overlap_over_d": float(np.mean(mean_overlap) / d),
        "max_pair_overlap_over_d": float(np.max(pair_overlap, initial=0.0) / d),
        "max_wall_overlap_over_d": float(np.max(wall_overlap, initial=0.0) / d),
        "coordination_bulk": z_bulk,
        "z12_fraction": z12,
        "n_bulk_particles": int(bulk_mask.sum()),
        "n_contacts_total": int(np.sum(n_contacts)),
        "profile_loglog_slope_bulk": slope,
        "profile": prof_raw,
        "profile_overlap_corrected": prof_eff,
    }



def _accumulate(target: np.ndarray, idx: np.ndarray, values: np.ndarray) -> None:
    """``target[idx] += values`` per component via ``bincount`` (fast on DEM sizes)."""
    if len(idx) == 0:
        return
    for k in range(target.shape[1]):
        target[:, k] += np.bincount(idx, weights=values[:, k],
                                    minlength=len(target))



def _instant_bulk_phi(pos: np.ndarray, spec: DepositionSpec, area: float,
                      r: float, d: float) -> float:
    """Bulk solid fraction of the *current* configuration (plateau witness)."""
    surf = float(np.max(pos[:, 2]) + r)
    prof = packing_profile(pos, np.full(len(pos), spec.particle_volume), area,
                           0.5 * d, surf)
    return _band_mean(prof, z_lo=2.5 * d, z_hi=surf - 2.5 * d)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _run_deposition_core(spec: DepositionSpec,
                         progress: Callable[[dict[str, Any]], None] | None = None
                         ) -> dict[str, Any]:
    """Inner loop (kept separate so its source hash identifies the physics)."""
    rng = np.random.default_rng(spec.seed + 7919)
    d = spec.diameter
    r = spec.radius
    width = spec.box_width
    area = width ** 2
    m = spec.mass
    inertia = spec.inertia
    g = spec.gravity
    kn = spec.kn
    kt = spec.kt_ratio * kn
    kr = 2.5 * kt * r * r           # rolling spring: U_r = 0.5 kr |roll|^2
    zeta = damping_ratio(spec.restitution)
    m_eff_pair = 0.5 * m
    gamma_pair = 2.0 * zeta * math.sqrt(kn * m_eff_pair)
    gamma_wall = 2.0 * zeta * math.sqrt(kn * m)
    mu = spec.mu
    mu_r = spec.mu_rolling
    mu_floor = spec.mu_floor
    n = spec.n_particles
    dt = spec.dt

    if spec.initial_positions is not None:
        pos = np.asarray(spec.initial_positions, dtype=np.float64).copy()
        pos[:, 0] %= width
        pos[:, 1] %= width
        placement = {"placement": "provided_initial_positions",
                     "initial_height": float(np.max(pos[:, 2]) + r)}
    else:
        height0 = spec.initial_height
        for _attempt in range(8):
            try:
                pos, placement = initial_configuration(n, width, height0, r,
                                                       spec.seed)
                break
            except RuntimeError:
                height0 *= 1.3  # RSA has a finite saturation density; lower it
        else:  # pragma: no cover - would need an absurd request
            raise RuntimeError("could not place a dilute initial configuration")
        placement["requested_initial_phi"] = spec.initial_phi
        placement["initial_height"] = height0
        placement["drop_height"] = height0 - spec.settled_height_guess
    pos_initial = pos.copy()
    vel = np.tile(np.asarray(spec.initial_velocity, dtype=np.float64), (n, 1))
    omega = np.tile(np.asarray(spec.initial_omega, dtype=np.float64), (n, 1))
    pairs = _PairList(skin=0.2 * d)
    wall_tang = np.zeros((n, 3))

    hist: dict[str, list[float]] = {k: [] for k in (
        "t", "ke_norm", "pe_norm", "elastic_norm", "total_norm",
        "n_pair_contacts", "n_wall_contacts", "max_overlap_over_d",
        "phi_bulk_instant")}
    t = 0.0
    step = 0
    next_sample = 0.0
    low_ke_since: float | None = None
    settled = False
    t_start = time.perf_counter()

    def contact_forces(p_pos: np.ndarray, p_vel: np.ndarray, p_omega: np.ndarray
                       ) -> tuple:
        """Forces, torques, contacts and elastic energy at one configuration."""
        nonlocal wall_tang
        if pairs.needs_rebuild(p_pos, width):
            pairs.rebuild(p_pos, r, width)
        force = np.zeros_like(p_pos)
        torque = np.zeros_like(p_vel)
        n_contacts = np.zeros(n)
        elastic = 0.0

        dvec, dist = pairs.branch_vectors(p_pos, width)
        overlap = d - dist
        active = overlap > 0.0
        pair_i = pairs.iu[active]
        pair_j = pairs.ju[active]
        pair_ov = overlap[active]
        n_pair = 0
        if len(pair_i):
            n_pair = len(pair_i)
            dist_a = dist[active]
            nvec = dvec[active] / dist_a[:, None]
            v_rel = p_vel[pair_j] - p_vel[pair_i]
            v_n = np.einsum("ij,ij->i", v_rel, nvec)
            f_n = np.maximum(kn * pair_ov - gamma_pair * v_n, 0.0)
            elastic += 0.5 * kn * float(np.sum(pair_ov ** 2))
            f_t = np.zeros_like(nvec)
            keys = pairs.keys[active]
            rows = pairs.rows_of(keys)
            if mu > 0.0:
                # contact-point velocity of j relative to i: the surface
                # velocity of i at (r_i + r n) is  v_i + w_i x (r n), that of
                # j at (r_j - r n) is  v_j - w_j x (r n); the sign of the i
                # term therefore differs from the j term (getting this wrong
                # anti-damps the spin and pumps energy -- it was the m16448
                # frictional-tier blow-up).
                surf_i = np.cross(p_omega[pair_i], nvec * r)
                surf_j = np.cross(p_omega[pair_j], -nvec * r)
                v_t = v_rel - v_n[:, None] * nvec + (surf_j - surf_i)
                tang = pairs.tang[rows] + v_t * dt
                tnorm = np.linalg.norm(tang, axis=1)
                cap = mu * f_n / kt
                scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-300), 1.0)
                tang *= scale[:, None]
                pairs.tang[rows] = tang
                f_t = -kt * tang
                elastic += 0.5 * kt * float(np.sum(tang ** 2))
            f_total = f_n[:, None] * nvec + f_t
            _accumulate(force, pair_i, -f_total)
            _accumulate(force, pair_j, f_total)
            if mu > 0.0:
                t_i = np.cross(-nvec * r, f_t)
                _accumulate(torque, pair_i, t_i)
                _accumulate(torque, pair_j, t_i)
            if mu_r > 0.0:
                spin = p_omega[pair_j] - p_omega[pair_i]
                roll = pairs.roll[rows] + spin * dt
                rnorm = np.linalg.norm(roll, axis=1)
                cap_r = (mu_r * r * f_n) / kr
                scale_r = np.where(rnorm > cap_r,
                                   cap_r / np.maximum(rnorm, 1e-300), 1.0)
                roll *= scale_r[:, None]
                pairs.roll[rows] = roll
                torque_r = -kr * roll
                elastic += 0.5 * kr * float(np.sum(roll ** 2))
                _accumulate(torque, pair_i, -torque_r)
                _accumulate(torque, pair_j, torque_r)
            np.add.at(n_contacts, pair_i, 1.0)
            np.add.at(n_contacts, pair_j, 1.0)  # counts, not forces: cheap

        # floor contact (plane z = 0)
        z = p_pos[:, 2]
        wall_ov = np.maximum(r - z, 0.0)
        wall_on = wall_ov > 0.0
        n_wall = int(wall_on.sum())
        if n_wall:
            v_n_w = p_vel[wall_on, 2]
            f_n_w = np.maximum(kn * wall_ov[wall_on] - gamma_wall * v_n_w, 0.0)
            elastic += 0.5 * kn * float(np.sum(wall_ov[wall_on] ** 2))
            idx_w = np.flatnonzero(wall_on)
            force[idx_w, 2] += f_n_w
            if mu_floor > 0.0:
                # contact-point velocity: v + omega x (-r z^)
                v_t = np.stack([p_vel[idx_w, 0] - r * p_omega[idx_w, 1],
                                p_vel[idx_w, 1] + r * p_omega[idx_w, 0]], axis=1)
                tang = wall_tang[idx_w, :2] + v_t * dt
                tnorm = np.linalg.norm(tang, axis=1)
                cap = mu_floor * f_n_w / kt
                scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-300), 1.0)
                tang *= scale[:, None]
                wall_tang[idx_w, :2] = tang
                wall_tang[idx_w, 2] = 0.0
                f_t = -kt * tang
                force[idx_w, 0] += f_t[:, 0]
                force[idx_w, 1] += f_t[:, 1]
                torque[idx_w, 0] += r * f_t[:, 1]
                torque[idx_w, 1] -= r * f_t[:, 0]
                elastic += 0.5 * kt * float(np.sum(tang ** 2))
            np.add.at(n_contacts, idx_w, 1.0)
        else:
            idx_w = np.zeros(0, dtype=np.int64)
        # particles that lost the floor must not keep a stale tangential spring
        off = ~wall_on
        if np.any(off):
            wall_tang[off] = 0.0

        force[:, 2] -= m * g
        return (force, torque, n_contacts, pair_i, pair_j, pair_ov,
                wall_ov, elastic, n_pair, n_wall)

    # ---- initial force pass --------------------------------------------------
    (force, torque, n_contacts, pair_i, pair_j, pair_ov, wall_ov, elastic,
     n_pair, n_wall) = contact_forces(pos, vel, omega)
    acc = force / m
    alpha = torque / inertia if mu > 0.0 and mu_r > 0.0 else torque / inertia

    while t < spec.t_max - 1e-12:
        vel += 0.5 * dt * acc
        omega += 0.5 * dt * alpha
        pos += dt * vel
        pos[:, 0] %= width
        pos[:, 1] %= width
        step += 1
        t += dt
        (force, torque, n_contacts, pair_i, pair_j, pair_ov, wall_ov, elastic,
         n_pair, n_wall) = contact_forces(pos, vel, omega)
        acc = force / m
        alpha = torque / inertia
        vel += 0.5 * dt * acc
        omega += 0.5 * dt * alpha

        if t >= next_sample:
            ke = 0.5 * m * float(np.sum(vel ** 2)) + 0.5 * inertia * float(
                np.sum(omega ** 2))
            pe = m * g * float(np.sum(pos[:, 2]))
            ke_norm = ke / (n * m * g * d)
            hist["t"].append(t)
            hist["ke_norm"].append(ke_norm)
            hist["pe_norm"].append(pe / (n * m * g * d))
            hist["elastic_norm"].append(elastic / (n * m * g * d))
            hist["total_norm"].append((ke + pe + elastic) / (n * m * g * d))
            hist["n_pair_contacts"].append(float(n_pair))
            hist["n_wall_contacts"].append(float(n_wall))
            hist["max_overlap_over_d"].append(
                float(max(np.max(pair_ov, initial=0.0),
                          np.max(wall_ov, initial=0.0)) / d))
            hist["phi_bulk_instant"].append(
                _instant_bulk_phi(pos, spec, area, r, d))
            if progress is not None:
                progress({"t": t, "ke_norm": ke_norm, "n_pair": n_pair,
                          "n_wall": n_wall, "step": step})
            next_sample = t + spec.sample_every
            if ke_norm < spec.settle_tol:
                if low_ke_since is None:
                    low_ke_since = t
                elif t - low_ke_since >= spec.settle_window:
                    settled = True
                    break
            else:
                low_ke_since = None

    wall_clock = time.perf_counter() - t_start
    metrics = measure_packing(spec, pos, pair_i, pair_j, pair_ov, wall_ov,
                              n_contacts)
    metrics["ke_norm_final"] = hist["ke_norm"][-1] if hist["ke_norm"] else float("inf")
    metrics["max_velocity_final"] = float(np.max(np.linalg.norm(vel, axis=1)))
    metrics["dt"] = dt
    metrics["kn"] = kn
    return {
        "positions": pos,
        "velocities": vel,
        "omegas": omega,
        "positions_initial": pos_initial,
        "metrics": metrics,
        "history": hist,
        "settled": settled,
        "final_time": t,
        "steps": step,
        "wall_clock": wall_clock,
        "placement": placement,
        "rebuilds": pairs.rebuilds,
        "impl_sha256": inspect_source_sha256(),
    }


def inspect_source_sha256() -> str:
    """SHA-256 of this module's source (provenance stamp for run artifacts)."""
    import pathlib
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


def run_deposition(spec: DepositionSpec,
                   progress: Callable[[dict[str, Any]], None] | None = None
                   ) -> DepositionResult:
    """Release ``spec.n_particles`` spheres and settle them under gravity."""
    out = _run_deposition_core(spec, progress=progress)
    return DepositionResult(
        spec=spec,
        positions=out["positions"],
        velocities=out["velocities"],
        omegas=out["omegas"],
        positions_initial=out["positions_initial"],
        metrics=out["metrics"],
        history=out["history"],
        settled=out["settled"],
        final_time=out["final_time"],
        steps=out["steps"],
        wall_clock=out["wall_clock"],
        impl_sha256=out["impl_sha256"],
        placement=out["placement"],
        rebuilds=out["rebuilds"],
    )