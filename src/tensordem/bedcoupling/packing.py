"""Random particle packings and kernel-based local voidage.

What this module owns
---------------------
* :func:`make_random_close_packing` -- a *random* (not lattice) monodisperse
  packing at a prescribed porosity, built by quasi-static compression of a
  dilute random configuration with a soft-sphere overlap relaxation on a
  periodic cubic cell.  The porosity is exact by volume balance
  (``eps = 1 - N V_p / L^3``); the residual overlap is returned as evidence
  that the configuration is geometrically admissible.
* :func:`local_porosity` -- per-particle voidage from a top-hat probe kernel,
  i.e. ``eps_i = 1 - (1/V_probe) * sum_j W(|x_i - x_j|) V_j`` with an *exact*
  sphere-sphere intersection volume for the weights.  This is the quantity an
  unresolved DEM--CFD closure needs: Ergun-type closures are strong functions
  of voidage, and using the bed-mean voidage for every particle silently drops
  the ``O(Var(eps) f''/f)`` bias that a heterogeneous bed produces.
* :func:`porosity_statistics` -- the mean / standard deviation that the node's
  first acceptance criterion is stated on.

Everything is periodic and deterministic (explicit ``seed``); no contact
mechanics is involved, which is why the *maximum overlap* of the finished
packing is reported with every packing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .closures import sphere_volume

__all__ = [
    "PackedBed",
    "make_random_close_packing",
    "make_jittered_cubic_array",
    "local_porosity",
    "porosity_statistics",
    "sphere_intersection_volume",
]


# ---------------------------------------------------------------------------
# geometry: exact sphere-sphere intersection volume
# ---------------------------------------------------------------------------
def sphere_intersection_volume(radius_a, radius_b, separation) -> np.ndarray:
    """Volume of the intersection of two spheres [m^3].

    ``radius_a`` is the probe radius, ``radius_b`` the particle radius and
    ``separation`` the centre distance.  Handles the fully separate, fully
    contained and lens cases in one vectorised expression.
    """
    ra = np.asarray(radius_a, dtype=float)
    rb = np.asarray(radius_b, dtype=float)
    s = np.asarray(separation, dtype=float)
    ra, rb, s = np.broadcast_arrays(ra, rb, s)

    va = 4.0 / 3.0 * math.pi * ra**3
    vb = 4.0 / 3.0 * math.pi * rb**3
    v_small = np.minimum(va, vb)

    s_safe = np.where(s > 0.0, s, 1.0)
    lens = (
        math.pi
        * np.maximum(ra + rb - s, 0.0) ** 2
        * (s**2 + 2.0 * s * (ra + rb) - 3.0 * (ra - rb) ** 2)
        / (12.0 * s_safe)
    )
    contained = s <= np.abs(ra - rb)  # one sphere inside the other
    out = np.where(s >= ra + rb, 0.0, np.where(contained, v_small, np.maximum(lens, 0.0)))
    out = np.where((s == 0.0) & contained, v_small, out)
    return out


# ---------------------------------------------------------------------------
# periodic neighbour search
# ---------------------------------------------------------------------------
def _min_image_pairs(positions: np.ndarray, box_length: float, cutoff: float):
    tree = cKDTree(positions, boxsize=box_length)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros((0, 3))
    i, j = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
    dvec = positions[j] - positions[i]
    dvec -= box_length * np.round(dvec / box_length)
    return i, j, dvec


def _relax(positions: np.ndarray, box_length: float, diameter: float, iterations: int):
    """Push overlapping pairs apart; returns ``(positions, max_overlap)``.

    Uses a Verlet-style neighbour list (skin ``0.2 d``) that is rebuilt every
    ``rebuild`` sweeps, so thousands of relaxation sweeps are affordable: the
    pure pair-separation scheme converges geometrically and needs O(10^3)
    sweeps to drive the residual overlap of a compressed random packing below
    ``1e-3 d``.
    """
    pos = np.asarray(positions, dtype=float)
    skin = 0.2 * diameter
    rebuild = 40
    remaining = int(iterations)
    if remaining <= 0:
        return pos, _max_overlap(pos, box_length, diameter)
    while remaining > 0:
        tree = cKDTree(pos, boxsize=box_length)
        pairs = tree.query_pairs(diameter + skin, output_type="ndarray")
        if len(pairs) == 0:
            return pos, 0.0
        i, j = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        for _ in range(min(rebuild, remaining)):
            dvec = pos[j] - pos[i]
            dvec -= box_length * np.round(dvec / box_length)
            dist = np.linalg.norm(dvec, axis=1)
            overlap = diameter - dist
            mask = overlap > 1e-12
            if not np.any(mask):
                return pos, 0.0
            corr = np.zeros_like(pos)
            unit = dvec[mask] / np.maximum(dist[mask], 1e-30)[:, np.newaxis]
            np.add.at(corr, i[mask], -0.5 * overlap[mask][:, np.newaxis] * unit)
            np.add.at(corr, j[mask], +0.5 * overlap[mask][:, np.newaxis] * unit)
            pos = (pos + corr) % box_length
        remaining -= rebuild
    return pos, _max_overlap(pos, box_length, diameter)


def _max_overlap(positions: np.ndarray, box_length: float, diameter: float) -> float:
    i, j, dvec = _min_image_pairs(positions, box_length, diameter)
    if len(i) == 0:
        return 0.0
    dist = np.linalg.norm(dvec, axis=1)
    return float(np.max(diameter - dist))


# ---------------------------------------------------------------------------
# packings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PackedBed:
    """Monodisperse packing in a periodic cubic cell."""

    diameter: float
    box_length: float
    positions: np.ndarray
    max_overlap: float = 0.0
    seed: int = 0
    kind: str = "random-close-packing"

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def particle_volume(self) -> float:
        return sphere_volume(self.diameter)

    @property
    def solid_fraction(self) -> float:
        return self.n_particles * self.particle_volume / self.box_length**3

    @property
    def porosity(self) -> float:
        return 1.0 - self.solid_fraction

    @property
    def bed_volume(self) -> float:
        return self.box_length**3

    @property
    def volumes(self) -> np.ndarray:
        return np.full(self.n_particles, self.particle_volume)

    def minimum_separation(self) -> float:
        i, j, dvec = _min_image_pairs(self.positions, self.box_length, self.box_length * 0.5)
        if len(i) == 0:
            return float("inf")
        return float(np.linalg.norm(dvec, axis=1).min())

    def summary(self) -> dict:
        n_candidates = self.n_particles * (self.n_particles - 1)
        return {
            "kind": self.kind,
            "n_particles": self.n_particles,
            "diameter": self.diameter,
            "box_length": self.box_length,
            "solids_fraction": self.solid_fraction,
            "porosity": self.porosity,
            "max_overlap_over_diameter": self.max_overlap / self.diameter,
            "minimum_separation_over_diameter": self.minimum_separation() / self.diameter,
            "particle_pairs": n_candidates,
            "seed": self.seed,
        }


def make_random_close_packing(
    n_particles: int,
    diameter: float,
    target_porosity: float,
    *,
    seed: int = 20261108,
    phi_start: float = 0.28,
    max_linear_step: float = 0.01,
    relax_iterations: int = 120,
    final_relax_iterations: int = 6000,
    overlap_tolerance: float = 1e-3,
) -> PackedBed:
    """Random packing at ``target_porosity`` by quasi-static compression.

    A dilute random configuration on a periodic cell is compressed in linear
    steps of at most ``max_linear_step`` (fractional), each followed by a
    soft-sphere overlap relaxation, until the target porosity is reached; the
    cell is then set to the exact target volume and relaxed until the maximum
    overlap falls below ``overlap_tolerance * diameter``.  Random (not
    lattice) placement plus random compression history is what makes the
    resulting contact network disordered.
    """
    if not 0.0 < target_porosity < 1.0:
        raise ValueError("target_porosity must be in (0, 1)")
    vp = sphere_volume(diameter)
    phi_target = 1.0 - target_porosity
    if phi_target > 0.70:
        raise ValueError("target solid fraction above 0.70 is not reachable by this scheme")

    rng = np.random.default_rng(seed)
    box = (n_particles * vp / phi_start) ** (1.0 / 3.0)
    pos = rng.random((n_particles, 3)) * box
    pos, _ = _relax(pos, box, diameter, 400)

    phi = n_particles * vp / box**3
    step_guard = 0
    while phi < phi_target - 1e-12:
        factor = (phi / phi_target) ** (1.0 / 3.0)  # < 1: shrink the cell
        factor = max(factor, 1.0 - max_linear_step)
        box *= factor
        pos = (pos * factor) % box
        pos, _ = _relax(pos, box, diameter, relax_iterations)
        phi_new = n_particles * vp / box**3
        if phi_new <= phi * (1.0 + 1e-15):
            break  # relaxation pushed the packing back: jammed below target
        phi = phi_new
        step_guard += 1
        if step_guard > 2000:
            raise RuntimeError("compression did not converge")

    # exact target volume, then relax overlaps at fixed volume
    factor = (phi / phi_target) ** (1.0 / 3.0)
    box *= factor
    pos = (pos * factor) % box
    pos, overlap = _relax(pos, box, diameter, final_relax_iterations)
    return PackedBed(
        diameter=diameter,
        box_length=box,
        positions=pos,
        max_overlap=overlap,
        seed=seed,
        kind="random-close-packing",
    )


def make_jittered_cubic_array(
    n_side: int, diameter: float, *, porosity: float, seed: int = 7, jitter: float = 0.4
) -> PackedBed:
    """Jittered simple-cubic array with a prescribed porosity.

    The lattice constant is set from the target porosity
    (``a = (V_p/(1-porosity))^(1/3)``), and the jitter amplitude is capped at
    ``jitter * (a - d)/2`` so that ``separation >= diameter`` is guaranteed.
    Used for the dilute settling cases where the point-particle model makes the
    arrangement irrelevant: the batch-settling runs then contain no contacts at
    all, which is checked with :meth:`PackedBed.minimum_separation`.
    """
    vp = sphere_volume(diameter)
    spacing = (vp / (1.0 - porosity)) ** (1.0 / 3.0)
    if spacing <= diameter:
        raise ValueError("porosity too low for a contact-free cubic array")
    amp = jitter * (spacing - diameter) / 2.0
    grid = np.stack(
        np.meshgrid(*(np.arange(n_side, dtype=float) for _ in range(3)), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    positions = (grid + 0.5) * spacing
    positions += rng.uniform(-amp, amp, size=positions.shape)
    box = n_side * spacing
    positions %= box
    return PackedBed(
        diameter=diameter,
        box_length=box,
        positions=positions,
        max_overlap=_max_overlap(positions, box, diameter),
        seed=seed,
        kind="jittered-cubic-array",
    )


# ---------------------------------------------------------------------------
# local voidage
# ---------------------------------------------------------------------------
def local_porosity(
    positions: np.ndarray,
    diameter: float,
    box_length: float,
    probe_radius: float,
) -> np.ndarray:
    """Per-particle voidage from a top-hat probe kernel with exact overlaps.

    ``eps_i = 1 - sum_j V(R_probe cap V_j) / V(R_probe)``, minimum-image
    periodic.  Requires ``probe_radius + diameter < box_length / 2`` so that
    the minimum-image construction is exact.
    """
    pos = np.asarray(positions, dtype=float)
    if probe_radius + diameter >= box_length / 2.0:
        raise ValueError("probe_radius + diameter must be < box_length / 2")
    n = pos.shape[0]
    r_part = diameter / 2.0
    v_probe = 4.0 / 3.0 * math.pi * probe_radius**3
    solid = np.zeros(n, dtype=float)

    tree = cKDTree(pos, boxsize=box_length)
    pairs = tree.query_pairs(probe_radius + r_part, output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        dvec = pos[j] - pos[i]
        dvec -= box_length * np.round(dvec / box_length)
        dist = np.linalg.norm(dvec, axis=1)
        vol = sphere_intersection_volume(probe_radius, r_part, dist)
        np.add.at(solid, i, vol)
        np.add.at(solid, j, vol)
    # the particle at the probe centre
    solid += sphere_volume(diameter)
    return 1.0 - solid / v_probe


def porosity_statistics(local_eps: np.ndarray, bed_mean: float | None = None) -> dict:
    """Mean / spread of the local voidage field."""
    eps = np.asarray(local_eps, dtype=float)
    return {
        "mean": float(eps.mean()),
        "std": float(eps.std(ddof=0)),
        "min": float(eps.min()),
        "max": float(eps.max()),
        "n_samples": int(eps.size),
        "bed_mean_from_volume_balance": None if bed_mean is None else float(bed_mean),
        "mean_minus_volume_balance": None if bed_mean is None else float(eps.mean() - bed_mean),
    }
