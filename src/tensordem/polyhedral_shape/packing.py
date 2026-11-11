"""Shape-dependent random packing of convex particles.

Two protocols are implemented, because "the packing fraction" of a shape family
is only meaningful together with its deposition history:

``compaction``
    Lubachevsky--Stillinger style isotropic compression in a periodic cube:
    particles start dilute, are grown quasi-statically, and overlaps are relaxed
    by successive projection onto the contact constraints.  The run stops at the
    jamming point (the largest scale at which overlaps can still be resolved),
    i.e. this is the *dense* (RCP) end of the packing spectrum.  For
    monodisperse spheres it converges to ``phi ~ 0.64``, the classical value.

``deposition``
    Ballistic/sequential deposition under gravity: particles are dropped one at
    a time with a random in-plane position and a random orientation and come to
    rest on the topmost admissible height, found by an exact convex overlap
    predicate (exponential search down from above, then bisection).  This is the
    *loose* (RLP) end.  ``trials > 1`` keeps the lowest of several candidate
    drops, which mimics tapping/rolling and interpolates towards the dense end.

Why two protocols: for spheres the two ends are ``0.60`` (RLP) and ``0.64``
(RCP); for cubes and spheroids the loose end is around ``0.55--0.62`` and the
dense end ``0.62--0.70``.  Reporting one number per shape without stating the
protocol would be meaningless, so every result carries ``protocol``.

Overlap predicates
------------------
* boxes: exact 15-axis separating-axis test, vectorised over all placed
  particles (standard box--box SAT, exact for boxes);
* arbitrary convex polyhedra / ellipsoids: GJK
  (``tensordem.convex_contact.gjk_distance``) decides the boolean overlap, with
  an AABB + bounding-sphere pre-filter.

The packing fraction is ``N V_particle / V_box`` with ``V_particle`` the exact
convex-hull volume; the box is periodic in ``x`` and ``y`` and bounded by a
floor at ``z = 0``, so wall effects enter only through the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..convex_contact import ConvexShape, contact, gjk_distance
from .shapes import Ellipsoid, PolyhedralShape, describe

__all__ = [
    "PackingResult",
    "pack_spheres_compaction",
    "pack_deposition",
    "packing_fraction_of_shapes",
    "PHI_SPHERE_RCP",
    "PHI_SPHERE_RLP",
]

PHI_SPHERE_RCP = 0.64   # classic random-close-packing value for hard spheres
PHI_SPHERE_RLP = 0.60   # random loose packing (frictional / deposited)


@dataclass
class PackingResult:
    """Outcome of one packing run."""

    protocol: str
    n_particles: int
    packing_fraction: float
    particle_volume: float
    box_volume: float
    height: float
    n_vertices: int
    sphericity: float
    aspect_ratio: float
    shape_name: str
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {
            "protocol": self.protocol,
            "n_particles": self.n_particles,
            "packing_fraction": self.packing_fraction,
            "particle_volume": self.particle_volume,
            "box_volume": self.box_volume,
            "height": self.height,
            "n_vertices": self.n_vertices,
            "sphericity": self.sphericity,
            "aspect_ratio": self.aspect_ratio,
            "shape_name": self.shape_name,
        }
        out.update({f"extra_{k}": v for k, v in self.extra.items()})
        return out


# ---------------------------------------------------------------------------
# sphere compaction (Lubachevsky--Stillinger)
# ---------------------------------------------------------------------------


def pack_spheres_compaction(n: int = 343, radius0: float = 0.02,
                            growth: float = 0.004, max_growth_steps: int = 4000,
                            relax_steps: int = 120, tol: float = 3.0e-3,
                            kick_rounds: int = 2, kick_amplitude: float = 0.05,
                            rng: Optional[np.random.Generator] = None,
                            box_length: float = 1.0) -> PackingResult:
    """Dense (RCP) packing of monodisperse spheres by isotropic compression."""
    from scipy.spatial import cKDTree

    n = int(n)
    if n < 8:
        raise ValueError("need at least 8 spheres")
    if box_length <= 0.0:
        raise ValueError("box_length must be positive")
    if growth <= 0.0:
        raise ValueError("growth must be positive")
    rng = np.random.default_rng(rng)
    L = float(box_length)
    pos = rng.random((n, 3)) * L
    radius = float(radius0)
    if 4.0 / 3.0 * math.pi * radius ** 3 * n / L ** 3 >= 0.4:
        raise ValueError("initial radius already too large for the box")

    def relax(r: float) -> float:
        """Project out overlaps; return the largest remaining overlap."""
        worst = 0.0
        for _ in range(relax_steps):
            tree = cKDTree(pos, boxsize=L)
            pairs = tree.query_pairs(2.0 * r, output_type="ndarray")
            if pairs.size == 0:
                return 0.0
            ii, jj = pairs[:, 0], pairs[:, 1]
            d = pos[jj] - pos[ii]
            d -= L * np.round(d / L)
            dist = np.linalg.norm(d, axis=1)
            dist = np.where(dist <= 1.0e-15, 1.0e-15, dist)
            overlap = 2.0 * r - dist
            worst = float(np.max(overlap))
            if worst <= 0.0:
                return 0.0
            move = 0.5 * overlap[:, None] * (d / dist[:, None])
            np.add.at(pos, ii, -move)
            np.add.at(pos, jj, move)
            pos[:] = pos % L
        return worst

    relax(radius)
    radius_feasible = radius

    def vibrate(r: float, amplitude: float) -> bool:
        """Small random kicks + relaxation: the standard cure for a jammed,
        under-coordinated state.  Returns True when the kicked state is kept."""
        nonlocal pos
        saved = pos.copy()
        pos[:] = pos + amplitude * r * rng.normal(size=pos.shape)
        pos[:] = pos % L
        if relax(r) <= tol * r:
            return True
        pos = saved
        return False

    step = float(growth)
    accepts = 0
    while accepts < max_growth_steps:
        trial = radius * (1.0 + step)
        worst = relax(trial)
        if worst <= tol * trial:
            radius = radius_feasible = trial
            accepts += 1
            for _ in range(kick_rounds):
                vibrate(radius, kick_amplitude)
            step = min(float(growth), step * 1.25)
            continue
        # infeasible: go back to the last feasible state and retry with a
        # smaller increment; the increment shrink is the jamming criterion
        relax(radius_feasible)
        step *= 0.45
        if step < 1.0e-6:
            break
    phi = n * 4.0 / 3.0 * math.pi * radius_feasible ** 3 / L ** 3
    return PackingResult(
        protocol="compaction",
        n_particles=n,
        packing_fraction=float(phi),
        particle_volume=float(4.0 / 3.0 * math.pi * radius_feasible ** 3),
        box_volume=float(L ** 3),
        height=float(L),
        n_vertices=0,
        sphericity=1.0,
        aspect_ratio=1.0,
        shape_name="sphere",
        extra={"radius": float(radius_feasible), "box_length": L, "growth": growth},
    )


# ---------------------------------------------------------------------------
# deposition helpers
# ---------------------------------------------------------------------------


def _box_sat_overlap(center_a: np.ndarray, rot_a: np.ndarray, half_a: np.ndarray,
                     centers_b: np.ndarray, rots_b: np.ndarray, halves_b: np.ndarray,
                     length: float) -> np.ndarray:
    """Exact 15-axis SAT: one oriented box vs many (periodic in x, y)."""
    m = centers_b.shape[0]
    if m == 0:
        return np.zeros(0, dtype=bool)
    axes: List[np.ndarray] = []
    for k in range(3):
        axes.append(np.broadcast_to(rot_a[:, k], (m, 3)))
        axes.append(np.broadcast_to(rots_b[:, :, k], (m, 3)))
    for i in range(3):
        for j in range(3):
            cross = np.cross(rot_a[:, i][None, :], rots_b[:, :, j])
            norm = np.linalg.norm(cross, axis=1)
            safe = norm > 1.0e-12
            cross = np.where(safe[:, None],
                             cross / np.where(safe, norm, 1.0)[:, None], 0.0)
            axes.append(cross)
    delta = centers_b - center_a[None, :]
    delta[:, 0] -= length * np.round(delta[:, 0] / length)
    delta[:, 1] -= length * np.round(delta[:, 1] / length)
    separated = np.zeros(m, dtype=bool)
    for axis in axes:
        # projection radii: sum_i h_i |(R^T a)_i| for each box
        ra = np.abs(np.einsum("ki,mk->mi", rot_a, axis)) @ half_a
        rb = np.abs(np.einsum("mki,mk->mi", rots_b, axis)) @ halves_b[0]
        d = np.abs(np.einsum("mk,mk->m", delta, axis))
        separated |= d > (ra + rb + 1.0e-12)
    return ~separated


class _Shifted(ConvexShape):
    """Translate a support body without re-hulling it."""

    def __init__(self, base: ConvexShape, offset: np.ndarray):
        self.base = base
        self.offset = np.asarray(offset, dtype=float)

    def support(self, direction):
        return self.base.support(direction) + self.offset

    def centroid(self):
        return self.base.centroid() + self.offset

    def bounding_radius(self):
        return self.base.bounding_radius()


class _Rotated(ConvexShape):
    """Rotate a centred support body without rebuilding its hull."""

    def __init__(self, base: ConvexShape, rot: np.ndarray):
        self.base = base
        self.rot = np.asarray(rot, dtype=float)

    def support(self, direction):
        d = np.asarray(direction, dtype=float)
        return self.rot @ self.base.support(self.rot.T @ d)

    def centroid(self):
        return self.rot @ self.base.centroid()

    def bounding_radius(self):
        return self.base.bounding_radius()


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-15:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _deepest_contact(base: ConvexShape, center: np.ndarray, rot: np.ndarray,
                     placed: Sequence["_Placed"], length: float,
                     half: np.ndarray):
    """Deepest contact of a movable particle against the frozen bed."""
    probe = _Shifted(_Rotated(base, rot), center)
    best = None
    for item in placed:
        delta = center - item.center
        delta[0] -= length * round(delta[0] / length)
        delta[1] -= length * round(delta[1] / length)
        if np.any(np.abs(delta) - (half + item.half) > 0.0):
            continue
        geom = contact(probe, _Shifted(item.shape, item.center), rel_tol=1.0e-12)
        if geom.separated:
            continue
        if best is None or geom.depth > best[0]:
            best = (float(geom.depth), np.asarray(geom.normal, dtype=float),
                    np.asarray(geom.point_a, dtype=float))
    return best


def _settle(base: ConvexShape, center: np.ndarray, rot: np.ndarray,
            placed: Sequence["_Placed"], length: float, half: np.ndarray,
            iters: int = 24, dz: float = 1.0e-3, drot: float = 0.02):
    """Quasi-static settling of one particle on a frozen bed.

    Each iteration: (i) push the particle out of the deepest penetration along
    its normal, (ii) attempt a gravity descent of ``dz`` and keep it if the body
    stays free, (iii) attempt a toppling rotation about the deepest contact
    point (gravity torque, axis ``z_hat x lever``) and keep it when the particle
    can descend lower afterwards.  This is the rolling/settling step that a
    ballistic drop without rotational dynamics lacks.
    """
    center = np.array(center, dtype=float)
    rot = np.array(rot, dtype=float)
    for _ in range(iters):
        hit = _deepest_contact(base, center, rot, placed, length, half)
        if hit is not None:
            depth, normal, point = hit
            center = center - normal * (depth + 1.0e-9)
            lever = center - point
            lever[2] = 0.0
            if float(np.linalg.norm(lever)) > 1.0e-6:
                axis = np.array([-lever[1], lever[0], 0.0])
                trial_rot = _axis_angle(axis, drot) @ rot
                trial_center = center - np.array([0.0, 0.0, dz])
                for _ in range(6):
                    hit2 = _deepest_contact(base, trial_center, trial_rot,
                                            placed, length, half)
                    if hit2 is None:
                        break
                    trial_center = trial_center - hit2[1] * (hit2[0] + 1.0e-9)
                if hit2 is None and trial_center[2] < center[2] - 1.0e-7:
                    center, rot = trial_center, trial_rot
                    continue
        moved = center - np.array([0.0, 0.0, dz])
        if _deepest_contact(base, moved, rot, placed, length, half) is None:
            center = moved
    return center, rot


@dataclass
class _Placed:
    shape: ConvexShape          # body frame, centroid at the origin
    center: np.ndarray
    half: np.ndarray            # AABB half-extents in the body frame axes
    rot: np.ndarray
    half_extents: Optional[np.ndarray] = None
    ext_hi: float = 0.0         # max z extent above the centroid
    ext_lo: float = 0.0         # min z extent relative to the centroid


def _aabb_half(shape: ConvexShape) -> np.ndarray:
    lo, hi = shape.aabb()
    return 0.5 * (hi - lo)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _convex_overlap(probe: ConvexShape, probe_center: np.ndarray,
                    probe_half: np.ndarray, placed: Sequence[_Placed],
                    length: float) -> bool:
    """Boolean overlap with AABB + bounding-sphere pre-filter, then GJK."""
    r_p = probe.bounding_radius()
    for item in placed:
        delta = probe_center - item.center
        delta[0] -= length * round(delta[0] / length)
        delta[1] -= length * round(delta[1] / length)
        gap = np.abs(delta) - (probe_half + item.half)
        if np.any(gap > 0.0):
            continue
        if float(np.linalg.norm(delta)) > r_p + item.shape.bounding_radius():
            continue
        if gjk_distance(_Shifted(probe, probe_center), _Shifted(item.shape, item.center),
                        rel_tol=1.0e-12).overlap:
            return True
    return False


def pack_deposition(shape_factory: Callable[[np.random.Generator], object],
                    n_particles: int = 120, trials: int = 1,
                    rng: Optional[np.random.Generator] = None,
                    length: Optional[float] = None,
                    bisection_steps: int = 28,
                    descriptor=None,
                    settle: bool = True,
                    tap_rounds: int = 4,
                    margin: float = 1.0e-9) -> PackingResult:
    """Loose (RLP) packing by sequential ballistic deposition.

    ``shape_factory(rng)`` must return a convex body with its centroid at the
    origin (``support``/``centroid``/``bounding_radius``; boxes additionally
    need ``half_extents`` and ``orientation`` so that the exact SAT path is
    used).  The random orientation is chosen by the caller.
    """
    n = int(n_particles)
    if n < 8:
        raise ValueError("need at least 8 particles")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    rng = np.random.default_rng(rng)
    ref = shape_factory(np.random.default_rng(12345))
    v_particle = _measure_volume(ref)
    is_box = hasattr(ref, "half_extents")
    if length is None:
        length = float((n * v_particle / 0.30) ** (1.0 / 3.0))
    length = float(length)
    if length <= 0.0:
        raise ValueError("length must be positive")

    placed: List[_Placed] = []
    box_centers: List[np.ndarray] = []
    box_rots: List[np.ndarray] = []
    box_halves: List[np.ndarray] = []
    z_top = 0.0

    for _ in range(n):
        best = None
        for _trial in range(trials):
            # a trial samples both the drop position and the orientation: the
            # lowest admissible rest height is kept (rolling/settling analogue)
            shape = shape_factory(rng)
            rot = np.asarray(getattr(shape, "orientation", np.eye(3)), dtype=float)
            half = _aabb_half(shape)
            ext_hi = float(np.max(_support_of(shape, np.array([0.0, 0.0, 1.0])) -
                                  shape.centroid()))
            ext_lo = float(np.min(_support_of(shape, np.array([0.0, 0.0, -1.0])) -
                                  shape.centroid()))
            floor = -ext_lo  # centroid height at which the body just touches z = 0
            x = float(rng.random()) * length
            y = float(rng.random()) * length

            def overlaps(z: float) -> bool:
                c = np.array([x, y, z])
                if is_box:
                    return bool(np.any(_box_sat_overlap(
                        c, rot, np.asarray(shape.half_extents, dtype=float),
                        np.asarray(box_centers) if box_centers else np.zeros((0, 3)),
                        np.asarray(box_rots) if box_rots else np.zeros((0, 3, 3)),
                        np.asarray(box_halves) if box_halves else np.zeros((0, 3)),
                        length)))
                return _convex_overlap(shape, c, half, placed, length)

            # exponential search downwards from a guaranteed-free height
            z_free = z_top + ext_hi + 1.0e-12
            step = max(2.0 * ext_hi, 1.0e-2)
            z_overlap = None
            for _ in range(80):
                z_try = max(floor, z_free - step)
                if overlaps(z_try):
                    z_overlap = z_try
                    break
                z_free = z_try
                if z_try <= floor:
                    break
                step *= 1.6
            if z_overlap is not None:
                lo, hi = z_overlap, z_free
                for _ in range(bisection_steps):
                    mid = 0.5 * (lo + hi)
                    if overlaps(mid):
                        lo = mid
                    else:
                        hi = mid
                z_rest = float(hi)
            else:
                z_rest = float(floor)
            if settle:
                z_rest, rot = _settle(shape, np.array([x, y, z_rest]), rot,
                                      placed, length, half)
                if isinstance(z_rest, tuple):  # pragma: no cover - defensive
                    z_rest, rot = z_rest
            if best is None or float(np.asarray(z_rest)[2] if np.ndim(z_rest) else z_rest) < best[0]:
                best = (float(np.asarray(z_rest)[2] if np.ndim(z_rest) else z_rest),
                        x, y, shape, rot, half, ext_hi)
        z, x, y, shape, rot, half, ext_hi = best
        center = np.array([x, y, z])
        placed.append(_Placed(shape, center, half, rot,
                              np.asarray(shape.half_extents, dtype=float)
                              if is_box else None,
                              ext_hi=float(ext_hi), ext_lo=float(ext_lo)))
        if is_box:
            box_centers.append(center)
            box_rots.append(rot)
            box_halves.append(np.asarray(shape.half_extents, dtype=float))
        z_top = max(z_top, z + ext_hi)

    if tap_rounds > 0:
        _tap(placed, length, rng, rounds=tap_rounds)
    height = max((float(p.center[2]) + float(p.ext_hi) for p in placed), default=0.0)
    box_volume = length * length * height
    phi = n * v_particle / box_volume
    desc = descriptor
    if desc is None and isinstance(ref, PolyhedralShape):
        desc = describe(ref)
    return PackingResult(
        protocol="deposition",
        n_particles=n,
        packing_fraction=float(phi),
        particle_volume=float(v_particle),
        box_volume=float(box_volume),
        height=float(height),
        n_vertices=int(desc.n_vertices) if desc is not None else 0,
        sphericity=float(desc.sphericity) if desc is not None else float("nan"),
        aspect_ratio=float(desc.aspect_ratio) if desc is not None else float("nan"),
        shape_name=str(getattr(ref, "name", type(ref).__name__)),
        extra={"length": length, "trials": trials,
               "box_kind": "box" if is_box else "convex"},
    )


def _descend(item: "_Placed", candidate: np.ndarray, placed: Sequence["_Placed"],
             skip: int, length: float, step: float) -> float:
    """Lowest overlap-free centroid height for `item` at a lateral candidate."""
    def free(z: float) -> bool:
        probe_center = np.array([candidate[0], candidate[1], z])
        if item.half_extents is not None:
            others = [p for k, p in enumerate(placed) if k != skip]
            if not others:
                return True
            box = _box_sat_overlap(
                probe_center, item.rot, item.half_extents,
                np.asarray([p.center for p in others]),
                np.asarray([p.rot for p in others]),
                np.asarray([p.half_extents for p in others]), length)
            return not bool(np.any(box))
        others = [p for k, p in enumerate(placed) if k != skip]
        return not _convex_overlap(item.shape, probe_center, item.half, others, length)

    z = float(candidate[2])
    floor = -item.ext_lo
    if not free(z):
        return z
    step_size = step
    z_overlap = None
    for _ in range(60):
        z_try = max(floor, z - step_size)
        if not free(z_try):
            z_overlap = z_try
            break
        z = z_try
        if z <= floor:
            break
        step_size *= 1.5
    if z_overlap is None:
        return float(z)
    lo, hi = z_overlap, z
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        if free(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def _tap(placed: List["_Placed"], length: float, rng: np.random.Generator,
         rounds: int = 4, lateral: float = 0.35, attempts: int = 4) -> None:
    """Quasi-static tapping: lateral re-placement followed by gravity descent.

    The ballistic deposit is columnar (a known artefact of deposition without
    lateral dynamics); tapping lets each particle roll into a neighbouring hole,
    which is the standard experimental route from RLP to a denser state.  Both
    the lateral trial and the accepted state are overlap-free, so the protocol
    stays geometric (no contact law is invented).
    """
    if not placed:
        return
    size = float(np.median([2.0 * max(p.half) for p in placed]))
    step = max(1.0e-3, 0.02 * size)
    for _ in range(int(rounds)):
        order = rng.permutation(len(placed))
        for idx in order:
            item = placed[idx]
            best = item.center.copy()
            for _a in range(int(attempts)):
                direction = rng.normal(size=2)
                norm = float(np.linalg.norm(direction))
                if norm < 1.0e-12:
                    continue
                radius = lateral * size * float(rng.random())
                candidate = item.center + np.array([direction[0] / norm * radius,
                                                    direction[1] / norm * radius, 0.0])
                candidate[0] %= length
                candidate[1] %= length
                z = _descend(item, candidate, placed, idx, length, step)
                if z < best[2] - 1.0e-6:
                    best = np.array([candidate[0], candidate[1], z])
            item.center = best


def _support_of(shape, direction: np.ndarray) -> np.ndarray:
    return np.asarray(shape.support(direction), dtype=float)


def _measure_volume(shape) -> float:
    """Particle volume for any of the supported shape objects."""
    volume = getattr(shape, "volume", None)
    if callable(volume):
        return float(volume())
    if volume is not None:
        return float(volume)
    half = getattr(shape, "half_extents", None)
    if half is not None:
        h = np.asarray(half, dtype=float)
        return float(8.0 * h[0] * h[1] * h[2])
    radius = getattr(shape, "radius", None)
    if radius is not None:
        return float(4.0 / 3.0 * math.pi * float(radius) ** 3)
    raise TypeError(f"{type(shape).__name__} has no volume")


def packing_fraction_of_shapes(shapes: Sequence[PolyhedralShape], *,
                               n_particles: int = 120, trials: int = 1,
                               rng: Optional[np.random.Generator] = None,
                               bisection_steps: int = 28) -> List[dict]:
    """Deposition packing fraction for a list of shapes (``phi`` vs ``Psi``, ``N_v``)."""
    rng = np.random.default_rng(rng)
    out: List[dict] = []
    for shape in shapes:
        desc = describe(shape)

        def factory(r: np.random.Generator, _verts=shape.local_vertices,
                    _name=shape.name):
            return PolyhedralShape(_verts, orientation=_random_rotation(r), name=_name)

        res = pack_deposition(factory, n_particles=n_particles, trials=trials,
                              rng=rng, bisection_steps=bisection_steps,
                              descriptor=desc)
        row = res.as_dict()
        row.update({"n_vertices": desc.n_vertices, "sphericity": desc.sphericity,
                    "roundness": desc.roundness, "aspect_ratio": desc.aspect_ratio})
        out.append(row)
    return out

# ---------------------------------------------------------------------------
# orientationally frozen compaction (generic convex shapes)
# ---------------------------------------------------------------------------


def _box_pair_mtd(centers: np.ndarray, rots: np.ndarray, halves: np.ndarray,
                  length: float):
    """Minimum-translation vector for every box pair (vectorised 15-axis SAT).

    Returns ``(i, j, depth, axis)`` for the pairs that overlap.  The SAT
    minimum-overlap axis *is* the minimum translation distance for convex
    polytopes, so pushing along it resolves the overlap exactly.
    """
    n = centers.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    delta = centers[jj] - centers[ii]
    delta[:, 0] -= length * np.round(delta[:, 0] / length)
    delta[:, 1] -= length * np.round(delta[:, 1] / length)
    ra_all = rots[ii]          # (p,3,3)
    rb_all = rots[jj]
    axes = []
    for k in range(3):
        axes.append(np.broadcast_to(ra_all[:, :, k], (ii.size, 3)))
        axes.append(np.broadcast_to(rb_all[:, :, k], (ii.size, 3)))
    for a in range(3):
        for b in range(3):
            cross = np.cross(ra_all[:, :, a], rb_all[:, :, b])
            norm = np.linalg.norm(cross, axis=1)
            safe = norm > 1.0e-12
            axes.append(np.where(safe[:, None],
                                 cross / np.where(safe, norm, 1.0)[:, None], 0.0))
    depth = np.full(ii.size, np.inf)
    best_axis = np.zeros((ii.size, 3))
    separated = np.zeros(ii.size, dtype=bool)
    for axis in axes:
        ra = np.abs(np.einsum("pki,pk->pi", ra_all, axis)) @ halves[0]
        rb = np.abs(np.einsum("pki,pk->pi", rb_all, axis)) @ halves[0]
        d = np.abs(np.einsum("pk,pk->p", delta, axis))
        # parallel edge pairs give a vanishing cross product: such an axis
        # carries no information and must not be read as a separating axis
        valid = np.linalg.norm(axis, axis=1) > 1.0e-12
        overlap = np.where(valid, ra + rb - d, np.inf)
        separated |= overlap <= 0.0
        better = (~separated) & (overlap < depth)
        depth = np.where(better, overlap, depth)
        best_axis = np.where(better[:, None], axis, best_axis)
    keep = (~separated) & np.isfinite(depth)
    axis_out = best_axis[keep]
    # orient every axis from i towards j so that a symmetric push separates
    sign = np.einsum("pk,pk->p", delta[keep], axis_out)
    axis_out = np.where(sign[:, None] < 0.0, -axis_out, axis_out)
    return ii[keep], jj[keep], depth[keep], axis_out


def pack_compaction_generic(bodies: Sequence[ConvexShape], *,
                            length0: float = 1.0,
                            shrink: float = 0.004,
                            max_steps: int = 900,
                            relax_iters: int = 40,
                            tol: float = 2.0e-3,
                            rng: Optional[np.random.Generator] = None,
                            descriptor=None) -> PackingResult:
    """Dense packing by box shrinking with orientationally frozen particles.

    Particles keep the supplied orientations; overlaps are removed by
    translating the pair apart along the exact minimum-translation vector (box
    SAT) or the EPA normal (general convex bodies).  Because the orientations
    are frozen this is the *loose* end for a given shape family: a jammed,
    orientation-disordered state rather than a crystallised one.
    """
    n = len(bodies)
    if n < 8:
        raise ValueError("need at least 8 particles")
    rng = np.random.default_rng(rng)
    volume = sum(_measure_volume(b) for b in bodies)
    centers = rng.random((n, 3)) * length0
    is_box = all(hasattr(b, "half_extents") for b in bodies)
    rots = np.asarray([np.asarray(getattr(b, "orientation", np.eye(3)), dtype=float)
                       for b in bodies])
    halves = np.asarray([np.asarray(b.half_extents, dtype=float) for b in bodies]) \
        if is_box else None
    radius = np.asarray([b.bounding_radius() for b in bodies])
    floors = [-float(np.asarray(b.support([0.0, 0.0, -1.0]), dtype=float)[2]) for b in bodies]

    def max_overlap(length: float) -> float:
        if is_box:
            _, _, depth, _ = _box_pair_mtd(centers, rots, halves, length)
            return float(np.max(depth)) if depth.size else 0.0
        worst = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                d[0] -= length * round(d[0] / length)
                d[1] -= length * round(d[1] / length)
                if float(np.linalg.norm(d)) > radius[i] + radius[j]:
                    continue
                geom = contact(_Shifted(bodies[i], centers[i]),
                               _Shifted(bodies[j], centers[j]), rel_tol=1.0e-11)
                if not geom.separated:
                    worst = max(worst, float(geom.depth))
        return worst

    def relax(length: float, iters: int = 200) -> float:
        """Project out overlaps until the largest overlap is below ``tol``."""
        for _ in range(iters):
            worst = max_overlap(length)
            if worst <= tol:
                return worst
            if is_box:
                ii, jj, depth, axis = _box_pair_mtd(centers, rots, halves, length)
                nrm = axis / np.maximum(np.linalg.norm(axis, axis=1), 1.0e-300)[:, None]
                move = 0.5 * depth[:, None] * nrm
                np.add.at(centers, ii, -move)
                np.add.at(centers, jj, move)
            else:
                for i in range(n):
                    for j in range(i + 1, n):
                        d = centers[i] - centers[j]
                        d[0] -= length * round(d[0] / length)
                        d[1] -= length * round(d[1] / length)
                        if float(np.linalg.norm(d)) > radius[i] + radius[j]:
                            continue
                        geom = contact(_Shifted(bodies[i], centers[i]),
                                       _Shifted(bodies[j], centers[j]),
                                       rel_tol=1.0e-11)
                        if geom.separated:
                            continue
                        move = 0.5 * float(geom.depth) * np.asarray(geom.normal, dtype=float)
                        centers[i] -= move
                        centers[j] += move
            centers[:, 0] %= length
            centers[:, 1] %= length
            centers[:, 2] = np.maximum(centers[:, 2], np.asarray(floors))
        return max_overlap(length)

    length = float(length0)
    relax(length)
    feas = length
    rate = float(shrink)
    for _ in range(max_steps):
        trial = length * (1.0 - rate)
        if n * (volume / n) / trial ** 3 > 1.0:
            break
        worst = relax(trial, iters=relax_iters)
        if worst <= tol:
            length = trial
            feas = trial
            rate = min(float(shrink), rate * 1.35)
            continue
        # the trial state is infeasible: restore the last feasible packing
        relax(length, iters=relax_iters)
        rate *= 0.4
        if rate < 1.0e-5:
            break
    phi = n * (volume / n) / feas ** 3
    desc = descriptor
    return PackingResult(
        protocol="compaction-frozen",
        n_particles=n,
        packing_fraction=float(phi),
        particle_volume=float(volume / n),
        box_volume=float(feas ** 3),
        height=float(feas),
        n_vertices=int(desc.n_vertices) if desc is not None else 0,
        sphericity=float(desc.sphericity) if desc is not None else float("nan"),
        aspect_ratio=float(desc.aspect_ratio) if desc is not None else float("nan"),
        shape_name=str(getattr(bodies[0], "name", type(bodies[0]).__name__)),
        extra={"shrink": shrink, "box_length": feas},
    )


def aligned_spheroid_affine_packing(sphere_result: PackingResult,
                                    aspect: float) -> Dict[str, object]:
    """Packing fraction of a *uniformly oriented* spheroid packing.

    An affine map ``x -> diag(1, aspect, 1) x`` applied to both the particles and
    the periodic container preserves every contact, every gap (scaled
    anisotropically) and the volume fraction exactly.  Hence a packing of
    aligned spheroids with aspect ratio ``aspect`` has the *same* packing
    fraction as the reference sphere packing it was mapped from -- this is an
    identity, not an approximation, and it is the anchor used for the
    ``1:2`` spheroid entry in the validation table.

    The identity is verified numerically by :func:`verify_affine_invariance`.
    """
    aspect = float(aspect)
    if aspect <= 0.0:
        raise ValueError("aspect must be positive")
    return {
        "protocol": "affine-aligned",
        "packing_fraction": float(sphere_result.packing_fraction),
        "source_protocol": sphere_result.protocol,
        "aspect": aspect,
        "n_particles": sphere_result.n_particles,
        "note": "phi is affine-invariant for a uniform orientation field",
    }


def verify_affine_invariance(aspect: float = 2.0) -> Dict[str, float]:
    """Numerically verify that the affine map preserves contacts and ``phi``.

    Contact is a topological statement (the two convex bodies touch / do not
    touch), so the affine image of a *touching* pair must still touch, and a
    separated pair must stay separated.  The packing fraction is invariant
    because both the particle volume and the container volume are multiplied by
    ``det A``; that part is exact by construction and is reported as such.
    """
    from ..convex_contact import Sphere, gjk_distance

    aspect = float(aspect)
    if aspect <= 0.0:
        raise ValueError("aspect must be positive")
    scale = np.array([1.0, aspect, 1.0])
    radius = 0.5
    # pairs at gap 0 (touching), and gaps +-0.05 (sep / overlap)
    results = {}
    for label, offset in (("touching", 2.0 * radius),
                          ("separated", 2.0 * radius + 0.05),
                          ("overlapping", 2.0 * radius - 0.05)):
        a = Sphere(np.zeros(3), radius)
        b = Sphere(np.array([offset, 0.0, 0.0]), radius)
        a2 = Ellipsoid(radius * scale, center=np.zeros(3))
        b2 = Ellipsoid(radius * scale, center=np.array([offset, 0.0, 0.0]) * scale)
        gap = gjk_distance(a, b).distance
        overlap = gjk_distance(a2, b2).overlap
        results[f"{label}_gap"] = float(gap)
        results[f"{label}_mapped_overlap"] = bool(overlap)
    phi_sphere = 4.0 / 3.0 * math.pi * radius ** 3
    phi_spheroid = 4.0 / 3.0 * math.pi * (radius ** 3 * aspect)
    box_sphere = 1.0
    box_spheroid = aspect
    results["phi_sphere"] = phi_sphere / box_sphere
    results["phi_spheroid"] = phi_spheroid / box_spheroid
    results["phi_relative_difference"] = abs(results["phi_sphere"] -
                                             results["phi_spheroid"]) / results["phi_sphere"]
    return results
