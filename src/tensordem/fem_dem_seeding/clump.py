"""Shape-resolved floe seeding: one floe -> a clump of inscribed union-free discs.

The problem this module solves
------------------------------
The first handoff (:mod:`tensordem.fem_dem_fracture.handoff`) emits one disc per
floe of *equivalent* area, ``D = sqrt(4A/pi)``, parked at the floe centroid.  The
mass and area books close exactly, but the body itself is a poor picture of the
floe: a non-convex or elongated fragment gets replaced by a disc that can reach
far outside its own floe (and therefore outside the plate, and into a neighbour).
That is fine for a mass ledger and wrong for a contact model, which is the point
of a handoff.

Here each floe is filled with a *clump* of discs that satisfy three properties by
construction:

1. **Insolid** -- every disc is inscribed in the floe polygon, so no part of the
   body lies outside the material it represents (``spill_area`` is then zero up
   to the raster error and, crucially, no clump crosses the plate rim).
2. **Union-free** -- disc centres of one clump are held at least ``2 r`` apart and
   two floes are separated by a fracture face, so every disc centre is at least
   ``r`` from that face on both sides; the clump is therefore non-overlapping both
   internally and against its neighbours (audited exactly in
   :func:`overlap_audit`, not on the raster).
3. **Mass preserving** -- the floe mass is split over its discs, so
   ``sum(m_disc) = m_floe`` to round-off for any number of discs, independently of
   how well the discs tile the polygon.

Property 3 is why coverage does not have to be forced to one: mass is carried per
disc, geometry is a separate, *measured* score (``coverage``, ``iou``).  The two
are reported separately instead of being conflated into "the clump equals the
floe", which it does not: a disc packing covers about 0.8 of the area at the
densities used here.  The disc set is the body's material; the polygon is the
exact footprint shipped alongside for shape-resolved contact models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..fem_dem_fracture.crack_network import polygon_area
from .raster import Raster, footprint_metrics, polygon_mask

__all__ = [
    "Disc",
    "FloeBody",
    "seed_inscribed_discs",
    "seed_equivalent_disc",
    "seed_floe_bodies",
    "overlap_audit",
    "body_ledger",
]

#: A clump needs at least this many discs; the campaign's floes are metres across
#: so below three discs the "shape resolved" claim is not credible.
MIN_DISCS = 3
#: Floor on the clump radius [m]; below this the JSON body becomes noise.
MIN_RADIUS = 0.05
#: Default clump resolution [m]: 0.4 of the FEM element (0.25 m).  A finer clump
#: than the continuum solution resolves adds cost without shape information, a
#: coarser one loses the fragments that are only one element wide.
RADIUS_HINT = 0.10
#: Target filling of the footprint by the disc union (triangular packing caps at 0.907).
FILL = 0.8


@dataclass(frozen=True)
class Disc:
    """One member disc of a clump (plan view, ``z`` = mid-plane)."""

    x: float
    y: float
    radius: float
    mass: float

    @property
    def area(self) -> float:
        return math.pi * self.radius * self.radius


@dataclass(frozen=True)
class FloeBody:
    """A DEM body built from one FEM floe."""

    floe_id: int
    mode: str                      # "clump" | "equivalent_disc"
    area: float                    # floe (material) area from the artefact [m^2]
    mass: float                    # floe mass [kg]
    centroid: tuple[float, float]
    thickness: float
    polygons: tuple                # exact footprint: (outer, *holes)
    discs: tuple[Disc, ...]
    metrics: dict = field(default_factory=dict)

    @property
    def n_discs(self) -> int:
        return len(self.discs)

    @property
    def disc_mass_sum(self) -> float:
        return float(sum(d.mass for d in self.discs))

    @property
    def disc_area(self) -> float:
        return float(sum(d.area for d in self.discs))

    @property
    def radius_mean(self) -> float:
        return float(np.mean([d.radius for d in self.discs])) if self.discs else 0.0

    def centres(self) -> np.ndarray:
        return np.array([[d.x, d.y] for d in self.discs], dtype=float).reshape(-1, 2)

    def radii(self) -> np.ndarray:
        return np.array([d.radius for d in self.discs], dtype=float)

    def as_payload(self) -> dict:
        return {
            "floe_id": self.floe_id,
            "mode": self.mode,
            "area_m2": self.area,
            "mass_kg": self.mass,
            "centroid_xy_m": list(self.centroid),
            "thickness_m": self.thickness,
            "n_discs": self.n_discs,
            "disc_radius_mean_m": self.radius_mean,
            "disc_area_m2": self.disc_area,
            "disc_mass_sum_kg": self.disc_mass_sum,
            "discs": [
                {"x_m": d.x, "y_m": d.y, "radius_m": d.radius, "mass_kg": d.mass}
                for d in self.discs
            ],
            "footprint_polygons": [[list(p) for p in ring] for ring in self.polygons],
            "footprint_metrics": self.metrics,
        }


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------
def _inner_distance(raster: Raster, rings) -> np.ndarray:
    """Distance [m] from each cell centre to the nearest outside cell."""
    mask = polygon_mask(raster, rings)
    if not mask.any():
        raise ValueError("polygon does not intersect the raster window")
    # ``distance_transform_edt`` measures to the nearest zero cell; padding the
    # border with zeros keeps the outside honest for floes touching the rim.
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dist = distance_transform_edt(padded)[1:-1, 1:-1] * raster.h
    return dist


def seed_inscribed_discs(raster: Raster, rings, *, target_discs: int) \
                            -> tuple[np.ndarray, np.ndarray]:
    """Inscribed, mutually non-overlapping discs filling a floe footprint.

    Returns ``(centres, radii)``.  The placement is a **triangular lattice** (the
    densest packing of equal discs, filling fraction ``pi / (2 sqrt 3) = 0.907``)
    clipped to the floe, with the lattice constant solved for so that the number of
    discs lands on ``target_discs``:

    1.  the lattice spacing is ``2 r`` (discs exactly touching, i.e. disjoint),
        rows are ``sqrt(3) r`` apart with alternate rows shifted by ``r``;
    2.  a lattice point is kept when the floe cell containing it is inside the
        polygon *and* at least ``r`` from the boundary (raster EDT, conservative);
    3.  each kept disc finally gets ``min(r, exact distance to its boundary)``, so
        the inscribed guarantee holds per disc; the radius only shrinks, hence the
        clump stays disjoint and cannot cross a fracture face;
    4.  ``r`` is picked from a geometric scan of ``count(r)`` (monotone decreasing)
        so that the count is as close to ``target_discs`` as the shape allows --
        the count is a property of the fragment, not a promise of the caller.

    ``max_scan`` caps the number of candidate radii evaluated (deterministic).
    """
    if target_discs < MIN_DISCS:
        raise ValueError(f"target_discs must be >= {MIN_DISCS}")
    mask = polygon_mask(raster, rings)
    if not mask.any():
        raise ValueError("polygon does not intersect the raster window")
    inner = _inner_distance(raster, rings)
    hull = np.asarray(rings[0], dtype=float)
    lo = hull.min(axis=0)
    hi = hull.max(axis=0)

    def points(r: float) -> tuple[np.ndarray, np.ndarray]:
        dy = math.sqrt(3.0) * r
        dx = 2.0 * r
        ky = np.arange(int(math.ceil((hi[1] - lo[1]) / dy)) + 2)
        ys = lo[1] + ky * dy
        cand = []
        for k, y in zip(ky, ys):
            shift = (k % 2) * r
            kx = np.arange(int(math.ceil((hi[0] - lo[0]) / dx)) + 2)
            xs = lo[0] + shift + kx * dx
            cand.append(np.column_stack([xs, np.full(xs.size, y)]))
        pts = np.concatenate(cand, axis=0)
        ii = np.clip(((pts[:, 0] - raster.x0) / raster.h).astype(int), 0, raster.nx - 1)
        jj = np.clip(((pts[:, 1] - raster.y0) / raster.h).astype(int), 0, raster.ny - 1)
        keep = mask[jj, ii] & (inner[jj, ii] >= r)
        return pts[keep], inner[jj[keep], ii[keep]]

    # Starting lattice constant: triangular cell area per disc is (sqrt3/2)(2r)^2.
    r0 = math.sqrt(max(hull_area(rings), 1e-12) / (2.0 * math.sqrt(3.0) * target_discs))
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_score = float("inf")

    def consider(r: float) -> int:
        """Evaluate one lattice constant, keeping the closest count so far."""
        nonlocal best, best_score
        if r < MIN_RADIUS:
            return 0
        pts, _ = points(r)
        n = pts.shape[0]
        if n < MIN_DISCS:
            return n
        radii = np.minimum(r, _boundary_distances(pts, rings))
        score = abs(n - target_discs)
        if score < best_score and np.all(radii > MIN_RADIUS * 0.5):
            best, best_score = (pts, radii), score
        return n

    # ``count(r)`` is monotone decreasing in ``r``, so a bisection on the lattice
    # constant hits the requested count as closely as the shape allows.  (A
    # geometric scan steps 25 % in r, i.e. ~56 % in count -- far too coarse.)
    a = MIN_RADIUS * 1.5        # far too many discs
    b = 3.0 * r0                # far too few discs
    n_a, n_b = consider(a), consider(b)
    if n_b < target_discs <= n_a:
        for _ in range(24):
            mid = math.sqrt(a * b)
            n = consider(mid)
            if n > target_discs:
                a = mid
            else:
                b = mid
    else:
        consider(r0)
    if best is None:
        raise RuntimeError(
            f"could not place {MIN_DISCS} inscribed discs in a floe of area "
            f"{hull_area(rings):.3f} m^2; the fragment is too thin for a clump"
        )
    return best


def hull_area(rings) -> float:
    """Shoelace area of the outer ring minus the holes [m^2]."""
    outer = abs(polygon_area(np.asarray(rings[0], dtype=float)))
    holes = sum(abs(polygon_area(np.asarray(h, dtype=float))) for h in list(rings)[1:])
    return float(outer - holes)


def seed_floe_bodies(faces, raster: Raster, *, target_discs=None,
                     modes=("clump", "equivalent_disc")) -> dict[str, list[FloeBody]]:
    """Seed every floe of the handoff artefact, one body set per mode.

    ``target_discs`` is a callable ``area -> int`` (default: the count implied by
    :data:`RADIUS_HINT` at the triangular packing density, floor
    :data:`MIN_DISCS`).  The ``equivalent_disc`` mode reproduces
    the first-handoff body exactly, so the report can quote a before/after pair
    measured with the same instrument.
    """
    if target_discs is None:
        def target_discs(area: float) -> int:      # noqa: F811 - documented default
            # Count implied by the clump resolution at the triangular packing
            # density: each disc owns (sqrt3/2)(2r)^2 of the footprint.
            n = round(area / (2.0 * math.sqrt(3.0) * RADIUS_HINT ** 2))
            return int(max(MIN_DISCS, min(4000, n)))

    thickness = float(faces.meta["plate"]["thickness_m"])
    out: dict[str, list[FloeBody]] = {m: [] for m in modes}
    for floe in faces.floes:
        rings = tuple(tuple(tuple(float(c) for c in v) for v in ring)
                      for ring in floe.polygons)
        n_t = target_discs(floe.area)
        if "clump" in modes:
            centres, radii = seed_inscribed_discs(raster, rings, target_discs=n_t)
            discs = _split_mass(centres, radii, floe.mass)
            metrics = footprint_metrics(raster, rings, centres, radii)
            out["clump"].append(FloeBody(
                floe_id=floe.floe_id, mode="clump", area=floe.area, mass=floe.mass,
                centroid=(float(floe.centroid[0]), float(floe.centroid[1])),
                thickness=thickness, polygons=rings, discs=tuple(discs),
                metrics=metrics,
            ))
        if "equivalent_disc" in modes:
            out["equivalent_disc"].append(
                seed_equivalent_disc(floe, raster, thickness=thickness))

    for mode in out:
        out[mode].sort(key=lambda b: b.floe_id)
    return out


def seed_equivalent_disc(floe, raster: Raster, *, thickness: float) -> FloeBody:
    """The first-handoff body: one disc of equivalent area at the centroid.

    Reproduced here so that the shape audit compares the two representations with
    one instrument.  Its analytic area equals the floe area exactly, which is why
    it passes every mass/area ledger and still misrepresents an elongated or
    non-convex fragment.
    """
    rings = tuple(tuple(tuple(float(c) for c in v) for v in ring)
                  for ring in floe.polygons)
    r_eq = math.sqrt(floe.area / math.pi)
    centre = np.array([[float(floe.centroid[0]), float(floe.centroid[1])]])
    discs = _split_mass(centre, np.array([r_eq]), floe.mass)
    metrics = footprint_metrics(raster, rings, centre, np.array([r_eq]))
    return FloeBody(
        floe_id=floe.floe_id, mode="equivalent_disc", area=floe.area, mass=floe.mass,
        centroid=(float(floe.centroid[0]), float(floe.centroid[1])),
        thickness=thickness, polygons=rings, discs=tuple(discs), metrics=metrics,
    )


def _boundary_distances(centres: np.ndarray, rings) -> np.ndarray:
    """Exact distance [m] from each centre to the polygon boundary.

    Computed against the segments themselves (clamped point-to-segment distance,
    vectorised), not on the raster, so the inscribed guarantee does not inherit the
    raster resolution.  Every segment of the handoff polygons is axis aligned, but
    the formula is general.
    """
    a = []
    b = []
    for ring in rings:
        pts = np.asarray(ring, dtype=float)
        if pts.shape[0] < 2:
            continue
        a.append(pts)
        b.append(np.roll(pts, -1, axis=0))
    a = np.concatenate(a, axis=0)
    b = np.concatenate(b, axis=0)
    ab = b - a
    seg_len2 = np.einsum("ij,ij->i", ab, ab)
    seg_len2 = np.where(seg_len2 > 0.0, seg_len2, 1e-300)
    d = centres[:, None, :] - a[None, :, :]                       # (n, m, 2)
    t = np.clip(np.einsum("nmk,mk->nm", d, ab) / seg_len2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    dist = np.hypot(centres[:, None, 0] - closest[:, :, 0],
                    centres[:, None, 1] - closest[:, :, 1])
    return dist.min(axis=1)


def _split_mass(centres: np.ndarray, radii: np.ndarray, mass: float) -> list[Disc]:
    """Split ``mass`` over the discs in proportion to their area.

    ``sum(m_disc) = mass`` holds by construction (the shares are normalised), so
    the clump total is independent of how well the discs cover the polygon: a
    partial footprint costs shape fidelity, never mass.
    """
    radii = np.asarray(radii, dtype=float).reshape(-1)
    areas = math.pi * radii * radii
    total = float(np.sum(areas))
    if total <= 0.0:
        raise ValueError("degenerate clump: total disc area is zero")
    return [Disc(x=float(p[0]), y=float(p[1]), radius=float(r),
                 mass=float(mass * a / total))
            for p, r, a in zip(centres, radii, areas)]


# --------------------------------------------------------------------------
# audits
# --------------------------------------------------------------------------
def overlap_audit(bodies) -> dict:
    """Exact pairwise overlap audit between bodies of *different* floes.

    Two discs of different floes are compared with ``d - (r_i + r_j)``; a negative
    value is a physical interpenetration at ``t = 0`` and would bias the first
    contact of the DEM run.  Discs of the same clump are rigidly attached, so their
    mutual overlap is a modelling choice, not an error, and is excluded.

    Pairs come from a ``cKDTree`` radius query, so the audit stays linear in the
    number of discs even when a clump carries thousands of them.
    """
    from scipy.spatial import cKDTree

    pts: list[np.ndarray] = []
    rad: list[np.ndarray] = []
    tag: list[np.ndarray] = []
    for body in bodies:
        pts.append(body.centres())
        rad.append(body.radii())
        tag.append(np.full(body.n_discs, body.floe_id))
    P = np.concatenate(pts, axis=0)
    R = np.concatenate(rad, axis=0)
    T = np.concatenate(tag, axis=0)
    if P.shape[0] < 2:
        return {"n_discs": int(P.shape[0]), "n_cross_pairs": 0,
                "n_overlapping_pairs": 0, "min_clearance_m": float("inf"),
                "n_same_body_pairs": 0}
    tree = cKDTree(P)
    pairs = tree.query_pairs(2.0 * float(np.max(R)), output_type="ndarray")
    if pairs.size == 0:
        return {"n_discs": int(P.shape[0]), "n_cross_pairs": 0,
                "n_overlapping_pairs": 0, "min_clearance_m": float("inf"),
                "n_same_body_pairs": 0}
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1])
    clearance = d - (R[i] + R[j])
    cross = T[i] != T[j]
    c = clearance[cross]
    return {
        "n_discs": int(P.shape[0]),
        "n_cross_pairs": int(c.size),
        "n_same_body_pairs": int(np.count_nonzero(~cross)),
        "n_overlapping_pairs": int(np.count_nonzero(c < -1e-9)),
        "min_clearance_m": float(np.min(c)) if c.size else float("inf"),
        "min_clearance_all_m": float(np.min(clearance)),
    }


def body_ledger(bodies, *, plate_area: float, plate_mass: float) -> dict:
    """Mass / area / disc ledger of a body set, with residual against the plate."""
    area = float(sum(b.area for b in bodies))
    mass = float(sum(b.mass for b in bodies))
    disc_mass = float(sum(b.disc_mass_sum for b in bodies))
    disc_area = float(sum(b.disc_area for b in bodies))
    covered = float(sum(b.metrics.get("covered_area", 0.0) for b in bodies))
    spill = float(sum(b.metrics.get("spill_area", 0.0) for b in bodies))
    return {
        "mode": bodies[0].mode if bodies else None,
        "n_bodies": len(bodies),
        "n_discs": int(sum(b.n_discs for b in bodies)),
        "plate_area_m2": plate_area,
        "body_area_sum_m2": area,
        "area_residual_m2": float(area - plate_area),
        "plate_mass_kg": plate_mass,
        "body_mass_sum_kg": mass,
        "mass_residual_kg": float(mass - plate_mass),
        "disc_mass_sum_kg": disc_mass,
        "disc_mass_residual_kg": float(disc_mass - plate_mass),
        "disc_area_sum_m2": disc_area,
        "covered_area_sum_m2": covered,
        "spill_area_sum_m2": spill,
        "mean_coverage": float(np.mean([b.metrics.get("coverage", 0.0) for b in bodies]))
        if bodies else 0.0,
        "mean_iou": float(np.mean([b.metrics.get("iou", 0.0) for b in bodies]))
        if bodies else 0.0,
    }