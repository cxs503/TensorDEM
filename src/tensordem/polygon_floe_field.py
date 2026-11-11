"""Unordered polygonal floe field: FSD-driven Laguerre tessellation + SAT
contacts with lever arms (IN32-N01).

Wave-1 (:mod:`tensordem.fsd_packing`, node IN22-N01) generated a floe field of
**equal-area disks on an ordered (size-graded) lattice**: it reaches high
coverage, but the topology is a lattice, the floe shape is a circle and a disk
contact has no lever arm.  This module replaces that geometry by the real thing:

1. the floe-size distribution (FSD) sets a target area for every floe;
2. a **Laguerre (power) tessellation** of the periodic ice channel realises those
   areas with convex polygons (:mod:`tensordem.laguerre_field`, prescribed-area
   Newton solve on the exact area Laplacian);
3. every polygon is winched inwards about its own centroid by
   :math:`s = \\sqrt{\\phi}`, which leaves a plateau gap and makes the field
   *strictly* non-overlapping up to :math:`\\phi = 0.90` (the wave-1 disc field
   failed at :math:`\\phi \\ge 0.85`) while keeping the realised floe areas equal
   to the FSD targets (:math:`s^2 \\cdot s^{-2} A_i`);
4. contacts are convex-polygon SAT contacts with a witness pair, hence a contact
   point, a **lever arm** and a **torque** (:mod:`tensordem.polygon_contact`);
   polygon-hull contacts use the same SAT machinery against the hull segments;
5. the field exports a floe table (centroid, local vertices, area, perimeter,
   equivalent radius, mass, polar moment of inertia, coordination) that can be
   fed to :mod:`tensordem.ice_channel` / :mod:`tensordem.floe_6dof`.

Realised metrics reported for every field: area moments ``m0/m1/m2`` of the
equivalent radius against the analytic FSD moments, mean number of polygon
vertices, minimum SAT signed gap, adjacency / coordination number, fabric tensor
and the measured generation time.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import math
import time
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .fsd_packing import DeltaFSD, FloeSizeDistribution, make_fsd
from .laguerre_field import LaguerreCells, LaguerreParams, jittered_sites, laguerre_prescribed_areas
from .polygon_contact import (PolygonContactConfig, candidate_pairs, contact_forces,
                              contact_set, coordination_number, fabric_tensor,
                              polygon_gap, polygon_hull_gap)

__all__ = [
    "PolygonFloeField",
    "sample_floe_radii",
    "generate_polygon_floe_field",
    "generate_from_spec",
]


# ---------------------------------------------------------------------------
# FSD sampling
# ---------------------------------------------------------------------------

def sample_floe_radii(fsd: FloeSizeDistribution, n: int, *, seed: int = 0,
                      shuffle: bool = True) -> np.ndarray:
    """Stratified (deterministic) sample of ``n`` floe radii from the FSD.

    Stratification ``u_k = (k + 1/2)/n`` makes the sample moments converge as
    ``O(n^-2)`` instead of ``O(n^-1/2)``, which is what lets the realised
    ``m1/m2`` match the analytic FSD moments to better than 1 % at a few hundred
    floes.  The *assignment* of radii to sites is then shuffled so the field is
    not size-graded.
    """
    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2")
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    try:
        radii = fsd.quantile(u)
    except Exception as exc:                              # pragma: no cover
        raise ValueError(f"FSD quantile evaluation failed: {exc}") from exc
    radii = np.asarray(getattr(radii, "numpy", lambda: radii)(), dtype=np.float64)
    radii = radii.reshape(n)
    if not np.isfinite(radii).all() or (radii <= 0.0).any():
        raise ValueError("sampled radii must be finite and positive")
    if shuffle:
        idx = np.random.default_rng(int(seed)).permutation(n)
        radii = radii[idx]
    return radii


# ---------------------------------------------------------------------------
# field container
# ---------------------------------------------------------------------------

@dataclass
class PolygonFloeField:
    """A periodic polygonal floe field."""

    box: Tuple[float, float]
    centroid: np.ndarray                 # (N, 2) folded into the box
    verts: np.ndarray                    # (N, cap, 2) local vertices (CCW)
    count: np.ndarray                    # (N,) live vertex count
    area: np.ndarray                     # (N,) polygon area [m^2]
    perimeter: np.ndarray                # (N,) [m]
    r_eq: np.ndarray                     # (N,) equal-area radius [m]
    thickness: np.ndarray                # (N,) [m]
    mass: np.ndarray                     # (N,) [kg]
    inertia: np.ndarray                  # (N,) polar moment about centroid [kg m^2]
    coverage: float
    seed: int
    fsd: Optional[FloeSizeDistribution] = None
    fsd_description: Dict[str, object] = dc_field(default_factory=dict)
    cells_summary: Dict[str, object] = dc_field(default_factory=dict)
    timings: Dict[str, float] = dc_field(default_factory=dict)
    density: float = 917.0
    #: diagnostics of the IN26-N01 compaction solver (None = buffered, contact-free field)
    compaction: Optional[Dict[str, object]] = None

    # -- basic accessors ---------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.centroid.shape[0])

    @property
    def circumradius(self) -> np.ndarray:
        return np.array([np.linalg.norm(self.verts[k, : self.count[k]], axis=1).max()
                         if self.count[k] else 0.0 for k in range(self.n)])

    def polygon(self, index: int) -> np.ndarray:
        """World vertices of floe ``index`` (may lie outside the box on the torus)."""
        i = int(index)
        return self.verts[i, : self.count[i]] + self.centroid[i]

    def total_area(self) -> float:
        return float(self.area.sum())

    def realised_coverage(self) -> float:
        return self.total_area() / (self.box[0] * self.box[1])

    # -- contacts ----------------------------------------------------------
    def contacts(self, cutoff: float = 0.0, *, brute: bool = False
                 ) -> Dict[str, np.ndarray]:
        """SAT contact set (``gap <= cutoff``); ``brute`` for the O(N^2) check."""
        return contact_set(self.centroid, self.verts, self.count, box=self.box,
                           cutoff=float(cutoff), brute=brute)

    def min_signed_gap(self, *, cutoff: Optional[float] = None) -> float:
        """Global minimum SAT signed gap (escalates the search radius safely)."""
        if self.n < 2:
            return float("inf")
        cut = cutoff
        if cut is None:
            cut = 0.25 * float(np.median(self.r_eq))
        # A pair outside the candidate list has disjoint bounding circles
        # separated by more than ``cut``, hence a gap above ``cut``: as soon as
        # the smallest discovered gap is <= cut it is the global minimum.
        for _ in range(8):
            cs = self.contacts(cutoff=cut)
            if cs["gap"].size:
                g = float(cs["gap"].min())
                if g <= cut:
                    return g
            cut *= 2.0
        return float(cs["gap"].min()) if cs["gap"].size else float("inf")

    def coordination_number(self, cutoff: float = 0.0) -> float:
        cs = self.contacts(cutoff=cutoff)
        return coordination_number(self.n, cs["i"], cs["j"])

    def fabric(self, cutoff: float = 0.0) -> Dict[str, float]:
        cs = self.contacts(cutoff=cutoff)
        return fabric_tensor(cs["normal"])

    def adjacency(self, cutoff: float = 0.0) -> np.ndarray:
        """Number of neighbours of every floe."""
        cs = self.contacts(cutoff=cutoff)
        out = np.zeros(self.n, dtype=np.int64)
        np.add.at(out, cs["i"], 1)
        np.add.at(out, cs["j"], 1)
        return out

    def hull_contacts(self, segments) -> List[Dict[str, object]]:
        """Contacts of every floe with a set of hull segments ``(S, 2, 2)``."""
        segs = np.asarray(segments, dtype=np.float64)
        if segs.ndim != 3 or segs.shape[1:] != (2, 2):
            raise ValueError("segments must have shape (S, 2, 2)")
        out = []
        for k in range(self.n):
            best = None
            for s in segs:
                res = polygon_hull_gap(self.centroid[k], self.verts[k, : self.count[k]],
                                      s[0], s[1])
                if best is None or res.gap < best.gap:
                    best = res
            if best is not None:
                out.append({
                    "floe": k, "gap": best.gap,
                    "normal": best.normal, "point": best.point + self.centroid[k],
                    "lever_arm": best.r_a, "arm_length": float(np.linalg.norm(best.r_a)),
                })
        return out

    def contact_forces(self, cutoff: float = 0.0, *,
                       config: Optional[PolygonContactConfig] = None,
                       velocity: Optional[np.ndarray] = None,
                       omega: Optional[np.ndarray] = None,
                       slip: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        cs = self.contacts(cutoff=cutoff)
        return contact_forces(self.centroid, self.mass, cs, config=config,
                              velocity=velocity, omega=omega, slip=slip)

    def centroid_drift(self, *, cutoff: float = 0.0, steps: int = 200,
                       dt: float = 1e-4) -> float:
        """Mass-weighted centroid drift of an isolated (force-free) field [m].

        Velocity Verlet on the contact forces only: with no external load the
        field must not move, which is the torque/force bookkeeping check.
        """
        cs = self.contacts(cutoff=cutoff)
        out = contact_forces(self.centroid, self.mass, cs,
                             config=PolygonContactConfig())
        acc = out["body_force"] / self.mass[:, None]
        vel = np.zeros_like(self.centroid)
        pos = self.centroid.copy()
        m_tot = float(self.mass.sum())
        x0 = (self.mass[:, None] * pos).sum(axis=0) / m_tot
        for _ in range(int(steps)):
            vel += 0.5 * dt * acc
            pos = pos + dt * vel
            acc = acc                       # no external force: acc is constant
            vel += 0.5 * dt * acc
        x1 = (self.mass[:, None] * pos).sum(axis=0) / m_tot
        return float(np.linalg.norm(x1 - x0))

    # -- reported metrics --------------------------------------------------
    def moments(self) -> Dict[str, float]:
        r = self.r_eq
        return {"m0": 1.0, "m1": float(r.mean()),
                "m2": float((r * r).mean()), "n": float(self.n)}

    def moment_errors(self) -> Dict[str, float]:
        realised = self.moments()
        analytic = self.fsd.moments() if self.fsd is not None else realised
        return {k: float(abs(realised[k] - analytic[k]) / analytic[k])
                for k in ("m1", "m2")}

    def metrics(self) -> Dict[str, object]:
        cs = self.contacts(cutoff=0.0)
        near = self.contacts(cutoff=0.25 * float(np.median(self.r_eq)))
        out: Dict[str, object] = {
            "n": self.n,
            "box": [float(self.box[0]), float(self.box[1])],
            "target_coverage": float(self.coverage),
            "realised_coverage": self.realised_coverage(),
            "min_signed_gap": self.min_signed_gap(),
            "contact_pairs": int(cs["i"].size),
            "near_pairs": int(near["i"].size),
            "coordination": coordination_number(self.n, cs["i"], cs["j"]),
            "coordination_near": coordination_number(self.n, near["i"], near["j"]),
            "vertices_mean": float(self.count.mean()),
            "vertices_min": int(self.count.min()),
            "vertices_max": int(self.count.max()),
            "r_eq_mean": float(self.r_eq.mean()),
            "r_eq_min": float(self.r_eq.min()),
            "r_eq_max": float(self.r_eq.max()),
            "area_min": float(self.area.min()),
            "area_max": float(self.area.max()),
            "perimeter_mean": float(self.perimeter.mean()),
            "m1_rel_error": self.moment_errors()["m1"],
            "m2_rel_error": self.moment_errors()["m2"],
            "fabric_anisotropy": fabric_tensor(near["normal"])["anisotropy"],
            "cells": dict(self.cells_summary),
            "timings": dict(self.timings),
        }
        return out

    def floe_table(self) -> Dict[str, np.ndarray]:
        """Flat table for the simulation modules (``ice_channel``/``floe_6dof``)."""
        return {
            "id": np.arange(self.n, dtype=np.int64),
            "x": self.centroid[:, 0].copy(),
            "y": self.centroid[:, 1].copy(),
            "n_vertices": self.count.copy(),
            "vertices": self.verts[:, : self.count.max()].copy(),
            "area": self.area.copy(),
            "perimeter": self.perimeter.copy(),
            "r_eq": self.r_eq.copy(),
            "thickness": self.thickness.copy(),
            "mass": self.mass.copy(),
            "inertia": self.inertia.copy(),
            "box": np.asarray(self.box, dtype=np.float64),
            "coverage": np.asarray([self.coverage], dtype=np.float64),
        }

    def with_vertex_scale(self, factor: float) -> "PolygonFloeField":
        """Copy of the field with every polygon scaled about its centroid.

        ``factor > 1`` inflates the floes, which turns the near-contact shell into
        a real overlapping contact network (used by the force/torque balance
        check); ``factor < 1`` increases the plateau gap.  Areas, radii, masses and
        moments of inertia scale consistently.
        """
        f = float(factor)
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError("factor must be finite and positive")
        verts = self.verts * f
        area = self.area * f * f
        return PolygonFloeField(
            box=self.box, centroid=self.centroid.copy(), verts=verts,
            count=self.count.copy(), area=area, perimeter=self.perimeter * f,
            r_eq=np.sqrt(area / math.pi), thickness=self.thickness.copy(),
            mass=self.mass * f * f, inertia=self.inertia * f ** 4,
            coverage=self.coverage * f * f, seed=self.seed, fsd=self.fsd,
            fsd_description=dict(self.fsd_description),
            cells_summary=dict(self.cells_summary), timings=dict(self.timings),
            density=self.density,
            compaction=(dict(self.compaction) if self.compaction is not None else None))

    def as_disc_equivalent(self) -> Dict[str, np.ndarray]:
        """Equal-area disk field (backward compatibility with the wave-1 API)."""
        return {"pos": self.centroid.copy(), "radius": self.r_eq.copy(),
                "thickness": self.thickness.copy(), "box": np.asarray(self.box)}

    def summary(self) -> Dict[str, object]:
        out: Dict[str, object] = {"fsd": dict(self.fsd_description),
                                  "metrics": self.metrics()}
        if self.compaction is not None:
            out["compaction"] = {k: v for k, v in self.compaction.items()
                                 if k not in ("energy_history", "coordination_history",
                                              "overlap_history")}
        return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _box_for(total_area: float, aspect: float) -> Tuple[float, float]:
    lx = math.sqrt(float(aspect) * float(total_area))
    return lx, float(total_area) / lx


def generate_polygon_floe_field(
    fsd: Optional[FloeSizeDistribution] = None, *,
    coverage: float = 0.85,
    n_floes: int = 400,
    box: Optional[Tuple[float, float]] = None,
    seed: int = 0,
    aspect: float = 1.6,
    jitter: float = 0.6,
    thickness: Optional[float] = None,
    density: float = 917.0,
    radii: Optional[np.ndarray] = None,
    params: Optional[LaguerreParams] = None,
) -> PolygonFloeField:
    """Generate a polygonal floe field at the requested coverage.

    The FSD supplies the target areas ``A_i = pi a_i^2``; the box is sized so that
    ``sum(A_i) = coverage * Lx Ly`` unless it is given explicitly (in which case
    the areas are rescaled, and the realised FSD moments are reported with that
    rescaling folded in).  ``thickness`` defaults to the ice-channel reference
    value (thickness within 10 % of the equivalent radius is *not* required: the
    field is geometrically 2D and the thickness only enters the mass).
    """
    coverage = float(coverage)
    if not (0.0 < coverage < 1.0):
        raise ValueError(f"coverage must lie in (0, 1), got {coverage}")
    n_floes = int(n_floes)
    if n_floes < 4:
        raise ValueError("n_floes must be at least 4")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    if aspect <= 0.0 or not math.isfinite(aspect):
        raise ValueError("aspect must be finite and positive")
    params = params or LaguerreParams()

    t_start = time.perf_counter()
    if radii is None:
        if fsd is None:
            fsd = DeltaFSD(5.0)
        radii = sample_floe_radii(fsd, n_floes, seed=seed)
    else:
        radii = np.asarray(radii, dtype=np.float64)
        if radii.shape != (n_floes,):
            raise ValueError("radii must have shape (n_floes,)")
        if not np.isfinite(radii).all() or (radii <= 0.0).any():
            raise ValueError("radii must be finite and positive")
    area_target = math.pi * radii ** 2

    if box is None:
        total = float(area_target.sum()) / coverage
        lx, ly = _box_for(total, aspect)
    else:
        lx, ly = float(box[0]), float(box[1])
        if not (lx > 0.0 and ly > 0.0):
            raise ValueError("box must be positive")
        total = lx * ly
        area_target = area_target * (coverage * total / area_target.sum())
    t_sampled = time.perf_counter()

    sites = jittered_sites(n_floes, (lx, ly), seed=seed, jitter=jitter)
    cell_targets = area_target / coverage
    cells: LaguerreCells = laguerre_prescribed_areas(sites, cell_targets, (lx, ly),
                                                     params=params)
    t_cells = time.perf_counter()

    s = math.sqrt(coverage)
    verts = cells.verts * s                       # local vertices about the centroid
    count = cells.count.copy()
    poly_area = cells.area * coverage             # = target areas (up to solver tol)
    poly_perimeter = cells.perimeter * s
    r_eq = np.sqrt(np.maximum(poly_area, 0.0) / math.pi)
    deck = (float(thickness) if thickness is not None
            else float(np.median(r_eq)) * 0.1)
    if deck <= 0.0 or not math.isfinite(deck):
        raise ValueError("thickness must be finite and positive")
    thick = np.full(n_floes, deck, dtype=np.float64)
    mass = density * poly_area * thick
    # polar moment of inertia of a polygon about its centroid (thin plate):
    # I = rho t / 12 * sum |cross(v_k, v_k+1)| (|v_k|^2 + v_k.v_k+1 + |v_k+1|^2)
    inertia = np.zeros(n_floes)
    for k in range(n_floes):
        v = verts[k, : count[k]]
        if count[k] < 3:
            continue
        vn = np.roll(v, -1, axis=0)
        cr = np.abs(v[:, 0] * vn[:, 1] - v[:, 1] * vn[:, 0])
        term = (v * v).sum(axis=1) + (vn * vn).sum(axis=1) + (v * vn).sum(axis=1)
        inertia[k] = density * thick[k] / 12.0 * float((cr * term).sum())
    t_done = time.perf_counter()

    field = PolygonFloeField(
        box=(lx, ly), centroid=cells.centroid.copy(), verts=verts, count=count,
        area=poly_area, perimeter=poly_perimeter, r_eq=r_eq, thickness=thick,
        mass=mass, inertia=inertia, coverage=coverage, seed=int(seed), fsd=fsd,
        fsd_description=(fsd.describe() if fsd is not None else {"kind": "radii"}),
        cells_summary=cells.summary(), density=float(density),
        timings={"sample_s": t_sampled - t_start,
                 "tessellation_s": t_cells - t_sampled,
                 "assembly_s": t_done - t_cells,
                 "total_s": t_done - t_start},
    )
    return field


def generate_from_spec(spec: Mapping[str, object], **kwargs) -> PolygonFloeField:
    """Build a field from a plain-data spec ``{"fsd": {...}, "coverage": ...}``."""
    cfg = dict(spec)
    fsd_spec = cfg.pop("fsd", None)
    fsd = make_fsd(fsd_spec) if fsd_spec is not None else DeltaFSD(5.0)
    merged = {**cfg, **kwargs}
    return generate_polygon_floe_field(fsd, **merged)