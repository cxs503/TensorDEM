"""Crack-path prediction with the maximum tangential stress (MTS) criterion.

For a mixed-mode tip the maximum hoop stress occurs at ``theta_c`` given by

    K_I sin(theta_c) + K_II (3 cos(theta_c) - 1) = 0

whose root in ``(-pi, pi)`` is

    theta_c = 2 atan2( K_I - sqrt(K_I^2 + 8 K_II^2), 4 K_II ).

Limits: pure mode I -> 0 rad; pure mode II (K_I = 0, K_II > 0) -> -70.53 deg,
the classical Erdogan & Sih (1963) kink angle.

:func:`trace_crack_path` grows a crack in the lattice: at every step K_I and K_II
come from the interaction integral, ``theta_c`` from MTS, and the bonds crossing
the segment from the current tip to the new tip are deleted (fracture).  Bond
deletion is irreversible, so the path is a genuine propagation history rather
than a post-processing of one elastic field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .jintegral import stress_intensity_from_m
from .lattice2d import BondedLattice2D

__all__ = ["mts_angle", "max_hoop_stress", "trace_crack_path", "path_overlap",
           "polyline_distance", "hoop_stress_ring", "mts_angle_from_field", "propagate"]


def max_hoop_stress(K_I: float, K_II: float, theta: np.ndarray | None = None,
                    n: int = 3601) -> np.ndarray:
    """``sigma_theta_theta * sqrt(2 pi r)`` as a function of theta."""
    if theta is None:
        theta = np.linspace(-np.pi, np.pi, n)
    theta = np.asarray(theta, dtype=float)
    return (np.cos(theta / 2.0) *
            (K_I * np.cos(theta / 2.0) ** 2 - 1.5 * K_II * np.sin(theta)))


def mts_angle(K_I: float, K_II: float) -> float:
    """Crack kink angle (rad) of the maximum tangential stress criterion."""
    K_I = float(K_I)
    K_II = float(K_II)
    root = np.hypot(K_I, 2.0 * np.sqrt(2.0) * K_II)
    if abs(K_II) < 1e-30 and K_I >= 0.0:
        return 0.0
    theta = 2.0 * np.arctan2(K_I - root, 4.0 * K_II)
    # fold into (-pi, pi]: the arctan2 branch jumps by 2 pi as K_II -> 0
    theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
    return float(theta)


def polyline_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from each point to the polyline ``reference``."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    if reference.shape[0] == 1:
        return np.hypot(*(points - reference[0]).T)
    seg_a = reference[:-1]
    seg_b = reference[1:]
    d = seg_b - seg_a
    L2 = np.einsum("ij,ij->i", d, d)
    L2 = np.where(L2 < 1e-30, 1e-30, L2)
    out = np.full(points.shape[0], np.inf)
    for k in range(seg_a.shape[0]):
        t = np.clip(((points - seg_a[k]) @ d[k]) / L2[k], 0.0, 1.0)
        proj = seg_a[k] + t[:, None] * d[k]
        out = np.minimum(out, np.hypot(*(points - proj).T))
    return out


def path_overlap(path: np.ndarray, reference: np.ndarray, tol: float) -> float:
    """Fraction of ``path`` vertices lying within ``tol`` of the reference path."""
    path = np.atleast_2d(np.asarray(path, dtype=float))
    dist = polyline_distance(path, reference)
    return float(np.mean(dist <= tol))


def _segment_hits_box(p0, p1, q0, q1) -> bool:
    """True if the 2-D segments ``p0-p1`` and ``q0-q1`` intersect."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1, o2 = orient(p0, p1, q0), orient(p0, p1, q1)
    o3, o4 = orient(q0, q1, p0), orient(q0, q1, p1)
    return (o1 * o2 <= 0.0) and (o3 * o4 <= 0.0)


def bonds_crossing(lat: BondedLattice2D, p0: Sequence[float], p1: Sequence[float],
                   active_only: bool = True) -> np.ndarray:
    """Indices of bonds whose segment crosses ``p0 -> p1``."""
    idx = np.flatnonzero(lat.active) if active_only else np.arange(lat.nbonds)
    n0, n1 = lat.bonds[idx, 0], lat.bonds[idx, 1]
    a0 = np.stack([lat.x[n0], lat.y[n0]], axis=1)
    a1 = np.stack([lat.x[n1], lat.y[n1]], axis=1)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    # bounding-box pre-filter: the orientation test alone reports collinear but
    # disjoint bonds as hits, which would delete bonds away from the crack front
    lo = np.minimum(p0, p1) - 1e-12
    hi = np.maximum(p0, p1) + 1e-12
    ok = (np.minimum(a0, a1) <= hi).all(axis=1) & (np.maximum(a0, a1) >= lo).all(axis=1)
    candidates = np.flatnonzero(ok)
    hits = []
    for k in candidates:
        if _segment_hits_box(p0, p1, a0[k], a1[k]):
            hits.append(idx[k])
    return np.asarray(hits, dtype=int)


@dataclass
class CrackPathResult:
    path: np.ndarray
    angles: list[float] = field(default_factory=list)
    k_history: list[tuple[float, float]] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)

    @property
    def kink_angles_deg(self) -> np.ndarray:
        return np.degrees(np.asarray(self.angles))


def trace_crack_path(lat: BondedLattice2D, tip: Sequence[float], steps: int,
                     r_in: float | None = None, r_out: float | None = None,
                     step_length: float | None = None) -> CrackPathResult:
    """Propagate a crack with MTS + M-integral on an already solved lattice."""
    a = lat.cfg.spacing
    r_in = 2.0 * a if r_in is None else float(r_in)
    r_out = 4.5 * a if r_out is None else float(r_out)
    step = a if step_length is None else float(step_length)
    tip = np.asarray(tip, dtype=float).copy()
    result = CrackPathResult(path=np.array([tip.copy()]))
    for _ in range(int(steps)):
        k1 = stress_intensity_from_m(lat, tip, "I", r_in, r_out)
        k2 = stress_intensity_from_m(lat, tip, "II", r_in, r_out)
        theta = mts_angle(k1, k2)
        new_tip = tip + step * np.array([np.cos(theta), np.sin(theta)])
        hits = bonds_crossing(lat, tip, new_tip)
        if hits.size == 0:
            break
        lat.kill_bonds(hits)
        result.removed.extend(int(h) for h in hits)
        result.angles.append(theta)
        result.k_history.append((k1, k2))
        tip = new_tip
        result.path = np.vstack([result.path, tip])
        lat.solve()
    return result

def hoop_stress_ring(lat: BondedLattice2D, tip: Sequence[float], radius: float,
                     n_theta: int = 181, n_steps: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Hoop stress ``sigma_theta_theta * sqrt(2 pi r)`` on a ring around the tip.

    The virial stress of the lattice is averaged radially over
    ``radius +/- (n_steps-1)/2`` cells before projecting on the hoop direction,
    which removes the bond-scale oscillation without touching the angular
    dependence that carries the mode mixity.
    """
    a = lat.cfg.spacing
    sig = lat.node_stress()
    xt, yt = float(tip[0]), float(tip[1])
    dx = lat.x - xt
    dy = lat.y - yt
    r = np.hypot(dx, dy)
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    out = np.zeros(n_theta)
    for k, th in enumerate(theta):
        best = 0.0
        for dr in range(-(n_steps // 2), n_steps // 2 + 1):
            rr = radius + dr * a
            if rr <= 0.5 * a:
                continue
            px, py = xt + rr * np.cos(th), yt + rr * np.sin(th)
            d2 = (lat.x - px) ** 2 + (lat.y - py) ** 2
            n = int(np.argmin(d2))
            if d2[n] > (1.2 * a) ** 2:
                continue
            s = sig[n]
            # hoop = n . sigma . n with n = (-sin, cos), plus radial->hoop is
            # the crack-tip convention theta measured from the crack line
            nx, ny = -np.sin(th), np.cos(th)
            hoop = nx * nx * s[0, 0] + ny * ny * s[1, 1] + 2.0 * nx * ny * s[0, 1]
            best += hoop
        out[k] = best / (n_steps)
    return theta, out * np.sqrt(2.0 * np.pi * radius)


def mts_angle_from_field(lat: BondedLattice2D, tip: Sequence[float],
                         radius: float | None = None) -> float:
    """Maximum-hoop-stress kink angle measured directly on the lattice field."""
    a = lat.cfg.spacing
    r = 1.6 * a if radius is None else float(radius)
    theta, hoop = hoop_stress_ring(lat, tip, r)
    # restrict to the forward sector to avoid the traction-free faces
    keep = np.abs(theta) < 2.0
    idx = np.flatnonzero(keep)[np.argmax(hoop[keep])]
    return float(theta[idx])


def propagate(lat: BondedLattice2D, tip: Sequence[float], steps: int,
              step_length: float | None = None,
              radius: float | None = None) -> CrackPathResult:
    """Grow a crack using the field-measured MTS angle (no K extraction)."""
    a = lat.cfg.spacing
    step = a if step_length is None else float(step_length)
    tip = np.asarray(tip, dtype=float).copy()
    result = CrackPathResult(path=np.array([tip.copy()]))
    for _ in range(int(steps)):
        theta = mts_angle_from_field(lat, tip, radius)
        new_tip = tip + step * np.array([np.cos(theta), np.sin(theta)])
        hits = bonds_crossing(lat, tip, new_tip)
        if hits.size == 0:
            break
        lat.kill_bonds(hits)
        result.removed.extend(int(h) for h in hits)
        result.angles.append(theta)
        tip = new_tip
        result.path = np.vstack([result.path, tip])
        lat.solve()
    return result
