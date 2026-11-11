"""DEM <-> PBM bridge: DEM-measured breakage rates and a particle-resolved
(Lagrangian) realisation of the same statistics.

Why this module exists
----------------------
The DEM side of the coupling (:mod:`tensordem.breakage_pbm.assembly_compression`)
is a contact-mechanics simulation: it returns, apart from the strength and
daughter statistics, an *event log* (time, parent volume, daughter volumes) and
PSD snapshots.  Those events are the ground truth that a population balance has
to be able to represent.  This module

1. turns the DEM event log into **section- and interval-resolved breakage
   rates** ``S_j`` (breakage events per particle in section ``j`` per second),
   which is the only input the continuum PBM takes from the DEM besides the
   daughter law (the micro-macro closure ``kappa`` is *not* used here, so the
   PSD comparison is not circular in the breakage rate -- the rate is measured,
   not modelled);
2. provides a **particle-resolved (Lagrangian, DEM-style) realisation** of the
   same rates and the same daughter law: every particle is tracked
   individually, its breakage is an exponential clock with rate ``S_j``, and on
   breakage it is replaced by daughter particles drawn from the DEM-calibrated
   daughter law.  This is the discrete-particle reference the sectional
   (Eulerian) population balance is cross-validated against;
3. provides the **first Wasserstein distance** of the volume-weighted diameter
   distributions, normalised by the initial mean diameter, which is the C4
   metric.

Both the Lagrangian and the Eulerian solvers consume exactly the same
``S_j(t)`` and the same daughter operator, so the residual difference is the
population-balance discretisation error (sectional projection + fixed-pivot
daughter placement + ODE integration), which is what the C4 metric has to
quantify.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dpbm import (
    VolumeGrid,
    breakage_rhs,
    equivalent_diameter,
    particle_volume,
    solve_breakage,
    wasserstein_from_cdf,
)

__all__ = [
    "RateProfile",
    "measure_section_rates",
    "section_of",
    "LagrangianBreakage",
    "psd_wasserstein",
    "cdf_from_psd",
]


def section_of(grid: VolumeGrid, volumes) -> np.ndarray:
    """Nearest-pivot section index of each particle volume."""
    v = np.asarray(volumes, dtype=np.float64).ravel()
    pivots = grid.pivots
    idx = np.searchsorted(pivots, v)
    idx = np.clip(idx, 1, pivots.size - 1)
    lower = pivots[idx - 1]
    upper = pivots[idx]
    choose_lower = np.abs(v - lower) <= np.abs(v - upper)
    return np.where(choose_lower, idx - 1, idx)


@dataclass(frozen=True)
class RateProfile:
    """Piecewise-constant (in time) sectional breakage-rate history."""

    edges: np.ndarray
    rates: np.ndarray          # shape (n_intervals, n_sections) [1/s]

    def __post_init__(self) -> None:
        e = np.asarray(self.edges, dtype=np.float64)
        r = np.asarray(self.rates, dtype=np.float64)
        if e.ndim != 1 or e.size < 2:
            raise ValueError("edges must be a 1-D array with >= 2 entries")
        if r.shape[0] != e.size - 1:
            raise ValueError("rates must have one row per time interval")

    def __call__(self, t: float) -> np.ndarray:
        t = float(t)
        idx = int(np.searchsorted(np.asarray(self.edges), t, side="right")) - 1
        if idx < 0:
            return np.zeros(self.rates.shape[1])
        if idx >= self.rates.shape[0]:
            return np.zeros(self.rates.shape[1])
        return self.rates[idx]

    @property
    def max_rate(self) -> float:
        return float(np.max(self.rates)) if self.rates.size else 0.0


def measure_section_rates(grid: VolumeGrid, events, snapshots, t_edges,
                          *, primary_only: bool = True) -> RateProfile:
    """DEM event log -> sectional breakage rate history ``S_j``.

    ``events`` is the DEM breakage log (``t``, ``parent_volume``,
    ``daughter_volumes``, optional ``generation``), ``snapshots`` maps ``t`` to
    the particle volumes at that time (the DEM PSD records), ``t_edges`` are the
    interval boundaries of the piecewise-constant rate.

    ``S_j`` is "breakage events per *breakable* particle in section ``j`` per
    second".  With ``primary_only`` (the default, matching
    :func:`tensordem.breakage_pbm.dpbm.solve_primary_breakage` and a DEM run with
    ``max_generations = 1``) only generation-0 events are counted and the
    denominator is the surviving mother population of section ``j``, which is
    obtained exactly from the initial population minus the generation-0 events
    that already happened -- no interpolation, no snapshot guesswork.
    """
    edges = np.asarray(sorted(float(t) for t in t_edges), dtype=np.float64)
    n_sec = int(grid.n_sections)
    if not events:
        return RateProfile(edges=edges, rates=np.zeros((edges.size - 1, n_sec)))
    times = np.asarray([float(e["t"]) for e in events], dtype=np.float64)
    parent_idx = section_of(grid, [float(e["parent_volume"]) for e in events])
    if primary_only:
        gen = np.asarray([int(e.get("generation", 0)) for e in events], dtype=np.int64)
        sel = gen == 0
        times, parent_idx = times[sel], parent_idx[sel]
    snap_t = np.array(sorted(float(t) for t in snapshots), dtype=np.float64)
    initial_vol = np.asarray(snapshots[snap_t[0]], dtype=np.float64)
    initial_counts = np.bincount(section_of(grid, initial_vol),
                                 minlength=n_sec).astype(np.float64)
    rates = np.zeros((edges.size - 1, n_sec))
    for k in range(edges.size - 1):
        lo, hi = float(edges[k]), float(edges[k + 1])
        width = hi - lo
        if width <= 0.0:
            continue
        held = np.bincount(parent_idx[times < lo], minlength=n_sec).astype(np.float64)
        alive = np.maximum(initial_counts - held, 0.0)
        event_counts = np.bincount(parent_idx[(times >= lo) & (times < hi)],
                                   minlength=n_sec).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(alive > 0.0, event_counts / np.maximum(alive, 1e-12) / width, 0.0)
        rates[k] = np.where(np.isfinite(s), s, 0.0)
    return RateProfile(edges=edges, rates=rates)


def cdf_from_psd(volumes, weights, points):
    """CDF of the *volume-weighted* diameter distribution on ``points``."""
    v = np.asarray(volumes, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if np.sum(w) <= 0.0:
        raise ValueError("PSD carries no mass")
    d = equivalent_diameter(v)
    order = np.argsort(d)
    d, w = d[order], w[order]
    cw = np.cumsum(w) / np.sum(w)
    return np.interp(np.asarray(points, dtype=np.float64), d, cw,
                     left=0.0, right=1.0)


def psd_wasserstein(volumes_a, weights_a, volumes_b, weights_b, norm: float,
                    *, n_pts: int = 4000) -> dict:
    """First Wasserstein distance of two PSDs, normalised by ``norm``."""
    d_all = np.concatenate([equivalent_diameter(volumes_a), equivalent_diameter(volumes_b)])
    lo, hi = float(np.min(d_all)), float(np.max(d_all))
    pts = np.linspace(lo, hi, int(n_pts))
    fa = cdf_from_psd(volumes_a, weights_a, pts)
    fb = cdf_from_psd(volumes_b, weights_b, pts)
    raw = wasserstein_from_cdf(pts, fa, fb)
    return {"wasserstein_m": float(raw), "normalised": float(raw) / float(norm),
            "points": pts, "cdf_a": fa, "cdf_b": fb}


class LagrangianBreakage:
    """Particle-resolved (DEM-style) realisation of a sectional breakage rate.

    Every particle is tracked individually.  Within one interval of the piecewise
    constant rate history, every particle in section ``j`` breaks with
    probability ``1 - exp(-S_j dt_sub)`` per sub-step (the exact Bernoulli
    survival probability of a Poisson clock), and is replaced by daughters drawn
    from the daughter rule with the parent's volume exactly conserved.  The
    sub-step is chosen so that ``max_j S_j dt_sub <= 0.2``.
    """

    def __init__(self, grid: VolumeGrid, rates: RateProfile, daughter, *,
                 seed: int = 0, max_substep: float = 0.2, max_generations: int = 1):
        self.grid = grid
        self.rates = rates
        self.daughter = daughter
        self.rng = np.random.default_rng(seed)
        self.max_substep = float(max_substep)
        self.max_generations = int(max_generations)

    def run(self, initial_volumes, t_eval):
        t_eval = sorted(float(t) for t in t_eval)
        volumes = np.asarray(initial_volumes, dtype=np.float64).ravel().copy()
        generation = np.zeros(volumes.size, dtype=np.int64)
        records = {}
        t = 0.0
        edges = np.asarray(self.rates.edges, dtype=np.float64)
        for target in t_eval:
            while t < target - 1e-12:
                s = np.asarray(self.rates(t), dtype=np.float64)
                s_max = float(np.max(s)) if s.size else 0.0
                later = edges[edges > t + 1e-15]
                stop = min(target, float(later.min())) if later.size else target
                if s_max > 0.0:
                    dt = min(stop - t, self.max_substep / s_max)
                else:
                    dt = stop - t
                dt = max(dt, 1e-12)
                if s_max > 0.0:
                    sec = section_of(self.grid, volumes)
                    p = 1.0 - np.exp(-s[sec] * dt)
                    hit = (self.rng.random(volumes.size) < p) & \
                        (generation < self.max_generations)
                    if np.any(hit):
                        keep_v = volumes[~hit]
                        keep_g = generation[~hit]
                        new_v, new_g = [], []
                        for v in volumes[hit]:
                            daughters = np.asarray(self.daughter.sample(float(v), self.rng),
                                                   dtype=np.float64)
                            daughters = daughters * (float(v) / float(np.sum(daughters)))
                            new_v.append(daughters)
                            new_g.append(np.full(daughters.size,
                                                 int(self.generation_of(generation, volumes, v)) + 1))
                        volumes = np.concatenate([keep_v] + new_v)
                        generation = np.concatenate([keep_g] + new_g)
                t += dt
                if stop - t <= 1e-12:
                    t = stop
            records[target] = volumes.copy()
        return records

    @staticmethod
    def generation_of(generation, volumes, v):
        """Generation of the particle that is about to break (largest match)."""
        idx = int(np.argmin(np.abs(volumes - v)))
        return generation[idx]
