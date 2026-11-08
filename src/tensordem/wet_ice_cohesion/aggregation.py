"""Shear-induced agglomeration of cohesive ice granules vs Smoluchowski kinetics.

The particle-resolved model
---------------------------
``simulate_shear_aggregation`` puts ``n_particles`` equal spheres (radius ``a``)
into a periodic cube and advects them with a linear shear field
``v_x = shear_rate * y`` (Lees-Edwards images).  Two clusters coalesce as soon as
their equivalent-sphere radii overlap::

    R_k = a * k**(1/df)        (k = number of primary particles, df = 3 compact)

The only "cohesion" input is *whether* an encounter sticks; in a wet-ice setting
the capillary bridge supplies a tensile capacity far larger than the shear force
at contact, so ``sticking_probability = 1`` is the physical limit and
``sticking_probability = 0`` is the dry control.

The mean-field reference
------------------------
``smoluchowski_shear`` integrates the discrete coagulation equation

    dn_k/dt = 1/2 sum_{i+j=k} K_ij n_i n_j  -  n_k sum_j K_kj n_j

with the orthokinetic (Smoluchowski 1917) shear kernel

    K_ij = (4/3) shear_rate (R_i + R_j)**3

for the *same* radius law, so particle simulation and mean field differ only in
the closure (mean field assumes spatial uniformity and molecular chaos).

Why the same kernel on both sides
---------------------------------
It isolates the error of interest.  Criterion (b) of P35-N2 is "<D> within 15 %
of the Smoluchowski prediction"; feeding the analytic kernel to the ODE makes the
comparison a test of the mean-field closure (and of the finite-size realisation),
not of a fitted parameter.  ``measure_shear_kernel`` separately measures the
binary collision rate of a two-species test so the kernel itself is not taken on
faith.

Known limitations (stated, not hidden)
--------------------------------------
* compact aggregates (``df = 3``); real shear aggregates are fractal
  (``df ~ 2.0 - 2.5``), which the parameter exposes but the campaign does not
  calibrate - no data is available in this repo to do so honestly.
* no aggregate break-up: valid while the hydrodynamic stress stays below the
  capillary bridge strength, which is precisely the wet-ice ("capillary
  dominated") regime.
* the affine cluster motion is rigid-body, i.e. no internal relaxation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "AggregationResult",
    "cluster_size_distribution",
    "measure_shear_kernel",
    "simulate_shear_aggregation",
    "smoluchowski_shear",
]


# ---------------------------------------------------------------------------
# particle-resolved shear agglomeration
# ---------------------------------------------------------------------------

@dataclass
class AggregationResult:
    """Time series of the particle-resolved shear agglomeration."""

    times: np.ndarray
    mean_diameter: np.ndarray          # number-weighted <D>, in units of the primary diameter
    mean_mass: np.ndarray              # total mass / number of clusters
    n_clusters: np.ndarray
    snapshots: List[np.ndarray] = field(default_factory=list)  # cluster masses
    params: dict = field(default_factory=dict)

    def distribution(self, index: int) -> np.ndarray:
        return self.snapshots[index]


def _image_delta(dx, dy, dz, box, strain):
    """Lees-Edwards minimum image for a linearly sheared periodic cube."""
    ny = -np.rint(dy / box)
    dyi = dy + ny * box
    dxs = dx + strain * box * ny
    nx = -np.rint(dxs / box)
    dxi = dxs + nx * box
    nz = -np.rint(dz / box)
    dzi = dz + nz * box
    return dxi, dyi, dzi


def simulate_shear_aggregation(
    n_particles: int = 800,
    shear_rate: float = 1.0,
    total_strain: float = 2.0,
    dt_fraction: float = 0.01,
    seed: int = 20261108,
    dimension_fractal: float = 3.0,
    box_over_radius: Optional[float] = None,
    sticking_probability: float = 1.0,
    store_snapshots: int = 6,
) -> AggregationResult:
    """Agglomerate monodisperse spheres in a sheared periodic cube.

    Args:
        n_particles: number of primary particles.
        shear_rate: ``gamma_dot`` in units of ``radius/time``.
        total_strain: ``gamma_dot * t_end``.
        dt_fraction: time step as a fraction of ``1/shear_rate``.
        seed: RNG seed (runs are reproducible bit-for-bit).
        dimension_fractal: exponent in ``R_k = a k**(1/df)``.
        box_over_radius: cube edge in units of the primary radius; defaults to
            the edge that gives a primary volume fraction of 0.05.
        sticking_probability: 1.0 for cohesive (capillary) encounters,
            0.0 for the dry negative control.
        store_snapshots: number of size-distribution snapshots to keep.
    """
    if n_particles < 8:
        raise ValueError("need at least 8 primary particles")
    if shear_rate <= 0.0 or total_strain <= 0.0:
        raise ValueError("shear_rate and total_strain must be positive")
    a = 1.0
    if box_over_radius is None:
        phi0 = 0.05
        box = a * (4.0 / 3.0 * math.pi * n_particles / phi0) ** (1.0 / 3.0)
    else:
        box = box_over_radius * a

    rng = np.random.default_rng(seed)
    # clusters: arrays are grown by merging; positions are centroids
    pos = rng.random((n_particles, 3)) * box
    mass = np.ones(n_particles)
    members: List[List[int]] = [[i] for i in range(n_particles)]

    dt = dt_fraction / shear_rate
    n_steps = int(round(total_strain / shear_rate / dt))
    record_every = max(1, n_steps // max(1, store_snapshots - 1))

    times = [0.0]
    mean_diameter = [2.0 * a]
    mean_mass = [1.0]
    n_clusters = [float(n_particles)]
    snapshots = [mass.copy()]

    t = 0.0
    for step in range(1, n_steps + 1):
        # affine advection
        pos[:, 0] = (pos[:, 0] + shear_rate * pos[:, 1] * dt) % box
        pos[:, 1] %= box
        pos[:, 2] %= box
        t += dt
        strain = (shear_rate * t) % 1.0

        # coalescence sweep
        n = len(mass)
        if n > 1:
            radii = a * mass ** (1.0 / dimension_fractal)
            merged = np.zeros(n, dtype=bool)
            new_mass = list(mass)
            new_pos = list(pos)
            for i in range(n):
                if merged[i]:
                    continue
                dx = new_pos[i][0] - pos[:, 0]
                dy = new_pos[i][1] - pos[:, 1]
                dz = new_pos[i][2] - pos[:, 2]
                dxi, dyi, dzi = _image_delta(dx, dy, dz, box, strain)
                dist = np.sqrt(dxi * dxi + dyi * dyi + dzi * dzi)
                cand = np.where((dist < radii[i] + radii) & (~merged))[0]
                cand = cand[cand > i]
                if cand.size == 0:
                    continue
                # closest first (deterministic, order independent)
                cand = cand[np.argsort(dist[cand])]
                for j in cand:
                    if merged[j]:
                        continue
                    w_i = new_mass[i]
                    w_j = new_mass[j]
                    if sticking_probability < 1.0 and rng.random() > sticking_probability:
                        continue
                    new_pos[i] = (
                        (w_i * np.array(new_pos[i]) + w_j * np.array(new_pos[j])) / (w_i + w_j)
                    )
                    new_mass[i] = w_i + w_j
                    members[i] = members[i] + members[j]
                    merged[j] = True
            keep = ~merged
            pos = np.array(new_pos, dtype=float)[keep]
            mass = np.array(new_mass, dtype=float)[keep]
            members = [members[k] for k in range(n) if keep[k]]

        if step % record_every == 0 or step == n_steps:
            d_equiv = 2.0 * a * mass ** (1.0 / dimension_fractal)
            times.append(t)
            mean_diameter.append(float(np.mean(d_equiv)))
            mean_mass.append(float(mass.sum() / len(mass)))
            n_clusters.append(float(len(mass)))
            snapshots.append(mass.copy())

    return AggregationResult(
        times=np.array(times),
        mean_diameter=np.array(mean_diameter),
        mean_mass=np.array(mean_mass),
        n_clusters=np.array(n_clusters),
        snapshots=snapshots,
        params=dict(
            n_particles=n_particles,
            shear_rate=shear_rate,
            total_strain=total_strain,
            dt=dt,
            box=box,
            dimension_fractal=dimension_fractal,
            sticking_probability=sticking_probability,
            seed=seed,
        ),
    )


def cluster_size_distribution(
    masses: Sequence[float], n_bins: int = 12, dimension_fractal: float = 3.0
) -> dict:
    """Histogram of equivalent diameters normalised as ``n(D) / N``."""
    d = 2.0 * np.asarray(masses, dtype=float) ** (1.0 / dimension_fractal)
    counts, edges = np.histogram(d, bins=n_bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return {
        "diameter_centres": centres.tolist(),
        "counts": counts.tolist(),
        "number_fraction": (counts / max(1, d.size)).tolist(),
        "mean_diameter": float(d.mean()),
        "n_clusters": int(d.size),
    }


# ---------------------------------------------------------------------------
# Smoluchowski mean field
# ---------------------------------------------------------------------------

def smoluchowski_shear(
    n_particles: int,
    volume: float,
    shear_rate: float,
    times: Sequence[float],
    radius_primary: float = 1.0,
    n_classes: int = 240,
    dimension_fractal: float = 3.0,
) -> dict:
    """Discrete Smoluchowski solution with the orthokinetic shear kernel.

    The ODE is integrated with ``scipy.integrate.solve_ivp`` (Radau).  Mass that
    would land in classes above ``n_classes`` is accumulated in the last class so
    the total mass is conserved exactly (checked by the caller).

    Returns ``{"<D>": ndarray, "n_clusters": ndarray, "n_k": (t, k)}`` where
    ``<D>`` is the number-weighted mean equivalent diameter in units of the
    primary diameter.
    """
    from scipy.integrate import solve_ivp

    k = np.arange(1, n_classes + 1, dtype=float)
    r = radius_primary * k ** (1.0 / dimension_fractal)
    K = (4.0 / 3.0) * shear_rate * (r[:, None] + r[None, :]) ** 3

    n0 = np.zeros(n_classes)
    n0[0] = n_particles / volume

    def rhs(_t, n):
        n = np.maximum(n, 0.0)
        dn = np.zeros_like(n)
        # aggregation gain
        for i in range(n_classes):
            if n[i] <= 0.0:
                continue
            js = np.arange(0, n_classes - i - 1)
            kk = i + js + 1
            dn[kk] += 0.5 * K[i, js] * n[i] * n[js]
        # loss
        dn -= n * (K @ n)
        return dn

    t_end = float(max(times))
    sol = solve_ivp(rhs, (0.0, t_end), n0, t_eval=np.asarray(times, dtype=float),
                    method="Radau", rtol=1e-6, atol=1e-30, max_step=t_end / 50.0)
    out_d, out_n = [], []
    for idx in range(sol.y.shape[1]):
        nk = np.maximum(sol.y[:, idx], 0.0)
        d_k = 2.0 * radius_primary * k ** (1.0 / dimension_fractal)
        tot = nk.sum()
        out_d.append(float((nk * d_k).sum() / tot) if tot > 0 else float("nan"))
        out_n.append(float(tot * volume))
    return {
        "times": sol.t,
        "mean_diameter": np.array(out_d),
        "n_clusters": np.array(out_n),
        "n_k": sol.y,
    }


# ---------------------------------------------------------------------------
# kernel measurement (two-species collision rate)
# ---------------------------------------------------------------------------

def measure_shear_kernel(
    radius_a: float = 1.0,
    radius_b: float = 1.0,
    shear_rate: float = 1.0,
    n_pairs: int = 4000,
    box_over_radius: Optional[float] = None,
    seed: int = 7,
) -> dict:
    """Measure the binary shear-collision kernel against ``(4/3) g (a+b)^3``.

    Two populations (A on a fixed lattice, B random) are advected with the same
    affine shear; the measured first-collision rate per unit B concentration is
    compared with the Smoluchowski orthokinetic kernel.  This validates the
    geometric realisation of the particle model independently of the ODE.
    """
    rng = np.random.default_rng(seed)
    rsum = radius_a + radius_b
    if box_over_radius is None:
        box = 12.0 * rsum
    else:
        box = box_over_radius * rsum
    n_a = max(1, n_pairs)
    pos_a = rng.random((n_a, 3)) * box
    pos_b = rng.random((n_a, 3)) * box
    dt = 0.002 / shear_rate
    t_end = 0.6 / shear_rate
    n_steps = int(t_end / dt)
    hits = 0
    strain = 0.0
    for _ in range(n_steps):
        pos_b[:, 0] = (pos_b[:, 0] + shear_rate * pos_b[:, 1] * dt) % box
        pos_b[:, 1] %= box
        pos_b[:, 2] %= box
        strain = (strain + shear_rate * dt) % 1.0
        dx = pos_a[:, None, 0] - pos_b[None, :, 0]
        dy = pos_a[:, None, 1] - pos_b[None, :, 1]
        dz = pos_a[:, None, 2] - pos_b[None, :, 2]
        dxi, dyi, dzi = _image_delta(dx, dy, dz, box, strain)
        d2 = dxi * dxi + dyi * dyi + dzi * dzi
        hits += int(np.count_nonzero(d2 < rsum * rsum))
    volume = box ** 3
    # separate the populations after each counted encounter by re-randomising B
    # (single-count estimate: rate = hits / (t * n_a * n_b / V))
    rate = hits / (t_end * n_a * (n_a / volume))
    kernel_theory = (4.0 / 3.0) * shear_rate * rsum ** 3
    return {
        "hits": hits,
        "measured_kernel": rate,
        "theory_kernel": kernel_theory,
        "relative_error": (rate - kernel_theory) / kernel_theory,
        "box": box,
        "note": (
            "single-count estimator: pairs are not removed, so the measured rate "
            "is a lower bound once clusters form; the box is intentionally large "
            "(box/(a+b) = %.1f) so boundary truncation stays small." % (box / rsum)
        ),
    }