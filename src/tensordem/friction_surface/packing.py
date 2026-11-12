"""Random close packing of monodisperse spheres by Lubachevsky-Stillinger inflation.

Purpose of this module in the friction-contact verification surface
-----------------------------------------------------------------
Packing fraction is the third anchor: an *independent* route to the textbook
random-close-packing value ``phi_RCP = 0.64`` (Scott & Kilgour 1969).  The route
shipped here is the dynamic one — spheres start diluted in a periodic cubic cell
and are inflated at a finite rate while overdamped dynamics removes overlaps;
the packing jams when the free volume is exhausted.

It is deliberately *not* the geometric route already in
:func:`tensordem.bedcoupling.packing.make_random_close_packing` (quasi-static
compression + local projection).  Two independent routes agreeing on 0.64 is a
stronger statement than either alone, and
:func:`tensordem.friction_surface.study.run_packing_anchor` reports the
cross-route difference.

Physics / numerics
------------------
* Periodic cubic cell, minimum-image distances, monodisperse spheres.
* Relaxation: a **capped steepest-descent quench** (Jacobi under-relaxation of the
  overlap-derived displacements, ``under_relax``), *not* the overdamped Langevin
  dynamics of the original Lubachevsky-Stillinger prescription.  Because the quench
  is inertia-free, a finite inflation rate cannot pump kinetic energy into the
  jammed state; in exchange the measured jamming point acquires a dependence on
  the relaxation budget (``relax_steps``) and on the inflation rate
  (``rho_growth``).  Both dependencies are *measured*, not assumed away:
  ``benchmarks/rcp_ls_extrapolation/`` scans them and extrapolates to the
  ``rate -> 0`` limit.  (An earlier revision of this docstring described the
  dynamics of the original L-S paper; the implementation has always been the
  quench, so the text was corrected rather than the code.)
* Contact: normal repulsion only (frictionless inflation) — the RCP value is a
  frictionless property.  A frictional variant is **not implemented**: a quench
  without a time history cannot carry a Cundall-Strack tangential displacement, so
  ``mu != 0`` is rejected instead of being silently ignored (it used to be echoed
  into the result dict while doing nothing).
* Frictionless contacts are *not* the same as the DEM kernel's tangential law;
  the normal force uses the same :func:`tensordem.friction_surface.laws.normal_force`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .laws import normal_force

__all__ = ["PackingCase", "random_close_packing", "solid_fraction"]


@dataclass(frozen=True)
class PackingCase:
    """Parameters of one L-S inflation run (dimensionless lengths, unit box)."""

    n_particles: int = 1000
    phi_start: float = 0.30
    rho_growth: float = 2.0e-4        # relative radius growth per inflation step
    relax_steps: int = 60             # capped-descent iterations per inflation step
    max_step_relax: float = 0.25      # max displacement per relaxation step, in radii
    under_relax: float = 0.5          # Jacobi under-relaxation factor
    max_inflation_steps: int = 6000
    overlap_stop: float = 1.0e-3      # mean overlap / diameter at which we stop
    kn: float = 1.0
    mu: float = 0.0                   # tangential/rolling friction (0 -> frictionless)
    seed: int = 20261110
    cell: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.n_particles, (int, np.integer)) or self.n_particles < 2:
            raise ValueError("n_particles must be an integer >= 2")
        if not (self.rho_growth > 0.0):
            raise ValueError("rho_growth must be positive (it is the per-step radius growth)")
        if not (self.relax_steps >= 1):
            raise ValueError("relax_steps must be >= 1")
        if not (self.overlap_stop > 0.0):
            raise ValueError("overlap_stop must be positive")
        if not (self.cell > 0.0):
            raise ValueError("cell must be positive")
        if self.mu != 0.0:
            raise NotImplementedError(
                "frictional inflation is not implemented: the capped-descent quench has no "
                "tangential displacement history, so a Coulomb cone cannot be evaluated. "
                "Use a time-stepping engine (tensordem.dem3d / SphereGas) for mu > 0 instead "
                "of relying on the (previously silent) `mu` argument."
            )

    @property
    def rate_per_step(self) -> float:
        """``d ln(phi) / d step`` injected by this inflation rate (phi ~ r^3)."""
        return 3.0 * math.log1p(self.rho_growth)

    @property
    def volume(self) -> float:
        return self.cell ** 3


def _min_image(dvec: np.ndarray, cell: float) -> np.ndarray:
    return dvec - cell * np.round(dvec / cell)


def solid_fraction(radius: float, n_particles: int, cell: float = 1.0) -> float:
    return n_particles * 4.0 / 3.0 * math.pi * radius ** 3 / cell ** 3


def _relax(pos, radius, cell, kn, max_step_relax, n_iter, under_relax=0.5):
    """Capped steepest-descent overlap removal; returns (pos, overlap_ratio, pressure)."""
    n = len(pos)
    overlap_ratio = 0.0
    pressure = 0.0
    for _ in range(n_iter):
        tree = cKDTree(pos, boxsize=cell)
        pairs = tree.query_pairs(2.0 * radius, output_type="ndarray")
        if pairs.size == 0:
            return pos, 0.0, 0.0
        iu = np.minimum(pairs[:, 0], pairs[:, 1])
        ju = np.maximum(pairs[:, 0], pairs[:, 1])
        dvec = _min_image(pos[ju] - pos[iu], cell)
        dist = np.linalg.norm(dvec, axis=1)
        overlap = 2.0 * radius - dist
        active = overlap > 0.0
        if not active.any():
            return pos, 0.0, 0.0
        iu, ju = iu[active], ju[active]
        dvec, dist, overlap = dvec[active], dist[active], overlap[active]
        nvec = dvec / np.maximum(dist, 1e-300)[:, None]
        f_n = normal_force(overlap, np.zeros_like(overlap), kn, 0.0)
        disp = f_n[:, None] * nvec / kn          # ~ overlap displacement
        norm = np.linalg.norm(disp, axis=1)
        cap = max_step_relax * radius
        scale = np.minimum(1.0, cap / np.maximum(norm, 1e-300))
        disp *= (scale * under_relax)[:, None]
        delta = np.zeros((n, 3))
        np.add.at(delta, iu, -disp)
        np.add.at(delta, ju, disp)
        pos = (pos + delta) % cell
        overlap_ratio = float(np.mean(overlap) / (2.0 * radius))
        pressure = float(np.sum(f_n * overlap) / (3.0 * cell ** 3))
    return pos, overlap_ratio, pressure


def random_close_packing(case: PackingCase, *, return_positions: bool = False) -> dict:
    """Inflate a diluted sphere gas until it jams; return the jamming statistics.

    Jamming is declared when the capped descent can no longer remove the overlaps
    within ``relax_steps`` iterations (``mean overlap / diameter`` exceeds
    ``overlap_stop``).  ``jammed_packing_fraction`` is linearly interpolated to
    ``overlap_stop`` between the last two inflation steps, i.e. the hard-sphere
    limit of the inflation curve.
    """
    rng = np.random.default_rng(case.seed)
    n, cell = case.n_particles, case.cell
    radius = (3.0 * case.phi_start * cell ** 3 / (4.0 * math.pi * n)) ** (1.0 / 3.0)
    pos = rng.random((n, 3)) * cell
    history = []
    jam_phi = None
    overlap_ratio = 0.0
    pressure = 0.0
    steps_used = 0
    prev = None
    for step in range(case.max_inflation_steps):
        radius *= 1.0 + case.rho_growth
        pos, overlap_ratio, pressure = _relax(pos, radius, cell, case.kn,
                                              case.max_step_relax, case.relax_steps,
                                              case.under_relax)
        phi = solid_fraction(radius, n, cell)
        history.append((phi, overlap_ratio, pressure))
        steps_used = step + 1
        if overlap_ratio > case.overlap_stop:
            if prev is not None:
                # linear interpolation of phi at the target overlap ratio
                phi0, ov0 = prev
                if overlap_ratio > ov0:
                    w = (case.overlap_stop - ov0) / (overlap_ratio - ov0)
                    jam_phi = phi0 + max(min(w, 1.0), 0.0) * (phi - phi0)
                else:
                    jam_phi = phi
            else:
                jam_phi = phi
            break
        prev = (phi, overlap_ratio)
    out = {
        "n_particles": n,
        "cell": cell,
        "radius": radius,
        "packing_fraction": solid_fraction(radius, n, cell),
        "jammed_packing_fraction": jam_phi,
        "mean_overlap_ratio": overlap_ratio,
        "pressure": pressure,
        "inflation_steps": steps_used,
        "overlap_stop": case.overlap_stop,
        "mu": case.mu,
        "relax_steps": case.relax_steps,
        "rho_growth": case.rho_growth,
        "rate_per_step": case.rate_per_step,
        "phi_start": solid_fraction(
            (3.0 * case.phi_start * cell ** 3 / (4.0 * math.pi * n)) ** (1.0 / 3.0), n, cell),
        "phi_max_reached": (history[-1][0] if history else None),
        "jammed_by_criterion": jam_phi is not None,
        "history": history,
        "protocol": ("Lubachevsky-Stillinger-style inflation + capped steepest-descent "
                     "quench, periodic cell, frictionless normal contacts, "
                     "jamming = mean overlap/diameter crossing overlap_stop"),
    }
    if return_positions:
        out["positions"] = pos
    return out
