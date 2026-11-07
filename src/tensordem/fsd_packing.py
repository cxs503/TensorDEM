"""Multi-disperse floe-field packing generator (IN22-N01).

The IN11-H1 channel-transit module :mod:`tensordem.ice_channel` places a
**monodisperse** floe field on a jittered hexagonal lattice: every floe radius is
the circle of equal area to one lattice cell.  Real fragmented ice covers are
*multi-disperse* -- the floe-size distribution (FSD) spans decades -- and the
resistance/inertial response of a hull depends on that distribution, not only on
the ice concentration.  This module provides

1. **FSD families** as normalised number distributions ``n(a)`` with closed-form
   ``cdf``/``quantile``/moments:

   * power law ``n(a) ~ a**-kappa`` on a finite support (``kappa`` = 1.5/2.5/3.5),
   * log-normal with prescribed coefficient of variation ``CV`` (0.3/0.5),
   * bimodal (coarse blocks + fine debris) mixture of two log-normals,
   * ``DeltaFSD`` -- the monodisperse limit ``CV -> 0``.

   Every family is anchored on the H1 reference median floe radius
   (``median = 5.0 m``, inside H1's 4.6-5.4 m band).

2. **A sequential compaction (压实) packing algorithm** instead of pure random
   sequential adsorption (RSA).  RSA jams near coverage 0.55 (H1, section 3 of
   ``docs/ice_channel_dem.md``) and cannot represent 6-8/10 ice.  The compaction
   algorithm drops the floes in descending size order; every floe is probed at
   ``probes`` random positions and is placed where its clearance to the
   already-placed floes and to the walls is largest, after which the floe and its
   neighbourhood are compacted by a short projection loop that pushes the
   neighbours aside.  A floe that does not fit the current void is therefore
   *squeezed in* rather than rejected, which is exactly where RSA blocks.  A
   final global collision-resolution pass removes the residual overlaps.  The
   default route is the *ordered* variant of the same idea (size-graded
   hexagonal rows), because a disordered packing in two dimensions jams at the
   random-close-packing limit (0.82-0.84) and cannot represent 8.5/10 ice --
   the measured jamming limit of both routes is in
   ``artifacts/fsd_packing.json``.  The realised coverage is
   ``sum(pi r^2)/(L W)`` and the overlap metric is the H1
   one used by ``IceChannelSimulation.initial_overlap``
   (``max(0, r_i + r_j - |x_i - x_j|)``), so the generated field can be handed to
   the H1 simulation unchanged.

3. **Generation-quality metrics** measured on every generated field: realised
   vs. target coverage, initial maximum overlap, mean coordination number,
   moments ``m0/m1/m2`` of the realised size distribution, iteration counts and
   wall time.

4. **The monodisperse limit is a regression lock**: for a degenerate FSD
   (``CV == 0``, i.e. ``LogNormalFSD(0.0)`` or ``DeltaFSD(r)``) the generator
   dispatches to ``IceChannelSimulation.from_field`` with the H1 keyword set and
   returns the *bit-identical* field (same ``pos``/``radius``/``thickness``/``z``
   arrays, same ``state_digest()``).  Raising ``CV`` from zero then moves the
   field continuously away from the H1 reference.

Determinism
-----------
All randomness comes from a single ``torch.Generator`` seeded with ``seed``; two
calls with the same arguments produce bit-identical fields
(:meth:`FloeField.field_digest`).
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from .ice_channel import IceChannelSimulation

__all__ = [
    "FloeSizeDistribution",
    "PowerLawFSD",
    "LogNormalFSD",
    "BimodalFSD",
    "DeltaFSD",
    "make_fsd",
    "FloeField",
    "generate_field",
    "rsa_placement",
    "DEAD_DEGENERATE_CV",
]

_DTYPE = torch.float64

#: A FSD whose coefficient of variation is below this value is treated as
#: degenerate (monodisperse limit) and routed to the H1 lattice placement.
DEAD_DEGENERATE_CV = 1.0e-6

#: H1 reference keyword set used by the monodisperse-limit dispatch.  Keeping
#: these values in one place is what makes the regression lock reproducible.
H1_REFERENCE_MEAN_RADIUS = 5.0
H1_REFERENCE_RADIUS_JITTER = 0.08
H1_REFERENCE_RADIUS_RANGE = (2.0, 8.0)
H1_REFERENCE_THICKNESS_RANGE = (0.4, 1.0)


# ---------------------------------------------------------------------------
# quadrature helper (analytic moments)
# ---------------------------------------------------------------------------

def _gauss_legendre_unit(n: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped from [-1, 1] to [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


_GL_U, _GL_W = _gauss_legendre_unit(256)


# ---------------------------------------------------------------------------
# floe size distributions
# ---------------------------------------------------------------------------

class FloeSizeDistribution:
    """Base class: a normalised *number* distribution ``n(a)`` of floe radii.

    Sub-classes implement :meth:`cdf`, :meth:`quantile` (inverse cdf) and
    :meth:`moments`.  ``m_k = E[a**k]`` for ``k = 0, 1, 2`` (``m0 = 1`` by
    normalisation, ``m1`` = mean radius, ``m2`` = mean square radius, which is
    the moment that sets the ice area ``sum(pi*R^2)``).
    """

    kind: str = "abstract"

    # -- distribution interface --------------------------------------------
    def cdf(self, a):
        raise NotImplementedError

    def quantile(self, u):
        raise NotImplementedError

    def support(self) -> Tuple[float, float]:
        raise NotImplementedError

    # -- derived quantities ------------------------------------------------
    def moments(self, n_quad: int = 256) -> Dict[str, float]:
        """Normalised moments via Gauss-Legendre quadrature of the quantile."""
        if int(n_quad) == 256:
            u, w = _GL_U, _GL_W
        else:
            u, w = _gauss_legendre_unit(n_quad)
        r = np.asarray(self.quantile(torch.as_tensor(u, dtype=_DTYPE)))
        return {
            "m0": 1.0,
            "m1": float((w * r).sum()),
            "m2": float((w * r ** 2).sum()),
        }

    @property
    def median(self) -> float:
        return float(self.quantile(torch.tensor(0.5, dtype=_DTYPE)))

    def mean_radius(self) -> float:
        return float(self.moments()["m1"])

    def mean_square_radius(self) -> float:
        return float(self.moments()["m2"])

    def mean_area(self) -> float:
        return math.pi * self.mean_square_radius()

    def radius_cv(self) -> float:
        m = self.moments()
        mean = m["m1"]
        var = max(m["m2"] - mean * mean, 0.0)
        return float(math.sqrt(var) / mean) if mean > 0 else 0.0

    def is_degenerate(self) -> bool:
        """True for the monodisperse limit (``CV -> 0``)."""
        return self.radius_cv() <= DEAD_DEGENERATE_CV

    def describe(self) -> Dict[str, object]:
        m = self.moments()
        lo, hi = self.support()
        return {
            "kind": self.kind,
            "median_radius_m": self.median,
            "mean_radius_m": m["m1"],
            "mean_square_radius_m2": m["m2"],
            "radius_cv": self.radius_cv(),
            "support_m": [lo, hi],
            "degenerate": self.is_degenerate(),
            "moments": m,
        }

    # -- sampling ----------------------------------------------------------
    def sample_radii(self, count: int, generator: torch.Generator,
                     mode: str = "stratified") -> torch.Tensor:
        """Draw ``count`` radii.

        ``mode='stratified'`` (default) uses *randomised stratified sampling* of
        the quantile variable: ``u_i = (i + U_i)/N`` with ``U_i ~ U(0,1)`` and a
        seeded random permutation.  Each radius keeps the correct marginal
        distribution, but the sample moments converge as ``O(N**-2)`` instead of
        ``O(N**-0.5)``, so the realised ``m1``/``m2`` -- and therefore the
        achievable coverage -- track the analytic FSD values closely (this is what
        criterion (d) requires).  ``mode='iid'`` is the plain independent-sample
        alternative, used in the benchmark to quantify sampling uncertainty.
        """
        count = int(count)
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        if mode not in ("stratified", "iid"):
            raise ValueError(f"unknown sampling mode {mode!r}")
        if mode == "iid":
            u = torch.rand(count, generator=generator, dtype=_DTYPE)
        else:
            jitter = torch.rand(count, generator=generator, dtype=_DTYPE)
            u = (torch.arange(count, dtype=_DTYPE) + jitter) / float(count)
            u = u[torch.randperm(count, generator=generator)]
        u = u.clamp(min=1e-12, max=1.0 - 1e-12)
        return self.quantile(u)


class PowerLawFSD(FloeSizeDistribution):
    """Truncated power law ``n(a) ~ a**-kappa`` on ``[a_lo, a_hi]``.

    The support is a geometric window of ratio ``upper_ratio = a_hi / a_lo``; the
    window is then *scaled* so that the median equals ``median`` (H1 reference
    5.0 m).  This keeps the compared slopes on the same median ice-floe size, so
    the sensitivity study isolates the shape of the FSD.
    """

    kind = "power_law"

    def __init__(self, kappa: float = 2.5, median: float = 5.0,
                 upper_ratio: float = 3.0) -> None:
        kappa = float(kappa)
        if not math.isfinite(kappa):
            raise ValueError("kappa must be finite")
        if abs(kappa - 1.0) < 1e-12:
            raise ValueError("kappa == 1 is the log-uniform limit and is not supported")
        if kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        ratio = float(upper_ratio)
        if not (ratio > 1.0):
            raise ValueError(f"upper_ratio must exceed 1, got {ratio}")
        med = float(median)
        if not (med > 0.0):
            raise ValueError("median must be positive")
        self.kappa = kappa
        self.upper_ratio = ratio
        self.median_target = med
        scale = (2.0 / (1.0 + ratio ** (1.0 - kappa))) ** (1.0 / (1.0 - kappa))
        self.a_lo = med * scale
        self.a_hi = self.a_lo * ratio

    def support(self) -> Tuple[float, float]:
        return self.a_lo, self.a_hi

    def cdf(self, a):
        p = 1.0 - self.kappa
        a = torch.as_tensor(a, dtype=_DTYPE) if not torch.is_tensor(a) else a
        num = a ** p - self.a_lo ** p
        den = self.a_hi ** p - self.a_lo ** p
        return (num / den).clamp(min=0.0, max=1.0)

    def quantile(self, u):
        p = 1.0 - self.kappa
        u = torch.as_tensor(u, dtype=_DTYPE) if not torch.is_tensor(u) else u
        base = self.a_lo ** p + u.clamp(min=0.0, max=1.0) * (self.a_hi ** p - self.a_lo ** p)
        return base ** (1.0 / p)

    def _analytic_moment(self, k: int) -> float:
        p = k + 1.0 - self.kappa
        if abs(p) < 1e-12:
            num = math.log(self.a_hi / self.a_lo)
        else:
            num = (self.a_hi ** p - self.a_lo ** p) / p
        q = 1.0 - self.kappa
        den = (self.a_hi ** q - self.a_lo ** q) / q
        return float(num / den)

    def moments(self, n_quad: int = 256) -> Dict[str, float]:
        return {"m0": 1.0, "m1": self._analytic_moment(1),
                "m2": self._analytic_moment(2)}

    def describe(self) -> Dict[str, object]:
        out = super().describe()
        out["kappa"] = self.kappa
        out["upper_ratio"] = self.upper_ratio
        return out


class LogNormalFSD(FloeSizeDistribution):
    """Log-normal radius distribution with prescribed coefficient of variation.

    ``sigma = sqrt(ln(1 + CV**2))``, ``mu = ln(median)``.  ``cv == 0`` collapses
    to the delta distribution at ``median`` (the monodisperse limit).
    """

    kind = "log_normal"

    def __init__(self, cv: float = 0.5, median: float = 5.0) -> None:
        cv = float(cv)
        if not math.isfinite(cv):
            raise ValueError("cv must be finite")
        if cv < 0.0:
            raise ValueError(f"cv must be non-negative, got {cv}")
        med = float(median)
        if not (med > 0.0):
            raise ValueError("median must be positive")
        self.cv = cv
        self.median_target = med
        self.sigma = math.sqrt(math.log(1.0 + cv * cv))
        self.mu = math.log(med)

    def support(self) -> Tuple[float, float]:
        if self.sigma == 0.0:
            return self.median_target, self.median_target
        z = 3.090232306167813    # 0.1 % / 99.9 % quantiles
        return (self.median_target * math.exp(-z * self.sigma),
                self.median_target * math.exp(z * self.sigma))

    def cdf(self, a):
        a = torch.as_tensor(a, dtype=_DTYPE) if not torch.is_tensor(a) else a
        if self.sigma == 0.0:
            return (a >= self.median_target).to(_DTYPE)
        z = (torch.log(a.clamp(min=1e-300)) - self.mu) / (self.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(z))

    def quantile(self, u):
        u = torch.as_tensor(u, dtype=_DTYPE) if not torch.is_tensor(u) else u
        u = u.clamp(min=1e-15, max=1.0 - 1e-15)
        if self.sigma == 0.0:
            return torch.full_like(u, self.median_target)
        z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        return torch.exp(self.mu + self.sigma * z)

    def moments(self, n_quad: int = 256) -> Dict[str, float]:
        if self.sigma == 0.0:
            r0 = self.median_target
            return {"m0": 1.0, "m1": r0, "m2": r0 * r0}
        return {
            "m0": 1.0,
            "m1": float(math.exp(self.mu + 0.5 * self.sigma ** 2)),
            "m2": float(math.exp(2.0 * self.mu + 2.0 * self.sigma ** 2)),
        }

    def describe(self) -> Dict[str, object]:
        out = super().describe()
        out["cv"] = self.cv
        out["sigma_log"] = self.sigma
        return out


class BimodalFSD(FloeSizeDistribution):
    """Mixture of a coarse (blocks) and a fine (debris) log-normal component.

    ``w`` is the *number* fraction of the fine component; it is solved
    numerically so that the mixture median equals ``median`` (5.0 m), which keeps
    this family comparable with the single-mode ones.  The fine component
    therefore dominates the particle count while the coarse component dominates
    the ice *area* -- the state of a fragmented, compacted floe field.
    """

    kind = "bimodal"

    def __init__(self, coarse_median: float = 7.2, fine_median: float = 2.2,
                 median: float = 5.0, coarse_cv: float = 0.2,
                 fine_cv: float = 0.35) -> None:
        if not (coarse_median > fine_median > 0.0):
            raise ValueError("bimodal requires coarse_median > fine_median > 0")
        self.coarse = LogNormalFSD(cv=coarse_cv, median=coarse_median)
        self.fine = LogNormalFSD(cv=fine_cv, median=fine_median)
        self.median_target = float(median)
        self.fine_fraction = self._solve_weight()

    def _solve_weight(self) -> float:
        """Bisection on the fine number fraction so that the median is exact."""
        target = self.median_target
        f_fine = float(self.fine.cdf(torch.tensor(target, dtype=_DTYPE)))
        f_coarse = float(self.coarse.cdf(torch.tensor(target, dtype=_DTYPE)))
        if abs(f_fine - f_coarse) < 1e-12:
            raise ValueError("the two components cannot be separated at the median")
        w_lin = (0.5 - f_coarse) / (f_fine - f_coarse)
        if not (0.0 < w_lin < 1.0):
            raise ValueError(
                "cannot reach the requested median with this bimodal "
                f"configuration (needs w={w_lin:.3f})")
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = (mid * f_fine + (1.0 - mid) * f_coarse)
            if f_mid < 0.5:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def support(self) -> Tuple[float, float]:
        lo_f, _ = self.fine.support()
        _, hi_c = self.coarse.support()
        return lo_f, hi_c

    def cdf(self, a):
        a = torch.as_tensor(a, dtype=_DTYPE) if not torch.is_tensor(a) else a
        w = self.fine_fraction
        return w * self.fine.cdf(a) + (1.0 - w) * self.coarse.cdf(a)

    def quantile(self, u):
        """Mixture quantile by vectorised bisection on a finite bracket."""
        u = torch.as_tensor(u, dtype=_DTYPE) if not torch.is_tensor(u) else u
        u = u.clamp(min=0.0, max=1.0)
        lo = torch.full_like(u, 1e-6)
        hi = torch.full_like(u, 1e6)
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < u
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def moments(self, n_quad: int = 256) -> Dict[str, float]:
        w = self.fine_fraction
        mf = self.fine.moments()
        mc = self.coarse.moments()
        return {"m0": 1.0,
                "m1": w * mf["m1"] + (1.0 - w) * mc["m1"],
                "m2": w * mf["m2"] + (1.0 - w) * mc["m2"]}

    def describe(self) -> Dict[str, object]:
        out = super().describe()
        out["fine_fraction_by_number"] = self.fine_fraction
        out["coarse_component"] = self.coarse.describe()
        out["fine_component"] = self.fine.describe()
        return out


class DeltaFSD(FloeSizeDistribution):
    """Monodisperse limit: every floe has radius ``radius``."""

    kind = "delta"

    def __init__(self, radius: float = 5.0) -> None:
        r = float(radius)
        if not (r > 0.0):
            raise ValueError(f"radius must be positive, got {r}")
        self.radius = r
        self.median_target = r

    def support(self) -> Tuple[float, float]:
        return self.radius, self.radius

    def cdf(self, a):
        a = torch.as_tensor(a, dtype=_DTYPE) if not torch.is_tensor(a) else a
        return (a >= self.radius).to(_DTYPE)

    def quantile(self, u):
        u = torch.as_tensor(u, dtype=_DTYPE) if not torch.is_tensor(u) else u
        return torch.full_like(u, self.radius)

    def moments(self, n_quad: int = 256) -> Dict[str, float]:
        return {"m0": 1.0, "m1": self.radius, "m2": self.radius * self.radius}

    def radius_cv(self) -> float:
        return 0.0


def make_fsd(spec: Mapping[str, object]) -> FloeSizeDistribution:
    """Build a FSD from a plain-data spec (``{"kind": ..., **params}``)."""
    params = dict(spec)
    kind = str(params.pop("kind", ""))
    if kind == "power_law":
        return PowerLawFSD(**params)
    if kind == "log_normal":
        return LogNormalFSD(**params)
    if kind == "bimodal":
        return BimodalFSD(**params)
    if kind == "delta":
        return DeltaFSD(**params)
    raise ValueError(f"unknown FSD kind {kind!r}")# ---------------------------------------------------------------------------
# neighbour search on a uniform grid (O(N) at fixed density)
# ---------------------------------------------------------------------------

def _candidate_pairs(pos: torch.Tensor, box: Tuple[float, float],
                     cell: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """All unordered neighbour pairs ``i < j`` whose cells are adjacent.

    The grid cell is ``cell`` wide (``cell >= 2 * max_radius``), so a pair can
    only overlap if both partners sit in the same 3x3 block of cells.  Cost is
    ``O(N)`` at fixed packing density, which is what makes the near-linear
    generation-time criterion reachable; a dense pairwise ``N x N`` tensor is
    ``O(N**2)`` and would violate it.
    """
    n = int(pos.shape[0])
    if n < 2:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    nx = max(1, int(math.ceil(box[0] / cell)))
    ny = max(1, int(math.ceil(box[1] / cell)))
    ix = torch.clamp((pos[:, 0] / cell).long(), 0, nx - 1)
    iy = torch.clamp((pos[:, 1] / cell).long(), 0, ny - 1)
    key = ix * ny + iy
    counts = torch.bincount(key, minlength=nx * ny)
    starts = torch.cumsum(counts, 0) - counts
    order = torch.argsort(key, stable=True)

    out_i = []
    out_j = []
    arange_n = torch.arange(n, dtype=torch.long)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            tx = ix + dx
            ty = iy + dy
            valid = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)
            if not bool(valid.any()):
                continue
            tkey = torch.where(valid, tx * ny + ty, torch.zeros_like(tx))
            lo = starts[tkey]
            cnt = torch.where(valid, counts[tkey], torch.zeros_like(lo))
            total = int(cnt.sum())
            if total == 0:
                continue
            cum = torch.cumsum(cnt, 0) - cnt
            idx_i = torch.repeat_interleave(arange_n, cnt)
            inner = torch.arange(total, dtype=torch.long) \
                - torch.repeat_interleave(cum, cnt) \
                + torch.repeat_interleave(lo, cnt)
            out_i.append(idx_i)
            out_j.append(order[inner])
    if not out_i:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    ii = torch.cat(out_i)
    jj = torch.cat(out_j)
    keep = ii < jj
    return ii[keep], jj[keep]


def _pair_overlap(pos: torch.Tensor, radius: torch.Tensor,
                  i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Signed overlap ``r_i + r_j - |x_i - x_j|`` for the given pairs [m]."""
    delta = pos[i] - pos[j]
    dist = torch.sqrt((delta ** 2).sum(dim=1) + 1e-30)
    return (radius[i] + radius[j]) - dist


def _push_apart(pos: torch.Tensor, radius: torch.Tensor, i: torch.Tensor,
                j: torch.Tensor, overlap: torch.Tensor,
                beta: float) -> torch.Tensor:
    """One Jacobi projection sweep over the overlapping pairs."""
    delta = pos[i] - pos[j]
    dist = torch.sqrt((delta ** 2).sum(dim=1) + 1e-30)
    normal = delta / dist[:, None]
    disp = torch.zeros_like(pos)
    disp.index_add_(0, i, normal * (0.5 * beta * overlap)[:, None])
    disp.index_add_(0, j, -normal * (0.5 * beta * overlap)[:, None])
    return pos + disp


def _clamp_to_box(pos: torch.Tensor, radius: torch.Tensor,
                  box: Tuple[float, float]) -> torch.Tensor:
    pos = pos.clone()
    pos[:, 0] = torch.minimum(torch.maximum(pos[:, 0], radius), box[0] - radius)
    pos[:, 1] = torch.minimum(torch.maximum(pos[:, 1], radius), box[1] - radius)
    return pos


def _max_overlap(pos: torch.Tensor, radius: torch.Tensor,
                 box: Tuple[float, float]) -> float:
    if pos.shape[0] < 2:
        return 0.0
    cell = 2.0 * float(radius.max())
    i, j = _candidate_pairs(pos, box, cell)
    if i.numel() == 0:
        return 0.0
    return float(_pair_overlap(pos, radius, i, j).max().clamp(min=0.0))


# ---------------------------------------------------------------------------
# field container
# ---------------------------------------------------------------------------

@dataclass
class FloeField:
    """A generated floe field plus its generation-quality metrics."""

    radius: torch.Tensor
    thickness: torch.Tensor
    pos: torch.Tensor
    domain: Tuple[float, float]
    target_coverage: float
    realised_coverage: float
    max_overlap: float
    coordination_number: float
    contact_pairs: int
    moments: Dict[str, float]
    fsd_moments: Dict[str, float]
    method: str
    seed: int
    generation_wall_s: float
    compression_steps: int = 0
    relaxation_sweeps: int = 0
    radius_correction: float = 1.0
    fsd: Optional[Dict[str, object]] = None
    extra: Dict[str, object] = dataclass_field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def floe_count(self) -> int:
        return int(self.radius.numel())

    def m1_deviation(self) -> float:
        """Relative deviation of realised ``m1`` from the analytic FSD ``m1``."""
        target = float(self.fsd_moments["m1"])
        if target <= 0.0:
            return float("nan")
        return abs(float(self.moments["m1"]) - target) / target

    def coverage_deviation(self) -> float:
        """Absolute deviation of the realised from the target coverage."""
        return abs(self.realised_coverage - self.target_coverage)

    # -- simulation bridge -------------------------------------------------
    def to_simulation(self, *, water=None, contact=None, hull=None,
                      dt: float = 5.0e-3, integrator: str = "euler"
                      ) -> IceChannelSimulation:
        """Build the matching :class:`IceChannelSimulation`.

        The vertical state is initialised at the Archimedes draught exactly as
        ``IceChannelSimulation.from_field`` does, so a multi-disperse field
        starts without a heave transient.
        """
        kwargs: Dict[str, object] = {"dt": dt, "integrator": integrator}
        if water is not None:
            kwargs["water"] = water
        if contact is not None:
            kwargs["contact"] = contact
        if hull is not None:
            kwargs["hull"] = hull
        sim = IceChannelSimulation([float(v) for v in self.radius],
                                   [float(v) for v in self.thickness], **kwargs)
        sim.pos = self.pos.clone()
        ratio = sim.water.ice_density / sim.water.water_density
        sim.z = 0.5 * sim.thickness - ratio * sim.thickness
        sim.domain = (float(self.domain[0]), float(self.domain[1]))
        sim.target_concentration = float(self.target_coverage)
        sim.seed = int(self.seed)
        return sim

    def field_digest(self) -> str:
        """SHA-256 over the generated arrays (placement regression lock)."""
        hasher = hashlib.sha256()
        for tensor in (self.pos, self.radius, self.thickness):
            hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()

    def state_digest(self, **sim_kwargs) -> str:
        return self.to_simulation(**sim_kwargs).state_digest()

    def summary(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "method": self.method,
            "seed": int(self.seed),
            "floe_count": self.floe_count,
            "domain_m": {"length": float(self.domain[0]),
                         "width": float(self.domain[1])},
            "target_coverage": float(self.target_coverage),
            "realised_coverage": float(self.realised_coverage),
            "coverage_deviation_abs": float(self.coverage_deviation()),
            "max_overlap_m": float(self.max_overlap),
            "coordination_number": float(self.coordination_number),
            "contact_pairs": int(self.contact_pairs),
            "moments_realised": {k: float(v) for k, v in self.moments.items()},
            "moments_fsd": {k: float(v) for k, v in self.fsd_moments.items()},
            "m1_deviation_rel": float(self.m1_deviation()),
            "radius_min_m": float(self.radius.min()),
            "radius_max_m": float(self.radius.max()),
            "mean_radius_m": float(self.radius.mean()),
            "mean_thickness_m": float(self.thickness.mean()),
            "radius_correction": float(self.radius_correction),
            "compression_steps": int(self.compression_steps),
            "relaxation_sweeps": int(self.relaxation_sweeps),
            "generation_wall_s": float(self.generation_wall_s),
            "field_digest": self.field_digest(),
            "state_digest": self.state_digest(),
        }
        if self.fsd is not None:
            out["fsd"] = self.fsd
        out.update(self.extra)
        return out


# ---------------------------------------------------------------------------
# generation-quality measurements
# ---------------------------------------------------------------------------

def _field_metrics(pos: torch.Tensor, radius: torch.Tensor,
                   box: Tuple[float, float], contact_tol: float
                   ) -> Tuple[float, float, int]:
    """``(max overlap, mean coordination number, contact pairs)``.

    A pair counts as a *contact* when its gap ``|x_i - x_j| - r_i - r_j`` is below
    ``contact_tol``; the mean coordination number is ``2 * pairs / N``.
    """
    n = int(pos.shape[0])
    if n < 2:
        return 0.0, 0.0, 0
    cell = 2.0 * float(radius.max())
    i, j = _candidate_pairs(pos, box, cell)
    if i.numel() == 0:
        return 0.0, 0.0, 0
    overlap = _pair_overlap(pos, radius, i, j)
    max_overlap = float(overlap.max().clamp(min=0.0))
    contacts = int((overlap > -float(contact_tol)).sum())
    return max_overlap, 2.0 * contacts / n, contacts


# ---------------------------------------------------------------------------
# compaction generator
# ---------------------------------------------------------------------------

def _relax(pos: torch.Tensor, radius: torch.Tensor, box: Tuple[float, float], *,
           tol: float, sweeps_per_round: int, rounds: int, beta: float,
           generator: torch.Generator, kick_scale: float
           ) -> Tuple[torch.Tensor, int, float]:
    """Iterative overlap resolution: projection sweeps + random-direction kicks.

    Each round runs ``sweeps_per_round`` Jacobi projection sweeps (push every
    overlapping pair apart by half its overlap, relaxation factor ``beta``) and
    then, if the overlap is still above ``tol``, injects a random-direction kick
    of amplitude ``kick_scale * mean radius``.  The kicks are what lets the
    packing re-order -- they are the "random direction" ingredient of the
    compaction protocol; without them the projection stalls in a local
    configuration.

    Returns ``(positions, sweeps_used, max_overlap)``.
    """
    n = int(pos.shape[0])
    if n < 2:
        return pos, 0, 0.0
    cell = 2.0 * float(radius.max())
    r_mean = float(radius.mean())
    sweeps = 0
    overlap = _max_overlap(pos, radius, box)
    for _ in range(max(1, int(rounds))):
        if overlap <= tol:
            break
        for _ in range(max(1, int(sweeps_per_round))):
            i, j = _candidate_pairs(pos, box, cell)
            if i.numel() == 0:
                return pos, sweeps, 0.0
            pen = _pair_overlap(pos, radius, i, j)
            active = pen > 0.0
            if not bool(active.any()):
                return pos, sweeps, 0.0
            pos = _push_apart(pos, radius, i[active], j[active], pen[active], beta)
            pos = _clamp_to_box(pos, radius, box)
            sweeps += 1
            overlap = float(pen[active].max())
            if overlap <= tol:
                break
        if overlap > tol:
            kick = torch.randn(pos.shape, generator=generator, dtype=_DTYPE)
            kick = kick / torch.linalg.norm(kick, dim=1, keepdim=True).clamp(min=1e-12)
            pos = pos + float(kick_scale) * r_mean * kick
            pos = _clamp_to_box(pos, radius, box)
            overlap = _max_overlap(pos, radius, box)
    return pos, sweeps, overlap


def _rescale_radii_to_coverage(radius: torch.Tensor, coverage: float,
                               area: float) -> Tuple[torch.Tensor, float]:
    """Single multiplicative correction so ``sum(pi r^2)/area == coverage``."""
    realised = float((math.pi * radius ** 2).sum()) / area
    if realised <= 0.0:
        raise ValueError("degenerate radius set (zero ice area)")
    factor = math.sqrt(float(coverage) / realised)
    return radius * factor, factor


def _hex_sites(box: Tuple[float, float], count: int, generator: torch.Generator
               ) -> torch.Tensor:
    """``count`` sites of a jittered hexagonal (triangular) lattice in ``box``.

    The lattice constant is chosen so that the sites are as evenly spread as the
    domain allows; surplus sites of the chosen column count are dropped at random
    (a few vacancies), which is the same construction family as the H1 field.
    """
    length, width = float(box[0]), float(box[1])
    area = length * width
    s = math.sqrt(2.0 * (area / max(count, 1)) / math.sqrt(3.0))
    best = None
    cols0 = max(1, int(round(length / s)))
    for cols in range(max(1, cols0 - 3), cols0 + 4):
        sx = length / cols
        rows = max(1, int(math.ceil(width / (0.5 * math.sqrt(3.0) * sx))))
        sy = width / rows
        pts = []
        for row in range(rows):
            offset = 0.5 * sx * (row % 2)
            n_col = cols + (row % 2)
            for col in range(n_col):
                x = offset + (col + 0.5) * sx
                if x < length - 1e-12:
                    pts.append((x, (row + 0.5) * sy))
        if len(pts) >= count and (best is None or len(pts) < len(best)):
            best = pts
        if len(pts) == count:
            best = pts
            break
    if best is None:                          # pragma: no cover - defensive
        best = [(0.5 * length, 0.5 * width)] * count
    if len(best) > count:
        keep = torch.randperm(len(best), generator=generator)[:count].tolist()
        best = [best[i] for i in sorted(keep)]
    return torch.tensor(best, dtype=_DTYPE)


def _lattice_pack(radius: torch.Tensor, box: Tuple[float, float],
                  generator: torch.Generator, *, coverage: float,
                  jitter: float = 0.015, relax_iters: int = 900,
                  beta: float = 0.7) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Area-matched hexagonal compaction for a multi-disperse radius set.

    The construction is H1's lattice idea generalised to a *distribution* of
    radii.  Sites of a hexagonal lattice are seeded evenly in the domain and one
    floe is assigned to each site; the neighbouring pair ``(a, b)`` then has the
    size-matched target spacing::

        d_ab >= lambda * (r_a + r_b),   lambda = sqrt(pi / (2 sqrt(3) phi))

    ``lambda`` is exactly the spacing of a hexagonal packing at coverage ``phi``
    (``lambda = 1`` is the touching hexagonal limit 0.907, ``lambda = 1.033`` at
    phi = 0.85, ``lambda = 1.229`` at phi = 0.60).  The sites are then relaxed
    under these pair constraints (Jacobi projection, walls included), which
    deforms the lattice locally so that large floes take the space of their
    small neighbours.  A disordered packing cannot do this: it jams at the
    random-close-packing limit (0.82-0.84 in 2D), which is why the ordered
    construction is the one that reaches 8.5/10 ice.

    Finally a small positional jitter is applied and any overlap it creates is
    removed by a bounded projection pass.
    """
    n = int(radius.numel())
    if n <= 0:
        raise ValueError("lattice packing needs at least one floe")
    length, width = float(box[0]), float(box[1])
    lam = math.sqrt(math.pi / (2.0 * math.sqrt(3.0) * float(coverage)))
    pos = _hex_sites(box, n, generator)
    spacing = math.sqrt((length * width) / n)
    pos = pos + (torch.rand(pos.shape, generator=generator, dtype=_DTYPE) - 0.5) \
        * (0.4 * jitter * spacing)
    pos = _clamp_to_box(pos, radius, box)

    # neighbour topology from the ideal lattice (fixed for the relaxation)
    cell = 2.0 * float(radius.max())
    i, j = _candidate_pairs(pos, box, cell)
    if i.numel() == 0:
        i, j = _candidate_pairs(pos, box, 1.5 * spacing)
    target = lam * (radius[i] + radius[j])
    sweeps = 0
    for _ in range(max(1, int(relax_iters))):
        delta = pos[i] - pos[j]
        dist = torch.linalg.norm(delta, dim=1).clamp(min=1e-30)
        need = target - dist
        act = need > 0.0
        if not bool(act.any()):
            break
        normal = delta[act] / dist[act][:, None]
        disp = torch.zeros_like(pos)
        disp.index_add_(0, i[act], normal * (0.5 * beta * need[act])[:, None])
        disp.index_add_(0, j[act], -normal * (0.5 * beta * need[act])[:, None])
        pos = _clamp_to_box(pos + disp, radius, box)
        sweeps += 1
    pos = _clamp_to_box(pos + torch.randn(pos.shape, generator=generator,
                                          dtype=_DTYPE) * (jitter * spacing), radius, box)
    pos, extra_sweeps, _ = _relax(pos, radius, box, tol=1e-3, sweeps_per_round=40,
                                  rounds=10, beta=0.8, generator=generator,
                                  kick_scale=0.002)
    return pos, {"lattice_spacing_target_scales": {"lambda": lam,
                                                   "mean_spacing_m": spacing},
                 "constraint_sweeps": int(sweeps),
                 "cleanup_sweeps": int(extra_sweeps),
                 "site_count_before_trim": int(n)}


def _insertion_pack(radius: torch.Tensor, box: Tuple[float, float],
                    generator: torch.Generator, *, probes: int = 24,
                    overlap_tol: float = 0.03, local_sweeps: int = 40,
                    neighbour_reach: Optional[float] = None,
                    max_retries: int = 4, beta: float = 0.6
                    ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Sequential compaction placement in descending size order.

    For every floe (largest first) ``probes`` random positions are scored by
    their clearance to the already-placed floes and to the walls; the floe is
    inserted at the best position and the local neighbourhood is compacted by a
    bounded projection loop (``local_sweeps``), which pushes neighbours aside to
    make room.  If the best ``max_retries`` positions still leave an overlap
    above ``overlap_tol``, the last trial position is kept and the residual
    overlap is handed to the global collision-resolution pass -- the placement
    therefore never rejects a floe, which is the qualitative difference to RSA.

    Neighbour queries go through a uniform-cell hash grid, so the cost per
    insertion is ``O(probes * occupancy)`` instead of ``O(probes * N)`` and the
    whole routine stays near-linear in the floe count.
    """
    n = int(radius.numel())
    if n <= 0:
        raise ValueError("insertion packing needs at least one floe")
    reach = float(neighbour_reach) if neighbour_reach else 2.0 * float(radius.max())
    cell = reach
    nx = max(1, int(math.ceil(box[0] / cell)))
    ny = max(1, int(math.ceil(box[1] / cell)))
    grid: Dict[Tuple[int, int], list] = {}

    def _cell_of(x: float, y: float) -> Tuple[int, int]:
        return (min(max(int(x / cell), 0), nx - 1), min(max(int(y / cell), 0), ny - 1))

    pos = torch.zeros((n, 2), dtype=_DTYPE)
    cell_of_floe: Dict[int, Tuple[int, int]] = {}
    placed = []
    local_sweeps_total = 0
    fallbacks = 0
    order = torch.argsort(radius, descending=True).tolist()

    for k in order:
        rk = float(radius[k])
        cx = torch.rand(probes, generator=generator, dtype=_DTYPE) * (box[0] - 2 * rk) + rk
        cy = torch.rand(probes, generator=generator, dtype=_DTYPE) * (box[1] - 2 * rk) + rk
        scored = []
        for c in range(int(probes)):
            x = float(cx[c])
            y = float(cy[c])
            clear = min(x - rk, box[0] - rk - x, y - rk, box[1] - rk - y)
            ix, iy = _cell_of(x, y)
            near = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for nb in grid.get((ix + dx, iy + dy), ()):
                        d = math.hypot(float(pos[nb, 0]) - x, float(pos[nb, 1]) - y) \
                            - float(radius[nb]) - rk
                        if d < clear:
                            clear = d
                        if d < 0.0:
                            near.append(nb)
            scored.append((clear, c, x, y, near))
        scored.sort(key=lambda t: -t[0])

        accepted = False
        for clear, c, x, y, near in scored[:max(1, int(max_retries))]:
            ids = [k] + near
            sub = torch.tensor(ids, dtype=torch.long)
            sub_r = radius[sub]
            p = pos[sub].clone()
            p[0] = torch.tensor([x, y], dtype=_DTYPE)
            p = _clamp_to_box(p, sub_r, box)
            used = 0
            for _ in range(max(1, int(local_sweeps))):
                dist = torch.cdist(p, p)
                pen = (sub_r[:, None] + sub_r[None, :]) - dist
                pen.fill_diagonal_(0.0)
                act = torch.triu(pen > 0.0, diagonal=1)
                if not bool(act.any()):
                    break
                ii, jj = act.nonzero(as_tuple=True)
                pen_ij = pen[ii, jj]
                delta = p[ii] - p[jj]
                dd = torch.linalg.norm(delta, dim=1).clamp(min=1e-30)
                normal = delta / dd[:, None]
                disp = torch.zeros_like(p)
                disp.index_add_(0, ii, normal * (0.5 * beta * pen_ij)[:, None])
                disp.index_add_(0, jj, -normal * (0.5 * beta * pen_ij)[:, None])
                p = _clamp_to_box(p + disp, sub_r, box)
                used += 1
            # residual overlap of the inserted floe with the committed field
            if near:
                d = torch.linalg.norm(pos[near] - p[0], dim=1) - radius[near] - rk
                residual = float((-d).max().clamp(min=0.0))
            else:
                residual = 0.0
            local_sweeps_total += used
            # commit the neighbourhood and the inserted floe
            for slot, idx in enumerate(ids):
                pos[idx] = p[slot]
            for idx in ids:
                old = cell_of_floe.get(idx)
                new = _cell_of(float(pos[idx, 0]), float(pos[idx, 1]))
                if old != new:
                    if old is not None:
                        bucket = grid.get(old)
                        if bucket and idx in bucket:
                            bucket.remove(idx)
                    grid.setdefault(new, []).append(idx)
                    cell_of_floe[idx] = new
            if k not in cell_of_floe:
                key = _cell_of(float(pos[k, 0]), float(pos[k, 1]))
                grid.setdefault(key, []).append(k)
                cell_of_floe[k] = key
            if k not in placed:
                placed.append(k)
            if residual > overlap_tol:
                fallbacks += 1
            accepted = True
            break
        if not accepted:                      # pragma: no cover - probes >= 1
            raise RuntimeError("insertion packing failed to place a floe")
    return pos, {"local_sweeps": int(local_sweeps_total),
                 "floes_placed": len(placed),
                 "floes_above_tolerance": int(fallbacks),
                 "probes": int(probes),
                 "overlap_tolerance_m": float(overlap_tol)}


def generate_field(fsd, coverage: float, *, domain_length: float,
                   domain_width: float, seed: int,
                   method: str = "lattice",
                   thickness_range: Tuple[float, float] = H1_REFERENCE_THICKNESS_RANGE,
                   max_overlap_target: float = 0.06,
                   probes: int = 24,
                   local_sweeps: int = 40,
                   contact_tol: float = 0.02,
                   sample_mode: str = "stratified",
                   h1_kwargs: Optional[Mapping[str, object]] = None,
                   sim_kwargs: Optional[Mapping[str, object]] = None
                   ) -> FloeField:
    """Generate a floe field of prescribed coverage and FSD.

    Parameters
    ----------
    fsd
        Floe-size distribution (a :class:`FloeSizeDistribution` or a
        ``{"kind": ..., **params}`` spec mapping).
    coverage
        Target ice concentration (fraction of domain area; ``0.6`` = 6/10).
    method
        ``"compaction"`` (default) -- dilute random start + progressive
        anisotropic box compression in randomly drawn axis directions + overlap
        resolution.  ``"rsa"`` -- random sequential adsorption, kept as the
        *blocking* negative control (it jams near 0.55).  ``"h1_lattice"`` -- the
        H1 jittered hexagonal lattice, required for (and auto-selected with) a
        degenerate FSD, where it reproduces the H1 field bit-for-bit.
    max_overlap_target
        Required ``max_initial_overlap`` of the produced field [m] (H1
        convention; criterion (b) asks for <= 0.06 m).
    sample_mode
        ``"stratified"`` (default) or ``"iid"`` FSD sampling (see
        :meth:`FloeSizeDistribution.sample_radii`).
    """
    fsd = _as_fsd(fsd)
    coverage = float(coverage)
    if not math.isfinite(coverage) or not (0.0 < coverage < 1.0):
        raise ValueError(f"coverage must lie in (0, 1), got {coverage!r}")
    length = float(domain_length)
    width = float(domain_width)
    if not (length > 0.0 and width > 0.0):
        raise ValueError("domain_length and domain_width must be positive")
    if method not in ("lattice", "compaction", "rsa", "h1_lattice"):
        raise ValueError(f"unknown method {method!r}")
    if method in ("lattice", "compaction") and fsd.is_degenerate():
        method = "h1_lattice"
    if int(probes) < 1:
        raise ValueError(f"probes must be >= 1, got {probes}")
    if method == "h1_lattice" and not fsd.is_degenerate():
        raise ValueError(
            "h1_lattice placement requires a degenerate (monodisperse) FSD")

    if method == "h1_lattice":
        return _h1_lattice_field(coverage, length, width, int(seed),
                                 thickness_range=thickness_range,
                                 h1_kwargs=h1_kwargs, sim_kwargs=sim_kwargs)

    area = length * width
    generator = torch.Generator().manual_seed(int(seed))
    tic = time.perf_counter()

    if method == "rsa":
        out = rsa_placement(fsd, coverage, domain_length=length,
                            domain_width=width, seed=int(seed),
                            thickness_range=thickness_range,
                            sample_mode=sample_mode)
        out.generation_wall_s = time.perf_counter() - tic
        return out

    # ---- particle count from the FSD area moment -------------------------
    mean_area = fsd.mean_area()
    count = max(1, int(round(coverage * area / mean_area)))
    radius = fsd.sample_radii(count, generator, mode=sample_mode)
    radius, correction = _rescale_radii_to_coverage(radius, coverage, area)
    r_max = float(radius.max())
    if 2.0 * r_max > min(length, width):
        raise ValueError(
            f"largest floe (2r = {2 * r_max:.2f} m) does not fit the domain "
            f"({length:.1f} x {width:.1f} m)")

    # thickness: stratified over the requested band, independent of the radius
    u = (torch.arange(count, dtype=_DTYPE) + 0.5) / count
    u = u[torch.randperm(count, generator=generator)]
    t_lo, t_hi = float(thickness_range[0]), float(thickness_range[1])
    thickness = t_lo + (t_hi - t_lo) * u

# ---- sequential compaction placement (descending radius) -------------
    # Floes are dropped in descending size order; every floe is probed at
    # ``probes`` random positions and placed where the local clearance to the
    # already-inserted floes and to the walls is largest (random-direction
    # probing = the "压实" step).  The inserted floe and its neighbourhood are
    # then compacted by a short projection loop, so a floe that does not fit the
    # current void is squeezed in by pushing its neighbours aside instead of
    # being rejected (which is exactly where RSA blocks).  A final global
    # collision-resolution pass removes the residual overlaps.
    if method == "lattice":
        pos, place_stats = _lattice_pack(radius, (length, width), generator,
                                         coverage=coverage)
        relaxed_sweeps = int(place_stats.get("constraint_sweeps", 0))
    else:
        pos, place_stats = _insertion_pack(
            radius, (length, width), generator, probes=int(probes),
            overlap_tol=float(max_overlap_target) * 0.5,
            local_sweeps=int(local_sweeps),
            neighbour_reach=float(2.0 * r_max))
        relaxed_sweeps = int(place_stats["local_sweeps"])
    relax_sweeps = relaxed_sweeps
    compression_steps = 0

    # ---- global collision resolution -------------------------------------
    pos, used, _ = _relax(pos, radius, (length, width), tol=1e-3,
                          sweeps_per_round=40, rounds=12, beta=0.8,
                          generator=generator, kick_scale=0.003)
    relax_sweeps += used
    wall = time.perf_counter() - tic

    realised = float((math.pi * radius ** 2).sum()) / (length * width)
    max_ov, coord, pairs = _field_metrics(pos, radius, (length, width), contact_tol)
    moments = {"m0": 1.0, "m1": float(radius.mean()),
               "m2": float((radius ** 2).mean())}
    return FloeField(
        radius=radius, thickness=thickness, pos=pos, domain=(length, width),
        target_coverage=coverage, realised_coverage=realised, max_overlap=max_ov,
        coordination_number=coord, contact_pairs=pairs, moments=moments,
        fsd_moments=fsd.moments(), method=method, seed=int(seed),
        generation_wall_s=wall, compression_steps=compression_steps,
        relaxation_sweeps=relax_sweeps, radius_correction=correction,
        fsd=fsd.describe(),
        extra={"sample_mode": sample_mode,
               "max_overlap_target": float(max_overlap_target),
               "placement": {"lattice": "area_matched_hex_lattice",
                             "compaction":
                                 "descending_radius_best_candidate_compaction"
                             }[method],
               "probes": int(probes),
               "placement_statistics": place_stats,
               "converged": bool(max_ov <= float(max_overlap_target))})


def _as_fsd(fsd) -> FloeSizeDistribution:
    if isinstance(fsd, FloeSizeDistribution):
        return fsd
    if isinstance(fsd, Mapping):
        return make_fsd(fsd)
    raise TypeError(
        f"expected a FloeSizeDistribution or a spec mapping, got {type(fsd)!r}")


def _h1_lattice_field(coverage: float, length: float, width: float, seed: int, *,
                      thickness_range, h1_kwargs=None,
                      sim_kwargs=None) -> FloeField:
    """Monodisperse limit: the H1 lattice placement, reproduced bit-for-bit."""
    h1 = dict(h1_kwargs or {})
    h1.setdefault("mean_radius", H1_REFERENCE_MEAN_RADIUS)
    h1.setdefault("radius_jitter", H1_REFERENCE_RADIUS_JITTER)
    h1.setdefault("radius_range", H1_REFERENCE_RADIUS_RANGE)
    h1.setdefault("thickness_range", tuple(thickness_range))
    tic = time.perf_counter()
    kwargs = dict(sim_kwargs or {})
    kwargs.update(h1)
    sim = IceChannelSimulation.from_field(
        float(coverage), domain_length=float(length), domain_width=float(width),
        seed=int(seed), **kwargs)
    wall = time.perf_counter() - tic
    area = float(length) * float(width)
    radius = sim.radius.clone()
    realised = float((math.pi * radius ** 2).sum()) / area
    max_ov, coord, pairs = _field_metrics(
        sim.pos, radius, (float(length), float(width)), 0.02)
    moments = {"m0": 1.0, "m1": float(radius.mean()),
               "m2": float((radius ** 2).mean())}
    fsd_ref = DeltaFSD(float(h1["mean_radius"]))
    return FloeField(
        radius=radius, thickness=sim.thickness.clone(), pos=sim.pos.clone(),
        domain=(float(length), float(width)), target_coverage=float(coverage),
        realised_coverage=realised, max_overlap=max_ov,
        coordination_number=coord, contact_pairs=pairs, moments=moments,
        fsd_moments=fsd_ref.moments(), method="h1_lattice", seed=int(seed),
        generation_wall_s=wall, compression_steps=0, relaxation_sweeps=0,
        radius_correction=1.0, fsd=fsd_ref.describe(),
        extra={"h1_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in h1.items()},
               "monodisperse_limit": True,
               "converged": bool(max_ov <= 0.06)})


# ---------------------------------------------------------------------------
# RSA placement (blocking negative control)
# ---------------------------------------------------------------------------

def rsa_placement(fsd, coverage: float, *, domain_length: float,
                  domain_width: float, seed: int,
                  thickness_range: Tuple[float, float] = H1_REFERENCE_THICKNESS_RANGE,
                  max_consecutive_failures: int = 500,
                  max_attempts: int = 400_000,
                  sample_mode: str = "stratified") -> FloeField:
    """Random sequential adsorption -- the *blocking* negative control.

    Floes of the prescribed FSD are inserted one at a time at uniformly random
    positions; an insertion is rejected when the disk overlaps an accepted floe.
    Placement stops after ``max_consecutive_failures`` consecutive rejections
    (the jamming/blocking point), so the realised coverage saturates near 0.55
    for near-monodisperse disks and stays well below a 0.85 target.  This is the
    control that criterion (f) contrasts with the compaction generator.
    """
    fsd = _as_fsd(fsd)
    coverage = float(coverage)
    if not math.isfinite(coverage) or not (0.0 < coverage < 1.0):
        raise ValueError(f"coverage must lie in (0, 1), got {coverage!r}")
    length, width = float(domain_length), float(domain_width)
    area = length * width
    generator = torch.Generator().manual_seed(int(seed))
    mean_area = fsd.mean_area()
    count = max(1, int(round(coverage * area / mean_area)))
    radius = fsd.sample_radii(count, generator, mode=sample_mode)
    radius, correction = _rescale_radii_to_coverage(radius, coverage, area)
    if 2.0 * float(radius.max()) > min(length, width):
        raise ValueError("largest floe does not fit the domain")

    tic = time.perf_counter()
    accepted_pos = []
    accepted_r = []
    accepted_t = []
    failures = 0
    attempts = 0
    t_lo, t_hi = float(thickness_range[0]), float(thickness_range[1])
    thicknesses = t_lo + (t_hi - t_lo) * torch.rand(count, generator=generator,
                                                    dtype=_DTYPE)
    while failures < int(max_consecutive_failures) and attempts < int(max_attempts):
        idx = min(len(accepted_pos), count - 1)
        attempts += 1
        r = float(radius[idx])
        x = float(torch.rand(1, generator=generator, dtype=_DTYPE)) * (length - 2 * r) + r
        y = float(torch.rand(1, generator=generator, dtype=_DTYPE)) * (width - 2 * r) + r
        ok = True
        if accepted_pos:
            pts = torch.tensor(accepted_pos, dtype=_DTYPE)
            rr = torch.tensor(accepted_r, dtype=_DTYPE)
            d = torch.sqrt(((pts - torch.tensor([x, y], dtype=_DTYPE)) ** 2).sum(dim=1))
            if bool((d < rr + r).any()):
                ok = False
        if ok:
            accepted_pos.append([x, y])
            accepted_r.append(r)
            accepted_t.append(float(thicknesses[idx]))
            failures = 0
            if len(accepted_pos) >= count:
                break
        else:
            failures += 1
    wall = time.perf_counter() - tic

    if accepted_pos:
        pos = torch.tensor(accepted_pos, dtype=_DTYPE)
        rad = torch.tensor(accepted_r, dtype=_DTYPE)
        thick = torch.tensor(accepted_t, dtype=_DTYPE)
    else:
        pos = torch.zeros((1, 2), dtype=_DTYPE)
        rad = torch.tensor([radius[0]], dtype=_DTYPE)
        thick = torch.tensor([float(thicknesses[0])], dtype=_DTYPE)
    realised = float((math.pi * rad ** 2).sum()) / area
    max_ov, coord, pairs = _field_metrics(pos, rad, (length, width), 0.02)
    placed = int(rad.numel())
    return FloeField(
        radius=rad, thickness=thick, pos=pos, domain=(length, width),
        target_coverage=coverage, realised_coverage=realised, max_overlap=max_ov,
        coordination_number=coord, contact_pairs=pairs,
        moments={"m0": 1.0, "m1": float(rad.mean()),
                 "m2": float((rad ** 2).mean())},
        fsd_moments=fsd.moments(), method="rsa", seed=int(seed),
        generation_wall_s=wall, compression_steps=0, relaxation_sweeps=0,
        radius_correction=correction, fsd=fsd.describe(),
        extra={"placed_floes": placed, "requested_floes": int(count),
               "attempts": int(attempts),
               "consecutive_failures_at_stop": int(failures),
               "blocked": bool(realised < coverage - 0.005)})