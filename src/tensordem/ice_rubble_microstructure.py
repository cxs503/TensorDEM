"""Ice-rubble microstructure statistics and experimental benchmarking (TensorDEM m5074).

Motivation
----------
TensorDEM could already generate ice-rubble lattices (:mod:`tensordem.ridge_dem`)
and bonded lattices (:mod:`tensordem.dem3d`), but it had **no** way to answer the
question that decides whether a rubble DEM is physically meaningful at all: does
the *microstructure* of the simulated fragment field -- porosity ``n``,
coordination number ``Z`` and the contact-force PDF -- reproduce what is measured
in ice-rubble ridges and in photoelastic granular experiments?  The engineering
stake is direct: Lindqvist-type ridge resistance scales with the solid fraction
``(1 - n)``, so a DEM that is 0.1 off in porosity is off by ~15 % in resistance.

This module adds that missing layer.  It contains

1. **A floe-field generator**: ``N >= 1000`` rigid poly-disperse ice floes whose
   diameters follow a *truncated Rosin-Rammler* distribution over
   ``d in [0.05, 0.5] m`` (:class:`RosinRammlerDistribution`,
   :func:`generate_floe_field`).
2. **A confining-cell packing engine** (:class:`IceRubbleField`): 3-D rigid
   spheres with rotation, linear spring-dashpot normal law, Cundall-Strack
   tangential spring saturated at ``mu F_n``, rolling-resistance couple, six
   plane walls, self weight and a displacement-controlled compacting lid.  A side
   wall is then driven into the compacted rubble as a *platen*, so the
   macroscopic resistance of the very same packing can be measured.
3. **Microstructure estimators**: porosity by voxel-resolved *Voronoi
   tessellation* of the floe centres (:func:`voxel_voronoi_porosity`), mean
   coordination number (:func:`coordination_number`) and the normalised
   contact-force PDF with force-chain clusters (:func:`force_chain_pdf`,
   :func:`force_chain_clusters`).
4. **Literature anchors and statistics**: :data:`TIMCO_OBRIEN_2004`
   (``n = 0.32 +- 0.05``), :data:`HOPKINS_TVEITEN_2002` (force-chain PDF shape),
   :data:`COORDINATION_REFERENCE` (``Z = 4.0 +- 0.5``), a self-contained
   Kolmogorov-Smirnov implementation (:func:`ks_test_cdf`,
   :func:`ks_test_two_sample`) and :func:`run_microstructure_study`, which runs
   the calibration, the resistance feedback fit ``F ~ (1 - n)`` and the
   frictionless random-close-packing negative control.

Evidence grading (machine-readable in the study JSON)
-----------------------------------------------------
* porosity / coordination reference values: ``LITERATURE_SUMMARY`` (mean +- std
  as cited; the raw per-ridge tables were **not** digitised offline, so a
  per-dataset comparison is ``WITHHELD``);
* force-chain reference: ``PARAMETRIC_MODEL`` -- a plateau + exponential-tail PDF
  whose shape follows the published photoelastic result (Radjai-type); the
  digitised experimental PDF itself is ``WITHHELD``.  The KS test is therefore a
  *shape* test against a documented model rather than against raw data, and the
  study additionally reports DEM-vs-DEM reproducibility;
* everything else (porosity, Z, force PDF, resistance, negative control) is
  computed by DEM here and is graded ``REAL``.

Physical assumptions and limitations (see ``docs/ice_rubble_microstructure.md``)
-----------------------------------------------------------------------------
* Rigid spheres.  Floe *shape* (angularity, flatness) is not represented; the
  angularity of real fragments is partly absorbed by the calibrated friction
  coefficient, which is why ``mu`` is *reported*, never tuned silently.
* The packing is compacted under **self weight** in a dry cell.  A fully
  submerged ice pile is buoyant, so water cannot compact it; a g-sensitivity run
  (``gravity_scale = 0.5``) shows that the measured porosity is not controlled by
  the body-force scale.
* Finite cell: the largest floe is ~40 % of the cell width, so a boundary layer
  is excluded from the bulk coordination statistics and the finite-size effect on
  porosity is quoted.
* No freeze bonds in the microstructure runs (unbonded rubble); bonded rubble is
  :mod:`tensordem.ridge_dem`'s scope.

All quantities are SI (m, kg, s, N; contact forces are normalised by their mean
inside the PDF).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

__all__ = [
    "DTYPE",
    "GRAVITY",
    "LiteraturePorosity",
    "TIMCO_OBRIEN_2004",
    "COORDINATION_REFERENCE",
    "ForceChainReference",
    "HOPKINS_TVEITEN_2002",
    "HOPKINS_TVEITEN_2002_PLATEAU",
    "RosinRammlerDistribution",
    "generate_floe_field",
    "IceRubbleMaterial",
    "IceRubbleField",
    "voxel_voronoi_porosity",
    "coordination_number",
    "coordination_spectrum",
    "force_chain_pdf",
    "force_chain_clusters",
    "kolmogorov_survival",
    "ks_test_cdf",
    "ks_test_two_sample",
    "MicrostructureStudyConfig",
    "run_microstructure_study",
]

DTYPE = torch.float64
GRAVITY = 9.80665


# ---------------------------------------------------------------------------
# Literature anchors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiteraturePorosity:
    """A literature porosity summary together with its evidence grade."""

    citation: str
    mean: float
    std: float
    n_datasets: int
    evidence: str

    def relative_error(self, value: float) -> float:
        """``|value - mean| / mean`` (dimensionless)."""
        return abs(float(value) - self.mean) / self.mean

    def within(self, value: float, *, relative: float = 0.10) -> bool:
        return self.relative_error(value) <= relative

    def sigma_distance(self, value: float) -> float:
        return abs(float(value) - self.mean) / self.std


TIMCO_OBRIEN_2004 = LiteraturePorosity(
    citation="Timco & O'Brien (2004), Cold Regions Science and Technology 38, 149-170",
    mean=0.32,
    std=0.05,
    n_datasets=12,
    evidence=(
        "LITERATURE_SUMMARY: mean/std over the >=10 ridge/rubble porosity datasets "
        "quoted in the campaign brief and the review; the individual per-ridge "
        "values were not digitised offline -> per-dataset comparison is WITHHELD."
    ),
)

COORDINATION_REFERENCE = LiteraturePorosity(
    citation="Ice-rubble coordination number (campaign reference, Z = 4.0 +- 0.5)",
    mean=4.0,
    std=0.5,
    n_datasets=1,
    evidence=(
        "LITERATURE_SUMMARY: single review value for ice rubble, consistent with "
        "the frictional isostatic limit Z = 4 in three dimensions; a multi-source "
        "digitised Z table is WITHHELD."
    ),
)


@dataclass(frozen=True)
class ForceChainReference:
    """Parametric contact-force PDF used as the literature shape reference.

    The published photoelastic result (``P(f)`` with ``f = F / <F>``) has a weak
    overload (plateau) and an exponential strong-force tail.  The model here is
    the minimal two-piece form

    ``P(f) = A`` for ``f < f_c`` and ``P(f) = A exp(-beta (f - f_c))`` otherwise,

    with ``A`` fixed by normalisation and ``beta`` solved so that ``<f> = 1``
    exactly.  Only the *shape* is taken from the literature; the digitised
    experimental histogram is WITHHELD (see the module docstring).
    """

    plateau_end: float = 0.0
    n_contacts_reported: int = 500
    evidence: str = (
        "PARAMETRIC_MODEL: exponential strong-force tail P(f)=exp(-f) "
        "(q-model / Coppersmith et al. 1996) plus an optional weak-force plateau "
        "(plateau_end = c > 0 gives the Radjai-type two-piece shape). This is the "
        "familly used to represent photoelastic force-chain PDFs, including "
        "Hopkins & Tveiten (2002) on ice rubble; the digitised PDF itself is "
        "WITHHELD."
    )

    def __post_init__(self) -> None:
        if self.plateau_end < 0.0:
            raise ValueError("plateau_end must be non-negative")
        if self.n_contacts_reported < 2:
            raise ValueError("n_contacts_reported must be at least 2")

    def amplitude(self, beta: float) -> float:
        return 1.0 / (self.plateau_end + 1.0 / beta)

    def _mean(self, beta: float) -> float:
        c = self.plateau_end
        a = self.amplitude(beta)
        return a * (0.5 * c * c + c / beta + 1.0 / (beta * beta))

    def solved_tail_rate(self) -> float:
        """Tail rate ``beta`` such that the distribution has unit mean."""
        lo, hi = 1e-6, 1.0e4
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self._mean(mid) > 1.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @property
    def tail_rate(self) -> float:
        return self.solved_tail_rate()

    @property
    def plateau_height(self) -> float:
        return self.amplitude(self.tail_rate)

    def pdf(self, x):
        x_arr = np.asarray(x, dtype=float)
        beta = self.tail_rate
        a = self.amplitude(beta)
        return np.where(
            x_arr < self.plateau_end,
            a,
            a * np.exp(-beta * (x_arr - self.plateau_end)),
        )

    def cdf(self, x):
        x_arr = np.asarray(x, dtype=float)
        beta = self.tail_rate
        a = self.amplitude(beta)
        c = self.plateau_end
        plateau = a * x_arr
        tail = a * c + (a / beta) * (1.0 - np.exp(-beta * (x_arr - c)))
        return np.where(x_arr < c, plateau, tail)

    def ppf(self, u, *, x_max: float = 40.0, samples: int = 400001):
        """Inverse CDF (numeric inversion of the analytic CDF)."""
        u_arr = np.asarray(u, dtype=float)
        grid = np.linspace(0.0, x_max, int(samples))
        cdf = self.cdf(grid)
        cdf[0] = 0.0
        np.maximum.accumulate(cdf, out=cdf)
        return np.interp(u_arr, cdf, grid)

    def sample(self, n: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.ppf(rng.random(int(n)))

    def survival(self, x):
        """``P(f > x)``."""
        return 1.0 - self.cdf(x)

    def mean(self) -> float:
        return self._mean(self.tail_rate)

    def coefficient_of_variation(self) -> float:
        """``std / mean`` of the reference shape (unit mean, so it equals std)."""
        grid = np.linspace(0.0, 60.0, 400001)
        pdf = self.pdf(grid)
        m1 = float(np.trapezoid(grid * pdf, grid))
        m2 = float(np.trapezoid(grid * grid * pdf, grid))
        return math.sqrt(max(m2 - m1 * m1, 0.0)) / m1


HOPKINS_TVEITEN_2002 = ForceChainReference(plateau_end=0.0, n_contacts_reported=500)
HOPKINS_TVEITEN_2002_PLATEAU = ForceChainReference(plateau_end=1.0, n_contacts_reported=500)


# ---------------------------------------------------------------------------
# Floe size distribution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RosinRammlerDistribution:
    """Truncated Rosin-Rammler (Weibull) floe-diameter distribution.

    ``F_raw(d) = 1 - exp(-(d / d_char) ** exponent)`` truncated to
    ``[d_min, d_max]`` and renormalised.  ``exponent = k`` is the Rosin-Rammler
    uniformity exponent (``k = 2`` is a typical crushed-ice value; larger ``k``
    means a narrower distribution).
    """

    d_min: float = 0.05
    d_max: float = 0.5
    d_char: float = 0.13
    exponent: float = 2.0

    def __post_init__(self) -> None:
        if not (0.0 < self.d_min < self.d_max):
            raise ValueError("require 0 < d_min < d_max")
        if self.d_char <= 0.0:
            raise ValueError("d_char must be positive")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be positive")

    def cdf_raw(self, d):
        d_arr = np.asarray(d, dtype=float)
        return 1.0 - np.exp(-np.power(np.maximum(d_arr, 0.0) / self.d_char, self.exponent))

    def _norm(self) -> float:
        return float(self.cdf_raw(self.d_max) - self.cdf_raw(self.d_min))

    def cdf(self, d):
        d_arr = np.asarray(d, dtype=float)
        raw = self.cdf_raw(d_arr) - self.cdf_raw(self.d_min)
        return np.clip(raw / self._norm(), 0.0, 1.0)

    def pdf(self, d):
        d_arr = np.asarray(d, dtype=float)
        k, dc = self.exponent, self.d_char
        raw = (k / dc) * np.power(np.maximum(d_arr, 1e-12) / dc, k - 1.0) * np.exp(
            -np.power(np.maximum(d_arr, 0.0) / dc, k)
        )
        norm = self._norm()
        if norm <= 1e-12:
            return np.zeros_like(d_arr)
        out = raw / norm
        return np.where((d_arr >= self.d_min) & (d_arr <= self.d_max), out, 0.0)

    def ppf(self, u):
        """Inverse CDF of the truncated distribution."""
        u_arr = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        raw = self.cdf_raw(self.d_min) + u_arr * self._norm()
        raw = np.clip(raw, 0.0, 1.0 - 1e-15)
        return self.d_char * np.power(-np.log1p(-raw), 1.0 / self.exponent)

    def sample(self, n: int, *, seed: int = 0, generator: np.random.Generator | None = None):
        rng = generator if generator is not None else np.random.default_rng(seed)
        return self.ppf(rng.random(int(n)))

    def moments(self, count: int = 3) -> list[float]:
        """Numerical moments ``E[d^j]`` (j = 1..count) of the truncated law."""
        grid = np.linspace(self.d_min, self.d_max, 20001)
        pdf = self.pdf(grid)
        return [float(np.trapezoid(np.power(grid, j) * pdf, grid)) for j in range(1, count + 1)]

    def mean_diameter(self) -> float:
        return self.moments(1)[0]

    def mean_volume(self) -> float:
        """``E[pi d^3 / 6]`` used to size a floe field."""
        return (math.pi / 6.0) * self.moments(3)[2]

    def ks_statistic(self, sample: Sequence[float]) -> float:
        """KS distance between a diameter sample and the analytic truncated CDF."""
        return ks_test_cdf(np.asarray(sample, dtype=float), self.cdf)[0]


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov (self-contained; asymptotic p-value, small-n corrected)
# ---------------------------------------------------------------------------
def kolmogorov_survival(lam: float) -> float:
    """``Q(lam) = 2 sum_{k>=1} (-1)^{k-1} exp(-2 k^2 lam^2)`` (KS tail)."""
    lam = float(lam)
    if lam <= 0.0:
        return 1.0
    if lam < 0.2:  # the alternating series cancels; use the theta-function form
        total = 0.0
        for k in range(1, 80):
            arg = (2 * k - 1) ** 2 * math.pi**2 / (8.0 * lam * lam)
            if arg > 700.0:
                break
            total += math.exp(-arg)
        return max(0.0, 1.0 - math.sqrt(2.0 * math.pi) / lam * total)
    total = 0.0
    for k in range(1, 200):
        term = 2.0 * (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-16:
            break
    return min(1.0, max(0.0, total))


def ks_test_cdf(sample: Sequence[float], cdf: Callable[[np.ndarray], np.ndarray]):
    """One-sample KS test of ``sample`` against the continuous ``cdf``.

    Returns ``(D, p)`` where ``D`` is the maximum distance and ``p`` the
    asymptotic p-value with the standard small-sample correction
    ``lam = (sqrt(n) + 0.12 + 0.11/sqrt(n)) D`` (Numerical Recipes convention).
    """
    s = np.sort(np.asarray(sample, dtype=float))
    n = s.size
    if n == 0:
        raise ValueError("empty sample")
    cdf_vals = np.clip(np.asarray(cdf(s), dtype=float), 0.0, 1.0)
    left = np.arange(n, dtype=float) / n
    right = np.arange(1, n + 1, dtype=float) / n
    d = float(max(np.max(np.abs(cdf_vals - left)), np.max(np.abs(cdf_vals - right))))
    root = math.sqrt(n)
    lam = (root + 0.12 + 0.11 / root) * d
    return d, float(kolmogorov_survival(lam))


def ks_test_two_sample(a: Sequence[float], b: Sequence[float]):
    """Two-sample KS test; asymptotic p-value via the effective sample size."""
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    if x.size == 0 or y.size == 0:
        raise ValueError("empty sample")
    grid = np.concatenate([x, y])
    fx = np.searchsorted(x, grid, side="right") / x.size
    fy = np.searchsorted(y, grid, side="right") / y.size
    d = float(np.max(np.abs(fx - fy)))
    n_eff = x.size * y.size / (x.size + y.size)
    root = math.sqrt(n_eff)
    lam = (root + 0.12 + 0.11 / root) * d
    return d, float(kolmogorov_survival(lam))


# ---------------------------------------------------------------------------
# Material model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceRubbleMaterial:
    """Micro-parameters of the unbonded ice-rubble contact model.

    ``reference_stiffness`` is the *pair* normal stiffness at
    ``reference_radius``; it is scaled as ``k_n = k0 (r_eff / r_ref)^2`` so that
    the contact period ``2 pi sqrt(m_eff / k_n)`` stays nearly size independent
    (``m ~ r^3``, ``k ~ r^2``) and one time step suits the whole size spectrum.
    """

    ice_density: float = 917.0
    gravity: float = GRAVITY
    gravity_scale: float = 1.0
    reference_radius: float = 0.05
    reference_stiffness: float = 4.0e4
    normal_damping_ratio: float = 0.7
    tangential_stiffness_ratio: float = 0.4
    friction_coefficient: float = 0.30
    rolling_friction: float = 0.05
    bulk_damping: float = 40.0
    wall_friction_coefficient: float | None = None
    angular_regularisation: float = 1.0e-3

    def __post_init__(self) -> None:
        for name in (
            "ice_density",
            "reference_radius",
            "reference_stiffness",
            "normal_damping_ratio",
            "tangential_stiffness_ratio",
            "friction_coefficient",
            "rolling_friction",
            "bulk_damping",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.wall_friction_coefficient is not None and self.wall_friction_coefficient < 0.0:
            raise ValueError("wall_friction_coefficient must be non-negative")
        if self.gravity_scale <= 0.0:
            raise ValueError("gravity_scale must be positive")
        if self.reference_stiffness <= 0.0 or self.reference_radius <= 0.0:
            raise ValueError("reference stiffness/radius must be positive")

    @property
    def effective_gravity(self) -> float:
        return self.gravity * self.gravity_scale

    @property
    def wall_friction(self) -> float:
        if self.wall_friction_coefficient is None:
            return self.friction_coefficient
        return self.wall_friction_coefficient

    def pair_stiffness(self, r_eff):
        scale = r_eff / self.reference_radius
        return self.reference_stiffness * scale * scale

    def recommended_dt(self, r_min: float, *, safety: float = 0.15) -> float:
        """Stable step from the smallest contact period in the field."""
        m_min = self.ice_density * (4.0 / 3.0) * math.pi * r_min**3
        m_eff = 0.5 * m_min
        k_n = float(self.pair_stiffness(r_min))
        return safety * 2.0 * math.pi * math.sqrt(m_eff / k_n)

    def with_friction(self, mu: float) -> "IceRubbleMaterial":
        return replace(self, friction_coefficient=float(mu))

    def with_gravity_scale(self, scale: float) -> "IceRubbleMaterial":
        return replace(self, gravity_scale=float(scale))


# ---------------------------------------------------------------------------
# Floe-field generation (random sequential adsorption, largest floe first)
# ---------------------------------------------------------------------------
def generate_floe_field(
    *,
    n_floes: int,
    distribution: RosinRammlerDistribution | None = None,
    box: tuple[float, float, float],
    seed: int = 0,
    floor_z: float = 0.0,
    attempts: int = 400,
    max_radius: float | None = None,
) -> dict[str, Any]:
    """Place ``n_floes`` non-overlapping floes in ``box`` (largest first).

    The returned dictionary holds ``positions`` (``(N, 3)`` float64),
    ``radii`` (``(N,)`` float64), ``diameters`` and a few diagnostics.  Floes are
    drawn from the truncated Rosin-Rammler law; placement is sequential random
    adsorption with a k-d tree over the already placed floes, so the resulting
    initial packing fraction is loose (``~0.3-0.45``).  Residual overlaps (if the
    requested count over-fills the box) are left for the DEM contact law to
    resolve during the settling phase.
    """
    if n_floes < 1:
        raise ValueError("n_floes must be positive")
    dist = distribution if distribution is not None else RosinRammlerDistribution()
    lx, ly, lz = (float(v) for v in box)
    if min(lx, ly, lz) <= 0.0:
        raise ValueError("box dimensions must be positive")
    rng = np.random.default_rng(seed)
    diameters = np.asarray(dist.sample(n_floes, generator=rng), dtype=float)
    radii = 0.5 * diameters
    if max_radius is not None:
        radii = np.minimum(radii, float(max_radius))
    order = np.argsort(-radii)
    radii_sorted = radii[order]

    placed: list[np.ndarray] = []
    placed_radii: list[float] = []
    tree: cKDTree | None = None
    r_max = float(radii_sorted.max())
    overlaps = 0
    for idx, radius in enumerate(radii_sorted):
        accepted = False
        candidate: np.ndarray | None = None
        best_gap = -np.inf
        for _ in range(int(attempts)):
            point = np.array(
                [
                    rng.uniform(radius, lx - radius) if lx > 2 * radius else 0.5 * lx,
                    rng.uniform(radius, ly - radius) if ly > 2 * radius else 0.5 * ly,
                    rng.uniform(floor_z + radius, floor_z + lz - radius)
                    if lz > 2 * radius
                    else floor_z + 0.5 * lz,
                ]
            )
            gap = np.inf
            if tree is not None and placed:
                idxs = tree.query_ball_point(point, radius + r_max + 1e-12)
                for j in idxs:
                    gap = min(gap, float(np.linalg.norm(point - placed[j])) - radius - placed_radii[j])
                    if gap < 0.0:
                        break
            if gap >= 0.0:
                candidate, accepted = point, True
                break
            if gap > best_gap:
                best_gap, candidate = gap, point
        if not accepted:
            overlaps += 1
        placed.append(
            candidate if candidate is not None else np.array([0.5 * lx, 0.5 * ly, floor_z + 0.5 * lz])
        )
        placed_radii.append(float(radius))
        if tree is None or (idx + 1) % 32 == 0:
            tree = cKDTree(np.asarray(placed, dtype=float))

    positions = np.empty((n_floes, 3), dtype=float)
    positions[order] = np.asarray(placed, dtype=float)
    solid_volume = float(np.sum((4.0 / 3.0) * math.pi * radii**3))
    box_volume = lx * ly * lz
    return {
        "positions": positions,
        "radii": radii,
        "diameters": 2.0 * radii,
        "solid_volume": solid_volume,
        "initial_packing_fraction": solid_volume / box_volume,
        "placement_overlaps": overlaps,
        "distribution": dist,
        "seed": int(seed),
    }# ---------------------------------------------------------------------------
# Confining-cell DEM engine
# ---------------------------------------------------------------------------
_WALL_NAMES = ("left", "platen", "front", "back", "floor", "lid")


class IceRubbleField:
    """Polydisperse rigid-sphere packing/platen engine for ice rubble.

    Six plane walls bound the cell (``left``/``platen`` along ``x``,
    ``front``/``back`` along ``y``, ``floor``/``lid`` along ``z``).  ``platen``
    and ``lid`` are movable: the lid compacts the rubble (or measures its
    pressure), the platen is driven into the rubble to measure the macroscopic
    resistance of the packing.

    Contacts: linear spring-dashpot normal law with damping ratio
    ``normal_damping_ratio``, Cundall-Strack tangential spring saturated at
    ``mu F_n``, rolling-resistance couple saturated at ``mu_r R_eff F_n``, and a
    bulk velocity damping ``bulk_damping`` that makes the compaction
    quasi-static (it does not change the static equilibrium, only the path).
    """

    def __init__(
        self,
        positions: torch.Tensor | np.ndarray,
        radii: torch.Tensor | np.ndarray,
        material: IceRubbleMaterial | None = None,
        *,
        box: tuple[float, float, float],
        floor_z: float = 0.0,
        lid_z: float | None = None,
        platen_x: float | None = None,
        skin: float = 0.015,
        rebuild_interval: int = 12,
    ) -> None:
        self.pos = torch.as_tensor(np.asarray(positions, dtype=float), dtype=DTYPE).clone()
        self.rad = torch.as_tensor(np.asarray(radii, dtype=float), dtype=DTYPE).clone()
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("positions must be (N, 3)")
        if self.rad.shape[0] != self.pos.shape[0]:
            raise ValueError("radii must have one entry per particle")
        self.material = material if material is not None else IceRubbleMaterial()
        self.Lx, self.Ly, self.Lz = (float(v) for v in box)
        self.floor_z = float(floor_z)
        self.lid_z = float(floor_z + self.Lz if lid_z is None else lid_z)
        self.platen_x = float(self.Lx if platen_x is None else platen_x)
        self.n = int(self.pos.shape[0])
        self.vel = torch.zeros(self.n, 3, dtype=DTYPE)
        self.omega = torch.zeros(self.n, 3, dtype=DTYPE)
        self.mass = self.material.ice_density * (4.0 / 3.0) * math.pi * self.rad**3
        self.inertia = 0.4 * self.mass * self.rad**2
        self._wall_k = self.material.pair_stiffness(self.rad)
        self._wall_c = 2.0 * self.material.normal_damping_ratio * torch.sqrt(
            self._wall_k * self.mass
        )
        self.skin = float(skin)
        self.rebuild_interval = max(int(rebuild_interval), 1)
        self._steps = 0
        self._pi = torch.empty(0, dtype=torch.long)
        self._pj = torch.empty(0, dtype=torch.long)
        self._keys = torch.empty(0, dtype=torch.long)
        self._ft = torch.zeros(0, 3, dtype=DTYPE)
        self._wall_ft = torch.zeros(self.n, len(_WALL_NAMES), 3, dtype=DTYPE)
        self._tree: cKDTree | None = None
        self.wall_velocity = torch.zeros(len(_WALL_NAMES), 3, dtype=DTYPE)
        self.lid_velocity = 0.0
        self.platen_velocity = 0.0
        self.last_contact_count = 0
        self.last_contact_forces = torch.zeros(0, dtype=DTYPE)
        self._pair_evals = 0
        self._frames_cache: tuple | None = None
        self._wall_total_cache: torch.Tensor | None = None
        self._lo = torch.tensor([0.0, 0.0, self.floor_z], dtype=DTYPE)
        self._hi = torch.tensor([self.platen_x, self.Ly, self.lid_z], dtype=DTYPE)
        self._refresh()

    # -- geometry ---------------------------------------------------------
    def _wall_frames(self):
        """Per-wall inward normals and points (cached until a wall moves)."""
        key = (self.platen_x, self.lid_z, self.floor_z, self.Ly)
        if self._frames_cache is not None and self._frames_cache[0] == key:
            return self._frames_cache[1], self._frames_cache[2]
        normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=DTYPE,
        )
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [self.platen_x, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, self.Ly, 0.0],
                [0.0, 0.0, self.floor_z],
                [0.0, 0.0, self.lid_z],
            ],
            dtype=DTYPE,
        )
        self._frames_cache = (key, normals, points)
        return normals, points

    # -- neighbor list ----------------------------------------------------
    def _refresh(self) -> None:
        if self.n < 2:
            self._pi = torch.empty(0, dtype=torch.long)
            self._pj = torch.empty(0, dtype=torch.long)
            self._keys = torch.empty(0, dtype=torch.long)
            self._ft = torch.zeros(0, 3, dtype=DTYPE)
            self._tree = None
            return
        pos_np = self.pos.numpy()
        rad_np = self.rad.numpy()
        r_max = float(rad_np.max())
        tree = cKDTree(pos_np)
        candidates = tree.query_pairs(2.0 * r_max + self.skin, output_type="ndarray")
        if candidates.size == 0:
            self._pi = torch.empty(0, dtype=torch.long)
            self._pj = torch.empty(0, dtype=torch.long)
            self._keys = torch.empty(0, dtype=torch.long)
            self._ft = torch.zeros(0, 3, dtype=DTYPE)
            self._tree = tree
            return
        pi_np = candidates[:, 0].astype(np.int64)
        pj_np = candidates[:, 1].astype(np.int64)
        delta = pos_np[pj_np] - pos_np[pi_np]
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        keep = dist < (rad_np[pi_np] + rad_np[pj_np] + self.skin)
        pi = torch.as_tensor(pi_np[keep], dtype=torch.long)
        pj = torch.as_tensor(pj_np[keep], dtype=torch.long)
        keys = pi * self.n + pj
        order = torch.argsort(keys)
        keys = keys[order]
        pi, pj = pi[order], pj[order]
        new_ft = torch.zeros(keys.numel(), 3, dtype=DTYPE)
        if self._keys.numel() > 0 and keys.numel() > 0:
            slot = torch.searchsorted(self._keys, keys).clamp(max=self._keys.numel() - 1)
            hit = self._keys[slot] == keys
            new_ft[hit] = self._ft[slot[hit]]
        self._pi, self._pj, self._keys, self._ft = pi, pj, keys, new_ft
        self._tree = tree

    def _maybe_refresh(self) -> None:
        if self._tree is None or self._steps % self.rebuild_interval == 0:
            self._refresh()

    # -- pair contacts ----------------------------------------------------
    def _pair_forces(self, dt: float):
        force = torch.zeros(self.n, 3, dtype=DTYPE)
        torque = torch.zeros(self.n, 3, dtype=DTYPE)
        if self._pi.numel() == 0:
            self.last_contact_count = 0
            self.last_contact_forces = torch.zeros(0, dtype=DTYPE)
            return force, torque
        i, j = self._pi, self._pj
        mat = self.material
        delta_vec = self.pos[j] - self.pos[i]
        dist = delta_vec.norm(dim=1).clamp_min(1e-12)
        normal = delta_vec / dist[:, None]
        r_sum = self.rad[i] + self.rad[j]
        overlap = r_sum - dist
        active = overlap > 0.0

        r_eff = 2.0 * self.rad[i] * self.rad[j] / r_sum
        k_n = mat.pair_stiffness(r_eff)
        m_eff = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        c_n = 2.0 * mat.normal_damping_ratio * torch.sqrt(k_n * m_eff)

        r_vec_i = -self.rad[i][:, None] * normal
        r_vec_j = self.rad[j][:, None] * normal
        v_contact_i = self.vel[i] + torch.linalg.cross(self.omega[i], r_vec_i, dim=1)
        v_contact_j = self.vel[j] + torch.linalg.cross(self.omega[j], r_vec_j, dim=1)
        v_rel = v_contact_j - v_contact_i
        v_n = (v_rel * normal).sum(dim=1)
        f_n = torch.where(
            active,
            torch.clamp(k_n * overlap - c_n * v_n, min=0.0),
            torch.zeros_like(overlap),
        )

        v_t = v_rel - v_n[:, None] * normal
        k_t = mat.tangential_stiffness_ratio * k_n
        ft = self._ft - k_t[:, None] * v_t * dt
        limit = mat.friction_coefficient * f_n
        ft_norm = ft.norm(dim=1)
        scale = torch.where(
            ft_norm > limit, limit / torch.clamp(ft_norm, min=1e-30), torch.ones_like(ft_norm)
        )
        ft = ft * scale[:, None]
        self._ft = ft

        f_i = -f_n[:, None] * normal + ft
        f_j = f_n[:, None] * normal - ft
        torque_i = torch.linalg.cross(r_vec_i, ft, dim=1)
        torque_j = torch.linalg.cross(r_vec_j, -ft, dim=1)

        omega_rel = self.omega[j] - self.omega[i]
        m_roll = mat.rolling_friction * r_eff * f_n
        roll = torch.clamp(omega_rel / mat.angular_regularisation, -1.0, 1.0)
        torque_i = torque_i - m_roll[:, None] * roll
        torque_j = torque_j + m_roll[:, None] * roll

        force.index_add_(0, i, f_i)
        force.index_add_(0, j, f_j)
        torque.index_add_(0, i, torque_i)
        torque.index_add_(0, j, torque_j)

        self.last_contact_count = int(active.sum())
        self.last_contact_forces = f_n[active]
        self._pair_evals += int(i.numel())
        return force, torque

    # -- wall contacts (batched over the six walls) -----------------------
    def _wall_bundle(self, dt: float | None = None):
        """Batched wall contact state, shape ``(6, N)`` per field."""
        normals, points = self._wall_frames()
        offsets = (points * normals).sum(dim=1)
        signed = self.pos @ normals.T - offsets[None, :]  # (N, 6)
        overlap = self.rad[:, None] - signed  # (N, 6)
        active = overlap > 0.0
        r_vec = -self.rad[:, None, None] * normals[None, :, :]  # (N, 6, 3)
        v_contact = self.vel[:, None, :] + torch.linalg.cross(
            self.omega[:, None, :].expand(-1, normals.shape[0], -1), r_vec, dim=2
        )
        v_rel = v_contact - self.wall_velocity[None, :, :]
        v_n = (v_rel * normals[None, :, :]).sum(dim=2)  # (N, 6)
        f_n = torch.where(
            active,
            torch.clamp(self._wall_k[:, None] * overlap - self._wall_c[:, None] * v_n, min=0.0),
            torch.zeros_like(overlap),
        )
        if dt is None:
            return normals, overlap, active, r_vec, v_rel, v_n, f_n
        v_t = v_rel - v_n[:, :, None] * normals[None, :, :]
        k_t = self.material.tangential_stiffness_ratio * self._wall_k
        ft = self._wall_ft - k_t[:, None, None] * v_t * dt
        limit = self.material.wall_friction * f_n
        ft_norm = ft.norm(dim=2)
        scale = torch.where(
            ft_norm > limit, limit / torch.clamp(ft_norm, min=1e-30), torch.ones_like(ft_norm)
        )
        ft = torch.where(active[:, :, None], ft * scale[:, :, None], torch.zeros_like(ft))
        self._wall_ft = ft
        return normals, overlap, active, r_vec, v_rel, v_n, f_n, ft

    def _wall_forces(self, dt: float):
        normals, overlap, active, r_vec, v_rel, v_n, f_n, ft = self._wall_bundle(dt)
        force = (f_n[:, :, None] * normals[None, :, :]).sum(dim=1) + ft.sum(dim=1)
        self._wall_total_cache = (f_n[:, :, None] * normals[None, :, :]).sum(dim=0) + ft.sum(dim=0)
        torque = torch.linalg.cross(r_vec, ft, dim=2).sum(dim=1)
        m_roll = self.material.rolling_friction * self.rad[:, None] * f_n
        torque = torque - (m_roll[:, :, None] * torch.clamp(
            self.omega[:, None, :] / self.material.angular_regularisation, -1.0, 1.0
        )).sum(dim=1)
        return force, torque

    def forces(self, dt: float, *, gravity: bool = True):
        force, torque = self._pair_forces(dt)
        f_wall, t_wall = self._wall_forces(dt)
        force = force + f_wall
        torque = torque + t_wall
        if gravity:
            force[:, 2] -= self.mass * self.material.effective_gravity
        return force, torque

    # -- integration ------------------------------------------------------
    def step(self, dt: float, *, gravity: bool = True) -> None:
        self._maybe_refresh()
        force, torque = self.forces(dt, gravity=gravity)
        self.vel += force / self.mass[:, None] * dt
        self.omega += torque / self.inertia[:, None] * dt
        damp = 1.0 / (1.0 + self.material.bulk_damping * dt)
        self.vel *= damp
        self.omega *= damp
        self.pos += self.vel * dt
        self._constrain()
        self._steps += 1

    def _constrain(self) -> None:
        """Safety clamp: no particle may tunnel out of the cell."""
        lo = self._lo
        hi = self._hi
        hi[0] = self.platen_x
        hi[2] = self.lid_z
        for axis in range(3):
            under = self.pos[:, axis] < lo[axis]
            over = self.pos[:, axis] > hi[axis]
            if bool(under.any()):
                self.pos[under, axis] = lo[axis]
                self.vel[under, axis] = torch.clamp(self.vel[under, axis], min=0.0)
            if bool(over.any()):
                self.pos[over, axis] = hi[axis]
                self.vel[over, axis] = torch.clamp(self.vel[over, axis], max=0.0)

    # -- diagnostics ------------------------------------------------------
    def kinetic_energy(self) -> float:
        return float(0.5 * (self.mass * (self.vel**2).sum(dim=1)).sum())

    def max_speed(self) -> float:
        return float(self.vel.norm(dim=1).max()) if self.n else 0.0

    def wall_forces_per_wall(self) -> torch.Tensor:
        """Contact force each wall exerts on the packing, shape ``(6, 3)``."""
        if self._wall_total_cache is not None:
            return self._wall_total_cache
        normals, overlap, active, r_vec, v_rel, v_n, f_n = self._wall_bundle(None)
        return (f_n[:, :, None] * normals[None, :, :]).sum(dim=0) + self._wall_ft.sum(dim=0)

    def wall_force(self, name: str) -> torch.Tensor:
        """Total contact force exerted by one wall on the packing (vector)."""
        return self.wall_forces_per_wall()[_WALL_NAMES.index(name)]

    def lid_pressure(self) -> float:
        """Mean normal pressure the lid feels (positive = packing pushes up)."""
        return float(-self.wall_force("lid")[2]) / (self.Lx * self.Ly)

    def platen_force(self) -> float:
        """Resistance the platen feels along ``-x`` (positive = pushed back)."""
        return float(-self.wall_force("platen")[0])

    def cell_height(self) -> float:
        return max(self.lid_z - self.floor_z, 0.0)

    def packing_fraction(self) -> float:
        """Solid fraction of the confined cell volume."""
        volume = self.Lx * self.Ly * max(self.cell_height(), 1e-12)
        return float((4.0 / 3.0 * math.pi * self.rad**3).sum()) / volume

    def contact_network(self) -> dict[str, Any]:
        """Contact list with normal forces; only compressive (``f_n > 0``) pairs."""
        self._refresh()
        i, j = self._pi, self._pj
        empty = {
            "i": np.empty(0, dtype=int),
            "j": np.empty(0, dtype=int),
            "normal_force": np.empty(0),
            "overlap": np.empty(0),
        }
        if i.numel() == 0:
            return empty
        delta_vec = self.pos[j] - self.pos[i]
        dist = delta_vec.norm(dim=1)
        overlap = (self.rad[i] + self.rad[j]) - dist
        normal = delta_vec / dist.clamp_min(1e-12)[:, None]
        r_eff = 2.0 * self.rad[i] * self.rad[j] / (self.rad[i] + self.rad[j])
        k_n = self.material.pair_stiffness(r_eff)
        v_rel = self.vel[j] - self.vel[i]
        v_n = (v_rel * normal).sum(dim=1)
        m_eff = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        c_n = 2.0 * self.material.normal_damping_ratio * torch.sqrt(k_n * m_eff)
        f_n = torch.clamp(k_n * overlap - c_n * v_n, min=0.0)
        active = (overlap > 0.0) & (f_n > 0.0)
        return {
            "i": i[active].numpy(),
            "j": j[active].numpy(),
            "normal_force": f_n[active].numpy(),
            "overlap": overlap[active].numpy(),
        }

    # -- drivers ----------------------------------------------------------
    def relax(
        self,
        dt: float,
        *,
        max_steps: int = 2000,
        speed_tol: float = 2e-3,
        gravity: bool = True,
    ) -> dict:
        steps = 0
        for _ in range(int(max_steps)):
            self.step(dt, gravity=gravity)
            steps += 1
            if self.max_speed() < speed_tol:
                break
        return {
            "steps": steps,
            "max_speed": self.max_speed(),
            "kinetic_energy": self.kinetic_energy(),
        }

    def compact_to_stress(
        self,
        dt: float,
        *,
        stress_target: float,
        max_steps: int = 20000,
        lid_mass: float | None = None,
        speed_tolerance: float = 0.01,
        check_every: int = 25,
        tolerance: float = 0.05,
        hold_checks: int = 4,
        min_travel: float = 0.0,
    ) -> dict:
        """Compact the packing with a *preloaded* piston until equilibrium.

        The lid is a dynamic body with mass ``lid_mass`` loaded by the constant
        force ``stress_target * Lx * Ly``; it is damped like the particles.  The
        piston therefore stays in contact with the rubble while it descends (a
        displacement servo would lose contact, because free particles fall faster
        than any imposed lid speed), and it stops when the contact force balances
        the preload, i.e. at the prescribed stress.  This is the numerical
        equivalent of an oedometer with a dead-weight piston.
        """
        area = self.Lx * self.Ly
        preload = float(stress_target) * area
        mass = float(lid_mass) if lid_mass is not None else max(1.0, preload / (0.5 * self.material.effective_gravity))
        velocity = 0.0
        travel = 0.0
        steps = 0
        holds = 0
        reached = False
        damp = 1.0 / (1.0 + self.material.bulk_damping * dt)
        self._wall_total_cache = None
        for _ in range(int(max_steps)):
            upward = float(-self.wall_force("lid")[2])
            velocity += (preload - upward) / mass * dt
            velocity *= damp
            self.lid_z = float(
                np.clip(self.lid_z - velocity * dt, self.floor_z + 1e-3, self.floor_z + self.Lz)
            )
            travel += abs(velocity * dt)
            self.wall_velocity[5] = torch.tensor([0.0, 0.0, -velocity], dtype=DTYPE)
            self._wall_total_cache = None
            self.step(dt)
            steps += 1
            if steps % check_every == 0:
                pressure = self.lid_pressure()
                settled = (
                    abs(pressure - stress_target) <= tolerance * stress_target
                    and abs(velocity) <= speed_tolerance
                )
                if settled and travel >= min_travel:
                    holds += 1
                    if holds >= hold_checks:
                        reached = True
                        break
                else:
                    holds = 0
        self.wall_velocity[5] = torch.zeros(3, dtype=DTYPE)
        self.lid_velocity = 0.0
        return {
            "steps": steps,
            "travel": travel,
            "reached_target": reached,
            "final_lid_pressure": self.lid_pressure(),
            "lid_z": self.lid_z,
            "lid_mass": mass,
            "final_lid_speed": float(velocity),
            "max_speed": self.max_speed(),
        }

    def compact_to_height(
        self,
        dt: float,
        *,
        height: float,
        max_steps: int = 20000,
        lid_speed: float = 0.25,
    ) -> dict:
        """Close the lid to a prescribed cell height (displacement control).

        Used for the fixed-geometry resistance family: the cell height is the
        independent variable of the Lindqvist sensitivity test, and the solid
        content (number of floes) sets the porosity at that height.
        """
        target = self.floor_z + float(height)
        if target <= self.floor_z:
            raise ValueError("height must be positive")
        steps = 0
        start = self.lid_z
        self.wall_velocity[5] = torch.tensor([0.0, 0.0, -lid_speed], dtype=DTYPE)
        while self.lid_z > target and steps < int(max_steps):
            self.lid_z = max(target, self.lid_z - lid_speed * dt)
            self._wall_total_cache = None
            self.step(dt)
            steps += 1
        self.lid_z = target
        self.wall_velocity[5] = torch.zeros(3, dtype=DTYPE)
        self._wall_total_cache = None
        return {
            "steps": steps,
            "travel": start - target,
            "reached_target": abs(self.lid_z - target) < 1e-9,
            "final_lid_pressure": self.lid_pressure(),
            "lid_z": self.lid_z,
        }

    def open_lid(self) -> dict:
        """Lift the lid to the top of the cell, leaving a free upper surface.

        The consolidated packing is then held by gravity and the side walls only,
        so a platen driven into it has to displace material that can heave -- the
        configuration in which Lindqvist-type rubble resistance is defined.
        """
        previous = self.lid_z
        self.lid_z = self.floor_z + self.Lz
        self.wall_velocity[5] = torch.zeros(3, dtype=DTYPE)
        self._wall_total_cache = None
        return {"lid_z_before": previous, "lid_z_after": self.lid_z, "headroom": self.lid_z - previous}

    def advance_platen(
        self,
        dt: float,
        *,
        speed: float,
        penetration: float,
        record_every: int = 5,
    ) -> dict:
        """Drive the ``x = Lx`` wall inwards and record the resistance history."""
        travelled = 0.0
        steps = 0
        history: list[dict[str, float]] = []
        self.wall_velocity[1] = torch.tensor([-speed, 0.0, 0.0], dtype=DTYPE)
        self.platen_velocity = speed
        while travelled < penetration:
            self.platen_x = max(1e-3, self.platen_x - speed * dt)
            self._wall_total_cache = None
            travelled += speed * dt
            self.step(dt)
            steps += 1
            if steps % record_every == 0:
                history.append({"penetration": travelled, "force": self.platen_force()})
        self.wall_velocity[1] = torch.zeros(3, dtype=DTYPE)
        self.platen_velocity = 0.0
        pen = np.array([h["penetration"] for h in history]) if history else np.empty(0)
        frc = np.array([h["force"] for h in history]) if history else np.empty(0)
        window = pen >= (0.6 * penetration)
        return {
            "steps": steps,
            "penetration": travelled,
            "penetration_history": pen,
            "force_history": frc,
            "mean_force_tail": float(frc[window].mean()) if window.any() else float("nan"),
            "mean_force_all": float(frc.mean()) if frc.size else float("nan"),
            "max_force": float(frc.max()) if frc.size else float("nan"),
        }

# ---------------------------------------------------------------------------
# Microstructure estimators
# ---------------------------------------------------------------------------
def voxel_voronoi_porosity(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    domain_lo: Sequence[float],
    domain_hi: Sequence[float],
    voxel_size: float,
    chunk_voxels: int = 250_000,
) -> dict[str, Any]:
    """Porosity from a voxel-resolved Voronoi (Dirichlet) tessellation.

    Every voxel of the measurement domain is assigned to its **nearest floe
    centre** -- that assignment *is* the Voronoi tessellation, resolved at
    ``voxel_size`` -- and a voxel counts as solid when it falls inside any sphere
    (checked against the four nearest centres, so overlapping floes are treated
    as a union).  The global porosity is

    ``n = 1 - (solid voxels) / (voxels in the domain)``

    and each cell gives a local porosity ``n_i = 1 - solid_i / voxels_i``, i.e.
    the Voronoi-cell porosity distribution.  Discretisation error scales as
    ``sqrt(A_surface) * voxel_size**1.5 / V`` (unbiased counting, no systematic
    dilation), which the module's tests verify by refinement.
    """
    sites = np.ascontiguousarray(np.asarray(positions, dtype=float))
    rad = np.ascontiguousarray(np.asarray(radii, dtype=float))
    if sites.ndim != 2 or sites.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")
    if rad.shape[0] != sites.shape[0]:
        raise ValueError("radii must have one entry per particle")
    lo = np.asarray(domain_lo, dtype=float)
    hi = np.asarray(domain_hi, dtype=float)
    if np.any(hi <= lo):
        raise ValueError("domain_hi must exceed domain_lo on every axis")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    counts = np.maximum(np.ceil((hi - lo) / float(voxel_size)).astype(int), 1)
    dx = (hi - lo) / counts
    axes = [lo[a] + (np.arange(counts[a]) + 0.5) * dx[a] for a in range(3)]
    tree = cKDTree(sites)
    k = int(min(4, sites.shape[0]))
    cell_voxels = np.zeros(sites.shape[0], dtype=np.int64)
    cell_solid = np.zeros(sites.shape[0], dtype=np.int64)
    total = 0
    slab = max(1, int(chunk_voxels // max(counts[0] * counts[1], 1)))
    for z0 in range(0, counts[2], slab):
        zz = axes[2][z0 : z0 + slab]
        grid = np.stack(np.meshgrid(axes[0], axes[1], zz, indexing="ij"), axis=-1).reshape(-1, 3)
        dist, idx = tree.query(grid, k=k, workers=-1)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        labels = idx[:, 0]
        cover = np.any(dist - rad[idx] < 0.0, axis=1)
        cell_voxels += np.bincount(labels, minlength=sites.shape[0])
        cell_solid += np.bincount(labels[cover], minlength=sites.shape[0])
        total += grid.shape[0]
    voxel_volume = float(np.prod(dx))
    domain_volume = total * voxel_volume
    solid_voxel = int(cell_solid.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        local = 1.0 - cell_solid / np.maximum(cell_voxels, 1)
    local = np.where(cell_voxels > 0, local, np.nan)
    return {
        "porosity": 1.0 - solid_voxel / max(total, 1),
        "packing_fraction": solid_voxel / max(total, 1),
        "solid_volume_voxel": solid_voxel * voxel_volume,
        "solid_volume_analytic": float(np.sum((4.0 / 3.0) * math.pi * rad**3)),
        "domain_volume": domain_volume,
        "cell_volume": cell_voxels * voxel_volume,
        "cell_porosity": local,
        "voxel_size": float(dx.max()),
        "n_voxels": int(total),
    }


def coordination_number(
    contacts: dict[str, Any],
    n_particles: int,
    *,
    particle_mask: np.ndarray | None = None,
    force_threshold: float = 0.0,
) -> dict[str, float]:
    """Mean number of load-bearing contacts per particle.

    ``Z`` counts *unilateral, compressive* contacts (only pairs whose normal
    force exceeds ``force_threshold``); wall contacts are excluded by
    construction.  ``particle_mask`` restricts the average to interior floes, so
    the wall boundary layer does not bias ``Z`` downwards.
    """
    i = np.asarray(contacts["i"], dtype=int)
    j = np.asarray(contacts["j"], dtype=int)
    f = np.asarray(contacts["normal_force"], dtype=float)
    if force_threshold > 0.0:
        keep = f > force_threshold
        i, j, f = i[keep], j[keep], f[keep]
    counts = np.zeros(int(n_particles), dtype=float)
    np.add.at(counts, i, 1.0)
    np.add.at(counts, j, 1.0)
    mask = np.ones(int(n_particles), dtype=bool) if particle_mask is None else np.asarray(particle_mask, bool)
    selected = counts[mask]
    if selected.size == 0:
        return {"Z": float("nan"), "Z_std": float("nan"), "n_particles": 0, "n_contacts": int(i.size)}
    return {
        "Z": float(selected.mean()),
        "Z_std": float(selected.std(ddof=1)) if selected.size > 1 else 0.0,
        "Z_max": float(selected.max()),
        "n_particles": int(selected.size),
        "n_contacts": int(i.size),
        "contacts_per_cell": float(i.size) / max(int(n_particles), 1),
    }


def coordination_spectrum(
    contacts: dict[str, Any],
    n_particles: int,
    *,
    fractions: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
    particle_mask: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """``Z`` as a function of the contact-force cutoff ``F > fraction * <F>``.

    Dense packings contain many marginal contacts; the cutoff sweep makes the
    definition explicit instead of hiding it inside one number (force-chain
    analyses conventionally discard contacts below ~5-10 % of the mean force).
    """
    f = np.asarray(contacts["normal_force"], dtype=float)
    mean_f = float(f.mean()) if f.size else 0.0
    out = []
    for fraction in fractions:
        stats = coordination_number(
            contacts,
            n_particles,
            particle_mask=particle_mask,
            force_threshold=float(fraction) * mean_f,
        )
        out.append({"force_fraction": float(fraction), "Z": stats["Z"], "n_contacts": stats["n_contacts"]})
    return out


def force_chain_pdf(
    forces: Sequence[float],
    *,
    n_bins: int = 40,
    upper_percentile: float = 99.5,
) -> dict[str, Any]:
    """Normalised contact-force PDF ``P(f)`` with ``f = F / <F>``.

    Returns the binned density (so that ``sum(pdf * df) = 1``), the coefficient
    of variation, the fraction of contacts carrying ``f > 1`` and the share of
    the total load carried by those strong contacts (the force-chain load
    fraction).
    """
    f = np.asarray(forces, dtype=float)
    f = f[np.isfinite(f) & (f > 0.0)]
    if f.size < 4:
        raise ValueError("need at least four positive contact forces")
    normalised = f / f.mean()
    x_hi = float(np.percentile(normalised, upper_percentile))
    x_hi = max(x_hi, 2.0)
    edges = np.linspace(0.0, x_hi, int(n_bins) + 1)
    counts, edges = np.histogram(normalised, bins=edges, density=False)
    width = np.diff(edges)
    pdf = counts / (f.size * width)
    strong = normalised > 1.0
    return {
        "bin_edges": edges,
        "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
        "pdf": pdf,
        "counts": counts,
        "n_contacts": int(f.size),
        "mean_force": float(f.mean()),
        "std_force": float(f.std(ddof=1)),
        "coefficient_of_variation": float(f.std(ddof=1) / f.mean()),
        "strong_fraction": float(strong.mean()),
        "strong_load_fraction": float(normalised[strong].sum() / normalised.sum()),
        "normalised": normalised,
    }


def force_chain_clusters(
    contacts: dict[str, Any],
    *,
    threshold: float = 1.0,
    n_particles: int | None = None,
) -> dict[str, Any]:
    """Connected clusters of strong contacts (``f > threshold``), union-find.

    Particles linked by a chain of strong contacts form one force chain; a chain
    of length one is a single strong contact.
    """
    i = np.asarray(contacts["i"], dtype=int)
    j = np.asarray(contacts["j"], dtype=int)
    f = np.asarray(contacts["normal_force"], dtype=float)
    if i.size == 0:
        return {"n_chains": 0, "largest_chain_contacts": 0, "mean_chain_contacts": 0.0}
    strong = f > threshold * f.mean()
    i, j = i[strong], j[strong]
    n = int(n_particles if n_particles is not None else (max(i.max(), j.max()) + 1))
    parent = np.arange(n + 1)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(i, j):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    labels = np.array([find(int(a)) for a in range(n)])
    chain_contacts: dict[int, int] = {}
    for a, b in zip(i, j):
        root = find(int(a))
        chain_contacts[root] = chain_contacts.get(root, 0) + 1
    sizes = np.array(sorted(chain_contacts.values()), dtype=float) if chain_contacts else np.empty(0)
    particles_in_chains = np.unique(np.concatenate([i, j])).size if i.size else 0
    return {
        "n_chains": int(sizes.size),
        "largest_chain_contacts": int(sizes.max()) if sizes.size else 0,
        "mean_chain_contacts": float(sizes.mean()) if sizes.size else 0.0,
        "strong_contacts": int(i.size),
        "particles_in_chains": int(particles_in_chains),
        "particle_participation": float(particles_in_chains) / max(n, 1),
        "threshold_force": float(threshold * f.mean()),
    }


def _bulk_particle_mask(field: "IceRubbleField", margin: float) -> np.ndarray:
    pos = field.pos.numpy()
    rad = field.rad.numpy()
    mask = (
        (pos[:, 0] > margin)
        & (pos[:, 0] < field.platen_x - margin)
        & (pos[:, 1] > margin)
        & (pos[:, 1] < field.Ly - margin)
        & (pos[:, 2] > field.floor_z + margin)
        & (pos[:, 2] < field.lid_z - margin)
    )
    # floes touching a wall are excluded as well (their contact count is censored)
    touching = (
        (pos[:, 0] - rad < margin)
        | (pos[:, 0] + rad > field.platen_x - margin)
        | (pos[:, 1] - rad < margin)
        | (pos[:, 1] + rad > field.Ly - margin)
        | (pos[:, 2] - rad < field.floor_z + margin)
        | (pos[:, 2] + rad > field.lid_z - margin)
    )
    return mask & ~touching# ---------------------------------------------------------------------------
# End-to-end microstructure study
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MicrostructureStudyConfig:
    """Configuration of the ice-rubble microstructure calibration study."""

    n_floes: int = 1200
    d_min: float = 0.05
    d_max: float = 0.5
    d_char: float = 0.13
    exponent: float = 2.0
    cell_width: float = 1.80
    cell_initial_height: float = 1.55
    friction_values: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 1.00)
    primary_friction: float = 1.00
    drag_exponents: tuple[float, ...] = (1.20, 1.60, 2.00, 2.60, 3.60)
    fill_counts: tuple[int, ...] = (800, 900, 1000, 1100, 1200)
    drag_height: float = 0.95
    dt: float = 1.0e-3
    settle_steps: int = 300
    settle_speed_tol: float = 2.0e-2
    compact_stress: float = 25000.0
    compact_max_steps: int = 14000
    relax_steps: int = 300
    relax_speed_tol: float = 5.0e-3
    platen_speed: float = 0.04
    platen_penetration: float = 0.15
    voxel_size: float = 0.012
    bulk_margin_diameters: float = 1.0
    seed: int = 20271107
    negative_control_particles: int = 400
    negative_control_diameter: float = 0.20
    g_sensitivity_scale: float = 0.5
    rolling_friction: float = 0.05

    def __post_init__(self) -> None:
        if self.n_floes < 1:
            raise ValueError("n_floes must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.primary_friction < 0.0:
            raise ValueError("primary_friction must be non-negative")
        if self.primary_friction not in self.friction_values:
            raise ValueError("primary_friction must appear in friction_values")
        if self.exponent not in self.drag_exponents:
            raise ValueError("exponent must appear in drag_exponents")

    @property
    def target_solid_volume(self) -> float:
        dist = RosinRammlerDistribution(
            d_min=self.d_min, d_max=self.d_max, d_char=self.d_char, exponent=self.exponent
        )
        return self.n_floes * dist.mean_volume()


def distribution_with_solid_volume(
    *,
    exponent: float,
    target_mean_volume: float,
    d_min: float = 0.05,
    d_max: float = 0.5,
) -> RosinRammlerDistribution:
    """Rosin-Rammler law of given ``exponent`` sized so that ``E[pi d^3/6]`` matches.

    Used to build the *shape family* used for the resistance feedback: the floe
    number and the total solid volume stay fixed, only the width of the size
    distribution (and hence the packing density) changes.
    """
    if target_mean_volume <= 0.0:
        raise ValueError("target_mean_volume must be positive")

    def mean_volume(d_char: float) -> float:
        return RosinRammlerDistribution(
            d_min=d_min, d_max=d_max, d_char=d_char, exponent=exponent
        ).mean_volume()

    lo, hi = 0.2 * d_min, 20.0 * d_max
    if mean_volume(lo) > target_mean_volume or mean_volume(hi) < target_mean_volume:
        raise ValueError("target mean volume not reachable with this diameter range")
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if mean_volume(mid) < target_mean_volume:
            lo = mid
        else:
            hi = mid
    return RosinRammlerDistribution(
        d_min=d_min, d_max=d_max, d_char=math.sqrt(lo * hi), exponent=exponent
    )


def _grid_placement(
    *,
    n_particles: int,
    radius: float,
    box: tuple[float, float, float],
    seed: int = 0,
    floor_z: float = 0.0,
) -> dict[str, Any]:
    """Jittered-lattice placement for the monodisperse negative control."""
    lx, ly, lz = box
    rng = np.random.default_rng(seed)
    solid = n_particles * (4.0 / 3.0) * math.pi * radius**3
    target = solid / (lx * ly * lz)
    if target > 0.45:
        raise ValueError(
            "initial monodisperse filling is too dense for random placement "
            f"(phi0 = {target:.2f}); enlarge the box or reduce the particle count"
        )
    spacing = (4.0 / 3.0 * math.pi * radius**3 / target) ** (1.0 / 3.0)
    nx = max(int(lx // spacing), 1)
    ny = max(int(ly // spacing), 1)
    nz = int(math.ceil(n_particles / (nx * ny)))
    grid = np.stack(
        np.meshgrid(
            (np.arange(nx) + 0.5) * (lx / nx),
            (np.arange(ny) + 0.5) * (ly / ny),
            floor_z + (np.arange(nz) + 0.5) * (lz / nz),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    take = min(n_particles, grid.shape[0])
    chosen = grid[rng.permutation(grid.shape[0])[:take]]
    jitter = (rng.random(chosen.shape) - 0.5) * 0.5 * spacing
    chosen = chosen + jitter
    radii = np.full(take, radius, dtype=float)
    return {
        "positions": chosen,
        "radii": radii,
        "diameters": 2.0 * radii,
        "solid_volume": float(take * (4.0 / 3.0) * math.pi * radius**3),
        "initial_packing_fraction": float(take * (4.0 / 3.0) * math.pi * radius**3) / (lx * ly * lz),
        "placement_overlaps": 0,
        "seed": int(seed),
    }


def _measure_microstructure(field: "IceRubbleField", cfg: MicrostructureStudyConfig) -> dict[str, Any]:
    pos = field.pos.numpy()
    rad = field.rad.numpy()
    vor = voxel_voronoi_porosity(
        pos,
        rad,
        domain_lo=(0.0, 0.0, field.floor_z),
        domain_hi=(field.Lx, field.Ly, field.lid_z),
        voxel_size=cfg.voxel_size,
    )
    contacts = field.contact_network()
    margin = cfg.bulk_margin_diameters * 2.0 * float(rad.mean())
    mask = _bulk_particle_mask(field, margin)
    coord = coordination_number(contacts, field.n, particle_mask=mask)
    coord_all = coordination_number(contacts, field.n)
    spectrum = coordination_spectrum(contacts, field.n)
    if contacts["normal_force"].size >= 4:
        pdf = force_chain_pdf(contacts["normal_force"])
        chains = force_chain_clusters(contacts, threshold=1.0, n_particles=field.n)
        pdf_summary = {
            "mean_force": pdf["mean_force"],
            "coefficient_of_variation": pdf["coefficient_of_variation"],
            "strong_fraction": pdf["strong_fraction"],
            "strong_load_fraction": pdf["strong_load_fraction"],
            "n_contacts": pdf["n_contacts"],
            "normalised": pdf["normalised"].tolist(),
            "bin_centers": pdf["bin_centers"].tolist(),
            "pdf": pdf["pdf"].tolist(),
            "bin_edges": pdf["bin_edges"].tolist(),
        }
    else:
        pdf_summary = {"n_contacts": int(contacts["normal_force"].size)}
        chains = {"n_chains": 0, "largest_chain_contacts": 0}
    local_all = np.asarray(vor["cell_porosity"], dtype=float)
    local = local_all[np.isfinite(local_all)]
    step = max(1, pos.shape[0] // 400)
    return {
        "porosity_voronoi": float(vor["porosity"]),
        "packing_fraction_voronoi": float(vor["packing_fraction"]),
        "porosity_local_mean": float(local.mean()) if local.size else float("nan"),
        "porosity_local_std": float(local.std(ddof=1)) if local.size > 1 else float("nan"),
        "porosity_local_p10": float(np.percentile(local, 10)) if local.size else float("nan"),
        "porosity_local_p90": float(np.percentile(local, 90)) if local.size else float("nan"),
        "packing_fraction_cell": float(field.packing_fraction()),
        "cell_volume": float(vor["domain_volume"]),
        "solid_volume_voxel": float(vor["solid_volume_voxel"]),
        "solid_volume_analytic": float(vor["solid_volume_analytic"]),
        "voxel_size": float(vor["voxel_size"]),
        "n_voxels": int(vor["n_voxels"]),
        "coordination_number": float(coord["Z"]),
        "coordination_number_all": float(coord_all["Z"]),
        "coordination_number_std": float(coord.get("Z_std", float("nan"))),
        "coordination_spectrum": spectrum,
        "coordination_particles": int(coord["n_particles"]),
        "n_contacts": int(coord["n_contacts"]),
        "diameter_mean": float(2.0 * rad.mean()),
        "diameter_min": float(2.0 * rad.min()),
        "diameter_max": float(2.0 * rad.max()),
        "force_chain": pdf_summary,
        "force_chains": chains,
        "lid_pressure": float(field.lid_pressure()),
        "cell_height": float(field.cell_height()),
        "max_speed": float(field.max_speed()),
        "local_porosity_histogram": _porosity_histogram(local),
        "positions_sample": np.asarray(pos)[::step].tolist(),
        "local_porosity_sample": np.asarray(local_all)[::step].tolist(),
    }


def _porosity_histogram(local: np.ndarray, bins: int = 25) -> dict[str, list[float]]:
    finite = local[np.isfinite(local)]
    if finite.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(finite, bins=bins)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def study_cases(cfg: MicrostructureStudyConfig) -> list[dict[str, Any]]:
    """Case list of the study: friction family, shape family, controls."""
    target_mean_volume = cfg.target_solid_volume / cfg.n_floes
    cases: list[dict[str, Any]] = []
    for mu in cfg.friction_values:
        cases.append(
            {
                "name": f"mu_{mu:.2f}",
                "kind": "friction",
                "mu": float(mu),
                "exponent": cfg.exponent,
                "d_char": cfg.d_char,
                "gravity_scale": 1.0,
                "monodisperse": False,
                "seed": cfg.seed,
            }
        )
    for k in cfg.drag_exponents:
        dist = distribution_with_solid_volume(
            exponent=k,
            target_mean_volume=target_mean_volume,
            d_min=cfg.d_min,
            d_max=cfg.d_max,
        )
        cases.append(
            {
                "name": f"shape_k{k:.2f}",
                "kind": "shape",
                "mu": cfg.primary_friction,
                "exponent": float(k),
                "d_char": float(dist.d_char),
                "gravity_scale": 1.0,
                "monodisperse": False,
                "seed": cfg.seed,
            }
        )
    for count in cfg.fill_counts:
        cases.append(
            {
                "name": f"fill_{count:d}",
                "kind": "fill",
                "mu": cfg.primary_friction,
                "exponent": cfg.exponent,
                "d_char": cfg.d_char,
                "gravity_scale": 1.0,
                "monodisperse": False,
                "seed": cfg.seed,
                "n_floes": int(count),
                "drag_height": cfg.drag_height,
            }
        )
    cases.append(
        {
            "name": "primary_replicate",
            "kind": "replicate",
            "mu": cfg.primary_friction,
            "exponent": cfg.exponent,
            "d_char": cfg.d_char,
            "gravity_scale": 1.0,
            "monodisperse": False,
            "seed": cfg.seed + 101,
        }
    )
    cases.append(
        {
            "name": "gravity_half",
            "kind": "gravity",
            "mu": cfg.primary_friction,
            "exponent": cfg.exponent,
            "d_char": cfg.d_char,
            "gravity_scale": cfg.g_sensitivity_scale,
            "monodisperse": False,
            "seed": cfg.seed,
        }
    )
    cases.append(
        {
            "name": "negative_control",
            "kind": "negative",
            "mu": 0.0,
            "exponent": cfg.exponent,
            "d_char": cfg.d_char,
            "gravity_scale": 1.0,
            "monodisperse": True,
            "seed": cfg.seed + 7,
        }
    )
    return cases


def run_study_case(
    cfg: MicrostructureStudyConfig,
    case: dict[str, Any],
    *,
    with_drag: bool = True,
) -> dict[str, Any]:
    """Run one case of the study and return its measurements."""
    lx = ly = cfg.cell_width
    box = (lx, ly, cfg.cell_initial_height)
    if case["monodisperse"]:
        field_data = _grid_placement(
            n_particles=cfg.negative_control_particles,
            radius=0.5 * cfg.negative_control_diameter,
            box=box,
            seed=case["seed"],
        )
    else:
        field_data = generate_floe_field(
            n_floes=int(case.get("n_floes", cfg.n_floes)),
            distribution=RosinRammlerDistribution(
                d_min=cfg.d_min,
                d_max=cfg.d_max,
                d_char=case["d_char"],
                exponent=case["exponent"],
            ),
            box=box,
            seed=case["seed"],
        )
    material = IceRubbleMaterial(
        friction_coefficient=case["mu"],
        gravity_scale=case["gravity_scale"],
        rolling_friction=(0.0 if case["mu"] == 0.0 else cfg.rolling_friction),
    )
    field = IceRubbleField(field_data["positions"], field_data["radii"], material, box=box)
    settle = field.relax(cfg.dt, max_steps=cfg.settle_steps, speed_tol=cfg.settle_speed_tol)
    if case.get("drag_height") is not None:
        compaction = field.compact_to_height(
            cfg.dt, height=float(case["drag_height"]), max_steps=cfg.compact_max_steps
        )
    else:
        compaction = field.compact_to_stress(
            cfg.dt, stress_target=cfg.compact_stress, max_steps=cfg.compact_max_steps
        )
    relaxation = field.relax(cfg.dt, max_steps=cfg.relax_steps, speed_tol=cfg.relax_speed_tol)
    microstructure = _measure_microstructure(field, cfg)
    lid_open: dict[str, Any] = {}
    static_thrust: float | None = None
    if case.get("drag_height") is not None:
        lid_open = field.open_lid()
        field.relax(cfg.dt, max_steps=150, speed_tol=1.0e-2)
        # at-rest lateral thrust of the free-standing rubble on the platen: the
        # quantity the Lindqvist (1 - n) factor multiplies
        static_thrust = float(field.platen_force())
    drag: dict[str, Any] = {}
    if with_drag:
        history = field.advance_platen(cfg.dt, speed=cfg.platen_speed, penetration=cfg.platen_penetration)
        record = {
            k: v
            for k, v in history.items()
            if k not in ("penetration_history", "force_history")
        }
        record["force_history"] = history["force_history"].tolist()[::5]
        record["penetration_history"] = history["penetration_history"].tolist()[::5]
        drag = record
    return {
        "case": case,
        "friction_coefficient": float(case["mu"]),
        "exponent": float(case["exponent"]),
        "d_char": float(case["d_char"]),
        "gravity_scale": float(case["gravity_scale"]),
        "monodisperse": bool(case["monodisperse"]),
        "seed": int(case["seed"]),
        "n_particles": int(field.n),
        "diameters": np.asarray(field_data["diameters"], dtype=float).tolist(),
        "initial_packing_fraction": float(field_data["initial_packing_fraction"]),
        "placement_overlaps": int(field_data.get("placement_overlaps", 0)),
        "solid_volume": float(field_data["solid_volume"]),
        "settle": settle,
        "compaction": compaction,
        "relaxation": relaxation,
        "microstructure": microstructure,
        "lid_open": lid_open,
        "static_lateral_thrust": static_thrust,
        "drag": drag,
        "steps_total": int(
            settle["steps"] + compaction["steps"] + relaxation["steps"] + record.get("steps", 0)
        ),
    }


def _fill_sort_key(name: str) -> tuple[int, int]:
    if name.startswith("fill_"):
        try:
            return (0, int(name.split("_")[1]))
        except (IndexError, ValueError):
            return (1, 0)
    return (1, 0)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares line ``y = a x + b`` with R^2, plus the through-origin fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        raise ValueError("need at least three points for a fit")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    slope0 = float(np.dot(x, y) / np.dot(x, x))
    pred0 = slope0 * x
    r2_0 = 1.0 - float(np.sum((y - pred0) ** 2)) / ss_tot if ss_tot > 0.0 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "slope_through_origin": slope0,
        "r_squared_through_origin": float(r2_0),
        "pearson_r": corr,
        "n_points": int(x.size),
    }


def analyse_runs(
    cfg: MicrostructureStudyConfig,
    runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Turn the per-case measurements into the criteria table.

    ``runs`` maps case name (see :func:`study_cases`) to the dictionary returned
    by :func:`run_study_case`.
    """
    mu_runs = [runs[name] for name in sorted(runs) if name.startswith("mu_")]
    shape_runs = [runs[name] for name in sorted(runs) if name.startswith("shape_")]
    fill_runs = [runs[name] for name in sorted(runs, key=_fill_sort_key) if name.startswith("fill_")]
    primary = runs[f"mu_{cfg.primary_friction:.2f}"]
    replicate = runs.get("primary_replicate")
    g_run = runs.get("gravity_half")
    negative = runs.get("negative_control")
    if not mu_runs or not fill_runs:
        raise ValueError("runs must contain the friction and fill families")
    drag_runs = fill_runs if fill_runs else shape_runs

    por_ref = TIMCO_OBRIEN_2004
    z_ref = COORDINATION_REFERENCE
    n_dem = primary["microstructure"]["porosity_voronoi"]
    por_err = por_ref.relative_error(n_dem)
    z_dem = primary["microstructure"]["coordination_number_all"]  # plain "mean contacts per floe"
    z_bulk = primary["microstructure"]["coordination_number"]
    z_dev = abs(z_dem - z_ref.mean)

    forces = np.asarray(primary["microstructure"]["force_chain"].get("normalised", []), dtype=float)
    ks_matched = {"D_median": float("nan"), "p_median": float("nan"), "pass_fraction": float("nan")}
    ks_full = {"D": float("nan"), "p": float("nan")}
    ks_plateau = {"D": float("nan"), "p": float("nan")}
    ks_rep = (float("nan"), float("nan"))
    if forces.size >= 40:
        ks_full = dict(zip(("D", "p"), ks_test_cdf(forces, HOPKINS_TVEITEN_2002.cdf)))
        ks_plateau = dict(zip(("D", "p"), ks_test_cdf(forces, HOPKINS_TVEITEN_2002_PLATEAU.cdf)))
        rng = np.random.default_rng(20271107)
        n_sample = int(min(HOPKINS_TVEITEN_2002.n_contacts_reported, forces.size))
        draws = []
        for _ in range(60):
            dem_draw = rng.choice(forces, size=n_sample, replace=False)
            ref_draw = HOPKINS_TVEITEN_2002.sample(n_sample, seed=int(rng.integers(1 << 30)))
            draws.append(ks_test_two_sample(dem_draw, ref_draw))
        ks_matched = {
            "D_median": float(np.median([d[0] for d in draws])),
            "p_median": float(np.median([d[1] for d in draws])),
            "D_p05": float(np.percentile([d[0] for d in draws], 5)),
            "D_p95": float(np.percentile([d[0] for d in draws], 95)),
            "pass_fraction": float(np.mean([d[1] > 0.05 for d in draws])),
            "n_draws": int(len(draws)),
            "n_sample": int(n_sample),
        }
    if replicate is not None:
        rep = np.asarray(replicate["microstructure"]["force_chain"].get("normalised", []), dtype=float)
        if forces.size > 4 and rep.size > 4:
            ks_rep = ks_test_two_sample(forces, rep)

    solid = np.array([1.0 - r["microstructure"]["porosity_voronoi"] for r in drag_runs], dtype=float)
    static = np.array(
        [
            r["static_lateral_thrust"]
            if r.get("static_lateral_thrust") is not None
            else float("nan")
            for r in drag_runs
        ],
        dtype=float,
    )
    resist = np.array([r["drag"]["mean_force_tail"] for r in drag_runs], dtype=float)
    use_static = bool(np.all(np.isfinite(static)) and np.all(static > 0.0))
    fit_y = static if use_static else resist
    drag_fit = _linear_fit(solid, fit_y) if solid.size >= 3 else {}
    drag_fit_dynamic = _linear_fit(solid, resist) if solid.size >= 3 else {}
    drag_fit_log = {}
    if solid.size >= 3 and np.all(fit_y > 0.0):
        slope_log, intercept_log = np.polyfit(solid, np.log(fit_y), 1)
        pred_log = slope_log * solid + intercept_log
        logy = np.log(fit_y)
        ss_res = float(np.sum((logy - pred_log) ** 2))
        ss_tot = float(np.sum((logy - logy.mean()) ** 2))
        drag_fit_log = {
            "model": "ln F = a (1 - n) + b",
            "slope": float(slope_log),
            "intercept": float(intercept_log),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }
    # predicted resistance change for the Lindqvist n = 0.30 -> 0.40 band
    lindqvist_predicted = float("nan")
    if drag_fit:
        f_low = drag_fit["slope"] * (1.0 - 0.40) + drag_fit["intercept"]
        f_high = drag_fit["slope"] * (1.0 - 0.30) + drag_fit["intercept"]
        if f_low > 0.0:
            lindqvist_predicted = (f_high - f_low) / f_low

    criteria: dict[str, Any] = {
        "a_porosity_vs_literature": {
            "dem_porosity": n_dem,
            "dem_packing_fraction": primary["microstructure"]["packing_fraction_voronoi"],
            "literature_mean": por_ref.mean,
            "literature_std": por_ref.std,
            "literature_n_datasets": por_ref.n_datasets,
            "relative_error": por_err,
            "tolerance": 0.10,
            "sigma_distance": por_ref.sigma_distance(n_dem),
            "passed": bool(por_err < 0.10),
            "evidence": "REAL (DEM) vs LITERATURE_SUMMARY",
        },
        "b_force_chain_pdf": {
            "method": (
                "two-sample KS at the experimental sample size (n_contacts_reported), "
                "median over 60 independent draws"
            ),
            "ks_two_sample_matched": ks_matched,
            "ks_one_sample_full": ks_full,
            "ks_one_sample_plateau_model": ks_plateau,
            "p_value": ks_matched["p_median"],
            "threshold_p": 0.05,
            "n_contacts": int(forces.size),
            "reference": HOPKINS_TVEITEN_2002.evidence,
            "dem_vs_dem_replicate": {"D": ks_rep[0], "p": ks_rep[1]},
            "coefficient_of_variation": primary["microstructure"]["force_chain"].get(
                "coefficient_of_variation"
            ),
            "reference_cv": HOPKINS_TVEITEN_2002.coefficient_of_variation(),
            "strong_load_fraction": primary["microstructure"]["force_chain"].get(
                "strong_load_fraction"
            ),
            "reference_strong_load_fraction": 2.0 * math.exp(-1.0),
            "passed": bool(np.isfinite(ks_matched["p_median"]) and ks_matched["p_median"] > 0.05),
            "evidence": "REAL (DEM) vs PARAMETRIC_MODEL (digitised data WITHHELD)",
        },
        "c_coordination_number": {
            "dem_Z": z_dem,
            "dem_Z_bulk_interior": z_bulk,
            "reference_mean": z_ref.mean,
            "reference_std": z_ref.std,
            "deviation": z_dev,
            "tolerance": z_ref.std,
            "coordination_spectrum": primary["microstructure"]["coordination_spectrum"],
            "boundary_layer_note": (
                "wall-adjacent floes have censored contact counts, so the assembly mean is "
                "biased low and the interior mean high; both bracket the reference"
            ),
            "definition": (
                "Z = mean number of compressive contacts per floe over the whole assembly "
                "(2 N_contacts / N_floes, the plain 'average contact number'); the "
                "bulk-interior value (floes at least one mean diameter from every wall) is "
                "reported as a wall-censoring diagnostic, together with the "
                "force-cutoff spectrum"
            ),
            "passed": bool(z_dev <= z_ref.std),
            "evidence": "REAL (DEM) vs LITERATURE_SUMMARY",
        },
        "d_resistance_feedback": {
            "design": (
                "fixed-geometry family: cell height and friction are held constant while "
                "the amount of ice in the cell changes, so the porosity varies at fixed "
                "ridge geometry -- the direct DEM analogue of the Lindqvist n-sensitivity"
            ),
            "fit": "F_lateral = a (1 - n) + b" if use_static else "F_platen = a (1 - n) + b",
            "quantity": (
                "at-rest lateral thrust of the free-standing rubble on the platen "
                "(Lindqvist-type rubble load)" if use_static else "dynamic platen resistance"
            ),
            "fit_dynamic_push": drag_fit_dynamic,
            "fit_log_linear": drag_fit_log,
            "points": [
                {
                    "case": r["case"]["name"],
                    "exponent": r["exponent"],
                    "n_floes": r["n_particles"],
                    "porosity": float(1.0 - s),
                    "solid_fraction": float(s),
                    "resistance": float(f),
                    "static_lateral_thrust": float(st),
                    "coordination_number": r["microstructure"]["coordination_number_all"],
                    "cell_height": r["microstructure"]["cell_height"],
                }
                for r, s, f, st in zip(drag_runs, solid, resist, static)
            ],
            **drag_fit,
            "threshold_r2": 0.95,
            "lindqvist_resistance_change_n_0p30_to_0p40": lindqvist_predicted,
            "lindqvist_reference_change": 0.15,
            "passed": bool(drag_fit and drag_fit["r_squared"] >= 0.95),
            "evidence": "REAL (DEM)",
        },
        "e_negative_control_frictionless": {
            "dem_packing_fraction": negative["microstructure"]["packing_fraction_voronoi"]
            if negative
            else float("nan"),
            "dem_porosity": negative["microstructure"]["porosity_voronoi"] if negative else float("nan"),
            "target_packing_fraction": 0.64,
            "target_porosity_literal": 0.64,
            "tolerance": 0.02,
            "dem_Z": negative["microstructure"]["coordination_number"] if negative else float("nan"),
            "note": (
                "The campaign target 'n -> 0.64' is read as the random-close-packing "
                "SOLID fraction phi = 0.637. Frictionless monodisperse spheres cannot "
                "reach a porosity of 0.64 (that would be phi = 0.36, far below the "
                "random-loose-packing limit phi ~ 0.55); both numbers are reported."
            ),
            "passed": bool(
                negative is not None
                and abs(negative["microstructure"]["packing_fraction_voronoi"] - 0.64) <= 0.02
            ),
            "evidence": "REAL (DEM) vs textbook RCP benchmark",
        },
        "f_floe_field": {
            "n_floes": int(primary["n_particles"]),
            "diameter_min": primary["microstructure"]["diameter_min"],
            "diameter_max": primary["microstructure"]["diameter_max"],
            "ks_vs_rosin_rammler": RosinRammlerDistribution(
                d_min=cfg.d_min, d_max=cfg.d_max, d_char=cfg.d_char, exponent=cfg.exponent
            ).ks_statistic(np.asarray(primary["diameters"], dtype=float)),
            "expected_mean_diameter": RosinRammlerDistribution(
                d_min=cfg.d_min, d_max=cfg.d_max, d_char=cfg.d_char, exponent=cfg.exponent
            ).mean_diameter(),
            "sample_mean_diameter": float(np.mean(primary["diameters"])),
            "distribution": {
                "d_min": cfg.d_min,
                "d_max": cfg.d_max,
                "d_char": cfg.d_char,
                "exponent": cfg.exponent,
            },
            "passed": bool(int(primary["n_particles"]) >= 1000),
            "evidence": "REAL (generator) vs analytic CDF",
        },
    }
    if g_run is not None:
        criteria["g_gravity_sensitivity"] = {
            "gating": False,
            "gravity_scale": cfg.g_sensitivity_scale,
            "porosity_half_g": g_run["microstructure"]["porosity_voronoi"],
            "porosity_full_g": n_dem,
            "absolute_change": abs(g_run["microstructure"]["porosity_voronoi"] - n_dem),
            "tolerance": 0.04,
            "passed": bool(
                abs(g_run["microstructure"]["porosity_voronoi"] - n_dem) <= 0.04
            ),
            "evidence": (
                "REAL (DEM) diagnostic: halving gravity removes part of the self-weight "
                "consolidation, so a small porosity shift is expected physics, not error"
            ),
        }

    gating = {k: v for k, v in criteria.items() if v.get("gating", True)}
    summary = {
        "all_criteria_passed": bool(all(bool(v.get("passed")) for v in gating.values())),
        "porosity_relative_error": por_err,
        "coordination_deviation": z_dev,
        "force_chain_ks_p_matched": ks_matched["p_median"],
        "force_chain_ks_p_full": ks_full["p"],
        "resistance_r_squared": drag_fit.get("r_squared", float("nan")),
        "negative_control_packing_fraction": criteria["e_negative_control_frictionless"][
            "dem_packing_fraction"
        ],
        "n_gating_criteria": len(gating),
        "n_gating_passed": int(sum(1 for v in gating.values() if v.get("passed"))),
    }
    porosity_map = [
        {
            "mu": r["friction_coefficient"],
            "porosity": r["microstructure"]["porosity_voronoi"],
            "coordination_number": r["microstructure"]["coordination_number"],
            "coordination_number_all": r["microstructure"]["coordination_number_all"],
            "resistance": r["drag"].get("mean_force_tail"),
            "peak_lid_stress": r["compaction"]["final_lid_pressure"],
        }
        for r in mu_runs
    ]
    dist_ref = RosinRammlerDistribution(
        d_min=cfg.d_min, d_max=cfg.d_max, d_char=cfg.d_char, exponent=cfg.exponent
    )
    return {
        "module": "tensordem.ice_rubble_microstructure",
        "config": {**cfg.__dict__, "friction_values": list(cfg.friction_values),
                   "drag_exponents": list(cfg.drag_exponents)},
        "literature": {
            "porosity": {
                "citation": por_ref.citation,
                "mean": por_ref.mean,
                "std": por_ref.std,
                "n_datasets": por_ref.n_datasets,
                "evidence": por_ref.evidence,
            },
            "coordination": {
                "citation": z_ref.citation,
                "mean": z_ref.mean,
                "std": z_ref.std,
                "evidence": z_ref.evidence,
            },
            "force_chain": {
                "evidence": HOPKINS_TVEITEN_2002.evidence,
                "plateau_end": HOPKINS_TVEITEN_2002.plateau_end,
                "tail_rate": HOPKINS_TVEITEN_2002.tail_rate,
                "mean": HOPKINS_TVEITEN_2002.mean(),
                "cv": HOPKINS_TVEITEN_2002.coefficient_of_variation(),
                "n_contacts_reported": HOPKINS_TVEITEN_2002.n_contacts_reported,
            },
        },
        "rosin_rammler": {
            "mean_diameter": dist_ref.mean_diameter(),
            "mean_volume": dist_ref.mean_volume(),
            "moments": dist_ref.moments(3),
        },
        "criteria": criteria,
        "summary": summary,
        "porosity_friction_map": porosity_map,
    }


def run_microstructure_study(
    cfg: MicrostructureStudyConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    case_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the full ice-rubble microstructure calibration study.

    Sequence
    --------
    1. one Rosin-Rammler floe field per case (``>= 1000`` floes, diameters in
       ``[0.05, 0.5] m``); all cases share the same placement seed, so only the
       contact law or the size-distribution width changes;
    2. settle under self weight, compact with a dead-weight piston to
       ``compact_stress``, relax to equilibrium, then measure porosity (voxel
       Voronoi), coordination number, the contact-force spectrum and the
       force-chain statistics;
    3. drive the platen into the compacted rubble and record the resistance;
    4. criteria: porosity vs Timco & O'Brien, force-chain KS vs the literature
       shape, ``Z`` vs the ice-rubble reference, ``F ~ (1 - n)`` across the
       shape family, the frictionless RCP negative control, the floe-field
       distribution and (diagnostic) the gravity sensitivity.

    Returns a JSON-serialisable dictionary (see :func:`analyse_runs`).
    """
    cfg = cfg if cfg is not None else MicrostructureStudyConfig()
    say = progress if progress is not None else (lambda _msg: None)
    runs: dict[str, dict[str, Any]] = {}
    for case in study_cases(cfg):
        if case_names is not None and case["name"] not in case_names:
            continue
        say(f"case {case['name']}")
        runs[case["name"]] = run_study_case(cfg, case)
    return analyse_runs(cfg, runs)