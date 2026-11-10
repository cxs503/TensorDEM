"""Weibull-strength controlled particle crushing (P96W2-N03).

Why this module
---------------
TensorDEM's bonded-sphere engines (:class:`tensordem.dem3d.IceDEM3D`,
:class:`tensordem.dem.IceDEM`) treat every particle as an **unbreakable** rigid
body: fracture is represented only by the *bond* network (``breaking_strain`` /
Weibull bond strength).  Granular crushing -- a particle that splits because the
*particle* strength is exceeded -- is a different, orthogonal mechanism: it
changes the particle-size distribution instead of the bond topology.

This module adds that missing mechanism as an opt-in DEM layer:

1. :class:`WeibullStrength` -- weakest-link single-particle strength
   ``P_s(d) = exp(-(d/d0)**3 * (sigma/sigma0)**m)``.  The ``(d/d0)**3`` volume
   scaling is exactly what produces the size effect: the characteristic
   strength is ``sigma0*(d0/d)**(3/m)``, i.e. a log-log slope ``-3/m``.
2. :class:`Fragmenter` -- replaces a parent sphere by ``k`` child spheres whose
   diameters follow the Gaudin-Schuhmann cumulative distribution
   ``F(d) = (d/d_max)**n``.  Child positions are volume-weighted samples inside
   the parent sphere (overlap-relaxed, centre of mass preserved); child
   velocities inherit the parent velocity plus an exactly momentum-free radial
   expansion funded by the released elastic energy.
3. :class:`MassMomentumLedger` -- per-event bookkeeping of ``sum(m)``,
   ``sum(m v)``, ``sum(1/2 m v**2)``, fracture surface energy ``Gamma * dA``
   and cumulative damping/friction dissipation, closed by the audit
   ``W_ext = dU + dK + E_diss + E_surface``.
4. :class:`CrushingDEM` -- a small 3-D DEM time loop that owns the crushing
   decision at the physically documented place: **after** the contact forces of
   the step have been evaluated and **before** positions are updated, so a
   particle can break at most once per step (``born_step`` gate).  Contacts use
   the same linear spring-dashpot + Coulomb-friction conventions as ``dem3d``,
   with approach-only damping so every energy term is exact by construction.
   An optional Weibull bond network provides the second, orthogonal fracture
   path required by the negative controls.

Boundary note (honest limitation): :class:`tensordem.dem3d.IceDEM3D` allocates
fixed-size particle arrays, so its particle count cannot change during a run;
the crushing loop therefore lives in this module instead of inside the
``dem3d`` integrator, which is left untouched (its trajectories stay
bit-for-bit identical -- see ``tests/test_crushing.py`` C7).  The module
follows the repository's contact and energy conventions; it is not a private
fork of the solver.

All quantities are SI (m, kg, s, Pa, J).  Every random draw comes from an
explicit ``numpy.random.Generator``, so a run is reproducible from its seed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import brentq

__all__ = [
    "WeibullStrength",
    "fit_weibull_strength",
    "fit_gaudin_schuhmann",
    "size_effect_slope",
    "Particle",
    "Fragmenter",
    "FragmentResult",
    "BreakageRecord",
    "MassMomentumLedger",
    "CrushingConfig",
    "CrushingDEM",
    "ContactDiagnostics",
    "nominal_particle_stress",
    "sphere_mass",
    "sphere_diameter",
]


# --------------------------------------------------------------------------
# 1. Weakest-link single-particle strength
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WeibullStrength:
    """Weibull weakest-link strength law for a single particle.

    ``P_s(d, sigma) = exp(-(d/d0)**3 * (sigma/sigma0)**m)``

    ``d`` is the equivalent particle diameter [m], ``sigma0`` the
    characteristic strength of a particle of diameter ``d0`` [Pa, 63.2 %
    failure) and ``m`` the Weibull modulus (scatter).  The exponent of the
    volume term is 3 because the strength-limiting flaw population is
    distributed over the particle volume.
    """

    d0: float = 2.0e-3
    sigma0: float = 1.5e6
    m: float = 3.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.d0) and self.d0 > 0.0):
            raise ValueError("d0 must be finite and positive")
        if not (math.isfinite(self.sigma0) and self.sigma0 > 0.0):
            raise ValueError("sigma0 must be finite and positive")
        if not (math.isfinite(self.m) and self.m > 0.0):
            raise ValueError("m must be finite and positive")

    # -- closed-form statistics -------------------------------------------
    @property
    def size_effect_exponent(self) -> float:
        """Theoretical log-log slope of characteristic strength vs diameter."""
        return -3.0 / self.m

    def volume_ratio(self, d: float) -> float:
        size = float(d)
        if not (math.isfinite(size) and size > 0.0):
            raise ValueError("particle diameter d must be finite and positive")
        return (size / self.d0) ** 3.0

    def characteristic_strength(self, d: float) -> float:
        """63.2 %-failure strength of a particle of diameter ``d``."""
        return float(self.sigma0 * (self.d0 / float(d)) ** (3.0 / self.m))

    def median_strength(self, d: float) -> float:
        """Median strength (the size-effect test statistic)."""
        return self.characteristic_strength(d) * math.log(2.0) ** (1.0 / self.m)

    def mean_strength(self, d: float) -> float:
        return self.characteristic_strength(d) * math.gamma(1.0 + 1.0 / self.m)

    def survival(self, sigma: Any, d: float) -> np.ndarray:
        s = np.asarray(sigma, dtype=float)
        return np.exp(-self.volume_ratio(d) * (np.maximum(s, 0.0) / self.sigma0) ** self.m)

    def failure(self, sigma: Any, d: float) -> np.ndarray:
        return 1.0 - self.survival(sigma, d)

    # -- sampling ---------------------------------------------------------
    def sample(self, n: int, d: float, *, seed: int | None = None,
               generator: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` independent strengths for particles of diameter ``d``.

        Inverse-CDF sampling of the weakest-link law:
        ``sigma = sigma_c(d) * (-ln u)**(1/m)`` with ``u ~ U(0, 1)``.
        """
        if n < 1:
            raise ValueError("n must be positive")
        if not (math.isfinite(float(d)) and float(d) > 0.0):
            raise ValueError("particle diameter d must be finite and positive")
        rng = generator if generator is not None else np.random.default_rng(seed)
        u = rng.random(int(n))
        u = np.clip(u, np.finfo(float).tiny, 1.0)
        return self.characteristic_strength(d) * (-np.log(u)) ** (1.0 / self.m)

    def sample_single(self, d: float, *, generator: np.random.Generator) -> float:
        return float(self.sample(1, d, generator=generator)[0])

    # -- inversion --------------------------------------------------------
    def mle_fit(self, samples: Sequence[float], d: float | None = None) -> dict[str, Any]:
        """Maximum-likelihood inversion of ``(m, sigma0)`` from a sample set.

        ``d`` is the common particle diameter of the samples; when given, the
        fitted location is rescaled to the reference size ``d0`` through the
        weakest-link volume term -- the exact inverse of :meth:`sample`.
        """
        fit = fit_weibull_strength(samples)
        result: dict[str, Any] = dict(fit)
        result["diameter"] = None if d is None else float(d)
        if d is not None:
            if not (math.isfinite(float(d)) and float(d) > 0.0):
                raise ValueError("d must be finite and positive")
            result["sigma0_hat"] = float(
                fit["sigma_at_size"] * (float(d) / self.d0) ** (3.0 / fit["m_hat"]))
        result["m_error"] = abs(result["m_hat"] - self.m) / self.m
        if "sigma0_hat" in result:
            result["sigma0_error"] = abs(result["sigma0_hat"] - self.sigma0) / self.sigma0
        return result


def fit_weibull_strength(samples: Sequence[float]) -> dict[str, float]:
    """Two-parameter Weibull MLE (shape ``m``, scale ``sigma``) of ``samples``.

    The shape solves the standard likelihood equation
    ``sum(x**m ln x)/sum(x**m) - 1/m - mean(ln x) = 0``; the scale is then
    ``(mean(x**m))**(1/m)``.  The sample is normalised by its mean before the
    root solve so the power stays inside float range for any realistic ``m``.
    """
    x = np.asarray([float(v) for v in samples], dtype=float)
    if x.size < 3:
        raise ValueError("at least three positive samples are required")
    if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("samples must be finite and positive")
    xbar = float(x.mean())
    y = x / xbar
    ly = np.log(y)
    mean_ly = float(ly.mean())

    def equation(m: float) -> float:
        ym = np.power(y, m)
        return float(np.sum(ym * ly) / np.sum(ym) - 1.0 / m - mean_ly)

    lo, hi = 1.0e-3, 200.0
    if equation(lo) * equation(hi) > 0.0:
        raise RuntimeError("Weibull MLE shape equation has no bracket")
    m_hat = float(brentq(equation, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))
    sigma_hat = xbar * float(np.mean(np.power(y, m_hat)) ** (1.0 / m_hat))
    loglik = float(np.sum(np.log(m_hat / sigma_hat)
                          + (m_hat - 1.0) * np.log(x / sigma_hat)
                          - (x / sigma_hat) ** m_hat))
    return {
        "m_hat": m_hat,
        "sigma_at_size": sigma_hat,
        "n": float(x.size),
        "mean": xbar,
        "log_likelihood": loglik,
    }


def fit_gaudin_schuhmann(diameters: Sequence[float]) -> dict[str, float]:
    """Fit ``F(d) = (d/d_max)**n`` to an empirical fragment size census.

    ``F`` is the number-based cumulative distribution built from Weibull
    plotting positions ``i/(N+1)`` on the sorted diameters; the fit is a
    least-squares line in log-log space (``ln F`` vs ``ln d``).  Returns the
    Gaudin-Schuhmann exponent ``n``, the fitted ``d_max`` (``F = 1`` intercept)
    and ``r_squared``.
    """
    d = np.asarray([float(v) for v in diameters], dtype=float)
    if d.size < 3:
        raise ValueError("at least three fragment diameters are required")
    if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        raise ValueError("fragment diameters must be finite and positive")
    d = np.sort(d)
    count = d.size
    f = np.arange(1, count + 1, dtype=float) / (count + 1.0)
    lx = np.log(d)
    ly = np.log(f)
    a, b = np.polyfit(lx, ly, 1)
    if not math.isfinite(float(a)) or a <= 0.0:
        raise RuntimeError("degenerate Gaudin-Schuhmann fit (non-positive exponent)")
    pred = a * lx + b
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    # Maximum-likelihood exponent with the anchor at the largest observed
    # fragment (Hill-type estimator): ``n = -N / sum(ln(d_i / d_max))``.  It is
    # reported next to the CDF regression slope because the log-log regression
    # of a *complete* power law carries a few-percent finite-sample bias.
    n_mle = float(-count / float(np.sum(np.log(d / d.max()))))
    return {"n": float(a), "d_max": float(math.exp(-b / a)),
            "r_squared": float(r2), "n_fragments": float(count),
            "n_mle": n_mle}


def size_effect_slope(diameters: Sequence[float],
                      strengths: Sequence[float]) -> dict[str, float]:
    """Least-squares slope of ``ln(strength)`` vs ``ln(diameter)``."""
    d = np.asarray([float(v) for v in diameters], dtype=float)
    s = np.asarray([float(v) for v in strengths], dtype=float)
    if d.size != s.size or d.size < 2:
        raise ValueError("need matched arrays of at least two diameters")
    if np.any(d <= 0.0) or np.any(s <= 0.0):
        raise ValueError("diameters and strengths must be positive")
    slope, intercept = np.polyfit(np.log(d), np.log(s), 1)
    ly = np.log(s)
    pred = slope * np.log(d) + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0}


# --------------------------------------------------------------------------
# 2. Particles and the fragmenter
# --------------------------------------------------------------------------
def _as_vec3(value: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(value), dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector components must be finite")
    return arr


@dataclass
class Particle:
    """A crushable spherical particle (SI units)."""

    position: np.ndarray
    velocity: np.ndarray
    radius: float
    mass: float
    strength: float
    lineage: int = 0
    generation: int = 0
    equivalent_diameter: float | None = None

    def __post_init__(self) -> None:
        self.position = _as_vec3(self.position)
        self.velocity = _as_vec3(self.velocity)
        self.radius = float(self.radius)
        self.equivalent_diameter = (2.0 * self.radius if self.equivalent_diameter is None
                                    else float(self.equivalent_diameter))
        self.mass = float(self.mass)
        self.strength = float(self.strength)
        self.lineage = int(self.lineage)
        self.generation = int(self.generation)
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError("radius must be finite and positive")
        if not (math.isfinite(self.mass) and self.mass > 0.0):
            raise ValueError("mass must be finite and positive")
        if not math.isfinite(self.strength):
            raise ValueError("strength must be finite")

    @property
    def diameter(self) -> float:
        """Mass-equivalent (PSD) diameter of the particle."""
        return float(self.equivalent_diameter)

    @property
    def momentum(self) -> np.ndarray:
        return self.mass * self.velocity

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(self.velocity @ self.velocity)


def sphere_mass(diameter: float, density: float) -> float:
    """Mass of a sphere ``rho * pi/6 * d**3``."""
    return float(density * math.pi / 6.0 * float(diameter) ** 3)


def sphere_diameter(mass: float, density: float) -> float:
    """Equivalent sphere diameter for a given mass."""
    return float((6.0 * float(mass) / (math.pi * float(density))) ** (1.0 / 3.0))


def nominal_particle_stress(force: float | torch.Tensor, radius: float) -> float:
    """Nominal particle stress ``sigma = F / (pi R**2)`` [Pa].

    Force-based single-particle strength criterion: the axial force carried by
    the particle divided by its mid-section area.  This is the standard DEM
    measure of particle strength (the force/area definition of particle
    breakage models) and the quantity compared with
    :meth:`WeibullStrength.sample`.
    """
    r = float(radius)
    if not (math.isfinite(r) and r > 0.0):
        raise ValueError("radius must be finite and positive")
    f = float(force)
    if not math.isfinite(f):
        raise ValueError("force must be finite")
    return f / (math.pi * r * r)


@dataclass
class FragmentResult:
    """Outcome of one parent -> ``k`` children replacement."""

    parent: Particle
    children: list[Particle]
    diameters: np.ndarray
    mass_fractions: np.ndarray
    surface_area_increase: float
    expansion_energy_J: float
    expansion_speed: float
    d_max: float

    @property
    def child_masses(self) -> np.ndarray:
        return np.asarray([c.mass for c in self.children], dtype=float)

    @property
    def child_velocities(self) -> np.ndarray:
        return np.asarray([c.velocity for c in self.children], dtype=float)

    def mass_residual(self) -> float:
        return float(abs(self.child_masses.sum() - self.parent.mass) / self.parent.mass)

    def momentum_residual(self) -> float:
        """Relative momentum error, scaled by the largest momentum in the event.

        The scale is ``max(|p_parent|, m_parent * max|v_child|)`` so the metric
        stays meaningful for a parent at rest (a pure ``0/0`` case for the naive
        relative residual).
        """
        p_child = (self.child_masses[:, None] * self.child_velocities).sum(axis=0)
        p_parent = self.parent.momentum
        diff = float(np.linalg.norm(p_child - p_parent))
        speed = float(np.linalg.norm(self.child_velocities, axis=1).max()) \
            if self.child_velocities.size else 0.0
        scale = max(float(np.linalg.norm(p_parent)), self.parent.mass * speed)
        return diff if scale <= 0.0 else diff / scale


@dataclass(frozen=True)
class Fragmenter:
    """Replace a parent sphere by ``k`` Gaudin-Schuhmann fragments.

    * ``k`` -- number of children per replacement (>= 2).
    * ``n`` -- Gaudin-Schuhmann exponent of the child *diameter* distribution:
      ``F(d) = (d/d_max)**n`` with ``d_max`` = parent diameter.
    * ``d_min_fraction`` -- numerical floor on child diameter as a fraction of
      the parent diameter (keeps the penalty contacts resolvable).
    * ``placement_radius_fraction`` -- children are placed inside this fraction
      of the parent radius, then overlap-relaxed, so the replacement is
      geometrically admissible.
    """

    k: int = 8
    n: float = 0.8
    d_min_fraction: float = 0.05
    placement_radius_fraction: float = 1.0
    overlap_relaxation_iterations: int = 60
    contact_radius_fraction: float = 0.6

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("k must be at least 2")
        if not (math.isfinite(self.n) and self.n > 0.0):
            raise ValueError("n must be finite and positive")
        if not (0.0 <= self.d_min_fraction < 1.0):
            raise ValueError("d_min_fraction must be in [0, 1)")
        if not (0.0 < self.placement_radius_fraction <= 1.0):
            raise ValueError("placement_radius_fraction must be in (0, 1]")
        if not (0.0 < self.contact_radius_fraction <= 1.0):
            raise ValueError("contact_radius_fraction must be in (0, 1]")

    # -- geometry helpers -------------------------------------------------
    def sample_diameters(self, parent_diameter: float,
                         rng: np.random.Generator) -> np.ndarray:
        """Stratified inverse-CDF draws from ``F(d) = (d/d_max)**n``.

        The ``k`` children are drawn from the ``k`` equal-probability strata of
        the Gaudin-Schuhmann law (``u_i = (i + xi_i)/k``), i.e. a Latin-hypercube
        sample of the inverse CDF.  Compared with ``k`` plain uniform draws this
        keeps the realised census an accurate power law for the *same* number of
        fragments, which matters because the mass-conservation rescale below is
        a single multiplicative factor: with plain sampling its scatter smears
        the pooled distribution of several parents.
        """
        u = (np.arange(self.k) + rng.random(self.k)) / float(self.k)
        u = np.clip(u, np.finfo(float).tiny, 1.0)
        d = float(parent_diameter) * u ** (1.0 / self.n)
        d_min = self.d_min_fraction * float(parent_diameter)
        return np.maximum(d, d_min)

    @staticmethod
    def _unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
        v = rng.normal(size=(count, 3))
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        return v / norm

    def _place_children(self, radii: np.ndarray, masses: np.ndarray,
                        parent_radius: float,
                        rng: np.random.Generator) -> np.ndarray:
        """Place the children inside the parent sphere without overlap.

        Largest-first greedy placement: every child is dropped at a random
        admissible point inside the parent and accepted as soon as it clears all
        previously placed children.  When the requested packing is geometrically
        impossible the fallback keeps the position with the largest clearance
        instead of silently accepting a deep overlap (a born overlap would
        inject spurious elastic energy into the energy ledger).
        """
        count = radii.size
        offsets = np.zeros((count, 3))
        placed: list[int] = []
        tolerance = 1.0e-9 * parent_radius
        for index in np.argsort(-radii):
            radius = float(radii[index])
            room = max(parent_radius - radius, 0.0)
            best = np.zeros(3)
            best_clearance = -math.inf
            for _ in range(400):
                direction = self._unit_vectors(1, rng)[0]
                rho = room * rng.random() ** (1.0 / 3.0)
                candidate = direction * rho
                clearance = min(
                    (float(np.linalg.norm(candidate - offsets[j])) - (radius + radii[j])
                     for j in placed), default=math.inf)
                if clearance > tolerance:
                    best, best_clearance = candidate, clearance
                    break
                if clearance > best_clearance:
                    best, best_clearance = candidate, clearance
            offsets[index] = best
            placed.append(int(index))
        return self._separate(offsets, radii, masses, parent_radius)[0]

    def _separate(self, offsets: np.ndarray, radii: np.ndarray, masses: np.ndarray,
                  parent_radius: float, *, iterations: int = 2000,
                  tolerance: float = 1.0e-6) -> tuple[np.ndarray, float]:
        """Remove the residual overlaps of the greedy placement.

        Jacobi projection sweeps over the full ``k x k`` pair matrix (vectorised):
        overlapping pairs are pushed apart inversely to their mass, every child
        is radially clamped inside the parent, and the centre of mass is
        restored.  Returns the refined offsets and the worst residual overlap
        ratio, which the caller can use to decide whether the contact-sphere
        proxy has to be shrunk.
        """
        count = radii.size
        pos = offsets.copy()
        total = float(masses.sum())
        upper = np.triu(np.ones((count, count), dtype=bool), 1)
        w = masses[None, :] / (masses[:, None] + masses[None, :])
        worst = 0.0
        for _ in range(iterations):
            delta = pos[None, :, :] - pos[:, None, :]
            dist = np.linalg.norm(delta, axis=2)
            touch = radii[:, None] + radii[None, :]
            ratio = np.where(upper, (touch - dist) / np.maximum(touch, 1e-300), 0.0)
            worst = float(ratio.max()) if ratio.size else 0.0
            if worst <= tolerance:
                break
            direction = delta / np.maximum(dist, 1e-15)[..., None]
            push = np.where(upper, np.maximum(touch - dist, 0.0), 0.0)
            forward = -np.where(upper[..., None], (push * w)[..., None] * direction,
                                0.0).sum(axis=1)
            backward = np.where(upper[..., None],
                                (push * (1.0 - w))[..., None] * direction,
                                0.0).sum(axis=0)
            pos += forward + backward
            reach = np.linalg.norm(pos, axis=1)
            limit = np.maximum(parent_radius - radii, 0.0)
            over = reach > limit
            if bool(np.any(over)):
                pos[over] *= (limit[over] / reach[over])[:, None]
            pos += ((offsets * masses[:, None]).sum(axis=0)[None, :]
                    - (pos * masses[:, None]).sum(axis=0)[None, :]) / total
        return pos, worst

    def distribute_expansion(self, children: list[Particle], *, centre: np.ndarray,
                             energy_J: float) -> float:
        """Give ``children`` an exactly momentum-free radial expansion.

        ``sum(m u) = 0`` by construction, so the parent momentum is inherited
        exactly; the expansion kinetic energy equals ``energy_J``.
        """
        if energy_J <= 0.0 or not children:
            return 0.0
        masses = np.asarray([c.mass for c in children], dtype=float)
        offsets = np.asarray([c.position for c in children], dtype=float) - np.asarray(centre)
        norm = np.linalg.norm(offsets, axis=1)
        unit = np.where(norm[:, None] > 0.0, offsets / np.maximum(norm, 1e-300)[:, None],
                        np.array([1.0, 0.0, 0.0]))
        mean_unit = (unit * masses[:, None]).sum(axis=0) / float(masses.sum())
        shape = unit - mean_unit[None, :]
        inertia = float((masses[:, None] * shape ** 2).sum())
        if inertia <= 0.0:
            return 0.0
        speed = math.sqrt(2.0 * float(energy_J) / inertia)
        for child, row in zip(children, speed * shape):
            child.velocity = child.velocity + row
        return float(energy_J)

    # -- replacement ------------------------------------------------------
    def split(self, parent: Particle, *, density: float,
              rng: np.random.Generator,
              separation_energy_J: float = 0.0,
              child_strengths: Sequence[float] | None = None) -> FragmentResult:
        """Split ``parent`` into ``k`` mass/momentum-conserving children."""
        if not (math.isfinite(density) and density > 0.0):
            raise ValueError("density must be finite and positive")
        if separation_energy_J < 0.0:
            raise ValueError("separation_energy_J must be nonnegative")

        d_raw = self.sample_diameters(parent.diameter, rng)
        # A uniform mass rescale keeps the realised diameter distribution a pure
        # Gaudin-Schuhmann power law (a multiplicative constant cannot change a
        # log-log slope) while making sum(m_child) = m_parent exact.
        raw_masses = np.pi / 6.0 * density * d_raw ** 3
        scale = (parent.mass / float(raw_masses.sum())) ** (1.0 / 3.0)
        diameters = d_raw * scale
        masses = np.pi / 6.0 * density * diameters ** 3
        masses = masses * (parent.mass / float(masses.sum()))
        # The contact sphere is smaller than the mass-equivalent sphere by the
        # volume shape factor: crushed fragments are angular, so the sphere
        # proxy used for contacts cannot fill the parent volume without
        # overlap.  Mass (and therefore the PSD) stays conservative; only the
        # contact geometry is reduced.
        radii = 0.5 * diameters * self.contact_radius_fraction

        # Volume-weighted placement inside the parent, then overlap relaxation.
        container = self.placement_radius_fraction * parent.radius
        offsets = self._place_children(radii, masses, container, rng)
        offsets -= (offsets * masses[:, None]).sum(axis=0)[None, :] / float(masses.sum())

        def _contain(offsets: np.ndarray) -> tuple[np.ndarray, float]:
            """Radial scale that keeps every child inside the parent sphere."""
            reach = np.linalg.norm(offsets, axis=1) + radii
            return offsets, float(min(1.0, container / float(reach.max())))

        offsets, shrink = _contain(offsets)
        if shrink < 1.0:
            # The contact-sphere proxy is shrunk, never the mass: the PSD is a
            # mass-equivalent property, the packing radius is numerical.
            radii = radii * shrink
            offsets = self._separate(offsets, radii, masses, container)[0]
            offsets, shrink = _contain(offsets)
            if shrink < 1.0:
                radii = radii * shrink
                offsets = offsets * shrink
        # Guarantee a born state with negligible overlap: the contact-sphere
        # proxy may be shrunk, the mass (and therefore the PSD) never is.
        residual = 0.0
        for i in range(self.k - 1):
            for j in range(i + 1, self.k):
                dist = float(np.linalg.norm(offsets[j] - offsets[i]))
                touch = radii[i] + radii[j]
                if touch > 0.0:
                    residual = max(residual, (touch - dist) / touch)
        if residual > 1.0e-6:
            radii = radii * (1.0 - residual)
            offsets = self._separate(offsets, radii, masses,
                                     self.placement_radius_fraction * parent.radius,
                                     iterations=200)[0]

        strengths = (np.asarray(list(child_strengths), dtype=float)
                     if child_strengths is not None
                     else np.full(self.k, parent.strength))
        if strengths.size != self.k:
            raise ValueError("child_strengths must have length k")
        children = [
            Particle(position=parent.position + offsets[i],
                     velocity=parent.velocity.copy(),
                     radius=radii[i], mass=masses[i], strength=strengths[i],
                     lineage=parent.lineage, generation=parent.generation + 1,
                     equivalent_diameter=diameters[i])
            for i in range(self.k)
        ]
        achieved = self.distribute_expansion(children, centre=parent.position,
                                            energy_J=separation_energy_J)
        eq_radii = 0.5 * diameters
        surface_increase = 4.0 * math.pi * (float((eq_radii ** 2).sum())
                                           - parent.radius ** 2)
        return FragmentResult(
            parent=parent, children=children, diameters=diameters,
            mass_fractions=masses / parent.mass,
            surface_area_increase=surface_increase,
            expansion_energy_J=achieved,
            expansion_speed=float(np.linalg.norm(children[0].velocity - parent.velocity)),
            d_max=float(diameters.max()),
        )


# --------------------------------------------------------------------------
# 3. Mass / momentum / energy ledger
# --------------------------------------------------------------------------
@dataclass
class BreakageRecord:
    """One mass/momentum/energy audit record for a parent -> children swap."""

    event_id: int
    parent_mass: float
    parent_momentum: np.ndarray
    child_masses: np.ndarray
    child_velocities: np.ndarray
    surface_energy_J: float
    released_elastic_energy_J: float
    expansion_energy_J: float
    requested_surface_energy_J: float = 0.0

    @property
    def mass_residual(self) -> float:
        return float(abs(self.child_masses.sum() - self.parent_mass) / self.parent_mass)

    @property
    def momentum_residual(self) -> float:
        """Relative momentum error, scaled by the largest momentum in the event."""
        p_child = (self.child_masses[:, None] * self.child_velocities).sum(axis=0)
        diff = float(np.linalg.norm(p_child - self.parent_momentum))
        speed = (float(np.linalg.norm(self.child_velocities, axis=1).max())
                 if self.child_velocities.size else 0.0)
        scale = max(float(np.linalg.norm(self.parent_momentum)),
                    self.parent_mass * speed)
        return diff if scale <= 0.0 else diff / scale

    @property
    def kinetic_energy_before_J(self) -> float:
        return 0.5 * float(self.parent_momentum @ self.parent_momentum) / self.parent_mass

    @property
    def kinetic_energy_after_J(self) -> float:
        return float(0.5 * np.sum(self.child_masses
                                  * np.sum(self.child_velocities ** 2, axis=1)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "parent_mass_kg": self.parent_mass,
            "child_count": int(self.child_masses.size),
            "mass_residual": self.mass_residual,
            "momentum_residual": self.momentum_residual,
            "kinetic_energy_before_J": self.kinetic_energy_before_J,
            "kinetic_energy_after_J": self.kinetic_energy_after_J,
            "surface_energy_J": self.surface_energy_J,
            "requested_surface_energy_J": self.requested_surface_energy_J,
            "released_elastic_energy_J": self.released_elastic_energy_J,
            "expansion_energy_J": self.expansion_energy_J,
        }


class MassMomentumLedger:
    """Cumulative mass/momentum/energy bookkeeping for a crushing run.

    The audited identity (:meth:`audit`) is::

        W_ext = (U - U_0) + (K - K_0) + E_diss + E_surface

    ``W_ext`` is the work done by the moving platens, ``U`` the elastic energy
    stored in contacts and surviving bonds, ``K`` the particle kinetic energy,
    ``E_diss`` the accumulated damping + friction dissipation and
    ``E_surface`` the fracture surface energy ``Gamma * dA`` actually booked.

    Booked surface energy is capped at the elastic energy released by the
    destroyed parent contacts; the deficit (if any) is reported as
    ``unfunded_surface_energy_J`` instead of being silently created.
    """

    def __init__(self, gamma: float = 0.1) -> None:
        if not (math.isfinite(gamma) and gamma >= 0.0):
            raise ValueError("gamma must be finite and nonnegative")
        self.gamma = float(gamma)
        self.records: list[BreakageRecord] = []
        self.external_work_J = 0.0
        self.dissipation_J = 0.0
        self.surface_energy_J = 0.0
        self.unfunded_surface_energy_J = 0.0
        self.released_elastic_energy_J = 0.0
        self.expansion_energy_J = 0.0

    def surface_energy_demand(self, area_increase: float) -> float:
        """``Gamma * dA`` for a given created surface area."""
        return self.gamma * max(float(area_increase), 0.0)

    def record_replacement(self, result: FragmentResult,
                           released_elastic_energy_J: float,
                           *, event_id: int | None = None) -> BreakageRecord:
        requested = self.surface_energy_demand(result.surface_area_increase)
        available = max(float(released_elastic_energy_J), 0.0)
        booked = min(requested, available)
        self.unfunded_surface_energy_J += max(requested - booked, 0.0)
        self.surface_energy_J += booked
        self.released_elastic_energy_J += available
        self.expansion_energy_J += result.expansion_energy_J
        record = BreakageRecord(
            event_id=len(self.records) if event_id is None else int(event_id),
            parent_mass=result.parent.mass,
            parent_momentum=result.parent.momentum.copy(),
            child_masses=result.child_masses.copy(),
            child_velocities=result.child_velocities.copy(),
            surface_energy_J=booked,
            released_elastic_energy_J=available,
            expansion_energy_J=result.expansion_energy_J,
            requested_surface_energy_J=requested,
        )
        self.records.append(record)
        return record

    def replacement_report(self) -> dict[str, float]:
        """Worst-case relative residuals over all replacement events."""
        if not self.records:
            return {"events": 0.0, "max_mass_residual": 0.0,
                    "max_momentum_residual": 0.0}
        return {
            "events": float(len(self.records)),
            "max_mass_residual": max(r.mass_residual for r in self.records),
            "max_momentum_residual": max(r.momentum_residual for r in self.records),
        }

    def audit(self, *, elastic_energy_J: float, kinetic_energy_J: float,
              initial_elastic_energy_J: float, initial_kinetic_energy_J: float,
              extra_sink_J: float = 0.0) -> dict[str, float]:
        """Close the energy account and return the residual and its scale."""
        delta_u = float(elastic_energy_J) - float(initial_elastic_energy_J)
        delta_k = float(kinetic_energy_J) - float(initial_kinetic_energy_J)
        sinks = self.surface_energy_J + self.dissipation_J + float(extra_sink_J)
        sources = self.external_work_J
        residual = sources - (delta_u + delta_k + sinks)
        scale = max(abs(sources), abs(delta_u) + abs(delta_k) + sinks, 1e-30)
        return {
            "external_work_J": sources,
            "delta_elastic_J": delta_u,
            "delta_kinetic_J": delta_k,
            "dissipation_J": self.dissipation_J,
            "surface_energy_J": self.surface_energy_J,
            "extra_sink_J": float(extra_sink_J),
            "unfunded_surface_energy_J": self.unfunded_surface_energy_J,
            "residual_J": residual,
            "residual_relative": abs(residual) / scale,
        }


# --------------------------------------------------------------------------
# 4. Crushing DEM time loop
# --------------------------------------------------------------------------
@dataclass
class CrushingConfig:
    """Physical and numerical controls of the crushing DEM."""

    density: float = 917.0                   # kg/m^3 (ice)
    contact_stiffness: float = 1.0e6         # N/m, normal penalty spring
    restitution_coefficient: float = 0.5     # e, sets the per-contact dashpot
    friction_coefficient: float = 0.4        # mu, Coulomb limit
    friction_velocity_scale: float = 1.0e-3  # m/s, tanh slip regularisation
    platen_stiffness: float | None = None    # N/m, None -> contact_stiffness
    platen_friction_coefficient: float = 0.0
    gamma: float = 0.1                       # J/m^2, fracture surface energy
    weibull: WeibullStrength = WeibullStrength()
    fragmenter: Fragmenter = Fragmenter()
    crushing_enabled: bool = True
    max_generation: int = 1
    max_particles: int = 1500
    min_fragment_diameter: float = 0.0
    # Optional Weibull bond network: the second, orthogonal fracture path.
    bond_stiffness: float | None = None
    bond_weibull: WeibullStrength | None = None
    bond_tension_only: bool = True
    dt: float | None = None
    dt_safety: float = 0.05
    gravity: float = 0.0
    seed: int = 0
    record_every: int = 1
    reseat_platen: bool = True

    def __post_init__(self) -> None:
        for name in ("density", "contact_stiffness", "gamma", "friction_velocity_scale"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 < self.restitution_coefficient <= 1.0):
            raise ValueError("restitution_coefficient must be in (0, 1]")
        if not (math.isfinite(self.friction_coefficient) and self.friction_coefficient >= 0.0):
            raise ValueError("friction_coefficient must be finite and nonnegative")
        if self.dt is not None and not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError("dt must be finite and positive")
        if not (0.0 < self.dt_safety <= 1.0):
            raise ValueError("dt_safety must be in (0, 1]")
        if self.max_generation < 0:
            raise ValueError("max_generation must be nonnegative")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")
        if self.bond_stiffness is not None:
            if self.bond_stiffness <= 0.0:
                raise ValueError("bond_stiffness must be positive when given")
            if self.bond_weibull is None:
                raise ValueError("bond_stiffness requires a bond_weibull strength law")


@dataclass
class ContactDiagnostics:
    """Forces and energy terms of one force evaluation."""

    forces: torch.Tensor                # (N, 3) total force on each particle
    elastic_energy_J: float              # contacts + platens + surviving bonds
    damping_dissipation_rate_W: float    # >= 0
    friction_dissipation_rate_W: float   # >= 0
    platen_work_rate_W: float            # power delivered by the platens
    platen_normal_force_N: float         # sum of the active platen contact forces
    max_compressive_force_N: torch.Tensor  # (N,)
    incident_elastic_energy_J: np.ndarray  # (N,) released if that particle dies
    pair_contacts: int
    platen_contacts: int


class CrushingDEM:
    """3-D DEM with opt-in Weibull particle crushing.

    Time-step ordering::

        forces -> platen work & dissipation accounting -> crushing decision
               -> bond breaking -> position/velocity integration

    The crushing decision therefore sees the *current* contact forces, and a
    particle created in this step can never break again in the same step
    (``born_step`` gate), which removes the same-frame double-breakage
    ambiguity.
    """

    def __init__(self, particles: Sequence[Particle],
                 config: CrushingConfig = CrushingConfig(),
                 *, seed: int | None = None) -> None:
        if not particles:
            raise ValueError("at least one particle is required")
        self.config = config
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.dtype = torch.float64
        self.step_count = 0
        self.time = 0.0
        self.crushing_events = 0
        self.bond_break_events = 0
        self.bond_cluster_events = 0
        self._bond_cluster_ref = 0
        self.fragment_sizes: list[float] = []
        self.first_generation_fragments: list[float] = []
        self.breakage_stress_Pa: list[float] = []
        self.history: list[dict[str, float]] = []
        self.ledger = MassMomentumLedger(gamma=config.gamma)
        self._load_particles(list(particles))
        self.platen_velocity = 0.0
        self.top_platen_z = float(self.positions[:, 2].max() + self.radii.max())
        self.bottom_platen_z = float(self.positions[:, 2].min() - self.radii.max())
        self._build_bonds()
        self._bond_cluster_ref = self.bond_cluster_count()
        self.dt = float(config.dt) if config.dt is not None else self._recommended_dt()
        self._initial_kinetic_J = self.kinetic_energy_J
        self._initial_elastic_J = self._elastic_energy_snapshot()
        self._prev_platen_force = 0.0
        self._prev_dissipation_rate = 0.0

    # -- construction helpers --------------------------------------------
    def _sync_particles(self) -> None:
        """Copy the integrated tensor state back onto the ``Particle`` objects.

        The tensors are the single source of truth for the integration; the
        particle objects are only used when a replacement has to rebuild the
        arrays, so they must be refreshed from the tensors first.
        """
        positions = self.positions.numpy()
        velocities = self.velocities.numpy()
        radii = self.radii.numpy()
        masses = self.masses.numpy()
        strengths = self.strengths.numpy()
        equivalent = self.diameters
        for index, particle in enumerate(self.particles):
            particle.position = positions[index].copy()
            particle.velocity = velocities[index].copy()
            particle.radius = float(radii[index])
            particle.mass = float(masses[index])
            particle.strength = float(strengths[index])
            particle.equivalent_diameter = float(equivalent[index])

    def _load_particles(self, particles: list[Particle]) -> None:
        self.particles = particles
        self.positions = torch.tensor(np.array([p.position for p in particles]),
                                      dtype=self.dtype)
        self.velocities = torch.tensor(np.array([p.velocity for p in particles]),
                                       dtype=self.dtype)
        self.radii = torch.tensor([p.radius for p in particles], dtype=self.dtype)
        self.masses = torch.tensor([p.mass for p in particles], dtype=self.dtype)
        self.strengths = torch.tensor([p.strength for p in particles], dtype=self.dtype)
        if not hasattr(self, "born_step"):
            self.lineage = [p.lineage for p in particles]
            self.generation = [p.generation for p in particles]
            self.born_step = [0 for _ in particles]

    @classmethod
    def single_sphere(cls, diameter: float, config: CrushingConfig = CrushingConfig(),
                      *, seed: int | None = None,
                      strength: float | None = None,
                      initial_overlap: float = 0.0) -> "CrushingDEM":
        """One particle squeezed between two platens (displacement control)."""
        radius = 0.5 * float(diameter)
        if radius <= 0.0:
            raise ValueError("diameter must be positive")
        rng = np.random.default_rng(config.seed if seed is None else seed)
        sigma = (config.weibull.sample_single(2.0 * radius, generator=rng)
                 if strength is None else float(strength))
        particle = Particle(position=np.zeros(3), velocity=np.zeros(3),
                            radius=radius,
                            mass=sphere_mass(2.0 * radius, config.density),
                            strength=sigma, lineage=0, generation=0)
        engine = cls([particle], config, seed=seed)
        engine.bottom_platen_z = -radius + float(initial_overlap)
        engine.top_platen_z = radius - float(initial_overlap)
        engine._initial_elastic_J = engine._elastic_energy_snapshot()
        engine._prev_platen_force = 0.0
        engine._prev_dissipation_rate = 0.0
        return engine

    @classmethod
    def column_bed(cls, count: int, diameter: float,
                   config: CrushingConfig = CrushingConfig(), *,
                   seed: int | None = None,
                   size_ratio: float | None = None,
                   initial_overlap: float = 0.0) -> "CrushingDEM":
        """Axially stacked 1-D column; ``size_ratio`` mixes two sizes 1:r."""
        if count < 2:
            raise ValueError("a bed needs at least two particles")
        rng = np.random.default_rng(config.seed if seed is None else seed)
        particles: list[Particle] = []
        z = 0.0
        previous_radius = 0.0
        for index in range(count):
            d = float(diameter)
            if size_ratio is not None and index % 2 == 1:
                d = float(diameter) * float(size_ratio)
            radius = 0.5 * d
            if index == 0:
                z = radius
            else:
                # ``initial_overlap`` pre-loads every contact, so a ``sigma_f ->
                # 0`` control breaks the whole bed in a single step.
                z += previous_radius + radius - float(initial_overlap)
            sigma = config.weibull.sample_single(d, generator=rng)
            particles.append(Particle(
                position=np.array([0.0, 0.0, z]), velocity=np.zeros(3),
                radius=radius, mass=sphere_mass(d, config.density),
                strength=sigma, lineage=index, generation=0))
            previous_radius = radius
        engine = cls(particles, config, seed=seed)
        engine.bottom_platen_z = (min(p.position[2] - p.radius for p in particles)
                                  + float(initial_overlap))
        engine.top_platen_z = (max(p.position[2] + p.radius for p in particles)
                               - float(initial_overlap))
        engine._initial_elastic_J = engine._elastic_energy_snapshot()
        engine._prev_platen_force = 0.0
        engine._prev_dissipation_rate = 0.0
        return engine

    # -- properties -------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())

    @property
    def total_momentum(self) -> torch.Tensor:
        return (self.masses[:, None] * self.velocities).sum(dim=0)

    @property
    def kinetic_energy_J(self) -> float:
        return float(0.5 * (self.masses * self.velocities.square().sum(dim=1)).sum())

    @property
    def contact_diameters(self) -> np.ndarray:
        return (2.0 * self.radii.numpy()).copy()

    @property
    def diameters(self) -> np.ndarray:
        """Mass-equivalent (PSD) diameters, recovered from the masses."""
        return np.asarray([sphere_diameter(float(m), self.config.density)
                           for m in self.masses.numpy()], dtype=float)

    @property
    def bond_breakage_active(self) -> bool:
        return self.config.bond_stiffness is not None

    def _platen_stiffness(self) -> float:
        return (self.config.platen_stiffness if self.config.platen_stiffness is not None
                else self.config.contact_stiffness)

    def damping_ratio(self) -> float:
        """Viscous damping ratio ``beta`` matching the restitution coefficient.

        ``beta = ln(1/e) / sqrt(ln(1/e)**2 + pi**2)`` -- the same
        restitution-coefficient parameterisation used by ``dem3d``.  The dashpot
        is then ``c = 2 beta sqrt(k m_eff)`` per contact, which keeps the
        explicit stability bound independent of the particle mass ratio (a
        fixed ``c`` over-damps small fragments and explodes).
        """
        e = float(self.config.restitution_coefficient)
        if e >= 1.0:
            return 0.0
        ln = math.log(1.0 / e)
        return ln / math.sqrt(ln * ln + math.pi ** 2)

    def _recommended_dt(self) -> float:
        """Explicit stability bound ``dt <= safety * min sqrt(m/k)``."""
        m_min = float(self.masses.min())
        k_max = max(self.config.contact_stiffness, self._platen_stiffness())
        if self.config.bond_stiffness:
            k_max = max(k_max, float(self.config.bond_stiffness))
        return float(self.config.dt_safety * math.sqrt(m_min / k_max))

    # -- bonds ------------------------------------------------------------
    def _build_bonds(self) -> None:
        """Create bonds between initially touching particles (opt-in)."""
        self.bond_i = np.zeros(0, dtype=int)
        self.bond_j = np.zeros(0, dtype=int)
        self.bond_rest = np.zeros(0)
        self.bond_strength = np.zeros(0)
        self.bond_alive = np.zeros(0, dtype=bool)
        self._bond_force = np.zeros(0)
        self._bond_extension = np.zeros(0)
        if self.config.bond_stiffness is None:
            return
        n = self.n
        pos = self.positions.numpy()
        rad = self.radii.numpy()
        pairs: list[tuple[int, int, float]] = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(pos[j] - pos[i]))
                if dist <= rad[i] + rad[j] + 1.0e-9:
                    pairs.append((i, j, dist))
        if not pairs:
            return
        self.bond_i = np.array([p[0] for p in pairs], dtype=int)
        self.bond_j = np.array([p[1] for p in pairs], dtype=int)
        self.bond_rest = np.array([p[2] for p in pairs], dtype=float)
        law = self.config.bond_weibull
        self.bond_strength = np.array([
            law.sample_single(2.0 * min(rad[i], rad[j]), generator=self.rng)
            * math.pi * min(rad[i], rad[j]) ** 2
            for (i, j, _) in pairs], dtype=float)
        self.bond_alive = np.ones(len(pairs), dtype=bool)
        self._bond_force = np.zeros(len(pairs))
        self._bond_extension = np.zeros(len(pairs))

    def bond_elastic_energy_J(self) -> float:
        """Elastic energy stored in the surviving bonds."""
        if self.bond_alive.size == 0 or not self.config.bond_stiffness:
            return 0.0
        k = float(self.config.bond_stiffness)
        extension = np.where(self.bond_alive, self._bond_extension, 0.0)
        return float(0.5 * k * float(np.sum(extension ** 2)))

    def bond_cluster_count(self) -> int:
        """Connected components of the surviving bond graph (union-find)."""
        n = self.n
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for index in np.nonzero(self.bond_alive)[0]:
            a, b = find(int(self.bond_i[index])), find(int(self.bond_j[index]))
            if a != b:
                parent[a] = b
        return len({find(i) for i in range(n)})

    def bond_derived_fragment_clusters(self) -> int:
        """Clusters that were *created by bond breaking* (0 if none broke).

        Crushing also detaches the bonds of the replaced particle, which changes
        the connectivity without being a bond-fracture event, so the counter is
        updated only inside :meth:`_apply_bond_breaking`.
        """
        return int(self.bond_cluster_events)

    # -- force pass -------------------------------------------------------
    @torch.no_grad()
    def _force_pass(self) -> ContactDiagnostics:
        cfg = self.config
        n = self.n
        k_n = float(cfg.contact_stiffness)
        beta = self.damping_ratio()
        mu = float(cfg.friction_coefficient)
        v_eps = float(cfg.friction_velocity_scale)
        forces = torch.zeros((n, 3), dtype=self.dtype)
        max_comp = torch.zeros(n, dtype=self.dtype)
        incident = np.zeros(n)
        elastic = 0.0
        damp_diss = 0.0
        fric_diss = 0.0
        platen_work = 0.0

        pos, vel, rad = self.positions, self.velocities, self.radii
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        d_vec = pos[j_idx] - pos[i_idx]
        dist = torch.linalg.vector_norm(d_vec, dim=1)
        overlap = rad[i_idx] + rad[j_idx] - dist
        active = overlap > 0.0
        pair_contacts = int(active.sum())
        if pair_contacts:
            i_a, j_a = i_idx[active], j_idx[active]
            normal = d_vec[active] / torch.clamp(dist[active], min=1e-15)[:, None]
            delta = overlap[active]
            v_rel = vel[j_a] - vel[i_a]
            v_n = (v_rel * normal).sum(dim=1)
            v_t_vec = v_rel - v_n[:, None] * normal
            v_t = torch.linalg.vector_norm(v_t_vec, dim=1)
            closing = torch.clamp(-v_n, min=0.0)
            m_eff = (self.masses[i_a] * self.masses[j_a]
                     / (self.masses[i_a] + self.masses[j_a]))
            c_pair = 2.0 * beta * torch.sqrt(k_n * m_eff)
            f_n = k_n * delta + c_pair * closing
            spring_energy = 0.5 * k_n * delta ** 2
            elastic += float(spring_energy.sum())
            damp_diss += float((c_pair * closing ** 2).sum())
            np.add.at(incident, i_a.numpy(), 0.5 * spring_energy.numpy())
            np.add.at(incident, j_a.numpy(), 0.5 * spring_energy.numpy())
            f_t_mag = mu * f_n * torch.tanh(v_t / v_eps)
            direction = v_t_vec / torch.clamp(v_t, min=1e-15)[:, None]
            f_t = -f_t_mag[:, None] * direction
            fric_diss += float(torch.sum(f_t_mag * v_t))
            f_vec = f_n[:, None] * normal + f_t
            # ``normal`` points i -> j, so the repulsive force pushes i along
            # -normal and j along +normal.
            forces.index_add_(0, i_a, -f_vec)
            forces.index_add_(0, j_a, f_vec)
            max_comp.scatter_reduce_(0, i_a, f_n, reduce="amax", include_self=True)
            max_comp.scatter_reduce_(0, j_a, f_n, reduce="amax", include_self=True)

        # -- platen contacts (top platen moves along -z when velocity < 0) --
        k_p = self._platen_stiffness()
        mu_p = float(cfg.platen_friction_coefficient)
        v_top = float(self.platen_velocity)
        top_overlap = (pos[:, 2] + rad) - self.top_platen_z
        bottom_overlap = self.bottom_platen_z - (pos[:, 2] - rad)
        platen_contacts = 0
        platen_force_sum = 0.0
        for mask, normal_z, delta_all, platen_vz in (
                (top_overlap > 0.0, -1.0, top_overlap, v_top),
                (bottom_overlap > 0.0, 1.0, bottom_overlap, 0.0)):
            if not bool(mask.any()):
                continue
            platen_contacts += int(mask.sum())
            delta = delta_all[mask]
            v_rel_z = vel[mask, 2] - platen_vz
            v_n = normal_z * v_rel_z                      # n = normal_z * z_hat
            closing = torch.clamp(-v_n, min=0.0)
            c_plane = 2.0 * beta * torch.sqrt(k_p * self.masses[mask])
            f_n = k_p * delta + c_plane * closing
            spring_energy = 0.5 * k_p * delta ** 2
            elastic += float(spring_energy.sum())
            damp_diss += float((c_plane * closing ** 2).sum())
            incident[mask.numpy()] += spring_energy.numpy()
            f_vec = torch.zeros((int(mask.sum()), 3), dtype=self.dtype)
            f_vec[:, 2] = normal_z * f_n
            if mu_p > 0.0:
                v_t_vec = vel[mask].clone()
                v_t_vec[:, 2] = 0.0
                v_t = torch.linalg.vector_norm(v_t_vec, dim=1)
                f_t_mag = mu_p * f_n * torch.tanh(v_t / v_eps)
                direction = v_t_vec / torch.clamp(v_t, min=1e-15)[:, None]
                f_vec -= f_t_mag[:, None] * direction
                fric_diss += float(torch.sum(f_t_mag * v_t))
            forces[mask] += f_vec
            # Only the *moving* platen does work; the weight v_platen / v_top
            # makes ``platen_force_sum * |v_top|`` the exact work rate.
            weight = (platen_vz / v_top) if v_top != 0.0 else 0.0
            platen_force_sum += float(f_n.sum()) * weight
            # Work done by the plane on the particles: F.particle * v_platen.
            platen_work += float(torch.sum(f_vec[:, 2] * platen_vz))
            max_comp[mask] = torch.maximum(max_comp[mask], f_n)

        # -- bonds (axial springs, optional) -------------------------------
        if self.bond_alive.size:
            k_b = float(self.config.bond_stiffness or 0.0)
            pos_np = pos.numpy()
            i_np, j_np = self.bond_i, self.bond_j
            delta_vec = pos_np[j_np] - pos_np[i_np]
            dist_b = np.linalg.norm(delta_vec, axis=1)
            extension = dist_b - self.bond_rest
            self._bond_extension = extension
            self._bond_force = np.where(self.bond_alive, k_b * extension, 0.0)
            energy_b = 0.5 * k_b * extension ** 2 * self.bond_alive
            elastic += float(energy_b.sum())
            np.add.at(incident, i_np, 0.5 * energy_b)
            np.add.at(incident, j_np, 0.5 * energy_b)
            unit = delta_vec / np.maximum(dist_b, 1e-15)[:, None]
            f_bond = self._bond_force[:, None] * unit
            f_bond[~self.bond_alive] = 0.0
            forces += torch.tensor(
                np.zeros((n, 3)) + self._scatter_np(n, i_np, f_bond) - self._scatter_np(n, j_np, f_bond),
                dtype=self.dtype)

        return ContactDiagnostics(
            forces=forces, elastic_energy_J=elastic,
            damping_dissipation_rate_W=damp_diss,
            friction_dissipation_rate_W=fric_diss,
            platen_work_rate_W=platen_work,
            platen_normal_force_N=platen_force_sum,
            max_compressive_force_N=max_comp,
            incident_elastic_energy_J=incident,
            pair_contacts=pair_contacts, platen_contacts=platen_contacts)

    @staticmethod
    def _scatter_np(n: int, index: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.zeros((n, 3))
        np.add.at(out, index, values)
        return out

    def _elastic_energy_snapshot(self) -> float:
        """Elastic energy (contacts + platens + bonds) at the current state."""
        return float(self._force_pass().elastic_energy_J)

    # -- fracture paths ---------------------------------------------------
    def _apply_bond_breaking(self) -> int:
        """Break bonds whose axial force exceeds their Weibull strength."""
        if self.bond_alive.size == 0:
            return 0
        k_b = float(self.config.bond_stiffness or 0.0)
        force = self._bond_force
        if self.config.bond_tension_only:
            broken = self.bond_alive & (force > self.bond_strength)
        else:
            broken = self.bond_alive & (np.abs(force) > self.bond_strength)
        if not bool(broken.any()):
            return 0
        for index in np.nonzero(broken)[0]:
            released = 0.5 * k_b * float(self._bond_extension[index]) ** 2
            i, j = int(self.bond_i[index]), int(self.bond_j[index])
            self._apply_recoil(i, j, released)
            self.bond_alive[index] = False
            self.bond_break_events += 1
            clusters = self.bond_cluster_count()
            if clusters > self._bond_cluster_ref:
                self.bond_cluster_events += clusters - self._bond_cluster_ref
            self._bond_cluster_ref = clusters
        return int(broken.sum())

    def _apply_recoil(self, i: int, j: int, energy_J: float) -> None:
        """Momentum-neutral recoil kick releasing bond energy into both ends."""
        if energy_J <= 0.0:
            return
        delta = (self.positions[j] - self.positions[i]).numpy()
        dist = float(np.linalg.norm(delta))
        if dist <= 0.0:
            return
        unit = delta / dist
        m_i = float(self.masses[i])
        m_j = float(self.masses[j])
        m_eff = m_i * m_j / (m_i + m_j)
        momentum = math.sqrt(2.0 * energy_J * m_eff)
        self.velocities[i] += torch.tensor(-momentum / m_i * unit, dtype=self.dtype)
        self.velocities[j] += torch.tensor(momentum / m_j * unit, dtype=self.dtype)

    def _apply_crushing(self, diag: ContactDiagnostics) -> int:
        """Split every particle whose nominal stress reached its strength."""
        cfg = self.config
        sigma = diag.max_compressive_force_N / (math.pi * self.radii ** 2)
        eligible = sigma >= self.strengths
        eligible &= torch.tensor([self.born_step[k] < self.step_count
                                  for k in range(self.n)], dtype=torch.bool)
        eligible &= torch.tensor([self.generation[k] < cfg.max_generation
                                  for k in range(self.n)], dtype=torch.bool)
        if cfg.min_fragment_diameter > 0.0:
            equivalent = torch.tensor(self.diameters, dtype=self.dtype)
            eligible &= equivalent >= cfg.min_fragment_diameter
        indices = torch.nonzero(eligible).flatten().tolist()
        for index in sorted(indices, reverse=True):
            stress = float(sigma[index])
            self._split_particle(int(index))
            self.breakage_stress_Pa.append(stress)
        return len(indices)

    def _split_particle(self, index: int) -> None:
        """Replace particle ``index`` by its fragments and book the accounting.

        The energy released by the replacement is the **net** drop of the total
        elastic energy across the swap (``U_before - U_after``): a child that
        inherits the parent's platen penetration keeps that energy, so it is not
        available for fracture.  The net drop funds the surface energy and the
        fragment expansion, which makes the ledger identity exact.
        """
        self._sync_particles()
        elastic_before = float(self._force_pass().elastic_energy_J)
        parent = self.particles[index]
        child_strengths = self.config.weibull.sample(
            self.config.fragmenter.k, parent.diameter, generator=self.rng)
        result = self.config.fragmenter.split(
            parent, density=self.config.density, rng=self.rng,
            separation_energy_J=0.0, child_strengths=child_strengths)
        requested = self.ledger.surface_energy_demand(result.surface_area_increase)

        self.fragment_sizes.extend(float(d) for d in result.diameters)
        if parent.generation == 0:
            self.first_generation_fragments.extend(float(d) for d in result.diameters)

        # Drop bonds incident to the dead parent, keep the rest addressable.
        if self.bond_alive.size:
            killed = (self.bond_i == index) | (self.bond_j == index)
            self.bond_alive[killed] = False
            self.bond_i = np.where(self.bond_i > index, self.bond_i - 1, self.bond_i)
            self.bond_j = np.where(self.bond_j > index, self.bond_j - 1, self.bond_j)
            self.bond_i = np.where(killed, 0, self.bond_i)
            self.bond_j = np.where(killed, 0, self.bond_j)
            self._bond_cluster_ref = self.bond_cluster_count()

        del self.born_step[index]
        del self.lineage[index]
        del self.generation[index]
        del self.particles[index]
        count = len(result.children)
        self.particles[index:index] = result.children
        self.born_step[index:index] = [self.step_count] * count
        self.lineage[index:index] = [c.lineage for c in result.children]
        self.generation[index:index] = [c.generation for c in result.children]
        self._load_particles(self.particles)

        if self.config.reseat_platen:
            # Servo re-seat: the spherical fragment proxy does not fill the
            # parent volume (contact_radius_fraction < 1), so the specimen top
            # surface drops at a breakage event.  The displacement-controlled
            # platen is re-seated on the new surface; it is instantaneous and
            # therefore does no work.
            self.top_platen_z = float((self.positions[:, 2] + self.radii).max())
            self.bottom_platen_z = float((self.positions[:, 2] - self.radii).min())

        elastic_after = float(self._force_pass().elastic_energy_J)
        released = max(elastic_before - elastic_after, 0.0)
        booked = min(requested, released)
        expansion = max(released - booked, 0.0)
        if expansion > 0.0:
            result.expansion_energy_J = self.config.fragmenter.distribute_expansion(
                result.children, centre=parent.position, energy_J=expansion)
            self._load_particles(self.particles)
        self.ledger.record_replacement(result, released)
        self.crushing_events += 1

    # -- time loop --------------------------------------------------------
    def step(self) -> None:
        """Advance one step: forces -> fracture -> integrate."""
        dt = self.dt
        before = self.n
        diag = self._force_pass()
        changed = 0
        if self.config.bond_stiffness is not None:
            changed += self._apply_bond_breaking()
        if self.config.crushing_enabled and self.n < self.config.max_particles:
            changed += self._apply_crushing(diag)
        if changed:
            # The fracture paths changed the topology: the force field used for
            # work accounting and integration must be the post-fracture one, and
            # the explicit-stability bound tightens for the lighter children.
            self.dt = min(self.dt, self._recommended_dt())
            diag = self._force_pass()
        # Trapezoidal quadrature in time: the platen force and the dissipation
        # rates oscillate with the contact ringing, so a one-sided rectangle
        # rule would leak O(dt) per step into the energy account.
        force_now = diag.platen_normal_force_N
        rate_now = diag.damping_dissipation_rate_W + diag.friction_dissipation_rate_W
        self.ledger.external_work_J += (
            0.5 * (force_now + self._prev_platen_force) * abs(float(self.platen_velocity)) * dt)
        self.ledger.dissipation_J += 0.5 * (rate_now + self._prev_dissipation_rate) * dt
        self._prev_platen_force = force_now
        self._prev_dissipation_rate = rate_now

        acceleration = diag.forces / self.masses[:, None]
        if self.config.gravity:
            acceleration[:, 2] -= float(self.config.gravity)
        self.velocities += acceleration * dt
        self.positions += self.velocities * dt
        self.top_platen_z += float(self.platen_velocity) * dt
        self.time += dt
        self.step_count += 1
        if not bool(torch.isfinite(self.positions).all()
                    and torch.isfinite(self.velocities).all()):
            raise FloatingPointError("non-finite crushing DEM state; reduce dt")
        if self.step_count % self.config.record_every == 0:
            self.history.append({
                "time_s": self.time,
                "particle_count": float(self.n),
                "kinetic_energy_J": self.kinetic_energy_J,
                "elastic_energy_J": float(diag.elastic_energy_J),
                "external_work_J": self.ledger.external_work_J,
                "dissipation_J": self.ledger.dissipation_J,
                "surface_energy_J": self.ledger.surface_energy_J,
                "max_stress_Pa": float(
                    (diag.max_compressive_force_N / (math.pi * self.radii ** 2)).max()),
                "top_platen_z": float(self.top_platen_z),
                "crushing_events": float(self.crushing_events),
            })
        if before != self.n and self.n > 20000:
            raise FloatingPointError("particle count explosion; check strength/fragmenter")

    def run(self, steps: int) -> dict[str, Any]:
        """Advance ``steps`` steps and return the diagnostics snapshot."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        for _ in range(int(steps)):
            self.step()
        return self.diagnostics()

    # -- diagnostics ------------------------------------------------------
    def energy_audit(self) -> dict[str, float]:
        """Close the energy account at the *current* state."""
        elastic = self._elastic_energy_snapshot()
        return self.ledger.audit(
            elastic_energy_J=elastic, kinetic_energy_J=self.kinetic_energy_J,
            initial_elastic_energy_J=self._initial_elastic_J,
            initial_kinetic_energy_J=self._initial_kinetic_J)

    def diagnostics(self) -> dict[str, Any]:
        diag = self._force_pass()
        report = self.ledger.replacement_report()
        diameters = self.diameters
        return {
            "time_s": self.time,
            "steps": self.step_count,
            "dt_s": self.dt,
            "particle_count": self.n,
            "total_mass_kg": self.total_mass,
            "total_momentum": [float(v) for v in self.total_momentum],
            "kinetic_energy_J": self.kinetic_energy_J,
            "elastic_energy_J": float(diag.elastic_energy_J),
            "external_work_J": self.ledger.external_work_J,
            "dissipation_J": self.ledger.dissipation_J,
            "surface_energy_J": self.ledger.surface_energy_J,
            "unfunded_surface_energy_J": self.ledger.unfunded_surface_energy_J,
            "crushing_events": self.crushing_events,
            "bond_break_events": self.bond_break_events,
            "bond_clusters": self.bond_cluster_count(),
            "bond_derived_fragment_clusters": self.bond_derived_fragment_clusters(),
            "bond_breakage_active": self.bond_breakage_active,
            "pair_contacts": diag.pair_contacts,
            "platen_contacts": diag.platen_contacts,
            "max_stress_Pa": float(
                (diag.max_compressive_force_N / (math.pi * self.radii ** 2)).max()),
            "mean_diameter_m": float(diameters.mean()),
            "min_diameter_m": float(diameters.min()),
            "max_diameter_m": float(diameters.max()),
            "diameters_m": [float(d) for d in diameters],
            "replacement_residuals": report,
            "top_platen_z": float(self.top_platen_z),
            "bottom_platen_z": float(self.bottom_platen_z),
            "energy_audit": self.energy_audit(),
        }
