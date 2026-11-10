"""Single-granule crushing verification surface (P97-N02).

Scope
-----
A verification surface for DEM single-granule crushing, built from three
coupled, independently testable pieces:

1. **Weibull strength with a size effect** (the "material law")::

       sigma_f(V) = sigma_0 * (V / V_0) ** (-1 / m)
       P_f(sigma, V) = 1 - exp[ -(sigma / sigma_0) ** m * (V / V_0) ]

   ``sigma_0`` is the characteristic strength at reference volume ``V_0``,
   ``m`` the Weibull modulus.  For a sphere of radius ``r`` with
   ``V = (4/3) pi r^3`` the size law reads ``sigma_f(r) = sigma_0 (r/r_0)^(-3/m)``.
   The median (50 % fractile) failure stress has the closed form

       sigma_med(V) = sigma_0 (V_0/V)^(1/m) (ln 2)^(1/m)

   which is used as the analytic reference of the simulation.

2. **Sub-granule replacement with a mass/momentum conservation ledger.**
   A parent granule of mass ``m_p`` and velocity ``v_p`` is replaced by
   ``N`` fragments whose diameters are drawn from a Gaudin-Schuhmann
   spectrum and then *uniformly rescaled* so that

       sum_i m_i = m_p          (exact, up to floating point)

   because the rescaling preserves the Gaudin shape (same ``alpha``,
   rescaled ``d_max``).  Fragment velocities are ``v_i = v_p + c_i`` with
   a mass-weighted zero-mean correction ``c_i = a_i - sum_j m_j a_j / m_p``
   so that

       sum_i m_i v_i = m_p v_p   (exact, up to floating point)

   i.e. the fracture impulse ``Delta p_fracture`` recorded by the ledger
   is zero for a closed event.  Every event is appended to a
   :class:`CrushLedger` that stores the per-event mass and momentum
   residuals and can be serialised to JSON/CSV for audit.

3. **Gaudin-Schuhmann fragment-size spectrum**::

       F(d) = (d / d_max) ** alpha          (number CDF, 0 < d <= d_max)
       f(d) = alpha * d**(alpha-1) / d_max**alpha

   with two alpha estimators (MLE with the observed maximum as ``d_max``,
   and a log-log CDF fit).  The mass-weighted CDF of the same spectrum is
   ``F_m(d) = (d/d_max) ** (alpha + 3)`` (mass in ``[d, d+dd]`` is
   proportional to ``d^3 f(d) dd``), a closed form used by the tests.

Crushing simulation
-------------------
``crush_single_granule`` loads a single sphere of radius ``r`` between two
rigid flat platens closing at ``platen_velocity``.  Each contact is a
sphere-on-plane Hertz contact, ``F_c = (4/3) E* sqrt(r) delta^(3/2)`` with
``E* = E / (2 (1 - nu^2))``; the total force is twice that (symmetric
loading) and the nominal (projected-area) stress is
``sigma = F / (pi r^2)``.  The loading is quasi-static: the platens impose
the displacement history and the granule inertia is neglected (standard for
single-granule crushing tests).  Failure is sampled exactly for the
monotonic load path: one uniform draw ``U`` per trial, and the granule
fails at the first time step where the *cumulative* Weibull probability
``P_f(sigma(t))`` reaches ``U`` (inverse-CDF sampling on the time grid).
With zero platen velocity the stress stays zero and no granule can fail
(the negative control).

Honesty note (read before citing numbers)
------------------------------------------
The "experimental" data sets used by the acceptance runner are
*manufactured solutions*: they are generated from the analytic Weibull law
and the analytic Gaudin spectrum with known parameters, then recovered by
the simulation + fitting pipeline.  The acceptance criteria therefore
validate the pipeline (simulation, sampling, conservation, fitting), not
the material parameters of real quartz or limestone.  Real-data
benchmarking is out of scope for this node and is flagged as a limitation.

All quantities are SI (kg, m, s, Pa).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

#: Volume of a sphere of radius ``r``.
def sphere_volume(radius: float) -> float:
    return (4.0 / 3.0) * math.pi * radius ** 3


# ---------------------------------------------------------------------------
# 1. Weibull strength model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeibullStrength:
    """Two-parameter Weibull strength with a power-law size effect.

    ``sigma_0`` : characteristic strength at reference volume ``V_0`` [Pa]
    ``m``       : Weibull modulus (dimensionless, > 0)
    ``V_0``     : reference volume [m^3]
    """

    sigma_0: float
    m: float
    V_0: float

    def __post_init__(self) -> None:
        if self.sigma_0 <= 0.0:
            raise ValueError(f"sigma_0 must be > 0, got {self.sigma_0}")
        if self.m <= 0.0:
            raise ValueError(f"Weibull modulus m must be > 0, got {self.m}")
        if self.V_0 <= 0.0:
            raise ValueError(f"V_0 must be > 0, got {self.V_0}")

    def failure_stress(self, volume: float) -> float:
        """Deterministic size-effect strength ``sigma_f(V)`` [Pa]."""
        if volume <= 0.0:
            raise ValueError(f"volume must be > 0, got {volume}")
        return self.sigma_0 * (volume / self.V_0) ** (-1.0 / self.m)

    def failure_probability(self, sigma: float, volume: float) -> float:
        """Cumulative failure probability at stress ``sigma`` [Pa]."""
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if volume <= 0.0:
            raise ValueError(f"volume must be > 0, got {volume}")
        x = (sigma / self.sigma_0) ** self.m * (volume / self.V_0)
        return 1.0 - math.exp(-x)

    def median_failure_stress(self, volume: float) -> float:
        """50 % fractile of the failure-stress distribution [Pa]."""
        return (
            self.sigma_0
            * (volume / self.V_0) ** (-1.0 / self.m)
            * math.log(2.0) ** (1.0 / self.m)
        )

    def mean_failure_stress(self, volume: float) -> float:
        """Mean failure stress, ``sigma_0 (V_0/V)^(1/m) Gamma(1+1/m)`` [Pa]."""
        from scipy.special import gamma

        return (
            self.sigma_0
            * (volume / self.V_0) ** (-1.0 / self.m)
            * float(gamma(1.0 + 1.0 / self.m))
        )


@dataclass(frozen=True)
class WeibullFitResult:
    """Result of fitting ``sigma_f(d) = C d^(-3/m)`` on log-log data."""

    m: float
    C: float
    r2: float
    slope: float
    n_points: int

    def sigma_0_at(self, V_0: float) -> float:
        """Characteristic strength ``sigma_0`` at reference volume ``V_0``.

        The fitted ``C`` is the median-law prefactor
        ``C = sigma_0 V_0^(1/m) (ln 2)^(1/m)``.
        """
        return self.C * V_0 ** (-1.0 / self.m) * math.log(2.0) ** (-1.0 / self.m)


def fit_weibull_size_law(d: Sequence[float], sigma_f: Sequence[float]) -> WeibullFitResult:
    """Least-squares log-log fit of ``log sigma = log C - (3/m) log d``.

    Returns the Weibull modulus ``m = -3/slope``, the prefactor ``C``,
    and the coefficient of determination ``R^2`` in linear space.
    """
    d = np.asarray(d, dtype=float)
    sigma_f = np.asarray(sigma_f, dtype=float)
    if d.shape != sigma_f.shape:
        raise ValueError("d and sigma_f must have the same shape")
    if d.size < 3:
        raise ValueError("need at least 3 (d, sigma_f) pairs for a fit")
    if np.any(d <= 0.0) or np.any(sigma_f <= 0.0):
        raise ValueError("d and sigma_f must be strictly positive")
    x = np.log(d)
    y = np.log(sigma_f)
    A = np.column_stack([x, np.ones_like(x)])
    (slope, log_C), *_ = np.linalg.lstsq(A, y, rcond=None)
    m = -3.0 / float(slope)
    C = float(math.exp(log_C))
    pred = C * d ** (-3.0 / m)
    ss_res = float(np.sum((sigma_f - pred) ** 2))
    ss_tot = float(np.sum((sigma_f - sigma_f.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return WeibullFitResult(m=m, C=C, r2=r2, slope=float(slope), n_points=int(d.size))


# ---------------------------------------------------------------------------
# 2. Gaudin-Schuhmann spectrum
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaudinSpectrum:
    """Gaudin-Schuhmann number CDF ``F(d) = (d/d_max)^alpha``."""

    alpha: float
    d_max: float

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.d_max <= 0.0:
            raise ValueError(f"d_max must be > 0, got {self.d_max}")

    def cdf(self, d: Sequence[float] | float) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return np.clip(d / self.d_max, 0.0, 1.0) ** self.alpha

    def pdf(self, d: Sequence[float] | float) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d)
        pos = d > 0.0
        out[pos] = self.alpha * d[pos] ** (self.alpha - 1.0) / self.d_max ** self.alpha
        return out

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Inverse-CDF sampling of ``n`` fragment diameters [m]."""
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        u = rng.random(n)
        return self.d_max * u ** (1.0 / self.alpha)

    def fit_alpha_mle(self, d: Sequence[float], d_max: Optional[float] = None) -> float:
        """MLE of ``alpha`` with ``d_max`` fixed (default: observed max).

        For ``F(d) = (d/d_max)^alpha`` the log-likelihood is
        ``L = alpha sum log(d_max/d_i) - n log d_max``; setting
        ``dL/dalpha = 0`` gives ``alpha_hat = n / sum log(d_max/d_i)``.
        """
        d = np.asarray(d, dtype=float)
        if d.size == 0:
            raise ValueError("no samples")
        if np.any(d <= 0.0):
            raise ValueError("diameters must be > 0")
        d_max = float(d.max()) if d_max is None else float(d_max)
        if d_max <= 0.0:
            raise ValueError("d_max must be > 0")
        return float(d.size / np.sum(np.log(d_max / d)))

    def fit_alpha_cdf(self, d: Sequence[float], d_max: Optional[float] = None) -> float:
        """Log-log CDF fit: ``log F = alpha log d - alpha log d_max``."""
        d = np.sort(np.asarray(d, dtype=float))
        if d.size < 3:
            raise ValueError("need at least 3 samples")
        d_max = float(d.max()) if d_max is None else float(d_max)
        F = np.arange(1, d.size + 1) / (d.size + 1)
        A = np.column_stack([np.log(d), np.full_like(np.log(d), -math.log(d_max))])
        sol, *_ = np.linalg.lstsq(A, np.log(F), rcond=None)
        return float(sol[0])


# ---------------------------------------------------------------------------
# 3. Sub-granule replacement + conservation ledger
# ---------------------------------------------------------------------------
@dataclass
class FragmentSet:
    """Fragments produced by one crushing event (SI units).

    ``scale`` is the uniform radius rescaling applied to the drawn
    Gaudin diameters to enforce the mass balance; the event's effective
    spectrum cutoff is therefore ``scale * spectrum.d_max``.
    """

    radii: np.ndarray          # (N,) [m]
    masses: np.ndarray          # (N,) [kg]
    velocities: np.ndarray      # (N, 3) [m/s]
    scale: float = 1.0

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())

    @property
    def total_momentum(self) -> np.ndarray:
        return self.masses @ self.velocities


def replace_granule(
    parent_radius: float,
    parent_mass: float,
    parent_velocity: Sequence[float],
    spectrum: GaudinSpectrum,
    n_fragments: int,
    rng: np.random.Generator,
    velocity_spread: float = 0.1,
) -> FragmentSet:
    """Replace a crushed parent granule by ``n_fragments`` children.

    Fragment diameters are drawn from ``spectrum`` and uniformly rescaled
    so that ``sum r_i^3 = r_parent^3``; with equal density this makes the
    mass balance exact while preserving the Gaudin shape.  Fragment
    velocities are the parent velocity plus a mass-weighted zero-mean
    random spread of relative scale ``velocity_spread * |v_p|`` (or of
    ``velocity_spread`` m/s when the parent is at rest), which makes the
    momentum balance exact.
    """
    if parent_radius <= 0.0:
        raise ValueError(f"parent_radius must be > 0, got {parent_radius}")
    if parent_mass <= 0.0:
        raise ValueError(f"parent_mass must be > 0, got {parent_mass}")
    if n_fragments < 2:
        raise ValueError(f"n_fragments must be >= 2, got {n_fragments}")
    v_p = np.asarray(parent_velocity, dtype=float).reshape(3)
    v_scale = max(float(np.linalg.norm(v_p)), 1e-12)

    d = spectrum.sample(n_fragments, rng)
    r = 0.5 * d
    scale = (parent_radius ** 3 / float(np.sum(r ** 3))) ** (1.0 / 3.0)
    r = scale * r
    masses = parent_mass * (r / parent_radius) ** 3

    sigma_c = velocity_spread * v_scale
    a = rng.normal(0.0, sigma_c, size=(n_fragments, 3))
    c = a - (masses @ a) / parent_mass
    velocities = v_p[None, :] + c
    return FragmentSet(radii=r, masses=masses, velocities=velocities, scale=float(scale))


@dataclass
class CrushLedger:
    """Per-event conservation ledger for sub-granule replacement."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        event_id: int,
        parent_mass: float,
        parent_velocity: Sequence[float],
        fragments: FragmentSet,
    ) -> dict[str, Any]:
        v_p = np.asarray(parent_velocity, dtype=float).reshape(3)
        p_parent = parent_mass * v_p
        p_frag = fragments.total_momentum
        dp = p_frag - p_parent
        mass_res = abs(fragments.total_mass - parent_mass) / parent_mass
        mom_scale = parent_mass * float(np.linalg.norm(v_p))
        mom_res = float(np.linalg.norm(dp)) / mom_scale if mom_scale > 0.0 else 0.0
        entry = {
            "event_id": int(event_id),
            "n_fragments": int(fragments.radii.size),
            "parent_mass_kg": float(parent_mass),
            "fragment_mass_kg": fragments.total_mass,
            "mass_residual_rel": float(mass_res),
            "momentum_residual_rel": mom_res,
            "delta_p_fracture_Ns": dp.tolist(),
            "parent_speed_m_s": float(np.linalg.norm(v_p)),
        }
        self.events.append(entry)
        return entry

    @property
    def max_mass_residual(self) -> float:
        return max((e["mass_residual_rel"] for e in self.events), default=0.0)

    @property
    def max_momentum_residual(self) -> float:
        return max((e["momentum_residual_rel"] for e in self.events), default=0.0)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.events, indent=2), encoding="utf-8")

    def to_csv(self, path: Path) -> None:
        if not self.events:
            path.write_text(
                "event_id,n_fragments,parent_mass_kg,fragment_mass_kg,"
                "mass_residual_rel,momentum_residual_rel,parent_speed_m_s\n",
                encoding="utf-8",
            )
            return
        lines = [
            "event_id,n_fragments,parent_mass_kg,fragment_mass_kg,"
            "mass_residual_rel,momentum_residual_rel,parent_speed_m_s"
        ]
        for e in self.events:
            lines.append(
                f"{e['event_id']},{e['n_fragments']},{e['parent_mass_kg']:.12e},"
                f"{e['fragment_mass_kg']:.12e},{e['mass_residual_rel']:.3e},"
                f"{e['momentum_residual_rel']:.3e},{e['parent_speed_m_s']:.6e}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Single-granule crushing simulation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GranuleCrushConfig:
    """Quasi-static two-platen Hertz loading of a single sphere."""

    radius: float
    density: float
    youngs_modulus: float
    poisson_ratio: float
    platen_velocity: float
    dt: float
    t_max: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {self.radius}")
        if self.density <= 0.0:
            raise ValueError(f"density must be > 0, got {self.density}")
        if self.youngs_modulus <= 0.0:
            raise ValueError(f"youngs_modulus must be > 0, got {self.youngs_modulus}")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError(f"poisson_ratio must be in [0, 0.5), got {self.poisson_ratio}")
        if self.platen_velocity < 0.0:
            raise ValueError(f"platen_velocity must be >= 0, got {self.platen_velocity}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.t_max <= self.dt:
            raise ValueError("t_max must be > dt")

    @property
    def volume(self) -> float:
        return sphere_volume(self.radius)

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def hertz_coefficient(self) -> float:
        """Total force coefficient: ``F = c * delta^(3/2)`` for both contacts."""
        e_star = self.youngs_modulus / (2.0 * (1.0 - self.poisson_ratio ** 2))
        return 2.0 * (4.0 / 3.0) * e_star * math.sqrt(self.radius)

    def stress_at(self, t: float) -> float:
        """Nominal (projected-area) stress at time ``t`` [Pa]."""
        delta = self.platen_velocity * t
        if delta <= 0.0:
            return 0.0
        force = self.hertz_coefficient * delta ** 1.5
        return force / (math.pi * self.radius ** 2)


@dataclass
class CrushResult:
    """Outcome of a batch of single-granule crushing trials."""

    n_trials: int
    n_failed: int
    failure_stress: np.ndarray  # (n_trials,) Pa, NaN where no failure
    failure_time: np.ndarray    # (n_trials,) s,  NaN where no failure

    @property
    def median_failure_stress(self) -> float:
        vals = self.failure_stress[np.isfinite(self.failure_stress)]
        return float(np.median(vals)) if vals.size else float("nan")


def crush_single_granule(
    cfg: GranuleCrushConfig,
    strength: WeibullStrength,
    rng: np.random.Generator,
    n_trials: int = 1,
) -> CrushResult:
    """Quasi-static two-platen Hertz loading with Weibull stochastic failure.

    One uniform draw ``U`` per trial; the granule fails at the first step
    where the cumulative probability ``P_f(sigma(t))`` reaches ``U``
    (exact inverse-CDF sampling for the monotonic load path).
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be > 0, got {n_trials}")
    U = rng.random(n_trials)
    failed = np.zeros(n_trials, dtype=bool)
    sigma_f = np.full(n_trials, np.nan)
    t_f = np.full(n_trials, np.nan)
    V = cfg.volume
    n_steps = int(math.ceil(cfg.t_max / cfg.dt))
    for k in range(1, n_steps + 1):
        if failed.all():
            break
        t = k * cfg.dt
        sigma = cfg.stress_at(t)
        if sigma <= 0.0:
            continue
        P = strength.failure_probability(sigma, V)
        hit = (~failed) & (U <= P)
        if hit.any():
            sigma_f[hit] = sigma
            t_f[hit] = t
            failed |= hit
    return CrushResult(
        n_trials=n_trials,
        n_failed=int(failed.sum()),
        failure_stress=sigma_f,
        failure_time=t_f,
    )


# ---------------------------------------------------------------------------
# 5. Acceptance studies (C1..C5)
# ---------------------------------------------------------------------------
def run_weibull_size_study(
    strength: WeibullStrength,
    material: dict[str, float],
    diameters: Sequence[float],
    n_trials: int,
    seed: int,
    platen_velocity: float = 0.01,
) -> dict[str, Any]:
    """C1: crush granules of several sizes, fit the Weibull size law.

    The "experimental" data are manufactured from the same analytic law
    (see module docstring, honesty note).  Returns the per-size median
    failure stresses, the fit, and the C1 verdict.
    """
    rng = np.random.default_rng(seed)
    medians: list[float] = []
    per_size: list[dict[str, Any]] = []
    for d in diameters:
        r = 0.5 * d
        cfg = GranuleCrushConfig(
            radius=r,
            density=material["density"],
            youngs_modulus=material["youngs_modulus"],
            poisson_ratio=material["poisson_ratio"],
            platen_velocity=platen_velocity,
            dt=material["dt"],
            t_max=material["t_max"],
        )
        res = crush_single_granule(cfg, strength, rng, n_trials=n_trials)
        med = res.median_failure_stress
        medians.append(med)
        per_size.append(
            {
                "d_m": d,
                "n_trials": n_trials,
                "n_failed": res.n_failed,
                "median_sigma_f_Pa": med,
                "analytic_median_Pa": strength.median_failure_stress(cfg.volume),
            }
        )
    fit = fit_weibull_size_law(list(diameters), medians)
    sigma0_fit = fit.sigma_0_at(strength.V_0)
    m_dev = abs(fit.m - strength.m) / strength.m
    verdict = bool(fit.r2 >= 0.95 and m_dev <= 0.20)
    return {
        "criterion": "C1",
        "verdict": "PASS" if verdict else "FAIL",
        "r2": fit.r2,
        "m_true": strength.m,
        "m_fit": fit.m,
        "m_rel_dev": m_dev,
        "sigma0_true_Pa": strength.sigma_0,
        "sigma0_fit_Pa": sigma0_fit,
        "sigma0_rel_dev": abs(sigma0_fit - strength.sigma_0) / strength.sigma_0,
        "fit_prefactor_C": fit.C,
        "per_size": per_size,
    }


def run_gaudin_spectrum_benchmark(
    alpha_true: float,
    d_max: float,
    n_samples: int,
    n_events: int,
    seed: int,
    density: float = 2650.0,
) -> dict[str, Any]:
    """C4: recover the Gaudin exponent from synthetic spectra.

    Three routes, all must recover ``alpha`` within 15 %:

    * ``direct``  -- sample ``n_samples`` diameters from the analytic
      spectrum and fit with both estimators (the "experiment" of C4).
    * ``pipeline`` -- run ``n_events`` full DEM replacement events and pool
      all fragments.  The per-event mass-conservation rescaling changes the
      effective ``d_max`` of every event, so the pooled sample is a mixture
      of Gaudin laws with different cutoffs; the log-log CDF fit is robust
      to that mixture, while the MLE (which assumes one global ``d_max``)
      is *known to be biased low* and is reported for transparency, not
      judged.
    * ``per_event`` -- the rescaling factor of every event is known, so
      ``x_i = d_i / (scale_i * d_max)`` is an *exact* Gaudin(``alpha``, 1)
      sample; the pooled normalised sample is fitted and its CDF checked
      against ``x^alpha``.
    """
    rng = np.random.default_rng(seed)
    spectrum = GaudinSpectrum(alpha=alpha_true, d_max=d_max)

    d_direct = spectrum.sample(n_samples, rng)
    alpha_mle = spectrum.fit_alpha_mle(d_direct)
    alpha_cdf = spectrum.fit_alpha_cdf(d_direct)

    all_frags: list[np.ndarray] = []
    per_event: list[dict[str, Any]] = []
    for _ in range(n_events):
        r_p = 0.5 * d_max * rng.uniform(0.5, 1.0)
        m_p = density * sphere_volume(r_p)
        v_p = rng.normal(0.0, 0.05, size=3)
        n_f = int(rng.integers(4, 13))
        frags = replace_granule(r_p, m_p, v_p, spectrum, n_f, rng)
        d = 2.0 * frags.radii
        all_frags.append(d)
        per_event.append({"d": d, "d_max_eff": frags.scale * d_max})
    d_pipeline = np.concatenate(all_frags)
    alpha_pipe_cdf = spectrum.fit_alpha_cdf(d_pipeline, d_max=d_max)
    alpha_pipe_mle = spectrum.fit_alpha_mle(d_pipeline, d_max=d_max)

    # per-event shape check: x_i = d_i / (scale_i * d_max) is an exact
    # Gaudin(alpha, 1) sample; pool over events and fit + CDF-check.
    x_pool = np.concatenate([ev["d"] / ev["d_max_eff"] for ev in per_event])
    alpha_norm_mle = spectrum.fit_alpha_mle(x_pool, d_max=1.0)
    alpha_norm_cdf = spectrum.fit_alpha_cdf(x_pool, d_max=1.0)
    grid = np.linspace(0.05, 1.0, 20)
    F_emp = np.array([(x_pool <= x).mean() for x in grid])
    cdf_dev = float(np.abs(F_emp - grid ** alpha_true).max())

    devs = {
        "direct_mle": abs(alpha_mle - alpha_true) / alpha_true,
        "direct_cdf": abs(alpha_cdf - alpha_true) / alpha_true,
        "pipeline_cdf": abs(alpha_pipe_cdf - alpha_true) / alpha_true,
        "per_event_mle": abs(alpha_norm_mle - alpha_true) / alpha_true,
        "per_event_cdf": abs(alpha_norm_cdf - alpha_true) / alpha_true,
    }
    verdict = all(v <= 0.15 for v in devs.values()) and cdf_dev <= 0.05
    return {
        "criterion": "C4",
        "verdict": "PASS" if verdict else "FAIL",
        "alpha_true": alpha_true,
        "alpha_direct_mle": alpha_mle,
        "alpha_direct_cdf": alpha_cdf,
        "alpha_pipeline_cdf": alpha_pipe_cdf,
        "alpha_pipeline_mle_biased": alpha_pipe_mle,
        "alpha_per_event_mle": alpha_norm_mle,
        "alpha_per_event_cdf": alpha_norm_cdf,
        "per_event_cdf_max_dev": cdf_dev,
        "rel_dev": devs,
        "n_direct_samples": n_samples,
        "n_pipeline_fragments": int(d_pipeline.size),
        "n_per_event_normalized": int(x_pool.size),
        "d_samples_direct": d_direct,
        "d_samples_pipeline": d_pipeline,
    }


def run_strain_rate_study(
    strength: WeibullStrength,
    material: dict[str, float],
    spectrum: GaudinSpectrum,
    radius: float,
    platen_velocities: Sequence[float],
    n_events: int,
    seed: int,
) -> dict[str, Any]:
    """Limestone-style study: Gaudin ``alpha`` recovered at several strain rates.

    For each platen velocity (strain rate ``eps_dot = v / r``) the crushing
    simulation decides which granules fail, the replacement pipeline
    generates the fragments, and the pooled (per-event scale-normalised)
    spectrum is fitted with the log-log CDF method.  The model has no
    rate-dependent strength, so ``alpha`` is expected to be rate-invariant;
    the study validates the pipeline at each rate and reports the spread.
    """
    rows: list[dict[str, Any]] = []
    for v in platen_velocities:
        rng = np.random.default_rng(seed + int(round(v * 1e6)))
        cfg = GranuleCrushConfig(
            radius=radius,
            density=material["density"],
            youngs_modulus=material["youngs_modulus"],
            poisson_ratio=material["poisson_ratio"],
            platen_velocity=v,
            dt=material["dt"],
            t_max=material["t_max"],
        )
        res = crush_single_granule(cfg, strength, rng, n_trials=n_events)
        n_failed = res.n_failed
        frags: list[np.ndarray] = []
        for _ in range(n_failed):
            m_p = material["density"] * sphere_volume(radius)
            v_p = rng.normal(0.0, 0.05, size=3)
            n_f = int(rng.integers(4, 13))
            fr = replace_granule(radius, m_p, v_p, spectrum, n_f, rng)
            frags.append(2.0 * fr.radii)
        d_pool = np.concatenate(frags) if frags else np.array([0.0])
        alpha_cdf = spectrum.fit_alpha_cdf(d_pool, d_max=spectrum.d_max) if frags else float("nan")
        rows.append(
            {
                "platen_velocity_m_s": v,
                "strain_rate_1_s": v / radius,
                "n_crushed": n_failed,
                "n_fragments": int(d_pool.size),
                "alpha_fit_cdf": alpha_cdf,
                "alpha_rel_dev": abs(alpha_cdf - spectrum.alpha) / spectrum.alpha,
            }
        )
    devs = [r["alpha_rel_dev"] for r in rows if np.isfinite(r["alpha_rel_dev"])]
    verdict = bool(devs) and all(d <= 0.15 for d in devs)
    return {
        "criterion": "C4-rate",
        "verdict": "PASS" if verdict else "FAIL",
        "alpha_true": spectrum.alpha,
        "rows": rows,
    }


def run_conservation_ledger(
    n_events: int,
    spectrum: GaudinSpectrum,
    seed: int,
    d_range: tuple[float, float] = (0.5e-3, 8.0e-3),
    density: float = 2650.0,
    speed_range: tuple[float, float] = (0.0, 0.5),
) -> dict[str, Any]:
    """C2/C3: mass and momentum conservation over ``n_events`` events."""
    rng = np.random.default_rng(seed)
    ledger = CrushLedger()
    for i in range(n_events):
        r_p = float(rng.uniform(*d_range))
        m_p = density * sphere_volume(r_p)
        speed = float(rng.uniform(*speed_range))
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-15)
        v_p = speed * direction
        n_f = int(rng.integers(4, 13))
        frags = replace_granule(r_p, m_p, v_p, spectrum, n_f, rng)
        ledger.record(i, m_p, v_p, frags)
    max_mass = ledger.max_mass_residual
    max_mom = ledger.max_momentum_residual
    verdict = bool(max_mass <= 1e-3 and max_mom <= 1e-3)
    return {
        "criterion": "C2+C3",
        "verdict": "PASS" if verdict else "FAIL",
        "n_events": n_events,
        "max_mass_residual_rel": max_mass,
        "max_momentum_residual_rel": max_mom,
        "bound": 1e-3,
        "ledger": ledger,
    }


def run_negative_control(
    strength: WeibullStrength,
    material: dict[str, float],
    radius: float,
    t_run: float,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    """C5: zero platen load must produce zero crushing events."""
    rng = np.random.default_rng(seed)
    cfg = GranuleCrushConfig(
        radius=radius,
        density=material["density"],
        youngs_modulus=material["youngs_modulus"],
        poisson_ratio=material["poisson_ratio"],
        platen_velocity=0.0,
        dt=material["dt"],
        t_max=t_run,
    )
    res = crush_single_granule(cfg, strength, rng, n_trials=n_trials)
    verdict = res.n_failed == 0
    return {
        "criterion": "C5",
        "verdict": "PASS" if verdict else "FAIL",
        "n_trials": n_trials,
        "n_crushed": res.n_failed,
        "t_run_s": t_run,
    }


def run_acceptance_suite(out_dir: Path, seed: int = 20261108) -> dict[str, Any]:
    """Run C1..C5, persist JSON/CSV artifacts, return the report dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quartz = {
        "density": 2650.0,
        "youngs_modulus": 37.0e9,
        "poisson_ratio": 0.17,
        "dt": 1.0e-6,
        "t_max": 200.0e-3,
    }
    strength = WeibullStrength(sigma_0=100.0e6, m=7.0, V_0=sphere_volume(1.0e-3))
    diameters = [0.5e-3, 1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3]

    c1 = run_weibull_size_study(strength, quartz, diameters, n_trials=300, seed=seed)
    c4 = run_gaudin_spectrum_benchmark(
        alpha_true=1.5, d_max=10.0e-3, n_samples=20000, n_events=1000, seed=seed + 1
    )
    c23 = run_conservation_ledger(
        n_events=1000,
        spectrum=GaudinSpectrum(alpha=1.5, d_max=8.0e-3),
        seed=seed + 2,
    )
    c5 = run_negative_control(strength, quartz, radius=1.0e-3, t_run=50.0e-3, n_trials=50, seed=seed + 3)

    limestone = {
        "density": 2500.0,
        "youngs_modulus": 50.0e9,
        "poisson_ratio": 0.25,
        "dt": 1.0e-6,
        "t_max": 200.0e-3,
    }
    limestone_strength = WeibullStrength(sigma_0=30.0e6, m=5.0, V_0=sphere_volume(1.0e-3))
    c4_rate = run_strain_rate_study(
        strength=limestone_strength,
        material=limestone,
        spectrum=GaudinSpectrum(alpha=1.5, d_max=8.0e-3),
        radius=2.0e-3,
        platen_velocities=(0.001, 0.01, 0.1),
        n_events=300,
        seed=seed + 4,
    )

    report = {
        "protocol": "tensordem-granule-crush-acceptance-v1",
        "seed": seed,
        "criteria": {"C1": c1, "C2C3": c23, "C4": c4, "C4_rate": c4_rate, "C5": c5},
    }
    report["verdict"] = "PASS" if all(
        c["verdict"] == "PASS" for c in (c1, c23, c4, c4_rate, c5)
    ) else "FAIL"

    # persist (arrays stripped from the JSON)
    c4_json = {k: v for k, v in c4.items() if not isinstance(v, np.ndarray)}
    c23_json = {k: v for k, v in c23.items() if k != "ledger"}
    report["criteria"]["C4"] = c4_json
    report["criteria"]["C2C3"] = c23_json
    report["criteria"]["C4_rate"] = c4_rate
    (out_dir / "acceptance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    with open(out_dir / "weibull_size_law.csv", "w", encoding="utf-8") as fh:
        fh.write("d_m,median_sigma_f_Pa,analytic_median_Pa\n")
        for row in c1["per_size"]:
            fh.write(f"{row['d_m']:.6e},{row['median_sigma_f_Pa']:.6e},"
                     f"{row['analytic_median_Pa']:.6e}\n")
    c23["ledger"].to_csv(out_dir / "conservation_ledger.csv")
    np.savez_compressed(
        out_dir / "gaudin_samples.npz",
        direct=c4["d_samples_direct"],
        pipeline=c4["d_samples_pipeline"],
    )
    return report
