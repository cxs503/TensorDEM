"""Andrade (fractional) primary creep and Mittag-Leffler relaxation for DEM ice.

Why this module exists next to :mod:`tensordem.ice_creep`
---------------------------------------------------------
:mod:`tensordem.ice_creep` (mission m6910 / IN13-W3-N02) carries the
*Glen / Norton* side of sustained-load ice behaviour: a Maxwell bond with a
non-linear Norton dashpot, a Kelvin branch for primary and Kachanov-Rabotnov
damage for tertiary creep.  The Kelvin branch relaxes *exponentially*
(``eps_K ~ 1 - exp(-t/tau_K)``), so it cannot represent the algebraic start of
primary creep, whereas the acceptance criterion of this node is written against
the Andrade / Bailey three-part law

    eps(t) = eps_0 + eps_1 t^(1/3) + eps_dot_s t
             (Andrade 1910; Glen 1955; the form Cole 1998 fits to lab creep)

The exact one-parameter DEM representation is a **fractional element**
(Scott-Blair "springpot"):

    sigma(t) = eta_alpha D^alpha eps(t),       0 < alpha < 1,

whose creep compliance under constant stress is

    J(t) = t^alpha / (eta_alpha Gamma(1 + alpha)),

i.e. exactly ``eps_1 t^(1/3)`` for ``alpha = 1/3``.  In series with the bond
spring it is a *fractional Maxwell* element whose stress relaxation is the
Mittag-Leffler function ``E_alpha(-(t/tau)^alpha)`` with
``tau = (eta_alpha / E)^(1/alpha)``:

* ``alpha = 1``   -> Newtonian dashpot -> ``exp(-t/tau)``   (limit / control)
* ``alpha = 1/2`` -> ``exp(x^2) erfc(x)``                    (exact check)
* ``alpha = 1/3`` -> Andrade primary creep                   (mission target)

What the benchmark suite pins down
----------------------------------
* the Andrade exponent ``alpha`` is recoverable from a numerically integrated
  DEM creep record to well inside 5 %,
* the elastic offset ``eps_0`` equals ``sigma/E`` (the ``eta -> inf`` /
  ``A -> 0`` no-creep limit),
* the relaxation time obeys ``tau ~ eta_alpha^(-1/alpha)`` (``~ A^-1`` for the
  Norton fluidity ``A``) -- the observable behind the "contact force decays as
  ``exp(-t/tau)`` with ``tau`` correlated to the creep parameter" criterion,
* a *single* exponential is exact only for ``alpha = 1``; for ``alpha = 1/3``
  the decay is slower than any exponential and the exponential-fit quality is
  reported, not hidden.

Numerics
--------
Fractional derivative: L1 scheme
``D^alpha y(t_n) ~ dt^-alpha / Gamma(2-alpha) sum_k b_k (y^{n-k} - y^{n-k-1})``
with ``b_k = (k+1)^(1-alpha) - k^(1-alpha)``.  Unconditionally stable for this
sub-diffusive problem and second-order accurate in the power-law regime, so a
fixed step is enough.  The fixed-displacement relaxation is linear in
``sigma^m`` and is solved in closed form each step (no Newton loop).

Start-up kernel: the plain L1 increment on the very first step underestimates
the creep by ``Gamma(1+alpha) Gamma(2-alpha)`` (0.806 at alpha = 1/3) because
``t^alpha`` is singular at the origin.  The first increment therefore uses the
exact local solution of a step load, ``dt^alpha/ Gamma(1+alpha)``, which removes
the start-up artefact; the benchmark suite reports the error with and without
this kernel so the claim is auditable.

Mittag-Leffler: convergent power series for ``x <= 3`` (at most ~9 digits of
cancellation in double precision) and the optimally truncated asymptotic series
for ``x > 3``.  Both branches are verified in
``tests/test_ice_creep_andrade_p46w3n02.py`` against ``exp(-x)`` (alpha = 1),
``exp(x^2) erfc(x)`` (alpha = 1/2, exact) and, when mpmath is available,
against Talbot Laplace inversion.

Limitations (stated, not hidden)
--------------------------------
* Uniaxial / 1-D chain only: the fractional element is scalar and acts on the
  axial stress.  A 3-D tensorial generalisation (fractional Glen flow) is *not*
  implemented.
* Time-fractional (``t^alpha``) memory only; temperature enters through
  ``eta_alpha`` (Arrhenius), which is a modelling choice, not a validated
  thermo-rheology.
* L1 keeps the full history: ``O(n^2)`` work, ``O(n)`` memory.  Fast-convolution
  acceleration is out of scope here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc, gamma as _gamma

__all__ = [
    "ALPHA_ANDRADE",
    "AndradeCreepLaw",
    "FractionalDemChain",
    "l1_weights",
    "l1_fractional_integral",
    "mittag_leffler_e",
    "mittag_leffler_e_series",
    "mittag_leffler_e_asymptotic",
    "fractional_relaxation_reference",
    "andrade_strain",
    "fit_andrade",
    "relaxation_time_constant",
    "exponential_fit_quality",
    "tau_scale_correlation",
]

#: Andrade (Bailey) primary-creep time exponent for ice: ``eps ~ t^(1/3)``.
ALPHA_ANDRADE = 1.0 / 3.0

#: Seam between the Mittag-Leffler series and asymptotic branches.
_ML_SEAM = 3.0


# --------------------------------------------------------------------------- #
# Fractional-calculus building blocks
# --------------------------------------------------------------------------- #
def l1_weights(n_steps: int, alpha: float) -> np.ndarray:
    """L1 weights ``b_k = (k+1)^(1-alpha) - k^(1-alpha)``, ``k = 0..n_steps``.

    ``b_0 = 1`` and ``b_k`` decreases monotonically towards 0, so the scheme is
    a convolution with positive summable weights (unconditionally stable for
    ``0 < alpha <= 1``).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if abs(alpha - 1.0) < 1e-15:
        # alpha = 1 degenerates to the backward difference: b = [1, 0, 0, ...]
        weights = np.zeros(n_steps + 1)
        weights[0] = 1.0
        return weights
    k = np.arange(n_steps + 1, dtype=float)
    return (k + 1.0) ** (1.0 - alpha) - k ** (1.0 - alpha)


def l1_fractional_integral(g, dt: float, alpha: float) -> np.ndarray:
    """Solve ``D^alpha y = g(t)`` with ``y(0) = 0`` using the L1 scheme.

    Reference implementation used by the tests; the production path is the
    incremental one inside :class:`FractionalDemChain`, which also carries the
    bond mechanics.  ``g`` is sampled at ``t = 0, dt, 2 dt, ...``.
    """
    g = np.asarray(g, dtype=float)
    n = g.size - 1
    if n < 1:
        raise ValueError("need at least two samples")
    b = l1_weights(n, alpha)
    scale = dt ** alpha * float(_gamma(2.0 - alpha))
    y = np.zeros(n + 1)
    dy = np.zeros(n + 1)
    for step in range(1, n + 1):
        history = float(np.dot(b[1:step], dy[step - 1:0:-1])) if step > 1 else 0.0
        y[step] = y[step - 1] + scale * g[step] - history
        dy[step] = y[step] - y[step - 1]
    return y


# --------------------------------------------------------------------------- #
# Mittag-Leffler function E_alpha(-x)
# --------------------------------------------------------------------------- #
def mittag_leffler_e_series(x: float, alpha: float, *, tol: float = 1e-17,
                            max_terms: int = 4000) -> float:
    """``E_alpha(-x) = sum_k (-x)^k / Gamma(1 + alpha k)`` via ``math.fsum``.

    Reliable in double precision for ``x <= ~3`` (the alternating series loses
    at most ~9 digits there); beyond that the asymptotic branch is used.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    terms = []
    for k in range(max_terms):
        term = ((-x) ** k) / float(_gamma(1.0 + alpha * k))
        terms.append(term)
        if k > 4 and abs(term) < tol * max(abs(sum(terms[-5:])), 1e-300):
            break
    return math.fsum(terms)


def mittag_leffler_e_asymptotic(x: float, alpha: float) -> float:
    """``E_alpha(-x)`` for large ``x`` from the optimally truncated expansion.

    ``E_alpha(-x) ~ - sum_{k>=1} (-x)^(-k) / Gamma(1 - alpha k)``, truncated at
    ``k = ceil(x) + 2``; relative error of order ``exp(-x)``.  Terms whose
    ``Gamma`` argument is a non-positive integer vanish.
    """
    if x <= 0.0:
        raise ValueError("x must be positive")
    k_max = int(math.ceil(x)) + 2
    total = 0.0
    for k in range(1, k_max + 1):
        arg = 1.0 - alpha * k
        if arg <= 0.0 and abs(arg - round(arg)) < 1e-12:
            continue                       # Gamma pole -> coefficient is zero
        total += ((-x) ** (-k)) / float(_gamma(arg))
    return -total


def mittag_leffler_e(x, alpha: float):
    """``E_alpha(-x)`` for ``x >= 0`` (vectorised, double precision).

    ``x <= 3`` uses the power series, ``x > 3`` the asymptotic expansion; the
    branches agree to ~1 % at the seam and to <1e-9 elsewhere (see tests).
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    if np.any(xs < 0.0):
        raise ValueError("mittag_leffler_e is implemented for x >= 0")
    if abs(alpha - 1.0) < 1e-15:
        return np.exp(-xs) if np.ndim(x) else float(np.exp(-xs[0]))
    out = np.empty_like(xs)
    for i, value in enumerate(xs):
        if value <= _ML_SEAM:
            out[i] = mittag_leffler_e_series(float(value), alpha)
        else:
            out[i] = mittag_leffler_e_asymptotic(float(value), alpha)
    return out if np.ndim(x) else float(out[0])


def fractional_relaxation_reference(t, tau: float, alpha: float) -> np.ndarray:
    """Normalised relaxation ``sigma(t)/sigma_0 = E_alpha(-(t/tau)^alpha)``.

    ``alpha = 1`` returns ``exp(-t/tau)``; ``alpha = 1/2`` returns
    ``exp(x^2) erfc(x)`` with ``x = sqrt(t/tau)``.  Both are exact and are used
    as controls for the numerical machinery.
    """
    t = np.asarray(t, dtype=float)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    x = (t / tau) ** alpha
    if abs(alpha - 1.0) < 1e-12:
        return np.exp(-x)
    if abs(alpha - 0.5) < 1e-12:
        # E_{1/2}(-z) = exp(z^2) erfc(z) with z = (t/tau)^(1/2)
        return np.exp(x * x) * erfc(x)
    return np.array([mittag_leffler_e(float(v), alpha)
                     for v in np.atleast_1d(x)])


# --------------------------------------------------------------------------- #
# Constitutive law
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AndradeCreepLaw:
    """Springpot law ``eps_c = sigma^m t^alpha / (eta_alpha Gamma(1+alpha))``.

    ``stress_exponent`` (``m``) lets the primary-creep coefficient scale with
    stress independently of the Glen exponent ``n = 3`` of the steady stage.
    ``m = 1`` is the linear-viscoelastic limit exercised by the DEM chain.
    """

    alpha: float = ALPHA_ANDRADE
    eta_alpha: float = 4.677e10
    stress_exponent: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must lie in (0, 1]")
        if not (math.isfinite(self.eta_alpha) and self.eta_alpha > 0.0):
            raise ValueError("eta_alpha must be finite and positive")
        if not (math.isfinite(self.stress_exponent) and self.stress_exponent > 0.0):
            raise ValueError("stress_exponent must be finite and positive")

    @property
    def compliance_coefficient(self) -> float:
        """``1/(eta_alpha Gamma(1+alpha))``, the multiplier of ``sigma^m t^alpha``."""
        return 1.0 / (self.eta_alpha * float(_gamma(1.0 + self.alpha)))

    def creep_strain(self, sigma, t, young_modulus: float | None = None):
        """``eps(t) = sigma/E + beta sigma^m t^alpha`` (``E`` optional)."""
        t = np.asarray(t, dtype=float)
        sigma = float(sigma)
        creep = (self.compliance_coefficient * abs(sigma) ** self.stress_exponent
                 * t ** self.alpha)
        if young_modulus is None:
            return creep
        return sigma / young_modulus + creep

    def beta(self, sigma: float) -> float:
        """Andrade coefficient ``eps_1 = sigma^m/(eta_alpha Gamma(1+alpha))``."""
        return self.compliance_coefficient * abs(sigma) ** self.stress_exponent

    def retardation_time(self, young_modulus: float) -> float:
        """``tau = (eta_alpha/E)^(1/alpha)`` of the fractional Maxwell element."""
        return (self.eta_alpha / young_modulus) ** (1.0 / self.alpha)


def andrade_strain(t, eps_0: float, eps_1: float, alpha: float = ALPHA_ANDRADE):
    """Analytic target of the mission criterion: ``eps_0 + eps_1 t^alpha``."""
    t = np.asarray(t, dtype=float)
    return eps_0 + eps_1 * t ** alpha


# --------------------------------------------------------------------------- #
# DEM chain: springpot in series with the bond spring
# --------------------------------------------------------------------------- #
class FractionalDemChain:
    """1-D assembly of ``n_bonds`` identical fractional-Maxwell DEM bonds.

    Each bond has rest length ``rest``, section ``pi radius^2``, axial stiffness
    ``k = E A_b / rest`` and a springpot ``eta_alpha`` in series.  A chain is
    used (rather than a single bond) so that the measured quantities are genuine
    DEM outputs: nodal displacements accumulated from bond elongations and the
    base reaction from the transmitted bond forces.  Because a series
    connection transmits a uniform force, the creep elongations add, so the
    macroscopic strain of the chain reproduces the element law exactly -- which
    is what makes the analytic comparison a test of the integrator and the
    assembly, not of a formula.
    """

    def __init__(self, *, n_bonds: int = 8, radius: float = 0.01,
                 rest: float = 0.02, young_modulus: float = 9.0e9,
                 law: AndradeCreepLaw | None = None):
        if n_bonds < 1:
            raise ValueError("n_bonds must be >= 1")
        self.n_bonds = int(n_bonds)
        self.radius = float(radius)
        self.rest = float(rest)
        self.young_modulus = float(young_modulus)
        self.law = law or AndradeCreepLaw()
        self.area = math.pi * self.radius ** 2
        self.stiffness = self.young_modulus * self.area / self.rest
        self.retardation_time = self.law.retardation_time(self.young_modulus)

    @property
    def length(self) -> float:
        return self.n_bonds * self.rest

    def elastic_strain(self, sigma0: float) -> float:
        return sigma0 / self.young_modulus

    # -- driver 1: constant stress ---------------------------------------- #
    def creep(self, sigma0: float, t_end: float, n_steps: int) -> dict:
        """Constant-stress creep test; strain read from nodal displacements."""
        if sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive")
        dt = t_end / float(n_steps)
        alpha = self.law.alpha
        b = l1_weights(n_steps, alpha)
        scale = dt ** alpha * float(_gamma(2.0 - alpha))
        scale_first = dt ** alpha / float(_gamma(1.0 + alpha))
        g = abs(sigma0) ** self.law.stress_exponent / self.law.eta_alpha
        eps_el = self.elastic_strain(sigma0)

        eps_c = np.zeros(n_steps + 1)
        delta = np.zeros(n_steps + 1)
        u_top = np.zeros(n_steps + 1)
        u_top[0] = eps_el * self.length          # load applied at t = 0
        for step in range(1, n_steps + 1):
            basis = scale_first if step == 1 else scale
            history = float(np.dot(b[1:step], delta[step - 1:0:-1])) if step > 1 else 0.0
            eps_c[step] = eps_c[step - 1] + basis * g - history
            delta[step] = eps_c[step] - eps_c[step - 1]
            u_top[step] = (eps_el + eps_c[step]) * self.length

        t = np.arange(n_steps + 1, dtype=float) * dt
        strain = (eps_el + eps_c)
        analytic_creep = self.law.beta(sigma0) * t ** alpha
        analytic = andrade_strain(t, eps_el, self.law.beta(sigma0), alpha)
        return {
            "t": t,
            "dt": dt,
            "strain": strain,                    # from nodal displacements
            "top_displacement": u_top,
            "creep_strain": eps_c,
            "elastic_strain": np.full_like(t, eps_el),
            "force": np.full_like(t, sigma0 * self.area),
            "analytic_strain": analytic,
            "analytic_creep": analytic_creep,
            "eps_0_analytic": eps_el,
            "eps_1_analytic": self.law.beta(sigma0),
            "alpha": alpha,
            "sigma0": sigma0,
            "chain": self,
        }

    # -- driver 2: fixed piston displacement (relaxation) ----------------- #
    def relax(self, total_strain: float, t_end: float, n_steps: int) -> dict:
        """Stress relaxation at constant total strain (piston held)."""
        if total_strain <= 0.0:
            raise ValueError("total_strain must be positive")
        if abs(self.law.stress_exponent - 1.0) > 1e-12:
            raise ValueError(
                "relax() solves the linear (stress_exponent = 1) element; "
                "a stress-dependent springpot needs an outer Newton iteration")
        dt = t_end / float(n_steps)
        alpha = self.law.alpha
        b = l1_weights(n_steps, alpha)
        scale = dt ** alpha * float(_gamma(2.0 - alpha))
        scale_first = dt ** alpha / float(_gamma(1.0 + alpha))
        sigma0 = self.young_modulus * total_strain
        tau = self.retardation_time

        sigma = np.zeros(n_steps + 1)
        eps_c = np.zeros(n_steps + 1)
        delta = np.zeros(n_steps + 1)
        sigma[0] = sigma0
        denom = 1.0 / self.young_modulus + scale / self.law.eta_alpha
        denom_first = 1.0 / self.young_modulus + scale_first / self.law.eta_alpha
        for step in range(1, n_steps + 1):
            history = float(np.dot(b[1:step], delta[step - 1:0:-1])) if step > 1 else 0.0
            if step == 1:
                sigma[step] = (total_strain - eps_c[0] + history) / denom_first
                eps_c[step] = (eps_c[0] + scale_first * sigma[step]
                               / self.law.eta_alpha - history)
            else:
                sigma[step] = (total_strain - eps_c[step - 1] + history) / denom
                eps_c[step] = (eps_c[step - 1] + scale * sigma[step]
                               / self.law.eta_alpha - history)
            delta[step] = eps_c[step] - eps_c[step - 1]

        t = np.arange(n_steps + 1, dtype=float) * dt
        normalised = sigma / sigma0
        return {
            "t": t,
            "dt": dt,
            "force": sigma * self.area,
            "force_normalised": normalised,
            "sigma": sigma,
            "reference_normalised": fractional_relaxation_reference(t, tau, alpha),
            "sigma0": sigma0,
            "tau": tau,
            "alpha": alpha,
            "chain": self,
        }


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def fit_andrade(t, strain, *, alpha_bounds=(0.05, 1.0)) -> dict:
    """Fit ``eps(t) = eps_0 + eps_1 t^alpha`` to a creep record.

    ``eps_0``, ``eps_1`` and ``alpha`` are all free (bounded non-linear least
    squares); the starting point is the analytic ``alpha = 1/3``.  Returns the
    fitted triple plus the fitted-curve misfit, so a caller can decide on the
    5 % criterion without trusting the fit blindly.
    """
    t = np.asarray(t, dtype=float)
    strain = np.asarray(strain, dtype=float)
    if t.shape != strain.shape or t.size < 4:
        raise ValueError("t and strain must be equal-length arrays (>=4)")
    positive = t > 0.0
    t_fit = t[positive]
    e_fit = strain[positive]

    def residual(p):
        eps_0, eps_1, alpha = p
        return eps_0 + eps_1 * t_fit ** alpha - e_fit

    p0 = np.array([strain[0], max(e_fit[-1] - strain[0], 1e-12) / t_fit[-1] ** ALPHA_ANDRADE,
                   ALPHA_ANDRADE])
    lower = np.array([-np.inf, 0.0, alpha_bounds[0]])
    upper = np.array([np.inf, np.inf, alpha_bounds[1]])
    sol = least_squares(residual, p0, bounds=(lower, upper), xtol=1e-14,
                        ftol=1e-14, gtol=1e-14)
    eps_0, eps_1, alpha = (float(v) for v in sol.x)
    model = eps_0 + eps_1 * t ** alpha
    scale = np.maximum(np.abs(strain), 1e-30)
    return {
        "eps_0": eps_0,
        "eps_1": eps_1,
        "alpha": alpha,
        "max_rel_error": float(np.max(np.abs(model - strain) / scale)),
        "rms_rel_error": float(np.sqrt(np.mean(((model - strain) / scale) ** 2))),
        "cost": float(sol.cost),
        "converged": bool(sol.success),
    }


def relaxation_time_constant(t, normalised, level: float = 1.0 / math.e) -> float:
    """Time at which a monotonically decaying history first crosses ``level``."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(normalised, dtype=float)
    if t.shape != y.shape or t.size < 2:
        raise ValueError("t and normalised must be equal-length arrays (>=2)")
    if y[0] <= level:
        return 0.0
    below = np.nonzero(y <= level)[0]
    if below.size == 0:
        raise ValueError("history never decays below the requested level")
    i = int(below[0])
    if i == 0:
        return float(t[0])
    y0, y1 = y[i - 1], y[i]
    if y0 == y1:
        return float(t[i])
    frac = (y0 - level) / (y0 - y1)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def exponential_fit_quality(t, normalised, level: float = 1.0 / math.e) -> dict:
    """Quality of the *single-exponential* approximation of a decay history.

    Returns the effective time constant (crossing of ``level``, default 1/e), the maximum and
    RMS relative deviation of ``exp(-t/tau_eff)`` from the history, and the
    ``R^2`` of a log-linear fit.  This is the honest accounting behind the
    "``F = F_0 exp(-t/tau)``" criterion: it is exact only for alpha = 1.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(normalised, dtype=float)
    tau = relaxation_time_constant(t, y, level=level)
    model = np.exp(-t / tau)
    scale = np.maximum(np.abs(y), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.log(np.where(y > 0.0, y, np.nan))
    valid = np.isfinite(log_y)
    if valid.sum() > 2:
        design = np.vstack([t[valid], np.ones(valid.sum())]).T
        coef, *_ = np.linalg.lstsq(design, log_y[valid], rcond=None)
        pred = design @ coef
        ss_res = float(np.sum((log_y[valid] - pred) ** 2))
        ss_tot = float(np.sum((log_y[valid] - log_y[valid].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
        log_slope = float(coef[0])
    else:
        r2 = float("nan")
        log_slope = float("nan")
    window = t <= 3.0 * tau
    late = int(np.argmin(np.abs(t - 10.0 * tau)))
    return {
        "tau_eff": tau,
        "level": level,
        "window_end_s": 3.0 * tau,
        "max_rel_deviation_window": float(np.max(np.abs(model[window] - y[window])
                                                 / scale[window])),
        "rms_rel_deviation_window": float(
            np.sqrt(np.mean(((model[window] - y[window]) / scale[window]) ** 2))),
        "tail_value_at_10_tau": float(y[late]),
        "tail_exponential_prediction": float(math.exp(-10.0)),
        "tail_ratio": float(y[late] / math.exp(-10.0)) if y[late] > 0 else float("inf"),
        "log_linear_r2": r2,
        "log_linear_slope": log_slope,
    }


def tau_scale_correlation(creep_parameter, tau) -> dict:
    """Log-log regression of the relaxation time on the creep parameter.

    ``creep_parameter`` is the fluidity ``A`` (Norton) or ``eta_alpha``
    (fractional).  Returns the exponent, the Pearson ``r`` of the log-log
    straight line and ``R^2``; ``|r| >= 0.9`` with a slope of ``-1`` is the
    quantitative form of the mission criterion "``tau`` correlates with the
    creep parameter".
    """
    x = np.asarray(creep_parameter, dtype=float)
    y = np.asarray(tau, dtype=float)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("need >= 3 paired positive samples")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("both arrays must be positive")
    lx = np.log(x)
    ly = np.log(y)
    design = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    r = float(np.corrcoef(lx, ly)[0, 1])
    return {
        "exponent": float(coef[0]),
        "prefactor": float(math.exp(coef[1])),
        "pearson_r": r,
        "abs_r": abs(r),
        "r2": r2,
        "points": int(x.size),
        "x_range": (float(x.min()), float(x.max())),
    }