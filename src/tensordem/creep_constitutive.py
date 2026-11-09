"""Glen flow law / Norton creep constitutive update for ice DEM contacts.

Why this module exists
----------------------
:class:`tensordem.ice_bond_viscoelastic` already carries a *linear*
Maxwell-Wiechert (Prony) bond law.  That law has a single relaxation spectrum and
therefore cannot represent the *non-linear* rate sensitivity of ice, whose
secondary (steady) creep rate obeys the Norton / Glen power law

    eps_dot_ij = (3/2) A * sigma_eq**(n-1) * s_ij ,      sigma_eq = sqrt(3/2 s:s)

with ``n ~ 3`` (Glen's exponent) and ``A`` the temperature dependent flow
coefficient.  This module supplies that law, the matching *implicit* stress
update, and the closed-form solutions the DEM results are pinned against.  It is
the constitutive kernel only: nothing here touches solver state, and the module
imports no contact model.

Conventions (three of them matter, all pinned by tests)
-------------------------------------------------------
1. **Normalisation.**  The task specification writes Glen's law as
   ``eps_dot_ij = A sigma_eq**(n-1) s_ij``.  Written that way the uniaxial
   reduction is ``eps_dot = (2/3) A sigma**n``, i.e. it does *not* equal the
   Norton uniaxial law ``eps_dot = A sigma**n``.  The two forms differ only by
   the 3/2 factor; the one that (a) reduces to ``eps_dot = A sigma**n`` under
   uniaxial stress and (b) is work conjugate, ``sigma:eps_dot_c = sigma_eq *
   eps_dot_eq``, carries the factor:

       ``normalization="work_conjugate"`` (default)
           eps_dot_ij = (3/2) A sigma_eq**(n-1) s_ij
       ``normalization="glen_literal"``
           eps_dot_ij = A sigma_eq**(n-1) s_ij      -> uniaxial = (2/3) A sigma**n

   Both are implemented because the choice is a real modelling decision, and
   ``test_glen_literal_normalization_is_two_thirds`` pins the -1/3 error the
   literal form makes against ``eps_dot = A sigma**n``.  (The usual field
   convention ``eps_dot_ij = A tau_e**(n-1) s_ij`` with ``tau_e = sqrt(1/2 s:s)``
   is identical to ``work_conjugate``: ``tau_e = sigma_eq / sqrt(3)`` and
   ``(3/2)sigma_eq**(n-1) = sqrt(3)/2 * 3**((n+1)/2) ... `` reduces to the same
   tensor; see ``docs/IN69_N01_creep_constitutive.md`` for the algebra.)

2. **Equivalent measures.**  ``sigma_eq = sqrt(3/2 s:s)`` (von Mises, so that
   uniaxial stress gives ``sigma_eq = sigma``) and ``eps_eq = sqrt(2/3 e:e)``
   (so that incompressible uniaxial strain gives ``eps_eq = eps_axial``).

3. **Units.**  stress MPa, time s, length mm.  Force = MPa*mm^2 = N and
   energy = MPa*mm^3 = mJ, so the whole module is consistent without any
   conversion factor.  ``A`` is in ``MPa**-n s**-1``; at -10 degC ice is
   ``A ~ 3.5e-7 MPa**-3 s**-1`` (Glen, n = 3).

Creep is taken to be **incompressible** (``tr(eps_dot_c) = 0``): the volumetric
response stays elastic, which is the standard assumption for ice at these
stress levels and is what makes the relaxation anchors below exact.

Implicit stress update (radial return)
--------------------------------------
Given the purely elastic *trial* stress of an increment, backward Euler on the
flow rule gives a single scalar equation for the equivalent stress

    sigma_eq + 3 G dt A sigma_eq**n = sigma_eq_trial ,      dgamma = dt A sigma_eq**n

whose left side is monotone increasing, so the root is unique and a safeguarded
Newton iteration (bisection fallback) always converges.  The deviator is then
scaled radially, ``s = s_trial * sigma_eq / sigma_eq_trial``, which is exactly
the J2 "radial return" of classical viscoplasticity.  ``n = 1`` degenerates to
the linear-viscoelastic closed form ``sigma_eq = sigma_eq_trial / (1 + 3 G dt A)``
and the relaxation becomes the exponential ``sigma0 exp(-E A t)`` -- that
degeneracy is the module's negative control for the power-law machinery.

Three-stage creep
-----------------
A pure Norton law has **no** primary and **no** tertiary stage: it predicts
``eps_dot = A sigma**n`` from ``t = 0`` to rupture.  The three-stage curve needs
two extra ingredients, both standard (Bailey 1926; Kachanov 1958; Rabotnov 1969):

* primary (transient): time hardening ``eps_dot_p = A_p sigma**n t**(-p)`` with
  ``p = 2/3`` gives ``eps_p = 3 A_p sigma**n t**(1/3)``;
* secondary: the Norton rate ``A_s sigma**n``;
* tertiary: scalar damage ``D`` with ``D_dot = B sigma**m (1-D)**(-m)``
  (so ``t_r = 1/((m+1) B sigma**m)`` and ``D -> 1`` at ``t_r``), strain rate
  ``eps_dot_c = A_s sigma**n (1-D)**(-q)``.

With ``q = m + 1`` the tertiary strain integrates in closed form
``-(A_s sigma**n / ((m+1) B sigma**m)) ln(1 - t/t_r)``, so every stage has an
independent analytic anchor.  :func:`integrate_three_stage_creep` integrates the
same ODEs numerically (sub-stepped, no closed form baked in), which is what makes
the "``eps ~ t**(1/3)``" fit an *emergent* result rather than an identity.

Contact force relaxation
------------------------
Constant displacement (relaxation) on a two-sphere Hertz contact: the overlap is
``delta = delta_e + delta_c`` with ``delta`` held fixed, so creep eats the
elastic overlap.  With mean contact pressure ``p = (4 E* / 3 pi) sqrt(delta_e/R)``
and a creep-active depth ``h = c_h a`` (``a = sqrt(R delta_e)``, the contact
radius) the overlap rate is

    delta_dot_e = -h A p**n = -K delta_e**((n+1)/2),
    K = c_h A (4 E* / (3 pi sqrt(R)))**n sqrt(R)

which separates, giving the exact force ratio

    F(t)/F0 = (delta_e/delta_0)**(3/2),
    delta_e**((1-n)/2) = delta_0**((1-n)/2) + ((n-1)/2) K t          (n != 1)
    delta_e            = delta_0 exp(-K t)                            (n == 1)

:class:`TwoSphereCreepRelaxation` steps that same law implicitly and
:func:`hertz_creep_relaxation` returns the closed form, so the long-time
``F(t)/F0`` comparison is a genuine integrator check on the contact level, with
an energy ledger ``-dU = int F d(delta_c)`` available alongside it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "DEFAULT_DTYPE",
    "GlenCreepParameters",
    "ThreeStageCreepParameters",
    "HertzCreepContact",
    "bulk_modulus_from_young",
    "creep_equivalent_strain_rate",
    "creep_strain_rate",
    "creep_stress_power",
    "deviator",
    "equivalent_strain",
    "hertz_creep_relaxation",
    "hydrostatic",
    "integrity_check",
    "integrate_three_stage_creep",
    "norton_relaxation",
    "norton_strain",
    "norton_strain_rate",
    "radial_return",
    "relaxation_histories",
    "shear_modulus_from_young",
    "three_stage_analytic",
    "uniaxial_stress",
    "von_mises",
]

DEFAULT_DTYPE = torch.float64
_EYE = torch.eye(3, dtype=DEFAULT_DTYPE)


# ---------------------------------------------------------------------------
# tensor helpers
# ---------------------------------------------------------------------------
def _as_stress(stress) -> torch.Tensor:
    t = torch.as_tensor(stress, dtype=DEFAULT_DTYPE)
    if t.shape[-2:] != (3, 3):
        raise ValueError(f"stress must have trailing shape (3, 3), got {tuple(t.shape)}")
    return t


def deviator(stress) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(s, p)`` with ``s`` the deviator and ``p = tr(stress)/3``."""
    t = _as_stress(stress)
    p = torch.diagonal(t, dim1=-2, dim2=-1).sum(-1) / 3.0
    return t - p[..., None, None] * _EYE, p


def hydrostatic(stress) -> torch.Tensor:
    """``tr(stress)/3`` (tension positive)."""
    return deviator(stress)[1]


def von_mises(stress) -> torch.Tensor:
    """``sqrt(3/2 s:s)`` -- equals the axial stress for a uniaxial state."""
    s, _ = deviator(stress)
    return torch.sqrt(1.5 * (s * s).sum(dim=(-2, -1)))


def equivalent_strain(strain) -> torch.Tensor:
    """``sqrt(2/3 e:e)`` -- equals the axial strain for incompressible uniaxial."""
    e = torch.as_tensor(strain, dtype=DEFAULT_DTYPE)
    if e.shape[-2:] != (3, 3):
        raise ValueError(f"strain must have trailing shape (3, 3), got {tuple(e.shape)}")
    return torch.sqrt((2.0 / 3.0) * (e * e).sum(dim=(-2, -1)))


def uniaxial_stress(sigma: float) -> torch.Tensor:
    """``diag(sigma, 0, 0)`` as a float64 tensor."""
    t = torch.zeros(3, 3, dtype=DEFAULT_DTYPE)
    t[0, 0] = float(sigma)
    return t


def shear_modulus_from_young(young_modulus: float, poisson_ratio: float) -> float:
    if not (young_modulus > 0.0):
        raise ValueError("young_modulus must be positive")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    return young_modulus / (2.0 * (1.0 + poisson_ratio))


def bulk_modulus_from_young(young_modulus: float, poisson_ratio: float) -> float:
    return young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))


# ---------------------------------------------------------------------------
# flow law
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GlenCreepParameters:
    """Glen / Norton flow-law parameters (stress MPa, time s, ``A`` in MPa^-n/s).

    ``normalization`` selects the tensor form; see the module docstring.  The
    default ``"work_conjugate"`` is the one that reduces to ``eps_dot = A sigma**n``.
    """

    A: float
    n: float
    normalization: str = "work_conjugate"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.A) and self.A > 0.0):
            raise ValueError("A must be a finite positive number (MPa**-n s**-1)")
        if not (math.isfinite(self.n) and self.n > 0.0):
            raise ValueError("n must be a finite positive number")
        if self.normalization not in ("work_conjugate", "glen_literal"):
            raise ValueError(
                "normalization must be 'work_conjugate' or 'glen_literal', "
                f"got {self.normalization!r}"
            )

    @property
    def flow_factor(self) -> float:
        """``3/2`` for the work-conjugate form, ``1`` for the literal one."""
        return 1.5 if self.normalization == "work_conjugate" else 1.0

    def strain_rate(self, sigma_eq) -> torch.Tensor:
        """Equivalent creep strain rate ``A sigma_eq**n`` (scalar or tensor)."""
        seq = torch.as_tensor(sigma_eq, dtype=DEFAULT_DTYPE)
        return self.A * torch.pow(torch.clamp(seq, min=0.0), self.n)


def integrity_check(stress, *, tol: float = 1e-12) -> dict:
    """Return deviator/volumetric split diagnostics (used by tests and reports)."""
    s, p = deviator(stress)
    seq = float(von_mises(stress))
    return {
        "sigma_eq": seq,
        "hydrostatic": float(p),
        "trace": float(3.0 * p),
        "deviator_norm": float(torch.sqrt((s * s).sum())),
        "tolerance": tol,
    }


def creep_strain_rate(stress, params: GlenCreepParameters) -> torch.Tensor:
    """``eps_dot_ij = flow_factor * A * sigma_eq**(n-1) * s_ij`` (incompressible)."""
    s, _ = deviator(stress)
    seq = von_mises(stress)
    if float(seq) <= 0.0:
        return torch.zeros_like(s)
    return params.flow_factor * params.A * torch.pow(seq, params.n - 1.0) * s


def creep_equivalent_strain_rate(stress, params: GlenCreepParameters) -> torch.Tensor:
    """``eps_dot_eq = sqrt(2/3 eps_dot:eps_dot)`` -- equals ``A sigma_eq**n``."""
    return equivalent_strain(creep_strain_rate(stress, params))


def creep_stress_power(stress, params: GlenCreepParameters) -> torch.Tensor:
    """``stress : eps_dot_c`` -- equals ``sigma_eq * eps_dot_eq`` when work conjugate."""
    rate = creep_strain_rate(stress, params)
    t = _as_stress(stress)
    return (t * rate).sum(dim=(-2, -1))


# ---------------------------------------------------------------------------
# uniaxial analytical anchors
# ---------------------------------------------------------------------------
def norton_strain_rate(sigma: float, params: GlenCreepParameters) -> float:
    """Uniaxial Norton rate ``A sigma**n`` (MPa, s)."""
    if sigma < 0.0:
        raise ValueError("uniaxial stress magnitude must be non-negative")
    return float(params.A * sigma ** params.n)


def norton_strain(t: float, sigma: float, params: GlenCreepParameters) -> float:
    """Steady creep strain ``eps(t) = A sigma**n t`` (creep strain only)."""
    return norton_strain_rate(sigma, params) * float(t)


def norton_relaxation(
    t: float,
    sigma0: float,
    params: GlenCreepParameters,
    *,
    young_modulus: float,
) -> float:
    """Closed-form relaxation ``sigma(t)`` at fixed total strain (uniaxial).

    Solves ``sigma_dot/E = -A sigma**n``:

        sigma(t) = sigma0 [1 + (n-1) E A sigma0**(n-1) t]**(-1/(n-1))   (n != 1)
        sigma(t) = sigma0 exp(-E A t)                                    (n == 1)
    """
    e = float(young_modulus)
    if not e > 0.0:
        raise ValueError("young_modulus must be positive")
    s0 = float(sigma0)
    if s0 < 0.0:
        raise ValueError("sigma0 must be non-negative")
    if s0 == 0.0:
        return 0.0
    n = float(params.n)
    if abs(n - 1.0) < 1e-12:
        return s0 * math.exp(-e * params.A * float(t))
    base = 1.0 + (n - 1.0) * e * params.A * s0 ** (n - 1.0) * float(t)
    if base <= 0.0:  # only reachable for n < 1 with a negative driver
        raise ValueError("relaxation closed form is singular at this time")
    return s0 * base ** (-1.0 / (n - 1.0))


# ---------------------------------------------------------------------------
# implicit radial return
# ---------------------------------------------------------------------------
def _solve_monotone(
    f, lo: float, hi: float, *, df=None, tol: float = 1e-13, max_iter: int = 100
) -> float:
    """Safeguarded Newton for a monotone increasing ``f`` with a bracketed root.

    Newton steps are taken when they stay inside the bracket and halve the
    bracket; otherwise the step is bisected.  Convergence is to ``tol`` relative
    and cannot stall outside the bracket, which is what makes both the
    constitutive and the contact solves unconditionally robust.
    """
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError("root is not bracketed by [lo, hi]")
    x = 0.5 * (lo + hi)
    for _ in range(max_iter):
        fx = f(x)
        if fx == 0.0:
            return x
        if fx < 0.0:
            lo, f_lo = x, fx
        else:
            hi, f_hi = x, fx
        if hi - lo <= tol * max(1.0, abs(x)):
            return x
        nxt = None
        if df is not None:
            d = df(x)
            if d > 0.0:
                candidate = x - fx / d
                if lo < candidate < hi:
                    nxt = candidate
        if nxt is None:
            nxt = 0.5 * (lo + hi)
        else:
            # keep the bracket shrinking: never allow a step that does not at
            # least halve the interval when the Newton step is close to an edge
            if abs(nxt - x) > 0.5 * (hi - lo):
                nxt = 0.5 * (lo + hi)
        x = nxt
    return x


def radial_return(
    trial_stress,
    dt: float,
    params: GlenCreepParameters,
    *,
    shear_modulus: float,
    max_iter: int = 200,
    tol: float = 1e-14,
) -> dict:
    """Implicit (backward Euler) creep update from an elastic *trial* stress.

    Solves ``sigma_eq + 3 G dt A sigma_eq**n = sigma_eq_trial`` and returns the
    corrected stress, the creep strain increment, the multiplier
    ``dgamma = dt A sigma_eq**n`` and the dissipation ``trial-independent``
    quantity ``(sigma_old + sigma_new)/2 : d_eps_c`` is *not* formed here --
    callers own their quadrature; this returns the per-step pieces.

    Returns a dict with keys ``stress``, ``d_creep_strain``, ``dgamma``,
    ``sigma_eq``, ``sigma_eq_trial``, ``iterations`` (always 0 for the
    bracketed solver, kept for API stability).
    """
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be a finite positive time increment")
    if not (math.isfinite(shear_modulus) and shear_modulus > 0.0):
        raise ValueError("shear_modulus must be positive")

    trial = _as_stress(trial_stress)
    s_tr, p = deviator(trial)
    seq_tr_t = von_mises(trial)
    seq_tr = float(seq_tr_t)
    out = {
        "stress": trial.clone(),
        "d_creep_strain": torch.zeros_like(trial),
        "dgamma": 0.0,
        "sigma_eq": seq_tr,
        "sigma_eq_trial": seq_tr,
        "iterations": 0,
    }
    if seq_tr <= 0.0:
        return out

    g = float(shear_modulus)
    c = 3.0 * g * dt * params.A
    n = float(params.n)

    def f(x: float) -> float:
        return x + c * x ** n - seq_tr

    def df(x: float) -> float:
        return 1.0 + c * n * x ** (n - 1.0)

    lo, hi = 0.0, seq_tr  # f(0) = -seq_tr < 0, f(seq_tr) >= 0
    seq = _solve_monotone(f, lo, hi, df=df, tol=tol, max_iter=max_iter)

    dgamma = dt * params.A * seq ** n
    s_new = s_tr * (seq / seq_tr)
    stress_new = s_new + p[..., None, None] * _EYE
    d_creep = params.flow_factor * dgamma * (s_new / seq)

    out.update(
        stress=stress_new,
        d_creep_strain=d_creep,
        dgamma=dgamma,
        sigma_eq=seq,
        sigma_eq_trial=seq_tr,
    )
    return out


# ---------------------------------------------------------------------------
# strain-controlled relaxation drivers
# ---------------------------------------------------------------------------
# Two different relaxation tests share the phrase "constant strain", and they
# relax with *different* moduli.  Mixing them up is a silent ~15% error, so both
# are implemented and separately anchored:
#
# * ``fixed_total_strain`` -- every component of the total strain is held.  Creep
#   is incompressible, so ``eps^e_dev`` absorbs the whole creep increment,
#   ``s_dot = -2G eps_dot_c`` and with ``eps_dot_eq = A sigma_eq**n``
#       sigma_eq_dot = -3 G A sigma_eq**n        ->  modulus 3G,
#   while ``p`` stays constant (``tr eps_dot_c = 0``).
# * ``free_lateral`` -- the textbook uniaxial relaxation test: the axial total
#   strain is held, the lateral surfaces are traction free.  Then
#       sigma_dot = -E A sigma**n               ->  modulus E = 2G(1+nu)
#   and ``p`` drifts because the lateral elastic strain changes.
def _relaxation_analytic(t: float, sigma0: float, params: GlenCreepParameters, modulus: float) -> float:
    n = float(params.n)
    if abs(n - 1.0) < 1e-12:
        return sigma0 * math.exp(-modulus * params.A * t)
    base = 1.0 + (n - 1.0) * modulus * params.A * sigma0 ** (n - 1.0) * t
    return sigma0 * base ** (-1.0 / (n - 1.0))


def _elastic_energy(sigma, shear_modulus: float, bulk_modulus: float) -> float:
    s, p = deviator(sigma)
    return float((s * s).sum()) / (4.0 * shear_modulus) + float(p * p) / (2.0 * bulk_modulus)


def relaxation_histories(
    sigma0: float,
    t_end: float,
    dt: float,
    params: GlenCreepParameters,
    *,
    young_modulus: float,
    poisson_ratio: float = 0.3,
    constraint: str = "fixed_total_strain",
    analytic: bool = True,
) -> dict:
    """Constant-strain uniaxial relaxation integrated by :func:`radial_return`.

    ``constraint="fixed_total_strain"``: every total strain component is held;
    the deviator relaxes at modulus ``3G`` and ``p`` is constant.
    ``constraint="free_lateral"``: only the axial total strain is held and the
    lateral total strain is solved each step so that ``sigma_22 = sigma_33 = 0``;
    the axial stress relaxes at modulus ``E``.  The latter exercises the mixed
    boundary-value path of the tensor update (one scalar secant iteration per
    step) rather than a single radial return.

    Returns ``times``, ``sigma_11``, ``sigma_eq``, ``elastic_energy``,
    ``dissipation`` (trapezoidal ``int sigma_eq eps_dot_eq dt``), the analytic
    ``sigma_analytic`` and the modulus used for it.
    """
    if not (t_end > 0.0 and 0.0 < dt <= t_end):
        raise ValueError("require 0 < dt <= t_end")
    if constraint not in ("fixed_total_strain", "free_lateral"):
        raise ValueError("constraint must be 'fixed_total_strain' or 'free_lateral'")
    g = shear_modulus_from_young(young_modulus, poisson_ratio)
    k_bulk = bulk_modulus_from_young(young_modulus, poisson_ratio)
    s0 = float(sigma0)
    if s0 <= 0.0:
        raise ValueError("sigma0 must be positive for a relaxation history")

    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError("dt too large for t_end")
    nu = float(poisson_ratio)
    eps_e = torch.zeros(3, 3, dtype=DEFAULT_DTYPE)
    eps_e[0, 0] = s0 / young_modulus
    eps_e[1, 1] = eps_e[2, 2] = -nu * s0 / young_modulus

    def stiffness(strain: torch.Tensor) -> torch.Tensor:
        return 2.0 * g * strain + (k_bulk - 2.0 * g / 3.0) * torch.diagonal(
            strain
        ).sum() * _EYE

    times, sig11, seq_hist, energy, diss = [0.0], [s0], [s0], [
        _elastic_energy(stiffness(eps_e), g, k_bulk)
    ], [0.0]
    diss_total = 0.0
    lateral_guess = 0.0
    lateral_slope = 2.0 * g / 3.0 + 2.0 * k_bulk
    sigma_old = stiffness(eps_e)
    seq_old = float(von_mises(sigma_old))
    for step in range(n_steps):
        if constraint == "fixed_total_strain":
            trial = stiffness(eps_e)
            res = radial_return(trial, dt, params, shear_modulus=g)
            eps_e = eps_e - res["d_creep_strain"]
            sigma_new = res["stress"]
        else:  # free_lateral: solve d_eps_lateral so that sigma_22 = sigma_33 = 0
            def sigma_at(dlat: float) -> tuple[torch.Tensor, dict]:
                dstrain = torch.zeros(3, 3, dtype=DEFAULT_DTYPE)
                dstrain[1, 1] = dstrain[2, 2] = dlat
                tr = radial_return(stiffness(eps_e + dstrain), dt, params, shear_modulus=g)
                return tr["stress"], tr

            lam = lateral_guess  # warm start from the previous step
            sigma_new, res = sigma_at(lam)
            for _ in range(4):
                f = float(sigma_new[1, 1])
                if abs(f) < 1e-13 * max(s0, 1.0):
                    break
                # d sigma_22 / d eps_lateral ~= 2G/3 + 2K for the elastic part;
                # incompressible creep reduces it slightly, so the slope is only
                # used as a Newton direction and the residual is re-evaluated.
                lam -= f / lateral_slope
                sigma_new, res = sigma_at(lam)
            lateral_guess = lam
            eps_e = eps_e + _dstrain_swapped(lam) - res["d_creep_strain"]

        seq_new = float(von_mises(sigma_new))
        diss_total += 0.5 * (seq_old + seq_new) * float(res["dgamma"])
        times.append((step + 1) * dt)
        sig11.append(float(sigma_new[0, 0]))
        seq_hist.append(seq_new)
        energy.append(_elastic_energy(sigma_new, g, k_bulk))
        diss.append(diss_total)
        seq_old = seq_new

    modulus = 3.0 * g if constraint == "fixed_total_strain" else float(young_modulus)
    out = {
        "times": times,
        "sigma_11": sig11,
        "sigma_eq": seq_hist,
        "elastic_energy": energy,
        "dissipation": diss,
        "constraint": constraint,
        "relaxation_modulus": modulus,
    }
    if analytic:
        out["sigma_analytic"] = [
            _relaxation_analytic(t, s0, params, modulus) for t in times
        ]
    return out


def _dstrain_swapped(lateral: float) -> torch.Tensor:
    t = torch.zeros(3, 3, dtype=DEFAULT_DTYPE)
    t[1, 1] = t[2, 2] = lateral
    return t


# ---------------------------------------------------------------------------
# three-stage creep (primary + secondary + tertiary)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThreeStageCreepParameters:
    """Composite creep model: Bailey primary, Norton secondary, Kachanov tertiary.

    ``primary`` and ``secondary`` share the stress exponent ``n`` (standard for
    ice); ``damage_m`` and ``damage_B`` control the tertiary stage through
    ``D_dot = B sigma**damage_m (1-D)**(-damage_m)``, i.e. rupture at
    ``t_r = 1/((damage_m+1) B sigma**damage_m)``.  The tertiary strain-rate
    exponent defaults to ``damage_m + 1``, the value that keeps the strain
    closed form elementary (log divergence at rupture).
    """

    glen: GlenCreepParameters
    primary_A: float
    primary_p: float = 2.0 / 3.0
    damage_B: float = 1e-9
    damage_m: float = 2.0
    tertiary_q: float = 3.0

    def __post_init__(self) -> None:
        if not (self.primary_A > 0.0):
            raise ValueError("primary_A must be positive")
        if not (0.0 <= self.primary_p < 1.0):
            raise ValueError("primary_p must lie in [0, 1)")
        if not (self.damage_B > 0.0 and self.damage_m > 0.0 and self.tertiary_q > 0.0):
            raise ValueError("damage_B, damage_m and tertiary_q must be positive")

    def rupture_time(self, sigma: float) -> float:
        """Kachanov rupture time ``1/((m+1) B sigma**m)``."""
        return 1.0 / ((self.damage_m + 1.0) * self.damage_B * float(sigma) ** self.damage_m)
    # NOTE: the docstring above intentionally avoids non-ASCII to keep the file
    # diff reviewable in terminals that mangle it.

    def damage(self, t: float, sigma: float) -> float:
        """``D(t) = 1 - (1 - t/t_r)**(1/(m+1))``."""
        tr = self.rupture_time(sigma)
        if t < 0.0 or t > tr:
            raise ValueError("t outside [0, t_r]")
        return 1.0 - (1.0 - t / tr) ** (1.0 / (self.damage_m + 1.0))

    def primary_strain(self, t: float, sigma: float) -> float:
        """Bailey primary strain ``3 A_p sigma**n t**(1/3)`` for ``p = 2/3``."""
        if t <= 0.0:
            return 0.0
        n, ap, pp = self.glen.n, self.primary_A, self.primary_p
        return ap * float(sigma) ** n * t ** (1.0 - pp) / (1.0 - pp)

    def secondary_strain(self, t: float, sigma: float) -> float:
        return norton_strain(t, sigma, self.glen)

    def tertiary_strain(self, t: float, sigma: float) -> float:
        """Tertiary strain ``-(A sigma**n / ((m+1) B sigma**m)) ln(1 - t/t_r)``.

        Valid for ``tertiary_q = damage_m + 1`` (enforced by the closed form);
        for other ``q`` the caller must integrate numerically.
        """
        if abs(self.tertiary_q - (self.damage_m + 1.0)) > 1e-12:
            raise ValueError("closed form requires tertiary_q == damage_m + 1")
        if t <= 0.0:
            return 0.0
        tr = self.rupture_time(sigma)
        if t >= tr:
            raise ValueError("tertiary strain diverges at the rupture time")
        rate = norton_strain_rate(sigma, self.glen)
        scale = rate / ((self.damage_m + 1.0) * self.damage_B * float(sigma) ** self.damage_m)
        return -scale * math.log(1.0 - t / tr)

    def strain_rate(self, t: float, sigma: float) -> float:
        """Total creep strain rate at time ``t`` (primary + secondary + tertiary)."""
        n, ap, pp = self.glen.n, self.primary_A, self.primary_p
        primary = ap * float(sigma) ** n * (t ** (-pp) if t > 0.0 else 0.0)
        sec = norton_strain_rate(sigma, self.glen)
        d = self.damage(min(t, self.rupture_time(sigma) * (1.0 - 1e-15)), sigma)
        tertiary = sec * (1.0 - d) ** (-self.tertiary_q)
        return primary + sec + tertiary


def three_stage_analytic(
    t: float,
    sigma: float,
    params: ThreeStageCreepParameters,
    *,
    young_modulus: float | None = None,
) -> dict:
    """Stage-resolved closed-form creep strain (adds the elastic strain if given)."""
    p = params
    out = {
        "elastic": (float(sigma) / young_modulus) if young_modulus else 0.0,
        "primary": p.primary_strain(t, sigma),
        "secondary": p.secondary_strain(t, sigma),
        "tertiary": p.tertiary_strain(t, sigma),
    }
    out["total"] = out["elastic"] + out["primary"] + out["secondary"] + out["tertiary"]
    return out


def integrate_three_stage_creep(
    sigma: float,
    t_end: float,
    dt: float,
    params: ThreeStageCreepParameters,
    *,
    young_modulus: float | None = None,
    t_start: float = 1e-3,
    growth: float = 0.02,
) -> dict:
    """Sub-stepped numerical integration of the three-stage creep ODEs.

    Explicit midpoint on the additive rates (Bailey primary time hardening +
    Norton secondary + Kachanov damage driven tertiary).  Nothing here knows
    about ``t**(1/3)`` or ``ln(1 - t/t_r)``: the 1/3 exponent and the
    acceleration are *emergent*, which is what the validation script measures.

    The only analytic input is the **initial instant** ``t_start``: the primary
    rate ``A_p sigma**n t**(-2/3)`` is integrable but singular at ``t = 0``, and a
    one-sided rule there is off by a factor 2 (``int_0^h t**(-2/3) dt =
    3 h**(1/3)`` versus the midpoint's ``2**(2/3) h**(1/3)``).  That single
    interval is therefore started from ``eps_p(t_start) = 3 A_p sigma**n
    t_start**(1/3)`` and everything after it is stepped numerically; the
    sensitivity to ``t_start`` is reported in the docs.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if not (t_end > 0.0 and 0.0 < dt <= t_end):
        raise ValueError("require 0 < dt <= t_end")
    if not (0.0 < t_start < t_end):
        raise ValueError("require 0 < t_start < t_end")
    if not (0.0 < growth <= 1.0):
        raise ValueError("growth must lie in (0, 1]")
    tr = params.rupture_time(sigma)
    g = params.glen
    n, ap, pp = g.n, params.primary_A, params.primary_p
    sec_rate = norton_strain_rate(sigma, g)
    elastic = float(sigma) / young_modulus if young_modulus else 0.0

    def prim_rate(t: float) -> float:
        return ap * sigma ** n * (t ** (-pp) if t > 0.0 else 0.0)

    prim_total = ap * sigma ** n * t_start ** (1.0 - pp) / (1.0 - pp)
    sec_total = sec_rate * t_start
    tert_total = 0.0

    times = [t_start]
    prim_hist = [prim_total]
    sec_hist = [sec_total]
    tert_hist = [0.0]
    rates = [prim_rate(t_start) + sec_rate]
    inv_tr = 1.0 / tr
    inv_mp1 = 1.0 / (params.damage_m + 1.0)
    q = params.tertiary_q
    damage = [1.0 - (1.0 - min(t_start * inv_tr, 1.0)) ** inv_mp1]
    t = t_start
    # the tertiary stage is meaningless past the rupture time, so the history ends there
    stop = min(t_end, tr * (1.0 - 1e-12))
    ruptured = bool(t_end >= tr)
    ap_sn = ap * sigma ** n
    while t < stop - 1e-15:
        # geometric refinement of the singular early stage: a uniform step that
        # starts at t_start loses O(t_start**(1/3)) of the primary integral, so
        # the step is capped at ``growth * t`` until it reaches ``dt``.
        h = min(dt, growth * t, stop - t)
        tm = t + 0.5 * h
        u = max(1.0 - min(tm * inv_tr, 1.0), 1e-12)
        tertiary_rate = sec_rate * u ** (-q * inv_mp1)
        prim_total += ap_sn * tm ** (-pp) * h
        sec_total += sec_rate * h
        tert_total += tertiary_rate * h
        t += h
        u = max(1.0 - min(t * inv_tr, 1.0), 1e-12)
        d_new = 1.0 - u ** inv_mp1
        times.append(t)
        prim_hist.append(prim_total)
        sec_hist.append(sec_total)
        tert_hist.append(tert_total)
        rates.append(ap_sn * t ** (-pp) + sec_rate * u ** (-q * inv_mp1))
        damage.append(d_new)
        if d_new > 1.0 - 1e-9:
            ruptured = True
            break
    creep = [a + b + c for a, b, c in zip(prim_hist, sec_hist, tert_hist)]
    return {
        "times": times,
        "strain": [elastic + c for c in creep],
        "creep_strain": creep,
        "primary": prim_hist,
        "secondary": sec_hist,
        "tertiary": tert_hist,
        "rate": rates,
        "damage": damage,
        "rupture_time": tr,
        "ruptured": ruptured,
    }


# ---------------------------------------------------------------------------
# two-particle Hertz contact with creep: long-time force relaxation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HertzCreepContact:
    """Two-sphere Hertz contact with a creep-relaxing overlap.

    ``reduced_modulus`` ``E*`` (MPa) and ``effective_radius`` ``R`` (mm) define
    the elastic geometry: ``F = 4/3 E* sqrt(R) delta_e**(3/2)``, contact radius
    ``a = sqrt(R delta_e)``, mean pressure ``p = F/(pi a^2)``.  ``creep_depth_factor``
    ``c_h`` sets the creep-active depth ``h = c_h a``: creep relaxes the elastic
    overlap at ``delta_dot_c = h * A p**n``.
    """

    reduced_modulus: float
    effective_radius: float
    params: GlenCreepParameters
    creep_depth_factor: float = 1.0

    def __post_init__(self) -> None:
        if not (self.reduced_modulus > 0.0 and self.effective_radius > 0.0):
            raise ValueError("reduced_modulus and effective_radius must be positive")
        if not (self.creep_depth_factor > 0.0):
            raise ValueError("creep_depth_factor must be positive")

    @classmethod
    def from_elastic(
        cls,
        young_modulus: float,
        poisson_ratio: float,
        radius: float,
        params: GlenCreepParameters,
        *,
        creep_depth_factor: float = 1.0,
    ) -> "HertzCreepContact":
        """Equal spheres of radius ``radius``: ``E* = E/(2(1-nu^2))``, ``R = r/2``."""
        e_star = young_modulus / (2.0 * (1.0 - poisson_ratio ** 2))
        return cls(e_star, radius / 2.0, params, creep_depth_factor)

    # -- elastic geometry ---------------------------------------------------
    def force(self, delta_e: float) -> float:
        if delta_e <= 0.0:
            return 0.0
        return 4.0 / 3.0 * self.reduced_modulus * math.sqrt(self.effective_radius) * delta_e ** 1.5

    def contact_radius(self, delta_e: float) -> float:
        return math.sqrt(self.effective_radius * max(delta_e, 0.0))

    def mean_pressure(self, delta_e: float) -> float:
        a = self.contact_radius(delta_e)
        if a <= 0.0:
            return 0.0
        return self.force(delta_e) / (math.pi * a * a)

    def sensitivity(self, delta_e: float) -> float:
        """``-d(delta_e)/dt / delta_e**((n+1)/2)`` = ``K`` of the separable ODE."""
        n = self.params.n
        c_sigma = 4.0 * self.reduced_modulus / (3.0 * math.pi * math.sqrt(self.effective_radius))
        return (
            self.creep_depth_factor
            * self.params.A
            * c_sigma ** n
            * math.sqrt(self.effective_radius)
        )

    def elastic_energy(self, delta_e: float) -> float:
        """Hertz elastic energy ``2/5 F delta_e`` (mJ with MPa/mm)."""
        return 0.4 * self.force(delta_e) * max(delta_e, 0.0)

    def analytic_force_ratio(self, t: float, delta0: float) -> float:
        """Closed-form ``F(t)/F0`` at fixed total overlap."""
        return hertz_creep_relaxation(self, t, delta0)["force_ratio"]

    # -- implicit stepping --------------------------------------------------
    def step(self, delta_e: float, delta_c: float, dt: float, *, tol: float = 1e-15) -> float:
        """Implicitly advance the creep overlap by ``dt``; returns ``delta_e_new``.

        Backward Euler on ``delta_c_dot = K delta_e**m`` (``m = (n+1)/2``) at
        fixed total overlap ``delta0 = delta_e + delta_c``:

            delta_e_new = delta0 - delta_c - dt K delta_e_new**m

        ``g(x) = delta0 - delta_c - dt K x**m - x`` is decreasing with
        ``g(0) > 0 > g(delta0)``, so ``-g`` is increasing and bracketed on
        ``[0, delta0]``: the root is unique and cannot be missed.
        """
        if not (dt > 0.0):
            raise ValueError("dt must be positive")
        m = 0.5 * (self.params.n + 1.0)
        k_sens = self.sensitivity(delta_e)
        delta0 = delta_e + delta_c

        def g(x: float) -> float:
            # increasing residual: (delta0 - delta_c - x) - dt K x**m ... in the
            # form solved below, f = -g is increasing with f(0) < 0 < f(delta0)
            return delta0 - delta_c - dt * k_sens * x ** m - x

        def dg(x: float) -> float:
            return 1.0 + dt * k_sens * m * x ** (m - 1.0)

        return _solve_monotone(lambda x: -g(x), 0.0, delta0, df=lambda x: dg(x),
                               tol=tol, max_iter=100)

    def run(self, delta0: float, t_end: float, dt: float) -> dict:
        """Constant-displacement relaxation history with an energy ledger."""
        if not (delta0 > 0.0):
            raise ValueError("delta0 must be positive")
        if not (t_end > 0.0 and 0.0 < dt <= t_end):
            raise ValueError("require 0 < dt <= t_end")
        n_steps = int(round(t_end / dt))
        delta_e = delta0
        delta_c = 0.0
        f0 = self.force(delta0)
        times = [0.0]
        ratios = [1.0]
        forces = [f0]
        u = [self.elastic_energy(delta0)]
        work = [0.0]
        accumulated = 0.0
        for i in range(n_steps):
            new_delta_e = self.step(delta_e, delta_c, dt)
            dc = delta_c + (delta_e - new_delta_e)
            # creep work done by the contact force: int F d(delta_c)
            accumulated += 0.5 * (self.force(delta_e) + self.force(new_delta_e)) * (
                delta_e - new_delta_e
            )
            delta_e, delta_c = new_delta_e, dc
            times.append((i + 1) * dt)
            forces.append(self.force(delta_e))
            ratios.append(self.force(delta_e) / f0)
            u.append(self.elastic_energy(delta_e))
            work.append(accumulated)
        return {
            "times": times,
            "force": forces,
            "force_ratio": ratios,
            "elastic_energy": u,
            "creep_work": work,
            "delta_e": delta_e,
            "delta_c": delta_c,
            "force0": f0,
            "analytic_force_ratio": [self.analytic_force_ratio(t, delta0) for t in times],
        }


def hertz_creep_relaxation(
    contact: HertzCreepContact, t: float | Sequence[float], delta0: float
) -> dict:
    """Closed-form constant-displacement Hertz creep relaxation.

    ``delta_e**((1-n)/2) = delta0**((1-n)/2) + ((n-1)/2) K t`` with
    ``K = c_h A (4E*/(3 pi sqrt(R)))**n sqrt(R)``; ``F/F0 = (delta_e/delta0)**(3/2)``.
    ``n = 1`` gives ``delta_e = delta0 exp(-K t)``.
    """
    if not (delta0 > 0.0):
        raise ValueError("delta0 must be positive")
    n = contact.params.n
    k_sens = contact.sensitivity(delta0)

    def one(tt: float) -> dict:
        if tt < 0.0:
            raise ValueError("t must be non-negative")
        if abs(n - 1.0) < 1e-12:
            delta_e = delta0 * math.exp(-k_sens * tt)
        else:
            exponent = 0.5 * (1.0 - n)
            base = delta0 ** exponent + 0.5 * (n - 1.0) * k_sens * tt
            if base <= 0.0:
                raise ValueError("relaxation time exceeded (delta_e -> 0)")
            delta_e = base ** (1.0 / exponent)
        return {
            "delta_e": delta_e,
            "delta_c": delta0 - delta_e,
            "force": contact.force(delta_e),
            "force_ratio": contact.force(delta_e) / contact.force(delta0),
            "mean_pressure": contact.mean_pressure(delta_e),
        }

    if isinstance(t, (int, float)):
        return one(float(t))
    out = [one(float(tt)) for tt in t]
    return {
        "times": list(t),
        "delta_e": [o["delta_e"] for o in out],
        "force": [o["force"] for o in out],
        "force_ratio": [o["force_ratio"] for o in out],
        "mean_pressure": [o["mean_pressure"] for o in out],
    }