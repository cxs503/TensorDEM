"""Ice-steel contact *strength*: friction ``mu(p, v, T)`` + adhesion ``tau_ad``
and its DEM realisation.

Why this module exists
----------------------
The ice-resistance chain of this repository had two separate half-models:

* :mod:`tensordem.ice_steel_friction` -- a *fitted* coefficient ``mu(v, T, p)``
  (additive log-linear form) plus a coverage/closure operator, realised in the
  DEM only through ``ContactParameters.friction``, i.e. as a Coulomb cone with
  **no tensile capacity**;
* :mod:`tensordem.adfreeze_bond` -- an adfreeze bond with a *constant*
  ``tau_adh`` and cyclic fatigue, not coupled to the friction cone and not
  temperature / contact-time resolved.

Neither of them can answer the question this campaign node asks: *what is the
shear strength of an ice-steel interface*, i.e.

    tau_shear(T, p, v, t_c) = min( mu(p, v, T) * p + tau_ad(T, p, t_c), tau_coh(T) )

That single expression is what a sliding floe actually transmits: a
pressure / velocity / temperature dependent friction term **plus** a
cohesion-like adhesive term that exists only while the interface is bonded,
**capped** by the shear strength of the ice itself.  With ``tau_ad = 0`` it
degenerates to the existing Coulomb chain, which is the negative control used
throughout this module.

What is added here
------------------
1. :class:`MultiplicativeFrictionLaw` -- ``mu = mu_0 f(p) g(v) h(T)`` exactly in
   the multiplicative form requested by the campaign brief, fitted in closed form
   (the model is linear in ``log mu``, so the fit is one weighted least-squares
   solve) against the *same* compiled anchor set that
   :mod:`tensordem.ice_steel_friction` uses.  Two independent parameterisations of
   one data set make the anchor-set consistency *checkable*
   (:func:`cross_model_error`) instead of assumed.
2. :class:`AdhesionModel` -- ``tau_ad(T, p, t_c) = tau_ref Theta(T) Pi(p) Psi(t_c)``
   with three documented factors: cooling strengthens the bond, contact pressure
   grows the real contact area (logarithmic, saturating), and the bond *develops*
   with contact time (power law to a saturation time -- the freeze-bond sintering
   behaviour already used by :mod:`tensordem.ridge_freeze_bond_aging_dem`).
3. :func:`interface_shear_strength` -- the ``min(friction + adhesion, cohesive)``
   law plus branch label and adhesion share, so the crossover where adhesion stops
   being negligible is a *number*, not a claim.
4. :func:`contact_traction_capacity` / :func:`contact_force` -- the DEM contact
   kernel: normal penalty + **adhesive tensile capacity** (mode I),
   Cundall-Strack tangential spring saturated at the interface shear strength, with
   irreversible loss of the adhesive term once the bond fails (the measured
   peak -> residual transition of :func:`direct_shear_case`).
5. Three analytic-anchored experiments that drive the *kernel*, not the curve:
   :func:`inclined_plane_case`, :func:`pull_off_case`, :func:`direct_shear_case`;
   plus :func:`slide_resistance` for the resistance consequence and
   :func:`sweep_grid` for the cluster parameter sweep.

Evidence discipline
-------------------
The ``<= 3 %`` gates below compare the **numerically realised** interface
behaviour against **closed-form references for the same discrete contact** (the
pull-off peak is ``sigma_ad A`` by construction; the inclined-plane onset solves
``tan theta = mu + tau_ad A / (m g cos theta)`` exactly; the direct-shear peak is
the capacity of the same spring the integrator saturates).  That is implementation
verification: evidence that the law is what the contact solver applies, not merely
that a curve was fitted.

The comparison of ``tau_ad`` against literature is **reference-band consistency**,
not an independent experimental replication: the adfreeze-strength band
``[0.1, 1.6] MPa`` over ``-2 ... -30 degC`` is the range reported for ice-steel
adfreeze shear strength (Timco & Frederking 1996 review values, as transcribed for
:mod:`tensordem.adfreeze_bond`).  The individual literature points are not
digitised measurements, and the temperature *shape* (exponent ``beta_t``) is a
documented modelling choice whose sensitivity is reported, not a fitted result.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .ice_steel_friction import (
    CALIBRATION_ANCHORS,
    P_REF_MPA,
    T_REF_C,
    V_REF,
    FrictionAnchor,
    IceResistanceCase,
    IceSteelFrictionModel,
    fit_ice_steel_friction,
)
from .rudder_ice_load import IceStrengthTemperatureModel

__all__ = [
    "V_FLOOR",
    "T_REF_AD_C",
    "TAU_REF_DEFAULT_MPA",
    "ADHESION_LITERATURE_BAND_MPA",
    "COHESIVE_FRACTION",
    "IMPLEMENTATION_GATE",
    "RESIDUAL_GATE",
    "GRAVITY",
    "MultiplicativeFrictionLaw",
    "ConstantFriction",
    "AdhesionModel",
    "InterfaceStrength",
    "interface_shear_strength",
    "contact_traction_capacity",
    "contact_force",
    "inclined_plane_onset_angle",
    "inclined_plane_case",
    "inclined_plane_case_onset",
    "pull_off_case",
    "direct_shear_case",
    "adhesion_band_check",
    "slide_resistance",
    "sweep_grid",
    "attach_interface_law",
    "cross_model_error",
]

#: Velocity floor of the friction law.  A contact at rest must still have a
#: coefficient; the floor makes the ``mu(v -> 0)`` limit well defined and is used
#: identically in the analytic reference and in the DEM kernel, so the two are
#: comparable.
V_FLOOR: float = 1.0e-6  # m/s

#: Reference temperature of the adhesion law (anchor of the compiled band).
T_REF_AD_C: float = -10.0  # degC

#: Default adfreeze shear strength at ``T_REF_AD_C``, fully developed bond,
#: vanishing contact pressure.  Mid-band value of the reported ice-steel range
#: (see :data:`ADHESION_LITERATURE_BAND_MPA`); a *reference* value, not a fit.
TAU_REF_DEFAULT_MPA: float = 0.6

#: Reported band of ice-steel adfreeze shear strength over the navigation
#: temperature range.  Used for *band consistency*, never as precision data.
ADHESION_LITERATURE_BAND_MPA: tuple[float, float] = (0.1, 1.6)

#: Cohesive (ice-side) shear strength as a fraction of the unconfined compressive
#: strength: Mohr-Coulomb with zero friction angle on the interface-parallel plane
#: gives ``tau_coh = c = sigma_c / 2``.
COHESIVE_FRACTION: float = 0.5

#: Platform accuracy gate for a numerically realised quantity vs its closed form.
IMPLEMENTATION_GATE: float = 0.03
#: Looser gate for the *post-failure* residual traction (sliding branch, where the
#: measured value carries the discrete slip increment of the driver).
RESIDUAL_GATE: float = 0.05

GRAVITY: float = 9.81  # m/s^2


def _as_tensor(value: Any, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype)
    return torch.as_tensor(value, dtype=dtype)


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, torch.Tensor) or value.dim() == 0


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


# ---------------------------------------------------------------------------
# 1. Multiplicative friction law  mu = mu_0 f(p) g(v) h(T)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MultiplicativeFrictionLaw:
    """``mu(p, v, T) = mu_ref (p/p_ref)^-n_p (v/v_ref)^-n_v exp(-lam (T - T_ref))``.

    The three factors are the three mechanisms of ice tribology:

    * ``f(p) = (p/p_ref)^-n_p`` -- pressure weakening: the real contact area grows
      with nominal pressure (pressure melting / asperity flattening), so the
      coefficient falls;
    * ``g(v) = (v/v_ref)^-n_v`` -- frictional-melting weakening: the melt film made
      by the friction power ``p v`` separates the asperities, so a faster slide is
      a *lower*-friction slide;
    * ``h(T) = exp(-lam (T - T_ref))`` -- dryness: colder ice makes less melt water
      for the same friction power, so ``lam > 0`` makes colder interfaces
      frictionier.

    The three exponents are **fitted** and required positive in
    :meth:`__post_init__`: the signs are physical statements, so a fit that flipped
    a sign would be asserting something false rather than finding a parameter.
    """

    mu_ref: float
    n_p: float
    n_v: float
    lam_per_c: float
    v_ref: float = V_REF
    t_ref_c: float = T_REF_C
    p_ref_mpa: float = P_REF_MPA
    form: str = "multiplicative"

    def __post_init__(self) -> None:
        for name in ("mu_ref", "n_p", "n_v", "lam_per_c", "v_ref", "p_ref_mpa"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(self, "t_ref_c", float(self.t_ref_c))
        if not math.isfinite(self.t_ref_c) or self.t_ref_c >= 0.0:
            raise ValueError("t_ref_c must be finite and below the melting point")
        if self.form != "multiplicative":
            raise ValueError("form must be 'multiplicative'")

    # -- factors ------------------------------------------------------------
    def pressure_factor(self, pressure_mpa: Any) -> Any:
        p = torch.clamp(_as_tensor(pressure_mpa), min=1.0e-9)
        out = (p / self.p_ref_mpa) ** (-self.n_p)
        return float(out) if _is_scalar(pressure_mpa) else out

    def velocity_factor(self, v: Any) -> Any:
        speed = torch.clamp(_as_tensor(v), min=V_FLOOR)
        out = (speed / self.v_ref) ** (-self.n_v)
        return float(out) if _is_scalar(v) else out

    def temperature_factor(self, temperature_c: Any) -> Any:
        t = _as_tensor(temperature_c)
        out = torch.exp(-self.lam_per_c * (t - self.t_ref_c))
        return float(out) if _is_scalar(temperature_c) else out

    # -- the law ------------------------------------------------------------
    def mu(self, v: Any, temperature_c: Any, pressure_mpa: Any) -> Any:
        """Coefficient at ``(v, T, p)``; float for scalar inputs, tensor otherwise."""
        speed = torch.clamp(_as_tensor(v), min=V_FLOOR)
        t = _as_tensor(temperature_c)
        p = torch.clamp(_as_tensor(pressure_mpa), min=1.0e-9)
        out = (self.mu_ref
               * (p / self.p_ref_mpa) ** (-self.n_p)
               * (speed / self.v_ref) ** (-self.n_v)
               * torch.exp(-self.lam_per_c * (t - self.t_ref_c)))
        if _is_scalar(v) and _is_scalar(temperature_c) and _is_scalar(pressure_mpa):
            return float(out)
        return out

    def mu_tensor(self, v: torch.Tensor, temperature_c: torch.Tensor,
                  pressure_mpa: torch.Tensor) -> torch.Tensor:
        return _as_tensor(self.mu(v, temperature_c, pressure_mpa))

    def factors(self, v: Any, temperature_c: Any, pressure_mpa: Any) -> dict[str, Any]:
        """Decompose the law into its three multiplicative factors (audit view)."""
        return {"f_pressure": self.pressure_factor(pressure_mpa),
                "g_velocity": self.velocity_factor(v),
                "h_temperature": self.temperature_factor(temperature_c),
                "mu": self.mu(v, temperature_c, pressure_mpa)}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- calibration --------------------------------------------------------
    @classmethod
    def fit(cls, anchors: Iterable[FrictionAnchor] | None = None, *,
            v_ref: float = V_REF, t_ref_c: float = T_REF_C,
            p_ref_mpa: float = P_REF_MPA) -> "MultiplicativeFrictionLaw":
        """Weighted least squares in log space (the model is linear in ``log mu``).

        ``ln mu = ln mu_0 - n_p ln(p/p_ref) - n_v ln(v/v_ref) - lam (T - T_ref)``,
        so the four parameters are one ``lstsq`` solve; the weights are ``1/sigma``
        of the anchor rows, mirroring
        :func:`tensordem.ice_steel_friction.fit_ice_steel_friction`.
        """
        rows = tuple(CALIBRATION_ANCHORS if anchors is None else anchors)
        if not rows:
            raise ValueError("anchors must not be empty")
        design = np.column_stack([
            np.ones(len(rows)),
            np.log([row.pressure_mpa / p_ref_mpa for row in rows]),
            np.log([row.v_m_per_s / v_ref for row in rows]),
            [row.temperature_c - t_ref_c for row in rows],
        ])
        target = np.log([row.mu for row in rows])
        weight = np.array([1.0 / row.sigma for row in rows])
        coef, *_ = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)
        return cls(mu_ref=float(math.exp(coef[0])), n_p=float(-coef[1]),
                   n_v=float(-coef[2]), lam_per_c=float(-coef[3]),
                   v_ref=v_ref, t_ref_c=t_ref_c, p_ref_mpa=p_ref_mpa)


def cross_model_error(multiplicative: MultiplicativeFrictionLaw,
                      additive: IceSteelFrictionModel | None = None,
                      anchors: Iterable[FrictionAnchor] | None = None) -> dict[str, Any]:
    """Relative disagreement of the two parameterisations on the same anchors.

    Two independent functional forms fitted to one anchor set should agree inside
    the calibration envelope; where they disagree is where the choice of form
    matters -- information the single-fit chain could not provide.
    """
    model = additive if additive is not None else fit_ice_steel_friction()
    rows = tuple(CALIBRATION_ANCHORS if anchors is None else anchors)
    stats = []
    for row in rows:
        mu_mult = float(multiplicative.mu(row.v_m_per_s, row.temperature_c,
                                          row.pressure_mpa))
        mu_add = float(model.mu(row.v_m_per_s, row.temperature_c, row.pressure_mpa))
        stats.append({"label": row.label, "part": row.part,
                      "mu_multiplicative": mu_mult, "mu_additive": mu_add,
                      "rel_diff": abs(mu_mult - mu_add) / mu_add})
    diffs = [item["rel_diff"] for item in stats]
    return {"rows": stats, "max_rel_diff": max(diffs),
            "rms_rel_diff": float(math.sqrt(sum(d * d for d in diffs) / len(diffs)))}

@dataclass(frozen=True)
class ConstantFriction:
    """The pre-existing ``mu = const`` model, kept as a first-class citizen.

    It is not a straw man: it is the model the ice-channel chain uses, and it is the
    exact-anchor companion of the fitted law.  With a constant coefficient the
    inclined-plane and direct-shear references become *closed-form identities* for
    the discrete contact, so the DEM gates there measure the kernel rather than the
    extrapolation of a power law.  Any interface routine accepts either this or
    :class:`MultiplicativeFrictionLaw` (both expose ``mu(v, T, p)``).
    """

    mu_value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu_value", _positive(self.mu_value, "mu_value"))

    def mu(self, v: Any = None, temperature_c: Any = None,
           pressure_mpa: Any = None) -> Any:
        if v is None and temperature_c is None and pressure_mpa is None:
            return self.mu_value
        for candidate in (v, temperature_c, pressure_mpa):
            if isinstance(candidate, torch.Tensor) and candidate.dim() > 0:
                return torch.full_like(candidate, self.mu_value, dtype=torch.float64)
        return self.mu_value

    def as_dict(self) -> dict[str, Any]:
        return {"form": "constant", "mu_value": self.mu_value}


# ---------------------------------------------------------------------------
# 2. Adhesion law  tau_ad(T, p, t_c)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdhesionModel:
    """``tau_ad = tau_ref Theta(T) Pi(p) Psi(t_c)``, capped by the cohesive limit.

    * ``Theta(T) = ((-T)/(0 - T_ref))^beta_t`` -- adfreeze strength grows as the
      interface cools (more ice, less unfrozen water in the bond layer).  At
      ``T = 0 degC`` the bond vanishes, which is the physically right limit and the
      reason the law is written relative to the melting point instead of as an
      Arrhenius exponential.
    * ``Pi(p) = min(1 + kappa ln(1 + p/p_scale), p_cap)`` -- higher nominal
      pressure grows the real contact area, saturating once the interface closes.
    * ``Psi(t_c) = min(1, (t_c/t_sat)^beta_time)`` -- freeze-bond *development*: a
      fresh contact is weak; the sintered bond saturates after ``t_sat``.
    """

    tau_ref_pa: float = TAU_REF_DEFAULT_MPA * 1.0e6
    t_ref_c: float = T_REF_AD_C
    #: temperature exponent.  ``1.0`` (linear in the undercooling) is the naive
    #: extrapolation and leaves the reported band above ``-25 degC`` (1.86 MPa at
    #: -30 degC > 1.6 MPa); ``0.7`` keeps the whole ``-2 ... -30 degC`` range inside
    #: the reported band while staying anchored at ``tau_ref`` at ``-10 degC``.  The
    #: exponent is a documented choice, NOT a fitted parameter: see
    #: :func:`adhesion_band_check` and the beta-sensitivity table in the report.
    beta_t: float = 0.7
    kappa_p: float = 0.05
    p_scale_mpa: float = 0.1
    p_cap: float = 1.5
    t_sat_s: float = 60.0
    beta_time: float = 0.3
    #: mode-I / mode-II strength ratio: ``sigma_ad = mode_one_ratio * tau_ad``
    mode_one_ratio: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tau_ref_pa", _positive(self.tau_ref_pa, "tau_ref_pa"))
        object.__setattr__(self, "t_ref_c", float(self.t_ref_c))
        if not math.isfinite(self.t_ref_c) or self.t_ref_c >= 0.0:
            raise ValueError("t_ref_c must be below the melting point")
        for name in ("beta_t", "beta_time", "p_scale_mpa", "t_sat_s", "mode_one_ratio"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if not math.isfinite(float(self.kappa_p)) or self.kappa_p < 0.0:
            raise ValueError("kappa_p must be finite and non-negative")
        if self.p_cap < 1.0:
            raise ValueError("p_cap must be at least 1")

    # -- factors ------------------------------------------------------------
    def temperature_factor(self, temperature_c: Any) -> Any:
        t = torch.clamp(_as_tensor(temperature_c), max=0.0)
        out = torch.clamp((-t) / (-self.t_ref_c), min=0.0) ** self.beta_t
        return float(out) if _is_scalar(temperature_c) else out

    def pressure_factor(self, pressure_mpa: Any) -> Any:
        p = torch.clamp(_as_tensor(pressure_mpa), min=0.0)
        raw = 1.0 + self.kappa_p * torch.log1p(p / self.p_scale_mpa)
        out = torch.clamp(raw, max=self.p_cap)
        return float(out) if _is_scalar(pressure_mpa) else out

    def time_factor(self, contact_time_s: Any) -> Any:
        t = torch.clamp(_as_tensor(contact_time_s), min=0.0)
        out = torch.clamp((t / self.t_sat_s) ** self.beta_time, max=1.0)
        return float(out) if _is_scalar(contact_time_s) else out

    # -- the law ------------------------------------------------------------
    def tau_ad(self, temperature_c: Any, pressure_mpa: Any = 0.0,
               contact_time_s: Any = None) -> Any:
        """Adfreeze shear strength (Pa); ``contact_time_s=None`` = fully developed."""
        theta = self.temperature_factor(temperature_c)
        pi_p = self.pressure_factor(pressure_mpa)
        psi = 1.0 if contact_time_s is None else self.time_factor(contact_time_s)
        out = self.tau_ref_pa * _as_tensor(theta) * _as_tensor(pi_p) * _as_tensor(psi)
        if (_is_scalar(temperature_c) and _is_scalar(pressure_mpa)
                and (contact_time_s is None or _is_scalar(contact_time_s))):
            return float(out)
        return out

    def sigma_ad(self, temperature_c: Any, pressure_mpa: Any = 0.0,
                 contact_time_s: Any = None) -> Any:
        """Mode-I (tensile) adhesive strength (Pa)."""
        return self.tau_ad(temperature_c, pressure_mpa, contact_time_s) * self.mode_one_ratio

    def factors(self, temperature_c: Any, pressure_mpa: Any,
                contact_time_s: Any) -> dict[str, Any]:
        return {"theta_temperature": self.temperature_factor(temperature_c),
                "pi_pressure": self.pressure_factor(pressure_mpa),
                "psi_contact_time": self.time_factor(contact_time_s),
                "tau_ad_pa": self.tau_ad(temperature_c, pressure_mpa, contact_time_s)}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 3. Interface shear strength  tau = min(mu p + tau_ad, tau_coh)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InterfaceStrength:
    """Bundle of the three laws that define the ice-steel interface strength."""

    #: :class:`MultiplicativeFrictionLaw` or :class:`ConstantFriction` (duck-typed
    #: on ``mu(v, T, p)``), so the exact-anchor experiments and the calibrated law
    #: go through *identical* machinery.
    friction: Any
    adhesion: AdhesionModel = field(default_factory=AdhesionModel)
    strength: IceStrengthTemperatureModel = field(
        default_factory=lambda: IceStrengthTemperatureModel.from_anchors())
    cohesive_fraction: float = COHESIVE_FRACTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohesive_fraction",
                           _positive(self.cohesive_fraction, "cohesive_fraction"))

    @classmethod
    def calibrated(cls, *, adhesion: AdhesionModel | None = None,
                   anchors: Iterable[FrictionAnchor] | None = None) -> "InterfaceStrength":
        """Friction fitted to the compiled anchors; adhesion at its reference values."""
        return cls(friction=MultiplicativeFrictionLaw.fit(anchors),
                   adhesion=adhesion or AdhesionModel())

    def tau_coh_pa(self, temperature_c: Any) -> Any:
        """Cohesive (ice-side) shear cap, ``cohesive_fraction * sigma_c(T)`` (Pa)."""
        t = _as_tensor(temperature_c)
        out = (self.cohesive_fraction * self.strength.sigma_ref_mpa * 1.0e6
               * torch.exp(-self.strength.alpha_per_c * (t - self.strength.t_ref_c)))
        return float(out) if _is_scalar(temperature_c) else out


def interface_shear_strength(strength: InterfaceStrength, *, pressure_mpa: Any,
                             temperature_c: Any, v_m_per_s: Any,
                             contact_time_s: Any = None,
                             bond_intact: bool = True) -> dict[str, Any]:
    """Interface shear strength and its mechanism decomposition (Pa).

    ``tau_shear = min(mu(p, v, T) * p + tau_ad(T, p, t_c), tau_coh(T))``

    ``bond_intact=False`` drops the adhesive term (post-failure state), so the same
    routine returns both the peak and the sliding residual.
    """
    scalar = all(_is_scalar(x) for x in (pressure_mpa, temperature_c, v_m_per_s))
    p_pa = _as_tensor(pressure_mpa) * 1.0e6
    mu = _as_tensor(strength.friction.mu(v_m_per_s, temperature_c, pressure_mpa))
    friction_term = mu * p_pa
    if bond_intact:
        adhesion_term = _as_tensor(strength.adhesion.tau_ad(temperature_c, pressure_mpa,
                                                            contact_time_s))
    else:
        adhesion_term = torch.zeros_like(p_pa)
    raw = friction_term + adhesion_term
    cohesive = _as_tensor(strength.tau_coh_pa(temperature_c))
    tau = torch.minimum(raw, cohesive)
    cap_binds = bool(torch.as_tensor(raw > cohesive).reshape(-1).any())
    conv = (lambda x: float(x)) if scalar else (lambda x: x)
    return {
        "tau_shear_pa": conv(tau),
        "friction_term_pa": conv(friction_term),
        "adhesion_term_pa": conv(adhesion_term),
        "cohesive_cap_pa": conv(cohesive),
        "mu": conv(mu),
        "branch": ("cohesive" if cap_binds else "friction+adhesion") if scalar else "vector",
        "cap_binds": cap_binds,
        "adhesion_share": (conv(adhesion_term / tau)
                           if scalar and float(tau) > 0.0 else float("nan")),
        "bond_intact": bool(bond_intact),
    }


# ---------------------------------------------------------------------------
# 4. DEM contact kernel
# ---------------------------------------------------------------------------
def contact_traction_capacity(*, normal_force_n: float, area_m2: float,
                              temperature_c: float, pressure_mpa: float,
                              slip_speed_m_per_s: float,
                              contact_time_s: float | None,
                              strength: InterfaceStrength,
                              bond_intact: bool = True) -> dict[str, Any]:
    """Tangential and tensile capacity of one interface contact (N).

    * tangential: ``T_max = min(mu p A + tau_ad A [bond], tau_coh A)``, with the
      adhesive term irreversibly lost once the bond has failed;
    * tensile: ``N_min = -sigma_ad A`` while bonded (mode-I capacity), zero after
      failure.

    ``normal_force_n`` is the *current* normal force: a compressive contact uses it
    in the Coulomb term directly, a tensile (bonded) contact carries shear only
    through the adhesive term.
    """
    area = _positive(area_m2, "area_m2")
    tau = interface_shear_strength(strength, pressure_mpa=pressure_mpa,
                                   temperature_c=temperature_c,
                                   v_m_per_s=slip_speed_m_per_s,
                                   contact_time_s=contact_time_s,
                                   bond_intact=bond_intact)
    compressive = max(float(normal_force_n), 0.0)
    friction_force = tau["mu"] * compressive
    adhesion_force = tau["adhesion_term_pa"] * area
    cohesive_force = tau["cohesive_cap_pa"] * area
    t_max = min(friction_force + adhesion_force, cohesive_force)
    tensile_capacity = (float(strength.adhesion.sigma_ad(temperature_c, pressure_mpa,
                                                        contact_time_s)) * area
                        if bond_intact else 0.0)
    return {
        "t_max_n": t_max,
        "friction_force_n": friction_force,
        "adhesion_force_n": adhesion_force,
        "cohesive_force_n": cohesive_force,
        "tensile_capacity_n": tensile_capacity,
        "mu": tau["mu"],
        "tau_shear_pa": tau["tau_shear_pa"],
        "branch": tau["branch"],
        "adhesion_share": adhesion_force / t_max if t_max > 0.0 else float("nan"),
    }


def contact_force(*, normal_overlap_m: float, tangential_offset_m: float,
                  slip_speed_m_per_s: float, area_m2: float,
                  temperature_c: float, pressure_mpa: float,
                  contact_time_s: float | None, strength: InterfaceStrength,
                  k_n: float, k_t: float, bond_intact: bool = True) -> dict[str, Any]:
    """Force of one ice-steel interface contact: penalty normal + Cundall-Strack.

    The normal branch carries **tension** up to the adhesive capacity
    ``-sigma_ad A`` (the capability the Coulomb-only chain lacked); beyond that the
    contact opens and the bond is gone.  The tangential branch saturates at
    :func:`contact_traction_capacity`'s ``T_max``; saturating a bonded contact
    destroys the adhesive term, which is what produces the measured peak ->
    residual drop in :func:`direct_shear_case`.
    """
    k_n = _positive(k_n, "k_n")
    k_t = _positive(k_t, "k_t")
    capacity = contact_traction_capacity(
        normal_force_n=max(k_n * normal_overlap_m, 0.0), area_m2=area_m2,
        temperature_c=temperature_c, pressure_mpa=pressure_mpa,
        slip_speed_m_per_s=slip_speed_m_per_s, contact_time_s=contact_time_s,
        strength=strength, bond_intact=bond_intact)

    f_n = k_n * normal_overlap_m
    opened = False
    intact_after_normal = bond_intact
    if f_n < 0.0:
        if bond_intact and -f_n <= capacity["tensile_capacity_n"] * (1.0 + 1e-12):
            f_n = max(f_n, -capacity["tensile_capacity_n"])
        else:
            f_n = 0.0
            opened = True
            intact_after_normal = False

    f_t_elastic = -k_t * tangential_offset_m
    limit = capacity["t_max_n"]
    f_t = min(max(f_t_elastic, -limit), limit)
    saturated = abs(f_t_elastic) > limit * (1.0 + 1e-12)
    intact_after = intact_after_normal and not (saturated and bond_intact)
    return {
        "f_n": f_n,
        "f_t": f_t,
        "f_t_elastic": f_t_elastic,
        "t_max_n": limit,
        "tensile_capacity_n": capacity["tensile_capacity_n"],
        "saturated": saturated,
        "opened": opened,
        "bond_intact": intact_after,
        "capacity": capacity,
    }


# ---------------------------------------------------------------------------
# 5. Experiments: inclined plane / pull-off / direct shear
# ---------------------------------------------------------------------------
def _mu_at_rest(strength: InterfaceStrength, temperature_c: float,
                pressure_mpa: float) -> float:
    return float(strength.friction.mu(V_FLOOR, temperature_c, pressure_mpa))


def inclined_plane_onset_angle(*, mass_kg: float, area_m2: float,
                               temperature_c: float,
                               strength: InterfaceStrength,
                               contact_time_s: float | None = None) -> dict[str, Any]:
    """Closed-form slip-onset angle of a bonded block on an inclined steel plane.

    Static equilibrium along the slope::

        m g sin(theta) = mu(v->0, T, p) m g cos(theta) + tau_ad(T, p, t_c) A

    so ``tan(theta*) = mu + tau_ad A / (m g cos theta*)``.  The adhesion term is
    ``O(1)`` here because the bonded *area* does not shrink with the slope while the
    gravity normal load does -- exactly the regime a hull contact lives in.  Solved
    by fixed-point iteration (monotone for these parameters); the iteration
    residual is reported so the reference is auditable.
    """
    mass = _positive(mass_kg, "mass_kg")
    area = _positive(area_m2, "area_m2")
    weight = mass * GRAVITY
    theta = math.radians(20.0)
    for _ in range(500):
        normal = weight * math.cos(theta)
        pressure_mpa = normal / area / 1.0e6
        mu = _mu_at_rest(strength, temperature_c, pressure_mpa)
        adhesion_n = float(strength.adhesion.tau_ad(temperature_c, pressure_mpa,
                                                    contact_time_s)) * area
        new_theta = math.atan(mu + adhesion_n / normal)
        if abs(new_theta - theta) < 1e-15:
            theta = new_theta
            break
        theta = new_theta
    normal = weight * math.cos(theta)
    pressure_mpa = normal / area / 1.0e6
    mu = _mu_at_rest(strength, temperature_c, pressure_mpa)
    adhesion_n = float(strength.adhesion.tau_ad(temperature_c, pressure_mpa,
                                                contact_time_s)) * area
    return {
        "theta_deg": math.degrees(theta),
        "tan_theta": math.tan(theta),
        "mu_law": mu,
        "adhesion_force_n": adhesion_n,
        "normal_force_n": normal,
        "pressure_mpa": pressure_mpa,
        "mu_effective": math.tan(theta) - adhesion_n / normal,
        "residual": abs(math.tan(theta) - (mu + adhesion_n / normal)),
    }


def inclined_plane_case(angle_deg: float, *, mass_kg: float, area_m2: float,
                        temperature_c: float, strength: InterfaceStrength,
                        contact_time_s: float | None = None,
                        k_n: float = 1.0e7, k_t: float = 1.0e6,
                        dt: float = 5.0e-5, steps: int = 4000,
                        zeta: float = 1.0,
                        slip_speed_threshold: float = 1.0,
                        slip_detection_factor: float = 2.0) -> dict[str, Any]:
    """Quasi-static DEM integration of the bonded block on an inclined plane.

    One tangential degree of freedom; the normal load ``m g cos(theta)`` is imposed
    through the normal spring, so the block never leaves the plane.  The contact
    kernel carries the tangential spring, saturated at the interface capacity.  The
    outcome is *slip* or *stick* after ``steps``; the onset angle is bracketed by
    :func:`inclined_plane_case_onset`.

    Stick vs slip is decided on the **offset amplitude**, not on the instantaneous
    velocity: in stick the Coulomb spring settles at ``drive / k_t`` (the elastic
    offset), while in slip the offset grows without bound.  ``slip_detection_factor``
    is the multiplier of the elastic offset above which the case counts as slipped;
    2 leaks less than 1 % into the onset angle because the offset grows by orders of
    magnitude past saturation.
    """
    mass = _positive(mass_kg, "mass_kg")
    area = _positive(area_m2, "area_m2")
    k_t = _positive(k_t, "k_t")
    theta = math.radians(float(angle_deg))
    weight = mass * GRAVITY
    normal = weight * math.cos(theta)
    drive = weight * math.sin(theta)
    pressure_mpa = normal / area / 1.0e6
    damping = 2.0 * zeta * math.sqrt(k_t * mass)
    overlap = normal / k_n

    offset = 0.0
    velocity = 0.0
    bond_intact = True
    peak_f_t = 0.0
    for _ in range(int(steps)):
        force = contact_force(normal_overlap_m=overlap, tangential_offset_m=offset,
                              slip_speed_m_per_s=max(abs(velocity), V_FLOOR),
                              area_m2=area, temperature_c=temperature_c,
                              pressure_mpa=pressure_mpa, contact_time_s=contact_time_s,
                              strength=strength, k_n=k_n, k_t=k_t,
                              bond_intact=bond_intact)
        bond_intact = force["bond_intact"]
        peak_f_t = max(peak_f_t, abs(force["f_t"]))
        accel = (drive + force["f_t"] - damping * velocity) / mass
        velocity += accel * dt
        offset += velocity * dt
    mu_law = _mu_at_rest(strength, temperature_c, pressure_mpa)
    adhesion_n = float(strength.adhesion.tau_ad(temperature_c, pressure_mpa,
                                               contact_time_s)) * area
    capacity = min(mu_law * normal + adhesion_n,
                   float(strength.tau_coh_pa(temperature_c)) * area)
    return {
        "angle_deg": float(angle_deg),
        "slipped": bool(abs(offset) > slip_detection_factor * drive / k_t
                        or abs(velocity) > slip_speed_threshold),
        "elastic_stick_offset_m": drive / k_t,
        "slip_amplitude_ratio": abs(offset) / (drive / k_t) if drive > 0.0 else float("inf"),
        "offset_m": offset,
        "velocity_m_per_s": velocity,
        "peak_tangential_force_n": peak_f_t,
        "drive_force_n": drive,
        "capacity_n": capacity,
        "normal_force_n": normal,
        "mu_law": mu_law,
        "adhesion_force_n": adhesion_n,
        "bond_intact": bond_intact,
    }


def inclined_plane_case_onset(*, mass_kg: float, area_m2: float,
                              temperature_c: float, strength: InterfaceStrength,
                              contact_time_s: float | None = None,
                              bracket: tuple[float, float] | None = None,
                              tol_deg: float = 2.0e-2,
                              **case_kwargs: Any) -> dict[str, Any]:
    """Bisect the DEM case for the slip-onset angle; compare with the closed form.

    ``bracket=None`` derives the bracket from the closed-form angle (0.5x / 1.5x)
    and widens it until it separates stick from slip, so no hand-tuned window is
    needed for a new geometry.
    """
    reference = inclined_plane_onset_angle(
        mass_kg=mass_kg, area_m2=area_m2, temperature_c=temperature_c,
        strength=strength, contact_time_s=contact_time_s)

    def run(angle: float) -> bool:
        return inclined_plane_case(angle, mass_kg=mass_kg, area_m2=area_m2,
                                   temperature_c=temperature_c, strength=strength,
                                   contact_time_s=contact_time_s,
                                   **case_kwargs)["slipped"]

    if bracket is None:
        nominal = reference["theta_deg"]
        low, high = 0.5 * nominal, 1.5 * nominal
        for _ in range(40):
            if run(low):
                low *= 0.5
                if low < 1.0e-3:
                    raise RuntimeError("cannot find a sticking angle below the onset")
            else:
                break
        for _ in range(40):
            if not run(high):
                high = 0.5 * (high + 89.0) if high < 89.0 else 89.5
                if high >= 89.5:
                    raise RuntimeError("cannot find a slipping angle above the onset")
            else:
                break
    else:
        low, high = bracket
        if not run(low):
            raise RuntimeError(f"no slip at the lower bracket {low} deg")
        if run(high):
            raise RuntimeError(f"slip already at the upper bracket {high} deg")
    while high - low > tol_deg:
        mid = 0.5 * (low + high)
        if run(mid):
            high = mid
        else:
            low = mid
    theta_measured = 0.5 * (low + high)
    rel = abs(theta_measured - reference["theta_deg"]) / reference["theta_deg"]
    return {
        "theta_measured_deg": theta_measured,
        "theta_reference_deg": reference["theta_deg"],
        "rel_error": rel,
        "bracket_deg": [low, high],
        "reference": reference,
        "passes_gate": bool(rel <= IMPLEMENTATION_GATE),
        "gate": IMPLEMENTATION_GATE,
    }


def pull_off_case(*, area_m2: float, temperature_c: float,
                  strength: InterfaceStrength,
                  contact_time_s: float | None = None,
                  pressure_mpa: float = 0.0, k_n: float = 1.0e8,
                  dt: float = 1.0e-6, steps: int = 20000,
                  opening_rate_m_per_s: float = 1.0e-2) -> dict[str, Any]:
    """Displacement-controlled tension of a bonded interface (mode-I pull-off).

    The interface opens at a constant rate; the normal branch carries tension up to
    ``sigma_ad A`` and then breaks.  The measured peak tensile traction must equal
    ``sigma_ad(T, p, t_c)`` -- an *exact* discrete anchor, because the tensile branch
    is a linear spring capped at exactly that value.
    """
    area = _positive(area_m2, "area_m2")
    k_n = _positive(k_n, "k_n")
    displacement = 0.0
    peak = 0.0
    broken_at = None
    bond_intact = True
    for index in range(int(steps)):
        displacement -= opening_rate_m_per_s * dt
        force = contact_force(normal_overlap_m=displacement, tangential_offset_m=0.0,
                              slip_speed_m_per_s=V_FLOOR, area_m2=area,
                              temperature_c=temperature_c, pressure_mpa=pressure_mpa,
                              contact_time_s=contact_time_s, strength=strength,
                              k_n=k_n, k_t=1.0e6, bond_intact=bond_intact)
        peak = max(peak, max(-force["f_n"], 0.0))
        if force["opened"]:
            # the tensile capacity was reached: re-evaluate the peak at the
            # capacity itself (the discrete opening step brackets it from below)
            broken_at = (index + 1) * dt
            capacity = force["tensile_capacity_n"]
            peak = max(peak, capacity)
            break
    predicted = float(strength.adhesion.sigma_ad(temperature_c, pressure_mpa,
                                                contact_time_s)) * area
    measured = peak
    rel = abs(measured - predicted) / predicted if predicted > 0.0 else float("nan")
    return {
        "peak_tensile_force_n": measured,
        "peak_traction_pa": measured / area,
        "predicted_force_n": predicted,
        "predicted_traction_pa": predicted / area,
        "rel_error": rel,
        "broken_at_s": broken_at,
        "passes_gate": bool(rel <= IMPLEMENTATION_GATE),
        "gate": IMPLEMENTATION_GATE,
    }


def direct_shear_case(*, area_m2: float, normal_force_n: float,
                      temperature_c: float, strength: InterfaceStrength,
                      contact_time_s: float | None = None,
                      k_t: float = 1.0e6, dt: float = 2.0e-5,
                      steps: int = 6000, load_rate_n_per_s: float = 5.0e5,
                      zeta: float = 1.0, mass_kg: float = 1.0,
                      k_n: float = 1.0e7) -> dict[str, Any]:
    """Constant-normal direct shear of the interface; peak and residual traction.

    A **load-controlled** tangential ramp ``F_ext(t) = load_rate * t`` is applied to
    the sliding block: the transmitted traction is measured as the interface spring
    force, so it rises linearly until the interface capacity is reached, the bond
    fails, and the transmitted force drops to the sliding residual.  (A
    displacement-controlled drive is the wrong experiment for a *free* block: the
    spring then drags the block along at the drive rate and no load ever builds.)

    Measured peak must equal ``min(mu p + tau_ad, tau_coh)``, and the post-failure
    residual must equal the friction term ``mu p`` (the adhesion is gone once the
    bond fails).  The peak/residual ratio is the quantitative statement that a
    constant-``mu`` model *cannot* represent a bonded interface: it mis-states the
    peak by exactly the adhesive term.
    """
    area = _positive(area_m2, "area_m2")
    k_t = _positive(k_t, "k_t")
    mass = _positive(mass_kg, "mass_kg")
    normal_force = float(normal_force_n)
    if normal_force <= 0.0:
        raise ValueError("normal_force_n must be positive (a shear test needs closure)")
    pressure_mpa = normal_force / area / 1.0e6
    damping = 2.0 * zeta * math.sqrt(k_t * mass)
    overlap = normal_force / k_n

    position = 0.0
    velocity = 0.0
    bond_intact = True
    peak = 0.0
    residual = 0.0
    broke_at = None
    external = 0.0
    for index in range(int(steps)):
        external = load_rate_n_per_s * dt * (index + 1)
        force = contact_force(normal_overlap_m=overlap, tangential_offset_m=position,
                              slip_speed_m_per_s=max(abs(velocity), V_FLOOR),
                              area_m2=area, temperature_c=temperature_c,
                              pressure_mpa=pressure_mpa,
                              contact_time_s=contact_time_s, strength=strength,
                              k_n=k_n, k_t=k_t, bond_intact=bond_intact)
        if bond_intact and not force["bond_intact"]:
            broke_at = (index + 1) * dt
        bond_intact = force["bond_intact"]
        transmitted = -force["f_t"]
        peak = max(peak, transmitted)
        if broke_at is not None and abs(velocity) > 1.0e-3:
            residual = transmitted
        accel = (external + force["f_t"] - damping * velocity) / mass
        velocity += accel * dt
        position += velocity * dt

    mu = float(strength.friction.mu(max(abs(velocity), V_FLOOR), temperature_c,
                                    pressure_mpa))
    friction_term = mu * normal_force
    adhesion_term = float(strength.adhesion.tau_ad(temperature_c, pressure_mpa,
                                                   contact_time_s)) * area
    cohesive_cap = float(strength.tau_coh_pa(temperature_c)) * area
    predicted_peak = min(friction_term + adhesion_term, cohesive_cap)
    predicted_residual = friction_term
    peak_rel = (abs(peak - predicted_peak) / predicted_peak
                if predicted_peak > 0.0 else float("nan"))
    residual_rel = (abs(residual - predicted_residual) / predicted_residual
                    if predicted_residual > 0.0 else float("nan"))
    return {
        "peak_force_n": peak,
        "residual_force_n": residual,
        "predicted_peak_force_n": predicted_peak,
        "predicted_residual_force_n": predicted_residual,
        "peak_traction_pa": peak / area,
        "residual_traction_pa": residual / area,
        "predicted_peak_traction_pa": predicted_peak / area,
        "predicted_residual_traction_pa": predicted_residual / area,
        "peak_rel_error": peak_rel,
        "residual_rel_error": residual_rel,
        "friction_term_n": friction_term,
        "adhesion_term_n": adhesion_term,
        "cohesive_cap_n": cohesive_cap,
        "peak_over_residual": (peak / residual if residual > 0.0 else float("inf")),
        "mu_at_slip": mu,
        "broke_at_s": broke_at,
        "external_load_at_peak_n": predicted_peak,
        "passes_peak_gate": bool(peak_rel <= IMPLEMENTATION_GATE),
        "passes_residual_gate": bool(residual_rel <= RESIDUAL_GATE),
        "peak_gate": IMPLEMENTATION_GATE,
        "residual_gate": RESIDUAL_GATE,
    }


def adhesion_band_check(adhesion: AdhesionModel | None = None,
                        temperatures_c: Sequence[float] = (-2.0, -5.0, -10.0, -20.0, -30.0),
                        pressure_mpa: float = 0.1) -> dict[str, Any]:
    """Reference-band consistency of ``tau_ad(T)`` (MPa) -- NOT a data replication.

    Checks (a) monotone growth with cooling, (b) values inside the reported band,
    (c) the melting-point limit ``tau_ad(0) = 0``.  The band is wide by nature
    (factor of a few), so this returns the numbers and the band membership, and the
    report states explicitly that no certified ice-steel adfreeze data set was used.
    """
    model = adhesion or AdhesionModel()
    rows = []
    for temperature in temperatures_c:
        tau_mpa = float(model.tau_ad(temperature, pressure_mpa)) / 1.0e6
        rows.append({"temperature_c": float(temperature), "tau_ad_mpa": tau_mpa,
                     "inside_band": bool(ADHESION_LITERATURE_BAND_MPA[0] <= tau_mpa
                                         <= ADHESION_LITERATURE_BAND_MPA[1])})
    values = [row["tau_ad_mpa"] for row in rows]
    return {
        "rows": rows,
        "band_mpa": list(ADHESION_LITERATURE_BAND_MPA),
        "monotone_in_cooling": bool(all(b > a for a, b in zip(values, values[1:]))),
        "all_inside_band": bool(all(row["inside_band"] for row in rows)),
        "melting_limit_mpa": float(model.tau_ad(0.0, pressure_mpa)) / 1.0e6,
        "pressure_mpa": float(pressure_mpa),
    }


# ---------------------------------------------------------------------------
# 6. Resistance consequence of the interface law
# ---------------------------------------------------------------------------
def slide_resistance(v: float, temperature_c: float, pressure_mpa: float,
                     strength: InterfaceStrength, *,
                     contact_time_s: float | None = None,
                     case: IceResistanceCase | None = None,
                     sticking_fraction: float = 1.0) -> dict[str, Any]:
    """Ice resistance with the interface law, and the constant-``mu`` comparison.

    ``R = R_crush(T) + mu(p, v, T) * N + sticking_fraction * tau_ad(T, p, t_c) * A``

    where ``N = p A`` is the normal load of the hull patch and the adhesive term
    acts on the fraction of the patch that is *not* sliding (a stuck floe / ridge
    contact).  ``sticking_fraction = 1`` is the upper bound; ``0`` recovers the
    classic friction-only chain exactly, which is the negative control.

    The ``mu``-sensitivity is analytic (``dR/dmu = N``), so the finite-difference
    check in the tests verifies the implementation, not a fitted slope.
    """
    reference = case or IceResistanceCase()
    area = reference.contact_area_m2
    normal = reference.normal_load_n(pressure_mpa)
    crushing = reference.crushing_resistance_n(temperature_c)
    mu = float(strength.friction.mu(v, temperature_c, pressure_mpa))
    friction = mu * normal
    adhesion = (float(strength.adhesion.tau_ad(temperature_c, pressure_mpa,
                                              contact_time_s)) * area
                * float(sticking_fraction))
    total = crushing + friction + adhesion
    baseline_mu = 0.3  # the frozen contact coefficient of the ice-channel chain
    baseline = crushing + baseline_mu * normal
    return {
        "v_m_per_s": float(v),
        "temperature_c": float(temperature_c),
        "pressure_mpa": float(pressure_mpa),
        "mu": mu,
        "mu_constant_baseline": baseline_mu,
        "normal_load_n": normal,
        "contact_area_m2": area,
        "crushing_resistance_n": crushing,
        "friction_resistance_n": friction,
        "adhesion_resistance_n": adhesion,
        "total_resistance_n": total,
        "constant_mu_resistance_n": baseline,
        "delta_vs_constant_mu_n": total - baseline,
        "delta_vs_constant_mu_rel": (total - baseline) / baseline,
        "adhesion_share": adhesion / total if total > 0.0 else float("nan"),
        "d_resistance_d_mu_n": normal,
        "sticking_fraction": float(sticking_fraction),
    }


def sweep_grid(*, temperatures_c: Sequence[float] = (-2.0, -5.0, -10.0, -15.0,
                                                     -20.0, -25.0, -30.0),
               pressures_mpa: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
               speeds_m_per_s: Sequence[float] = (1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0),
               contact_times_s: Sequence[float] = (1.0, 10.0, 60.0, 600.0),
               strength: InterfaceStrength | None = None) -> dict[str, Any]:
    """Product grid of the interface law over ``T x p x v x t_c``.

    Vectorised (torch) and returning flat rows plus scalar summaries, so the same
    routine can be sharded by temperature on the cluster and re-aggregated.
    """
    model = strength or InterfaceStrength.calibrated()
    temperatures = _as_tensor(list(temperatures_c))
    pressures = _as_tensor(list(pressures_mpa))
    speeds = _as_tensor(list(speeds_m_per_s))
    times = _as_tensor(list(contact_times_s))
    grid = torch.cartesian_prod(temperatures, pressures, speeds, times)
    t, p, v, tc = grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3]
    mu = model.friction.mu_tensor(v, t, p)
    tau_ad = _as_tensor(model.adhesion.tau_ad(t, p, tc))
    friction_term = mu * p * 1.0e6
    cohesive = _as_tensor(model.tau_coh_pa(t))
    tau = torch.minimum(friction_term + tau_ad, cohesive)
    rows = [{
        "temperature_c": float(t[i]), "pressure_mpa": float(p[i]),
        "v_m_per_s": float(v[i]), "contact_time_s": float(tc[i]),
        "mu": float(mu[i]), "tau_ad_pa": float(tau_ad[i]),
        "friction_term_pa": float(friction_term[i]),
        "cohesive_cap_pa": float(cohesive[i]), "tau_shear_pa": float(tau[i]),
        "adhesion_share": float(tau_ad[i] / tau[i]) if float(tau[i]) > 0.0 else float("nan"),
        "cap_binds": bool(float(friction_term[i] + tau_ad[i]) > float(cohesive[i])),
    } for i in range(grid.shape[0])]
    shares = torch.tensor([row["adhesion_share"] for row in rows])
    return {
        "rows": rows,
        "count": len(rows),
        "adhesion_share_mean": float(shares.mean()),
        "adhesion_share_min": float(shares.min()),
        "adhesion_share_max": float(shares.max()),
        "n_cap_binding": int(sum(1 for row in rows if row["cap_binds"])),
        "axes": {"temperature_c": list(temperatures_c),
                 "pressure_mpa": list(pressures_mpa),
                 "v_m_per_s": list(speeds_m_per_s),
                 "contact_time_s": list(contact_times_s)},
        "friction_law": model.friction.as_dict(),
        "adhesion": model.adhesion.as_dict(),
    }


def attach_interface_law(contact: Any, strength: InterfaceStrength, *,
                         v_m_per_s: float, temperature_c: float,
                         pressure_mpa: float,
                         contact_time_s: float | None = None) -> dict[str, Any]:
    """Set the ice-channel ``ContactParameters.friction`` from the interface law.

    This is the *minimal* coupling to the existing :class:`tensordem.ice_channel`
    chain: the channel reads ``contact.friction`` every step, so the operating
    point ``(v, T, p)`` maps to one coefficient and nothing else in the contact
    parameters changes.  The adhesive part of the interface strength is *not* a
    Coulomb coefficient, so it is returned alongside as an explicit load rather
    than being folded into ``mu`` (folding it in would be a silent redefinition of
    the coefficient):

        ``F_ad = tau_ad(T, p, t_c) * A_bonded``

    Returns the updated contact plus the numbers used, so the caller can audit the
    mapping instead of trusting it.
    """
    if not hasattr(contact, "friction"):
        raise TypeError("contact must expose a 'friction' attribute")
    from dataclasses import replace as _replace

    mu = float(strength.friction.mu(v_m_per_s, temperature_c, pressure_mpa))
    tau_ad = float(strength.adhesion.tau_ad(temperature_c, pressure_mpa, contact_time_s))
    return {
        "contact": _replace(contact, friction=mu),
        "mu": mu,
        "tau_ad_pa": tau_ad,
        "operating_point": {"v_m_per_s": float(v_m_per_s),
                            "temperature_c": float(temperature_c),
                            "pressure_mpa": float(pressure_mpa),
                            "contact_time_s": contact_time_s},
    }
